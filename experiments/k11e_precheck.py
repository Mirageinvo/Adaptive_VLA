"""K-11e: структурная проверка НЕДОСЧИТАННОГО прогона, без вердикта.

ЗАЧЕМ. Агрегатор по частичному набору ячеек запускать нельзя: правило
зарегистрировано до прогона (ровно 400 пар, 10 задач, добора не будет), и
промежуточный интервал не добавляет знания, зато создаёт соблазн «починить»
голову или остановить прогон. Но ждать десять часов, чтобы узнать, что у
одной руки не пишется `init_hash_full` или разъехались sha, — потеря всего
прогона.

Здесь проверяется ТОЛЬКО то, что не зависит от результата: происхождение,
парность начальных состояний, полнота групп. Успех не читается.

КАК ЭТО ГАРАНТИРОВАНО, А НЕ ОБЕЩАНО. Поля `success` и агрегат `summary`
удаляются из каждой ячейки сразу при загрузке (`load_structural`). Дальше их
в памяти нет: напечатать долю успеха этот скрипт не может, даже если его
потом неосторожно расширят. Самопроверка это подтверждает.

Запуск:
    python3 experiments/k11e_precheck.py --selftest
    python3 experiments/k11e_precheck.py --cells data/k11e/cells \\
        --proto data/k11e/protocol.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ARMS = ("fullbar", "coarse24", "joint12", "hicora_s0", "hicora_s1")
# Поля, которые этот скрипт обязан НЕ видеть. Удаляются при загрузке.
FORBIDDEN_EP = ("success",)
FORBIDDEN_TOP = ("summary",)


def load_structural(obj):
    """Ячейка без всего, что говорит о результате.

    Не «мы не будем это печатать», а «этого здесь нет». Перечисленные поля
    вырезаются до возврата, поэтому любой дальнейший код физически не может
    сообщить долю успеха.
    """
    d = {k: v for k, v in obj.items() if k not in FORBIDDEN_TOP}
    d["episodes"] = [{k: v for k, v in e.items() if k not in FORBIDDEN_EP}
                     for e in obj.get("episodes", [])]
    return d


def group_key(c):
    return (c.get("task_id"), c.get("init_start"))


def check_pairing(by_arm):
    """Начальные состояния одной группы обязаны совпадать у всех рук.

    Сравнивается `init_hash_full` (он включает камеру запястья, в которую
    политика тоже смотрит) по `env_index`. Отсутствие хеша — дефект, а не
    повод пропустить сравнение: именно на нём держится парность.
    """
    bad = []
    ref_arm = sorted(by_arm)[0]
    ref = {e.get("env_index"): e.get("init_hash_full")
           for e in by_arm[ref_arm]["episodes"]}
    if not ref or any(v is None for v in ref.values()):
        bad.append(f"{ref_arm}: нет init_hash_full")
        return bad
    for arm in sorted(by_arm):
        cur = {e.get("env_index"): e.get("init_hash_full")
               for e in by_arm[arm]["episodes"]}
        if any(v is None for v in cur.values()):
            bad.append(f"{arm}: нет init_hash_full")
        elif set(cur) != set(ref):
            bad.append(f"{arm}: другой набор сред {sorted(set(cur))} против "
                       f"{sorted(set(ref))}")
        else:
            diff = sorted(i for i in ref if cur[i] != ref[i])
            if diff:
                bad.append(f"{arm}: начальные состояния расходятся с "
                           f"{ref_arm} в средах {diff}")
    return bad


def check_provenance(cells, proto=None):
    """Единая версия стенда, чекпойнт, черновик; отпечаток руки единственен."""
    bad = []
    sha = {c.get("script_sha1") for c in cells}
    if len(sha) > 1:
        bad.append(f"разные версии стенда: {sorted(sha)}")
    ck = {c.get("ckpt") for c in cells}
    if len(ck) > 1:
        bad.append(f"разные чекпойнты: {sorted(ck)}")
    wj, fp, hh = defaultdict(set), defaultdict(set), defaultdict(set)
    for c in cells:
        a = c.get("arm_label")
        j = c.get("joint")
        if not isinstance(j, dict):
            if a in ("joint12", "hicora_s0", "hicora_s1"):
                bad.append(f"{a}: нет словаря joint")
            continue
        wj[a].add(j.get("weights_sha1", "?"))
        if a.startswith("hicora_"):
            hh[a].add(j.get("hicora_sha1", "?"))
            fp[a].add(str(j.get("arm_fingerprint") or "?"))
    joints = {s for ss in wj.values() for s in ss}
    if "?" in joints:
        bad.append("черновик Joint12 не записан в части ячеек")
    elif len(joints) > 1:
        bad.append(f"руки идут от РАЗНЫХ Joint12: {sorted(joints)}")
    for a in sorted(hh):
        if len(hh[a]) > 1 or "?" in hh[a]:
            bad.append(f"{a}: голова не единственна {sorted(hh[a])}")
        if len(fp[a]) > 1 or "?" in fp[a]:
            bad.append(f"{a}: отпечаток руки не единственен {sorted(fp[a])}")
    s0 = next(iter(hh.get("hicora_s0", {"?"})))
    s1 = next(iter(hh.get("hicora_s1", {"?"})))
    if s0 != "?" and s0 == s1:
        bad.append("hicora_s0 и hicora_s1 идут от ОДНОЙ головы: это не "
                   "репликация по сиду")
    if proto:
        for a, key in (("hicora_s0", "head_s0_sha1"),
                       ("hicora_s1", "head_s1_sha1")):
            got = hh.get(a)
            if got and proto.get(key) and next(iter(got)) != proto[key]:
                bad.append(f"{a}: голова {next(iter(got))} расходится с "
                           f"протоколом {proto[key]}")
        if joints and proto.get("joint_sha1") and \
                joints != {proto["joint_sha1"]}:
            bad.append(f"черновик {sorted(joints)} расходится с протоколом "
                       f"{proto['joint_sha1']}")
    return bad


def report(cells, proto=None, expect_groups=40, out=sys.stdout):
    by_group = defaultdict(dict)
    for c in cells:
        by_group[group_key(c)][c.get("arm_label")] = c
    n_arm = defaultdict(int)
    for c in cells:
        n_arm[c.get("arm_label")] += 1
    full = sorted(g for g, d in by_group.items() if set(d) >= set(ARMS))
    part = sorted(g for g, d in by_group.items() if set(d) < set(ARMS))

    print(f"\n  ЯЧЕЕК {len(cells)} из {expect_groups * len(ARMS)}", file=out)
    for a in ARMS:
        print(f"    {a:>12}: {n_arm.get(a, 0):>3}", file=out)
    extra = sorted(set(n_arm) - set(ARMS) - {None})
    if extra:
        print(f"    посторонние метки: {extra}", file=out)
    print(f"  групп (задача, блок): полных {len(full)} из {expect_groups}, "
          f"незавершённых {len(part)}", file=out)
    if part:
        for g in part[:4]:
            print(f"    {g}: нет {sorted(set(ARMS) - set(by_group[g]))}",
                  file=out)

    bad = []
    for g in full:
        for b in check_pairing(by_group[g]):
            bad.append(f"группа {g}: {b}")
    bad += check_provenance(cells, proto)

    # ПАРНОСТЬ ПРОВЕРЯЕТСЯ ТОЛЬКО НА ПОЛНЫХ ГРУППАХ: в незавершённой части
    # рук просто ещё нет, и это не дефект.
    # Число эпизодов берётся фактическое, а не 10: `n_envs` — параметр, и
    # захардкоженный множитель напечатал бы неверный объём при другом блоке.
    n_ep = sum(len(by_group[g][ARMS[0]]["episodes"]) for g in full)
    print(f"  парность проверена на {len(full)} полных группах "
          f"({n_ep} эпизодов на руку)", file=out)
    if bad:
        print("\n  ДЕФЕКТЫ, КОТОРЫЕ ОБЕСЦЕНЯТ ПРОГОН:", file=out)
        for b in bad:
            print(f"    {b}", file=out)
        print("  Досчитывать при таких дефектах смысла нет: пары не "
              "сопоставимы\n  или руки идут от разных весов.", file=out)
        return False
    print("\n  СТРУКТУРА ЦЕЛА: начальные состояния совпадают у всех рук, "
          "версия стенда,\n  чекпойнт и черновик едины, головы s0/s1 различны "
          "и совпадают с протоколом.", file=out)
    print("  ВЕРДИКТ ЗДЕСЬ НЕ СЧИТАЕТСЯ. Доля успеха не прочитана: поля "
          "success и summary\n  удалены при загрузке. Анализ — после 200 "
          "ячеек, через k6h_summarize.py.", file=out)
    return True


def selftest():
    def ep(i, h, ok=True):
        return dict(env_index=i, init_hash="кратк", init_hash_full=h,
                    success=ok, env_steps=31, policy_calls=4)

    def cell(arm, t=0, i=0, h=("a", "b"), joint=True, wj="WJ", head=None,
             fp="FP", sha="SC"):
        c = dict(arm_label=arm, policy="x", task_id=t, init_start=i,
                 script_sha1=sha, ckpt="CK",
                 episodes=[ep(0, h[0]), ep(1, h[1])],
                 summary=dict(success_rate=0.9))
        if joint:
            c["joint"] = dict(weights_sha1=wj, arm_fingerprint=fp)
            if head is not None:
                c["joint"]["hicora_sha1"] = head
        return c

    # --- НЕЧИТАЕМОСТЬ УСПЕХА: это свойство, а не обещание ------------------
    raw = cell("fullbar")
    d = load_structural(raw)
    assert "summary" not in d, "summary должен быть вырезан"
    assert all("success" not in e for e in d["episodes"]), "success остался"
    assert "success" in raw["episodes"][0], "исходный объект не портим"
    blob = json.dumps(d)
    assert "success" not in blob and "summary" not in blob

    def group(t=0, i=0, h=("a", "b"), **kw):
        out = []
        for a in ARMS:
            k = dict(kw)
            if a.startswith("hicora_"):
                k.setdefault("head", "H0" if a.endswith("_s0") else "H1")
            out.append(load_structural(cell(a, t, i, h, **k)))
        return out

    ok = group()
    assert not check_pairing({c["arm_label"]: c for c in ok})
    # расхождение начальных состояний у одной руки
    bad_g = {c["arm_label"]: c for c in ok}
    bad_g["joint12"] = load_structural(cell("joint12", h=("a", "ИНОЕ")))
    assert check_pairing(bad_g), "расхождение init_hash_full принято"
    # отсутствие хеша
    no_h = {c["arm_label"]: c for c in ok}
    no_h["coarse24"]["episodes"][0].pop("init_hash_full")
    assert check_pairing(no_h), "отсутствие init_hash_full принято"
    # другой набор сред
    few = {c["arm_label"]: c for c in group()}
    few["fullbar"]["episodes"] = few["fullbar"]["episodes"][:1]
    assert check_pairing(few), "разный набор сред принят"

    assert not check_provenance(group())
    for mut, why in (
            (dict(wj="ДРУГОЙ"), "разные Joint12"),
            (dict(sha="ДРУГАЯ"), "разные версии стенда"),
    ):
        cs = group()
        cs[2] = load_structural(cell("joint12", **mut))
        assert check_provenance(cs), f"принято: {why}"
    # одна голова на оба сида
    cs = group()
    cs[4] = load_structural(cell("hicora_s1", head="H0"))
    assert check_provenance(cs), "одна голова на s0 и s1 принята"
    # голова расходится с протоколом
    assert check_provenance(group(), dict(head_s0_sha1="ИНАЯ")), \
        "расхождение с протоколом принято"
    assert not check_provenance(group(), dict(head_s0_sha1="H0",
                                             head_s1_sha1="H1",
                                             joint_sha1="WJ"))
    # отпечаток руки или голова не единственны ВНУТРИ метки. Ячейку надо
    # ДОБАВИТЬ, а не заменить: заменив единственную, получим снова одно
    # значение, и тест ничего не проверит.
    for mut, why in ((dict(fp="ИНОЙ"), "два отпечатка у одной руки"),
                     (dict(head="ИНАЯ"), "две головы у одной руки"),
                     (dict(wj="ИНОЙ"), "два черновика у одной руки")):
        cs = group()
        cs.append(load_structural(
            cell("hicora_s0", i=10, **dict(dict(head="H0"), **mut))))
        assert check_provenance(cs), f"принято: {why}"
    # добавленная ячейка той же руки с ТЕМИ ЖЕ значениями — не дефект
    cs = group()
    cs.append(load_structural(cell("hicora_s0", i=10, head="H0")))
    assert not check_provenance(cs), "второй блок той же руки отвергнут"
    # нет словаря joint у руки, которая обязана его иметь
    cs = group()
    cs[2] = load_structural(cell("joint12", joint=False))
    assert check_provenance(cs), "отсутствие joint принято"

    # отчёт на неполном наборе не падает и не выдаёт вердикта
    import io
    buf = io.StringIO()
    part = group() + group(t=1)[:3]
    assert report(part, None, expect_groups=2, out=buf) is True
    txt = buf.getvalue()
    assert "незавершённых 1" in txt and "ВЕРДИКТ ЗДЕСЬ НЕ СЧИТАЕТСЯ" in txt
    # объём берётся фактический: одна полная группа по два эпизода
    assert "(2 эпизодов на руку)" in txt, txt
    assert "успех" not in txt.lower().replace("доля успеха", "")

    print("самопроверка k11e_precheck пройдена: success и summary вырезаются "
          "при загрузке\n  и в отчёт попасть не могут; расхождение начальных "
          "состояний, отсутствие хеша,\n  разные Joint12, одна голова на два "
          "сида и расхождение с протоколом отвергаются")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cells", default="data/k11e/cells")
    ap.add_argument("--proto", default="data/k11e/protocol.json")
    ap.add_argument("--expect-groups", type=int, default=40)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    files = sorted(glob.glob(os.path.join(a.cells, "*.json")))
    if not files:
        raise SystemExit(f"нет ячеек в {a.cells}")
    cells = []
    for f in files:
        with open(f) as fh:
            cells.append(load_structural(json.load(fh)))
    proto = None
    if os.path.exists(a.proto):
        with open(a.proto) as fh:
            proto = json.load(fh)
        print(f"  протокол: черновик {proto.get('joint_sha1')}, головы "
              f"{proto.get('head_s0_sha1')} / {proto.get('head_s1_sha1')}")
    else:
        print(f"  ВНИМАНИЕ: протокола {a.proto} нет, сверка с ним пропущена")
    if not report(cells, proto, a.expect_groups):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
