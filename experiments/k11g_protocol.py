"""K-11g: протокол прогона, сверка ячеек и анализ окна исследования.

ЗАЧЕМ ОДИН МОДУЛЬ НА ТРИ ДЕЛА. Регистрация, сверка и чтение результата
опираются на одну и ту же схему ячейки. Разведи их по файлам — и поля
разойдутся ровно так, как уже разошлись разметчики событий в K-10.

ЗАРЕГИСТРИРОВАННОЕ ПРАВИЛО (записано ДО прогона). Годной считается одна и та
же МИНИМАЛЬНАЯ sigma, у которой НА ОБЕИХ головах:
    инварианты, паритет и происхождение пройдены;
    rms_median >= 0.01  — медианное изменение действия не ниже 1% диапазона
        канала. Порог зафиксирован ПОСЛЕ пилотного smoke на задаче 0, но ДО
        прогона на отложенных состояниях, и так и описывается; независимо
        измеренного «дрожания декодера» у нас нет;
    насыщение < 10%;
    падение успеха от СВОЕЙ детерминированной руки <= 10 пп.

ЧЕГО В ПРАВИЛЕ НЕТ, И ПОЧЕМУ. Дискордантность исходов в критерий НЕ входит,
хотя и считается обязательно. Она мерит пересечение границы бинарного успеха,
а не разнообразие поведения: пять испорченных успехов и ни одного улучшения
дали бы дискордантность 10% и падение ровно 10 пп, то есть формально прошли
бы — критерий выбрал бы чисто разрушительный шум. Обратный случай тоже
возможен: траектории заметно разные, но все успешны, и дискордантность нуль.

ЭТО ИНЖЕНЕРНЫЙ ФИЛЬТР ДЛЯ ВЫБОРА НАЧАЛЬНОЙ sigma PPO, а не доказательство
статистической не-худшести: правило не использует интервалов вовсе, а при
пятидесяти эпизодах на ячейку они были бы широкими.

Запуск:
    python3 experiments/k11g_protocol.py --selftest
    python3 experiments/k11g_protocol.py init --proto ... --cfg '{...}'
    python3 experiments/k11g_protocol.py check --proto ... --cell ... \
        --head s0 --sigma 0.05 --task 3
    python3 experiments/k11g_protocol.py analyze --proto ... --cells ... \
        --out ...
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HEADS = ("s0", "s1")
MIN_RMS = 0.01
MAX_SAT = 0.10
MAX_DROP = 10.0
# Поля ячейки, которые обязаны совпасть с протоколом. Всё, что определяет,
# ЧТО именно исполнялось.
CELL_VS_PROTO = (
    ("run_tag", "run_tag"), ("ckpt", "ckpt"), ("suite", "suite"),
    ("n_envs", "n_envs"), ("init_start", "init_start"),
    ("horizon", "horizon"), ("max_steps", "max_steps"),
    ("waiting_steps", "waiting_steps"), ("ensemble", "ensemble"),
    ("seed", "seed"), ("rollout_seed_mode", "rollout_seed_mode"),
    ("eps_salt", "eps_salt"), ("preprocess", "preprocess"),
    ("image_size", "image_size"), ("dtype", "dtype"),
    ("joint_sha1", "joint_sha1"), ("res_norm_sha1", "res_norm_sha1"),
    ("basis_sha1", "basis_sha1"), ("rho_sha1", "rho_sha1"),
    ("offset_table_sha1", "offset_table_sha1"),
    ("script_sha1", "cell_script_sha1"),
    ("hicora_g_sha1", "hicora_g_sha1"),
    ("hicora_vla_sha1", "hicora_vla_sha1"),
    ("joint12_vla_sha1", "joint12_vla_sha1"),
    ("min_rms", "min_rms"),
    ("device", "device"),
    ("k9h_sha1", "k9h_sha1"),
    ("rollout_seed", "rollout_seed"),
)


def sha12(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:12]


def load_json(path, what="файл"):
    """Чтение с ВНЯТНЫМ отказом. Оборванная ячейка от убитого процесса давала
    сырой JSONDecodeError, и причина тонула в трейсбеке."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"{what} ПОВРЕЖДЁН: {path}\n    {e}\n  Скорее всего процесс "
            f"убили посреди записи. Удалите этот файл — он пересчитается.")
    except OSError as e:
        raise SystemExit(f"{what} не читается: {path}\n    {e}")


def cell_name(task, head, sigma):
    """Имя ячейки. sigma печатается с двумя знаками, чтобы 0.1 и 0.10 не
    породили две разные ячейки одной конфигурации."""
    return f"t{int(task)}_{head}_sig{float(sigma):.2f}.json"


def build_protocol(cfg):
    p = dict(cfg)
    # ПОРОГИ И ВЕРСИИ АНАЛИЗА ТОЖЕ В ПРОТОКОЛЕ. Прежде max_sat и max_drop_pp
    # жили только константами модуля: после записи протокола их можно было
    # изменить и получить другой вердикт без единого отказа.
    p.setdefault("max_sat", MAX_SAT)
    p.setdefault("max_drop_pp", MAX_DROP)
    p.setdefault("protocol_script_sha1", sha12(os.path.abspath(__file__)))
    miss = [k for k in ("run_tag", "ckpt", "suite", "tasks", "sigmas",
                        "n_envs", "init_start", "eps_salt", "joint_sha1",
                        "head_s0_sha1", "head_s1_sha1", "min_rms",
                        "max_sat", "max_drop_pp", "device", "k9h_sha1",
                        "rollout_seed")
            if p.get(k) is None]
    if miss:
        raise SystemExit(f"в конфигурации протокола нет полей: {miss}")
    if 0.0 not in [float(x) for x in p["sigmas"]]:
        raise SystemExit("в сетке нет sigma=0: не будет детерминированной "
                         "опоры, от которой считается падение успеха")
    if p["head_s0_sha1"] == p["head_s1_sha1"]:
        raise SystemExit("головы s0 и s1 — один файл: это не две руки")
    for k, want in (("min_rms", MIN_RMS), ("max_sat", MAX_SAT),
                    ("max_drop_pp", MAX_DROP)):
        if abs(float(p[k]) - want) > 1e-12:
            raise SystemExit(f"{k}={p[k]} вместо зарегистрированного {want}")
    return p


def verify_protocol(old, cur):
    diff = [k for k in sorted(set(old) | set(cur))
            if str(old.get(k)) != str(cur.get(k))]
    if diff:
        raise SystemExit(
            f"ПРОТОКОЛ K-11g РАСХОДИТСЯ С ЗАПИСАННЫМ по полям {diff}.\n"
            f"  Продолжать прогон другой конфигурацией нельзя: часть ячеек "
            f"была бы посчитана иначе,\n  а пропуск готовых это скрыл бы.")
    return True


def check_cell(cell, proto, head, sigma, task):
    """Ячейка обязана соответствовать протоколу. Отсутствие поля — отказ."""
    bad = []

    def eq(got, want, name):
        if got is None:
            bad.append(f"{name}: нет поля")
        elif str(got) != str(want):
            bad.append(f"{name}: {got!r}, ожидалось {want!r}")

    eq(cell.get("head"), head, "head")
    eq(cell.get("task_id"), task, "task_id")
    if cell.get("sigma") is None:
        bad.append("sigma: нет поля")
    elif abs(float(cell["sigma"]) - float(sigma)) > 1e-12:
        bad.append(f"sigma: {cell['sigma']} вместо {sigma}")
    want_mode = "deterministic" if float(sigma) == 0.0 else "gaussian"
    eq(cell.get("mode"), want_mode, "mode")
    for cf, pf in CELL_VS_PROTO:
        if pf in proto:
            eq(cell.get(cf), proto[pf], cf)
    eq(cell.get("head_sha1"), proto.get(f"head_{head}_sha1"), "head_sha1")
    if not (cell.get("parity") or {}).get("ok"):
        bad.append("parity: не сошёлся")
    if float(sigma) > 0 and not cell.get("eps_sha1"):
        bad.append("eps_sha1: нет, общий поток шума нечем подтвердить")
    if bad:
        raise SystemExit(
            f"ЯЧЕЙКА НЕ СООТВЕТСТВУЕТ ПРОТОКОЛУ ({head}, sigma={sigma}, "
            f"задача {task}):\n    " + "\n    ".join(bad)
            + "\n  Пропуск такой ячейки дал бы вердикт не для "
              "зарегистрированной конфигурации.")
    return True


def check_eps_shared(cells):
    """Поток шума зависит от (задача, состояние, вызов, соль) — НЕ от головы и
    НЕ от sigma. Значит внутри задачи все ненулевые sigma обеих голов обязаны
    дать ОДИН eps_sha1. Это прямая проверка общего потока."""
    by_task = {}
    for (hd, sg, t), c in cells.items():
        if float(sg) == 0.0:
            continue
        by_task.setdefault(t, {})[(hd, sg)] = c.get("eps_sha1")
    bad = []
    for t, d in sorted(by_task.items()):
        vals = {v for v in d.values()}
        if None in vals or "" in vals:
            bad.append(f"задача {t}: eps_sha1 не записан у "
                       f"{[k for k, v in d.items() if not v]}")
        elif len(vals) > 1:
            bad.append(f"задача {t}: РАЗНЫЕ потоки шума {sorted(vals)}")
    if bad:
        raise SystemExit("ПОТОК ШУМА НЕ ОБЩИЙ:\n    " + "\n    ".join(bad)
                         + "\n  Тогда ячейки отличаются не только масштабом "
                           "шума, и сравнивать sigma нельзя.")
    return True


def check_shared_initial_states(cells):
    """Один pair_key — ОДНО фактическое начальное состояние у всех рук.

    ЗАЧЕМ СВЕРХ НЕПУСТОГО init_hash_full. Ячейка требовала лишь наличие хеша,
    а `discordance` соединяет эпизоды по строке `suite|task|init_state_id`.
    Значит два РАЗНЫХ фактических состояния под одним номером образовали бы
    пару, и парное сравнение перестало бы быть парным — молча.
    """
    by_key = {}
    for (hd, sg, t), c in cells.items():
        for e in c["episodes"]:
            by_key.setdefault(e["pair_key"], {})[(hd, sg, t)] = (
                e.get("init_hash_full"), e.get("rollout_seed"))
    bad = []
    for k, d in sorted(by_key.items()):
        hs = {v[0] for v in d.values()}
        rs = {v[1] for v in d.values()}
        if None in rs:
            bad.append(f"{k}: rollout_seed не записан в части эпизодов")
            rs = {v for v in rs if v is not None}
        if None in hs or "" in hs:
            bad.append(f"{k}: init_hash_full отсутствует у "
                       f"{[a for a, v in d.items() if not v[0]][:3]}")
        elif len(hs) > 1:
            bad.append(f"{k}: РАЗНЫЕ начальные состояния, хешей "
                       f"{len(hs)} на {len(d)} рук")
        if len(rs) > 1:
            bad.append(f"{k}: разные сиды раскатки {sorted(rs, key=str)}")
    if bad:
        raise SystemExit(
            "НАЧАЛЬНЫЕ СОСТОЯНИЯ НЕ ОБЩИЕ:\n    " + "\n    ".join(bad[:8])
            + ("\n    ..." if len(bad) > 8 else "")
            + "\n  Парное сравнение по номеру состояния тогда соединяло бы "
              "разные эпизоды.")
    return True


def discordance(test_eps, ref_eps):
    """Парное сравнение исходов по pair_key. Считает АГРЕГАТОР, не воркер.

    Воркер видит только свою руку, и сравнивать ему не с чем. Пары — по
    (suite, task, init_state_id), то есть по одному и тому же начальному
    состоянию.
    """
    a = {e["pair_key"]: bool(e["success"]) for e in test_eps}
    b = {e["pair_key"]: bool(e["success"]) for e in ref_eps}
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("нет общих пар: исходы не сопоставимы")
    win = sum(1 for k in keys if a[k] and not b[k])
    loss = sum(1 for k in keys if b[k] and not a[k])
    return dict(pairs=len(keys), unpaired=len(set(a) ^ set(b)),
                win=win, loss=loss, discordant=win + loss,
                discordant_frac=(win + loss) / len(keys),
                paired_diff_pp=100.0 * (win - loss) / len(keys))


def pool(cells_of_arm, expect_episodes=None):
    """Сводка руки по всем задачам. RMS — медиана ПО ЭПИЗОДАМ всех задач.

    СВОДКА ЯЧЕЙКИ ПЕРЕСЧИТЫВАЕТСЯ ЗДЕСЬ ЗАНОВО, а не берётся на веру: гейтующий
    `rms_median` — то самое число, подмена которого меняет вердикт. Воркер его
    тоже сверяет, но агрегатор обязан не зависеть от честности воркера.
    """
    import k11g_cell as kc
    per_ep, sat, dzf, lad, eps_rows = [], [], [], {}, []
    chunks = succ = n_ep = grip = 0
    for c in cells_of_arm:
        s = kc.summarize(c["chunks"], c["episodes"], c["mode"])
        per_ep += list(s.get("rms_per_episode") or [])
        sat.append((s["sat_frac_mean"], s["chunks"]))
        dzf.append(s["dz_frac_max"])
        for k, v in (s.get("changed_frac_ladder") or {}).items():
            lad.setdefault(k, []).append((v, s["chunks"]))
        chunks += s["chunks"]
        grip += int(round(s["grip_flip_frac"] * s["chunks"]))
        succ += s["successes"]
        n_ep += s["episodes"]
        eps_rows += c["episodes"]
    def wmean(pairs):
        w = sum(n for _, n in pairs)
        return (sum(v * n for v, n in pairs) / w) if w else 0.0
    return dict(
        episodes=n_ep, successes=succ,
        success=(succ / n_ep) if n_ep else 0.0, chunks=chunks,
        rms_median=float(np.median(per_ep)) if per_ep else 0.0,
        rms_n_episodes=len(per_ep),
        sat_frac_mean=wmean(sat), dz_frac_max=max(dzf) if dzf else 0.0,
        grip_flip_frac=(grip / chunks) if chunks else 0.0,
        changed_frac_ladder={k: wmean(v) for k, v in sorted(lad.items())},
        invariants_ok=all(kc.summarize(c["chunks"], c["episodes"],
                                       c["mode"])["invariants_ok"]
                          for c in cells_of_arm),
        episodes_rows=eps_rows)


def read_window(arms, min_rms=MIN_RMS, max_sat=MAX_SAT, max_drop=MAX_DROP):
    """Вердикт по зарегистрированному правилу. Минимальная годная sigma."""
    out, passing = {}, []
    sigmas = sorted({sg for _, sg in arms if float(sg) > 0.0})
    for sg in sigmas:
        per, ok = {}, True
        for hd in HEADS:
            a, base = arms.get((hd, sg)), arms.get((hd, 0.0))
            if a is None or base is None:
                per[hd] = dict(reason="нет руки")
                ok = False
                continue
            drop = (base["success"] - a["success"]) * 100.0
            r = dict(rms_median=a["rms_median"], sat=a["sat_frac_mean"],
                     success=a["success"], det_success=base["success"],
                     drop_pp=drop, invariants_ok=a["invariants_ok"],
                     ok_rms=bool(a["rms_median"] >= min_rms),
                     ok_sat=bool(a["sat_frac_mean"] < max_sat),
                     ok_drop=bool(drop <= max_drop))
            r["ok"] = bool(r["invariants_ok"] and r["ok_rms"] and r["ok_sat"]
                           and r["ok_drop"])
            per[hd] = r
            ok = ok and r["ok"]
        out[sg] = dict(per_head=per, ok=bool(ok))
        if ok:
            passing.append(sg)
    return dict(per_sigma=out, passing=sorted(passing),
                chosen=(min(passing) if passing else None),
                rule=dict(min_rms=min_rms, max_sat=max_sat,
                          max_drop=max_drop,
                          note="дискордантность НЕ входит в критерий"))


def selftest():
    # --- имя ячейки ---------------------------------------------------------
    assert cell_name(3, "s0", 0.1) == cell_name(3, "s0", 0.10)
    assert cell_name(3, "s0", 0.05) == "t3_s0_sig0.05.json"

    # --- протокол -----------------------------------------------------------
    cfg = dict(run_tag="k11g", ckpt="A/B", suite="10", tasks=list(range(10)),
               sigmas=[0.0, 0.03, 0.05, 0.07, 0.10, 0.30, 0.50], n_envs=5,
               init_start=40, eps_salt=1, joint_sha1="wj",
               head_s0_sha1="h0", head_s1_sha1="h1", min_rms=MIN_RMS,
               max_sat=MAX_SAT, max_drop_pp=MAX_DROP, device="cuda:0",
               k9h_sha1="k9", rollout_seed=0,
               horizon=8, max_steps=600, waiting_steps=10, ensemble="off",
               seed=0, rollout_seed_mode="block", preprocess="pp",
               image_size=224, dtype="float16", res_norm_sha1="rn",
               basis_sha1="bs", rho_sha1="rh", offset_table_sha1="ot",
               cell_script_sha1="cs", hicora_g_sha1="hg",
               hicora_vla_sha1="hv", joint12_vla_sha1="jv")
    p = build_protocol(cfg)
    assert verify_protocol(p, build_protocol(dict(cfg)))
    for mut, why in ((dict(sigmas=[0.03, 0.10]), "нет sigma=0"),
                     (dict(head_s1_sha1="h0"), "одна голова на два сида"),
                     (dict(min_rms=0.001), "иной порог исследования"),
                     (dict(max_sat=0.5), "иной порог насыщения"),
                     (dict(max_drop_pp=40.0), "иной порог падения"),
                     (dict(device=None), "нет карты"),
                     (dict(k9h_sha1=None), "нет версии k9h"),
                     (dict(joint_sha1=None), "нет поля")):
        try:
            build_protocol(dict(cfg, **mut))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"принято: {why}")
    try:
        verify_protocol(p, build_protocol(dict(cfg, init_start=0)))
    except SystemExit:
        pass
    else:
        raise AssertionError("смена состояний при возобновлении принята")

    # --- сверка ячейки ------------------------------------------------------
    def mk_cell(head="s0", sigma=0.05, task=3, n=5, succ=4, rms=0.02,
                sat=0.01, bad=None):
        eps = [dict(env_index=i, init_state_id=40 + i,
                    pair_key=f"10|{task}|{40 + i}", success=(i < succ),
                    init_hash="a", init_hash_full="b", env_steps=80,
                    policy_calls=10, rollout_seed=0) for i in range(n)]
        c = dict(run_tag="k11g", head=head, sigma=sigma, task_id=task,
                 mode=("deterministic" if sigma == 0.0 else "gaussian"),
                 ckpt="A/B", suite="10", n_envs=n, init_start=40, horizon=8,
                 max_steps=600, waiting_steps=10, ensemble="off", seed=0,
                 rollout_seed_mode="block", eps_salt=1, preprocess="pp",
                 image_size=224, dtype="float16", joint_sha1="wj",
                 head_sha1=("h0" if head == "s0" else "h1"),
                 res_norm_sha1="rn", basis_sha1="bs", rho_sha1="rh",
                 offset_table_sha1="ot", script_sha1="cs",
                 hicora_g_sha1="hg", hicora_vla_sha1="hv",
                 joint12_vla_sha1="jv", min_rms=MIN_RMS, device="cuda:0",
                 k9h_sha1="k9", rollout_seed=0,
                 eps_sha1=(None if sigma == 0.0 else "E" + str(task)),
                 parity=dict(ok=True), episodes=eps)
        # ФИКСТУРА САМОСОГЛАСОВАНА: сводка считается из строк тем же кодом,
        # что и в воркере. Иначе агрегатор, который теперь пересчитывает
        # сводку сам, падал бы на собственной самопроверке.
        import k11g_cell as kc
        ch = [dict(call=j // n, env_index=j % n, rms=rms, max=2 * rms,
                   changed=bool(2 * rms > 0.01), grip_flip=False,
                   sat_frac=sat, dz_frac=0.5, layers_run=24,
                   log_prob_u=(None if sigma == 0.0 else -10.0))
              for j in range(2 * n)]
        c["chunks"] = ch
        c["summary"] = kc.summarize(ch, eps, c["mode"])
        if bad:
            c.update(bad)
        return c

    assert check_cell(mk_cell(), p, "s0", 0.05, 3)
    for mut, why in ((dict(head="s1"), "чужая голова"),
                     (dict(sigma=0.07), "чужая sigma"),
                     (dict(task_id=4), "чужая задача"),
                     (dict(mode="deterministic"), "не тот режим"),
                     (dict(init_start=0), "иные состояния"),
                     (dict(eps_salt=0), "иная соль шума"),
                     (dict(preprocess="иное"), "иной препроцессинг"),
                     (dict(hicora_g_sha1="иной"), "иная версия головы"),
                     (dict(offset_table_sha1="иной"), "иная таблица смещений"),
                     (dict(head_sha1="h1"), "веса чужой головы"),
                     (dict(parity=dict(ok=False)), "паритет не сошёлся"),
                     (dict(eps_sha1=None), "нет eps_sha1"),
                     (dict(min_rms=0.001), "иной порог"),
                     (dict(device="cuda:1"), "иная карта"),
                     (dict(k9h_sha1="иной"), "иная версия k9h"),
                     (dict(rollout_seed=7), "иной сид раскатки"),
                     (dict(ensemble="on"), "включён ансамбль")):
        try:
            check_cell(mk_cell(bad=mut), p, "s0", 0.05, 3)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"принято: {why}")

    # --- общий поток шума ---------------------------------------------------
    good = {}
    for hd in HEADS:
        for sg in (0.0, 0.05, 0.07):
            for t in (0, 1):
                good[(hd, sg, t)] = mk_cell(head=hd, sigma=sg, task=t)
    assert check_eps_shared(good)
    bad1 = dict(good)
    bad1[("s1", 0.07, 1)] = mk_cell(head="s1", sigma=0.07, task=1,
                                    bad=dict(eps_sha1="ДРУГОЙ"))
    try:
        check_eps_shared(bad1)
    except SystemExit:
        pass
    else:
        raise AssertionError("разные потоки шума приняты")
    bad2 = dict(good)
    bad2[("s0", 0.05, 0)] = mk_cell(sigma=0.05, task=0,
                                    bad=dict(eps_sha1=None))
    try:
        check_eps_shared(bad2)
    except SystemExit:
        pass
    else:
        raise AssertionError("отсутствующий eps_sha1 принят")

    # --- ОБЩИЕ НАЧАЛЬНЫЕ СОСТОЯНИЯ МЕЖДУ РУКАМИ ---------------------------
    # Непустого хеша мало: пары соединяются по номеру состояния, и разные
    # фактические состояния под одним номером прошли бы молча.
    assert check_shared_initial_states(good)
    bh = dict(good)
    c_ = mk_cell(head="s1", sigma=0.05, task=0)
    c_["episodes"][2]["init_hash_full"] = "ДРУГОЕ_СОСТОЯНИЕ"
    bh[("s1", 0.05, 0)] = c_
    try:
        check_shared_initial_states(bh)
    except SystemExit as e:
        assert "РАЗНЫЕ начальные состояния" in str(e), str(e)
    else:
        raise AssertionError("чужое начальное состояние принято")
    bs_ = dict(good)
    c_ = mk_cell(head="s1", sigma=0.05, task=1)
    c_["episodes"][0]["rollout_seed"] = 999
    bs_[("s1", 0.05, 1)] = c_
    try:
        check_shared_initial_states(bs_)
    except SystemExit as e:
        assert "сиды раскатки" in str(e), str(e)
    else:
        raise AssertionError("разные сиды раскатки приняты")
    nh = dict(good)
    c_ = mk_cell(task=0)
    c_["episodes"][1]["init_hash_full"] = ""
    nh[("s0", 0.05, 0)] = c_
    try:
        check_shared_initial_states(nh)
    except SystemExit:
        pass
    else:
        raise AssertionError("отсутствующий хеш принят")

    # --- ПОРОГИ БЕРУТСЯ ИЗ ПРОТОКОЛА --------------------------------------
    # Подмена константы модуля после записи протокола не должна менять
    # вердикт: read_window обязан принимать пороги аргументами.
    arms_t = {(hd, s_): dict(success=0.9, rms_median=0.012,
                             sat_frac_mean=0.01, invariants_ok=True)
              for hd in HEADS for s_ in (0.0, 0.07)}
    assert read_window(arms_t, min_rms=0.01)["chosen"] == 0.07
    assert read_window(arms_t, min_rms=0.02)["chosen"] is None
    assert read_window(arms_t, min_rms=0.01, max_sat=0.005)["chosen"] is None
    drop_t = dict(arms_t)
    drop_t[("s0", 0.07)] = dict(success=0.70, rms_median=0.012,
                                sat_frac_mean=0.01, invariants_ok=True)
    assert read_window(drop_t, min_rms=0.01, max_drop=10.0)["chosen"] is None
    assert read_window(drop_t, min_rms=0.01, max_drop=40.0)["chosen"] == 0.07

    # --- ПЕРЕСЧЁТ СВОДКИ АГРЕГАТОРОМ ------------------------------------
    # Воспроизведённый обход: в сводке ячейки rms_median подменён на 0.999.
    # pool обязан взять пересчитанное из строк значение, а не записанное.
    import k11g_cell as kc
    ep_f = [dict(env_index=i, init_state_id=40 + i, pair_key=f"10|0|{40+i}",
                 init_hash="a", init_hash_full="b", success=True,
                 env_steps=80, policy_calls=2, rollout_seed=0)
            for i in range(2)]
    ch_f = [dict(call=c, env_index=c % 2, rms=0.002, max=0.004, changed=False,
                 grip_flip=False, sat_frac=0.0, dz_frac=0.1, layers_run=24,
                 log_prob_u=-1.0) for c in range(4)]
    forged = dict(mode="gaussian", episodes=ep_f, chunks=ch_f,
                  summary=dict(kc.summarize(ch_f, ep_f, "gaussian"),
                               rms_median=0.999,
                               rms_per_episode=[0.999, 0.999]))
    got = pool([forged])["rms_median"]
    assert abs(got - 0.002) < 1e-12, f"агрегатор поверил подделке: {got}"

    # --- дискордантность ----------------------------------------------------
    ref = [dict(pair_key=f"10|0|{i}", success=True) for i in range(10)]
    tst = [dict(pair_key=f"10|0|{i}", success=(i >= 3)) for i in range(10)]
    d = discordance(tst, ref)
    assert d == dict(pairs=10, unpaired=0, win=0, loss=3, discordant=3,
                     discordant_frac=0.3, paired_diff_pp=-30.0), d
    up = discordance([dict(pair_key="10|0|0", success=True)],
                     [dict(pair_key="10|0|0", success=False),
                      dict(pair_key="10|0|1", success=True)])
    assert up["pairs"] == 1 and up["unpaired"] == 1 and up["win"] == 1
    try:
        discordance([dict(pair_key="x", success=True)],
                    [dict(pair_key="y", success=True)])
    except SystemExit:
        pass
    else:
        raise AssertionError("несопоставимые пары приняты")

    # --- правило окна -------------------------------------------------------
    def arm(succ, rms, sat=0.01, inv=True):
        return dict(success=succ, rms_median=rms, sat_frac_mean=sat,
                    invariants_ok=inv)

    arms = {(hd, 0.0): arm(0.90, 0.0) for hd in HEADS}
    for hd in HEADS:
        arms[(hd, 0.03)] = arm(0.90, 0.0053)    # исследования нет
        arms[(hd, 0.05)] = arm(0.89, 0.0095)    # всё ещё ниже порога
        arms[(hd, 0.07)] = arm(0.87, 0.0139)    # годна
        arms[(hd, 0.10)] = arm(0.86, 0.0202)    # годна, но больше чем надо
        arms[(hd, 0.30)] = arm(0.50, 0.0600, sat=0.40)
    w = read_window(arms)
    assert w["passing"] == [0.07, 0.10], w["passing"]
    assert w["chosen"] == 0.07
    assert not w["per_sigma"][0.03]["ok"] and not w["per_sigma"][0.05]["ok"]
    assert not w["per_sigma"][0.30]["ok"]
    # ПАДЕНИЕ СЧИТАЕТСЯ ОТ СВОЕЙ ОПОРЫ
    sh = dict(arms)
    sh[("s1", 0.0)] = arm(0.70, 0.0)
    sh[("s1", 0.07)] = arm(0.68, 0.0139)
    assert read_window(sh)["per_sigma"][0.07]["ok"]
    # ОДНА НЕГОДНАЯ ГОЛОВА ЛОМАЕТ sigma
    ob = dict(arms)
    ob[("s1", 0.07)] = arm(0.87, 0.0139, sat=0.50)
    assert read_window(ob)["chosen"] == 0.10
    # НАРУШЕННЫЕ ИНВАРИАНТЫ И ОТСУТСТВИЕ РУКИ
    iv = dict(arms)
    iv[("s0", 0.07)] = arm(0.90, 0.02, inv=False)
    assert not read_window(iv)["per_sigma"][0.07]["ok"]
    ms = {k: v for k, v in arms.items() if k != ("s1", 0.0)}
    assert not read_window(ms)["per_sigma"][0.07]["ok"]
    none_ok = {(hd, s): arm(0.2, 0.0001) for hd in HEADS for s in (0.0, 0.07)}
    assert read_window(none_ok)["chosen"] is None

    # --- сведение по задачам ------------------------------------------------
    po = pool([mk_cell(task=t, rms=0.02 + 0.001 * t, succ=4) for t in range(3)])
    assert po["episodes"] == 15 and po["successes"] == 12
    assert po["rms_n_episodes"] == 15
    assert abs(po["rms_median"] - 0.021) < 1e-9, po["rms_median"]
    assert po["invariants_ok"]

    print("самопроверка k11g_protocol пройдена: протокол требует sigma=0, "
          "двух разных голов\n  и зарегистрированных порогов; ячейка "
          "сверяется по семнадцати подменам; поток\n  шума обязан быть общим "
          "внутри задачи, начальные состояния и сиды — у всех\n  рук; пороги "
          "приходят ИЗ ПРОТОКОЛА; агрегатор пересчитывает сводку и не\n  "
          "верит подделанному rms_median; дискордантность парная и в критерий "
          "НЕ входит;\n  окно берёт минимальную годную sigma и падает от "
          "одной негодной головы")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["init", "check", "analyze"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--proto", default="data/k11g/protocol.json")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--cells", default="data/k11g/cells")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest or a.cmd is None:
        selftest()
        return

    if a.cmd == "init":
        cfg = json.loads(a.cfg) if a.cfg else {}
        cur = build_protocol(cfg)
        if os.path.exists(a.proto):
            verify_protocol(load_json(a.proto, "протокол"), cur)
            print(f"  протокол сверен: {a.proto}")
            return
        n = len(glob.glob(os.path.join(a.cells, "*.json")))
        if n:
            raise SystemExit(
                f"в {a.cells} уже {n} ячеек, а протокола нет. Ставить "
                f"протокол задним числом нельзя:\n  он подстроился бы под то, "
                f"что посчитано. Перенесите ячейки или смените каталог.")
        os.makedirs(os.path.dirname(os.path.abspath(a.proto)) or ".",
                    exist_ok=True)
        tmp = a.proto + ".tmp"
        json.dump(cur, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, a.proto)
        print(f"  протокол записан: {a.proto}")
        return

    proto = load_json(a.proto, "протокол")
    # ПРОТОКОЛ ПЕРЕПРОВЕРЯЕТСЯ ПРИ ЧТЕНИИ. Пороги теперь берутся из него, и
    # без этого их можно было бы отредактировать В ФАЙЛЕ после записи и
    # получить другой вердикт. build_protocol отказывает на любом значении,
    # кроме зарегистрированного.
    build_protocol(dict(proto))
    if a.cmd == "check":
        check_cell(load_json(a.cell, "ячейка"), proto, a.head,
                   a.sigma, a.task)
        print(f"  ячейка сверена: {a.cell}")
        return

    # --- analyze ------------------------------------------------------------
    cells, missing = {}, []
    for t in proto["tasks"]:
        for hd in HEADS:
            for sg in proto["sigmas"]:
                f = os.path.join(a.cells, cell_name(t, hd, sg))
                if not os.path.exists(f):
                    missing.append(os.path.basename(f))
                    continue
                c = load_json(f, "ячейка")
                check_cell(c, proto, hd, float(sg), t)
                cells[(hd, float(sg), t)] = c
    if missing:
        raise SystemExit(
            f"НЕ ХВАТАЕТ {len(missing)} ЯЧЕЕК ИЗ "
            f"{len(proto['tasks']) * 2 * len(proto['sigmas'])}: "
            f"{missing[:6]}{'...' if len(missing) > 6 else ''}\n"
            f"  Вердикт по неполному набору относился бы не к "
            f"зарегистрированному пилоту.")
    check_eps_shared(cells)
    check_shared_initial_states(cells)
    n_exp = int(proto["n_envs"]) * len(proto["tasks"])
    print(f"  ячеек {len(cells)}, поток шума общий внутри каждой задачи, "
          f"начальные состояния общие у всех рук")

    arms, disc = {}, {}
    for hd in HEADS:
        for sg in [float(x) for x in proto["sigmas"]]:
            a_ = pool([cells[(hd, sg, t)] for t in proto["tasks"]])
            # ОЖИДАЕМОЕ ЧИСЛО ЭПИЗОДОВ — ИЗ ПРОТОКОЛА. Рука, собранная из
            # меньшего числа эпизодов, дала бы медиану RMS по другому набору.
            if a_["rms_n_episodes"] != n_exp and float(sg) > 0:
                raise SystemExit(
                    f"{hd}, sigma={sg}: RMS посчитан по "
                    f"{a_['rms_n_episodes']} эпизодам вместо {n_exp}")
            if a_["episodes"] != n_exp:
                raise SystemExit(f"{hd}, sigma={sg}: эпизодов "
                                 f"{a_['episodes']} вместо {n_exp}")
            arms[(hd, sg)] = a_
    for hd in HEADS:
        for sg in [float(x) for x in proto["sigmas"]]:
            if sg == 0.0:
                continue
            disc[f"{hd}|{sg}"] = discordance(
                arms[(hd, sg)]["episodes_rows"],
                arms[(hd, 0.0)]["episodes_rows"])
    # ПОРОГИ БЕРУТСЯ ИЗ ПРОТОКОЛА, а не из константов модуля: иначе их можно
    # было изменить после записи протокола и получить другой вердикт.
    win = read_window(arms, min_rms=float(proto["min_rms"]),
                      max_sat=float(proto["max_sat"]),
                      max_drop=float(proto["max_drop_pp"]))
    print(f"\n  ОКНО ИССЛЕДОВАНИЯ (правило записано до прогона): "
          f"RMS >= {proto['min_rms']}, насыщение < {proto['max_sat']}, "
          f"падение <= {proto['max_drop_pp']} пп")
    print(f"    {'sigma':>6}{'голова':>8}{'RMS':>9}{'нас':>7}{'усп':>7}"
          f"{'пад пп':>8}{'иссл':>6}{'без':>5}{'вердикт':>9}")
    for sg in sorted(win["per_sigma"]):
        for hd in HEADS:
            r = win["per_sigma"][sg]["per_head"].get(hd, {})
            if "rms_median" not in r:
                print(f"    {sg:>6.2f}{hd:>8}{'нет руки':>9}")
                continue
            print(f"    {sg:>6.2f}{hd:>8}{r['rms_median']:>9.4f}"
                  f"{100 * r['sat']:>6.1f}%{100 * r['success']:>6.1f}%"
                  f"{r['drop_pp']:>+8.1f}{('да' if r['ok_rms'] else 'НЕТ'):>6}"
                  f"{('да' if r['ok_sat'] and r['ok_drop'] else 'НЕТ'):>5}"
                  f"{('годна' if win['per_sigma'][sg]['ok'] else '-'):>9}")
    print(f"\n  ДИСКОРДАНТНОСТЬ — обязательная диагностика, в критерий НЕ "
          f"входит")
    print(f"    {'рука':>12}{'пар':>6}{'лучше':>7}{'хуже':>6}{'дискорд':>9}"
          f"{'парная разн':>13}")
    for k in sorted(disc, key=lambda x: (x.split("|")[0], float(x.split("|")[1]))):
        d = disc[k]
        print(f"    {k:>12}{d['pairs']:>6}{d['win']:>7}{d['loss']:>6}"
              f"{100 * d['discordant_frac']:>8.1f}%"
              f"{d['paired_diff_pp']:>+12.1f}")
    print(f"\n  ЛЕСТНИЦА ИЗМЕНЕНИЯ ДЕЙСТВИЯ — диагностика (доля при 1% "
          f"насыщается)")
    for hd in HEADS:
        for sg in sorted({s for _, s in arms}):
            if sg == 0.0:
                continue
            lad = arms[(hd, sg)]["changed_frac_ladder"]
            print(f"    {hd}, sigma={sg:<5}: "
                  + "  ".join(f"{k}:{100 * v:.0f}%"
                              for k, v in sorted(lad.items(),
                                                 key=lambda kv: float(kv[0]))))
    out = dict(protocol=proto, window=win, discordance=disc,
               arms={f"{hd}|{sg}": {k: v for k, v in arms[(hd, sg)].items()
                                    if k != "episodes_rows"}
                     for hd, sg in arms},
               n_cells=len(cells),
               script_sha1=sha12(os.path.abspath(__file__)))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {a.out}")
    if win["chosen"] is None:
        print("\n  ОКНА НЕТ. Ни одна sigma не даёт медианного изменения от 1% "
              "диапазона,\n  оставаясь безопасной на ОБЕИХ головах. PPO в "
              "таком виде запускать нельзя.")
        raise SystemExit(1)
    print(f"\n  ОКНО ЕСТЬ. Годные sigma: {win['passing']}; для PPO берём "
          f"МИНИМАЛЬНУЮ: {win['chosen']}")
    print("  ЭТО ИНЖЕНЕРНЫЙ ФИЛЬТР, НЕ ДОКАЗАТЕЛЬСТВО не-худшести: правило "
          "не использует\n  интервалов, а при 50 эпизодах на ячейку они были "
          "бы широкими.\n  Окончательная оценка RL — на НОВЫХ начальных "
          "состояниях.")


if __name__ == "__main__":
    main()
