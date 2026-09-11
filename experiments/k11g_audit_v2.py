"""K-11g: СТРОГИЙ пересчёт результата. Отдельным файлом, а не правкой.

ЗАЧЕМ ОТДЕЛЬНЫЙ ФАЙЛ. `k11g_protocol.py` с sha fa5964396866 — тот код, которым
вердикт был получен, и его sha записан в логах прогона. Править его значило бы
задним числом менять инструмент, выдавший результат. Здесь пересчёт по более
строгим правилам; если вердикт совпадёт, это независимое подтверждение, если
нет — прежний результат подлежит отзыву.

ЧТО ЗДЕСЬ СТРОЖЕ, ЧЕМ В АНАЛИЗАТОРЕ ПРОГОНА. Четыре лазейки, найденные
проверяющим агентом уже после прогона:

  1. `analyze` вызывал `check_cell` (сверку с протоколом), но НЕ
     `k11g_cell.check_cell_self` (внутреннюю непротиворечивость ячейки).
     Значит ячейка с подделанной сводкой или с правдоподобием в
     детерминированном режиме прошла бы сверку с протоколом.
  2. `check_shared_initial_states` сверяла только ПРИСУТСТВУЮЩИЕ руки и не
     требовала ровно 14 записей на каждый pair_key. Воспроизведённый обход:
     4 эпизода в одной ячейке и 6 в другой дают в сумме 50, и набор
     принимался.
  3. `discordance` считала `unpaired`, но анализатор при `unpaired > 0` не
     отказывал.
  4. Проверка протокола была fail-open: `protocol_script_sha1` принимался
     через `setdefault`, обязательными были лишь несколько полей, а
     `check_cell` МОЛЧА пропускал сравнение поля, отсутствующего в протоколе.

Запуск:
    python3 experiments/k11g_audit_v2.py --selftest
    python3 experiments/k11g_audit_v2.py --proto data/k11g/protocol.json \
        --cells data/k11g/cells --out data/k11g/analysis/window_v2.json
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import k11g_cell as kc          # noqa: E402
import k11g_protocol as kp      # noqa: E402

# ПОЛНЫЙ НАБОР ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ ПРОТОКОЛА. Прежде отсутствие поля означало
# «сравнение пропускается», то есть удалив строку из protocol.json можно было
# снять проверку.
REQUIRED_PROTOCOL_FIELDS = (
    "run_tag", "ckpt", "suite", "tasks", "sigmas", "n_envs", "init_start",
    "eps_salt", "horizon", "max_steps", "waiting_steps", "ensemble", "seed",
    "rollout_seed_mode", "rollout_seed", "device", "preprocess", "image_size",
    "dtype", "joint_sha1", "head_s0_sha1", "head_s1_sha1", "res_norm_sha1",
    "basis_sha1", "rho_sha1", "offset_table_sha1", "cell_script_sha1",
    "hicora_g_sha1", "hicora_vla_sha1", "joint12_vla_sha1", "k9h_sha1",
    "min_rms", "max_sat", "max_drop_pp",
)


def sha12(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:12]


def check_protocol_complete(proto, analyzer_sha=None):
    """Все поля обязательны; sha анализатора сверяется явно."""
    bad = [f"нет поля {f}" for f in REQUIRED_PROTOCOL_FIELDS
           if proto.get(f) is None]
    for f in ("min_rms", "max_sat", "max_drop_pp"):
        want = {"min_rms": kp.MIN_RMS, "max_sat": kp.MAX_SAT,
                "max_drop_pp": kp.MAX_DROP}[f]
        if proto.get(f) is not None and abs(float(proto[f]) - want) > 1e-12:
            bad.append(f"{f}={proto[f]} вместо зарегистрированного {want}")
    if 0.0 not in [float(x) for x in (proto.get("sigmas") or [])]:
        bad.append("в сетке нет sigma=0: нет детерминированной опоры")
    if proto.get("head_s0_sha1") == proto.get("head_s1_sha1"):
        bad.append("головы s0 и s1 — один файл")
    # SHA АНАЛИЗАТОРА СВЕРЯЕТСЯ ЯВНО. В протоколе лежит sha того кода, которым
    # считался вердикт прогона; здесь он ДРУГОЙ по построению, и это должно
    # быть видно, а не проскочить через setdefault.
    got = proto.get("protocol_script_sha1")
    if not got:
        bad.append("нет protocol_script_sha1: чем считался вердикт — неизвестно")
    if bad:
        raise SystemExit("ПРОТОКОЛ НЕПОЛОН ИЛИ НЕ ЗАРЕГИСТРИРОВАН:\n    "
                         + "\n    ".join(bad))
    return dict(protocol_script_sha1=got, analyzer_sha1=analyzer_sha,
                same_as_run=bool(analyzer_sha and got == analyzer_sha))


def check_cell_vs_protocol_strict(cell, proto, head, sigma, task):
    """Как `kp.check_cell`, но БЕЗ пропуска отсутствующих в протоколе полей."""
    bad = []
    for cf, pf in kp.CELL_VS_PROTO:
        if pf not in proto:
            bad.append(f"{pf}: нет в протоколе, сравнение {cf} невозможно")
        elif cell.get(cf) is None:
            bad.append(f"{cf}: нет в ячейке")
        elif str(cell[cf]) != str(proto[pf]):
            bad.append(f"{cf}: {cell[cf]!r} против {proto[pf]!r}")
    if bad:
        raise SystemExit(
            f"ЯЧЕЙКА НЕ СООТВЕТСТВУЕТ ПРОТОКОЛУ СТРОГО ({head}, "
            f"sigma={sigma}, задача {task}):\n    " + "\n    ".join(bad))
    kp.check_cell(cell, proto, head, sigma, task)
    return True


def check_arms_complete(cells, proto):
    """У КАЖДОГО pair_key обязаны быть ВСЕ руки, и ровно по одной записи.

    Воспроизведённый обход: 4 эпизода в одной ячейке и 6 в другой дают в сумме
    50, и прежняя проверка, сверявшая лишь присутствующие руки, это принимала.
    """
    arms = [(hd, float(sg)) for hd in kp.HEADS
            for sg in proto["sigmas"]]
    n_arms = len(arms)
    per_key = {}
    for (hd, sg, t), c in cells.items():
        for e in c["episodes"]:
            per_key.setdefault(e["pair_key"], []).append((hd, sg, t))
    want_keys = {f"{proto['suite']}|{t}|{proto['init_start'] + i}"
                 for t in proto["tasks"]
                 for i in range(int(proto["n_envs"]))}
    bad = []
    missing = sorted(want_keys - set(per_key))
    extra = sorted(set(per_key) - want_keys)
    if missing:
        bad.append(f"нет ключей: {missing[:5]} (всего {len(missing)})")
    if extra:
        bad.append(f"посторонние ключи: {extra[:5]} (всего {len(extra)})")
    for k in sorted(per_key):
        got = per_key[k]
        if len(got) != n_arms:
            bad.append(f"{k}: записей {len(got)} вместо {n_arms}")
        elif len({(a, s) for a, s, _ in got}) != n_arms:
            bad.append(f"{k}: руки повторяются или не все")
    if bad:
        raise SystemExit(
            f"НАБОР РУК НЕПОЛОН (ожидалось {n_arms} рук на каждый из "
            f"{len(want_keys)} ключей):\n    " + "\n    ".join(bad[:8])
            + ("\n    ..." if len(bad) > 8 else ""))
    return dict(keys=len(want_keys), arms_per_key=n_arms)


def discordance_strict(test_eps, ref_eps):
    """Как `kp.discordance`, но ЛЮБАЯ непарная запись — отказ."""
    d = kp.discordance(test_eps, ref_eps)
    if d["unpaired"]:
        raise SystemExit(
            f"НЕПАРНЫХ ЗАПИСЕЙ {d['unpaired']}: парное сравнение исходов на "
            f"таком наборе\n  не является парным.")
    return d


def selftest():
    # --- полнота протокола --------------------------------------------------
    proto = dict(run_tag="k11gT", ckpt="A/B", suite="10", tasks=[0, 1],
                 sigmas=[0.0, 0.07], n_envs=2, init_start=40, eps_salt=1,
                 horizon=8, max_steps=600, waiting_steps=10, ensemble="off",
                 seed=0, rollout_seed_mode="block", rollout_seed=40000,
                 device="cuda:0", preprocess="pp", image_size=224,
                 dtype="float16", joint_sha1="wj", head_s0_sha1="h0",
                 head_s1_sha1="h1", res_norm_sha1="rn", basis_sha1="bs",
                 rho_sha1="rh", offset_table_sha1="ot", cell_script_sha1="cs",
                 hicora_g_sha1="hg", hicora_vla_sha1="hv",
                 joint12_vla_sha1="jv", k9h_sha1="k9", min_rms=kp.MIN_RMS,
                 max_sat=kp.MAX_SAT, max_drop_pp=kp.MAX_DROP,
                 protocol_script_sha1="old")
    info = check_protocol_complete(proto, analyzer_sha="new")
    assert info["same_as_run"] is False
    assert check_protocol_complete(proto, "old")["same_as_run"] is True
    # КАЖДОЕ обязательное поле: удаление — отказ. Прежде это снимало проверку.
    for f in REQUIRED_PROTOCOL_FIELDS:
        try:
            check_protocol_complete({k: v for k, v in proto.items() if k != f},
                                    "x")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"удаление поля {f} принято")
    for mut in (dict(protocol_script_sha1=None), dict(max_drop_pp=40.0),
                dict(min_rms=0.001), dict(max_sat=0.9),
                dict(sigmas=[0.07]), dict(head_s1_sha1="h0")):
        try:
            check_protocol_complete(dict(proto, **mut), "x")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"принято: {mut}")

    # --- полнота рук на каждый ключ ----------------------------------------
    def mk(hd, sg, t, n=2, succ=2):
        eps = [dict(env_index=i, init_state_id=40 + i,
                    pair_key=f"10|{t}|{40+i}", success=(i < succ),
                    init_hash="a", init_hash_full=f"H{t}_{40+i}",
                    env_steps=80, policy_calls=2, rollout_seed=40000)
               for i in range(n)]
        ch = [dict(call=j // max(n, 1), env_index=j % max(n, 1), rms=0.013,
                   max=0.03, changed=True, grip_flip=False, sat_frac=0.0,
                   dz_frac=0.5, layers_run=24,
                   log_prob_u=(None if sg == 0.0 else -10.0))
              for j in range(2 * n)]
        c = dict(run_tag="k11gT", head=hd, sigma=sg, task_id=t,
                 mode=("deterministic" if sg == 0.0 else "gaussian"),
                 ckpt="A/B", suite="10", n_envs=n, init_start=40, horizon=8,
                 max_steps=600, waiting_steps=10, ensemble="off", seed=0,
                 rollout_seed_mode="block", rollout_seed=40000, eps_salt=1,
                 preprocess="pp", image_size=224, dtype="float16",
                 joint_sha1="wj", head_sha1=("h0" if hd == "s0" else "h1"),
                 res_norm_sha1="rn", basis_sha1="bs", rho_sha1="rh",
                 offset_table_sha1="ot", script_sha1="cs", hicora_g_sha1="hg",
                 hicora_vla_sha1="hv", joint12_vla_sha1="jv", k9h_sha1="k9",
                 device="cuda:0", min_rms=kp.MIN_RMS, pos_offset=4,
                 eps_sha1=(None if sg == 0.0 else f"E{t}"),
                 parity=dict(ok=True), episodes=eps, chunks=ch)
        c["summary"] = kc.summarize(ch, eps, c["mode"])
        return c

    good = {(hd, sg, t): mk(hd, sg, t) for hd in kp.HEADS
            for sg in (0.0, 0.07) for t in (0, 1)}
    assert check_arms_complete(good, proto)["arms_per_key"] == 4
    # ВОСПРОИЗВЕДЁННЫЙ ОБХОД: 1 эпизод в одной ячейке и 3 в другой.
    skew = dict(good)
    skew[("s1", 0.07, 0)] = mk("s1", 0.07, 0, n=1, succ=1)
    try:
        check_arms_complete(skew, proto)
    except SystemExit as e:
        assert "записей" in str(e) or "нет ключей" in str(e), str(e)
    else:
        raise AssertionError("перекошенный набор эпизодов принят")
    gone = {k: v for k, v in good.items() if k != ("s1", 0.07, 1)}
    try:
        check_arms_complete(gone, proto)
    except SystemExit:
        pass
    else:
        raise AssertionError("отсутствующая рука принята")

    # --- непарность — отказ -------------------------------------------------
    ref = [dict(pair_key=f"10|0|{i}", success=True) for i in range(4)]
    tst = [dict(pair_key=f"10|0|{i}", success=(i > 0)) for i in range(4)]
    assert discordance_strict(tst, ref)["loss"] == 1
    try:
        discordance_strict(tst[:3], ref)
    except SystemExit:
        pass
    else:
        raise AssertionError("непарная запись принята")

    # --- строгая сверка с протоколом ---------------------------------------
    assert check_cell_vs_protocol_strict(mk("s0", 0.07, 0), proto,
                                         "s0", 0.07, 0)
    # УДАЛЕНИЕ ПОЛЯ ИЗ ПРОТОКОЛА больше не снимает сравнение.
    thin = {k: v for k, v in proto.items() if k != "preprocess"}
    try:
        check_cell_vs_protocol_strict(mk("s0", 0.07, 0), thin, "s0", 0.07, 0)
    except SystemExit as e:
        assert "нет в протоколе" in str(e), str(e)
    else:
        raise AssertionError("отсутствие поля в протоколе сняло проверку")

    # --- check_cell_self вызывается ----------------------------------------
    forged = mk("s0", 0.07, 0)
    forged["summary"] = dict(forged["summary"], rms_median=0.999)
    try:
        kc.check_cell_self(forged)
    except SystemExit:
        pass
    else:
        raise AssertionError("подделанная сводка прошла check_cell_self")

    print("самопроверка k11g_audit_v2 пройдена: все 34 поля протокола "
          "обязательны,\n  sha анализатора сверяется явно, у каждого ключа "
          "требуется полный набор рук\n  и ровно по одной записи, "
          "перекошенное число эпизодов отвергается,\n  любая непарная запись "
          "— отказ, отсутствие поля в протоколе больше не\n  снимает сравнение")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--proto", default="data/k11g/protocol.json")
    ap.add_argument("--cells", default="data/k11g/cells")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    me = sha12(os.path.abspath(__file__))
    proto = kp.load_json(a.proto, "протокол")
    info = check_protocol_complete(proto, kp.sha12(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "k11g_protocol.py")))
    print(f"  аудитор {me}, анализатор прогона "
          f"{info['protocol_script_sha1']}, текущий k11g_protocol "
          f"{info['analyzer_sha1']}"
          + ("" if info["same_as_run"] else
             "  — ОТЛИЧАЕТСЯ, пересчёт независимый"))

    cells = {}
    for t in proto["tasks"]:
        for hd in kp.HEADS:
            for sg in proto["sigmas"]:
                f = os.path.join(a.cells, kp.cell_name(t, hd, sg))
                if not os.path.exists(f):
                    raise SystemExit(f"нет ячейки {os.path.basename(f)}")
                c = kp.load_json(f, "ячейка")
                check_cell_vs_protocol_strict(c, proto, hd, float(sg), t)
                kc.check_cell_self(c)          # ВНУТРЕННЯЯ НЕПРОТИВОРЕЧИВОСТЬ
                cells[(hd, float(sg), t)] = c
    print(f"  ячеек {len(cells)}, каждая сверена с протоколом СТРОГО и прошла "
          f"check_cell_self")
    kp.check_eps_shared(cells)
    kp.check_shared_initial_states(cells)
    ac = check_arms_complete(cells, proto)
    print(f"  ключей {ac['keys']}, на каждом полный набор из "
          f"{ac['arms_per_key']} рук, по одной записи")

    n_exp = int(proto["n_envs"]) * len(proto["tasks"])
    arms, disc = {}, {}
    for hd in kp.HEADS:
        for sg in [float(x) for x in proto["sigmas"]]:
            arm = kp.pool([cells[(hd, sg, t)] for t in proto["tasks"]])
            if arm["episodes"] != n_exp:
                raise SystemExit(f"{hd}/{sg}: эпизодов {arm['episodes']} "
                                 f"вместо {n_exp}")
            if sg > 0 and arm["rms_n_episodes"] != n_exp:
                raise SystemExit(f"{hd}/{sg}: RMS по "
                                 f"{arm['rms_n_episodes']} эпизодам")
            arms[(hd, sg)] = arm
    for hd in kp.HEADS:
        for sg in [float(x) for x in proto["sigmas"]]:
            if sg == 0.0:
                continue
            disc[f"{hd}|{sg}"] = discordance_strict(
                arms[(hd, sg)]["episodes_rows"],
                arms[(hd, 0.0)]["episodes_rows"])
    win = kp.read_window(arms, min_rms=float(proto["min_rms"]),
                         max_sat=float(proto["max_sat"]),
                         max_drop=float(proto["max_drop_pp"]))
    sat_max = max(c["summary"]["sat_frac_mean"] for c in cells.values())
    sat_chunk = max(r["sat_frac"] for c in cells.values()
                    for r in c["chunks"])

    print(f"\n  СТРОГИЙ ПЕРЕСЧЁТ: RMS >= {proto['min_rms']}, насыщение < "
          f"{proto['max_sat']}, падение <= {proto['max_drop_pp']} пп")
    print(f"    {'sigma':>6}{'голова':>8}{'RMS':>9}{'нас':>8}{'усп':>8}"
          f"{'пад пп':>8}{'иссл':>6}{'без':>5}{'вердикт':>9}")
    for sg in sorted(win["per_sigma"]):
        for hd in kp.HEADS:
            r = win["per_sigma"][sg]["per_head"][hd]
            print(f"    {sg:>6.2f}{hd:>8}{r['rms_median']:>9.4f}"
                  f"{100 * r['sat']:>7.4f}%{100 * r['success']:>7.1f}%"
                  f"{r['drop_pp']:>+8.1f}{('да' if r['ok_rms'] else 'НЕТ'):>6}"
                  f"{('да' if r['ok_sat'] and r['ok_drop'] else 'НЕТ'):>5}"
                  f"{('годна' if win['per_sigma'][sg]['ok'] else '-'):>9}")
    print(f"\n  насыщение: максимум по ячейке {100 * sat_max:.4f}%, по "
          f"отдельному чанку {100 * sat_chunk:.4f}%")
    print(f"  дискордантность (не критерий): "
          + ", ".join(f"{k} {100 * v['discordant_frac']:.0f}%"
                      f"/{v['paired_diff_pp']:+.0f}"
                      for k, v in sorted(disc.items())))
    out = dict(auditor_sha1=me, protocol=proto, protocol_info=info,
               window=win, discordance=disc, n_cells=len(cells),
               arms_per_key=ac, sat_max_cell=sat_max,
               sat_max_chunk=sat_chunk,
               arms={f"{hd}|{sg}": {k: v for k, v in arms[(hd, sg)].items()
                                    if k != "episodes_rows"}
                     for hd, sg in arms})
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {a.out}")
    if win["chosen"] is None:
        print("\n  СТРОГИЙ ПЕРЕСЧЁТ: ОКНА НЕТ. Прежний вердикт подлежит "
              "отзыву.")
        raise SystemExit(1)
    print(f"\n  СТРОГИЙ ПЕРЕСЧЁТ ПОДТВЕРЖДАЕТ: годные sigma "
          f"{win['passing']}, минимальная {win['chosen']}")


if __name__ == "__main__":
    main()
