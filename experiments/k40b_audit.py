#!/usr/bin/env python3
"""K-40 B, CPU-аудит: доступность архива c2 и K-21, объём трёх наборов.

1. c2: план (полная сверка кода и головок); все 12 частей — штатной полной
   проверкой K-37 (k37_confirm.build_merge: журналы, записи, точное
   множество 2880 eid, у каждой сцены все 6 эпизодов); пересчитанные строки
   сверяются с merged.json; наличие k_a2 / j_a_codec в T2 и T2P.
   Статистика — только по проверенным строкам.
2. K-21: prep.npz и кадры (load_prepared без учителя), train_rows/val_rows.
3. Спасения: для case_id × seed с провалом T1 — успешный T2P, иначе
   успешный T2. Разбиение train/val 80/20 по физическим сценам (хеш внутри
   задачи; оба сида сцены в одной части). В группе задача × сид × split:
   m = min(N_спасений, ⌊N_успехов T1 / 2⌋); спасения R, успехи T1a
   (общие) и T1b (только контроль) — первые по хешу. Смешанный набор
   R + T1a, контроль TreeB-T1 — T1a + T1b; повторов 3m. Окна (до 8
   полных восьмишаговых на эпизод): W_R, W_T1a, W_T1b и объём после
   выравнивания min(W_R, W_T1b) + W_T1a на набор.
При отсутствии входов отчёт всё равно пишется со списком проблем.
Выход: reports/k40b/audit.json (архив не изменяется).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

PROTOCOL = "k40b-audit-v1"
VAL_SHARE = 0.2
WIN, MAX_WIN = 8, 8
N_EXPECTED = 2880
CMP = ("eid", "part", "task", "case_id", "seed", "policy", "success",
       "phys")


def rank(key):
    return hashlib.sha256(f"{PROTOCOL}/{key}".encode()).hexdigest()


def split_scenes(rows):
    """case_id -> train | val; 20% сцен каждой задачи по хешу."""
    out = {}
    for t in sorted({r["task"] for r in rows}):
        cs = sorted({r["case_id"] for r in rows if r["task"] == t}, key=rank)
        n_val = round(VAL_SHARE * len(cs))
        for i, c in enumerate(cs):
            out[c] = "val" if i < n_val else "train"
    return out


def windows(phys):
    return min(MAX_WIN, max((phys - 10) // WIN, 0))


def budget(rows):
    """Проверенные строки c2 -> группы, наборы, повторы, окна."""
    by = defaultdict(dict)
    for r in rows:
        key = (r["task"], r["case_id"], r["seed"])
        if r["policy"] in by[key]:
            raise SystemExit(f"{key} {r['policy']}: дубль")
        by[key][r["policy"]] = r
    scenes = {(t, c) for t, c, _s in by}
    for t, c in scenes:
        for s in (0, 1):
            if set(by.get((t, c, s), {})) != {"T1", "T2", "T2P"}:
                raise SystemExit(f"{t}/{c}/s{s}: неполный набор политик")
    split = split_scenes(rows)
    resc, t1s = defaultdict(list), defaultdict(list)
    src = Counter()
    for (t, c, s), d in sorted(by.items()):
        g = (t, s, split[c])
        if d["T1"]["success"]:
            t1s[g].append(d["T1"])
            continue
        pick = "T2P" if d["T2P"]["success"] else (
            "T2" if d["T2"]["success"] else None)
        if pick:
            resc[g].append(d[pick])
            src[pick] += 1
    groups, tot = {}, Counter()
    for g in sorted(set(resc) | set(t1s)):
        nr, ns = len(resc[g]), len(t1s[g])
        m = min(nr, ns // 2)
        R = sorted(resc[g], key=lambda r: rank(r["eid"]))[:m]
        T = sorted(t1s[g], key=lambda r: rank(r["eid"]))
        Ta, Tb = T[:m], T[m:2 * m]
        wr, wa, wb = (sum(windows(r["phys"]) for r in x) for x in (R, Ta, Tb))
        aligned = min(wr, wb) + wa
        groups["t{}_s{}_{}".format(*g)] = dict(
            rescues=nr, t1_success=ns, m=m, replays=3 * m, win_R=wr,
            win_T1a=wa, win_T1b=wb, win_aligned_per_set=aligned)
        for k_, v in (("rescues", nr), ("m", m), ("replays", 3 * m),
                      (f"m_{g[2]}", m), ("win_R", wr), ("win_T1a", wa),
                      ("win_T1b", wb), ("win_aligned_per_set", aligned)):
            tot[k_] += v
    return dict(groups=groups, totals=dict(tot), rescue_source=dict(src),
                n_scenes=len(split),
                val_scenes=sum(v == "val" for v in split.values()))


def verified_rows(archive, rep):
    """-> проверенные строки c2 или None; проблемы — в rep."""
    import k36_protocol as pc
    import k37_confirm as kcf
    import k36_harness as kh
    try:
        plan, k36 = kcf.load_plan(archive)
    except (SystemExit, OSError, KeyError, ValueError) as e:
        rep["problems"].append(f"план c2: {type(e).__name__}: {e}")
        return None
    rep["archive_plan"] = plan["sha256"]
    missing = [k for k in range(kcf.N_PARTS)
               if not Path(kcf.part_dir(archive, k)).is_dir()]
    if missing:
        rep["problems"].append(f"нет каталогов частей {missing}")
        return None
    prob, rows, parts = kcf.build_merge(archive, plan, k36)
    rep["parts"] = parts
    if prob:
        rep["problems"] += [f"c2: {x}" for x in prob[:10]]
        return None
    if len(rows) != N_EXPECTED or len({r["eid"] for r in rows}) != N_EXPECTED:
        rep["problems"].append(f"строк {len(rows)} ≠ {N_EXPECTED}")
        return None
    mp = Path(archive) / "merged.json"
    if not mp.exists():
        rep["problems"].append("нет merged.json")
    else:
        m = pc.load_json(str(mp))
        old = {r["eid"]: {k: r.get(k) for k in CMP} for r in m.get("rows", [])}
        new = {r["eid"]: {k: r.get(k) for k in CMP} for r in rows}
        if m.get("ok") is not True or m.get("plan_sha256") != plan["sha256"]:
            rep["problems"].append("merged.json не подтверждает план")
        if old != new:
            diff = sorted(e for e in set(old) | set(new)
                          if old.get(e) != new.get(e))
            rep["problems"].append(f"merged.json ≠ проверенным записям: "
                                   f"{len(diff)} eid, напр. {diff[:2]}")
    keys = {}
    for k in range(kcf.N_PARTS):
        kcf.install(kcf.part_dir(archive, k))
        for w in kcf.part_wants(plan, k36, k):
            pol = dict(T="T1").get(w["kind"], w["kind"])
            if pol in keys:
                continue
            got = kh.load_ep(w)
            if got is not None:
                keys[pol] = sorted(x for x in got[1] if x in (
                    "j_a_codec", "k_a2", "actions"))
        if len(keys) == 3:
            break
    rep["npz_keys"] = keys
    for pol in ("T2", "T2P"):
        if "k_a2" not in keys.get(pol, []):
            rep["problems"].append(f"{pol}: в записи нет k_a2")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default="reports/k37c/runs/c2")
    ap.add_argument("--out", default="reports/k40b/audit.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import k36_protocol as pc
    rep = dict(protocol=PROTOCOL, archive=a.archive, problems=[])
    rows = verified_rows(a.archive, rep)
    try:
        import k21_data as k21
        D = k21.load_prepared(require_teacher=False)
        rep["k21"] = dict(ok=True, train_rows=int(len(D["train_rows"])),
                          val_rows=int(len(D["val_rows"])),
                          images=D["prep_report"]["inputs"]["images"]["path"])
    except (SystemExit, OSError, KeyError, ValueError) as e:
        rep["k21"] = dict(ok=False, error=f"{type(e).__name__}: {e}")
        rep["problems"].append(f"K-21: {e}")
    if rows:
        try:
            rep["budget"] = budget(rows)
        except SystemExit as e:
            rep["problems"].append(f"бюджет: {e}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pc.save_json(a.out, rep)
    print(f"  c2: план {rep.get('archive_plan', '—')[:16]}; строк "
          f"{len(rows) if rows else 0}; ключи {rep.get('npz_keys')}")
    print(f"  K-21: {rep['k21']}")
    if "budget" in rep:
        b = rep["budget"]
        print(f"  спасений {b['totals'].get('rescues')} (источник "
              f"{b['rescue_source']}); сцен {b['n_scenes']}, val "
              f"{b['val_scenes']}")
        for g, v in b["groups"].items():
            print(f"    {g}: спасений {v['rescues']}, успехов T1 "
                  f"{v['t1_success']}, m {v['m']}, повторов {v['replays']}; "
                  f"окна R/T1a/T1b {v['win_R']}/{v['win_T1a']}/"
                  f"{v['win_T1b']}, после выравнивания "
                  f"{v['win_aligned_per_set']} на набор")
        t = b["totals"]
        print(f"  итого m {t.get('m')} (train {t.get('m_train', 0)}, val "
              f"{t.get('m_val', 0)}); повторов архива {t.get('replays')}; "
              f"окон на набор после выравнивания "
              f"{t.get('win_aligned_per_set')}")
    print("  аудит: " + ("ок" if not rep["problems"] else
                         "ПРОБЛЕМЫ: " + "; ".join(rep["problems"][:5])))
    return 0 if not rep["problems"] else 3


def selftest():
    rows = []
    for t in (0, 6, 8):
        for c in range(20):
            for s in (0, 1):
                fail = (c + s + t) % 4 == 0
                for pol in ("T1", "T2", "T2P"):
                    ok = not fail if pol == "T1" else (
                        (pol == "T2P" and c % 2 == 0) or
                        (pol == "T2" and c % 3 == 0))
                    rows.append(dict(eid=f"{t}/{c}/{s}/{pol}", task=t,
                                     case_id=f"t{t}_c{c}", seed=s,
                                     policy=pol, success=ok,
                                     phys=210 if pol == "T1" else 58))
    sp = split_scenes(rows)
    assert sum(v == "val" for v in sp.values()) == 12   # 4 из 20 на задачу
    b = budget(rows)
    for g, v in b["groups"].items():
        assert v["m"] == min(v["rescues"], v["t1_success"] // 2)
        assert v["replays"] == 3 * v["m"]
        assert v["win_R"] == 6 * v["m"] and v["win_T1a"] == 8 * v["m"]
        assert v["win_aligned_per_set"] == min(6 * v["m"], 8 * v["m"]) + \
            8 * v["m"]
    n_resc = 0
    for t in (0, 6, 8):
        for c in range(20):
            for s in (0, 1):
                if (c + s + t) % 4 == 0 and (c % 2 == 0 or c % 3 == 0):
                    n_resc += 1
    assert b["totals"]["rescues"] == n_resc
    assert windows(210) == 8 and windows(50) == 5
    for bad in (rows[:-3], rows + [dict(rows[0])]):
        try:
            budget(bad)
            raise AssertionError("неполный набор или дубль принят")
        except SystemExit:
            pass
    print("самопроверка k40b_audit пройдена: разбиение 80/20 по задачам, "
          "спасения T2P → T2, m = min(спасений, ⌊успехов/2⌋), повторы 3m, "
          "окна и выравнивание; неполный старт и дубль — отказ")
    return 0


if __name__ == "__main__":
    import sys
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
