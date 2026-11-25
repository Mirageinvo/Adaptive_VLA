#!/usr/bin/env python3
"""K-40 B, CPU-аудит: доступность архива c2 и K-21, объём трёх наборов.

1. c2: план (с полной сверкой кода и головок), merged.json, все 12 частей
   полной проверкой K-37 (part_problems), наличие k_a2 / j_a_codec в
   записях T2 и T2P.
2. K-21: prep.npz и кадры (load_prepared без учителя), число строк.
3. Спасения: для case_id × seed с провалом T1 — успешный T2P, иначе
   успешный T2. Разбиение train/val 80/20 по физическим сценам (хеш внутри
   задачи; оба сида сцены в одной части). В группе задача × сид × split:
   m = min(N_спасений, ⌊N_успехов T1 / 2⌋). Смешанный набор: m спасений +
   m успехов T1; контроль TreeB-T1: те же m успехов T1 + m других.
   Повторов архива: m спасений + 2m успехов T1 на группу.
Выход: reports/k40b/audit.json (архив не изменяется).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

PROTOCOL = "k40b-audit-v1"
VAL_SHARE = 0.2
WIN, MAX_WIN = 8, 8


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
    steps = phys - 10
    return min(MAX_WIN, max(steps // WIN, 0))


def budget(rows):
    """rows: merged.json c2 -> группы, наборы, объём повторов."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["task"], r["case_id"], r["seed"])][r["policy"]] = r
    split = split_scenes(rows)
    resc, t1s = defaultdict(list), defaultdict(list)
    src = Counter()
    for (t, c, s), d in sorted(by.items()):
        if set(d) != {"T1", "T2", "T2P"}:
            raise SystemExit(f"{t}/{c}/s{s}: неполный набор политик")
        g = (t, s, split[c])
        if d["T1"]["success"]:
            t1s[g].append(d["T1"])
            continue
        pick = "T2P" if d["T2P"]["success"] else (
            "T2" if d["T2"]["success"] else None)
        if pick:
            resc[g].append(d[pick])
            src[pick] += 1
    groups = {}
    tot = Counter()
    for g in sorted(set(resc) | set(t1s)):
        nr, ns = len(resc[g]), len(t1s[g])
        m = min(nr, ns // 2)
        groups["t{}_s{}_{}".format(*g)] = dict(
            rescues=nr, t1_success=ns, m=m, replays=3 * m,
            windows_rescue=sum(windows(r["phys"])
                               for r in sorted(resc[g], key=lambda r: rank(
                                   r["eid"]))[:m]))
        tot["rescues"] += nr
        tot["m"] += m
        tot["replays"] += 3 * m
        tot[f"m_{g[2]}"] += m
    return dict(groups=groups, totals=dict(tot), rescue_source=dict(src),
                n_scenes=len(split),
                val_scenes=sum(v == "val" for v in split.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default="reports/k37c/runs/c2")
    ap.add_argument("--out", default="reports/k40b/audit.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    rep = dict(protocol=PROTOCOL, archive=a.archive, problems=[])
    try:
        plan, k36 = kcf.load_plan(a.archive)
        rep["archive_plan"] = plan["sha256"]
    except SystemExit as e:
        rep["problems"].append(f"план c2: {e}")
        plan = None
    rows = []
    if plan is not None:
        C = kh.consts(k36)
        mp = Path(a.archive) / "merged.json"
        if not mp.exists():
            rep["problems"].append("нет merged.json")
        else:
            m = pc.load_json(str(mp))
            if m.get("ok") is not True or m.get("plan_sha256") != \
                    plan["sha256"]:
                rep["problems"].append("merged.json не подтверждает план")
            rows = m.get("rows", [])
        parts = {}
        keys = Counter()
        for k in range(kcf.N_PARTS):
            if not Path(kcf.part_dir(a.archive, k)).is_dir():
                parts[str(k)] = "нет каталога"
                rep["problems"].append(f"часть {k}: нет каталога")
                continue
            r = kcf.part_problems(plan, k36, a.archive, k, C)
            parts[str(k)] = dict(ok=r.get("ok"), n=r.get("n"),
                                 problems=r.get("problems", [])[:3])
            if not r.get("ok"):
                rep["problems"].append(f"часть {k}: {r['problems'][:2]}")
                continue
            kcf.install(kcf.part_dir(a.archive, k))
            seen = set()
            for w in kcf.part_wants(plan, k36, k):
                pol = dict(T="T1").get(w["kind"], w["kind"])
                if pol in seen:
                    continue
                got = kh.load_ep(w)
                if got is not None:
                    seen.add(pol)
                    keys[pol] = sorted(x for x in got[1]
                                       if x.startswith(("j_a_codec", "k_a2",
                                                        "actions")))
        rep["parts"] = parts
        rep["npz_keys"] = {p: v for p, v in keys.items()}
        for pol in ("T2", "T2P"):
            if "k_a2" not in rep["npz_keys"].get(pol, []):
                rep["problems"].append(f"{pol}: в записи нет k_a2")
    try:
        import k21_data as k21
        D = k21.load_prepared(require_teacher=False)
        rep["k21"] = dict(ok=True, rows=int(len(D.get("rows_train",
                                                      D.get("rows", [])))),
                          keys=sorted(k for k in D if not k.startswith(
                              ("IMG", "prep_report")))[:20],
                          images=D["prep_report"]["inputs"]["images"]["path"])
    except SystemExit as e:
        rep["k21"] = dict(ok=False, error=str(e))
        rep["problems"].append(f"K-21: {e}")
    if rows:
        rep["budget"] = budget(rows)
    pc.save_json(a.out, rep)
    print(f"  c2: {'план ' + rep['archive_plan'][:16] if plan else 'нет'}; "
          f"частей ок {sum(1 for v in rep.get('parts', {}).values() if isinstance(v, dict) and v['ok'])}"
          f"/{kcf.N_PARTS}; ключи {rep.get('npz_keys')}")
    print(f"  K-21: {rep['k21']}")
    if rows:
        b = rep["budget"]
        print(f"  спасений {b['totals'].get('rescues')} (источник "
              f"{b['rescue_source']}); сцен {b['n_scenes']}, val "
              f"{b['val_scenes']}")
        for g, v in b["groups"].items():
            print(f"    {g}: спасений {v['rescues']}, успехов T1 "
                  f"{v['t1_success']}, m {v['m']}, повторов {v['replays']}, "
                  f"окон спасений {v['windows_rescue']}")
        print(f"  итого m {b['totals'].get('m')} (train "
              f"{b['totals'].get('m_train', 0)}, val "
              f"{b['totals'].get('m_val', 0)}); повторов архива "
              f"{b['totals'].get('replays')}")
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
                                     policy=pol, success=ok, phys=210))
    sp = split_scenes(rows)
    assert sum(v == "val" for v in sp.values()) == 12   # 4 из 20 на задачу
    b = budget(rows)
    for g, v in b["groups"].items():
        assert v["m"] == min(v["rescues"], v["t1_success"] // 2)
        assert v["replays"] == 3 * v["m"]
    n_fail = sum(1 for r in rows if r["policy"] == "T1" and not r["success"])
    n_resc = sum(1 for (t, c, s) in {(r["task"], r["case_id"], r["seed"])
                                     for r in rows}
                 if (int(c.split("_c")[1]) + s + t) % 4 == 0 and
                 (int(c.split("_c")[1]) % 2 == 0 or
                  int(c.split("_c")[1]) % 3 == 0))
    assert b["totals"]["rescues"] == n_resc <= n_fail
    assert windows(210) == 8 and windows(50) == 5
    try:
        budget(rows[:-1])
        raise AssertionError("неполный набор принят")
    except SystemExit:
        pass
    print("самопроверка k40b_audit пройдена: разбиение сцен 80/20 по "
          "задачам, спасения T2P → T2, m = min(спасений, ⌊успехов/2⌋), "
          "повторы 3m, неполный набор — отказ")
    return 0


if __name__ == "__main__":
    import sys
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
