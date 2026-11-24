#!/usr/bin/env python3
"""K-39 2×2: вклад обновлённого схвата и руки при пересчёте T32 в o4.

Задача 8, архив c2 (T1), сцены не из natural_probe/v1. Точка — онлайн-
триггер: первый чанк, где после четырёх исполненных шагов запланированный
хвост B[4:8] начинает закрытие (три open перед ним в исполненной истории и
плане, все доступные в B команды от него — close). Из общего o4:
  T1          рука B, схват B (архивный повтор)
  GripOnly    рука B, схват R[:4]
  ArmOnly     рука R[:4], схват B
  RefreshOnce рука R, схват R (R — полный T32 в o4, elapsed t + 4)
Выборка: 24 доступных провала и 24 доступных успеха T1 (12 + 12 на сид),
разные case_id, порядок по хешу. Главный критерий — взвешенный прирост SR
против T1: p_f · спасения/n_f − p_s · потери/n_s, где p_f, p_s — доли
доступных провалов и успехов среди всех T1 задачи 8 в c2. Сигнал: прирост
≥ +3 п.п. (бюджетное решение, ДИ — bootstrap по сценам внутри групп).

Команды: --prepare-only | --part K --device D (части 0/1) | --summarize.
"""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import socket

import numpy as np

from k31_collect import ENV, EXEC_MAX, Ledger

TASK = 8
PER_GROUP = 12                       # на сид и группу (провал / успех)
PROTOCOL = "k39-grip-only-v1"
ARMS = ("GripOnly", "ArmOnly", "RefreshOnce")
SCOPE = "grip_probe"
CASE_RESERVE = 4 * EXEC_MAX
N_PARTS = 2
THRESHOLD_PP = 3.0
BOOT = dict(n=50_000, seed=39002)
OPEN, CLOSE = -1.0, 1.0
CODE = ("k39_grip_probe.py", "k39_failure_probe.py", "k39_cache.py",
        "k39_pulse.py")


def rank(key):
    return hashlib.sha256(f"{PROTOCOL}/{key}".encode()).hexdigest()


def online_point(A, J):
    """A — исполненные действия, J — журнал (elapsed, a_codec). -> (g, t)
    первого срабатывания или None. g — абсолютный индекс первой close."""
    import k31_collect as kcol
    A = np.asarray(A, np.float64)
    for i in np.argsort(J["elapsed"]):
        t = int(J["elapsed"][i])
        if len(A) <= t + 4:                 # нужен живой o4 и хотя бы шаг хвоста
            break
        plan = -np.asarray(J["a_codec"][i], np.float64)[4:8, 6]
        hist = np.concatenate([[kcol.DUMMY[6]] * 3, A[:t + 4, 6]])
        seq = np.concatenate([hist, plan])          # смещение 3
        for j in range(4, 8):
            k = t + j + 3
            if (seq[k - 3:k] == OPEN).all() and \
                    (seq[k:t + 8 + 3] == CLOSE).all():
                return t + j, t
    return None


def select(jobs, used):
    out, taken = [], set(used)
    for seed in (0, 1):
        for succ in (False, True):
            pool = sorted((j for j in jobs if j["want"]["seed"] == seed and
                           j["success"] is succ),
                          key=lambda j: rank(j["want"]["eid"]))
            got = []
            for j in pool:
                if j["want"]["case_id"] in taken:
                    continue
                got.append(j)
                taken.add(j["want"]["case_id"])
                if len(got) == PER_GROUP:
                    break
            if len(got) < PER_GROUP:
                raise SystemExit(f"s{seed} {'успех' if succ else 'провал'}: "
                                 f"доступно {len(got)} < {PER_GROUP}")
            out += got
    out.sort(key=lambda j: (j["want"]["seed"], rank(j["want"]["eid"])))
    for i, j in enumerate(out):
        j["part"] = i % N_PARTS
    return out


def prepare(archive, v1):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf
    import k39_failure_probe as kfp
    plan, k36 = kcf.load_plan(str(archive))
    C = kh.consts(k36)
    merged = pc.load_json(archive / "merged.json")
    if merged.get("ok") is not True or \
            merged.get("plan_sha256") != plan["sha256"]:
        raise SystemExit("merged.json не подтверждает архивный план")
    rows = {r["eid"]: r for r in merged["rows"]
            if r["policy"] == "T1" and r["task"] == TASK}
    v1p = pc.load_json(v1 / "plan.json")
    if v1p.get("protocol") != kfp.PROTOCOL:
        raise SystemExit("v1: не план natural_probe")
    used = {j["want"]["case_id"] for j in v1p["jobs"]}
    cov, cand, n_all = Counter(), [], Counter()
    for part in range(kcf.N_PARTS):
        led = pc.load_json(Path(kcf.part_dir(str(archive), part)) /
                           "ledger.json")
        if led["pending"] is not None or led["caps"] != kcf.CAP_PART:
            raise SystemExit(f"часть {part}: открытый резерв или лимиты")
        for w in kcf.part_wants(plan, k36, part):
            if w["task"] != TASK or w["kind"] != "T":
                continue
            r = rows.pop(w["eid"], None)
            if r is None or r["part"] != part or r["seed"] != w["seed"] or \
                    r["case_id"] != w["case_id"]:
                raise SystemExit(f"{w['eid']}: merged.json ≠ плану")
            rec, z = kfp.reference(archive, part, w)
            succ = rec["summary"]["success"]
            pr = kh.episode_problems(rec, z, w, C)
            if pr or succ is not r["success"] or led["done"].get(
                    w["eid"]) != {"scope": "confirm", "actual": rec["phys"]}:
                raise SystemExit(f"{w['eid']}: архив не прошёл сверку ({pr})")
            grp = f"s{w['seed']}_{'success' if succ else 'fail'}"
            n_all[grp] += 1
            J = {k[2:]: z[k] for k in z if k.startswith("j_")}
            pt = online_point(z["actions"], J)
            arch = kfp.anchor(z["actions"])
            if arch is not None:
                cov[f"{grp}_archive_point"] += 1
            if pt is None:
                continue
            g, t = pt
            cov[f"{grp}_online"] += 1
            if w["case_id"] in used:
                cov[f"{grp}_online_used_v1"] += 1
                continue
            cand.append(dict(want=w, arch_part=part, success=succ, g=g, t=t,
                             physical=rec["phys"], npz_sha1=rec["npz_sha1"]))
    if rows:
        raise SystemExit("в merged.json остались записи T1 вне плана")
    N = sum(n_all.values())
    weights = dict(
        N=N, p_fail=sum(cov[f"s{s}_fail_online"] for s in (0, 1)) / N,
        p_success=sum(cov[f"s{s}_success_online"] for s in (0, 1)) / N,
        per_seed={str(s): dict(
            N=n_all[f"s{s}_fail"] + n_all[f"s{s}_success"],
            p_fail=cov[f"s{s}_fail_online"] /
            (n_all[f"s{s}_fail"] + n_all[f"s{s}_success"]),
            p_success=cov[f"s{s}_success_online"] /
            (n_all[f"s{s}_fail"] + n_all[f"s{s}_success"])) for s in (0, 1)})
    print(f"  t{TASK}: всего {dict(sorted(n_all.items()))}; триггер "
          f"{dict(sorted(cov.items()))}", flush=True)
    print(f"  веса: p_fail {weights['p_fail']:.4f}, p_success "
          f"{weights['p_success']:.4f} (N {N})", flush=True)
    jobs = select(cand, used)
    here = Path(__file__).resolve().parent
    return dict(protocol=PROTOCOL, archive_plan=plan["sha256"],
                v1_plan=v1p["sha256"], coverage=dict(cov),
                totals=dict(n_all), weights=weights,
                code={n: kcf.sha256(here / n) for n in CODE},
                threshold_pp=THRESHOLD_PP, boot=BOOT,
                caps={SCOPE: (len(jobs) // N_PARTS + 1) * CASE_RESERVE},
                jobs=jobs), k36, C


def grip_tail(B4, R4):
    t = np.array(B4, np.float32, copy=True)
    t[:, 6] = R4[:, 6]
    return t


def first_close(phys):
    i = np.flatnonzero(np.asarray(phys) == CLOSE)
    return int(i[0]) if len(i) else None


def arm_tail(B4, R4):
    t = np.array(R4, np.float32, copy=True)
    t[:, 6] = B4[:, 6]
    return t


def grip_info(B4, R4):
    import k39_failure_probe as kfp
    d = kfp.grip_commands(B4, R4)
    a, b = first_close(d["t1"]), first_close(d["refresh"])
    d["shift"] = ("same" if a == b else "removed" if b is None else
                  "added" if a is None else "later" if b > a else "earlier")
    return d


def branches(sim, io, arm, job, rec, z):
    from k31_collect import run_episode
    from k36_harness import SUMMARY, episode_number
    import k39_failure_probe as kfp
    w, t = job["want"], job["t"]
    eid = w["eid"]
    B, R4, o0, o4, hashes, frames = kfp.replay(sim, io, arm, job, rec, z)
    tails = dict(GripOnly=grip_tail(B[4:], R4), ArmOnly=arm_tail(B[4:], R4),
                 RefreshOnce=R4)
    k_ = int(rank(eid)[-1], 16) % len(ARMS)       # порядок ветвей по хешу
    order = list(ARMS[k_:] + ARMS[:k_])
    A = z["actions"]
    res = {}
    for name in order:
        suffix = tails[name]
        wrapped = kfp.CheckedMidpoint(sim, ENV["waiting_steps"] + t + 4, o4)
        arm.episode = episode_number(eid)
        ep = run_episode(wrapped, io, arm, w["task"], w["reset_id"],
                         prefix=A[:t], fork=dict(
                             tau=t, hash=o0,
                             a_codec=np.concatenate((B[:4], suffix))))
        if wrapped.count < wrapped.steps or \
                (ep["init_hash"], ep["init_hash_full"]) != hashes or \
                not np.array_equal(ep["actions"][:t + 4], A[:t + 4]):
            raise SystemExit(f"{eid} {name}: общий префикс ≠ архиву")
        if np.array_equal(suffix, B[4:]):
            keys = ("success", "success_step", "done_step", "steps",
                    "actions_sha1")
            if any(ep[k] != rec["summary"][k] for k in keys):
                raise SystemExit(f"{eid} {name}: хвост = B, эпизод ≠ T1")
        res[name] = {k: ep[k] for k in (*SUMMARY, "seconds")}
    for x in ARMS:
        for y in ARMS:
            if x < y and np.array_equal(tails[x], tails[y]) and any(
                    res[x][k] != res[y][k]
                    for k in ("success", "physical", "actions_sha1")):
                raise SystemExit(f"{eid}: одинаковые хвосты {x}/{y} дали "
                                 f"разные эпизоды")
    arm_d = R4[:, :6].astype(np.float64) - B[4:, :6]
    return dict(arms=res, o0=o0, o4=o4, grip=grip_info(B[4:], R4),
                grip_same=bool(np.array_equal(R4[:, 6], B[4:, 6])),
                arm_same=bool(np.array_equal(R4[:, :6], B[4:, :6])),
                arm_rms_codec=float(np.sqrt(np.mean(arm_d ** 2)))), frames


# --- запись, восстановление ---------------------------------------------------------------
def part_dir(out, k):
    return out / f"part-{int(k)}"


def scene_files(out, job, plan):
    w = job["want"]
    name = f"s{w['seed']}_{w['case_id']}"
    d = part_dir(out, job["part"])
    ident = dict(eid=w["eid"], plan=plan["sha256"], seed=w["seed"],
                 t1=job["success"], t=job["t"], g=job["g"],
                 archive_npz_sha1=job["npz_sha1"], physical=job["physical"])
    return name, d / f"{name}.json", d / f"{name}.png", ident


def checked_result(path, frame_path, ident):
    import k36_harness as kh
    import k36_protocol as pc
    row = pc.load_json(path)
    if not isinstance(row, dict) or any(
            type(row.get(k)) is not type(v) or row[k] != v
            for k, v in ident.items()) or not frame_path.is_file():
        raise SystemExit(f"{path}: запись не соответствует плану")
    if any(not isinstance(row.get(k), str) or not row[k]
           for k in ("o0", "o4")):
        raise SystemExit(f"{path}: нет хешей наблюдений")
    gr = row.get("grip")
    if not isinstance(gr, dict) or gr.get("changed_steps") != [
            i for i, (a, b) in enumerate(zip(gr.get("t1", []),
                                             gr.get("refresh", [])), 5)
            if a != b] or len(gr.get("t1", [])) != 4 or \
            row.get("grip_same") is not (not gr["changed_steps"]):
        raise SystemExit(f"{path}: неверные данные схвата")
    arms = row.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise SystemExit(f"{path}: неполный набор ветвей")
    for name, ep in arms.items():
        pr = kh.summary_problems(dict(phys=ep.get("physical")), ep,
                                 ep.get("steps"))
        if pr or ep["steps"] < row["t"] + 4:
            raise SystemExit(f"{path} {name}: неверная сводка ({pr})")
    if not isinstance(row.get("arm_same"), bool):
        raise SystemExit(f"{path}: нет arm_same")
    if row["grip_same"] and (arms["GripOnly"]["success"] != row["t1"] or
                             arms["ArmOnly"] != arms["RefreshOnce"]):
        raise SystemExit(f"{path}: схват R = B, а исходы не сходятся")
    if row["arm_same"] and (arms["ArmOnly"]["success"] != row["t1"] or
                            arms["GripOnly"] != arms["RefreshOnce"]):
        raise SystemExit(f"{path}: рука R = B, а исходы не сходятся")
    return row


def case_spend(row):
    return row["physical"] + sum(e["physical"] for e in row["arms"].values())


def restore_rows(out, plan, part, ledger):
    jobs = {f"{plan['sha256']}/{j['want']['eid']}": j for j in plan["jobs"]
            if j["part"] == part}
    done, pending = ledger.d["done"], ledger.d["pending"]
    pid = ledger.pending_id()
    if not set(done).issubset(jobs):
        raise SystemExit("ledger: учтённые сцены вне части")
    if pending is not None and (pid not in jobs or pid in done or pending !=
                                dict(id=pid, scope=SCOPE,
                                     amount=CASE_RESERVE)):
        raise SystemExit("ledger: открытый резерв не из части")
    for c in ledger.d["crashes"]:
        if c.get("id") not in jobs or c.get("scope") != SCOPE or \
                c.get("amount") != CASE_RESERVE:
            raise SystemExit("ledger: авария не из части")
    rows = {}
    for key, j in jobs.items():
        _n, path, fp, ident = scene_files(out, j, plan)
        if path.exists():
            row = checked_result(path, fp, ident)
            if key != pid and done.get(key) != dict(scope=SCOPE,
                                                    actual=case_spend(row)):
                raise SystemExit(f"{path}: расход ≠ записи журнала")
            rows[key] = row
        elif key in done:
            raise SystemExit(f"{path}: учтена, результата нет")
    spent = sum(case_spend(rows[k]) for k in done) + \
        CASE_RESERVE * len(ledger.d["crashes"])
    if ledger.d["spent"] != {SCOPE: spent}:
        raise SystemExit("ledger: расход не сходится")
    if pid in rows:
        ledger.settle(pid, case_spend(rows[pid]))
    return rows


def collect(a, plan, k36):
    import torch
    import k36_bank as kb36
    import k36_harness as kh
    import k36_protocol as pc
    import k37_bank as kbk
    import k37_confirm as kcf
    import k39_failure_probe as kfp
    from k38_grip_run import threads_problems
    if a.part not in range(N_PARTS):
        raise SystemExit(f"часть {a.part}")
    pd_ = part_dir(a.out, a.part)
    pd_.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(pd_ / "ledger.json"), plan["caps"])
    plock = kcf.Lock(str(pd_ / ".part.lock"), f"part {a.part}", kcf.Busy)
    try:
        rows = restore_rows(a.out, plan, a.part, ledger)
        n_part = sum(j["part"] == a.part for j in plan["jobs"])
        todo = [j for j in plan["jobs"] if j["part"] == a.part and
                f"{plan['sha256']}/{j['want']['eid']}" not in rows]
        if not todo:
            print(f"  part {a.part}: готова ({len(rows)})", flush=True)
            return
        if pr := threads_problems(torch):
            raise SystemExit(f"потоки: {pr}")
        idx = kcf.dev_index(a.device)
        gpu = kcf.gpu_query(idx)
        lock = kcf.Lock(str(Path(kcf.lock_dir()) / (gpu["uuid"] + ".lock")),
                        gpu["uuid"], kcf.Busy)
        sim = None
        try:
            kcf.check_gpu(idx, kcf.RES["min_free_gb"])
            kcf.check_disk(a.out)
            binding = kcf.bind_device(torch, idx, gpu)
            adm = dict(host=socket.gethostname(), uuid=gpu["uuid"],
                       pci=gpu["pci"], binding=binding,
                       threads=torch.get_num_threads())
            sim = kb36.BankSim("bank", kbk.SPACE, bank_dir=kbk.BANK_DIR)
            for seed in (0, 1):
                src = arm = None
                for j in todo:
                    w = j["want"]
                    if w["seed"] != seed:
                        continue
                    key = f"{plan['sha256']}/{w['eid']}"
                    name, path, fp, ident = scene_files(a.out, j, plan)
                    if src is None:
                        kcf.check_gpu(idx, kcf.RES["min_free_gb"])
                        src = kh.load_source(torch, seed, a.device)
                        arm = kh.make_arm(torch, k36, f"T32_s{seed}", src)
                    rec, z = kfp.reference(a.archive, j["arch_part"], w)
                    if rec["npz_sha1"] != j["npz_sha1"] or \
                            rec["phys"] != j["physical"] or \
                            rec["summary"]["success"] is not j["success"]:
                        raise SystemExit(f"{w['eid']}: архив изменился")
                    kcf.check_disk(a.out)
                    ledger.reserve(SCOPE, key, amount=CASE_RESERVE)
                    result, frames = branches(sim, src["io"], arm, j, rec, z)
                    row = dict(ident, **result, gpu=adm)
                    kfp.save_frames(fp, frames)
                    pc.save_json(str(path), row)
                    if checked_result(path, fp, ident) != row:
                        raise SystemExit(f"{path}: запись изменилась")
                    ledger.commit(key, case_spend(row))
                    rows[key] = row
                    ar = row["arms"]
                    print(f"  {name}: T1 {int(row['t1'])}, GripOnly "
                          f"{int(ar['GripOnly']['success'])}, RefreshOnce "
                          f"{int(ar['RefreshOnce']['success'])}, схват "
                          f"{row['grip']['shift']} {row['grip']['changed_steps']}"
                          f"; {len(rows)}/{n_part}", flush=True)
                if src is not None:
                    src = arm = None
                    kh.free(torch)
        finally:
            if sim is not None:
                kcf.close_sim(sim)
            lock.release()
        print(f"  part {a.part}: готова", flush=True)
    finally:
        plock.release()


# --- анализ ------------------------------------------------------------------------------
def counts(rows, arm):
    f = [r for r in rows if not r["t1"]]
    s = [r for r in rows if r["t1"]]
    return dict(n_fail=len(f), n_success=len(s),
                rescued=sum(r["arms"][arm]["success"] for r in f),
                lost=sum(not r["arms"][arm]["success"] for r in s))


def weighted(rows, arm, W, n_boot=BOOT["n"], seed=BOOT["seed"]):
    f = np.array([r["arms"][arm]["success"] for r in rows if not r["t1"]],
                 float)
    s = np.array([not r["arms"][arm]["success"] for r in rows if r["t1"]],
                 float)
    est = 100 * (W["p_fail"] * f.mean() - W["p_success"] * s.mean())
    rng = np.random.default_rng(seed)
    bf = f[rng.integers(0, len(f), (n_boot, len(f)))].mean(1)
    bs = s[rng.integers(0, len(s), (n_boot, len(s)))].mean(1)
    b = 100 * (W["p_fail"] * bf - W["p_success"] * bs)
    lo, hi = np.quantile(b, [0.025, 0.975])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def summarize(rows, plan):
    W = plan["weights"]
    out = dict(totals={a: counts(rows, a) for a in ARMS},
               by_seed={str(s): {a: counts([r for r in rows
                                            if r["seed"] == s], a)
                                 for a in ARMS} for s in (0, 1)},
               weighted={a: weighted(rows, a, W) for a in ARMS},
               weighted_by_seed={str(s): {a: weighted(
                   [r for r in rows if r["seed"] == s], a,
                   W["per_seed"][str(s)]) for a in ARMS} for s in (0, 1)},
               grip_shift=dict(Counter(r["grip"]["shift"] for r in rows)),
               grip_same=sum(r["grip_same"] for r in rows),
               arm_same=sum(r["arm_same"] for r in rows))
    pair = Counter()
    inter = []
    for r in rows:
        y = {a: int(r["arms"][a]["success"]) for a in ARMS}
        pair[f"{'fail' if not r['t1'] else 'success'}_G{y['GripOnly']}"
             f"A{y['ArmOnly']}R{y['RefreshOnce']}"] += 1
        inter.append(y["RefreshOnce"] - y["GripOnly"] - y["ArmOnly"] +
                     int(r["t1"]))
    out["pairs"] = dict(sorted(pair.items()))
    out["interaction"] = dict(sum=int(sum(inter)),
                              pos=sum(x > 0 for x in inter),
                              neg=sum(x < 0 for x in inter))
    est = {a: out["weighted"][a]["est"] for a in ARMS}
    out["signals"] = [a for a in ARMS if est[a] >= THRESHOLD_PP]
    out["decision"] = ("+".join(out["signals"]) + "_signal"
                       if out["signals"] else "no_gain")
    return out


def summarize_cmd(a, plan):
    import k36_protocol as pc
    rows = []
    for k in range(N_PARTS):
        p = part_dir(a.out, k) / "ledger.json"
        if not p.exists():
            raise SystemExit(f"part {k}: нет журнала")
        led = Ledger(str(p), plan["caps"])
        rr = restore_rows(a.out, plan, k, led)
        if len(rr) != sum(j["part"] == k for j in plan["jobs"]) or \
                led.pending_id() is not None:
            raise SystemExit(f"part {k}: неполная")
        rows += list(rr.values())
    res = summarize(rows, plan)
    pc.save_json(str(a.out / "summary.json"), dict(
        res, plan=plan["sha256"], rows=rows))
    for arm in ARMS:
        c, wv = res["totals"][arm], res["weighted"][arm]
        print(f"  {arm}: спасено {c['rescued']}/{c['n_fail']}, потеряно "
              f"{c['lost']}/{c['n_success']}; взвешенно {wv['est']:+.1f} "
              f"п.п. 95% [{wv['lo']:+.1f}; {wv['hi']:+.1f}]; по сидам "
              f"{ {s: res['by_seed'][s][arm] for s in ('0', '1')} }")
    print(f"  схват R к T1: {res['grip_shift']}; схват R = B на "
          f"{res['grip_same']} сценах")
    print(f"  исходы (T1 · G A R): {res['pairs']}")
    print(f"  взаимодействие R − G − A + T1 по сценам: {res['interaction']}")
    print(f"  K-39 2×2: {res['decision']}")


def selftest():
    import k31_collect as kcol
    # онлайн-триггер
    def mk(seq_exec, plans):
        A = np.zeros((len(seq_exec), 7))
        A[:, 6] = seq_exec
        J = dict(elapsed=np.arange(len(plans)) * 8,
                 a_codec=np.zeros((len(plans), 8, 7)))
        for i, p in enumerate(plans):
            J["a_codec"][i][:, 6] = -np.asarray(p)
        return A, J
    o, c = OPEN, CLOSE
    p0 = [o] * 8
    p1 = [o] * 5 + [c] * 3                 # закрытие на шаге 13
    A, J = mk([o] * 8 + [o] * 5 + [c] * 3 + [c] * 4, [p0, p1, [c] * 8])
    assert online_point(A, J) == (13, 8)
    p1b = [o] * 7 + [c]                    # закрытие на последнем шаге плана
    A, J = mk([o] * 15 + [c] + [c] * 4, [p0, p1b, [c] * 8])
    assert online_point(A, J) == (15, 8)
    p1c = [o] * 6 + [c, o]                 # не устойчиво в плане
    A, J = mk([o] * 16 + [o] * 4, [p0, p1c, p0])
    assert online_point(A, J) is None
    p1d = [o, o, c, c, c, c, c, c]         # закрытие в префиксе — не триггер
    A, J = mk([o] * 8 + [o, o, c, c, c, c, c, c] + [c] * 4,
              [p0, p1d, [c] * 8])
    assert online_point(A, J) is None
    A, J = mk([o] * 12, [p0, p1])          # эпизод кончился до хвоста
    assert online_point(A, J) is None
    assert kcol.DUMMY[6] == OPEN
    # схват
    B4 = np.zeros((4, 7), np.float32)
    B4[:, 6] = [1, 1, -1, -1]              # физ. [-1, -1, 1, 1]
    R4 = B4.copy()
    R4[:, :6] = 0.1
    R4[:, 6] = [1, 1, 1, -1]
    gt = grip_tail(B4, R4)
    assert np.array_equal(gt[:, :6], B4[:, :6]) and \
        np.array_equal(gt[:, 6], R4[:, 6])
    at = arm_tail(B4, R4)
    assert np.array_equal(at[:, :6], R4[:, :6]) and \
        np.array_equal(at[:, 6], B4[:, 6])
    assert grip_info(B4, R4)["shift"] == "later"
    assert grip_info(B4, B4)["shift"] == "same"
    R5 = B4.copy()
    R5[:, 6] = 1
    assert grip_info(B4, R5)["shift"] == "removed"
    # отбор: группы, разные case_id, исключение v1, части
    jobs = [dict(want=dict(seed=s, eid=f"e{s}_{i}", case_id=f"c{i}"),
                 success=(i + s) % 3 != 0, g=10, t=8)
            for s in (0, 1) for i in range(150)]
    sel = select(jobs, {"c1", "c2"})
    assert len(sel) == 48 and len({j["want"]["case_id"] for j in sel}) == 48
    assert not {"c1", "c2"} & {j["want"]["case_id"] for j in sel}
    for k in range(N_PARTS):
        pj = [j for j in sel if j["part"] == k]
        assert len(pj) == 24
    assert Counter((j["want"]["seed"], j["success"]) for j in sel) == {
        (0, False): 12, (0, True): 12, (1, False): 12, (1, True): 12}
    # анализ: взвешивание и решение
    W = dict(p_fail=0.17, p_success=0.47)
    rows = []
    for i in range(24):
        rows.append(dict(t1=False, seed=i % 2, grip=dict(shift="later"),
                         grip_same=False, arm_same=False, arms=dict(
                             GripOnly=dict(success=i < 6),
                             ArmOnly=dict(success=i < 1),
                             RefreshOnce=dict(success=i < 3))))
        rows.append(dict(t1=True, seed=i % 2, grip=dict(shift="later"),
                         grip_same=False, arm_same=False, arms=dict(
                             GripOnly=dict(success=True),
                             ArmOnly=dict(success=i >= 2),
                             RefreshOnce=dict(success=i >= 1))))
    plan = dict(weights=dict(W, per_seed={"0": W, "1": W}))
    sm = summarize(rows, plan)
    assert sm["totals"]["GripOnly"] == dict(n_fail=24, n_success=24,
                                            rescued=6, lost=0)
    assert abs(sm["weighted"]["GripOnly"]["est"] - 100 * 0.17 * 0.25) < 1e-9
    assert abs(sm["weighted"]["RefreshOnce"]["est"] -
               100 * (0.17 * 3 / 24 - 0.47 / 24)) < 1e-9
    assert sm["decision"] == "GripOnly_signal"
    assert sm["totals"]["ArmOnly"]["lost"] == 2
    # взаимодействие R − G − A + T1: провалы i=0 → −1, i=1,2 → 0,
    # i=3..5 → −1; успехи i=1 → +1, остальные 0
    assert sm["interaction"] == dict(sum=-3, pos=1, neg=4)
    print("самопроверка k39_grip_probe пройдена: онлайн-триггер (граница "
          "плана, неустойчивое и раннее закрытие, конец эпизода); GripOnly = "
          "рука B + схват R, ArmOnly = рука R + схват B; взаимодействие; сдвиг закрытия; отбор 4 × 12 без v1, части по "
          "24; взвешенный прирост и решение")
    return 0


def main():
    import sys
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path,
                    default=Path("reports/k37c/runs/c2"))
    ap.add_argument("--v1", type=Path,
                    default=Path("reports/k39/natural_probe/v1"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/k39/grip_probe/v1"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    import k36_protocol as pc
    import k37_confirm as kcf
    if a.out.resolve().is_relative_to(a.archive.resolve()):
        raise SystemExit("каталог результатов должен быть вне архива c2")
    a.out.mkdir(parents=True, exist_ok=True)
    plan, k36, _C = prepare(a.archive, a.v1)
    plan["sha256"] = kcf.canon_sha(plan)
    pp = a.out / "plan.json"
    lk = kcf.Lock(str(a.out / ".plan.lock"), str(a.out), kcf.Busy)
    try:
        if pp.exists() and pc.load_json(str(pp)) != plan:
            raise SystemExit("план изменился, укажите новый --out")
        pc.save_json(str(pp), plan)
    finally:
        lk.release()
    print(f"  {len(plan['jobs'])} сцен, SHA {plan['sha256'][:16]}",
          flush=True)
    if a.prepare_only:
        return 0
    if a.summarize:
        summarize_cmd(a, plan)
        return 0
    if a.part is None:
        raise SystemExit("нужен --part 0|1")
    collect(a, plan, k36)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
