#!/usr/bin/env python3
"""Screen one RefreshOnce intervention on natural T1 failures in c2.

Task 8, six failures and six matched successes per seed, distinct scenes.
The intervention point is o4 in the chunk containing the first sustained
gripper closure. This selection uses the archive and is for diagnosis only.
Replay T1, then compare a fresh T32 tail and a random sign control from the
same prefix. The control shares the refresh gripper and the absolute arm
correction per step and axis. Only the sign of selected whole axes changes.
"""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import socket

import numpy as np

from k31_collect import ENV, EXEC_MAX, Ledger


TASK = 8
PER_SEED = 6
PROTOCOL = "k39-natural-failure-v1"
ARMS = ("RefreshOnce", "Random")
SCOPE = "natural_probe"
CASE_RESERVE = 3 * EXEC_MAX
CODE = ("k39_failure_probe.py", "k39_cache.py", "k39_pulse.py",
        "k38_grip_operator.py", "k38_grip_run.py")


def rank(key):
    return hashlib.sha256(f"{PROTOCOL}/{key}".encode()).hexdigest()


def anchor(actions):
    from k39_pulse import grip_event, grip_t

    g = grip_event(actions)
    t = grip_t(g)
    if t is None or g >= t + 8:
        return None
    return g, t


def select(jobs, per_seed=PER_SEED):
    selected = []
    used = set()
    for seed in (0, 1):
        failures = sorted((j for j in jobs if j["want"]["seed"] == seed
                           and not j["success"]),
                          key=lambda j: rank(j["want"]["eid"]))
        failures = [j for j in failures if j["want"]["case_id"] not in used]
        if len(failures) < per_seed:
            raise SystemExit(f"s{seed}: мало подходящих неудач ({len(failures)})")
        failures = failures[:per_seed]
        used.update(j["want"]["case_id"] for j in failures)
        selected.extend(failures)
    for failure in selected[:]:
        seed = failure["want"]["seed"]
        pool = [j for j in jobs if j["want"]["seed"] == seed and j["success"]
                and j["want"]["case_id"] not in used]
        if not pool:
            raise SystemExit(f"s{seed}: не хватает успешных контрольных сцен")
        control = min(pool, key=lambda j: (abs(j["g"] - failure["g"]),
                                          rank(j["want"]["eid"])))
        used.add(control["want"]["case_id"])
        selected.append(control)
    return sorted(selected, key=lambda j: (j["want"]["seed"],
                                            rank(j["want"]["eid"])))


def reference(archive, part, want):
    import k36_harness as kh
    import k37_confirm as kcf

    kcf.install(kcf.part_dir(str(archive), part))
    found = kh.load_ep(want)
    if found is None:
        raise SystemExit(f"{want['eid']}: отсутствует эпизод c2")
    return found


def prepare(archive):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf

    plan, k36 = kcf.load_plan(str(archive))
    C = kh.consts(k36)
    merged = pc.load_json(archive / "merged.json")
    if merged.get("ok") is not True or merged.get("plan_sha256") != plan["sha256"]:
        raise SystemExit("merged.json не подтверждает архивный план")
    rows = [r for r in merged["rows"] if r["policy"] == "T1" and r["task"] == TASK]
    indexed = {r["eid"]: r for r in rows}
    if len(rows) != len(indexed):
        raise SystemExit("повтор T1 в merged.json")

    coverage, eligible = Counter(), []
    for part in range(kcf.N_PARTS):
        ledger = pc.load_json(Path(kcf.part_dir(str(archive), part)) / "ledger.json")
        if ledger["pending"] is not None or ledger["caps"] != kcf.CAP_PART:
            raise SystemExit(f"часть {part}: открытый резерв или неверные лимиты")
        for w in kcf.part_wants(plan, k36, part):
            if w["task"] != TASK or w["kind"] != "T":
                continue
            r = indexed.pop(w["eid"], None)
            if r is None or r["part"] != part or r["seed"] != w["seed"] or \
                    r["case_id"] != w["case_id"] or type(r["success"]) is not bool:
                raise SystemExit(f"{w['eid']}: запись merged.json ≠ плану")
            rec, z = reference(archive, part, w)
            success = rec["summary"]["success"]
            problems = kh.episode_problems(rec, z, w, C)
            done = ledger["done"].get(w["eid"])
            if problems or success is not r["success"] or done != {
                    "scope": "confirm", "actual": rec["phys"]}:
                raise SystemExit(f"{w['eid']}: архив не прошёл сверку ({problems})")
            group = f"s{w['seed']}_{'success' if success else 'fail'}"
            coverage[group] += 1
            point = anchor(z["actions"])
            if point is not None:
                g, t = point
                coverage[f"{group}_eligible"] += 1
                eligible.append(dict(want=w, part=part, success=success, g=g, t=t,
                                     physical=rec["phys"], npz_sha1=rec["npz_sha1"]))
    if indexed:
        raise SystemExit("в merged.json остались записи T1 вне плана")
    print(f"  t{TASK}, покрытие точки развилки: {dict(sorted(coverage.items()))}", flush=True)
    jobs = select(eligible)
    modules_dir = Path(__file__).resolve().parent
    return dict(protocol=PROTOCOL, archive_plan=plan["sha256"], coverage=dict(coverage),
                code={name: kcf.sha256(modules_dir / name) for name in CODE},
                caps={SCOPE: len(jobs) * CASE_RESERVE}, jobs=jobs), k36, C


def random_tail(base, fresh, bounds, key):
    base_arm = base[:, :6].astype(np.float64)
    delta = fresh[:, :6].astype(np.float64) - base_arm
    reflected = (base_arm - delta).astype(np.float32)
    lo, hi = (np.asarray(bounds[k], np.float32) for k in ("L", "U"))
    feasible = np.flatnonzero(((lo <= reflected) & (reflected <= hi)).all(0) &
                              (reflected != fresh[:, :6]).any(0))
    rng = np.random.default_rng(int(rank(key)[:16], 16))
    mask = int(rng.integers(1, 2 ** len(feasible))) if len(feasible) else 0
    axes = [int(axis) for i, axis in enumerate(feasible) if mask & (1 << i)]
    tail = fresh.copy()
    tail[:, axes] = reflected[:, axes]
    d_random = (tail[:, :6].astype(np.float64) - base_arm) / bounds["half"]
    d_fresh = delta / bounds["half"]
    if not np.allclose(np.abs(d_random), np.abs(d_fresh), rtol=0, atol=1e-6):
        raise SystemExit("контроль: абсолютная поправка не совпала")
    return tail, axes, float(np.sqrt(np.mean(d_fresh ** 2)))


def grip_commands(base, fresh):
    """Было/стало в физических знаках, шаги чанка с единицы."""
    if not np.isin([base[:, 6], fresh[:, 6]], (-1, 1)).all():
        raise SystemExit("схват: ожидались команды ±1")
    return dict(t1=(-base[:, 6]).astype(int).tolist(),
                refresh=(-fresh[:, 6]).astype(int).tolist(),
                changed_steps=(5 + np.flatnonzero(base[:, 6] != fresh[:, 6])).tolist())


def replay(sim, io, arm, job, rec, z):
    from k31_collect import obs_hash
    from k36_harness import episode_number
    from k39_cache import CAMERAS, Replay, teacher

    w, t, g = job["want"], job["t"], job["g"]
    task, init, eid = w["task"], w["reset_id"], w["eid"]
    rp = Replay(sim, io, task, init)
    q = rec["summary"]
    hashes = q["init_hash"], q["init_hash_full"]
    if (rp.ih[0][0], rp.ih[1][0]) != hashes:
        raise SystemExit(f"{eid}: начальные хеши ≠ архиву")
    A = z["actions"]
    indices = np.flatnonzero(z["j_elapsed"] == t)
    if len(indices) != 1:
        raise SystemExit(f"{eid}: нет единственной архивной команды в t={t}")
    (i,) = indices
    B = z["j_a_codec"][i]
    frame_steps = {g, min(g + 3, len(A)), min(g + 12, len(A)), min(g + 24, len(A))}
    frames = {}
    for step in range(len(A) + 1):
        if step in frame_steps:
            frames[step] = [np.array(rp.obs[k][0, :, ::-1], copy=True) for k in CAMERAS]
        if step == t:
            o0 = obs_hash(rp.obs)
            desc = io.task_desc(task)
            proposal, _ = teacher(arm.torch, arm, io, desc, task, init, episode_number(eid),
                                  t // 8, rp.obs, t)
            if not np.array_equal(proposal, B):
                raise SystemExit(f"{eid}: прогноз T32 в o0 ≠ архивному")
        if step == t + 4:
            o4 = obs_hash(rp.obs)
            R, _ = teacher(arm.torch, arm, io, desc, task, init, episode_number(eid),
                           t // 8 + 1, rp.obs, step)
        if step < len(A):
            rp.step(A[step])
    if (rp.reward >= 1, rp.succ_step, rp.done_step, rp.steps) != (
            q["success"], q["success_step"], q["done_step"], q["steps"]):
        raise SystemExit(f"{eid}: повтор T1 ≠ архивному исходу или terminal")
    return B, R[:4], o0, o4, hashes, frames


class CheckedMidpoint:
    def __init__(self, sim, steps, observation_hash):
        self.sim = sim
        self.steps = steps
        self.observation_hash = observation_hash

    def reset(self, task, starts):
        self.count = 0
        return self.sim.reset(task, starts)

    def step(self, actions):
        from k31_collect import obs_hash

        result = self.sim.step(actions)
        self.count += 1
        if self.count == self.steps and obs_hash(result[0]) != self.observation_hash:
            raise SystemExit("наблюдение перед продолжением ≠ повтору T1")
        return result


def branches(sim, io, arm, job, rec, z, bounds):
    from k31_collect import run_episode
    from k36_harness import SUMMARY, episode_number

    w, t = job["want"], job["t"]
    eid = w["eid"]
    B, fresh, o0, o4, hashes, frames = replay(sim, io, arm, job, rec, z)
    grip = grip_commands(B[4:], fresh)
    tail, axes, correction = random_tail(B[4:], fresh, bounds, eid)
    choices = [("RefreshOnce", fresh), ("Random", tail)]
    if int(rank(eid)[-1], 16) % 2:
        choices.reverse()
    results = {}
    A = z["actions"]
    for name, suffix in choices:
        wrapped = CheckedMidpoint(sim, ENV["waiting_steps"] + t + 4, o4)
        arm.episode = episode_number(eid)
        chunk = np.concatenate((B[:4], suffix))
        ep = run_episode(wrapped, io, arm, w["task"], w["reset_id"], prefix=A[:t],
                         fork=dict(tau=t, hash=o0, a_codec=chunk))
        actual_init = ep["init_hash"], ep["init_hash_full"]
        if (wrapped.count < wrapped.steps or actual_init != hashes or
                not np.array_equal(ep["actions"][:t + 4], A[:t + 4])):
            raise SystemExit(f"{eid} {name}: общий префикс ≠ архиву")
        if np.array_equal(suffix, B[4:]):
            keys = ("success", "success_step", "done_step", "steps", "actions_sha1")
            if any(ep[k] != rec["summary"][k] for k in keys):
                raise SystemExit(f"{eid} {name}: нулевая поправка ≠ T1")
        results[name] = {k: ep[k] for k in (*SUMMARY, "seconds")}
    if not axes and any(results["RefreshOnce"][k] != results["Random"][k]
                        for k in ("success", "physical", "actions_sha1")):
        raise SystemExit(f"{eid}: одинаковые хвосты дали разные эпизоды")
    return dict(arms=results, o0=o0, o4=o4, axes=axes, grip=grip,
                correction_rms=correction), frames


def save_frames(path, frames):
    from PIL import Image, ImageDraw

    w, h = 192, 192
    ordered = sorted(frames.items())
    canvas = Image.new("RGB", (w * len(ordered), 2 * h + 18), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (step, cameras) in enumerate(ordered):
        draw.text((w * i + 4, 2), f"step {step}", fill="black")
        for camera, pixels in enumerate(cameras):
            canvas.paste(Image.fromarray(pixels).resize((w, h)),
                         (w * i, 18 + h * camera))
    tmp = path.with_suffix(".tmp")
    canvas.save(tmp, format="PNG")
    tmp.replace(path)


def outcome_counts(rows):
    return {name: dict(
        rescued=sum(r["arms"][name]["success"] for r in rows if not r["t1"]),
        lost=sum(not r["arms"][name]["success"] for r in rows if r["t1"]))
        for name in ARMS}


def case_spend(row):
    return row["physical"] + sum(ep["physical"] for ep in row["arms"].values())


def summarize(rows):
    by_seed = {str(s): outcome_counts([r for r in rows if r["seed"] == s])
               for s in (0, 1)}
    grip_groups = {}
    for name, changed in [("changed", True), ("unchanged", False)]:
        group = [r for r in rows if bool(r["grip"]["changed_steps"]) == changed]
        grip_groups[name] = dict(cases=len(group), t1_failures=sum(not r["t1"] for r in group),
                                 t1_successes=sum(r["t1"] for r in group),
                                 outcomes=outcome_counts(group))
    totals = outcome_counts(rows)
    fresh, random = totals["RefreshOnce"], totals["Random"]
    positive = (fresh["rescued"] >= 3 and fresh["rescued"] - random["rescued"] >= 2
                and fresh["lost"] == 0)
    return dict(by_seed=by_seed, by_grip=grip_groups, totals=totals,
                physical=sum(case_spend(r) for r in rows),
                identical_controls=sum(not r["axes"] for r in rows),
                decision="recovery_signal" if positive else "hold_training")


def scene_files(out, job, plan):
    w = job["want"]
    name = f"s{w['seed']}_{w['case_id']}"
    identity = dict(eid=w["eid"], plan=plan["sha256"], seed=w["seed"], t1=job["success"],
                    t=job["t"], g=job["g"], archive_npz_sha1=job["npz_sha1"],
                    physical=job["physical"])
    return name, out / f"{name}.json", out / f"{name}.png", identity


def checked_result(path, frame_path, identity):
    import k36_harness as kh
    import k36_protocol as pc

    row = pc.load_json(path)
    if (not isinstance(row, dict) or
            any(type(row.get(k)) is not type(v) or row[k] != v for k, v in identity.items()) or
            not frame_path.is_file()):
        raise SystemExit(f"{path}: запись не соответствует плану")
    if any(not isinstance(row.get(k), str) or not row[k] for k in ("o0", "o4")):
        raise SystemExit(f"{path}: нет хешей наблюдений")
    grip = row.get("grip")
    if not isinstance(grip, dict):
        raise SystemExit(f"{path}: нет данных схвата")
    g0, g1 = grip.get("t1", []), grip.get("refresh", [])
    if (not isinstance(g0, list) or not isinstance(g1, list) or
            len(g0) != 4 or len(g1) != 4 or
            any(type(x) is not int or x not in (-1, 1) for x in (*g0, *g1)) or
            grip.get("changed_steps") != [i for i, (a, b) in enumerate(zip(g0, g1), 5) if a != b]):
        raise SystemExit(f"{path}: неверные данные схвата")
    arms = row.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise SystemExit(f"{path}: неполный набор продолжений")
    for name, ep in arms.items():
        if not isinstance(ep, dict):
            raise SystemExit(f"{path} {name}: сводка не словарь")
        problems = kh.summary_problems(dict(phys=ep.get("physical")), ep, ep.get("steps"))
        if problems or ep["steps"] < row["t"] + 4:
            raise SystemExit(f"{path} {name}: неверная сводка ({problems})")
    return row


def restore_rows(out, plan, ledger):
    """Проверить все готовые сцены и закрыть записанный резерв до нового."""
    jobs = {f"{plan['sha256']}/{j['want']['eid']}": j for j in plan["jobs"]}
    done, pending, crashes = (ledger.d[k] for k in ("done", "pending", "crashes"))
    pending_id = ledger.pending_id()
    if not set(done).issubset(jobs):
        raise SystemExit("ledger.json: учтённые сцены вне плана")
    if pending is not None and (pending_id not in jobs or pending_id in done or
                                pending != dict(id=pending_id, scope=SCOPE, amount=CASE_RESERVE)):
        raise SystemExit("ledger.json: открытый резерв не соответствует плану")
    for crash in crashes:
        if (crash.get("id") not in jobs or crash.get("scope") != SCOPE or
                crash.get("amount") != CASE_RESERVE):
            raise SystemExit("ledger.json: авария не соответствует плану")

    rows = {}
    for key, j in jobs.items():
        _name, path, frame_path, identity = scene_files(out, j, plan)
        if path.exists():
            row = checked_result(path, frame_path, identity)
            if key != pending_id and done.get(key) != dict(scope=SCOPE, actual=case_spend(row)):
                raise SystemExit(f"{path}: расход не соответствует закрытой записи журнала")
            rows[key] = row
        elif key in done:
            raise SystemExit(f"{path}: сцена учтена, результата нет")
    spent = sum(case_spend(rows[k]) for k in done) + CASE_RESERVE * len(crashes)
    reserved = CASE_RESERVE if pending is not None else 0
    if ledger.d["spent"] != {SCOPE: spent} or spent + reserved > plan["caps"][SCOPE]:
        raise SystemExit("ledger.json: расход не сходится")
    if pending_id in rows:
        ledger.settle(pending_id, case_spend(rows[pending_id]))
    return rows


def run_seed(seed, sim, torch, a, plan, k36, C, admission, ledger, rows):
    import k36_harness as kh
    import k36_protocol as pc
    import k37_confirm as kcf

    source = arm = None
    for j in plan["jobs"]:
        w = j["want"]
        key = f"{plan['sha256']}/{w['eid']}"
        if w["seed"] != seed or key in rows:
            continue
        name, path, frame_path, identity = scene_files(a.out, j, plan)
        if source is None:
            source = kh.load_source(torch, seed, a.device)
            arm = kh.make_arm(torch, k36, f"T32_s{seed}", source)
        rec, z = reference(a.archive, j["part"], w)
        if (rec["npz_sha1"] != j["npz_sha1"] or rec["phys"] != j["physical"] or
                rec["summary"]["success"] is not j["success"]):
            raise SystemExit(f"{w['eid']}: архив изменился после подготовки")
        kcf.check_disk(a.out)
        ledger.reserve(SCOPE, key, amount=CASE_RESERVE)
        result, frames = branches(sim, source["io"], arm, j, rec, z, C)
        row = dict(identity, **result, gpu=admission)
        save_frames(frame_path, frames)
        pc.save_json(str(path), row)
        loaded = checked_result(path, frame_path, identity)
        if loaded != row:
            raise SystemExit(f"{path}: записанный результат изменился до commit")
        ledger.commit(key, case_spend(loaded))
        rows[key] = loaded
        print(f"  {name}: T1 {int(row['t1'])}, RefreshOnce "
              f"{int(row['arms']['RefreshOnce']['success'])}, Random "
              f"{int(row['arms']['Random']['success'])}, g={j['g']}, "
              f"схват на шагах {row['grip']['changed_steps']}", flush=True)


def collect(a, plan, k36, C):
    ledger = Ledger(str(a.out / "ledger.json"), plan["caps"])
    rows = restore_rows(a.out, plan, ledger)
    if len(rows) == len(plan["jobs"]):
        return list(rows.values()), ledger.d

    import torch
    import k36_bank as kb36
    import k36_harness as kh
    import k37_bank as kbk
    import k37_confirm as kcf
    from k38_grip_run import threads_problems

    if problems := threads_problems(torch):
        raise SystemExit(f"потоки: {problems}")
    idx = kcf.dev_index(a.device)
    gpu = kcf.gpu_query(idx)
    lock = kcf.Lock(str(Path(kcf.lock_dir()) / (gpu["uuid"] + ".lock")),
                    gpu["uuid"], kcf.Busy)
    sim = None
    try:
        kcf.check_gpu(idx, kcf.RES["min_free_gb"])
        kcf.check_disk(a.out)
        binding = kcf.bind_device(torch, idx, gpu)
        admission = dict(host=socket.gethostname(), uuid=gpu["uuid"], pci=gpu["pci"],
                         binding=binding, threads=torch.get_num_threads())
        sim = kb36.BankSim("bank", kbk.SPACE, bank_dir=kbk.BANK_DIR)
        for seed in (0, 1):
            run_seed(seed, sim, torch, a, plan, k36, C, admission, ledger, rows)
            kh.free(torch)
        rows = restore_rows(a.out, plan, ledger)
        if len(rows) != len(plan["jobs"]) or ledger.pending_id() is not None:
            raise SystemExit("неполный сбор или открытый резерв")
        return list(rows.values()), ledger.d
    finally:
        if sim is not None:
            kcf.close_sim(sim)
        lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("reports/k37c/runs/c2"))
    parser.add_argument("--out", type=Path, default=Path("reports/k39/natural_probe/v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    a = parser.parse_args()

    import k36_protocol as pc
    import k37_confirm as kcf

    if a.out.resolve().is_relative_to(a.archive.resolve()):
        raise SystemExit("каталог результатов должен быть вне архива c2")
    a.out.mkdir(parents=True, exist_ok=True)
    lock = kcf.Lock(str(a.out / ".run.lock"), str(a.out), kcf.Busy)
    try:
        plan, k36, C = prepare(a.archive)
        plan["sha256"] = kcf.canon_sha(plan)
        plan_path = str(a.out / "plan.json")
        if Path(plan_path).exists() and pc.load_json(plan_path) != plan:
            raise SystemExit("план изменился, укажите новый --out")
        pc.save_json(plan_path, plan)
        print(f"  {len(plan['jobs'])} сцен, SHA {plan['sha256'][:16]}", flush=True)
        if not a.prepare_only:
            rows, ledger = collect(a, plan, k36, C)
            result = summarize(rows)
            pc.save_json(str(a.out / "summary.json"), dict(result, rows=rows, plan=plan["sha256"],
                         ledger={k: ledger[k] for k in ("caps", "spent", "crashes")}))
            print(f"  изм. схвата RefreshOnce к T1: "
                  f"{result['by_grip']['changed']['cases']}/{len(rows)} сцен", flush=True)
            print(f"  {result['totals']}; {result['decision']}", flush=True)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
