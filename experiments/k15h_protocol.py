#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15h: инвентаризация стартов, план, проверки входов и блоков.

РЕЖИМЫ:
  inventory   какие (задача, init_state_id) LIBERO-10 уже участвовали в
              экспериментах: JSON с эпизодами, планы с блоками, логи
              харнессов («задача N … блок a-b»), включая многоблочные
              прогоны. Размер канонического банка и отпечатки файлов
              начальных состояний — из LIBERO. Выбор свежего банка — ДО
              любых раскаток q0 на нём, без отбора по исходам: по
              возрастанию ID первые три полностью неиспользованных блока
              по 5 на каждой задаче (запасной вариант, заданный заранее, —
              два блока на всех задачах). Каталоги, где история не
              восстановлена, перечисляются отдельно.
  smoke-plan  план smoke: старые блоки (задачи 0 и 8, блок 0), пять
              политик, смешанные направления внутри батча, эталоны: q0
              переписи M1-cold и p4r3p из K-15g.
  plan        окончательный план (нужны инвентаризация и чистый smoke).
  check-block / check-run / smoke-check — проверки (см. check_block).

ПЛАН фиксирует: входы (базис, отчёт тождества, инвентаризацию), код,
банк, таблицу направлений (choice_seed=20261013, отдельный генератор;
при возобновлении читается из плана), контракт раскатки и базиса, правила
анализа.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

KIND = "k15h_plan"
CHOICE_SEED = 20261013
N_ENVS = 5
BLOCKS_WANT, BLOCKS_FALLBACK = 3, 2
TASKS = list(range(10))
SUITE = "10"
INVENTORY = "reports/k15h/inventory.json"
PLAN = "reports/k15h/plan.json"
SMOKE_PLAN = "reports/k15h/smoke_plan.json"
SMOKE_BANK = {0: [0], 8: [0]}
SMOKE_DIRS = {0: ["0p", "1m", "2p", "3m", "0m"],
              8: ["1p", "2m", "3p", "0m", "3m"]}
CODE = ("k15h_policy.py", "k15h_harness.py", "k15h_protocol.py",
        "k15h_static_basis.py", "k15h_analyze.py", "k9h_multiarm_gate.py",
        "k15f_policy.py", "k15f_continuous_refine.py",
        "k15f_check_identity.py", "k15g_local_probe.py")
ANALYSIS = dict(
    unit="начальное состояние", main="среднее SR по 10 задачам с равными "
    "весами", primary=[["p4r3p", "q0ref"],
                       ["learned_uniform", "arm_dir_uniform"],
                       ["learned_uniform", "static_uniform"]],
    test="точный двусторонний Мак-Немар, поправка Holm на 3 сравнения",
    ci="95% парный бутстрэп ΔSR внутри задач, вектор исходов состояния "
       "целиком", bootstrap_n=10000, bootstrap_seed=20261013,
    secondary=["остальные политики против q0ref", "arm_dir − static",
               "только достигнутые импульсы"])
H = 8
MOMENT = 4


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def reached(done_step, moment):
    return bool(done_step < 0 or done_step >= H * (moment - 1))


# --- инвентаризация ------------------------------------------------------------
LOG_TASK = re.compile(r"suite (\S+), задача (\d+)")
LOG_BLOCK = re.compile(r"--- блок (\d+)-(\d+)")


def states_from_json(obj, add, suite_default=SUITE):
    """Рекурсивно: эпизоды с init_state_id, заголовки блоков, планы."""
    if isinstance(obj, list):
        for x in obj:
            states_from_json(x, add, suite_default)
        return
    if not isinstance(obj, dict):
        return
    suite = str(obj.get("suite", obj.get("task_suite", suite_default)))
    t = obj.get("task_id")
    if t is not None and isinstance(obj.get("episodes"), list):
        for e in obj["episodes"]:
            if isinstance(e, dict) and "init_state_id" in e:
                add(suite, int(t), int(e["init_state_id"]))
    if t is not None and "init_start" in obj and "n_envs" in obj:
        try:
            for s in range(int(obj["init_start"]),
                           int(obj["init_start"]) + int(obj["n_envs"])):
                add(suite, int(t), s)
        except (TypeError, ValueError):
            pass
    bl = obj.get("blocks")
    if isinstance(bl, dict):
        for tk, bs in bl.items():
            if str(tk).isdigit() and isinstance(bs, list):
                for b in bs:
                    if isinstance(b, int):
                        for s in range(b, b + N_ENVS):
                            add(suite, int(tk), s)
    for v in obj.values():
        if isinstance(v, (dict, list)):
            states_from_json(v, add, suite)


def states_from_log(text, add):
    suite = task = None
    for line in text.splitlines():
        m = LOG_TASK.search(line)
        if m:
            suite, task = m.group(1), int(m.group(2))
            continue
        m = LOG_BLOCK.search(line)
        if m and task is not None:
            for s in range(int(m.group(1)), int(m.group(2)) + 1):
                add(suite, task, s)


def scan(roots, exclude=("reports/k15h",)):
    """{(задача, состояние): число источников}, сводка по файлам,
    каталоги без восстановленной истории."""
    used, per_file, other_suite = {}, {}, set()
    dirs_seen, dirs_found = {}, set()
    for root in roots:
        for dp, _dn, fn in os.walk(root):
            if any(os.path.normpath(dp).startswith(os.path.normpath(x))
                   for x in exclude):
                continue
            for f in fn:
                p = os.path.join(dp, f)
                # каталог эксперимента: первый уровень под корнем
                top = os.path.join(root, os.path.relpath(p, root).split(
                    os.sep)[0])
                dirs_seen.setdefault(top, 0)
                dirs_seen[top] += 1
                got = set()

                def add(suite, t, s):
                    if str(suite) != SUITE:
                        other_suite.add((str(suite), t, s))
                        return
                    got.add((t, s))
                try:
                    if f.endswith(".json"):
                        states_from_json(json.load(open(p)), add)
                    elif f.endswith(".log"):
                        states_from_log(open(p, errors="replace").read(),
                                        add)
                except (ValueError, OSError, UnicodeDecodeError):
                    continue
                if got:
                    per_file[p] = len(got)
                    dirs_found.add(top)
                    for k in got:
                        used[k] = used.get(k, 0) + 1
    unrecovered = sorted(d for d in dirs_seen if d not in dirs_found)
    return used, per_file, unrecovered, sorted(other_suite)


def libero_bank():
    """Размер канонического банка и отпечатки файлов начальных состояний."""
    import torch
    from libero.libero import benchmark, get_libero_path
    bench = benchmark.get_benchmark_dict()[f"libero_{SUITE}"]()
    out = {}
    for t in range(bench.n_tasks):
        task = bench.get_task(t)
        path = os.path.join(get_libero_path("init_states"),
                            task.problem_folder, task.init_states_file)
        x = torch.load(path, weights_only=False)
        out[str(t)] = dict(n_states=int(len(x)), path=path,
                           sha1=hashlib.sha1(open(path, "rb").read())
                           .hexdigest(), task=task.language)
    return out


def select_bank(used, n_states, tasks=TASKS):
    """По возрастанию ID первые свободные блоки по 5. Возвращает
    (банк {задача: [блоки]}, число блоков на задачу, свободные блоки)."""
    free = {}
    for t in tasks:
        n = int(n_states[t])
        free[t] = [b for b in range(0, n - N_ENVS + 1, N_ENVS)
                   if not any((t, s) in used for s in range(b, b + N_ENVS))]
    for k in (BLOCKS_WANT, BLOCKS_FALLBACK):
        if all(len(free[t]) >= k for t in tasks):
            return {t: free[t][:k] for t in tasks}, k, free
    raise SystemExit("свободных блоков недостаточно: " + ", ".join(
        f"{t}: {len(v)}" for t, v in free.items()))


def directions_table(bank, seed=CHOICE_SEED):
    """Отдельный локальный генератор; порядок — задачи и состояния по
    возрастанию; каждая из 8 пар равновероятна."""
    import k15h_policy as hp
    rng = np.random.default_rng(seed)
    out = {}
    for t in sorted(bank):
        out[str(t)] = {}
        for b in sorted(bank[t]):
            for s in range(b, b + N_ENVS):
                out[str(t)][str(s)] = hp.DIRS[int(rng.integers(8))]
    return out


def inventory_mode(out=INVENTORY):
    if os.path.exists(out):
        raise SystemExit(f"{out} уже есть: инвентаризация не перезаписывается")
    if glob.glob("reports/k15h/run/*/*.json"):
        raise SystemExit("раскатки K-15h уже есть: банк выбирается ДО них")
    roots = [r for r in ("reports", "logs") if os.path.isdir(r)]
    used, per_file, unrec, other = scan(roots)
    bank_info = libero_bank()
    n_states = {int(t): v["n_states"] for t, v in bank_info.items()}
    bank, k, free = select_bank(set(used), n_states)
    res = dict(
        kind="k15h_inventory", created=datetime.datetime.now().isoformat(
            timespec="seconds"),
        roots=roots, suite=SUITE, libero_bank=bank_info,
        used={str(t): sorted(s for (t_, s) in used if t_ == t)
              for t in TASKS},
        used_sources_files=len(per_file),
        files=dict(sorted(per_file.items())),
        other_suites=len(other),
        unrecovered_dirs=unrec,
        free_blocks={str(t): v for t, v in free.items()},
        rule=(f"по возрастанию ID первые {BLOCKS_WANT} полностью "
              f"неиспользованных блока по {N_ENVS}; запасной — "
              f"{BLOCKS_FALLBACK} на всех задачах"),
        blocks_per_task=k,
        bank={str(t): sorted(v) for t, v in bank.items()},
        limitations=(
            "история восстановлена по JSON и логам в reports/ и logs/ на "
            "кластере; прогоны, не оставившие ни JSON с эпизодами, ни лога "
            "харнесса, не учтены; каталоги без восстановленных состояний — "
            "unrecovered_dirs"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(out + ".tmp", out)
    print(f"  источников с состояниями LIBERO-{SUITE}: {len(per_file)}; "
          f"каталогов без истории: {len(unrec)}")
    for t in TASKS:
        u = res["used"][str(t)]
        print(f"    задача {t}: банк {n_states[t]}, использовано {len(u)} "
              f"({_ranges(u)}); свободные блоки {free[t]}; выбрано "
              f"{bank[t]}")
    print(f"  блоков на задачу: {k}; сводка: {out}")
    return 0


def _ranges(xs):
    xs = sorted(xs)
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{xs[i]}" if i == j else f"{xs[i]}–{xs[j]}")
        i = j + 1
    return ", ".join(out) or "—"


# --- план --------------------------------------------------------------------
def expected_contract(basis, identity):
    import torch
    ck = torch.load(basis, map_location="cpu", weights_only=False)
    rep = json.load(open(identity))
    return dict(
        basis_sha1=sha(basis), basis_state_sha1=ck["state_sha1"],
        stats_sha1=ck["stats_sha1"], amp_factor=float(ck["amp_factor"]),
        control_seed=int(ck["control_seed"]),
        control_kind=ck.get("control_kind"),
        identity_report_sha1=sha(identity),
        refine_module_sha1=sha(os.path.join(HERE,
                                            "k15f_continuous_refine.py")),
        k15f_policy_sha1=sha(os.path.join(HERE, "k15f_policy.py")),
        k15h_policy_sha1=sha(os.path.join(HERE, "k15h_policy.py")),
        harness_adapter_sha1=sha(os.path.join(HERE, "k15h_harness.py")),
        frozen_content_sha=rep.get("frozen_sha1"),
        joint_sha1=rep.get("joint_sha1"), code_version=rep.get("code_version"))


def make_plan(*, basis, identity, device, smoke, inventory=INVENTORY,
              snapshot="reports/k15f/m1cold_snapshot.json",
              k15g_run=None):
    import k15g_local_probe as pr
    import k15h_policy as hp
    inputs = dict(basis=basis, identity_report=identity)
    if smoke:
        bank = {t: list(v) for t, v in SMOKE_BANK.items()}
        dirs = {str(t): {str(b + i): SMOKE_DIRS[t][i] for b in bl
                         for i in range(N_ENVS)} for t, bl in bank.items()}
        import k15f_m1cold_snapshot as msn
        snap = msn.verify(snapshot)
        refs = dict(q0={str(t): os.path.join(snap["cold_dir"],
                                             f"q0_t{t}_i{b}.json")
                        for t, bl in bank.items() for b in bl})
        if not k15g_run:
            cand = sorted(glob.glob("reports/k15g/local/run/p*/"))
            if len(cand) != 1:
                raise SystemExit(f"каталог K-15g неоднозначен: {cand}")
            k15g_run = cand[0]
        refs["p4r3p"] = {"8": os.path.join(k15g_run, "p4r3p_t8_i0.json")}
        for grp in refs.values():
            for p in grp.values():
                if not os.path.exists(p):
                    raise SystemExit(f"нет эталона {p}")
        inputs["snapshot"] = snapshot
        ref_sha = {k: {t: sha(p) for t, p in v.items()}
                   for k, v in refs.items()}
    else:
        if not os.path.exists(inventory):
            raise SystemExit("нет инвентаризации (inventory)")
        inv = json.load(open(inventory))
        bank = {int(t): v for t, v in inv["bank"].items()}
        dirs = directions_table(bank)
        inputs["inventory"] = inventory
        refs, ref_sha = None, None
    return dict(
        kind=KIND, smoke=bool(smoke), created=datetime.datetime.now()
        .isoformat(timespec="seconds"),
        labels=list(hp.LABELS), moment=MOMENT, n_envs=N_ENVS,
        choice_seed=None if smoke else CHOICE_SEED,
        tasks=sorted(bank), blocks={str(t): sorted(v) for t, v in
                                    bank.items()},
        directions=dirs, device=str(device),
        setup=pr.expected_setup(), expected_contract=expected_contract(
            basis, identity),
        inputs={k: os.path.abspath(v) for k, v in inputs.items()},
        inputs_sha1={k: sha(v) for k, v in inputs.items()},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        refs=refs, refs_sha1=ref_sha, analysis=ANALYSIS)


def check_plan_inputs(plan, plan_path=None):
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-15h"]
    for k, f in plan["inputs"].items():
        if not os.path.exists(f) or sha(f) != plan["inputs_sha1"][k]:
            p.append(f"вход {k} изменён или отсутствует ({f})")
    for f, s_ in plan["code"].items():
        if sha(os.path.join(HERE, f)) != s_:
            p.append(f"код {f} изменён после плана")
    for grp, d in (plan.get("refs") or {}).items():
        for t, f in d.items():
            if not os.path.exists(f) or sha(f) != plan["refs_sha1"][grp][t]:
                p.append(f"эталон {grp} t{t} изменён ({f})")
    # таблица направлений основного плана воспроизводится по сиду
    if not plan.get("smoke"):
        bank = {int(t): v for t, v in plan["blocks"].items()}
        if directions_table(bank, plan["choice_seed"]) != plan["directions"]:
            p.append("таблица направлений не воспроизводится по сиду")
    return p


# --- проверки блока ----------------------------------------------------------
class Basis:
    """Статистики чекпойнта для проверки формул на CPU."""

    def __init__(self, path):
        import torch
        import k15f_continuous_refine as kf
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.U = ck["stats"]["U"].numpy().astype(np.float64)
        self.rho = ck["stats"]["rho"].numpy().astype(np.float64)
        self.sigma = ck["stats"]["sigma_arm"].numpy().astype(np.float64)
        self.sigma_g = float(ck["stats"]["sigma_g"])
        self.f = float(ck["amp_factor"])
        self.R = kf.rotation(int(ck["control_seed"])).numpy().astype(
            np.float64)


def combine_exec(a0, t, sigma, sigma_g, eps=1e-3):
    """Исполняемая часть combine K-15f на numpy (float64)."""
    a0 = np.asarray(a0, np.float64)
    arm = a0[..., :6] + sigma * t[..., :6]
    c = 1.0 - eps
    lg0 = 2.0 * np.arctanh(np.clip(a0[..., 6], -c, c))
    lg = lg0 + sigma_g * t[..., 6]
    g = a0[..., 6] + (np.tanh(lg * 0.5) - np.tanh(lg0 * 0.5))
    return np.concatenate([arm, g[..., None]], axis=-1)


def check_arrays(label, z, dirs, bs, tol=1e-4):
    """Поведение политики по журналу блока. Список нарушений."""
    import k15h_policy as hp
    import k15h_static_basis as sb
    p = []
    need = ("k15d_a0", f"k15d_level_{label}", "k15f_basis",
            "k15h_basis_used", "k15g_chunk", "k15g_active",
            "k15h_coeff_planned", "k15h_coeff_applied", "k15h_h18")
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{label}: в журнале нет {miss}"]
    chunk = np.asarray(z["k15g_chunk"])
    active = np.asarray(z["k15g_active"], bool)
    C = len(chunk)
    if chunk.tolist() != list(range(1, C + 1)):
        p.append(f"{label}: нумерация чанков нарушена")
    want = (chunk == MOMENT) if label != "q0ref" else np.zeros(C, bool)
    if active.tolist() != want.tolist():
        p.append(f"{label}: импульс не ровно в чанке {MOMENT}")
    planned = hp.planned_coeffs(label, dirs)
    if not np.array_equal(z["k15h_coeff_planned"], planned):
        p.append(f"{label}: запланированные коэффициенты не из плана")
    ap = z["k15h_coeff_applied"]
    if ap[~active].any() or (active.any() and not np.array_equal(
            ap[active], np.broadcast_to(planned, ap[active].shape))):
        p.append(f"{label}: применённые коэффициенты не равны плану в "
                 f"импульсе и нулю вне его")
    a0, lv = z["k15d_a0"], z[f"k15d_level_{label}"]
    if (~active).any() and not np.array_equal(a0[~active], lv[~active]):
        p.append(f"{label}: вне импульса действие не равно a0")
    L, used = z["k15f_basis"].astype(np.float64), z["k15h_basis_used"]
    if label in ("q0ref", "learned_uniform"):
        ok = np.array_equal(z["k15h_basis_used"], z["k15f_basis"])
    elif label == "p4r3p":
        ok = np.allclose(used, L @ bs.R, rtol=1e-4, atol=tol)
    elif label == "static_uniform":
        ok = np.allclose(used, sb.static_basis(bs.U, bs.rho, bs.f)[None,
                                                                  None],
                         rtol=1e-5, atol=tol)
    else:
        ok = np.allclose(used, sb.arm_dir_basis(
            L.reshape(-1, 4, 56), bs.U, bs.sigma).reshape(L.shape),
            rtol=1e-4, atol=tol)
    if not ok:
        p.append(f"{label}: базис не соответствует формуле варианта")
    if active.any():
        k = int(np.nonzero(active)[0][0])
        t = np.einsum("bk,bkd->bd", ap[k].astype(np.float64),
                      used[k].astype(np.float64)).reshape(-1, H, 7)
        exp = combine_exec(a0[k], t, bs.sigma, bs.sigma_g)
        if not np.allclose(lv[k], exp, rtol=1e-4, atol=tol):
            p.append(f"{label}: действие в импульсе не равно combine("
                     f"a0, базис·c) (max|Δ| "
                     f"{float(np.abs(lv[k] - exp).max()):.2e})")
        if label != "q0ref" and np.array_equal(lv[k], a0[k]):
            p.append(f"{label}: импульс не изменил действие")
    if z["k15h_h18"].shape[0] != min(C, hp.H18_CALLS):
        p.append(f"{label}: h18 записан не для вызовов 1–"
                 f"{hp.H18_CALLS}")
    return p


def check_vs_ref(label, d, z, rd, rz):
    """Против q0ref того же блока: старты, префикс до импульса, исход без
    импульса, h18/a0 вызовов 1–4 у достигнувших сред."""
    import k15h_policy as hp
    p = []
    eps, reps = d["episodes"], rd["episodes"]
    for i, (e, r) in enumerate(zip(eps, reps)):
        for f in ("init_state_id", "init_hash", "init_hash_full",
                  "rollout_seed"):
            if str(e.get(f)) != str(r.get(f)):
                p.append(f"{label}: среда {i}: {f} не равен q0ref")
    A, Ar = z["actions"], rz["actions"]
    ds, dr = np.asarray(z["done_step"]), np.asarray(rz["done_step"])
    n_pre = H * (MOMENT - 1)
    for b in range(A.shape[1]):
        if reached(int(dr[b]), MOMENT):
            if not np.array_equal(A[:n_pre, b], Ar[:n_pre, b]):
                p.append(f"{label}: префикс среды {b} до импульса не "
                         f"совпал с q0ref")
        else:
            end = int(dr[b]) + 1
            if int(ds[b]) != int(dr[b]) or \
                    eps[b].get("success") != reps[b].get("success") or \
                    not np.array_equal(A[:end, b], Ar[:end, b]):
                p.append(f"{label}: среда {b} без импульса разошлась с "
                         f"q0ref")
    hh, hr = z.get("k15h_h18"), rz.get("k15h_h18")
    if hh is None or hr is None:
        return p + [f"{label}: нет h18"]
    for k in range(min(len(hh), len(hr), hp.H18_CALLS)):
        for b in range(A.shape[1]):
            if not reached(int(dr[b]), k + 1):
                continue
            if not (np.array_equal(hh[k, b], hr[k, b]) and np.array_equal(
                    z["k15d_a0"][k, b], rz["k15d_a0"][k, b])):
                p.append(f"{label}: h18/a0 вызова {k + 1} среды {b} не "
                         f"равны q0ref")
    return p


def load(path):
    d = json.load(open(path))
    z = np.load(os.path.join(os.path.dirname(path), d["actions_npz"]),
                allow_pickle=True)
    return d, {k: z[k] for k in z.files}


def check_block(plan, plan_path, path, label, task, block, ref_path=None,
                bs=None):
    import k15g_local_probe as pr
    p = pr.validate_block(path, label, int(task), int(block),
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = load(path)
    j = d.get("joint") or {}
    for k, v in plan["expected_contract"].items():
        if json.dumps(j.get(k), sort_keys=True) != json.dumps(
                v, sort_keys=True):
            p.append(f"{label}: контракт {k} = {j.get(k)!r}, план {v!r}")
    dirs = [plan["directions"][str(task)][str(s)]
            for s in range(int(block), int(block) + plan["n_envs"])]
    if j.get("label") != label or j.get("directions") != dirs or \
            j.get("plan_sha1") != sha(plan_path):
        p.append(f"{label}: метка, направления или план руки не из плана")
    bs = bs or Basis(plan["inputs"]["basis"])
    p += check_arrays(label, z, dirs, bs)
    if label != "q0ref":
        if not ref_path or not os.path.exists(ref_path):
            return p + [f"{label}: нет q0ref блока"]
        rd, rz = load(ref_path)
        p += check_vs_ref(label, d, z, rd, rz)
    return p


def check_run(plan, plan_path, run_dir):
    """Все блоки плана: (проверенные {(метка, задача, блок): путь},
    непригодные/отсутствующие с причиной)."""
    bs = Basis(plan["inputs"]["basis"])
    ok, gaps = {}, []
    for t, bl in plan["blocks"].items():
        for b in bl:
            ref = os.path.join(run_dir, f"q0ref_t{t}_i{b}.json")
            for lab in plan["labels"]:
                pth = os.path.join(run_dir, f"{lab}_t{t}_i{b}.json")
                if not os.path.exists(pth):
                    gaps.append(dict(label=lab, task=int(t), block=int(b),
                                     why="не готов"))
                    continue
                if lab != "q0ref" and ("q0ref", int(t), int(b)) not in ok:
                    gaps.append(dict(label=lab, task=int(t), block=int(b),
                                     why="нет проверенного q0ref"))
                    continue
                prob = check_block(plan, plan_path, pth, lab, t, b, ref, bs)
                if prob:
                    gaps.append(dict(label=lab, task=int(t), block=int(b),
                                     why="; ".join(prob[:2])))
                else:
                    ok[(lab, int(t), int(b))] = pth
    return ok, gaps


def smoke_check(plan, plan_path, run_dir):
    """Smoke: всё проверено; q0ref = q0 переписи M1-cold; новый p4r3p
    задачи 8 = p4r3p K-15g по исполненным действиям, исходам и завершению."""
    ok, gaps = check_run(plan, plan_path, run_dir)
    p = [f"{g['label']} t{g['task']} i{g['block']}: {g['why']}"
         for g in gaps]
    for t, ref in plan["refs"]["q0"].items():
        b = plan["blocks"][t][0]
        mine = ok.get(("q0ref", int(t), int(b)))
        if mine:
            p += same_episodes(mine, ref, f"q0ref t{t} против q0 M1-cold",
                               actions=False)
    for t, ref in plan["refs"]["p4r3p"].items():
        b = plan["blocks"][t][0]
        mine = ok.get(("p4r3p", int(t), int(b)))
        if mine:
            p += same_episodes(mine, ref, f"p4r3p t{t} против K-15g",
                               actions=True)
    return p, ok


def same_episodes(mine, ref, what, actions):
    p = []
    d, z = load(mine)
    rd, rz = load(ref)
    for f in ("action_sha1", "success", "done_step", "init_hash"):
        a_ = [str(e.get(f)) for e in d["episodes"]]
        b_ = [str(e.get(f)) for e in rd["episodes"]]
        if a_ != b_:
            p.append(f"{what}: {f} не совпал")
    if actions:
        for b in range(z["actions"].shape[1]):
            ds = int(z["done_step"][b])
            end = ds + 1 if ds >= 0 else len(z["actions"])
            if not np.array_equal(z["actions"][:end, b],
                                  rz["actions"][:end, b]):
                p.append(f"{what}: исполненные действия среды {b} "
                         f"не совпали")
    return p


# --- самопроверка ------------------------------------------------------------
def selftest():
    import tempfile
    import k15h_policy as hp
    import k15h_static_basis as sb
    # инвентаризация: эпизоды, заголовки блоков, планы, логи, чужой набор
    got = set()

    def add(suite, t, s):
        if suite == SUITE:
            got.add((t, s))
    states_from_json(dict(task_id=3, suite="10", episodes=[
        dict(init_state_id=7), dict(init_state_id=8)]), add)
    states_from_json(dict(task_id=4, init_start=10, n_envs=5), add)
    states_from_json(dict(blocks={"2": [20]}), add)
    states_from_json(dict(task_id=5, suite="goal", episodes=[
        dict(init_state_id=1)]), add)
    states_from_log("=== suite 10, задача 6, офсет 1\n  --- блок 25-29, "
                    "сид\n=== suite 10, задача 7, офсет 2\n--- блок 0-4", add)
    assert {(3, 7), (3, 8)} <= got and all((4, s) in got
                                           for s in range(10, 15))
    assert all((2, s) in got for s in range(20, 25)) and (5, 1) not in got
    assert (6, 27) in got and (7, 0) in got and (6, 0) not in got
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "reports", "a"))
        os.makedirs(os.path.join(td, "reports", "empty"))
        json.dump(dict(task_id=1, episodes=[dict(init_state_id=3)]),
                  open(os.path.join(td, "reports", "a", "x.json"), "w"))
        open(os.path.join(td, "reports", "empty", "y.npz"), "w").write("z")
        open(os.path.join(td, "reports", "a", "bad.json"), "w").write("{")
        used, per_file, unrec, _o = scan([os.path.join(td, "reports")])
        assert (1, 3) in used and len(per_file) == 1
        assert any("empty" in u for u in unrec)
    # выбор банка: первые свободные блоки, запасной вариант, отказ
    n = {t: 50 for t in TASKS}
    used = {(t, s) for t in TASKS for s in range(25)}
    used.add((3, 31))
    bank, k, free = select_bank(used, n)
    assert k == 3 and bank[0] == [25, 30, 35] and bank[3] == [25, 35, 40]
    used2 = used | {(5, s) for s in range(25, 40)}
    bank2, k2, _f = select_bank(used2, n)
    assert k2 == 2 and bank2[5] == [40, 45]
    try:
        select_bank(used2 | {(5, 40)}, n)
        raise AssertionError("принят банк меньше запасного")
    except SystemExit:
        pass
    # таблица направлений: детерминирована сидом, порядок фиксирован
    t1 = directions_table(bank)
    assert t1 == directions_table(bank) and len(t1["0"]) == 15
    assert t1 != directions_table(bank, seed=1)
    assert set(v for d in t1.values() for v in d.values()) <= set(hp.DIRS)
    # проверка поведения политики по журналу
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.standard_normal((56, 4)))[0].T
    bs = type("B", (), {})()
    bs.U, bs.rho, bs.f = U, np.array([2.0, 1.5, 1.0, 0.8]), 1.0
    bs.sigma, bs.sigma_g = rng.uniform(0.02, 0.2, 6), 0.7
    bs.R = np.eye(56)
    C, B = 6, 5
    dirs = SMOKE_DIRS[0]
    L = np.repeat((bs.rho[:, None] * U)[None, None], C, 0).repeat(B, 1)
    L = L + rng.standard_normal(L.shape) * 0.01
    a0 = rng.uniform(-0.5, 0.5, (C, B, H, 7))

    def journal(label, used_fn=None, chunk_on=MOMENT, planned=None,
                applied_fix=None):
        planned = hp.planned_coeffs(label, dirs) if planned is None \
            else planned
        if label == "static_uniform":
            used_ = np.broadcast_to(sb.static_basis(bs.U, bs.rho, bs.f),
                                    L.shape).copy()
        elif label == "arm_dir_uniform":
            used_ = sb.arm_dir_basis(L.reshape(-1, 4, 56), bs.U,
                                     bs.sigma).reshape(L.shape)
        elif label == "p4r3p":
            used_ = L @ bs.R
        else:
            used_ = L.copy()
        if used_fn:
            used_ = used_fn(used_)
        active = (np.arange(1, C + 1) == chunk_on) if label != "q0ref" \
            else np.zeros(C, bool)
        applied = np.zeros((C, B, 4))
        applied[active] = planned
        if applied_fix:
            applied = applied_fix(applied)
        lv = a0.copy()
        for k in np.nonzero(active)[0]:
            t = np.einsum("bk,bkd->bd", applied[k], used_[k]).reshape(B, H, 7)
            lv[k] = combine_exec(a0[k], t, bs.sigma, bs.sigma_g)
        return {"k15d_a0": a0, f"k15d_level_{label}": lv, "k15f_basis": L,
                "k15h_basis_used": used_, "k15g_chunk": np.arange(1, C + 1),
                "k15g_active": active, "k15h_coeff_planned": planned,
                "k15h_coeff_applied": applied,
                "k15h_h18": np.zeros((4, B, 16, 8))}
    for lab in hp.LABELS:
        assert check_arrays(lab, journal(lab), dirs, bs) == [], lab
    # ошибки протокола, которые обязаны ловиться
    one_dir = np.tile(hp.planned_coeffs("learned_uniform", dirs)[:1], (B, 1))
    assert check_arrays("learned_uniform", journal(
        "learned_uniform", planned=one_dir), dirs, bs)       # одно на батч
    other = ["3m"] + dirs[1:]                                 # иной выбор
    assert check_arrays("learned_uniform", journal("learned_uniform"),
                        other, bs)
    assert check_arrays("static_uniform", journal(
        "static_uniform", used_fn=lambda u: u * bs.f * 2), dirs, bs) or \
        bs.f != 1.0
    assert check_arrays("static_uniform", journal(
        "static_uniform", used_fn=lambda u: u * 1.5), dirs, bs)  # амплитуда
    def renorm(u):
        u = u.copy()
        u[..., sb.ARM] *= 1.2
        return u
    assert check_arrays("arm_dir_uniform", journal(
        "arm_dir_uniform", used_fn=renorm), dirs, bs)         # нормировка
    def grip(u):
        u = u.copy()
        u[..., sb.GRIP] += 0.1
        return u
    assert check_arrays("arm_dir_uniform", journal(
        "arm_dir_uniform", used_fn=grip), dirs, bs)           # схват
    assert check_arrays("learned_uniform", journal(
        "learned_uniform", chunk_on=3), dirs, bs)             # другой чанк
    jj = journal("learned_uniform")
    jj["k15d_level_learned_uniform"] = jj[
        "k15d_level_learned_uniform"].copy()
    jj["k15d_level_learned_uniform"][5] += 0.01               # после импульса
    assert check_arrays("learned_uniform", jj, dirs, bs)
    # против q0ref: префикс, недостигнутая среда, h18
    eps = [dict(init_state_id=s, init_hash=f"h{s}", init_hash_full=f"H{s}",
                rollout_seed=101, success=False) for s in range(5)]
    A = rng.uniform(-1, 1, (40, B, 7)).astype(np.float32)
    rz = dict(actions=A, done_step=np.array([-1, 10, -1, -1, 30]),
              k15h_h18=np.ones((4, B, 3)), k15d_a0=a0)
    z = dict(actions=A.copy(), done_step=rz["done_step"].copy(),
             k15h_h18=np.ones((4, B, 3)), k15d_a0=a0)
    rd, d = dict(episodes=eps), dict(episodes=[dict(e) for e in eps])
    assert check_vs_ref("x", d, z, rd, rz) == []
    z2 = dict(z, actions=A.copy())
    z2["actions"][20, 0] += 1                                 # префикс
    assert any("префикс" in x for x in check_vs_ref("x", d, z2, rd, rz))
    z3 = dict(z, actions=A.copy())
    z3["actions"][30, 0] += 1                                 # после — можно
    assert check_vs_ref("x", d, z3, rd, rz) == []
    z4 = dict(z, actions=A.copy())
    z4["actions"][5, 1] += 1                                  # хвост без
    assert any("без импульса" in x for x in check_vs_ref("x", d, z4, rd,
                                                          rz))
    d5 = dict(episodes=[dict(e) for e in eps])
    d5["episodes"][1]["success"] = True
    assert any("без импульса" in x for x in check_vs_ref("x", d5, z, rd,
                                                          rz))
    h = np.ones((4, B, 3))
    h[3, 0] = 2                                               # вызов 4
    assert any("вызова 4" in x for x in check_vs_ref(
        "x", d, dict(z, k15h_h18=h), rd, rz))
    h2 = np.ones((4, B, 3))
    h2[3, 1] = 2           # среда 1 завершилась на шаге 10: вызов 4 не её
    assert check_vs_ref("x", d, dict(z, k15h_h18=h2), rd, rz) == []
    print("самопроверка k15h_protocol пройдена: инвентаризация, банк и "
          "запасной вариант, таблица направлений, формулы вариантов, одно "
          "направление на батч, повторный выбор, амплитуда, нормировка, "
          "схват, чанк, префикс, хвост без импульса, h18 до импульса")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-15h: протокол")
    ap.add_argument("mode", choices=("inventory", "smoke-plan", "plan",
                                     "check-block", "smoke-check",
                                     "check-plan", "selftest"))
    ap.add_argument("--plan", default=None)
    ap.add_argument("--basis", default="data/k15f/basis_s0.pt")
    ap.add_argument("--identity", default="reports/k15f/identity_cuda1.json")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--block", nargs=4, default=None,
                    metavar=("JSON", "МЕТКА", "ЗАДАЧА", "БЛОК"))
    ap.add_argument("--smoke-report", default="reports/k15h/smoke_check.json")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if a.mode == "inventory":
        return inventory_mode()
    if a.mode in ("smoke-plan", "plan"):
        smoke = a.mode == "smoke-plan"
        out = a.plan or (SMOKE_PLAN if smoke else PLAN)
        if os.path.exists(out):
            raise SystemExit(f"{out} уже зафиксирован")
        if not smoke:
            rep = (json.load(open(a.smoke_report))
                   if os.path.exists(a.smoke_report) else {})
            if rep.get("code") != 0:
                raise SystemExit("окончательный план — только после чистого "
                                 "smoke (smoke_check.json с кодом 0)")
        plan = make_plan(basis=a.basis, identity=a.identity,
                         device=a.device, smoke=smoke)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=1)
        os.replace(out + ".tmp", out)
        n = sum(len(v) for v in plan["blocks"].values())
        print(f"  план {out}: задачи {plan['tasks']}, блоков {n}, процессов "
              f"{n * len(plan['labels'])}, эпизодов "
              f"{n * len(plan['labels']) * N_ENVS}")
        for t, dd in plan["directions"].items():
            print(f"    задача {t}: " + " ".join(
                f"{s}:{v}" for s, v in dd.items()))
        return 0
    plan_path = a.plan or SMOKE_PLAN
    plan = json.load(open(plan_path))
    if a.mode == "check-plan":
        prob = check_plan_inputs(plan, plan_path)
        print("\n".join(prob) if prob else "  входы плана сверены")
        return 3 if prob else 0
    if a.mode == "check-block":
        path, lab, t, b = a.block
        ref = os.path.join(os.path.dirname(path), f"q0ref_t{t}_i{b}.json")
        prob = check_block(plan, plan_path, path, lab, t, b, ref)
        if prob:
            print("; ".join(prob[:6]))
        return 2 if prob else 0
    prob, ok = smoke_check(plan, plan_path, a.run_dir)
    rep = dict(code=3 if prob else 0, problems=prob, plan=plan_path,
               plan_sha1=sha(plan_path), run_dir=a.run_dir,
               verified=len(ok))
    with open(a.smoke_report + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    os.replace(a.smoke_report + ".tmp", a.smoke_report)
    print(f"  smoke: проверено блоков {len(ok)}; проблем {len(prob)}"
          + (f": {prob[:8]}" if prob else ""))
    return rep["code"]


if __name__ == "__main__":
    sys.exit(main())
