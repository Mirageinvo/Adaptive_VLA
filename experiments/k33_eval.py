#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-33, вторая очередь: план машины, очередь, транзакции и проверки блоков.

plan --machine A|B — reports/k33/plan_<M>.json (A = ccmplanner, B = cds2):
  банк (bank_id, SHA файлов, пройденный bank-check), снимок BAR и настройки
  раскатки (expected_setup K-9h + путь снимка этой машины), шесть рук с
  контентными отпечатками источников (k33_policy.make_source), SHA кода,
  разбиение B5-групп между машинами (группа g = блок 5g; g чётное — A,
  нечётное — B: по 30 групп каждой задачи на машину), очередь SR этой
  машины, ячейки legacy (только A), dev и repeat, лимиты бюджета.
  contract_fp — отпечаток всего, кроме машинных путей: у A и B он обязан
  совпасть. Готовый план не перезаписывается (другой контракт — отказ).
Очередь SR: порядок задач — перестановка seed 33010; на позиции p задачи
  руки идут в порядке ORDER0, циклически сдвинутом на p; внутри руки —
  группы машины по возрастанию. Один процесс K-9h = один блок (холодный
  старт, как во всех действующих оценках): 1800 процессов на машину.
Транзакция блока: block --op reserve (scope sr с лимитом машины либо
  checks) -> процесс K-9h -> атомарный JSON+npz -> block --op finish:
  проверка блока, расход = 5·(env_steps + 10) из JSON, commit/settle.
  Готовый результат при открытом резерве закрывается settle без повтора;
  проверка не пройдена — карантин и технический стоп.
Проверка блока: validate_block (5 эпизодов start…start+4, action SHA,
  настройки, одиночный процесс), ревизия BAR, метаданные руки, журнал руки
  (router и debug у T/F, токены у BC, исполненные действия = журнал), для
  банка — post-wait хеши каждого эпизода = банку.
legacy-compare / dev-check / repeat-compare / forecast / progress — см.
  функции ниже.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k33_bank as kb  # noqa: E402
import k33_policy as kp  # noqa: E402

KIND = "k33_plan"
MACHINES = dict(A="ccmplanner", B="cds2")
PLAN = "reports/k33/plan_{m}.json"
OUT = "reports/k33"
N_GROUPS = 60
TASK_ORDER_SEED = 33010
EP_CAP = 610                         # 600 управляющих + 10 ожидания
BLOCK_CAP = 5 * EP_CAP
LEGACY = ((0, 30), (8, 30))
DEV_START = 0
REPEAT_TASKS = (6, 8)                # холодные повторы dev на A
PORT_TASKS = (0,)                    # строгий перенос на B: dev t0
CHECKS_LIMIT = dict(A=331_700, B=18_300)   # с прежним расходом машины
REJECTED = "reports/k33/rejected"
PHASES = ("legacy_cur", "legacy_new", "dev", "repeat", "port", "sr")
STATS = dict(alpha=0.05, holm=2, boot_n=100_000, boot_seed=33011,
             ni_margin=0.03, ni_tail=0.05 / 8, ni_family=8,
             mcnemar="exact two-sided")
CODE = ("k33_bank.py", "k33_harness.py", "k33_policy.py", "k33_eval.py",
        "k33_protocol.py", "k33_analyze.py", "k33_latency.py", "k33_run.sh",
        "k18_collector.py", "k9h_multiarm_gate.py", "k16_harness.py",
        "k16_protocol.py", "k15g_local_probe.py", "k15h_protocol.py",
        "k15f_policy.py", "k26_policy.py", "k26_protocol.py",
        "k26_attention.py", "k26_model.py", "k21_cached_policy.py",
        "k21_cached_protocol.py", "k21_barcache.py", "k21_main_policy.py",
        "k21_policy.py", "k21_data.py", "k21_protocol.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def save_json(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def plan_path(m):
    return PLAN.format(m=m)


# --- разбиение и очередь ------------------------------------------------------------
def task_order():
    return [int(t) for t in np.random.default_rng(TASK_ORDER_SEED)
            .permutation(kb.N_TASKS)]


def arm_order(pos):
    k = pos % len(kp.ORDER0)
    return list(kp.ORDER0[k:] + kp.ORDER0[:k])


def machine_of_group(g):
    return "A" if g % 2 == 0 else "B"


def machine_starts(m):
    return [5 * g for g in range(N_GROUPS) if machine_of_group(g) == m]


def sr_queue(m):
    """[(task, arm, start)] — очередь SR машины в порядке исполнения."""
    q = []
    for pos, t in enumerate(task_order()):
        for arm in arm_order(pos):
            q += [(t, arm, s) for s in machine_starts(m)]
    return q


def debug_for(arm, phase):
    """Полный debug первого вызова router — у T/F в legacy (как в
    smoke-ячейках K-26) и в development-проверках; в SR — нет."""
    if kp.ARMS[arm]["kind"] not in ("tree", "flat"):
        return False
    return phase in ("legacy_cur", "legacy_new", "dev", "repeat", "port")


def cells(m, phase):
    """[(arm, task, start)] фазы на машине m в порядке исполнения."""
    if phase in ("legacy_cur", "legacy_new"):
        return [] if m != "A" else [(a, t, s) for a in kp.ORDER0
                                    for t, s in LEGACY]
    if phase == "dev":
        return [] if m != "A" else [(a, t, DEV_START) for t in task_order()
                                    for a in kp.ORDER0]
    if phase == "repeat":
        return [] if m != "A" else [(a, t, DEV_START) for t in REPEAT_TASKS
                                    for a in kp.ORDER0]
    if phase == "port":
        return [] if m != "B" else [(a, t, DEV_START) for t in PORT_TASKS
                                    for a in kp.ORDER0]
    if phase == "sr":
        return [(a, t, s) for t, a, s in sr_queue(m)]
    raise SystemExit(f"фаза {phase}")


def phase_mode(phase):
    if phase in ("legacy_cur", "legacy_new"):
        return "legacy", None
    if phase == "sr":
        return "bank", 1
    return "bank", 0


def block_path(m, phase, arm, task, start):
    return os.path.join(OUT, phase, m, arm, f"t{task}_i{start}.json")


def out_pattern(m, phase, arm, task):
    return os.path.join(OUT, phase, m, arm, f"t{task}_i{{i0}}.json")


def eid(plan, phase, arm, task, start):
    if phase == "sr":
        return f"sr/{plan['bank']['bank_id'][:12]}/{arm}/t{task}/b{start}"
    return f"checks/k33/{phase}/{plan['machine']}/{arm}/t{task}/b{start}"


def limits(m):
    n_sr = len(sr_queue(m))
    return dict(sr=n_sr * BLOCK_CAP, checks=CHECKS_LIMIT[m],
                checks_new=(len(cells(m, "legacy_cur")) +
                            len(cells(m, "legacy_new")) +
                            len(cells(m, "dev")) + len(cells(m, "repeat")) +
                            len(cells(m, "port"))) * BLOCK_CAP)


def wants_bsnap(arm, mode, space):
    """Контрольный снимок B (токены -> decode -> команды) — в development."""
    return arm == "B" and mode == "bank" and space == 0


def schedule_problems(plan):
    """Ячейки, очередь и лимиты плана = расписанию из кода для его машины;
    ни одна обязательная фаза не пуста."""
    m = plan.get("machine")
    if m not in MACHINES:
        return [f"машина {m}"]
    p = []
    for ph in PHASES:
        want = [list(x) for x in cells(m, ph)]
        if plan.get("cells", {}).get(ph) != want:
            p.append(f"ячейки {ph} ≠ расписанию машины {m}")
    if plan.get("queue_sr") != [list(x) for x in sr_queue(m)]:
        p.append("очередь SR ≠ расписанию")
    if plan.get("limits") != limits(m):
        p.append("лимиты ≠ расписанию")
    need = (("legacy_cur", "legacy_new", "dev", "repeat", "sr") if m == "A"
            else ("port", "sr"))
    p += [f"фаза {ph} пуста" for ph in need if not plan.get("cells", {}).get(ph)]
    return p


# --- план ----------------------------------------------------------------------------
def bank_record():
    import k33_protocol as kpr
    m = json.load(open(kb.MANIFEST, encoding="utf-8"))
    bad = [p for p, s in m["files"].items()
           if not os.path.exists(p) or kpr.sha(p) != s]
    if bad:
        raise SystemExit(f"файлы банка изменены {bad[:3]}")
    if not os.path.exists(kpr.CHECK):
        raise SystemExit("нет пройденного bank-check")
    c = json.load(open(kpr.CHECK, encoding="utf-8"))
    if c.get("bank_id") != m["bank_id"] or c.get("ok") is not True:
        raise SystemExit("bank-check не пройден для этого банка")
    return dict(bank_id=m["bank_id"], manifest=os.path.abspath(kb.MANIFEST),
                manifest_sha1=sha(kb.MANIFEST), files=m["files"],
                dir=os.path.abspath(kb.BANK_DIR),
                bank_check_sha1=sha(kpr.CHECK))


def contract_of(plan):
    """Всё содержательное, без машинных путей и времени."""
    arms = {a: dict(label=v["label"], policy=v["policy"],
                    k9h_args=v["k9h_args"], content=v["source"]["content"])
            for a, v in plan["arms"].items()}
    setup = {k: v for k, v in plan["setup"].items() if k != "ckpt"}
    c = dict(bank_id=plan["bank"]["bank_id"],
             bank_files=sorted(plan["bank"]["files"].items()),
        arms=arms, setup=setup, bar_revision=plan["bar_revision"],
        prep_id=plan["prep_id"], bounds=plan["bounds"], code=plan["code"],
        task_order=plan["task_order"], groups=plan["groups"],
        stats=plan["stats"], ep_cap=EP_CAP)
    return hashlib.sha1(json.dumps(c, sort_keys=True,
                                   default=str).encode()).hexdigest()


def make_plan(m):
    import k15g_local_probe as pr
    import k21_data as d21
    if m not in MACHINES:
        raise SystemExit(f"машина {m}")
    prep = json.load(open(d21.REPORT, encoding="utf-8"))
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    plan = dict(
        kind=KIND, machine=m, host=MACHINES[m],
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        bank=bank_record(),
        setup=dict(pr.expected_setup(), ckpt=snap),
        bar_revision=prep["bar_pin"]["revision"], prep_id=prep["prep_id"],
        bounds=prep["bounds_codec"],
        arms={a: dict(label=kp.ARMS[a]["label"],
                      policy=kp.ARMS[a]["policy"],
                      k9h_args=kp.k9h_args(a), source=kp.make_source(a))
              for a in kp.ORDER0},
        code={f: sha(os.path.join(HERE, f)) for f in CODE},
        task_order=task_order(),
        groups=dict(n=N_GROUPS, rule="g чётное — A, нечётное — B",
                    A=machine_starts("A"), B=machine_starts("B")),
        queue_sr=[list(x) for x in sr_queue(m)],
        cells={ph: [list(x) for x in cells(m, ph)] for ph in PHASES},
        limits=limits(m), stats=dict(STATS),
        one_block_per_process=True,
        note="LIBERO-10, свежий банк native-reset K-33; не официальные 50 "
             "стартов и не новый benchmark")
    plan["contract_fp"] = contract_of(plan)
    return plan


def plan_problems(plan, arms=None):
    """Код, банк, снимок BAR и источники рук (по умолчанию всех) = плану."""
    import k33_protocol as kpr
    p = []
    if plan.get("kind") != KIND:
        return ["не план K-33"]
    p += schedule_problems(plan)
    for f, s_ in plan["code"].items():
        q = os.path.join(HERE, f)
        if not os.path.exists(q) or sha(q) != s_:
            p.append(f"код {f} изменён после плана")
    b = plan["bank"]
    if not os.path.exists(b["manifest"]) or sha(b["manifest"]) != \
            b["manifest_sha1"]:
        p.append("манифест банка изменён")
    else:
        bad = [f for f, s_ in b["files"].items()
               if not os.path.exists(f) or kpr.sha(f) != s_]
        if bad:
            p.append(f"файлы банка изменены {bad[:2]}")
    if contract_of(plan) != plan.get("contract_fp"):
        p.append("contract_fp ≠ содержимому плана")
    for a in (arms if arms is not None else list(plan["arms"])):
        p += kp.source_problems(a, plan["arms"][a]["source"])
    return p


def cell_problems(plan, cell):
    """Ячейка процесса — из плана этой машины (фаза определяется режимом,
    пространством, блоком и debug)."""
    arm, t, s = cell["arm"], int(cell["task"]), int(cell["start"])
    ok = False
    for ph in PHASES:
        mode, space = phase_mode(ph)
        if mode != cell["mode"] or space != cell["space"]:
            continue
        if [arm, t, s] in plan["cells"][ph] and \
                bool(cell["debug"]) == debug_for(arm, ph):
            ok = True
    return [] if ok else [f"ячейка {arm} t{t} i{s} ({cell['mode']}, "
                          f"space {cell['space']}, debug {cell['debug']}) "
                          f"не из плана машины {plan['machine']}"]


def plan_cmd(a):
    p = plan_path(a.machine)
    new = make_plan(a.machine)
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
        if old.get("contract_fp") == new["contract_fp"] and \
                old["setup"] == new["setup"] and not plan_problems(old):
            print(f"  план {p}: закреплён, контракт {new['contract_fp'][:12]}")
            return 0
        raise SystemExit(f"{p}: закреплённый план другого контракта или "
                         f"путей — новый план только в новом namespace")
    save_json(p, new)
    other = plan_path("B" if a.machine == "A" else "A")
    tail = ""
    if os.path.exists(other):
        o = json.load(open(other, encoding="utf-8"))
        tail = (", контракт = другой машине" if o["contract_fp"] ==
                new["contract_fp"] else ", КОНТРАКТ ≠ ДРУГОЙ МАШИНЕ")
    print(f"  план {p}: машина {a.machine} ({MACHINES[a.machine]}), "
          f"bank_id {new['bank']['bank_id'][:12]}, контракт "
          f"{new['contract_fp'][:12]}, SR-блоков {len(new['queue_sr'])}, "
          f"лимит sr {new['limits']['sr']}{tail}")
    return 0


# --- проверка блока ----------------------------------------------------------------------
def bank_parity(space, task, start, eps, load=None):
    load = load or kb.load_task
    cases, _m = load(int(space), int(task))
    p = []
    for i, e in enumerate(eps):
        c = cases[int(start) + i]
        if int(e["init_state_id"]) != int(start) + i:
            p.append(f"эпизод {i}: init_state_id {e['init_state_id']}")
        elif e["init_hash"] != c["init_hash"] or \
                e["init_hash_full"] != c["init_hash_full"]:
            p.append(f"{c['case_id']}: post-wait хеш ≠ банку")
    return p


def block_problems(plan, plan_file, phase, arm, task, start, path=None):
    import k15g_local_probe as pr
    import k16_protocol as p16
    path = path or block_path(plan["machine"], phase, arm, task, start)
    if not os.path.exists(path):
        return [f"нет блока {path}"]
    lab = kp.ARMS[arm]["label"]
    p = pr.validate_block(path, lab, int(task), int(start),
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = p16.load(path)
    if os.path.basename(os.path.normpath(str(d.get("ckpt")))) != \
            plan["bar_revision"]:
        p.append(f"{arm}: снимок BAR блока не закреплённой ревизии")
    dbg = debug_for(arm, phase)
    if phase != "legacy_cur":
        p += kp.joint_problems(arm, d, plan["arms"][arm]["source"],
                               sha(plan_file), dbg)
    p += kp.journal_problems(arm, z, plan["bounds"], dbg)
    mode, space = phase_mode(phase)
    if mode == "bank":
        p += bank_parity(space, task, start, d["episodes"])
    if wants_bsnap(arm, mode, space):
        import k15h_protocol as hpc
        p += kp.bsnap_problems(path, z, hpc.harness_scale())
    return p


def block_actual(path):
    d = json.load(open(path, encoding="utf-8"))
    eps = d["episodes"]
    steps = int(eps[0]["env_steps"])
    if any(int(e["env_steps"]) != steps for e in eps) or len(eps) != 5:
        raise SystemExit(f"{path}: env_steps эпизодов различаются")
    return len(eps) * (steps + int(d.get("waiting_steps", 10)))


def reject_path(e):
    return os.path.join(REJECTED, e.replace("/", "__") + ".json")


def mark_rejected(e, reasons, q):
    save_json(reject_path(e), dict(eid=e, reasons=reasons, quarantine=q,
                                   time=time.strftime("%Y-%m-%dT%H:%M:%S")))


def quarantine(path):
    q = os.path.join(OUT, "quarantine", time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(q, exist_ok=True)
    for f in (path, path[:-5] + ".actions.npz"):
        if os.path.exists(f):
            shutil.move(f, os.path.join(q, os.path.basename(
                os.path.dirname(f)) + "_" + os.path.basename(f)))
    return q


def ledger_for(plan):
    import k33_protocol as kpr
    return kpr.ledger()


def scope_of(phase):
    return "sr" if phase == "sr" else "checks"


def txn_reserve(L, plan, phase, arm, task, start):
    e = eid(plan, phase, arm, task, start)
    path = block_path(plan["machine"], phase, arm, task, start)
    if os.path.exists(reject_path(e)):
        raise SystemExit(f"{e}: результат был отвергнут ({reject_path(e)}) — "
                         f"технический стоп; новая попытка только по решению")
    if os.path.exists(path):
        raise SystemExit(f"{path}: результат уже есть — нужен finish")
    if L.is_done(e):
        raise SystemExit(f"{e}: учтён, но результата нет")
    if plan["machine"] == "B":
        pr_ = contract_ab_problems(plan)
        if pr_:
            raise SystemExit("; ".join(pr_))
    lim = plan["limits"]["sr"] if phase == "sr" else plan["limits"]["checks"]
    L.reserve(scope_of(phase), e, amount=BLOCK_CAP, limit=lim)
    return e


def contract_ab_problems(plan):
    """B исполняет только при плане A того же контракта."""
    pa = plan_path("A")
    if not os.path.exists(pa):
        return ["нет плана A (reports/k33/plan_A.json) для сверки контракта"]
    a = json.load(open(pa, encoding="utf-8"))
    p = schedule_problems(a)
    if a.get("contract_fp") != plan["contract_fp"] or \
            contract_of(a) != a.get("contract_fp"):
        p.append("контракт B ≠ контракту A")
    return p


def txn_finish(L, plan, plan_file, phase, arm, task, start):
    """Проверка результата и учёт: commit при своём открытом резерве,
    сверка при уже учтённом; чужое состояние журнала — отказ."""
    e = eid(plan, phase, arm, task, start)
    path = block_path(plan["machine"], phase, arm, task, start)
    if os.path.exists(reject_path(e)):
        raise SystemExit(f"{e}: результат был отвергнут — технический стоп")
    pr_ = block_problems(plan, plan_file, phase, arm, task, start, path)
    if pr_:
        q = quarantine(path) if os.path.exists(path) else None
        mark_rejected(e, pr_, q)
        raise SystemExit(f"{arm} t{task} i{start}: " + "; ".join(pr_[:4]) +
                         (f" -> карантин {q}" if q else "") +
                         " (маркер отказа сохранён)")
    actual = block_actual(path)
    done = L.d["done"].get(e)
    if done is not None and done.get("scope") != scope_of(phase):
        raise SystemExit(f"{e}: учтён в scope {done.get('scope')}")
    pd = L.d["pending"]
    if done is None and not (pd and pd.get("id") == e):
        raise SystemExit(f"{e}: результат есть, но не учтён и не "
                         f"зарезервирован")
    if pd and pd.get("id") == e and pd.get("scope") != scope_of(phase):
        raise SystemExit(f"{e}: резерв в scope {pd.get('scope')}")
    L.settle(e, actual)
    return actual


def block_cmd(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    # источники руки (полные SHA весов) сверяет процесс K-33 перед
    # раскаткой; здесь — код, банк и контракт
    pr_ = plan_problems(plan, arms=[])
    if pr_:
        raise SystemExit("план: " + "; ".join(pr_[:4]))
    if [a.arm, a.task, a.start] not in plan["cells"][a.phase]:
        raise SystemExit(f"{a.arm} t{a.task} i{a.start} не из фазы {a.phase}")
    L = ledger_for(plan)
    if a.op == "reserve":
        e = txn_reserve(L, plan, a.phase, a.arm, a.task, a.start)
        print(f"    резерв {e}")
        return 0
    if a.op == "status":
        path = block_path(plan["machine"], a.phase, a.arm, a.task, a.start)
        return 0 if os.path.exists(path) else 1
    act = txn_finish(L, plan, a.plan, a.phase, a.arm, a.task, a.start)
    sc = scope_of(a.phase)
    lim = plan["limits"]["sr"] if sc == "sr" else plan["limits"]["checks"]
    print(f"    {a.arm} t{a.task} i{a.start}: проверен, шагов {act}; {sc} "
          f"{L.spent(sc)}/{lim}")
    return 0


def cells_cmd(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    pr_ = schedule_problems(plan)
    if pr_:
        raise SystemExit("расписание плана: " + "; ".join(pr_[:3]))
    if not plan["cells"][a.phase]:
        raise SystemExit(f"фаза {a.phase} пуста на машине {plan['machine']}")
    print(f"#{len(plan['cells'][a.phase])}")
    for arm, t, s in plan["cells"][a.phase]:
        mode, space = phase_mode(a.phase)
        print(f"{arm}:{t}:{s}:{mode}:{space if space is not None else '-'}:"
              f"{int(debug_for(arm, a.phase))}")
    return 0


# --- сверки фаз -------------------------------------------------------------------------
GROUPS_CMP = (("init_hash", "init_hash_full"),
              ("action_sha1", "env_steps", "policy_calls"),
              ("success", "success_step", "done_step", "own_steps"))


def pair_problems(fa, fb, tag):
    """Два блока одной руки и ячейки: сцена, действия, исходы побитово."""
    import k16_protocol as p16
    da, za = p16.load(fa)
    db, zb = p16.load(fb)
    p = []
    for i, (ea, eb) in enumerate(zip(da["episodes"], db["episodes"])):
        for grp in GROUPS_CMP:
            diff = [k for k in grp if ea.get(k) != eb.get(k)]
            if diff:
                p.append(f"{tag} эпизод {i}: {diff}")
                break
    if len(da["episodes"]) != len(db["episodes"]):
        p.append(f"{tag}: число эпизодов")
    if not np.array_equal(za["actions"], zb["actions"]):
        p.append(f"{tag}: исполненные действия не совпали побитово")
    return p


GATES = dict(A=("legacy", "dev", "repeat"), B=("port",))
GATE_PHASES = dict(legacy=("legacy_cur", "legacy_new"), dev=("dev",),
                   repeat=("repeat",), port=("port",))


def gate_path(tag, m):
    return os.path.join(OUT, f"gate_{tag}_{m}.json")


def block_files(path):
    fs = [path, path[:-5] + ".actions.npz"]
    if os.path.exists(kp.bsnap_path(path)):
        fs.append(kp.bsnap_path(path))
    return fs


def expected_blocks(tag, m):
    """JSON блоков, на которых стоит гейт (включая эталоны сравнения)."""
    out = []
    for ph in GATE_PHASES[tag]:
        out += [block_path(m, ph, a, t, s) for a, t, s in cells(m, ph)]
    if tag == "port":
        out += [block_path("A", "dev", a, t, s) for a, t, s in cells(m, "port")]
    return out


def write_gate(tag, plan, plan_file, prob, extra=None):
    m = plan["machine"]
    blocks = {}
    for f in expected_blocks(tag, m):
        for q in (block_files(f) if os.path.exists(f) else []):
            blocks[q] = sha(q)
    rep = dict(tag=tag, machine=m, plan=os.path.abspath(plan_file),
               plan_sha1=sha(plan_file), contract_fp=plan["contract_fp"],
               bank_id=plan["bank"]["bank_id"],
               cells={ph: [list(x) for x in cells(m, ph)]
                      for ph in GATE_PHASES[tag]},
               blocks=blocks, ok=not prob, problems=prob[:100],
               time=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
    save_json(gate_path(tag, m), rep)
    return rep


def verify_gate(tag, m, plan, plan_file):
    """Отчёт гейта этой машины и этого плана: ok, контракт, банк, SHA
    плана, полный ожидаемый набор ячеек и неизменённые проверенные блоки."""
    f = gate_path(tag, m)
    if not os.path.exists(f):
        return [f"нет отчёта гейта {tag} машины {m}"]
    r = json.load(open(f, encoding="utf-8"))
    p = []
    if r.get("ok") is not True:
        p.append(f"гейт {tag} {m} не пройден")
    for k, v in (("machine", m), ("plan_sha1", sha(plan_file)),
                 ("contract_fp", plan["contract_fp"]),
                 ("bank_id", plan["bank"]["bank_id"])):
        if r.get(k) != v:
            p.append(f"гейт {tag} {m}: {k} не этого плана")
    want = {ph: [list(x) for x in cells(m, ph)] for ph in GATE_PHASES[tag]}
    if r.get("cells") != want or not any(want.values()):
        p.append(f"гейт {tag} {m}: набор ячеек ≠ расписанию или пуст")
    blocks = r.get("blocks") or {}
    for q in expected_blocks(tag, m):
        if q not in blocks:
            p.append(f"гейт {tag} {m}: блок {q} не проверен")
    for q, h in blocks.items():
        if not os.path.exists(q) or sha(q) != h:
            p.append(f"гейт {tag} {m}: {q} отсутствует или изменён")
    return p


def legacy_compare(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    prob = schedule_problems(plan)
    for arm, t, s in plan["cells"]["legacy_new"]:
        fc = block_path("A", "legacy_cur", arm, t, s)
        fn = block_path("A", "legacy_new", arm, t, s)
        for ph, f in (("legacy_cur", fc), ("legacy_new", fn)):
            prob += [f"{ph}: {x}" for x in block_problems(
                plan, a.plan, ph, arm, t, s, f)]
        if os.path.exists(fc) and os.path.exists(fn):
            prob += pair_problems(fc, fn, f"{arm} t{t}")
    rep = write_gate("legacy", plan, a.plan, prob)
    n = len(plan["cells"]["legacy_new"])
    print("  legacy-smoke 6 рук: " + (
        f"совпадение побитово ({n} блоков × 5: сцены, действия, исходы)"
        if rep["ok"] else "РАСХОЖДЕНИЕ: " + "; ".join(prob[:4])))
    return 0 if rep["ok"] else 3


def dev_check(a):
    """Все блоки фазы проверены; сцены и pre-policy хеши одинаковы у всех
    рук и = банку; SR — описательно. Для port (B) — строгое побитовое
    совпадение с эталонными dev-блоками A (обязательны) и контракт = A."""
    plan = json.load(open(a.plan, encoding="utf-8"))
    ph = a.phase
    prob = schedule_problems(plan)
    sr, hashes = {}, {}
    for arm, t, s in plan["cells"][ph]:
        f = block_path(plan["machine"], ph, arm, t, s)
        prob += block_problems(plan, a.plan, ph, arm, t, s, f)
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        sr.setdefault(arm, []).extend(bool(e["success"])
                                      for e in d["episodes"])
        for e in d["episodes"]:
            hashes.setdefault((t, e["init_state_id"]), set()).add(
                (e["init_hash"], e["init_hash_full"]))
    multi = [k for k, v in hashes.items() if len(v) != 1]
    if multi:
        prob.append(f"pre-policy хеши различаются между руками: {multi[:3]}")
    if not plan["cells"][ph]:
        prob.append(f"фаза {ph} пуста")
    extra = dict(sr_descriptive={k: float(np.mean(v)) for k, v in sr.items()},
                 n={k: len(v) for k, v in sr.items()})
    if ph == "port":
        prob += contract_ab_problems(plan)
        prob += port_problems(plan)
    tag = "port" if ph == "port" else "dev"
    rep = write_gate(tag, plan, a.plan, prob, extra)
    print(f"  {ph}: " + (f"блоков {len(plan['cells'][ph])}, все проверки "
                         f"пройдены; сцены рук совпадают; SR (описательно) "
                         + ", ".join(f"{k} {v:.2f}" for k, v in
                                     extra["sr_descriptive"].items())
                         if rep["ok"] else "ОТКАЗ: " + "; ".join(prob[:4])))
    return 0 if rep["ok"] else 3


def port_problems(plan):
    """Перенос на B: эталонные dev-блоки A (скопированные, проверенные по
    плану A) и блоки port B совпадают побитово (сцены, действия, исходы)."""
    pa = plan_path("A")
    if not os.path.exists(pa):
        return ["нет плана A"]
    plan_a = json.load(open(pa, encoding="utf-8"))
    p = []
    for arm, t, s in plan["cells"]["port"]:
        fa = block_path("A", "dev", arm, t, s)
        fb = block_path("B", "port", arm, t, s)
        if not os.path.exists(fa):
            p.append(f"нет эталонного блока A {fa}")
            continue
        p += [f"эталон A: {x}" for x in block_problems(
            plan_a, pa, "dev", arm, t, s, fa)]
        if os.path.exists(fb):
            p += pair_problems(fa, fb, f"перенос {arm} t{t}")
    return p


def repeat_compare(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    prob = schedule_problems(plan)
    for arm, t, s in plan["cells"]["repeat"]:
        fr = block_path("A", "repeat", arm, t, s)
        fd = block_path("A", "dev", arm, t, s)
        prob += [f"repeat: {x}" for x in block_problems(
            plan, a.plan, "repeat", arm, t, s, fr)]
        if not os.path.exists(fd):
            prob.append(f"нет dev-блока {fd}")
        elif os.path.exists(fr):
            prob += pair_problems(fd, fr, f"{arm} t{t}")
    rep = write_gate("repeat", plan, a.plan, prob)
    print(f"  холодные повторы dev ({'/'.join(f't{t}' for t in REPEAT_TASKS)}"
          f" × 6 рук): " + ("действия и исходы совпали побитово" if rep["ok"]
                           else "РАСХОЖДЕНИЕ: " + "; ".join(prob[:4])))
    return 0 if rep["ok"] else 3


def gates_problems(plan, plan_file):
    """Технические гейты до SR машины: свои; для B — ещё гейты A по
    скопированным плану, отчётам и блокам A и контракт = A."""
    m = plan["machine"]
    p = schedule_problems(plan)
    for tag in GATES[m]:
        p += verify_gate(tag, m, plan, plan_file)
    if m == "B":
        p += contract_ab_problems(plan)
        pa = plan_path("A")
        if os.path.exists(pa):
            plan_a = json.load(open(pa, encoding="utf-8"))
            for tag in GATES["A"]:
                p += verify_gate(tag, "A", plan_a, pa)
    return p


def gates_cmd(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    p = gates_problems(plan, a.plan) + plan_problems(plan)
    print("  гейты SR: " + ("пройдены" if not p else "ОТКАЗ: " +
                            "; ".join(p[:4])))
    return 0 if not p else 3


# --- время, диск, прогресс ----------------------------------------------------------------
TIMING = "reports/k33/timing_{m}.tsv"


def forecast(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    tf = TIMING.format(m=plan["machine"])
    if not os.path.exists(tf):
        raise SystemExit(f"нет {tf}")
    rows = [ln.split("\t") for ln in open(tf, encoding="utf-8")
            if ln.strip()]
    by = {}
    for ph, arm, _t, _s, t0, t1 in rows:
        if ph in ("dev", "port", "repeat"):
            by.setdefault(arm, []).append(float(t1) - float(t0))
    if set(by) != set(kp.ORDER0):
        raise SystemExit(f"время есть не для всех рук: {sorted(by)}")
    n_sr = {arm: sum(1 for x in plan["queue_sr"] if x[1] == arm)
            for arm in kp.ORDER0}
    hours = {arm: float(np.mean(v)) * n_sr[arm] / 3600 for arm, v in
             by.items()}
    sizes = []
    for ph in ("dev", "port"):
        d = os.path.join(OUT, ph, plan["machine"])
        if os.path.isdir(d):
            for root, _d, fs in os.walk(d):
                sizes += [os.path.getsize(os.path.join(root, f)) for f in fs]
    blocks = max(1, len(sizes) // 2)
    disk_gb = sum(sizes) / blocks * len(plan["queue_sr"]) / 2 ** 30
    rep = dict(machine=plan["machine"], sec_per_block={
        k: float(np.mean(v)) for k, v in by.items()}, sr_blocks=n_sr,
        hours=hours, total_hours=sum(hours.values()), disk_gb=disk_gb,
        note="по фактическому времени процессов dev (загрузка модели, "
             "reset, рендер, препроцессинг, модель)")
    save_json(os.path.join(OUT, f"forecast_{plan['machine']}.json"), rep)
    print(f"  прогноз SR машины {plan['machine']}: {rep['total_hours']:.0f} ч "
          f"({rep['total_hours'] / 24:.1f} сут), диск ≈ {disk_gb:.2f} ГиБ; "
          f"с/блок: " + ", ".join(f"{k} {v:.0f}" for k, v in
                                  rep["sec_per_block"].items()))
    return 0


def progress(a):
    plan = json.load(open(a.plan, encoding="utf-8"))
    done = sum(1 for t, arm, s in plan["queue_sr"]
               if os.path.exists(block_path(plan["machine"], "sr", arm, t, s)))
    print(f"  SR машины {plan['machine']}: {done}/{len(plan['queue_sr'])} "
          f"блоков")
    return 0


# --- самопроверка ---------------------------------------------------------------------------
def gates_regression():
    """Гейты: пустой/чужой/неполный/изменённый отчёт — отказ; перенос без
    эталона A или с расхождением — отказ; отвергнутый блок — стоп до
    резерва; ячейки фазы получаются только по расписанию."""
    import tempfile
    import k31_collect as kcol
    import k33_protocol as kpr
    g = globals()
    keep = dict(OUT=g["OUT"], REJECTED=g["REJECTED"], PLAN=g["PLAN"],
                block_problems=g["block_problems"])
    with tempfile.TemporaryDirectory() as td:
        try:
            g.update(OUT=td, REJECTED=os.path.join(td, "rej"),
                     PLAN=os.path.join(td, "plan_{m}.json"))
            g["block_problems"] = lambda *aa, **kw: []

            def mkplan(m, fp="C1"):
                pl = dict(machine=m, contract_fp=fp,
                          bank=dict(bank_id="bank1"),
                          cells={ph: [list(x) for x in cells(m, ph)]
                                 for ph in PHASES},
                          queue_sr=[list(x) for x in sr_queue(m)],
                          limits=limits(m))
                f = plan_path(m)
                save_json(f, pl)
                return pl, f

            def mkblock(f, acts=0.0, succ=True):
                os.makedirs(os.path.dirname(f), exist_ok=True)
                npz = f[:-5] + ".actions.npz"
                np.savez(npz, actions=np.full((3, 5, 7), acts, np.float32))
                json.dump(dict(actions_npz=os.path.basename(npz), episodes=[
                    dict(init_hash="h", init_hash_full="f", action_sha1="a",
                         env_steps=3, policy_calls=1, success=succ,
                         success_step=1, done_step=1, own_steps=2,
                         init_state_id=i) for i in range(5)]), open(f, "w"))
            pa, fa = mkplan("A")
            # гейт без блоков — отказ при проверке
            write_gate("dev", pa, fa, [])
            assert verify_gate("dev", "A", pa, fa)
            for ph in ("dev",):
                for arm, t, s_ in cells("A", ph):
                    mkblock(block_path("A", ph, arm, t, s_))
            write_gate("dev", pa, fa, [])
            assert verify_gate("dev", "A", pa, fa) == []
            # чужой контракт/банк/план — отказ
            pa2 = dict(pa, contract_fp="C2")
            assert verify_gate("dev", "A", pa2, fa)
            assert verify_gate("dev", "A", dict(pa, bank=dict(bank_id="x")),
                               fa)
            # изменённый блок после гейта — отказ
            f0 = block_path("A", "dev", *cells("A", "dev")[0])
            mkblock(f0, acts=1.0)
            assert verify_gate("dev", "A", pa, fa)
            mkblock(f0)
            write_gate("dev", pa, fa, [])
            # отчёт с усечённым набором ячеек — отказ
            r = json.load(open(gate_path("dev", "A")))
            r["cells"]["dev"] = r["cells"]["dev"][:3]
            save_json(gate_path("dev", "A"), r)
            assert verify_gate("dev", "A", pa, fa)
            write_gate("dev", pa, fa, [])
            # нет гейтов legacy/repeat — SR A не допускается
            assert gates_problems(pa, fa)
            # перенос: нет эталона A — отказ; эталон есть и совпадает — ок;
            # расхождение действий — отказ
            pb, fb = mkplan("B")
            for arm, t, s_ in cells("B", "port"):
                mkblock(block_path("B", "port", arm, t, s_))
            assert port_problems(pb) == []                 # эталоны dev A есть
            fref = block_path("A", "dev", *cells("B", "port")[0])
            os.rename(fref, fref + ".x")
            assert port_problems(pb)
            os.rename(fref + ".x", fref)
            mkblock(block_path("B", "port", *cells("B", "port")[1]),
                    acts=0.5)
            assert port_problems(pb)
            mkblock(block_path("B", "port", *cells("B", "port")[1]))
            # контракт B ≠ A — отказ до резерва
            g_contract = g["contract_of"]
            g["contract_of"] = lambda plan_: plan_["contract_fp"]
            assert contract_ab_problems(pb) == []
            assert contract_ab_problems(dict(pb, contract_fp="C9"))
            L = kcol.Ledger(os.path.join(td, "l.json"), kpr.CAPS)
            t0_, a0_, s0_ = sr_queue("B")[0]
            try:
                txn_reserve(L, dict(pb, contract_fp="C9"), "sr", a0_, t0_,
                            s0_)
                raise AssertionError("B резервирует при чужом контракте")
            except SystemExit as ex:
                assert "контракт" in str(ex.code), ex.code
            assert L.d["pending"] is None
            txn_reserve(L, pb, "sr", a0_, t0_, s0_)       # тот же контракт
            L._crash_pending()
            # гейты B требуют и гейты A
            write_gate("port", pb, fb, [])
            assert gates_problems(pb, fb)
            g["contract_of"] = g_contract
            # маркер отказа: после карантина новый резерв той же ячейки —
            # отказ (стоп), журнал не меняется
            g["block_problems"] = lambda *aa, **kw: ["плохо"]
            e = eid(pa, "dev", "B", 0, 0)
            L.reserve("checks", e, amount=BLOCK_CAP)
            fbad = block_path("A", "dev", "B", 0, 0)
            mkblock(fbad)
            try:
                txn_finish(L, pa, fa, "dev", "B", 0, 0)
                raise AssertionError("плохой блок принят")
            except SystemExit:
                pass
            assert os.path.exists(reject_path(e)) and not os.path.exists(fbad)
            h = json.dumps(L.d, sort_keys=True)
            try:
                txn_reserve(L, pa, "dev", "B", 0, 0)
                raise AssertionError("повторный резерв отвергнутой ячейки")
            except SystemExit:
                pass
            assert json.dumps(L.d, sort_keys=True) == h
            # предел checks машины применяется при резерве
            for arm, t, s_ in cells("B", "port"):
                for q in block_files(block_path("B", "port", arm, t, s_)):
                    os.remove(q)
            L2 = kcol.Ledger(os.path.join(td, "l2.json"), kpr.CAPS)
            pb2 = dict(pb, limits=dict(pb["limits"], checks=2 * BLOCK_CAP))
            g["contract_of"] = lambda plan_: plan_["contract_fp"]
            g["block_problems"] = lambda *aa, **kw: []
            txn_reserve(L2, pb2, "port", "T0", 0, 0)
            L2._crash_pending()
            txn_reserve(L2, pb2, "port", "F0", 0, 0)
            L2._crash_pending()
            try:
                txn_reserve(L2, pb2, "port", "BC", 0, 0)
                raise AssertionError("предел checks машины не сработал")
            except SystemExit:
                pass
            g["contract_of"] = g_contract
        finally:
            g.update(keep)


def selftest():
    import tempfile
    import k31_collect as kcol
    import k33_protocol as kpr
    # разбиение и очередь
    assert sorted(machine_starts("A") + machine_starts("B")) == \
        list(range(0, 300, 5))
    assert not set(machine_starts("A")) & set(machine_starts("B"))
    for m in MACHINES:
        q = sr_queue(m)
        assert len(q) == 10 * 6 * 30 and len(set(q)) == len(q)
        for t in range(10):
            for arm in kp.ORDER0:
                assert sum(1 for x in q if x[0] == t and x[1] == arm) == 30
    qa = sr_queue("A")
    assert [x[0] for x in qa[:180]] == [task_order()[0]] * 180
    assert [qa[30 * i][1] for i in range(6)] == arm_order(0) == \
        list(kp.ORDER0)
    assert arm_order(1)[0] == kp.ORDER0[1] and arm_order(7) == arm_order(1)
    assert task_order() == task_order() and sorted(task_order()) == \
        list(range(10))
    assert len(cells("A", "legacy_new")) == 12 and \
        len(cells("A", "dev")) == 60 and len(cells("A", "repeat")) == 12 and \
        cells("B", "dev") == [] and len(cells("B", "port")) == 6
    assert debug_for("T0", "dev") and not debug_for("T0", "sr") and \
        not debug_for("B", "dev") and not debug_for("BC", "legacy_new")
    lim = limits("A")
    assert lim["sr"] == 1800 * BLOCK_CAP and lim["checks"] == 331_700 and \
        lim["checks_new"] == (24 + 60 + 12) * BLOCK_CAP
    assert limits("B")["checks_new"] == 6 * BLOCK_CAP == \
        limits("B")["checks"]
    assert lim["checks_new"] + limits("B")["checks_new"] == 311_100
    assert wants_bsnap("B", "bank", 0) and not wants_bsnap("B", "bank", 1) \
        and not wants_bsnap("BC", "bank", 0)
    # ячейки плана
    plan = dict(machine="A", cells={ph: [list(x) for x in cells("A", ph)]
                                    for ph in PHASES},
                bank=dict(bank_id="abcdef0123456789"),
                queue_sr=[list(x) for x in sr_queue("A")], limits=limits("A"))
    assert schedule_problems(plan) == []
    for ph in ("legacy_new", "dev", "repeat"):        # удалённые ячейки
        bad = dict(plan, cells=dict(plan["cells"], **{ph: []}))
        assert schedule_problems(bad), ph
    assert schedule_problems(dict(plan, queue_sr=plan["queue_sr"][:-1]))
    assert schedule_problems(dict(plan, limits=dict(lim, sr=1)))
    ok = dict(arm="T0", task=task_order()[0], start=0, mode="bank", space=1,
              debug=False)
    assert cell_problems(plan, ok) == []
    assert cell_problems(plan, dict(ok, start=5))          # группа машины B
    assert cell_problems(plan, dict(ok, debug=True))       # debug в SR
    assert cell_problems(plan, dict(ok, space=0, start=0, debug=True)) == []
    assert cell_problems(plan, dict(ok, space=0, start=0, debug=False))
    assert cell_problems(plan, dict(arm="BC", task=8, start=30, mode="legacy",
                                    space=None, debug=False)) == []
    assert cell_problems(plan, dict(arm="BC", task=3, start=30,
                                    mode="legacy", space=None, debug=False))
    assert eid(plan, "sr", "B", 3, 10) == "sr/abcdef012345/B/t3/b10"
    assert eid(plan, "dev", "T1", 3, 0) == "checks/k33/dev/A/T1/t3/b0"
    # паритет банка
    cases = [dict(case_id=f"c{i}", init_hash=f"h{i}", init_hash_full=f"f{i}")
             for i in range(10)]
    eps = [dict(init_state_id=5 + i, init_hash=f"h{5 + i}",
                init_hash_full=f"f{5 + i}") for i in range(5)]
    assert bank_parity(1, 0, 5, eps, lambda s, t: (cases, {})) == []
    eps[2]["init_hash"] = "x"
    assert bank_parity(1, 0, 5, eps, lambda s, t: (cases, {}))
    # транзакция блока: резерв, результат до commit -> settle, лимит sr
    with tempfile.TemporaryDirectory() as td:
        g = globals()
        keep = dict(OUT=g["OUT"], block_problems=g["block_problems"],
                    ledger_for=g["ledger_for"], REJECTED=g["REJECTED"])
        L = kcol.Ledger(os.path.join(td, "l.json"), kpr.CAPS)
        try:
            g["OUT"] = td
            g["REJECTED"] = os.path.join(td, "rejected")
            g["block_problems"] = lambda *aa, **kw: []
            g["ledger_for"] = lambda plan_: L
            pl = dict(plan, limits=dict(sr=2 * BLOCK_CAP))
            e = txn_reserve(L, pl, "sr", "T0", 3, 0)
            f = block_path("A", "sr", "T0", 3, 0)
            os.makedirs(os.path.dirname(f))
            json.dump(dict(waiting_steps=10, episodes=[
                dict(env_steps=200)] * 5), open(f, "w"))
            L2 = kcol.Ledger(os.path.join(td, "l.json"), kpr.CAPS)
            assert txn_finish(L2, pl, "p", "sr", "T0", 3, 0) == 1050
            assert L2.is_done(e) and L2.spent("sr") == 1050
            assert txn_finish(L2, pl, "p", "sr", "T0", 3, 0) == 1050
            assert L2.spent("sr") == 1050                  # без двойного учёта
            try:
                txn_reserve(L2, pl, "sr", "T0", 3, 0)
                raise AssertionError("резерв при готовом результате")
            except SystemExit:
                pass
            txn_reserve(L2, pl, "sr", "T0", 3, 10)
            L2._crash_pending()                            # авария: 3050
            try:
                txn_reserve(L2, pl, "sr", "T0", 3, 20)     # 1050+3050+3050
                raise AssertionError("лимит машины не сработал")
            except SystemExit:
                pass
            # результат без резерва и без учёта — отказ
            f2 = block_path("A", "dev", "B", 0, 0)
            os.makedirs(os.path.dirname(f2))
            json.dump(dict(waiting_steps=10, episodes=[
                dict(env_steps=100)] * 5), open(f2, "w"))
            try:
                txn_finish(L2, pl, "p", "dev", "B", 0, 0)
                raise AssertionError("неучтённый результат принят")
            except SystemExit:
                pass
            # проверка не пройдена — карантин
            g["block_problems"] = lambda *aa, **kw: ["плохо"]
            L2.reserve("checks", eid(pl, "dev", "B", 0, 0), amount=BLOCK_CAP)
            try:
                txn_finish(L2, pl, "p", "dev", "B", 0, 0)
                raise AssertionError("плохой блок принят")
            except SystemExit:
                pass
            assert not os.path.exists(f2) and os.path.isdir(
                os.path.join(td, "quarantine"))
        finally:
            g.update(keep)
    gates_regression()
    # сверка пар блоков
    with tempfile.TemporaryDirectory() as td:
        def wr(name, A, succ):
            np.savez(os.path.join(td, name + ".npz"), actions=A)
            json.dump(dict(actions_npz=name + ".npz", episodes=[dict(
                init_hash="h", init_hash_full="f", action_sha1="a",
                env_steps=3, policy_calls=1, success=succ, success_step=1,
                done_step=1, own_steps=2)] * 5),
                open(os.path.join(td, name + ".json"), "w"))
            return os.path.join(td, name + ".json")
        A = np.zeros((3, 5, 7), np.float32)
        fa, fb = wr("a", A, True), wr("b", A.copy(), True)
        assert pair_problems(fa, fb, "x") == []
        A2 = A.copy()
        A2[1, 2, 3] = 1e-7
        assert pair_problems(fa, wr("c", A2, True), "x")
        assert pair_problems(fa, wr("d", A, False), "x")
    print("самопроверка k33_eval пройдена: разбиение групп A/B (по 30 на "
          "задачу, без пересечений), очередь (перестановка задач 33010, "
          "сдвиг рук, 1800 блоков на машину), ячейки и debug по фазам, eid, "
          "паритет хешей банка, транзакция блока (settle без двойного учёта, "
          "лимит машины, авария, неучтённый результат — отказ, карантин), "
          "побитовая сверка пар блоков; расписание из кода (удалённые "
          "ячейки, очередь, лимиты — отказ); гейты привязаны к контракту, "
          "банку, плану и SHA блоков (пустой, чужой, усечённый, изменённый "
          "— отказ); B — только при контракте A и гейтах A; перенос строгий "
          "(нет эталона A или расхождение — отказ); маркер отказа "
          "останавливает новый резерв; предел checks машины")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-33: вторая очередь")
    ap.add_argument("cmd", choices=("plan", "block", "cells", "legacy-compare",
                                    "dev-check", "repeat-compare", "gates",
                                    "forecast", "progress", "plan-path"))
    ap.add_argument("--machine", choices=tuple(MACHINES))
    ap.add_argument("--plan")
    ap.add_argument("--phase", choices=PHASES)
    ap.add_argument("--op", choices=("reserve", "finish", "status"))
    ap.add_argument("--arm", choices=tuple(kp.ARMS))
    ap.add_argument("--task", type=int)
    ap.add_argument("--start", type=int)
    a = ap.parse_args()
    if a.cmd == "plan-path":
        print(plan_path(a.machine))
        return 0
    try:
        return dict(plan=plan_cmd, block=block_cmd, cells=cells_cmd,
                    gates=gates_cmd, forecast=forecast, progress=progress,
                    **{"legacy-compare": legacy_compare,
                       "dev-check": dev_check,
                       "repeat-compare": repeat_compare})[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
