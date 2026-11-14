#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35 (K35-A): подготовка, панели, оракул, гейты, селекторы, планы,
транзакции блоков и решение после пилота.

prepare — reports/k35/prepare.json + reports/k35/panels.json (неизменяемы):
  источник K-34 T0 (parents_attn24 s0): best.pt и отчёт обучения K-34
  (SHA1 и SHA256), отпечаток весов, best_update = 3000, спецификация руки
  K-34 (k34_protocol.arm_spec, источник K-26), контракт кода K-34 из
  checkpoint, подготовка K-21, ревизия BAR, пределы, версии библиотек,
  таблица размещения; панели oracle-train / oracle-val / checkpoint-val.
plan --kind smoke|pilot — reports/k35/plan_<kind>.json:
  smoke — t0 init 20 (один B5-блок на руку): принудительные c = 0/17/35,
          N, A (селекторы init — нулевые выходы), S; декодер init (веса
          K-34 T0), отладочный снимок первого вызова;
  pilot — A/C/N/S × 10 задач × init 20–24, 40 блоков, 200 эпизодов;
          порядок задач — перестановка seed 35010, руки циклически
          сдвигаются; создаётся только при свежих пройденных гейтах.
block --op reserve|finish — reserve -> K-9h -> атомарный JSON+npz -> finish
  (проверка, расход 5·(env_steps + 10), commit/settle; отказ — карантин и
  маркер, повтор ячейки — только по решению). data/k35/ledger.json: checks
  20 000, pilot 122 000.
smoke-check — проверки блоков smoke + пересчёт трёх голов из весов на
  входах снимка (тот же GPU и fp16-autocast — гейт y/grip 1e−5, arm 1e−6;
  CPU FP32 — описательно), маршрут A из записанных X; паритет хешей.
oracle — шарды data/k35/oracle/<панель>_<задача>.npz (+ .json — запись
  завершённости), 36 прореженных forward на строку общего best K35-A;
  source_val — полный выход 24 источника K-34 на oracle-val.
oracle-gate — c_const по oracle-train, гейт потенциала (§10.3) ->
  reports/k35/oracle_gate.json: passed | stopped_low_oracle.
selectors — OOF, два селектора (CPU) -> data/k35/selectors.pt и отчёт;
selector-gate — гейт обучаемости (§11.5): passed | stopped_selector.
pilot-gate — успехи A/C/N/S из 50 и правило §14 -> pilot_signal |
  pilot_no_signal | technical_stop.
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

import k35_oracle as ko  # noqa: E402
import k35_policy as kp  # noqa: E402
import k35_topology as tp  # noqa: E402

OUT = "reports/k35"
PREPARE = "reports/k35/prepare.json"
LEDGER = "data/k35/ledger.json"
CAPS = dict(checks=20_000, pilot=122_000)
BLOCK_CAP = 5 * 610
REJECTED = "reports/k35/rejected"
TASK_SEED = 35010
SOURCE_BEST_UPDATE = 3000
KINDS = ("smoke", "pilot")
SCOPE = dict(smoke="checks", pilot="pilot")
SMOKE_CELLS = ((0, 20),)
SMOKE_CUTS = (0, 17, 35)
PILOT_BLOCKS = (20,)
PILOT_ARMS = ("k35_A", "k35_C", "k35_N", "k35_S")
SMOKE_ARMS = tuple(f"k35i_c{c}" for c in SMOKE_CUTS) + ("k35i_N", "k35i_A",
                                                         "k35_S")
PILOT_RULE = dict(c_vs_s=5, a_vs_c=1, a_vs_n=1)
DBG_TOL = kp.TOL
SELECTORS = "data/k35/selectors.pt"
SEL_REPORT = "reports/k35/selectors.json"
SEL_GATE = "reports/k35/selector_gate.json"
GATE_FILE = "reports/k35/pilot_gate.json"
CODE = ("k35_topology.py", "k35_model.py", "k35_train.py", "k35_oracle.py",
        "k35_selector.py", "k35_policy.py", "k35_harness.py",
        "k35_protocol.py", "k35_analyze.py", "k35_latency.py", "k35_run.sh",
        "k34_model.py", "k34_train.py", "k34_policy.py",
        "k9h_multiarm_gate.py", "k16_harness.py", "k16_protocol.py",
        "k15g_local_probe.py", "k15h_protocol.py", "k15f_policy.py",
        "k21_main_policy.py", "k21_policy.py", "k21_protocol.py",
        "k26_policy.py", "k26_model.py", "k26_attention.py", "k21_tree.py",
        "k21_model.py", "k21_data.py")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def save_json(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def load_json(path):
    return json.load(open(path, encoding="utf-8"))


def ledger():
    import k31_collect as kcol
    return kcol.Ledger(LEDGER, CAPS)


def code_shas(files=CODE):
    return {f: sha(os.path.join(HERE, f)) for f in files
            if os.path.exists(os.path.join(HERE, f))}


def plan_path(kind):
    return os.path.join(OUT, f"plan_{kind}.json")


# --- подготовка ----------------------------------------------------------------------
def libs():
    out = {}
    for m in ("torch", "numpy", "transformers"):
        try:
            out[m] = __import__(m).__version__
        except Exception as e:  # noqa: BLE001
            out[m] = f"нет ({type(e).__name__})"
    return out


def make_prepare():
    import torch
    import k21_data as d21
    import k34_protocol as pc34
    import k34_train as k34t
    prep = load_json(d21.REPORT)
    spec = pc34.arm_spec("pilot", "parents_attn24", 0, 24)
    if int(spec["best_update"]) != SOURCE_BEST_UPDATE:
        raise SystemExit(f"best main K-34 T0 = {spec['best_update']}, план "
                         f"K-35 закрепляет {SOURCE_BEST_UPDATE}")
    ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    pr = k34t.trained_ckpt_problems(ck, "parents_attn24", 0, spec)
    if pr:
        raise SystemExit("источник K-34: " + "; ".join(pr))
    files = dict(ckpt=spec["ckpt"], train_report=spec["train_report"],
                 k34_prepare=os.path.abspath(pc34.PREPARE),
                 k26_ckpt=spec["source"]["ckpt"])
    return dict(
        kind="k35_prepare", prep_id=prep["prep_id"],
        bar_revision=prep["bar_pin"]["revision"],
        bounds=prep["bounds_codec"],
        source=dict(ckpt=spec["ckpt"], ckpt_sha1=spec["ckpt_sha1"],
                    weights_hash=spec["weights_hash"],
                    best_update=int(spec["best_update"]),
                    train_report=spec["train_report"],
                    train_report_sha1=spec["train_report_sha1"],
                    k34_prepare_sha1=sha(pc34.PREPARE),
                    k34_code=ck["config"]["code"],
                    architecture=ck["config"]["architecture"], spec=spec),
        placement={k: dict(path=v, sha256=sha256(v)) for k, v in
                   files.items()},
        dtype="float16", libraries=libs(),
        topology=dict(pairs=list(tp.PAIRS), n_cuts=tp.N_CUTS,
                      active=tp.ACTIVE, router=tp.ROUTER_ACTIVE))


def prepare(a):
    import k21_data as d21
    new = make_prepare()
    if os.path.exists(PREPARE):
        old = load_json(PREPARE)
        diff = sorted(k for k in set(old) | set(new) if k != "created" and
                      old.get(k) != new.get(k))
        if diff:
            raise SystemExit(f"{PREPARE} закреплён иначе: {diff}")
        print(f"  {PREPARE}: закреплён, источник совпадает")
    else:
        save_json(PREPARE, dict(new, created=datetime.datetime.now()
                                .isoformat(timespec="seconds")))
        print(f"  {PREPARE}: записан")
    D = d21.load_prepared(require_teacher=False)
    pan = ko.make_panels(D)
    if os.path.exists(ko.PANELS):
        if load_json(ko.PANELS) != json.loads(json.dumps(pan)):
            raise SystemExit(f"{ko.PANELS} закреплён с другими строками")
        print(f"  {ko.PANELS}: закреплены, строки совпадают")
    else:
        save_json(ko.PANELS, pan)
        print(f"  {ko.PANELS}: записаны")
    s = new["source"]
    print(f"    источник K-34 T0: best.pt {s['ckpt_sha1'][:12]}, best main "
          f"{s['best_update']}, веса {s['weights_hash'][:12]}")
    for nm in ("oracle_train", "oracle_val", "checkpoint_val"):
        q = pan[nm]
        print(f"    {nm}: строк {q['n_rows']}, задач {q['n_tasks']}, "
              f"эпизодов {q['n_episodes']}")
    return 0


def prepare_problems():
    if not os.path.exists(PREPARE) or not os.path.exists(ko.PANELS):
        return ["нет подготовки или панелей"]
    p = load_json(PREPARE)
    out = []
    for k, v in p["placement"].items():
        if not os.path.exists(v["path"]) or sha256(v["path"]) != v["sha256"]:
            out.append(f"размещение {k}: файл изменён или отсутствует")
    return out


# --- оракул ---------------------------------------------------------------------------
ORACLE_PANELS = ("oracle_train", "oracle_val")


def shard_name(panel, ti):
    return os.path.join(ko.ORACLE_DIR, f"{panel}_t{ti:02d}")


def shard_list():
    pan = load_json(ko.PANELS)
    out = []
    for panel in ORACLE_PANELS:
        for ti, t in enumerate(sorted(pan[panel]["by_task"])):
            out.append((panel, ti, t, pan[panel]["by_task"][t]))
    return out


def decoder_binding():
    import k35_train as k35t
    r = load_json(k35t.REPORT)
    return dict(decoder_sha1=r["best_sha1"],
                decoder_weights_hash=r["best_weights_hash"],
                train_report_sha1=sha(k35t.REPORT),
                panels_sha1=sha(ko.PANELS), code=k35t.code_now(),
                prepare_sha1=sha(PREPARE))


def shard_problems(base, panel, task, rows, bind):
    if not os.path.exists(base + ".json") or not os.path.exists(base +
                                                                ".npz"):
        return ["нет шарда"]
    m = load_json(base + ".json")
    p = []
    want = dict(panel=panel, task=task, rows=[int(r) for r in rows],
                rows_sha1=ko.sha_arr(rows), npz_sha1=sha(base + ".npz"),
                cuts=list(range(36)), **bind)
    for k, v in want.items():
        if m.get(k) != v:
            p.append(f"шард {os.path.basename(base)}: {k} ≠ ожидаемому")
    return p


def oracle_cmd(a):
    import torch
    import k21_data as d21
    import k21_model as km
    import k35_train as k35t
    pr = prepare_problems() + k35t.train_problems(k35t.micro_now(), "float16")
    if pr:
        raise SystemExit("; ".join(pr))
    bind = decoder_binding()
    todo = [(panel, ti, t, rows) for panel, ti, t, rows in shard_list()
            if shard_problems(shard_name(panel, ti), panel, t, rows, bind)]
    src_base = os.path.join(ko.ORACLE_DIR, "source_val")
    pan = load_json(ko.PANELS)
    vrows = pan["oracle_val"]["rows"]
    need_src = bool(shard_problems(src_base, "source_val", "all", vrows,
                                   bind))
    print(f"  оракул: шардов к расчёту {len(todo)} из "
          f"{len(shard_list())}; источник S на oracle-val: "
          f"{'нужен' if need_src else 'готов'}", flush=True)
    if not todo and not need_src:
        return 0
    D = d21.load_prepared(require_teacher=False)
    dev = torch.device(a.device)
    dt = torch.float16
    os.makedirs(ko.ORACLE_DIR, exist_ok=True)
    if todo:
        mod, bar, _ck = k35t.load_trained(torch, a.device)
        for j, (panel, ti, t, rows) in enumerate(todo):
            base = shard_name(panel, ti)
            for f in (base + ".json",):
                if os.path.exists(f):
                    os.remove(f)                 # чужая запись завершённости
            t0 = time.time()
            r = ko.oracle_rows(torch, km, mod, bar, D, np.asarray(rows), dev,
                               dt, a.micro, d21.make_batch)
            np.savez(base + ".tmp.npz", rows=np.asarray(rows),
                     task=np.asarray([str(D["task"][x]) for x in rows]),
                     episode=np.asarray([int(D["episode"][x]) for x in rows]),
                     **{f"C_{k}": v for k, v in r["C"].items()},
                     roots=r["roots"], iv_root=r["iv_root"], mids=r["mids"],
                     iv_mid=r["iv_mid"])
            os.replace(base + ".tmp.npz", base + ".npz")
            save_json(base + ".json", dict(
                panel=panel, task=t, rows=[int(x) for x in rows],
                rows_sha1=ko.sha_arr(rows), npz_sha1=sha(base + ".npz"),
                cuts=list(range(36)), feat_dtype="float16 autocast -> "
                "float32 (norm после слоя 8 / 16)", micro=a.micro, **bind))
            el = time.time() - t0
            if j == 0:
                sz = os.path.getsize(base + ".npz")
                print(f"  оракул: первый шард {len(rows)} строк за {el:.0f} с"
                      f" ({36 * len(rows) / max(el, 1e-9):.1f} forward/с), "
                      f"{sz / 2 ** 20:.1f} МБ; прогноз ~"
                      f"{el * len(todo) / 3600:.1f} ч, диск ~"
                      f"{sz * len(todo) / 2 ** 30:.2f} ГБ; память "
                      f"{torch.cuda.max_memory_allocated(dev) / 2 ** 30:.1f}"
                      f" ГБ", flush=True)
            print(f"    шард {os.path.basename(base)}: {len(rows)} строк, "
                  f"{el:.0f} с", flush=True)
        del mod, bar
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if need_src:
        import k34_policy as kp34
        spec = load_json(PREPARE)["source"]["spec"]
        src, bar, _c, _p = kp34.load_trained(torch, "parents_attn24", 0, 24,
                                             spec, a.device)
        r = ko.source_costs(torch, km, src, bar, D, np.asarray(vrows), dev,
                            dt, a.micro, d21.make_batch)
        np.savez(src_base + ".tmp.npz", rows=np.asarray(vrows),
                 **{f"S_{k}": v for k, v in r.items()})
        os.replace(src_base + ".tmp.npz", src_base + ".npz")
        save_json(src_base + ".json", dict(
            panel="source_val", task="all", rows=[int(x) for x in vrows],
            rows_sha1=ko.sha_arr(vrows), npz_sha1=sha(src_base + ".npz"),
            cuts=list(range(36)), source_ckpt_sha1=spec["ckpt_sha1"],
            **bind))
        print(f"  источник S на oracle-val: {len(vrows)} строк")
    return 0


def load_oracle(panel):
    """Шарды панели в порядке панели -> O (локальные индексы)."""
    parts = [np.load(shard_name(p, ti) + ".npz")
             for p, ti, _t, _r in shard_list() if p == panel]
    cat = {k: np.concatenate([z[k] for z in parts]) for k in parts[0].files}
    return dict(C={k: cat[f"C_{k}"] for k in ("full", "arm20", "arm8",
                                               "grip")},
                roots=cat["roots"], iv_root=cat["iv_root"], mids=cat["mids"],
                iv_mid=cat["iv_mid"], rows=cat["rows"], task=cat["task"],
                episode=cat["episode"])


def oracle_fresh_problems():
    p = prepare_problems()
    if p:
        return p
    bind = decoder_binding()
    for panel, ti, t, rows in shard_list():
        p += shard_problems(shard_name(panel, ti), panel, t, rows, bind)
    vrows = load_json(ko.PANELS)["oracle_val"]["rows"]
    p += shard_problems(os.path.join(ko.ORACLE_DIR, "source_val"),
                        "source_val", "all", vrows, bind)
    return p


def oracle_shas():
    out = {}
    for panel, ti, _t, _r in shard_list():
        b = shard_name(panel, ti)
        out[b] = sha(b + ".npz")
    b = os.path.join(ko.ORACLE_DIR, "source_val")
    out[b] = sha(b + ".npz")
    return out


def oracle_gate_compute():
    Otr, Ova = load_oracle("oracle_train"), load_oracle("oracle_val")
    ntr, nva = len(Otr["rows"]), len(Ova["rows"])
    c_const, m_tr = ko.c_const_of(Otr["C"]["full"], np.arange(ntr),
                                  Otr["task"])
    S = np.load(os.path.join(ko.ORACLE_DIR, "source_val.npz"))
    if not np.array_equal(S["rows"], Ova["rows"]):
        raise SystemExit("строки источника S ≠ oracle-val")
    L_S = ko.task_mean(S["S_full"], np.arange(nva), Ova["task"])
    met, pr = ko.gate_problems(Ova["C"], np.arange(nva), Ova["task"],
                               c_const, L_S)
    per_task = {}
    for t in sorted(set(Ova["task"].tolist())):
        i = np.flatnonzero(Ova["task"] == t)
        f = Ova["C"]["full"][i]
        per_task[t] = dict(L_C=float(f[:, c_const].mean()),
                           L_O=float(f.min(1).mean()),
                           L_N=float(f.mean(1).mean()),
                           L_S=float(S["S_full"][i].mean()), n=int(len(i)))
    met.update(train_cut_means=m_tr.tolist(), per_task=per_task,
               n_train=ntr, n_val=nva)
    return met, pr, c_const


def oracle_gate(a):
    p = oracle_fresh_problems()
    if p:
        save_json(ko.GATE, dict(status="technical_stop", problems=p[:30]))
        print("  гейт оракула: ТЕХНИЧЕСКИЙ СТОП — " + "; ".join(p[:4]))
        return 3
    met, pr, c_const = oracle_gate_compute()
    st = "passed" if not pr else "stopped_low_oracle"
    save_json(ko.GATE, dict(status=st, problems=pr, metrics=met,
                            c_const=c_const, rule=ko.GATE_RULE,
                            shards=oracle_shas(), binding=decoder_binding()))
    print(f"  гейт оракула: c_const {c_const}; L_C {met['L_C']:.4f}, L_O "
          f"{met['L_O']:.4f}, L_N {met['L_N']:.4f}, L_S {met['L_S']:.4f}; "
          f"gap_full {met['gap_full']:.3f}, arm-8 {met['arm8_gain']:.3f} — "
          f"{st}" + ("" if not pr else ": " + "; ".join(pr)))
    return 0


def oracle_gate_problems():
    """Гейт оракула пройден и свеж (шарды, декодер, код) — пересчёт."""
    if not os.path.exists(ko.GATE):
        return ["нет гейта оракула"]
    g = load_json(ko.GATE)
    p = oracle_fresh_problems()
    if p:
        return p
    if g.get("shards") != oracle_shas() or g.get("binding") != \
            decoder_binding():
        return ["гейт оракула: шарды или декодер изменены"]
    met, pr, c_const = oracle_gate_compute()
    st = "passed" if not pr else "stopped_low_oracle"
    if g.get("status") != st or g.get("c_const") != c_const:
        return ["гейт оракула ≠ пересчёту"]
    return [] if st == "passed" else [f"гейт оракула: {st}"]


# --- селекторы ---------------------------------------------------------------------------
def selectors_cmd(a):
    import torch
    import k35_selector as ks
    if os.path.exists(SELECTORS):
        raise SystemExit(f"{SELECTORS} уже обучен (неизменяем); повтор — "
                         f"только по решению")
    p = oracle_gate_problems()
    if p:
        raise SystemExit("; ".join(p))
    torch.manual_seed(0)
    Otr, Ova = load_oracle("oracle_train"), load_oracle("oracle_val")
    ntr, nva = len(Otr["rows"]), len(Ova["rows"])
    task_all = np.concatenate([Otr["task"], Ova["task"]])
    rows_tr, rows_va = np.arange(ntr), ntr + np.arange(nva)
    S = ks.train_selectors(torch, Otr, Ova, rows_tr, rows_va,
                           Otr["episode"], task_all)
    g = load_json(ko.GATE)
    meta = dict(oracle_gate_sha1=sha(ko.GATE), c_const=g["c_const"],
                selector_code=sha(os.path.join(HERE, "k35_selector.py")),
                scales=S["scales"])
    h = ks.save_bundle(torch, S, SELECTORS, meta)
    rep = dict(bundle_hash=h, bundle_sha1=sha(SELECTORS), meta=meta,
               folds=S["fold_report"], root_history=S["root_history"],
               b_oof_freq=np.bincount(S["b_oof"].reshape(-1),
                                      minlength=6).tolist())
    save_json(SEL_REPORT, rep)
    print(f"  селекторы: набор {h[:12]}, best шаг корней "
          f"{S['root_history']['best_update']}")
    return 0


def selector_gate_compute():
    import torch
    import k35_selector as ks
    B = ks.load_bundle(torch, SELECTORS)
    nets = ks.nets_of(torch, B)
    S = dict(Tr=B["Tr"], **nets)
    Ova = load_oracle("oracle_val")
    nva = len(Ova["rows"])
    a, b, c = ks.choose(torch, S, Ova, np.arange(nva))
    g = load_json(ko.GATE)
    met, pr = ks.selector_gate(Ova["C"]["full"], np.arange(nva), Ova["task"],
                               c, g["c_const"], Ova["C"]["arm8"])
    Cv = Ova["C"]["full"]
    o = np.argmin(Cv, 1)
    oa, ob = o // 6, o % 6
    best_b_given_a = np.argmin(Cv.reshape(nva, 6, 6)[np.arange(nva), a], 1)
    met.update(first_decision_err=float(np.mean(a != oa)),
               second_decision_err_given_a=float(np.mean(b != best_b_given_a)),
               per_task={t: dict(
                   L_A=float(Cv[np.flatnonzero(Ova["task"] == t),
                                c[Ova["task"] == t]].mean()),
                   L_C=float(Cv[Ova["task"] == t, g["c_const"]].mean()),
                   L_O=float(Cv[Ova["task"] == t].min(1).mean()))
                   for t in sorted(set(Ova["task"].tolist()))})
    return met, pr, B["bundle_hash"]


def selector_gate(a):
    p = oracle_gate_problems()
    if not os.path.exists(SELECTORS) or not os.path.exists(SEL_REPORT):
        p.append("нет селекторов")
    elif load_json(SEL_REPORT).get("bundle_sha1") != sha(SELECTORS):
        p.append("файл селекторов ≠ отчёту")
    elif load_json(SEL_REPORT)["meta"].get("oracle_gate_sha1") != \
            sha(ko.GATE):
        p.append("селекторы обучены при другом гейте оракула")
    if p:
        save_json(SEL_GATE, dict(status="technical_stop", problems=p[:30]))
        print("  гейт селекторов: ТЕХНИЧЕСКИЙ СТОП — " + "; ".join(p[:4]))
        return 3
    met, pr, h = selector_gate_compute()
    st = "passed" if not pr else "stopped_selector"
    save_json(SEL_GATE, dict(status=st, problems=pr, metrics=met,
                             bundle_hash=h, bundle_sha1=sha(SELECTORS),
                             oracle_gate_sha1=sha(ko.GATE),
                             selector_code=sha(os.path.join(
                                 HERE, "k35_selector.py"))))
    print(f"  гейт селекторов: L_A {met['L_A']:.4f}, L_C {met['L_C']:.4f}, "
          f"L_O {met['L_O']:.4f}, L_N {met['L_N']:.4f}; capture "
          f"{met['capture']:.3f}; arm-8 A {met['arm8_A']:.4f} / C "
          f"{met['arm8_C']:.4f} — {st}" + ("" if not pr else ": " +
                                            "; ".join(pr)))
    return 0


def selector_gate_problems():
    p = oracle_gate_problems()
    if p:
        return p
    if not os.path.exists(SEL_GATE) or not os.path.exists(SELECTORS):
        return ["нет гейта селекторов"]
    g = load_json(SEL_GATE)
    if g.get("bundle_sha1") != sha(SELECTORS) or \
            g.get("oracle_gate_sha1") != sha(ko.GATE) or \
            g.get("selector_code") != sha(os.path.join(HERE,
                                                       "k35_selector.py")):
        return ["гейт селекторов: привязки изменены"]
    met, pr, h = selector_gate_compute()
    st = "passed" if not pr else "stopped_selector"
    if g.get("status") != st or g.get("bundle_hash") != h:
        return ["гейт селекторов ≠ пересчёту"]
    return [] if st == "passed" else [f"гейт селекторов: {st}"]


def offline_problems():
    return selector_gate_problems()


# --- планы ---------------------------------------------------------------------------------
def task_order():
    return [int(t) for t in np.random.default_rng(TASK_SEED).permutation(10)]


def arms_for(kind):
    return list(SMOKE_ARMS if kind == "smoke" else PILOT_ARMS)


def cells(kind):
    arms = arms_for(kind)
    if kind == "smoke":
        return [(lab, t, b) for t, b in SMOKE_CELLS for lab in arms]
    out = []
    for pos, t in enumerate(task_order()):
        k = pos % len(arms)
        rot = arms[k:] + arms[:k]
        for lab in rot:
            out += [(lab, t, b) for b in PILOT_BLOCKS]
    return out


def s_spec():
    src = load_json(PREPARE)["source"]
    return dict(kind="S", k34=src["spec"], ckpt_sha1=src["ckpt_sha1"])


def arm_spec(kind, lab):
    import torch
    import k35_selector as ks
    import k35_train as k35t
    if lab == "k35_S":
        return s_spec()
    src = load_json(PREPARE)["source"]
    if kind == "smoke":
        ctrl = "N" if lab == "k35i_N" else "A" if lab == "k35i_A" else "C"
        c = int(lab[len("k35i_c"):]) if ctrl == "C" else None
        return dict(kind="K35", controller=ctrl, c_const=c, decoder="init",
                    decoder_weights_hash=src["weights_hash"],
                    source_ckpt=src["ckpt"], decoder_ckpt=None,
                    decoder_sha1=None, selectors="init",
                    selectors_hash=None)
    r = load_json(k35t.REPORT)
    g = load_json(ko.GATE)
    B = ks.load_bundle(torch, SELECTORS)
    return dict(kind="K35", controller=lab[-1],
                c_const=g["c_const"] if lab == "k35_C" else None,
                decoder="trained",
                decoder_ckpt=os.path.abspath(os.path.join(k35t.RUN_DIR,
                                                          "best.pt")),
                decoder_sha1=r["best_sha1"],
                decoder_weights_hash=r["best_weights_hash"],
                source_ckpt=src["ckpt"], selectors="trained",
                selectors_path=os.path.abspath(SELECTORS),
                selectors_sha1=sha(SELECTORS),
                selectors_hash=B["bundle_hash"])


def init_selectors_hash():
    import torch
    import k35_selector as ks
    H = None
    sp = load_json(PREPARE)["source"]["spec"]
    ck = torch.load(sp["ckpt"], map_location="cpu", weights_only=False)
    for k, v in ck["weights"].items():
        if k == "q0" or k.endswith(".q0"):
            H = int(v.shape[-1])
    if H is None:
        raise SystemExit("не найдена ширина H в checkpoint K-34")
    return ks.init_bundle(torch, H)["bundle_hash"]


def make_plan(kind, device):
    import k15g_local_probe as pr
    import k21_data as d21
    import k35_train as k35t
    if kind not in KINDS:
        raise SystemExit(f"вид плана {kind}")
    p = prepare_problems()
    if kind == "pilot":
        p += offline_problems()
    if p:
        raise SystemExit("план: " + "; ".join(p[:4]))
    prep = load_json(d21.REPORT)
    snap = d21.check_pin(prep["bar_pin"], verify_content=True)
    arms = {lab: dict(spec=arm_spec(kind, lab)) for lab in arms_for(kind)}
    if kind == "smoke":
        h = init_selectors_hash()
        for lab, arm in arms.items():
            if arm["spec"]["kind"] == "K35":
                arm["spec"]["selectors_hash"] = h
    binding = dict(prepare_sha1=sha(PREPARE), panels_sha1=sha(ko.PANELS))
    if kind == "pilot":
        binding.update(train_report_sha1=sha(k35t.REPORT),
                       oracle_gate_sha1=sha(ko.GATE),
                       selector_gate_sha1=sha(SEL_GATE),
                       selectors_sha1=sha(SELECTORS))
    return dict(kind="k35_plan", plan_kind=kind,
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                arms=arms, cells=[list(c) for c in cells(kind)],
                debug=kind == "smoke", scope=SCOPE[kind],
                setup=dict(pr.expected_setup(), ckpt=snap),
                bar_revision=prep["bar_pin"]["revision"],
                prep_id=prep["prep_id"], bounds=prep["bounds_codec"],
                binding=binding, task_order=task_order(), n_envs=5,
                env_seed=101, rule=PILOT_RULE, code=code_shas(),
                device=str(device))


def plan_cmd(a):
    p = plan_path(a.kind)
    if os.path.exists(p):
        old = load_json(p)
        pr = plan_problems(old)
        if pr:
            raise SystemExit(f"{p}: закреплённый план не сходится — "
                             + "; ".join(pr[:3]))
        print(f"  план {p}: закреплён")
        return 0
    save_json(p, make_plan(a.kind, a.device))
    print(f"  план {p}: {len(cells(a.kind))} блоков, руки {arms_for(a.kind)}")
    return 0


def plan_problems(plan, offline=True):
    p = []
    if plan.get("kind") != "k35_plan":
        return ["не план K-35"]
    k = plan["plan_kind"]
    if plan["cells"] != [list(c) for c in cells(k)]:
        p.append("ячейки ≠ расписанию")
    b = plan.get("binding") or {}
    for key, f in (("prepare_sha1", PREPARE), ("panels_sha1", ko.PANELS)):
        if not os.path.exists(f) or sha(f) != b.get(key):
            p.append(f"{f} изменён после плана")
    for f, s in plan["code"].items():
        q = os.path.join(HERE, f)
        if not os.path.exists(q) or sha(q) != s:
            p.append(f"код {f} изменён после плана")
    p += prepare_problems()
    for lab, arm in plan["arms"].items():
        sp = arm["spec"]
        if sp.get("decoder") == "trained" and (
                not os.path.exists(sp["decoder_ckpt"]) or
                sha(sp["decoder_ckpt"]) != sp["decoder_sha1"]):
            p.append(f"{lab}: best.pt K-35 изменён или отсутствует")
        if sp.get("selectors") == "trained" and (
                not os.path.exists(sp["selectors_path"]) or
                sha(sp["selectors_path"]) != sp["selectors_sha1"]):
            p.append(f"{lab}: селекторы изменены или отсутствуют")
    if k == "pilot" and offline:
        import k35_train as k35t
        for key, f in (("train_report_sha1", k35t.REPORT),
                       ("oracle_gate_sha1", ko.GATE),
                       ("selector_gate_sha1", SEL_GATE),
                       ("selectors_sha1", SELECTORS)):
            if not os.path.exists(f) or sha(f) != b.get(key):
                p.append(f"{f} изменён после плана")
        if not p:
            p += offline_problems()
    return p


# --- транзакции ------------------------------------------------------------------------------
def block_path(kind, lab, task, start):
    return os.path.join(OUT, kind, lab, f"t{task}_i{start}.json")


def eid(kind, lab, task, start):
    return f"{SCOPE[kind]}/k35/{kind}/{lab}/t{task}/b{start}"


def reject_path(e):
    return os.path.join(REJECTED, e.replace("/", "__") + ".json")


def quarantine(path):
    q = os.path.join(OUT, "quarantine", time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(q, exist_ok=True)
    for f in (path, path[:-5] + ".actions.npz"):
        if os.path.exists(f):
            shutil.move(f, os.path.join(q, os.path.basename(
                os.path.dirname(f)) + "_" + os.path.basename(f)))
    return q


def block_actual(path):
    d = load_json(path)
    eps = d["episodes"]
    steps = int(eps[0]["env_steps"])
    if len(eps) != 5 or any(int(e["env_steps"]) != steps for e in eps):
        raise SystemExit(f"{path}: эпизодов/env_steps")
    return 5 * (steps + int(d.get("waiting_steps", 10)))


def ctx_of(plan, task, start):
    return dict(task=int(task), inits=[int(start) + i for i in
                                       range(int(plan["n_envs"]))],
                env_seed=int(plan["env_seed"]))


def block_problems(plan, plan_file, kind, lab, task, start, path=None):
    import k15g_local_probe as pr
    import k15h_protocol as hpc
    import k16_protocol as p16
    import k21_protocol as pp
    path = path or block_path(kind, lab, task, start)
    if not os.path.exists(path):
        return [f"нет блока {path}"]
    p = pr.validate_block(path, lab, int(task), int(start),
                          require_single=True, setup=plan["setup"])
    if p:
        return p
    d, z = p16.load(path)
    if os.path.basename(os.path.normpath(str(d.get("ckpt")))) != \
            plan["bar_revision"]:
        p.append(f"{lab}: снимок BAR блока не закреплённой ревизии")
    sp = plan["arms"][lab]["spec"]
    j = d.get("joint") or {}
    ctx = ctx_of(plan, task, start)
    if sp["kind"] == "S":
        want = dict(label=lab, k35_kind="S", k34_kind="K34", k34_depth=24,
                    k34_ckpt_sha1=sp["k34"]["ckpt_sha1"],
                    k35_plan_sha1=sha(plan_file))
    else:
        want = dict(label=lab, k35_kind="K35",
                    k35_controller=sp["controller"],
                    k35_c_const=sp.get("c_const"),
                    k35_decoder=sp["decoder"],
                    k35_decoder_weights_hash=sp["decoder_weights_hash"],
                    k35_selectors_hash=sp["selectors_hash"],
                    k35_task=ctx["task"], k35_inits=ctx["inits"],
                    k35_env_seed=ctx["env_seed"],
                    k35_debug=bool(plan["debug"]),
                    k35_plan_sha1=sha(plan_file))
    p += [f"{lab}: {k} = {j.get(k)!r}, ожидалось {v!r}"
          for k, v in want.items() if j.get(k) != v]
    p += pp.journal_problems(lab, z, plan["bounds"])
    p += kp.journal_problems(lab, z, sp, ctx, plan["bounds"],
                             debug=bool(plan["debug"]) and sp["kind"] != "S")
    if not p:
        p += hpc.actions_problems(lab, z, hpc.harness_scale())
    return p


def block_cmd(a):
    plan = load_json(a.plan)
    kind = plan["plan_kind"]
    pr_ = plan_problems(plan)
    if pr_:
        raise SystemExit("план: " + "; ".join(pr_[:3]))
    if [a.label, a.task, a.start] not in plan["cells"]:
        raise SystemExit(f"{a.label} t{a.task} i{a.start} не из плана")
    e = eid(kind, a.label, a.task, a.start)
    path = block_path(kind, a.label, a.task, a.start)
    L = ledger()
    if os.path.exists(reject_path(e)):
        raise SystemExit(f"{e}: результат был отвергнут — технический стоп")
    if a.op == "reserve":
        if os.path.exists(path):
            raise SystemExit(f"{path}: результат уже есть — нужен finish")
        if L.is_done(e):
            raise SystemExit(f"{e}: учтён, но результата нет")
        L.reserve(SCOPE[kind], e, amount=BLOCK_CAP)
        print(f"    резерв {e}")
        return 0
    pr_ = block_problems(plan, a.plan, kind, a.label, a.task, a.start, path)
    if pr_:
        q = quarantine(path) if os.path.exists(path) else None
        save_json(reject_path(e), dict(eid=e, reasons=pr_, quarantine=q))
        raise SystemExit(f"{a.label} t{a.task} i{a.start}: " +
                         "; ".join(pr_[:4]) + f" -> карантин {q}")
    act = block_actual(path)
    done = L.d["done"].get(e)
    pd = L.d["pending"]
    if done is None and not (pd and pd.get("id") == e):
        raise SystemExit(f"{e}: результат есть, но не учтён и не "
                         f"зарезервирован")
    L.settle(e, act)
    sc = SCOPE[kind]
    print(f"    {a.label} t{a.task} i{a.start}: проверен, шагов {act}; {sc} "
          f"{L.spent(sc)}/{CAPS[sc]}")
    return 0


def cells_cmd(a):
    plan = load_json(a.plan)
    pr_ = plan_problems(plan)
    if pr_:
        raise SystemExit("план: " + "; ".join(pr_[:3]))
    print(f"#{len(plan['cells'])}")
    for lab, t, b in plan["cells"]:
        print(f"{lab}:{t}:{b}:{int(plan['debug'])}")
    return 0


def parity_problems(plan, kind):
    h = {}
    for lab, t, b in plan["cells"]:
        f = block_path(kind, lab, t, b)
        if not os.path.exists(f):
            continue
        for e in load_json(f)["episodes"]:
            h.setdefault((t, int(e["init_state_id"])), set()).add(
                (e["init_hash"], e["init_hash_full"]))
    return [f"t{k[0]} init {k[1]}: начальные хеши рук различаются"
            for k, v in h.items() if len(v) != 1]


# --- smoke ----------------------------------------------------------------------------------
def smoke_check(a):
    import torch
    import k16_protocol as p16
    import k35_selector as ks
    plan = load_json(a.plan)
    if plan["plan_kind"] != "smoke":
        raise SystemExit("нужен план smoke")
    prob, rep, sizes = [], {}, []
    src = load_json(PREPARE)["source"]
    ck = torch.load(src["ckpt"], map_location="cpu", weights_only=False)
    L, U = np.asarray(plan["bounds"]["L"]), np.asarray(plan["bounds"]["U"])
    for lab, t, b in plan["cells"]:
        f = block_path("smoke", lab, t, b)
        pr_ = block_problems(plan, a.plan, "smoke", lab, t, b, f)
        prob += pr_
        if pr_:
            continue
        sizes += [os.path.getsize(q) for q in (f, f[:-5] + ".actions.npz")]
        sp = plan["arms"][lab]["spec"]
        if sp["kind"] == "S":
            rep[lab] = dict(kind="S")
            continue
        _d, z = p16.load(f)
        same = kp.head_recompute(torch, z, ck["weights"], L, U,
                                 device=a.device,
                                 autocast=a.device.startswith("cuda"))
        cpu = kp.head_recompute(torch, z, ck["weights"], L, U)
        bad = {k: v for k, v in same.items() if not (v <= DBG_TOL[k])}
        if set(same) != set(DBG_TOL):
            bad["поля"] = sorted(same)
        if bad:
            prob.append(f"{lab}: пересчёт голов на том же устройстве — {bad}")
        H = int(np.asarray(z["k21_dbg_x_root"]).shape[-1])
        route = kp.route_recompute(torch, z, ks.init_bundle(torch, H),
                                   device=a.device)
        if not all(route[k] <= kp.ROUTE_TOL for k in ("net_root",
                                                       "net_mid")):
            prob.append(f"{lab}: маршрут из записанных X ≠ scores {route}")
        c = np.asarray(z["k21_c"])
        rep[lab] = dict(same_device=same, cpu_fp32=cpu, route=route,
                        cuts=np.bincount(c.reshape(-1), minlength=36)
                        .tolist(), calls=int(c.shape[0]),
                        distinct_routes_first_call=int(len(set(
                            c[0].tolist()))))
    prob += parity_problems(plan, "smoke")
    if "k35i_N" in rep and rep["k35i_N"].get("distinct_routes_first_call",
                                             0) < 2:
        prob.append("N: на первом вызове все элементы B5 получили один срез")
    per_block = sum(sizes) / max(1, len(sizes) // 2)
    rep.update(ok=not prob, problems=prob, bytes_per_block=per_block,
               disk_forecast_gb=dict(pilot=per_block * len(cells("pilot")) /
                                     2 ** 30),
               tolerances=DBG_TOL, plan_sha1=sha(a.plan),
               note="гейт — пересчёт голов под тем же fp16-autocast "
                    "харнесса; CPU float32 — описательно; руки smoke — "
                    "init (декодер K-34 T0, селекторы с нулевым выходом)")
    save_json(os.path.join(OUT, "smoke_check.json"), rep)
    for lab in [c[0] for c in plan["cells"]]:
        if lab in rep and "same_device" in rep[lab]:
            print(f"  smoke {lab}: головы (то же устройство) "
                  f"{rep[lab]['same_device']}; CPU {rep[lab]['cpu_fp32']}")
    print(f"  smoke: блок ≈ {per_block / 2 ** 20:.2f} МБ; диск пилот "
          f"{rep['disk_forecast_gb']['pilot']:.2f} ГБ — " +
          ("пройден" if not prob else "ОТКАЗ: " + "; ".join(prob[:3])))
    return 0 if not prob else 3


def smoke_verify(a):
    f = os.path.join(OUT, "smoke_check.json")
    p = []
    if not os.path.exists(f):
        p.append("нет smoke_check.json")
    else:
        r = load_json(f)
        if not r.get("ok"):
            p.append("smoke не пройден")
        pf = plan_path("smoke")
        if not os.path.exists(pf) or r.get("plan_sha1") != sha(pf):
            p.append("smoke другого плана")
        else:
            p += plan_problems(load_json(pf))
    print("  smoke: " + ("пройден, план свеж" if not p else
                         "ОТКАЗ — " + "; ".join(p[:3])))
    return 0 if not p else 3


# --- пилот -------------------------------------------------------------------------------------
def sr_of(kind, plan, lab):
    succ = []
    for l_, t, b in plan["cells"]:
        if l_ == lab:
            d = load_json(block_path(kind, lab, t, b))
            succ += [bool(e["success"]) for e in d["episodes"]]
    return int(sum(succ)), len(succ)


def pilot_decision(s):
    """s: {A, C, N, S} -> успехи из 50 -> (статус, детали)."""
    c1 = s["C"] >= s["S"] - PILOT_RULE["c_vs_s"]
    c2 = s["A"] >= s["C"] + PILOT_RULE["a_vs_c"]
    c3 = s["A"] >= s["N"] + PILOT_RULE["a_vs_n"]
    st = "pilot_signal" if (c1 and c2 and c3) else "pilot_no_signal"
    return st, dict(c_vs_s=c1, a_vs_c=c2, a_vs_n=c3)


def _block_shas(kind, plan):
    out = {}
    for lab, t, b in plan["cells"]:
        f = block_path(kind, lab, t, b)
        for q in (f, f[:-5] + ".actions.npz"):
            out[q] = sha(q) if os.path.exists(q) else None
    return out


def pilot_gate(a):
    plan = load_json(a.plan)
    prob = plan_problems(plan)
    if plan.get("plan_kind") != "pilot":
        prob.append("не план пилота")
    for lab, t, b in plan["cells"]:
        prob += block_problems(plan, a.plan, "pilot", lab, t, b)
    prob += parity_problems(plan, "pilot")
    if prob:
        save_json(GATE_FILE, dict(status="technical_stop", problems=prob[:50],
                                  plan_sha1=sha(a.plan)))
        print("  пилот: ТЕХНИЧЕСКИЙ СТОП — " + "; ".join(prob[:4]))
        return 3
    s = {}
    for lab in PILOT_ARMS:
        k, n = sr_of("pilot", plan, lab)
        if n != 50:
            raise SystemExit(f"{lab}: эпизодов {n} из 50")
        s[lab[-1]] = k
    st, det = pilot_decision(s)
    save_json(GATE_FILE, dict(status=st, successes=s, details=det,
                              rule=PILOT_RULE, plan_sha1=sha(a.plan),
                              blocks=_block_shas("pilot", plan)))
    print(f"  пилот: успехи из 50 — A {s['A']}, C {s['C']}, N {s['N']}, S "
          f"{s['S']}; статус {st} ({det})")
    return 0


# --- самопроверка ------------------------------------------------------------------------------
def selftest():
    assert sorted(task_order()) == list(range(10))
    c = cells("pilot")
    assert len(c) == 40 and len(set(c)) == 40
    assert {x[0] for x in c} == set(PILOT_ARMS)
    for t in range(10):
        assert sum(1 for x in c if x[1] == t) == 4
    first = [x[0] for x in c[:4]]
    second = [x[0] for x in c[4:8]]
    assert first[1:] + first[:1] == second
    assert len(cells("smoke")) == 6 and \
        {x[0] for x in cells("smoke")} == set(SMOKE_ARMS)
    assert 40 * BLOCK_CAP == CAPS["pilot"] - 0 and \
        6 * BLOCK_CAP <= CAPS["checks"]
    # правило пилота
    assert pilot_decision(dict(A=46, C=44, N=43, S=46))[0] == "pilot_signal"
    assert pilot_decision(dict(A=44, C=44, N=40, S=44))[0] == \
        "pilot_no_signal"                                  # A = C
    assert pilot_decision(dict(A=45, C=40, N=45, S=44))[0] == \
        "pilot_no_signal"                                  # A = N
    assert pilot_decision(dict(A=45, C=38, N=40, S=44))[0] == \
        "pilot_no_signal"                                  # C < S − 5
    assert pilot_decision(dict(A=40, C=39, N=39, S=44))[0] == "pilot_signal"
    # шард: чужая запись завершённости и подмена — отказ
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "s")
        rows = [3, 5, 9]
        np.savez(base + ".npz", rows=np.asarray(rows))
        bind = dict(decoder_sha1="d", decoder_weights_hash="w",
                    train_report_sha1="r", panels_sha1="p", code={},
                    prepare_sha1="q")
        save_json(base + ".json", dict(panel="oracle_val", task="t0",
                                       rows=rows, rows_sha1=ko.sha_arr(rows),
                                       npz_sha1=sha(base + ".npz"),
                                       cuts=list(range(36)), **bind))
        assert shard_problems(base, "oracle_val", "t0", rows, bind) == []
        assert shard_problems(base, "oracle_val", "t0", rows,
                              dict(bind, decoder_sha1="x"))
        assert shard_problems(base, "oracle_val", "t0", [3, 5, 8], bind)
        np.savez(base + ".npz", rows=np.asarray([3, 5, 8]))
        assert shard_problems(base, "oracle_val", "t0", rows, bind)
        os.remove(base + ".json")
        assert shard_problems(base, "oracle_val", "t0", rows, bind)
    # обход отрицательного офлайн-гейта: plan_problems пилота вызывает
    # offline_problems
    g = globals()
    keep = {k: g[k] for k in ("offline_problems", "prepare_problems")}
    try:
        g["prepare_problems"] = lambda: []
        g["offline_problems"] = lambda: ["гейт селекторов: stopped_selector"]
        fake = dict(kind="k35_plan", plan_kind="pilot",
                    cells=[list(x) for x in cells("pilot")], binding={},
                    code={}, arms={})
        pr = plan_problems(fake)
        assert pr and any("изменён" in x or "stopped" in x for x in pr)
        try:
            make_plan("pilot", "cpu")
            raise AssertionError("план пилота при отрицательном гейте")
        except SystemExit:
            pass
    finally:
        g.update(keep)
    print("самопроверка k35_protocol пройдена: расписание (smoke 6 блоков, "
          "пилот 40 блоков, сдвиг рук по задачам, бюджеты = блоки × 3050), "
          "правило пилота (C к S ≤ 5, A > C и A > N хотя бы на 1), шард "
          "оракула — чужая запись и подмена отвергаются, пилот без "
          "пройденных офлайн-гейтов не планируется")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-35: протокол")
    ap.add_argument("cmd", choices=(
        "prepare", "plan", "block", "cells", "smoke-check", "smoke-verify",
        "oracle", "oracle-gate", "selectors", "selector-gate",
        "offline-verify", "pilot-gate"))
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--plan")
    ap.add_argument("--op", choices=("reserve", "finish"))
    ap.add_argument("--label")
    ap.add_argument("--task", type=int)
    ap.add_argument("--start", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=8)
    a = ap.parse_args()

    def offline_verify(_a):
        p = offline_problems()
        print("  офлайн-гейты: " + ("пройдены и свежи" if not p else
                                    "; ".join(p[:3])))
        return 0 if not p else 4
    try:
        return dict(prepare=prepare, plan=plan_cmd, block=block_cmd,
                    cells=cells_cmd, oracle=oracle_cmd,
                    selectors=selectors_cmd, **{
                        "smoke-check": smoke_check,
                        "smoke-verify": smoke_verify,
                        "oracle-gate": oracle_gate,
                        "selector-gate": selector_gate,
                        "offline-verify": offline_verify,
                        "pilot-gate": pilot_gate})[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
