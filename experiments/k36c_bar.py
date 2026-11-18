#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36, этап C: настоящая BAR в общем цикле B1 и её предварительные проверки.

BAR — штатный original fullbar + официальный ActionCodec, исполняемый
неизменённым k31_collect.run_episode (B1, 10 waiting, до 600 команд, тот
же масштаб и знак схвата k29_policy.scale_exec, что в K-9h): вход —
k29_policy.Inputs (= подготовка K-9h), генерация —
model.generate(do_sample=False, initial_position_shift=1), команды —
сумма трёх уровней кодбуков -> codec._decode (как decode K-9h для
fullbar), x[..., :7]. На первом вызове процесса — контрольный снимок:
токены -> официальный proc.action_processor.decode -> |Δ| ≤ 1e−3 к своей
сборке (порог K-9h); снимок сохраняется в эпизоде. Источник K-25 не
используется. Схват — собственный BAR.
Журнал вызова: токены [48], команды кодека [T, 7], elapsed, k. Проверка
эпизода: формы, конечность, токены в словаре, исполненное = scale_exec
(команд)[:k] побитово, последовательность elapsed + k, сводка эпизода
(k36_harness.summary_problems), начальные хеши.
Предварительные проверки (§6.4–6.5, checks): legacy t0/t8 init 30 в двух
путях (k18 Sim и BankSim legacy) — побитово; bank-smoke 6 (dev-сцены
0/6/8 × 2) — хеши = банку, контрольный снимок; cold-repeat 3 (первая
dev-сцена каждой задачи, отдельный процесс) = smoke побитово.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k31_collect as kcol  # noqa: E402
import k36_bank as kb36  # noqa: E402
import k36_harness as kh  # noqa: E402
import k36_protocol as pc  # noqa: E402

LABEL = "BAR"
N_LEVEL, N_POS = 3, 16
DECODE_TOL = 1e-3
STAGE_SCOPE = dict(bar_legacy="checks", bar_smoke="checks",
                   bar_repeat="checks", confirm_c="bar_c")
GATES = dict(bar_legacy=os.path.join(pc.OUT, "bar_legacy.json"),
             bar_smoke=os.path.join(pc.OUT, "bar_smoke.json"),
             bar_repeat=os.path.join(pc.OUT, "bar_repeat.json"))
for _k, _v in STAGE_SCOPE.items():
    if _k != "confirm_c":
        kh.STAGE_SCOPE.setdefault(_k, _v)


# --- рука BAR ---------------------------------------------------------------------
def load_bar(torch, device, prep):
    """Original BAR закреплённой в prepare K-36 ревизии; иначе отказ до
    загрузки (и до любого BAR-резерва)."""
    import k21_data as d21
    import k29_policy as kp
    import k31_harness as kh31
    prep21 = json.load(open(d21.REPORT, encoding="utf-8"))
    want = prep["k32_plan"]["bar_revision"]
    if prep21["bar_pin"]["revision"] != want:
        raise SystemExit(f"ревизия BAR {prep21['bar_pin']['revision']} ≠ "
                         f"закреплённой в prepare K-36 {want}")
    bar = d21.load_bar(torch.device(device), prep21["bar_pin"])
    io = kh31.IO(kp.Inputs(torch, bar["proc"], device))
    return dict(bar=bar, io=io, revision=prep21["bar_pin"]["revision"])


class BarArm:
    def __init__(self, torch, bar):
        self.torch, self.bar = torch, bar
        self.model, self.codec, self.E = bar["model"], bar["codec"], bar["E"]
        self.dev = self.E.device
        self.call = 0
        self.episode = 0
        self.keep_first = False
        self.first = None
        self.last = None

    def begin(self, task, init, episode=None):
        self.call = 0
        self.first = None
        self.last = None

    def assemble(self, toks):
        torch = self.torch
        K = np.asarray(toks).reshape(-1, N_LEVEL, N_POS)
        with torch.no_grad():
            z = sum(self.E[j][torch.as_tensor(K[:, j, :]).long().to(self.dev)]
                    for j in range(N_LEVEL))
            x, _ = self.codec._decode(z, embodiment_ids=0)
        return x[..., :7].float().cpu().numpy()

    def decide(self, batch, po, s8, elapsed):
        torch = self.torch
        self.call += 1
        with torch.no_grad():
            tk = self.model.generate(**batch, position_offset=int(po),
                                     do_sample=False,
                                     initial_position_shift=1)
        t = tk.cpu().numpy()
        if t.shape != (1, N_LEVEL * N_POS):
            raise SystemExit(f"BAR: токены формы {t.shape}")
        a = self.assemble(t)[0]
        if not np.isfinite(a).all():
            raise SystemExit("BAR: нечисловые команды")
        if self.call == 1 and self.keep_first:
            self.first = official_check(self.bar["proc"], t, a)
        rec = dict(elapsed=int(elapsed), toks=t[0].astype(np.int64),
                   a_codec=a.astype(np.float32))
        self.last = dict(rec=rec)
        return a.astype(np.float32), rec

    def executed(self, k):
        self.last["rec"]["k"] = int(k)


def official_check(proc, toks, a):
    """Контроль первого вызова: официальный decode(...)[0] — [B, T, D];
    сравниваются ВСЕ шаги decoded[0, :, :7] со своей сборкой [T, 7] после
    проверки форм; |Δ| ≤ 1e−3 (порог K-9h)."""
    dec = np.asarray(proc.action_processor.decode(
        np.asarray(toks).tolist())[0], np.float64)
    if dec.ndim != 3 or dec.shape[0] != 1 or dec.shape[1] != a.shape[0] \
            or dec.shape[2] < 7 or a.shape[1] != 7:
        raise SystemExit(f"BAR: формы официального decode {dec.shape} и "
                         f"своей сборки {a.shape} несовместимы")
    ref = dec[0, :, :7]
    d = float(np.abs(a.astype(np.float64) - ref).max())
    if not np.isfinite(d) or not d <= DECODE_TOL:
        raise SystemExit(f"BAR: своя сборка ≠ официальному decode ({d:.2e})")
    return dict(toks=np.asarray(toks)[0].copy(), a_codec=a.copy(),
                official=ref.astype(np.float32), delta=d)


def snapshot_problems(z):
    """Повторная проверка снимка по сохранённым массивам (не по записанному
    delta): формы, пересчёт |Δ| по f_official, связь с первым вызовом."""
    need = ("f_toks", "f_a_codec", "f_official")
    if any(k not in z for k in need):
        return ["нет контрольного снимка"]
    a, o = np.asarray(z["f_a_codec"], np.float64), np.asarray(
        z["f_official"], np.float64)
    if a.shape != o.shape or a.ndim != 2 or a.shape[1] != 7:
        return [f"снимок: формы {a.shape} / {o.shape}"]
    d = float(np.abs(a - o).max())
    if not np.isfinite(d) or not d <= DECODE_TOL:
        return [f"снимок: |Δ| к официальному decode {d:.2e}"]
    if not np.array_equal(z["f_a_codec"], z["j_a_codec"][0]) or \
            not np.array_equal(z["f_toks"], z["j_toks"][0]):
        return ["снимок не первого вызова журнала"]
    return []


# --- эпизоды BAR ----------------------------------------------------------------------
def want_bar(prep, stage, task, key, reset_id, sim_mode, revision,
             space=None, case_id=None, want_hash=None):
    return dict(eid=f"{STAGE_SCOPE[stage]}/{stage}/{LABEL}/t{task}/{key}",
                scope=STAGE_SCOPE[stage], stage=stage, label=LABEL,
                kind="BAR", seed=None, task=int(task), key=str(key),
                reset_id=int(reset_id), sim_mode=sim_mode,
                space=None if space is None else int(space), case_id=case_id,
                prepare_sha1=prep["sha1"], contract=dict(bar=revision),
                actor_hash=None, source_weights_hash=revision,
                want_hash=None if want_hash is None else list(want_hash))


def pack(ep):
    calls = ep["calls"]
    n = len(calls)
    T = calls[0]["rec"]["a_codec"].shape[0] if n else 0
    J = dict(elapsed=np.zeros(n, np.int64), k=np.zeros(n, np.int64),
             toks=np.zeros((n, N_LEVEL * N_POS), np.int64),
             a_codec=np.zeros((n, T, 7), np.float32))
    for i, c in enumerate(calls):
        r = c["rec"]
        J["elapsed"][i], J["k"][i] = r["elapsed"], c["k"]
        J["toks"][i], J["a_codec"][i] = r["toks"], r["a_codec"]
    return J


def save_bar_ep(want, ep, first=None):
    jp, np_ = kh.ep_paths(want["stage"], LABEL, want["task"], want["key"])
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    arr = dict(actions=ep["actions"].astype(np.float32),
               **{f"j_{k}": v for k, v in pack(ep).items()})
    if first is not None:
        arr.update({f"f_{k}": np.asarray(v) for k, v in first.items()})
    with open(np_ + ".tmp", "wb") as fh:
        np.savez_compressed(fh, **arr)
    os.replace(np_ + ".tmp", np_)
    rec = dict({k: want[k] for k in kh.IDENT}, want_hash=want["want_hash"],
               npz_sha1=kh.sha(np_), phys=int(ep["physical"]),
               seconds=float(ep["seconds"]),
               summary={k: ep[k] for k in kh.SUMMARY})
    pc.save_json(jp, rec)


def bar_episode_problems(rec, z, want, q, vocab):
    import k29_policy as kp
    p = []
    bad = [k for k in kh.IDENT if rec.get(k) != want[k]]
    if bad:
        return [f"идентификаторы {bad}"]
    s = rec["summary"]
    wh = want["want_hash"]
    if wh is not None and (s["init_hash"] != wh[0] or
                           (wh[1] is not None and
                            s["init_hash_full"] != wh[1])):
        p.append("начальный хеш ≠ банку/эталону")
    A = z.get("actions")
    if A is None or kcol.actions_sha(A) != s["actions_sha1"]:
        return p + ["действия ≠ SHA"]
    need = ("j_elapsed", "j_k", "j_toks", "j_a_codec")
    if any(k not in z for k in need) or len(z["j_k"]) == 0:
        return p + ["журнал BAR неполон или пуст"]
    n = len(z["j_k"])
    if z["j_toks"].shape != (n, N_LEVEL * N_POS) or z["j_a_codec"].ndim != 3 \
            or z["j_a_codec"].shape[0] != n or z["j_a_codec"].shape[2] != 7 \
            or z["j_a_codec"].shape[1] < kcol.ENV["horizon"]:
        return p + ["журнал BAR: формы"]
    if not np.isfinite(z["j_a_codec"]).all():
        return p + ["журнал BAR: нечисловые команды"]
    if (z["j_toks"] < 0).any() or (z["j_toks"] >= vocab).any():
        p.append("журнал BAR: токен вне словаря")
    k = z["j_k"]
    if ((k < 1) | (k > kcol.ENV["horizon"])).any() or \
            int(z["j_elapsed"][0]) != 0 or any(
                int(z["j_elapsed"][i]) + int(k[i]) != int(z["j_elapsed"][i + 1])
                for i in range(n - 1)):
        p.append("журнал BAR: последовательность вызовов")
    ex = np.concatenate([kp.scale_exec(z["j_a_codec"][i][None], q)[0]
                         [:int(k[i])] for i in range(n)])
    if not np.array_equal(ex.astype(np.float32), np.asarray(A, np.float32)):
        p.append("исполненное ≠ масштабу записанных команд BAR")
    p += kh.summary_problems(rec, s, len(A))
    return p


def run_bar_ep(want, sim, io, arm, L, q, vocab, keep_first=False,
               crash=None, allowed=None, woe=None, validate=None):
    """allowed — допустимые открытые резервы [(scope, префикс eid)];
    чужой открытый резерв — отказ до списания."""
    import k36b_gate as kg
    if os.path.exists(kh.reject_path(want["eid"])):
        raise SystemExit(f"{want['eid']}: был отвергнут — технический стоп")
    got = kh.load_ep(want)
    if got is None:
        if L.is_done(want["eid"]):
            raise SystemExit(f"{want['eid']}: учтён, но результата нет")
        kg.before_reserve(
            L, want, allowed or [(want["scope"],
                                  f"{want['scope']}/{want['stage']}/")],
            woe or (lambda eid: (_ for _ in ()).throw(
                SystemExit(f"{eid}: нет ожидаемого эпизода"))),
            validate or (lambda r, z, w2: bar_episode_problems(
                r, z, w2, q, vocab)))
        L.reserve(want["scope"], want["eid"], amount=pc.EP)
        arm.keep_first = bool(keep_first)
        ep = kcol.run_episode(sim, io, arm, want["task"], want["reset_id"])
        save_bar_ep(want, ep, arm.first if keep_first else None)
        if crash == "after_write":
            raise KeyboardInterrupt("после записи эпизода")
        L.commit(want["eid"], ep["physical"])
        got = kh.load_ep(want)
    rec, z = got
    pr = kh.ledger_owner_problems(L, want["eid"], rec["phys"], want["scope"])
    if pr:
        raise SystemExit(pr[0])
    L.settle(want["eid"], rec["phys"])
    pr = bar_episode_problems(rec, z, want, q, vocab)
    if pr:
        q_ = kh.quarantine(want, pr)
        raise SystemExit(f"{want['eid']}: эпизод BAR отвергнут ({pr[:3]}) -> "
                         f"карантин {q_}")
    return rec, z


# --- ожидаемые наборы и гейты ------------------------------------------------------------
def bar_wants(prep, kind, revision):
    out = []
    if kind == "bar_legacy":
        for t in kh.LEGACY_TASKS:
            wh = (prep["k32_plan"]["init_ref"][f"{t}:{kh.LEGACY_INIT}"], None)
            for path in ("old", "new"):
                out.append(want_bar(prep, kind, t, f"i{kh.LEGACY_INIT}_{path}",
                                    kh.LEGACY_INIT,
                                    "k18" if path == "old" else "legacy",
                                    revision, want_hash=wh))
        return out
    n = 2 if kind == "bar_smoke" else 1
    for t in kh.SMOKE_TASKS:
        for cid, pos, ih, ihf in kb36.dev_cases(t, n):
            out.append(want_bar(prep, kind, t, cid, pos, "bank", revision,
                                space=kb36.SPACE_DEV, case_id=cid,
                                want_hash=(ih, ihf)))
    return out


def consts_bar(prep):
    return np.asarray(prep["k32_plan"]["scale_q"])


def gate_problems(kind, prep, revision, vocab):
    path = GATES[kind]
    if not os.path.exists(path):
        return [f"нет {path}"]
    g = json.load(open(path, encoding="utf-8"))
    p = []
    if g.get("ok") is not True or g.get("problems") != [] or \
            g.get("prepare_sha1") != prep["sha1"] or \
            g.get("revision") != revision or g.get("vocab") != vocab:
        return [f"{kind}: не пройден / другой prepare или снимок BAR"]
    wants = bar_wants(prep, kind, revision)
    if set(g.get("files") or {}) != kh.files_of(wants):
        return [f"{kind}: набор файлов ≠ ожидаемому"]
    for f, s in g["files"].items():
        if not os.path.exists(f) or kh.sha(f) != s:
            return [f"{kind}: файл {f} изменён"]
    q = consts_bar(prep)
    got = {}
    for w in wants:
        r = kh.load_ep(w)
        if r is None:
            return [f"{kind}: нет {w['eid']}"]
        pr = bar_episode_problems(*r, w, q, vocab)
        if pr:
            return [f"{kind} {w['eid']}: {pr[:2]}"]
        got[w["key"], w["task"]] = r
    if kind == "bar_legacy":
        for t in kh.LEGACY_TASKS:
            d = kh.compare_eps(got[f"i{kh.LEGACY_INIT}_old", t],
                               got[f"i{kh.LEGACY_INIT}_new", t])
            if d:
                p.append(f"bar_legacy t{t}: {d[:3]}")
    elif kind == "bar_smoke":
        for (key, t), (rec, z) in got.items():
            p += [f"bar_smoke t{t} {key}: {x}" for x in snapshot_problems(z)]
    else:
        for w in wants:
            w0 = dict(w, stage="bar_smoke",
                      eid=w["eid"].replace("bar_repeat", "bar_smoke"))
            r0 = kh.load_ep(w0)
            if r0 is None:
                p.append(f"bar_repeat {w['key']}: нет пары smoke")
                continue
            z0 = {k: v for k, v in r0[1].items() if not k.startswith("f_")}
            d = kh.compare_eps((r0[0], z0), got[w["key"], w["task"]])
            if d:
                p.append(f"bar_repeat t{w['task']}: {d[:3]}")
    return p


def preflight_problems(prep, revision, vocab):
    p = []
    for k in GATES:
        p += gate_problems(k, prep, revision, vocab)
    return p


def bar_check(a, kind, sims=None, src=None):
    """Одна из трёх предварительных проверок BAR (отдельный процесс)."""
    import k18_collector as k18c
    if sims is None:
        sims = dict(old=k18c.Sim(1, "persistent"),
                    new=kb36.BankSim("legacy")) if kind == "bar_legacy" else \
            dict(new=kb36.BankSim("bank", kb36.SPACE_DEV))
    import torch
    import k36b_screen as kbs
    try:
        gb = kbs.gate_b_problems()
        if gb:
            raise SystemExit(f"{kind} без passed_B_candidate: " +
                             "; ".join(gb[:3]))
        prep = pc.load_prepare()
        q = consts_bar(prep)
        pr = kh.resources_problems(torch, a.device)
        if pr:
            raise SystemExit("; ".join(pr))
        if kind != "bar_legacy":
            prev = "bar_legacy" if kind == "bar_smoke" else "bar_smoke"
            src0 = src or load_bar(torch, a.device, prep)
            vocab = int(src0["bar"]["codec"].vocab_size)
            p0 = gate_problems(prev, prep, src0["revision"], vocab)
            if p0:
                raise SystemExit(f"{kind} без {prev}: " + "; ".join(p0))
            src = src0
        src = src or load_bar(torch, a.device, prep)
        vocab = int(src["bar"]["codec"].vocab_size)
        L = pc.ledger()
        arm = BarArm(torch, src["bar"])
        files = {}
        wants = bar_wants(prep, kind, src["revision"])
        by_eid = {w["eid"]: w for w in wants}
        for w in wants:
            sim = sims["old" if w["key"].endswith("_old") else "new"]
            run_bar_ep(w, sim, src["io"], arm, L, q, vocab,
                       keep_first=(kind == "bar_smoke"),
                       woe=lambda eid: by_eid[eid])
            files.update(kh.ep_files(w))
        pc.save_json(GATES[kind], dict(ok=True, problems=[], files=files,
                                       prepare_sha1=prep["sha1"],
                                       revision=src["revision"],
                                       vocab=vocab))
        p = gate_problems(kind, prep, src["revision"], vocab)
        pc.save_json(GATES[kind], dict(ok=not p, problems=p, files=files,
                                       prepare_sha1=prep["sha1"],
                                       revision=src["revision"], vocab=vocab,
                                       spent_checks=L.spent("checks")))
        print(f"  {kind}: эпизодов {len(files) // 2}; " +
              ("пройден" if not p else "ОТКАЗ: " + "; ".join(p[:3])))
        return 0 if not p else 3
    finally:
        for s_ in sims.values():
            s_.close()


def selftest():
    import tempfile
    import k32_ppo as kppo
    q = np.array([0.9] * 6 + [1.0])
    prep = dict(sha1="PREP", k32_plan=dict(scale_q=q.tolist(),
                                           init_ref={"0:30": "0" * 16,
                                                     "8:30": "0" * 16}))

    class FakeBarArm(BarArm):
        """Сборка/генерация заменены детерминированной игрушкой."""

        def __init__(self):
            self.call, self.episode, self.keep_first = 0, 0, False
            self.first = self.last = None

        def decide(self, batch, po, s8, elapsed):
            self.call += 1
            x = float(batch["x"])
            t = (np.arange(48) * 7 + int(abs(x) * 1e3)) % 512
            a = np.zeros((20, 7), np.float32)
            a[:, 0] = 0.02 * np.sin(x + np.arange(20))
            a[:, 6] = -1.0
            if self.call == 1 and self.keep_first:
                self.first = dict(toks=t.copy(), a_codec=a.copy(),
                                  official=a.copy(), delta=0.0)
            rec = dict(elapsed=int(elapsed), toks=t.astype(np.int64),
                       a_codec=a)
            self.last = dict(rec=rec)
            return a, rec
    g_ = vars(kh)
    keep = {k: g_[k] for k in ("OUT", "REJECTED")}
    with tempfile.TemporaryDirectory() as td:
        try:
            g_.update(OUT=os.path.join(td, "r"),
                      REJECTED=os.path.join(td, "r", "rejected"))
            L = kcol.Ledger(os.path.join(td, "led.json"), pc.CAPS)
            arm = FakeBarArm()
            w = want_bar(prep, "confirm_c", 0, "k33s1_t0_c00007", 7, "bank",
                         "REV", space=1, case_id="k33s1_t0_c00007")
            rec, z = run_bar_ep(w, kppo.ToySim(), kcol.FakeIO(), arm, L, q,
                                512, keep_first=True)
            assert bar_episode_problems(rec, z, w, q, 512) == []
            assert L.spent("bar_c") == rec["phys"]
            assert "f_toks" in z and np.array_equal(z["f_toks"],
                                                    z["j_toks"][0])
            # повтор без исполнения
            assert run_bar_ep(w, kppo.ToySim(), kcol.FakeIO(), arm, L, q,
                              512)[0] == rec and L.spent("bar_c") == \
                rec["phys"]
            for nm, f in (
                    ("действие", lambda zz: zz["actions"].__setitem__(
                        (0, 0), zz["actions"][0, 0] + 1e-6)),
                    ("команда", lambda zz: zz["j_a_codec"].__setitem__(
                        (0, 0, 0), zz["j_a_codec"][0, 0, 0] + 1e-5)),
                    ("токен", lambda zz: zz["j_toks"].__setitem__((0, 0),
                                                                  9999)),
                    ("NaN", lambda zz: zz["j_a_codec"].__setitem__(
                        (0, 1, 1), np.nan)),
                    ("последний вызов", lambda zz: [zz.__setitem__(
                        k, v[:-1]) for k, v in list(zz.items())
                        if k.startswith("j_")]),
                    ("пустой", lambda zz: [zz.__setitem__(k, v[:0]) for k, v
                                           in list(zz.items())
                                           if k.startswith("j_")])):
                zz = {k: np.array(v, copy=True) for k, v in z.items()}
                f(zz)
                assert bar_episode_problems(rec, zz, w, q, 512), nm
            assert bar_episode_problems(dict(rec, prepare_sha1="X"), z, w, q,
                                        512)
            assert bar_episode_problems(rec, z, dict(
                w, want_hash=["f" * 16, None]), q, 512)
            # авария между записью и commit
            w2 = want_bar(prep, "confirm_c", 0, "k33s1_t0_c00011", 11,
                          "bank", "REV", space=1, case_id="k33s1_t0_c00011")
            try:
                run_bar_ep(w2, kppo.ToySim(), kcol.FakeIO(), arm, L, q, 512,
                           crash="after_write")
                raise AssertionError("авария не сработала")
            except KeyboardInterrupt:
                pass
            L2 = kcol.Ledger(os.path.join(td, "led.json"), pc.CAPS)
            sp = L2.spent("bar_c")
            r2 = run_bar_ep(w2, kppo.ToySim(), kcol.FakeIO(), arm, L2, q, 512)
            assert L2.spent("bar_c") == sp + r2[0]["phys"] and \
                not L2.d["crashes"]
            # открытый резерв этапа A: BAR отказывает без списания
            L2.reserve("confirm_a", "confirm_a/confirm_a/R_s0/t0/k")
            before = json.dumps(L2.d, sort_keys=True)
            w3 = want_bar(prep, "confirm_c", 0, "k33s1_t0_c00013", 13,
                          "bank", "REV", space=1, case_id="k33s1_t0_c00013")
            try:
                run_bar_ep(w3, kppo.ToySim(), kcol.FakeIO(), arm, L2, q, 512)
                raise AssertionError("чужой резерв списан BAR")
            except SystemExit:
                pass
            assert json.dumps(L2.d, sort_keys=True) == before
            # вход BAR без passed_B_candidate — стоп до резерва
            import types
            import k36b_screen as kbs
            keep_g = kbs.gate_b_problems
            closed = []

            class S:
                def close(self):
                    closed.append(1)
            try:
                kbs.gate_b_problems = lambda: ["нет решения B"]
                try:
                    bar_check(types.SimpleNamespace(device="cpu"),
                              "bar_legacy", sims=dict(old=S(), new=S()))
                    raise AssertionError("BAR без решения B")
                except SystemExit:
                    pass
            finally:
                kbs.gate_b_problems = keep_g
            assert len(closed) == 2 and json.dumps(
                L2.d, sort_keys=True) == before
        finally:
            g_.update(keep)
    assert kh.STAGE_SCOPE["bar_smoke"] == "checks"
    # контроль официального decode: [B, T, D] против [T, 7] по всем шагам
    class _AP:
        def __init__(self, arr):
            self.arr = arr

        def decode(self, toks):
            return [self.arr]

    class _P:
        def __init__(self, arr):
            self.action_processor = _AP(arr)
    T_ = 20
    aa = np.cumsum(np.full((T_, 7), 0.01, np.float32), 0)    # меняющийся
    full = np.concatenate([aa, np.zeros((T_, 1), np.float32)], 1)[None]
    toks = np.arange(48)[None]
    f0 = official_check(_P(full), toks, aa)
    assert f0["delta"] == 0.0 and f0["official"].shape == (T_, 7)
    bad = full.copy()
    bad[0, -1, 0] += 1.0                                     # только последний шаг
    try:
        official_check(_P(bad), toks, aa)
        raise AssertionError("расхождение последнего шага принято")
    except SystemExit:
        pass
    try:
        official_check(_P(full[:, :8]), toks, aa)           # другая длина T
        raise AssertionError("несовместимые формы приняты")
    except SystemExit:
        pass
    zz = dict(f_toks=toks[0], f_a_codec=aa, f_official=f0["official"],
              j_a_codec=aa[None], j_toks=toks)
    assert snapshot_problems(zz) == []
    z2 = dict(zz, f_official=bad[0, :, :7])
    assert snapshot_problems(z2)                    # пересчёт, не delta
    z3 = dict(zz, j_a_codec=(aa + 1e-3)[None])
    assert snapshot_problems(z3)
    # ревизия BAR K-21 ≠ prepare K-36 — отказ до загрузки модели и в плане C
    import k21_data as d21
    import k36c_confirm as kc
    keep_r = d21.REPORT
    with tempfile.TemporaryDirectory() as td:
        try:
            d21.REPORT = os.path.join(td, "k21.json")
            pc.save_json(d21.REPORT, dict(bar_pin=dict(revision="OTHER")))
            prep_r = dict(k32_plan=dict(bar_revision="PINNED"))
            for fn in (lambda: load_bar(None, "cpu", prep_r),
                       lambda: kc.bar_revision(prep_r)):
                try:
                    fn()
                    raise AssertionError("чужая ревизия BAR принята")
                except SystemExit:
                    pass
            pc.save_json(d21.REPORT, dict(bar_pin=dict(revision="PINNED")))
            assert kc.bar_revision(prep_r) == "PINNED"
        finally:
            d21.REPORT = keep_r
    print("самопроверка k36c_bar пройдена: эпизод BAR (B1, свой схват) — "
          "журнал токенов и команд, исполненное = масштабу команд; подмены "
          "действия/команды/токена/NaN/последнего вызова/пустой журнал, "
          "чужой prepare, неверный хеш — отказ; контрольный снимок первого "
          "вызова = журналу; повтор без исполнения; авария между записью и "
          "commit — settle без списания; чужой открытый резерв и вход без "
          "passed_B_candidate — отказ до списания; контроль decode по всем "
          "шагам [B, T, D] -> [T, 7] (расхождение последнего шага и чужая "
          "длина — отказ), снимок пересчитывается по f_official; ревизия "
          "BAR ≠ prepare — отказ до загрузки и в плане C")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36 C: BAR")
    ap.add_argument("cmd", choices=("bar-legacy", "bar-smoke", "bar-repeat"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    kind = a.cmd.replace("-", "_")
    try:
        return bar_check(a, kind)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
