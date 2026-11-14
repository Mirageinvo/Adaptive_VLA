#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-35: руки A / C / N / S для штатного холодного харнесса K-9h.

A, C, N — один загруженный K-35 (общий best K35-A либо, в collector-smoke,
состояние init = веса K-34 T0) с контроллером двух решений:
  A — решения по scores селекторов (k35_selector.decide, ничья — меньший ID);
  C — постоянный срез c_const (a = c // 6, b = c % 6);
  N — равномерные a, b; локальный RNG на элемент с ключом
      (35, training_seed, task, init, env_seed, call_index, stage).
У A/C/N обе сети выбора ВСЕГДА вычисляются (одинаковый состав модулей);
у C/N их scores записываются и не используются. C_fast (только для замера
задержки) пропускает сети выбора.
Признаки — в GPU FP32 вне autocast из snap модели (roots_n, iv, mids_n,
iv_mid, канонические ID средних, one-hot a) по закреплённым PCA и
стандартизациям — та же формула, что k35_selector.X_first / X_second.
S — неизменённый выход 24 K-34 T0 (k34_policy, журнал K-34).
Журнал каждого вызова (K-35): контроллер, a/b/c, канонические конечные
ID, родители листьев, X и scores обоих селекторов, счётчики, router по
шести точкам, partial y трёх уровней и итоговый y. В отладочных блоках
(smoke) на первом вызове — снимок: признаки уровней, интервалы, веса,
входы и выходы трёх голов, y/arm/grip и фактические dtype.
"""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k35_model as k35m  # noqa: E402
import k35_selector as ks  # noqa: E402
import k35_topology as tp  # noqa: E402

KINDS = ("A", "C", "N")
KIND_ID = dict(A=0, C=1, N=2, C_fast=3)
TRAIN_SEED = 0
TOL = dict(y=1e-5, grip_logit=1e-5, arm=1e-6)
ROUTE_TOL = 1e-5
TAB = tp.tables()
LEVELS = ("root", "mid", "leaf")
N_TERM = dict(root=2, mid=2, leaf=4)


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def n_key(task, init, env_seed, call, stage, seed=TRAIN_SEED):
    return [35, int(seed), int(task), int(init), int(env_seed), int(call),
            int(stage)]


def n_choice(task, init, env_seed, call, stage, seed=TRAIN_SEED):
    return int(np.random.default_rng(n_key(task, init, env_seed, call, stage,
                                           seed)).integers(6))


# --- контроллер ---------------------------------------------------------------------
class Controller:
    """first/second для K35.forward; scores в GPU FP32 вне autocast."""

    def __init__(self, torch, kind, bundle, device, c_const=None, ctx=None,
                 compute_selectors=True):
        if kind not in KINDS + ("C_fast",):
            raise SystemExit(f"контроллер {kind}")
        if kind in ("C", "C_fast") and c_const is None:
            raise SystemExit("C без c_const")
        self.torch, self.kind = torch, kind
        self.dev = torch.device(device)
        self.c_const = None if c_const is None else int(c_const)
        self.ctx = ctx or {}
        self.compute = compute_selectors and kind != "C_fast"
        if kind == "A" and not self.compute:
            raise SystemExit("A без сетей выбора")
        f = dict(device=self.dev, dtype=torch.float32)
        T = bundle["Tr"]
        self.t = {f"{k}.{kk}": torch.as_tensor(vv, **f)
                  for k, v in T.items() for kk, vv in v.items()}
        nets = ks.nets_of(torch, bundle)
        self.net_root = nets["net_root"].to(self.dev).float()
        self.net_mid = nets["net_mid"].to(self.dev).float()
        self.bundle_hash = bundle["bundle_hash"]
        self.rec = {}

    def _pca(self, pre, x):
        return (x.float() - self.t[f"{pre}.mean"]) @ self.t[f"{pre}.comp"].T

    def raw1(self, roots_n, iv):
        B = roots_n.shape[0]
        z = self._pca("P_root", roots_n)
        return self.torch.cat([z, iv.float() / 20.0], -1).reshape(B, -1)

    def x1(self, roots_n, iv):
        return (self.raw1(roots_n, iv) - self.t["S1.mu"]) / self.t["S1.sd"]

    def x2(self, roots_n, iv, mids_n, iv_mid, mid_ids, a):
        torch = self.torch
        B = roots_n.shape[0]
        z = self._pca("P_mid", mids_n)
        ids = torch.as_tensor(mid_ids, device=self.dev).float() / 7.0
        f2 = torch.cat([z, iv_mid.float() / 20.0, ids[..., None]], -1)
        cont = torch.cat([self.raw1(roots_n, iv), f2.reshape(B, -1)], -1)
        cont = (cont - self.t["S2.mu"]) / self.t["S2.sd"]
        oh = torch.nn.functional.one_hot(
            torch.as_tensor(a, device=self.dev).long(), 6).float()
        return torch.cat([cont, oh], -1)

    def begin(self, call):
        self.call = int(call)
        self.rec = {}

    def _decide(self, stage, scores, B):
        if self.kind == "A":
            return ks.decide(scores)
        if self.kind in ("C", "C_fast"):
            v = self.c_const // 6 if stage == 1 else self.c_const % 6
            return np.full(B, v, np.int64)
        inits = self.ctx["inits"]
        if len(inits) != B:
            raise SystemExit(f"N: элементов {B}, init {len(inits)}")
        return np.asarray([n_choice(self.ctx["task"], i, self.ctx["env_seed"],
                                    self.call, stage) for i in inits],
                          np.int64)

    def _scores(self, net, X):
        torch = self.torch
        with torch.no_grad(), torch.autocast(self.dev.type, enabled=False):
            return net(X).float()

    def first(self, info):
        torch = self.torch
        B = int(info["B"])
        sc = X = None
        if self.compute:
            with torch.autocast(self.dev.type, enabled=False):
                X = self.x1(info["roots_n"], info["iv"])
            sc = self._scores(self.net_root, X).cpu().numpy()
            if not np.isfinite(sc).all():
                raise SystemExit("селектор корней: нечисловые scores")
        a = self._decide(1, sc, B)
        self.rec.update(a=a, scores1=sc, X1=None if X is None else
                        X.cpu().numpy())
        return torch.as_tensor(a)

    def second(self, info):
        torch = self.torch
        B = int(info["B"])
        a = np.asarray(info["a"]).reshape(-1)
        sc = X = None
        if self.compute:
            with torch.autocast(self.dev.type, enabled=False):
                X = self.x2(info["roots_n"], info["iv"], info["mids_n"],
                            info["iv_mid"], np.asarray(info["mids"]), a)
            sc = self._scores(self.net_mid, X).cpu().numpy()
            if not np.isfinite(sc).all():
                raise SystemExit("селектор средних: нечисловые scores")
        b = self._decide(2, sc, B)
        self.rec.update(b=b, scores2=sc, X2=None if X is None else
                        X.cpu().numpy())
        return torch.as_tensor(b)


# --- загрузка -------------------------------------------------------------------------
def load_decoder(torch, spec, device):
    """K-35 декодер: init (веса K-34 T0) или обученный best K35-A."""
    import k35_train as k35t
    if spec["decoder"] == "init":
        mod, bar = k35t.load_k35(torch, device, checkpoint=False)
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
        return mod, bar, dict(weights_hash=spec["decoder_weights_hash"],
                              update=None)
    if sha(spec["decoder_ckpt"]) != spec["decoder_sha1"]:
        raise SystemExit("best.pt K-35 ≠ плану")
    mod, bar, ck = k35t.load_trained(torch, device, checkpoint=False)
    if ck["weights_hash"] != spec["decoder_weights_hash"]:
        raise SystemExit("отпечаток best.pt K-35 ≠ плану")
    return mod, bar, ck


def load_selectors(torch, spec, H):
    if spec["selectors"] == "init":
        B = ks.init_bundle(torch, H)
        if B["bundle_hash"] != spec["selectors_hash"]:
            raise SystemExit("набор init селекторов ≠ плану")
        return B
    if sha(spec["selectors_path"]) != spec["selectors_sha1"]:
        raise SystemExit("файл селекторов ≠ плану")
    return ks.load_bundle(torch, spec["selectors_path"],
                          spec["selectors_hash"])


def router_record(infos):
    import k26_attention as ka
    if len(infos) != 6:
        raise SystemExit(f"router: агрегатов {len(infos)}, ожидалось 6")
    f = [ka.info_floats(x) for x in infos]
    return dict(attn_pad=np.float32([x["pad_mass"] for x in f]),
                attn_sumerr=np.float32([x["sum_err"] for x in f]),
                attn_finite=np.float32([float(x["finite"]) for x in f]))


def build_arm(device, basis, lab, torch, *, spec, plan_sha1, ctx,
              debug=False):
    """spec['kind']: K35 (A/C/N) или S (K-34 T0, выход 24)."""
    if spec["kind"] == "S":
        return build_s(device, basis, lab, torch, spec=spec,
                       plan_sha1=plan_sha1)
    import k21_main_policy as kmp
    import k26_attention as ka
    ctrl_kind = spec["controller"]
    orig = kmp.load_policy
    holder = {}

    def loader(torch_, l_, c_, dev_):
        import k21_data as d21
        import json
        mod, bar, ck = load_decoder(torch_, spec, dev_)
        holder["ck"] = ck
        prep = json.load(open(d21.REPORT, encoding="utf-8"))
        return mod, bar, dict(weights_hash=spec["decoder_weights_hash"],
                              update=ck.get("update")), prep
    kmp.load_policy = loader
    try:
        arm = kmp.build_arm(device, basis, lab, torch,
                            ckpt=spec.get("decoder_ckpt") or
                            spec["source_ckpt"], plan_sha1=plan_sha1)
    finally:
        kmp.load_policy = orig
    mod = arm.k21
    mod.collect = True
    H = int(mod.q0.shape[-1])
    bundle = load_selectors(torch, spec, H)
    ctrl = Controller(torch, ctrl_kind, bundle, device,
                      c_const=spec.get("c_const"),
                      ctx=dict(task=int(ctx["task"]),
                               inits=[int(x) for x in ctx["inits"]],
                               env_seed=int(ctx["env_seed"])))
    mod.controller = ctrl
    last, st = {}, dict(call=0, dbg_done=False, dbg=None)
    fwd0 = mod.forward

    def forward(*a, **k):
        out = fwd0(*a, **k)
        last["out"] = out
        return out
    mod.forward = forward
    part0 = mod._partial

    def partial(level, h, iv, w, cnt):
        y = part0(level, h, iv, w, cnt)
        if st["dbg"] is not None:
            x = mod.bar.action_expert.norm(h)
            st["dbg"].update({
                f"dbg_x_{level}": x.float().cpu().numpy(),
                f"dbg_iv_{level}": iv.float().cpu().numpy(),
                f"dbg_w_{level}": w.float().cpu().numpy(),
                f"dbg_y_{level}": y.float().cpu().numpy(),
                f"dbg_dtype_x_{level}": np.asarray(str(x.dtype)),
                f"dbg_dtype_y_{level}": np.asarray(str(y.dtype))})
        return y
    mod._partial = partial
    act0 = arm.act

    def act(batch, pos_off, autocast, first):
        if first:
            st["call"] = 0
        st["call"] += 1
        last.clear()
        ctrl.begin(st["call"])
        keep = debug and not st["dbg_done"]
        st["dbg"] = {} if keep else None
        res = act0(batch, pos_off, autocast, first)
        prob = k35m.counters_problems(mod.counters, decisions=2)
        if prob or "out" not in last:
            raise SystemExit(f"рука {lab}: " + "; ".join(prob or [
                "нет выхода модели"]))
        o = last["out"]
        rec = router_record(o.get("router") or [])
        if (rec["attn_pad"] != 0).any() or \
                (rec["attn_sumerr"] > ka.SUM_TOL).any() or \
                (rec["attn_finite"] != 1).any():
            raise SystemExit(f"рука {lab}: router — padding/суммы/"
                             f"конечность")
        a = o["a"].numpy().astype(np.int64)
        b = o["b"].numpy().astype(np.int64)
        c = o["c"].numpy().astype(np.int64)
        if not (np.array_equal(a, ctrl.rec["a"]) and
                np.array_equal(b, ctrl.rec["b"])):
            raise SystemExit(f"рука {lab}: решения модели ≠ контроллеру")
        B = len(c)
        nan6 = np.full((B, 6), np.nan, np.float32)
        cnt = mod.counters
        rec.update(
            ctrl=np.int32(KIND_ID[ctrl_kind]), a=a, b=b, c=c,
            term_roots=TAB["term_roots"][c], mids=TAB["mids"][c],
            term_mid_local=TAB["term_mid_local"][c], leaves=TAB["leaves"][c],
            leaf_parent_local=TAB["leaf_parent_local"][c],
            scores1=nan6 if ctrl.rec["scores1"] is None else
            ctrl.rec["scores1"].astype(np.float32),
            scores2=nan6 if ctrl.rec["scores2"] is None else
            ctrl.rec["scores2"].astype(np.float32),
            X1=ctrl.rec["X1"].astype(np.float32),
            X2=ctrl.rec["X2"].astype(np.float32),
            n_vision=np.int32(cnt["vision"]), n_kv=np.int32(cnt["kv"]),
            n_text=np.int32(sum(cnt["text"])),
            active=np.int32(cnt["active"]),
            router_q=np.int32(cnt["router"]),
            n_expand=np.int32(cnt["expand"]),
            n_heads=np.int32(len(cnt["heads"])),
            n_decisions=np.int32(cnt["decisions"]),
            y=o["y"].float().cpu().numpy(),
            **{f"partial_{k}": o["partial"][k].float().cpu().numpy()
               for k in LEVELS})
        if keep:
            sn = o["snap"]
            rec.update(st["dbg"])
            rec.update(
                dbg_roots_n=sn["roots_n"].float().cpu().numpy(),
                dbg_iv_root_all=sn["iv_root"].float().cpu().numpy(),
                dbg_W0=sn["W0"].float().cpu().numpy(),
                dbg_mids_n=sn["mids_n"].float().cpu().numpy(),
                dbg_iv_mid_all=sn["iv_mid"].float().cpu().numpy(),
                dbg_W_mid=sn["W_mid"].float().cpu().numpy(),
                dbg_iv_leaf_all=o["iv_leaf"].float().cpu().numpy(),
                dbg_W_leaf=o["W_leaf"].float().cpu().numpy(),
                dbg_arm=o["arm"].float().cpu().numpy(),
                dbg_grip=o["grip_logit"].float().cpu().numpy(),
                dbg_y_final=o["y"].float().cpu().numpy(),
                dbg_dtype_roots_n=np.asarray(str(sn["roots_n"].dtype)),
                dbg_dtype_arm=np.asarray(str(o["arm"].dtype)))
            st["dbg_done"] = True
        st["dbg"] = None
        arm.log.add(**rec)
        return res
    arm.act = act
    arm.meta.update(
        k35_kind="K35", k35_controller=ctrl_kind,
        k35_c_const=spec.get("c_const"), k35_decoder=spec["decoder"],
        k35_decoder_weights_hash=spec["decoder_weights_hash"],
        k35_decoder_sha1=spec.get("decoder_sha1"),
        k35_selectors=spec["selectors"],
        k35_selectors_hash=bundle["bundle_hash"],
        k35_task=int(ctx["task"]), k35_inits=[int(x) for x in ctx["inits"]],
        k35_env_seed=int(ctx["env_seed"]), k35_debug=bool(debug),
        k35_train_seed=TRAIN_SEED, layers_per_call=24,
        router_version=ka.VERSION,
        k35_policy_sha1=sha(os.path.abspath(__file__)))
    return arm


def build_s(device, basis, lab, torch, *, spec, plan_sha1):
    """S: K-34 T0 выход 24 через k34_policy; журнал K-34, метка K-35."""
    import k34_policy as kp34
    arm = kp34.build_arm(device, basis, "parents_attn24_s0_d24", torch,
                         spec=spec["k34"], plan_sha1=plan_sha1, debug=False)
    arm.log.label = lab
    arm.meta.update(label=lab, k35_kind="S",
                    k35_policy_sha1=sha(os.path.abspath(__file__)))
    return arm


# --- проверки журнала -------------------------------------------------------------------
def assemble(partials, L, U):
    return k35m.assemble_np(partials, L, U)


def journal_problems(lab, z, spec, ctx, bounds, debug=False):
    """Журнал K-35 (A/C/N): счётчики, маршрут ↔ c, правило контроллера,
    сборка y и физическая рука ↔ записанные partial y, router; снимок."""
    if spec["kind"] == "S":
        import k34_policy as kp34
        return kp34.journal_problems("parents_attn24_s0_d24", z)
    kind = spec["controller"]
    need = ("k21_ctrl", "k21_a", "k21_b", "k21_c", "k21_term_roots",
            "k21_mids", "k21_term_mid_local", "k21_leaves",
            "k21_leaf_parent_local", "k21_scores1", "k21_scores2", "k21_X1",
            "k21_X2", "k21_n_vision", "k21_n_kv", "k21_n_text", "k21_active",
            "k21_router_q", "k21_n_expand", "k21_n_heads", "k21_n_decisions",
            "k21_y", "k21_partial_root", "k21_partial_mid",
            "k21_partial_leaf", "k21_attn_pad", "k21_attn_sumerr",
            "k21_attn_finite", "k15g_chunk", f"k15d_level_{lab}")
    miss = [k for k in need if k not in z]
    if miss:
        return [f"{lab}: в журнале нет {miss[:5]}"]
    p = []
    C = len(z["k15g_chunk"])
    if C == 0:
        return [f"{lab}: пустой журнал"]
    short = [k for k in need if np.asarray(z[k]).shape[:1] != (C,)]
    if short:
        return [f"{lab}: число записей ≠ числу вызовов: {short[:4]}"]
    a, b, c = (np.asarray(z[f"k21_{k}"]) for k in ("a", "b", "c"))
    B = len(ctx["inits"])
    for k in ("a", "b", "c"):
        if np.asarray(z[f"k21_{k}"]).shape != (C, B):
            return [f"{lab}: {k} формы {np.asarray(z[f'k21_{k}']).shape}"]
    for k, v in (("k21_ctrl", KIND_ID[kind]), ("k21_n_vision", 1),
                 ("k21_n_kv", 1), ("k21_n_text", 24), ("k21_n_expand", 2),
                 ("k21_n_heads", 3), ("k21_n_decisions", 2)):
        x = np.asarray(z[k])
        if x.shape[0] != C or not (x == v).all():
            p.append(f"{lab}: {k[4:]} ≠ {v} на части вызовов")
    if not (np.asarray(z["k21_active"]) == np.int32(tp.ACTIVE)).all() or \
            not (np.asarray(z["k21_router_q"]) ==
                 np.int32(tp.ROUTER_ACTIVE)).all():
        p.append(f"{lab}: активные query / router-Q ≠ 4/8/12")
    if (a < 0).any() or (a > 5).any() or (b < 0).any() or (b > 5).any() or \
            not np.array_equal(c, 6 * a + b):
        return p + [f"{lab}: c ≠ 6a + b или вне диапазона"]
    for k in ("term_roots", "mids", "term_mid_local", "leaves",
              "leaf_parent_local"):
        if not np.array_equal(np.asarray(z[f"k21_{k}"]), TAB[k][c]):
            p.append(f"{lab}: {k} ≠ топологии среза")
    s1 = np.asarray(z["k21_scores1"], np.float64)
    s2 = np.asarray(z["k21_scores2"], np.float64)
    X1 = np.asarray(z["k21_X1"])
    X2 = np.asarray(z["k21_X2"])
    if s1.shape != (C, B, 6) or s2.shape != (C, B, 6) or \
            X1.shape != (C, B, ks.D_FIRST) or X2.shape != (C, B, ks.D_SECOND):
        return p + [f"{lab}: формы scores/X"]
    if not (np.isfinite(s1).all() and np.isfinite(s2).all() and
            np.isfinite(X1).all() and np.isfinite(X2).all()):
        p.append(f"{lab}: scores/X не конечны")
    elif not np.array_equal(X2[..., -6:], np.eye(6, dtype=X2.dtype)[a]):
        p.append(f"{lab}: one-hot a второго селектора ≠ решению a")
    if kind == "A" and not p:
        da = np.stack([ks.decide(s1[i]) for i in range(C)])
        db = np.stack([ks.decide(s2[i]) for i in range(C)])
        if not (np.array_equal(da, a) and np.array_equal(db, b)):
            p.append(f"{lab}: решения A ≠ правилу по записанным scores")
    elif kind == "C":
        if not (c == int(spec["c_const"])).all():
            p.append(f"{lab}: срез C ≠ c_const")
    elif kind == "N":
        ch = np.asarray(z["k15g_chunk"])
        ea = np.asarray([[n_choice(ctx["task"], i, ctx["env_seed"], k, 1)
                          for i in ctx["inits"]] for k in ch])
        eb = np.asarray([[n_choice(ctx["task"], i, ctx["env_seed"], k, 2)
                          for i in ctx["inits"]] for k in ch])
        if not (np.array_equal(ea, a) and np.array_equal(eb, b)):
            p.append(f"{lab}: решения N ≠ закреплённому RNG")
    # сборка и физическая команда
    parts = {k: np.asarray(z[f"k21_partial_{k}"], np.float32) for k in LEVELS}
    yz = np.asarray(z["k21_y"], np.float32)
    if any(v.shape != (C, B, 20, 7) for v in parts.values()) or \
            yz.shape != (C, B, 20, 7):
        return p + [f"{lab}: формы partial/y"]
    y, arm, grip = assemble(parts, bounds["L"], bounds["U"])
    ey = np.abs(y - yz)
    if not (np.isfinite(ey).all() and float(ey.max()) <= TOL["y"]):
        p.append(f"{lab}: y ≠ сумме partial ({float(np.nanmax(ey))})")
    lv = np.asarray(z[f"k15d_level_{lab}"], np.float32)
    if lv.shape != (C, B, 8, 7):
        return p + [f"{lab}: журнал команд формы {lv.shape}"]
    ea = np.abs(arm[:, :, :8] - lv[..., :6])
    if not (np.isfinite(ea).all() and float(ea.max()) <= TOL["arm"]):
        p.append(f"{lab}: команда руки ≠ сборке из partial "
                 f"({float(np.nanmax(ea))})")
    g = np.where(yz[:, :, :8, 6] > 0, 1.0, -1.0).astype(np.float32)
    if not np.array_equal(g, lv[..., 6]):
        p.append(f"{lab}: схват ≠ знаку записанного y")
    import k26_attention as ka
    pad = np.asarray(z["k21_attn_pad"])
    if pad.shape != (C, 6) or (pad != 0).any() or \
            (np.asarray(z["k21_attn_sumerr"]) > ka.SUM_TOL).any() or \
            (np.asarray(z["k21_attn_finite"]) != 1).any():
        p.append(f"{lab}: router — padding/суммы/конечность")
    if debug:
        p += debug_problems(lab, z, bounds)
    return p


def debug_problems(lab, z, bounds):
    """Снимок первого вызова: формы, конечность, сборка, связь с журналом."""
    keys = [f"k21_dbg_{n}_{lv}" for lv in LEVELS for n in ("x", "iv", "w",
                                                          "y")]
    keys += ["k21_dbg_arm", "k21_dbg_grip", "k21_dbg_y_final",
             "k21_dbg_roots_n", "k21_dbg_mids_n"]
    miss = [k for k in keys if k not in z]
    if miss:
        return [f"{lab}: нет снимка {miss[:4]}"]
    g = {k[8:]: np.asarray(z[k])[0] for k in keys}
    B = g["arm"].shape[0]
    p = []
    for lv in LEVELS:
        n = N_TERM[lv]
        want = dict(iv=(B, n, 2), w=(B, 20, n), y=(B, 20, 7))
        for nm, sh in want.items():
            if g[f"{nm}_{lv}"].shape != sh:
                p.append(f"{lab}: снимок {nm}_{lv} формы "
                         f"{g[f'{nm}_{lv}'].shape}, ожидалось {sh}")
        x = g[f"x_{lv}"]
        if x.ndim != 3 or x.shape[:2] != (B, n):
            p.append(f"{lab}: снимок x_{lv} формы {x.shape}")
    if p:
        return p
    bad = [k for k, v in g.items() if not np.isfinite(v).all()]
    if bad:
        return [f"{lab}: снимок не конечен: {bad}"]
    y, arm, grip = assemble({lv: g[f"y_{lv}"] for lv in LEVELS},
                            bounds["L"], bounds["U"])
    if float(np.abs(y - g["y_final"]).max()) > TOL["y"] or \
            float(np.abs(arm - g["arm"]).max()) > TOL["arm"] or \
            float(np.abs(grip - g["grip"]).max()) > TOL["grip_logit"]:
        p.append(f"{lab}: снимок — сборка из partial ≠ записанной")
    for lv in LEVELS:
        if not np.array_equal(g[f"y_{lv}"],
                              np.asarray(z[f"k21_partial_{lv}"])[0]):
            p.append(f"{lab}: снимок {lv} не первого вызова")
    l0 = np.asarray(z[f"k15d_level_{lab}"])[0]
    if not np.array_equal(l0[..., :6], g["arm"][:, :8].astype(l0.dtype)):
        p.append(f"{lab}: снимок не первого вызова — рука ≠ журналу")
    return p


def head_recompute(torch, z, weights, L, U, device="cpu", autocast=False):
    """Три головы из весов (без модели) на входах снимка -> max|Δ| y, arm,
    grip к записанным (NaN при несовпадении форм или нечисловых)."""
    import k21_tree as kt
    dev = torch.device(device)
    x0 = np.asarray(z["k21_dbg_x_root"])[0]
    H = x0.shape[-1]
    ys = {}
    for lv in LEVELS:
        pre = k35m.HEADS[lv] + "."
        sd = {k[len(pre):]: v for k, v in weights.items()
              if k.startswith(pre)}
        R = kt.build(torch)["Readout"](H, L, U).to(dev)
        R.load_state_dict(sd)
        x = torch.as_tensor(np.asarray(z[f"k21_dbg_x_{lv}"])[0], device=dev)
        iv = torch.as_tensor(np.asarray(z[f"k21_dbg_iv_{lv}"])[0], device=dev)
        w = torch.as_tensor(np.asarray(z[f"k21_dbg_w_{lv}"])[0], device=dev)
        with torch.no_grad():
            if autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    ys[lv] = R(x, iv, w)["y"].float()
            else:
                ys[lv] = R(x, iv, w)["y"].float()
    with torch.no_grad():
        y = ys["root"] + ys["mid"] + ys["leaf"]
        Lb = torch.as_tensor(np.asarray(L, np.float32), device=dev)
        Ub = torch.as_tensor(np.asarray(U, np.float32), device=dev)
        mid, half = (Ub + Lb) / 2, (Ub - Lb) / 2
        arm = mid + half * torch.tanh(y[..., :6] / half)
    out = {}
    for k, got, nm in (("y", y, "k21_dbg_y_final"), ("arm", arm, "k21_dbg_arm"),
                       ("grip_logit", y[..., 6], "k21_dbg_grip")):
        ref = np.asarray(z[nm])[0]
        g = got.cpu().numpy()
        out[k] = float(np.abs(g - ref).max()) if g.shape == ref.shape and \
            np.isfinite(g).all() and np.isfinite(ref).all() else float("nan")
    return out


def route_recompute(torch, z, bundle, device="cpu"):
    """scores селекторов из записанных X (по вызову, B как в харнессе) ->
    max|Δ| к записанным scores; решения A по пересчитанным scores
    (описательно: ничьи могут разрешаться иначе)."""
    nets = ks.nets_of(torch, bundle)
    dev = torch.device(device)
    out = {}
    for nm, X, S in (("net_root", "k21_X1", "k21_scores1"),
                     ("net_mid", "k21_X2", "k21_scores2")):
        net = nets[nm].to(dev).float()
        Xa = np.asarray(z[X], np.float32)
        Sa = np.asarray(z[S], np.float32)
        worst, flips = 0.0, 0
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            for i in range(len(Xa)):
                s = net(torch.as_tensor(Xa[i], device=dev)).float().cpu()\
                    .numpy()
                if s.shape != Sa[i].shape or not np.isfinite(s).all():
                    return {nm: float("nan")}
                worst = max(worst, float(np.abs(s - Sa[i]).max()))
                flips += int((ks.decide(s) != ks.decide(Sa[i])).sum())
        out[nm] = worst
        out[f"{nm}_decision_flips"] = flips
    return out


def selftest():
    import torch
    # 1. признаки контроллера (torch) = k35_selector.X_first / X_second
    rng = np.random.default_rng(0)
    H, n = 16, 24
    O = dict(roots=rng.normal(size=(n, 4, H)).astype(np.float32),
             iv_root=rng.uniform(0, 20, (n, 4, 2)).astype(np.float32),
             mids=rng.normal(size=(n, 6, 4, H)).astype(np.float32),
             iv_mid=rng.uniform(0, 20, (n, 6, 4, 2)).astype(np.float32))
    keep = ks.P_PCA
    try:
        ks.P_PCA = 8
        Tr = ks.fit_transforms(O, np.arange(n))
    finally:
        ks.P_PCA = keep
    D1, D2 = 4 * (8 + 2), 4 * (8 + 2) + 4 * (8 + 3) + 6
    S = dict(Tr=Tr, net_root=ks.make_net(torch, D1, 1),
             net_mid=ks.make_net(torch, D2, 2))
    with torch.no_grad():
        for net in (S["net_root"], S["net_mid"]):
            net[2].weight.normal_(0, 0.3, generator=torch.Generator()
                                  .manual_seed(5))
    kd = ks.D_FIRST, ks.D_SECOND
    try:
        ks.D_FIRST, ks.D_SECOND = D1, D2
        B_ = dict(ks.bundle_state(S), meta={})
        B_["bundle_hash"] = ks.bundle_hash(B_)
        ctl = Controller(torch, "A", B_, "cpu")
        idx = np.arange(5)
        X1n = ks.X_first(Tr, O, idx)
        X1t = ctl.x1(torch.as_tensor(O["roots"][idx]),
                     torch.as_tensor(O["iv_root"][idx])).numpy()
        assert np.abs(X1n - X1t).max() < 1e-4
        a = np.array([0, 3, 5, 1, 2])
        X2n = ks.X_second(Tr, O, idx, a)
        X2t = ctl.x2(torch.as_tensor(O["roots"][idx]),
                     torch.as_tensor(O["iv_root"][idx]),
                     torch.as_tensor(O["mids"][idx, a]),
                     torch.as_tensor(O["iv_mid"][idx, a]),
                     ks.mid_ids_of(a), a).numpy()
        assert np.abs(X2n - X2t).max() < 1e-4
        # решения A через контроллер = k35_selector.choose
        ctl.begin(1)
        ra = ctl.first(dict(roots_n=torch.as_tensor(O["roots"][idx]),
                            iv=torch.as_tensor(O["iv_root"][idx]),
                            B=5)).numpy()
        aa, bb, cc = ks.choose(torch, S, O, idx)
        assert np.array_equal(ra, aa)
        rb = ctl.second(dict(roots_n=torch.as_tensor(O["roots"][idx]),
                             iv=torch.as_tensor(O["iv_root"][idx]),
                             mids_n=torch.as_tensor(O["mids"][idx, aa]),
                             iv_mid=torch.as_tensor(O["iv_mid"][idx, aa]),
                             mids=ks.mid_ids_of(aa), a=aa, B=5)).numpy()
        assert np.array_equal(rb, bb)
        # 2. сохранение / загрузка набора; порча — отказ
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "sel.pt")
            h = ks.save_bundle(torch, S, f, dict(x=1))
            L_ = ks.load_bundle(torch, f, h)
            nets = ks.nets_of(torch, L_)
            Xt = torch.as_tensor(X1n)
            assert torch.equal(nets["net_root"](Xt), S["net_root"](Xt))
            bad = torch.load(f, weights_only=False)
            bad["net_mid"]["0.bias"] = bad["net_mid"]["0.bias"] + 1e-3
            torch.save(bad, f)
            try:
                ks.load_bundle(torch, f, h)
                raise AssertionError("испорченный набор принят")
            except SystemExit:
                pass
        # 3. C всегда c_const; N — ключевой RNG, разные элементы, не
        # сдвигает глобальные RNG; C_fast — без сетей
        ctx = dict(task=3, inits=[20, 21, 22, 23, 24], env_seed=101)
        st_np = np.random.get_state()[1].copy()
        st_t = torch.random.get_rng_state().clone()
        info1 = dict(roots_n=torch.as_tensor(O["roots"][idx]),
                     iv=torch.as_tensor(O["iv_root"][idx]), B=5)
        cn = Controller(torch, "N", B_, "cpu", ctx=ctx)
        cn.begin(7)
        an = cn.first(info1).numpy()
        assert an.tolist() == [n_choice(3, i, 101, 7, 1) for i in ctx["inits"]]
        cn.begin(7)
        assert np.array_equal(cn.first(info1).numpy(), an)
        assert np.array_equal(np.random.get_state()[1], st_np) and \
            torch.equal(torch.random.get_rng_state(), st_t)
        many = [n_choice(3, 20 + i % 5, 101, k, s) for k in range(1, 200)
                for i in range(5) for s in (1, 2)]
        assert set(many) == set(range(6))
        assert len({tuple(n_choice(3, i, 101, 1, 1) for i in range(20, 25)),
                    tuple(n_choice(3, i, 101, 2, 1) for i in range(20, 25))}
                   ) == 2
        assert cn.rec["scores1"] is not None          # сети вычислены
        cc_ = Controller(torch, "C", B_, "cpu", c_const=17)
        cc_.begin(1)
        assert (cc_.first(info1).numpy() == 2).all()
        cf = Controller(torch, "C_fast", B_, "cpu", c_const=17)
        cf.begin(1)
        assert (cf.first(info1).numpy() == 2).all() and \
            cf.rec["scores1"] is None
        try:
            Controller(torch, "C", B_, "cpu")
            raise AssertionError("C без c_const принят")
        except SystemExit:
            pass
    finally:
        ks.D_FIRST, ks.D_SECOND = kd
    # 4. набор init: scores = 0 -> решения 0 (правило ничьей)
    Bi = ks.init_bundle(torch, H)
    assert Bi["bundle_hash"] == ks.init_bundle(torch, H)["bundle_hash"]
    ci = Controller(torch, "A", Bi, "cpu")
    ci.begin(1)
    r = ci.first(dict(roots_n=torch.randn(5, 4, H), iv=torch.rand(5, 4, 2),
                      B=5)).numpy()
    assert (r == 0).all() and (ci.rec["scores1"] == 0).all()
    # 5. проверка журнала: правильный проходит; подмены — отказ
    Lb, Ub = [-1.0] * 6, [2.0] * 6
    bounds = dict(L=Lb, U=Ub)
    for kind in ("A", "C", "N"):
        z, spec = fake_journal(kind, ctx, bounds, rng)
        lab = f"k35_{kind}"
        assert journal_problems(lab, z, spec, ctx, bounds, debug=True) == [], \
            journal_problems(lab, z, spec, ctx, bounds, debug=True)
        for mut in MUTATIONS:
            zz = {k: np.array(v, copy=True) for k, v in z.items()}
            name = mut(zz, lab, kind)
            if name is None:
                continue
            assert journal_problems(lab, zz, spec, ctx, bounds,
                                    debug=True), (kind, name)
    print("самопроверка k35_policy пройдена: признаки контроллера (torch) = "
          "X_first/X_second, решения A = choose; набор селекторов — "
          "сохранение/загрузка, порча — отказ; C = c_const, N — ключевой "
          "RNG без сдвига глобальных, разные элементы; C_fast без сетей; "
          "набор init даёт scores 0; журнал: правильный проходит, подмены "
          "(scores/решение, RNG N, c_const, ID, родитель, partial y, "
          "физическая команда, схват, пустой, чужой снимок, счётчики) — "
          "отказ")
    return 0


# --- синтетический журнал для самопроверок ----------------------------------------------
def fake_journal(kind, ctx, bounds, rng, C=4):
    B = len(ctx["inits"])
    lab = f"k35_{kind}"
    s1 = rng.normal(size=(C, B, 6)).astype(np.float32)
    s2 = rng.normal(size=(C, B, 6)).astype(np.float32)
    ch = np.arange(1, C + 1)
    if kind == "A":
        a = np.stack([ks.decide(s1[i]) for i in range(C)])
        b = np.stack([ks.decide(s2[i]) for i in range(C)])
    elif kind == "C":
        a, b = np.full((C, B), 2), np.full((C, B), 5)
    else:
        a = np.asarray([[n_choice(ctx["task"], i, ctx["env_seed"], k, 1)
                         for i in ctx["inits"]] for k in ch])
        b = np.asarray([[n_choice(ctx["task"], i, ctx["env_seed"], k, 2)
                         for i in ctx["inits"]] for k in ch])
    c = 6 * a + b
    parts = {k: rng.normal(size=(C, B, 20, 7)).astype(np.float32)
             for k in LEVELS}
    y, arm, grip = assemble(parts, bounds["L"], bounds["U"])
    lv = np.concatenate([arm[:, :, :8], np.where(
        y[:, :, :8, 6:7] > 0, 1.0, -1.0)], -1).astype(np.float32)
    X2 = rng.normal(size=(C, B, ks.D_SECOND)).astype(np.float32)
    X2[..., -6:] = np.eye(6, dtype=np.float32)[a]
    z = dict(k15g_chunk=ch, **{f"k15d_level_{lab}": lv},
             k21_ctrl=np.full(C, KIND_ID[kind]), k21_a=a, k21_b=b, k21_c=c,
             k21_scores1=s1, k21_scores2=s2,
             k21_X1=rng.normal(size=(C, B, ks.D_FIRST)).astype(np.float32),
             k21_X2=X2, k21_n_vision=np.ones(C), k21_n_kv=np.ones(C),
             k21_n_text=np.full(C, 24), k21_active=np.tile(tp.ACTIVE, (C, 1)),
             k21_router_q=np.tile(tp.ROUTER_ACTIVE, (C, 1)),
             k21_n_expand=np.full(C, 2), k21_n_heads=np.full(C, 3),
             k21_n_decisions=np.full(C, 2), k21_y=y,
             k21_attn_pad=np.zeros((C, 6)), k21_attn_sumerr=np.zeros((C, 6)),
             k21_attn_finite=np.ones((C, 6)),
             **{f"k21_partial_{k}": v for k, v in parts.items()})
    for k in ("term_roots", "mids", "term_mid_local", "leaves",
              "leaf_parent_local"):
        z[f"k21_{k}"] = TAB[k][c]
    H = 12
    for lv_, n in N_TERM.items():
        z[f"k21_dbg_x_{lv_}"] = rng.normal(size=(1, B, n, H)).astype(
            np.float32)
        z[f"k21_dbg_iv_{lv_}"] = np.zeros((1, B, n, 2), np.float32)
        z[f"k21_dbg_w_{lv_}"] = np.zeros((1, B, 20, n), np.float32)
        z[f"k21_dbg_y_{lv_}"] = parts[lv_][:1].copy()
    z.update(k21_dbg_arm=arm[:1].copy(), k21_dbg_grip=grip[:1].copy(),
             k21_dbg_y_final=y[:1].copy(),
             k21_dbg_roots_n=rng.normal(size=(1, B, 4, H)).astype(np.float32),
             k21_dbg_mids_n=rng.normal(size=(1, B, 4, H)).astype(np.float32))
    spec = dict(kind="K35", controller=kind,
                c_const=17 if kind == "C" else None)
    return z, spec


def _m_scores(z, lab, kind):
    if kind != "A":
        return None
    s = z["k21_scores1"]
    i = int(np.argmax(s[0, 0]))
    s[0, 0, i] = s[0, 0].min() - 1.0     # решение больше не соответствует
    return "scores"


def _m_decision(z, lab, kind):
    z["k21_b"][1, 2] = (z["k21_b"][1, 2] + 1) % 6
    z["k21_c"][1, 2] = 6 * z["k21_a"][1, 2] + z["k21_b"][1, 2]
    for k in ("term_roots", "mids", "term_mid_local", "leaves",
              "leaf_parent_local"):
        z[f"k21_{k}"] = TAB[k][z["k21_c"]]
    return "решение"


def _m_c(z, lab, kind):
    z["k21_c"][0, 0] = (z["k21_c"][0, 0] + 1) % 36
    return "c"


def _m_parent(z, lab, kind):
    z["k21_leaf_parent_local"][0, 0, 0] = 3 - z["k21_leaf_parent_local"][
        0, 0, 0]
    return "родитель"


def _m_ids(z, lab, kind):
    z["k21_leaves"][2, 1, 0] += 1
    return "ID"


def _m_partial(z, lab, kind):
    z["k21_partial_mid"][2, 3, 4, 1] += 0.01
    return "partial"


def _m_action(z, lab, kind):
    z[f"k15d_level_{lab}"][3, 0, 2, 1] += 1e-3
    return "физическая команда"


def _m_grip(z, lab, kind):
    z[f"k15d_level_{lab}"][1, 1, 0, 6] *= -1
    return "схват"


def _m_empty(z, lab, kind):
    for k in list(z):
        if k.startswith("k21_") or k == "k15g_chunk" or k.startswith(
                "k15d_level"):
            z[k] = z[k][:0]
    return "пустой"


def _m_counter(z, lab, kind):
    z["k21_n_expand"][1] = 3
    return "счётчик"


def _m_router(z, lab, kind):
    z["k21_router_q"][0, 5] = 16
    return "router-Q"


def _m_snapshot(z, lab, kind):
    z["k21_dbg_y_leaf"] = z["k21_dbg_y_leaf"] + 0.0
    z["k21_dbg_y_leaf"][0] = np.asarray(z["k21_partial_leaf"])[1]
    return "чужой снимок"


def _m_nan(z, lab, kind):
    z["k21_scores2"][0, 0, 0] = np.nan
    return "NaN"


def _m_onehot(z, lab, kind):
    z["k21_X2"][0, 0, -6:] = np.roll(z["k21_X2"][0, 0, -6:], 1)
    return "one-hot"


def _m_missing_last(z, lab, kind):
    for k in list(z):
        if (k.startswith("k21_") and not k.startswith("k21_dbg")) or \
                k.startswith("k15d_level"):
            z[k] = z[k][:-1]
    return "нет последнего вызова"


MUTATIONS = (_m_scores, _m_decision, _m_c, _m_parent, _m_ids, _m_partial,
             _m_action, _m_grip, _m_empty, _m_counter, _m_router,
             _m_snapshot, _m_nan, _m_onehot, _m_missing_last)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k35_harness.py")
