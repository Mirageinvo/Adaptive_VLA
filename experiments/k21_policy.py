#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: рука flat24 / tree_adapt24 для холодного харнесса k9h.

Интерфейс руки k15f (act(batch, pos_off, autocast, first) -> (a, q0c),
log, meta, model, proc). Модель — исходный BAR из закреплённого снимка
подготовки + K21 нужного режима; веса — best.pt обучения (сверяются
отпечаток весов, prep_id и SHA кода из контракта обучения).
Вход: батч харнесса (тот же процессор, кадры, подсказка), смещение задачи,
состояние руки из обёртки process_state (те же 8 чисел: позиция, axis-angle,
пальцы) -> state11. Выход: полный чанк 20 × 7 в координатах кодека: рука —
ограниченный выход модели, схват — +1 при логите > 0, иначе −1 (обратное
преобразование для среды делает харнесс). Ансамбля, импульсов и выбора
действий нет; исполняются первые 8 команд (харнесс).
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LABELS = ("flat24", "tree_adapt24")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


class K21Log:
    def __init__(self, label):
        self.label = label
        self.reset()

    def reset(self):
        self.rows = collections.defaultdict(list)

    def add(self, **kw):
        for k, v in kw.items():
            self.rows[k].append(np.asarray(v))

    def take(self):
        if not self.rows:
            self.reset()
            return dict(calls=0), {}
        arr = {k: np.stack(v) for k, v in self.rows.items()}
        out = {"k15d_a0": arr["act"], f"k15d_level_{self.label}": arr.pop(
            "act"), "k15g_chunk": arr.pop("chunk")}
        out.update({f"k21_{k}": v for k, v in arr.items()})
        summ = dict(calls=int(len(out["k15g_chunk"])), levels=[self.label])
        self.reset()
        return summ, out


def load_policy(torch, label, ckpt, device):
    """K21 + веса best.pt со сверкой происхождения."""
    import k21_data as d21
    import k21_model as km
    import k21_train as kt
    if label not in LABELS:
        raise SystemExit(f"метка {label!r}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg["mode"] != label:
        raise SystemExit(f"чекпойнт режима {cfg['mode']}, метка {label}")
    prep = json.load(open(d21.REPORT))
    if prep.get("prep_id") != cfg["prep_id"]:
        raise SystemExit("чекпойнт обучен на другой подготовке")
    bad = [f for f, s in cfg["code"].items()
           if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"код модели изменён после обучения: {bad}")
    dev = torch.device(device)
    bar = d21.load_bar(dev, prep["bar_pin"])
    m = bar["model"]
    m.action_expert.float()
    M = km.build(torch)
    L = np.asarray(prep["bounds_codec"]["L"])
    U = np.asarray(prep["bounds_codec"]["U"])
    mod = M["K21"](m, label, L, U, prep["state_norm"]["mean"],
                   prep["state_norm"]["std"], checkpoint=False,
                   seed=kt.SEED).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own:
                raise SystemExit(f"в модели нет {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if kt.state_hash(kt.trainable_state(mod)) != ck["weights_hash"]:
        raise SystemExit("веса после загрузки ≠ отпечатку чекпойнта")
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod, bar, ck, prep


def build_arm(device, basis, label, torch, *, ckpt, plan_sha1):
    import k16_policy as kp
    import k21_model as km
    mod, bar, ck, prep = load_policy(torch, label, ckpt, device)
    dev = torch.device(device)
    L = torch.as_tensor(prep["bounds_codec"]["L"], device=dev)
    U = torch.as_tensor(prep["bounds_codec"]["U"], device=dev)
    log = K21Log(label)
    st = dict(nbuf=0, call=0)

    def act(batch, pos_off, autocast, first):
        if first:
            st["nbuf"] = len(kp.STATE_BUF) - 1
            st["call"] = 0
        if len(kp.STATE_BUF) != st["nbuf"] + 1:
            raise SystemExit("рука k21: нет состояния текущего вызова")
        st["nbuf"] = len(kp.STATE_BUF)
        st["call"] += 1
        state8 = np.asarray(kp.STATE_BUF[-1], np.float64)
        B = state8.shape[0]
        s11 = torch.as_tensor(km.state11(state8), device=dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.no_grad(), autocast:
            out = mod(batch, int(pos_off), s11)
        torch.cuda.synchronize(dev)
        ms = (time.perf_counter() - t0) * 1e3
        arm = out["arm"].float()
        if not (torch.isfinite(arm).all() and torch.isfinite(
                out["grip_logit"]).all()):
            raise SystemExit("рука k21: выход не конечен")
        if not bool(((arm >= L - 1e-4) & (arm <= U + 1e-4)).all()):
            raise SystemExit("рука k21: выход вне пределов")
        g = torch.where(out["grip_logit"].float() > 0, 1.0, -1.0)
        a = torch.cat([arm, g[..., None]], -1).cpu().numpy().astype(
            np.float32)
        near = ((arm - L).abs() < 0.01 * (U - L)) | \
            ((U - arm).abs() < 0.01 * (U - L))
        rec = dict(act=a[:, :8], chunk=st["call"], state=state8.astype(
            np.float32), grip_logit=out["grip_logit"][:, :8].float().cpu()
            .numpy(), saturated=near.float().mean((1, 2)).cpu().numpy(),
            ms=ms)
        if "root" in out:
            rec["root"] = out["root"].float().cpu().numpy()
        log.add(**rec)
        if first:
            print(f"    проверка k21: {label}, вызов на {B} сред "
                  f"{ms:.1f} мс, активные query "
                  f"{mod.counters['active'][::8]}", flush=True)
        return a, np.zeros((B, 16), np.int64)

    meta = dict(
        arm="k21", label=label, plan_sha1=plan_sha1,
        ckpt=os.path.abspath(ckpt), ckpt_sha1=sha(ckpt),
        weights_hash=ck["weights_hash"], best_update=ck["update"],
        prep_id=prep["prep_id"], bar_revision=prep["bar_pin"]["revision"],
        k21_policy_sha1=sha(os.path.abspath(__file__)),
        coeffs=[0.0], control=False, basis_sha1=None, amp_factor=None,
        decodes_per_call=0, layers_per_call=24)
    meta["model_fingerprint"] = hashlib.sha1("|".join(str(meta[k]) for k in (
        "label", "ckpt_sha1", "weights_hash", "prep_id", "k21_policy_sha1",
        "plan_sha1")).encode()).hexdigest()[:12]
    return SimpleNamespace(model=mod.bar, proc=bar["proc"], act=act,
                           log=log, meta=meta, k21=mod)


def selftest():
    lg = K21Log("tree_adapt24")
    for c in range(3):
        lg.add(act=np.zeros((5, 8, 7), np.float32), chunk=c + 1,
               state=np.zeros((5, 8)), grip_logit=np.zeros((5, 8)),
               saturated=np.zeros(5), ms=1.0, root=np.zeros((5, 4, 2)))
    summ, arr = lg.take()
    assert summ["calls"] == 3 and arr["k15g_chunk"].tolist() == [1, 2, 3]
    assert arr["k15d_level_tree_adapt24"].shape == (3, 5, 8, 7)
    assert "k21_root" in arr and lg.take()[0]["calls"] == 0
    print("самопроверка k21_policy пройдена: журнал руки")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("модуль руки; запуск — через k21_harness.py")
