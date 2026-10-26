#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin: real-check пары flat24 / parents24 на настоящем BAR.

Каждая архитектура — в ОТДЕЛЬНОМ процессе (--arch), чтобы замер памяти не
искажали остатки другой модели. В процессе: снимок BAR загружается заново,
начальные обучаемые тензоры сохраняются до пробных шагов (для сверки
общих тензоров пары), затем для micro 1/2/4 на одних строках разных задач:
bf16 autocast, checkpointing, forward, потеря, backward, clip 1.0 и ПОЛНЫЙ
шаг AdamW с настоящими LR групп (эксперт 2e-5, LoRA 1e-5, новые 1e-4;
wd 0.01) и проверкой конечности обновлённых параметров. Проверки: счётчики,
конечность, суставы в пределах и схваты в [0, 1], float32-команды, градиенты
по группам (k23_smoke), пик памяти. Сохраняются форма pixel_values и
контактная картинка: склейка камер и результат предобработки.
Итог (родительский процесс): общий micro — наибольший делитель 32,
прошедший у обеих моделей с пиком ≤ MEM_MARGIN доступной в начале памяти;
полный контракт (код, подготовка, BAR, устройство, dtype, micros, версии)
в reports/rt1/realcheck.json. Код 3 при технической проблеме.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LR = dict(expert=2e-5, lora=1e-5, new=1e-4)
WD, CLIP, EFF_BATCH = 0.01, 1.0, 32
MEM_MARGIN = 0.85
OUT = "reports/rt1/realcheck.json"
ARCHS = ("flat24", "parents24")
DEPS = ("rt1_common.py", "rt1_data.py", "rt1_model.py", "rt1_prepare.py",
        "rt1_realcheck.py", "k21_model.py", "k21_tree.py", "k22_model.py",
        "k21_data.py", "k23_train.py", "k23_smoke.py", "k21_train.py")


def sha(p):
    import hashlib
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def pick_rows(D, n):
    tr = D.rows("train")
    task = D.ix["task"][tr]
    tasks = np.unique(task)
    out, rng = [], np.random.default_rng(20261025)
    for k in range(n):
        out.append(int(rng.choice(tr[task == tasks[k % len(tasks)]])))
    return out


def contract(a, D, device_name):
    import torch
    import transformers
    import rt1_common as rc
    return dict(code={f: sha(os.path.join(HERE, f)) for f in DEPS},
                prep_id=D.prep["prep_id"], bar_revision=rc.BAR_REVISION,
                device=device_name, dtype=a.dtype, micros=sorted(a.micros),
                seed=a.seed, torch=torch.__version__,
                transformers=transformers.__version__,
                mem_margin=MEM_MARGIN)


def save_contact(batch, img, path):
    import cv2
    t = np.asarray(img).transpose(1, 2, 0)[..., ::-1]          # RGB -> BGR
    cv2.imwrite(path + "_tile.png", t)
    pv = batch["pixel_values"][0].float().cpu().numpy()       # [n, 3, H, W]
    pv = pv.reshape((-1,) + pv.shape[-3:])
    tiles = []
    for x in pv:
        x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6) * 255
        tiles.append(x.transpose(1, 2, 0)[..., ::-1].astype(np.uint8))
    cv2.imwrite(path + "_processed.png", np.concatenate(tiles, 1))


def run_arch(a):
    import torch
    import k21_train as ktr
    import k23_smoke as sm
    import rt1_common as rc
    import rt1_data as rd
    import rt1_model as rm
    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    free0, total = torch.cuda.mem_get_info(dev)
    D = rd.Data(check_files=False)
    M = rm.build(torch)
    rows = pick_rows(D, max(a.micros))
    bar = rc.load_bar(dev, a.dtype)
    m = bar["model"]
    m.action_expert.float()
    mod = M[a.arch](m, D.L, D.U, checkpoint=True, seed=a.seed).to(dev)
    torch.save(ktr.trainable_state(mod), a.init_out)
    gr = mod.groups()
    gn = sm.groups_named(mod)
    named = [(n, p) for n, p in mod.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=gr[k], lr=LR[k]) for k in LR],
                            weight_decay=WD)
    last = len(m.vlm.text_model.layers) - 1
    want = dict(flat24=[16] * 24, parents24=[4] * 8 + [12] * 8 + [28] * 8)
    Lt = torch.as_tensor(D.L[list(rc.ARM_IDX)], device=dev)
    Ut = torch.as_tensor(D.U[list(rc.ARM_IDX)], device=dev)
    res = dict(arch=a.arch, free_gb_at_start=free0 / 2 ** 30,
               total_gb=total / 2 ** 30, rows=rows, micros={}, problems=[])
    for mb in a.micros:
        r = dict(ok=False)
        try:
            batch, po, st, tgt, val = D.make_batch(rows[:mb], bar, dev, dt)
            r["prefix_len"] = int(batch["input_ids"].shape[1])
            r["pixel_values"] = list(batch["pixel_values"].shape)
            if mb == min(a.micros) and a.contact:
                save_contact(batch, D.sample(rows[0])[0], a.contact)
                res["contact"] = a.contact
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
            t0 = time.time()
            mod.train()
            with torch.autocast("cuda", dtype=dt):
                out = mod(batch, po, st)
            loss, la, lg = rm.loss_fn(torch, out, tgt, val)
            loss.mean().backward()
            gp, none, norms = sm.grad_problems(named, gn, last)
            torch.nn.utils.clip_grad_norm_([p for _n, p in named], CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(dev)
            c = mod.counters
            arm, grip = out["arm"], out["grip"]
            checks = dict(
                counters=c["active"] == want[a.arch] and c["vision"] == 1
                and all(x == 1 for x in c["text"]),
                finite=bool(torch.isfinite(out["action"]).all()
                            and torch.isfinite(loss).all()),
                float32=out["action"].dtype == torch.float32,
                bounds=bool(((arm >= Lt - 1e-4) & (arm <= Ut + 1e-4)).all()
                            and (grip >= 0).all() and (grip <= 1).all()),
                grads=not gp,
                params_finite=all(bool(torch.isfinite(p).all())
                                  for _n, p in named))
            r.update(ok=all(checks.values()), checks=checks,
                     seconds=time.time() - t0,
                     peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                     loss=float(loss.mean()), l1_arm=float(la.mean()),
                     l1_grip=float(lg.mean()), group_grad_norms=norms,
                     no_grad_params=none, grad_problems=gp)
            bad = [k for k, v in checks.items() if not v]
            if bad:
                res["problems"].append(f"{a.arch} micro {mb}: {bad}")
        except torch.cuda.OutOfMemoryError:
            r.update(oom=True)
            opt.zero_grad(set_to_none=True)
        res["micros"][str(mb)] = r
        print(f"  {a.arch} micro {mb}: " + (
            f"пик {r['peak_gb']:.1f} ГБ, {r['seconds']:.2f} с, префикс "
            f"{r.get('prefix_len')}, pixel_values {r.get('pixel_values')}, "
            f"проверки {r['checks']}" if "peak_gb" in r else
            "НЕ ПОМЕСТИЛСЯ (OOM)"), flush=True)
        if r.get("oom"):
            break
    res["device"] = torch.cuda.get_device_name(dev)
    res["contract"] = contract(a, D, res["device"])
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    return 0


def orchestrate(a):
    import torch
    import k23_train as k23t
    prob, per = [], {}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for arch in ARCHS:
            out = os.path.join(td, f"{arch}.json")
            init = os.path.join(td, f"{arch}.pt")
            cmd = [sys.executable, os.path.abspath(__file__), "--arch", arch,
                   "--device", a.device, "--dtype", a.dtype, "--seed",
                   str(a.seed), "--out", out, "--init-out", init,
                   "--micros", *map(str, a.micros)]
            if arch == ARCHS[0]:
                cmd += ["--contact", "reports/rt1/contact"]
            rc_ = subprocess.call(cmd)
            if rc_ != 0 or not os.path.exists(out):
                prob.append(f"{arch}: процесс завершился с кодом {rc_}")
                continue
            per[arch] = json.load(open(out, encoding="utf-8"))
            per[arch]["init"] = torch.load(init, map_location="cpu",
                                           weights_only=False)
        rep = dict(archs={k: {kk: vv for kk, vv in v.items() if kk != "init"}
                          for k, v in per.items()})
        if len(per) == len(ARCHS):
            cp = k23t.init_compare(per["flat24"]["init"],
                                   per["parents24"]["init"])
            prob += k23t.pair_problems(cp)
            rep["pair_compare"] = {k: len(v) for k, v in cp.items()}
            if per["flat24"]["contract"] != dict(
                    per["parents24"]["contract"]):
                prob.append("контракты процессов различаются")
            rep["contract"] = per["flat24"]["contract"]
    for arch, r in per.items():
        prob += r["problems"]
    ok_both = []
    for mb in a.micros:
        good = True
        for arch in ARCHS:
            r = (per.get(arch) or {}).get("micros", {}).get(str(mb), {})
            free = (per.get(arch) or {}).get("free_gb_at_start", 0)
            if not (r.get("ok") and r.get("peak_gb", 1e9) <=
                    MEM_MARGIN * free):
                good = False
        if good and EFF_BATCH % mb == 0:
            ok_both.append(mb)
    common = max(ok_both) if ok_both else None
    rep["common_micro"] = common
    rep["accumulation"] = EFF_BATCH // common if common else None
    if common is None:
        prob.append("ни один microbatch не прошёл у обеих моделей с запасом "
                    "памяти")
    rep["problems"] = prob
    json.dump(rep, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"  пара: {rep.get('pair_compare')}; общий micro {common} "
          f"(накопление {rep['accumulation']}); запас памяти {MEM_MARGIN}")
    print("  " + ("ОТКАЗ: " + "; ".join(prob[:6]) if prob else
                  "real-check RoboTwin пройден") + f"; отчёт {OUT}")
    return 3 if prob else 0


def main():
    ap = argparse.ArgumentParser(description="RoboTwin: real-check пары")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--micros", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=ARCHS, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--init-out", default=None)
    ap.add_argument("--contact", default=None)
    a = ap.parse_args()
    if a.arch:
        return run_arch(a)
    return orchestrate(a)


if __name__ == "__main__":
    sys.exit(main())
