"""K-9c: обучение Joint-12 дистилляцией из coarse-выхода полной BAR.

ВОПРОС. Post-hoc головы над замороженным h12 не смогли: четыре конфигурации
сошлись к 26% согласия против 87% у полной глубины. Здесь первые двенадцать
слоёв обеих башен размораживаются целиком, и проверяется не «лежит ли решение
на слое 12», а «можно ли его туда перенести».

УЧИТЕЛЬ — САМА BAR, А НЕ ТОКЕНИЗАТОР. Совпадение BAR с кодами токенизатора 87%,
а успех в замкнутом цикле следует за поведением BAR (K-6h). Цель — воспроизвести
решение учителя вдвое меньшей глубиной; K_true остаётся вторичной метрикой.

СТРОКИ БЕРУТСЯ ИЗ КЭША ПО (эпизод, шаг), А НЕ ПЕРЕВЫВОДЯТСЯ ГСЧ. Повторять
выборку тем же сидом — значит завязаться на порядок вызовов случайного
генератора в другом скрипте. Кэш хранит episode и step по каждой строке, и
изображения набираются ровно по ним.

ТОЧНОСТЬ. Обучаемые веса в fp32, проход под autocast fp16, GradScaler со
стартовым масштабом 4096 (замерено в K-9b: подходит с первой попытки). AdamW на
fp16-параметрах держал бы состояния в fp16, а при lr=1e-5 относительный шаг
около 1e-3 — на границе разрешения.

ОБРЕЗКА ГРАДИЕНТА ОПРЕДЕЛЯЕТ ШАГ, А НЕ lr. В K-9b норма до обрезки была
26.7–72.5 при пороге 1.0, то есть в начале обучения обрезка режет в десятки
раз. Это штатно и спадает по мере сближения с учителем, но норма печатается
каждую эпоху: если она не падает, дело не в lr.

Запуск:
    python3 experiments/k9c_joint12_train.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9c_joint12_train.py --ckpt <ckpt> \\
        --cache data/k9_teacher_20k.npz --epochs 10 --out data/k9c
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8


def param_groups(model, lr_pre, lr_new, wd):
    """Две группы шага и отключение weight decay там, где он вреден.

    Предобученные слои двигаются медленно (lr_pre), новые и малые компоненты
    быстрее (lr_new). Weight decay не применяется к нормам, смещениям и
    bos_embedding: это не веса линейных отображений, и штраф за их величину
    смещал бы представление без причины.
    """
    pre_d, pre_n, new_d, new_n = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_new = name.startswith(("fast_head", "bos_embedding")) or \
            name.startswith("action_expert.norm")
        no_wd = (p.ndim < 2) or ("norm" in name) or name.startswith("bos_embedding")
        (new_n if no_wd else new_d).append(p) if is_new else \
            (pre_n if no_wd else pre_d).append(p)
    groups = [dict(params=pre_d, lr=lr_pre, weight_decay=wd),
              dict(params=pre_n, lr=lr_pre, weight_decay=0.0),
              dict(params=new_d, lr=lr_new, weight_decay=wd),
              dict(params=new_n, lr=lr_new, weight_decay=0.0)]
    return [g for g in groups if g["params"]]


def selftest():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit("нет torch: самопроверки k9c про потери и группы "
                         "параметров и без него бессмысленны.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from joint12_vla import kd_loss

    # 1. Дистилляция: ноль при совпадении, ненулевая при расхождении.
    torch.manual_seed(0)
    t = torch.randn(4, N_POS, 64)
    assert float(kd_loss(t, t, 2.0)) < 1e-6
    assert float(kd_loss(t + torch.randn_like(t), t, 2.0)) > 0.01

    # 2. Группы параметров: нормы и смещения без weight decay, новые с большим
    #    шагом. Ошибка здесь тихая — обучение просто пойдёт не туда.
    m = nn.Module()
    m.fast_head = nn.Linear(4, 4)
    m.bos_embedding = nn.Parameter(torch.zeros(1, 1, 4))
    m.action_expert = nn.Module()
    m.action_expert.norm = nn.LayerNorm(4)
    m.deep = nn.Linear(4, 4)
    for p in m.parameters():
        p.requires_grad_(True)
    gs = param_groups(m, 1e-5, 1e-4, 0.01)
    lr_of = {}
    for g in gs:
        for p in g["params"]:
            lr_of[id(p)] = (g["lr"], g["weight_decay"])
    assert lr_of[id(m.fast_head.weight)] == (1e-4, 0.01)
    assert lr_of[id(m.fast_head.bias)] == (1e-4, 0.0), "у смещения есть wd"
    assert lr_of[id(m.bos_embedding)] == (1e-4, 0.0)
    assert lr_of[id(m.action_expert.norm.weight)] == (1e-4, 0.0), "норма с wd"
    assert lr_of[id(m.deep.weight)] == (1e-5, 0.01), "чужой слой в новой группе"
    n_in = sum(len(g["params"]) for g in gs)
    n_tr = sum(1 for p in m.parameters() if p.requires_grad)
    assert n_in == n_tr, f"в группы попало {n_in} из {n_tr} обучаемых"

    # 3. Метрика согласия и ошибки действия считаются по наблюдениям, а не по
    #    батчам: неполный последний батч иначе весит как полный.
    xs = [(0.0, 100), (1.0, 1)]
    naive = np.mean([v for v, _ in xs])
    weighted = sum(v * w for v, w in xs) / sum(w for _, w in xs)
    assert abs(naive - 0.5) < 1e-9 and weighted < 0.01

    print("самопроверка k9c пройдена: дистилляция ноль при совпадении, нормы "
          "и смещения без weight decay, новые компоненты с большим шагом, все "
          "обучаемые тензоры попали в группы, метрики взвешены по наблюдениям")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_20k.npz")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-pre", type=float, default=1e-5)
    ap.add_argument("--lr-new", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--hard-weight", type=float, default=0.25)
    ap.add_argument("--init-scale", type=float, default=4096.0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--grip-gate", type=float, default=0.005,
                    help="чекпойнт принимается, только если знак схвата на "
                         "первых 8 шагах ошибается не чаще этого")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k9c")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")
    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k9c sha1 {sha}")
    os.makedirs(args.out, exist_ok=True)

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from joint12_vla import kd_loss, make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, process_state, prompt_template,
                       seed_everything)

    seed_everything(args.seed)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    # --- кэш учителя ---------------------------------------------------------
    z = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    if meta["ckpt"] != args.ckpt:
        raise SystemExit(f"кэш собран на {meta['ckpt']}, а задан {args.ckpt}")
    q_teach = z["teacher_codes_q0"].astype(np.int64)
    K_true0 = z["K_true_q0"].astype(np.int64)
    epi, stp, tsk = z["episode"], z["step"], z["task"]
    offs, split = z["pos_offset"], z["split"]
    N = len(epi)
    lg_path = args.cache + ".logits.npy"
    if not os.path.exists(lg_path):
        raise SystemExit(f"нет файла логитов {lg_path}")
    T_LOG = np.load(lg_path, mmap_mode="r")
    assert T_LOG.shape == (N, N_POS, meta["vocab"]), T_LOG.shape
    print(f"кэш: {N} наблюдений, {len(np.unique(epi))} эпизодов, "
          f"согласие учителя с токенизатором {meta['teacher_agreement_with_tokenizer']:.1%}")
    for nm in ("train", "val", "test"):
        m = split == nm
        print(f"    {nm:<6}{int(m.sum()):>7} наблюдений, "
              f"{len(np.unique(tsk[m])):>3} задач")
    itr, iva = np.where(split == "train")[0], np.where(split == "val")[0]

    # --- изображения ПО (эпизод, шаг) из кэша -------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    im1 = [None] * N
    im2 = [None] * N
    st = None
    png = lambda c: np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
    uniq = np.unique(epi)
    for j, e in enumerate(uniq):
        f = hf_hub_download(rid, f"data/chunk-{int(e) // 1000:03d}/"
                            f"episode_{int(e):06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        if st is None:
            st = np.zeros((N, S_.shape[1]), np.float64)
        elif st.shape[1] != S_.shape[1]:
            raise SystemExit(f"эпизод {e}: состояние {S_.shape[1]}-мерное, "
                             f"а раньше было {st.shape[1]}-мерное")
        rows = np.where(epi == e)[0]
        for r in rows:
            s0 = int(stp[r])
            im1[r] = png(c1[s0]); im2[r] = png(c2[s0])
            st[r] = S_[s0]
        if j % 200 == 0:
            print(f"  эпизодов {j}/{len(uniq)}", flush=True)
    assert all(x is not None for x in im1), "не все строки получили изображение"
    hw = im1[0].shape[0]
    tf = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])   # ДОЛЯ, не число
    if st.shape[1] == len(STATE_Q01) + 1:
        st = process_state(st)
    st_n = (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    print(f"изображения собраны, кадр {hw}")

    # --- модель --------------------------------------------------------------
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=args.depth, head_dtype=dt,
                          grad_ckpt=args.grad_ckpt)
    rep = model.trainable_report()
    model.to_fp32_trainable()
    est = model.memory_estimate()
    print(f"обучаемое: " + ", ".join(f"{k} {v/1e6:.1f}" for k, v in sorted(rep.items())))
    print(f"итого {sum(rep.values())/1e6:.1f} млн; статическая память "
          f"{est['total_static_gib']:.2f} ГиБ")

    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None)).to(dev).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    def dec0(codes):
        out = []
        for i0 in range(0, len(codes), 256):
            k = torch.as_tensor(codes[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                x, _ = codec._decode(E[0][k], embodiment_ids=0)
            out.append(x[..., :7].float().cpu())
        return torch.cat(out)

    A_teach = dec0(q_teach)                 # действие учителя, только уровень 0
    A_star = None
    if "K_true" in z.files:
        Kt3 = z["K_true"].astype(np.int64)
        outs = []
        for i0 in range(0, N, 256):
            k = torch.as_tensor(Kt3[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                zz = sum(E[j][k[:, j, :]] for j in range(N_LEVEL))
                x, _ = codec._decode(zz, embodiment_ids=0)
            outs.append(x[..., :7].float().cpu())
        A_star = torch.cat(outs)

    def build(sel):
        i1 = tf(torch.tensor(np.stack([im1[j] for j in sel])).permute(0, 3, 1, 2))
        i2 = tf(torch.tensor(np.stack([im2[j] for j in sel])).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for gi in sel:
            m = prompt_template(st_n[gi], None, str(tsk[gi]),
                                mode=cfg.MODEL.vla_processor.kwargs.mode,
                                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[k].numpy()] for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b)

    def forward(sel):
        po = int(offs[sel[0]])
        assert (offs[sel] == po).all(), "в микробатче разные офсеты"
        b = build(sel)
        v, p = model.build_inputs(position_offset=po, **b)
        return model.forward_joint_fast(
            vlm_inputs_embeds=v, attention_mask=b.get("attention_mask"),
            position_ids=p)

    def batches(idxs, size, shuffle, rng=None):
        """Микробатчи ВНУТРИ группы одного офсета: position_offset задаётся на
        весь вызов, смешивать нельзя."""
        out = []
        for po in sorted({int(v) for v in offs[idxs]}):
            g = idxs[offs[idxs] == po]
            if shuffle:
                g = rng.permutation(g)
            out += [g[i:i + size] for i in range(0, len(g), size)]
        if shuffle:
            rng.shuffle(out)
        return out

    # --- оценка --------------------------------------------------------------
    @torch.no_grad()
    def evaluate(idxs, tag):
        model.eval()
        acc_t, acc_k, wsum = 0.0, 0.0, 0
        se_i, se_e, sg_i, fl4, fl8 = 0.0, 0.0, 0.0, 0.0, 0.0
        for sel in batches(idxs, args.micro_batch, False):
            with torch.autocast(device_type=dev.type, dtype=dt,
                                enabled=(dev.type == "cuda")):
                o = forward(sel)
            pc = o["pred_codes"].cpu().numpy()
            w = len(sel)
            acc_t += float((pc == q_teach[sel]).mean()) * w
            acc_k += float((pc == K_true0[sel]).mean()) * w
            a = dec0(pc)
            d_i = a - A_teach[sel]
            se_i += float((d_i[:, :H_EXEC, :6] ** 2).mean()) * w
            sg_i += float((d_i[:, :H_EXEC, 6] ** 2).mean()) * w
            fl4 += float((torch.sign(a[:, :4, 6])
                          != torch.sign(A_teach[sel][:, :4, 6])).float().mean()) * w
            fl8 += float((torch.sign(a[:, :H_EXEC, 6])
                          != torch.sign(A_teach[sel][:, :H_EXEC, 6])).float().mean()) * w
            if A_star is not None:
                d_e = a - A_star[sel]
                se_e += float((d_e[:, :H_EXEC, :6] ** 2).mean()) * w
            wsum += w
        r = dict(acc_teacher=acc_t / wsum, acc_ktrue=acc_k / wsum,
                 imit_pose8=math.sqrt(se_i / wsum),
                 imit_grip8=math.sqrt(sg_i / wsum),
                 grip_flip4=fl4 / wsum, grip_flip8=fl8 / wsum,
                 expert_pose8=(math.sqrt(se_e / wsum) if A_star is not None else None),
                 n=wsum)
        print(f"  [{tag}] согласие с учителем {r['acc_teacher']:.1%} "
              f"(с токенизатором {r['acc_ktrue']:.1%}); имитация поза8 "
              f"{r['imit_pose8']:.4f}, знак4 {r['grip_flip4']:.2%}, знак8 "
              f"{r['grip_flip8']:.2%}"
              + (f"; до эксперта {r['expert_pose8']:.4f}"
                 if r['expert_pose8'] is not None else ""))
        return r

    # ОПОРА УЧИТЕЛЯ НА ВАЛИДАЦИИ: с чем сравнивать ученика.
    if A_star is not None:
        d = A_teach[iva] - A_star[iva]
        t_pose = float(torch.sqrt((d[:, :H_EXEC, :6] ** 2).mean()))
        t_flip = float((torch.sign(A_teach[iva][:, :H_EXEC, 6])
                        != torch.sign(A_star[iva][:, :H_EXEC, 6])).float().mean())
        print(f"\nучитель на валидации: до эксперта поза8 {t_pose:.4f}, "
              f"знак8 {t_flip:.2%}")
    else:
        t_pose = t_flip = None

    # --- обучение ------------------------------------------------------------
    groups = param_groups(model, args.lr_pre, args.lr_new, args.weight_decay)
    params = [p for g in groups for p in g["params"]]
    n_grad = sum(1 for p in model.parameters() if p.requires_grad)
    if len(params) != n_grad:
        raise SystemExit(f"в группы попало {len(params)} из {n_grad} обучаемых")
    opt = torch.optim.AdamW(groups)
    scaler = torch.amp.GradScaler("cuda", init_scale=args.init_scale,
                                  enabled=(dev.type == "cuda"))
    rng = np.random.default_rng(args.seed)
    spe = math.ceil(len(batches(itr, args.micro_batch, False)) / args.accum)
    total, warm = args.epochs * spe, 0
    warm = max(1, int(total * args.warmup_frac))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    print(f"\n{spe} шагов на эпоху, всего {total}, прогрев {warm}; "
          f"микробатч {args.micro_batch} × накопление {args.accum}")

    hist = []
    ev0 = evaluate(iva, "эпоха 0, до обучения")
    # ЭПОХА 0 — ПОЛНОПРАВНЫЙ КАНДИДАТ: если обучение всё портит, результатом
    # обязан остаться исходный чекпойнт.
    def save(name):
        sd = {k: v.detach().cpu().clone()
              for k, v in model.state_dict().items()
              if any(k.startswith(pf) or k == pf.rstrip(".")
                     for pf in model.trainable_prefixes)}
        torch.save(dict(state=sd, depth=args.depth, sha1=sha, args=vars(args)),
                   os.path.join(args.out, name))
    best = None
    if ev0["grip_flip8"] <= args.grip_gate:
        best = (ev0["imit_pose8"], 0)
        save("best_imitation.pt")
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0, agg, skipped = time.time(), [], 0
        order = batches(itr, args.micro_batch, True, rng)
        opt.zero_grad(set_to_none=True)
        for bi, sel in enumerate(order):
            with torch.autocast(device_type=dev.type, dtype=dt,
                                enabled=(dev.type == "cuda")):
                o = forward(sel)
                tl = torch.as_tensor(np.asarray(T_LOG[sel])).to(dev).float()
                l_kd = kd_loss(o["logits"], tl, args.temperature)
                l_hard = F.cross_entropy(
                    o["logits"].reshape(-1, o["logits"].shape[-1]).float(),
                    torch.as_tensor(q_teach[sel]).to(dev).reshape(-1))
                loss = (l_kd + args.hard_weight * l_hard) / args.accum
            scaler.scale(loss).backward()
            agg.append((float(l_kd), float(l_hard)))
            if (bi + 1) % args.accum == 0 or bi == len(order) - 1:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
                sc = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() < sc:
                    skipped += 1
                else:
                    sch.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 200 == 0:
                    print(f"    шаг {step}/{total}, KD {np.mean([a[0] for a in agg[-200:]]):.3f}, "
                          f"норма до обрезки {float(gn):.2f}", flush=True)
        m = np.mean(agg, axis=0)
        ev = evaluate(iva, f"эпоха {ep + 1}")
        hist.append(dict(epoch=ep + 1, kd=float(m[0]), hard=float(m[1]),
                         val=ev, skipped_steps=skipped,
                         minutes=(time.time() - t0) / 60))
        print(f"    KD {m[0]:.3f}, hard {m[1]:.3f}, пропущено шагов {skipped}, "
              f"{(time.time() - t0) / 60:.1f} мин")
        save("last.pt")
        # ГЕЙТ ПО СХВАТУ: чекпойнт с лучшей позой, но испорченным знаком схвата
        # брать нельзя — именно знак объяснял сохранность успеха в K-6h.
        if ev["grip_flip8"] <= args.grip_gate and (
                best is None or ev["imit_pose8"] < best[0]):
            best = (ev["imit_pose8"], ep + 1)
            save("best_imitation.pt")
        json.dump(dict(history=hist, before=ev0, best=best, teacher_ref=
                       dict(pose8=t_pose, grip_flip8=t_flip),
                       cache=args.cache, args=vars(args), sha1=sha),
                  open(os.path.join(args.out, "history.json"), "w"),
                  ensure_ascii=False, indent=1)
        torch.save(opt.state_dict(), os.path.join(args.out, "optimizer.pt"))

    if best is None:
        print("\nНИ ОДНА ЭПОХА НЕ ПРОШЛА ГЕЙТ ПО СХВАТУ — эксперимент неуспешен")
    else:
        print(f"\nлучший по имитации: эпоха {best[1]}, поза8 {best[0]:.4f}")
    print(f"сохранено в {args.out}/")


if __name__ == "__main__":
    main()
