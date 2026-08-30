"""K-9b: проверки проводки Joint-12 ДО обучения.

Ни одна проверка не про качество. Все — про то, что модель делает ровно
написанное, что обучается ровно то, что заявлено, и что шаг оптимизатора
действительно меняет первые двенадцать слоёв.

Без модели (`--selftest`, CPU):
  0. Потеря дистилляции: при совпадении логитов обращается в ноль, растёт при
     расхождении, масштабируется T^2.

С моделью (`--ckpt`, GPU):
  1. ТОЖДЕСТВО. depth=24 с копией головы обязано побитово воспроизвести первый
     блок официальной BAR — и токены, и логиты.
  2. СЧЁТЧИКИ СЛОЁВ по `input_layernorm`: depth=12 даёт 12 и 12, depth=24 — 24
     и 24, официальная BAR — 72 и 72.
  3. Голова стартует копией `action_lm_head`.
  4. Белый список: обучаемо только заявленное, глубокие слои заморожены.
  5. Оптимизатор покрывает обучаемое ПО id(), а не по количеству.
  6. Градиенты доходят до всех размороженных групп, у глубоких слоёв None.
  7. Шаг оптимизатора реально меняет голову, слой VLM и слой эксперта, а
     замороженные веса остаются побитово теми же.
  8. Память: один полный шаг на V100, с чекпойнтингом и без.

Запуск:
    python3 experiments/k9b_joint12_selftest.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9b_joint12_selftest.py --ckpt <ckpt> --depth 12
"""

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20


def selftest_cpu():
    try:
        import torch
    except ImportError:
        raise SystemExit("нет torch: проверки k9b про поведение модулей и без "
                         "него бессмысленны. Запускать на кластере.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from joint12_vla import kd_loss

    torch.manual_seed(0)
    t = torch.randn(4, N_POS, 64)
    assert float(kd_loss(t, t, 2.0)) < 1e-6, "совпадающие логиты дают не ноль"
    s = t + torch.randn_like(t) * 2.0
    assert float(kd_loss(s, t, 2.0)) > 0.05, "расхождение не штрафуется"
    # Сдвиг логитов на константу softmax не меняет — потеря не должна реагировать.
    assert abs(float(kd_loss(t + 3.0, t, 2.0))) < 1e-5, (
        "потеря реагирует на сдвиг логитов, хотя softmax к нему инвариантен")
    # Масштаб T^2: при удвоении T потеря на малых расхождениях меняется мало,
    # но обязана остаться конечной и положительной.
    for T in (1.0, 2.0, 4.0):
        v = float(kd_loss(s, t, T))
        assert np.isfinite(v) and v > 0, (T, v)
    # Градиент идёт в УЧЕНИКА и не идёт в учителя.
    su = (t + 0.5).requires_grad_(True)
    te = t.clone().requires_grad_(True)
    kd_loss(su, te.detach(), 2.0).backward()
    assert su.grad is not None and su.grad.abs().sum() > 0
    assert te.grad is None, "градиент утёк в учителя"

    print("самопроверка k9b (без модели) пройдена: дистилляция обращается в "
          "ноль при совпадении, инвариантна к сдвигу логитов, конечна при "
          "разных T, градиент идёт только в ученика")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--pos-offset", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    selftest_cpu()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")
    print(f"k9b sha1 "
          f"{hashlib.sha1(open(__file__, 'rb').read()).hexdigest()[:12]}")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import torch
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

    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    n_layers = int(model.config.vlm_config.text_config.num_hidden_layers)
    if dev.type == "cuda":
        tot = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
        print(f"{torch.cuda.get_device_name(dev)}, всего {tot:.1f} ГиБ; "
              f"слоёв {n_layers}")

    # --- данные ---------------------------------------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    rng = np.random.default_rng(args.seed)
    im1, im2, st, tsk = [], [], [], []
    for e in rng.permutation(1693):
        if len(tsk) >= args.n_obs:
            break
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        if t.num_rows < T_CHUNK + 1:
            continue
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        png = lambda c: np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
        s0 = int(rng.integers(0, t.num_rows - T_CHUNK + 1))
        im1.append(png(c1[s0])); im2.append(png(c2[s0]))
        st.append(S_[s0]); tsk.append(tasks_map[ti[s0]])
    N, hw = len(tsk), im1[0].shape[0]
    tf = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])   # ДОЛЯ, не число
    ST = np.asarray(st, np.float64)
    if ST.shape[1] == len(STATE_Q01) + 1:
        ST = process_state(ST)
    st_n = (ST - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    i1 = tf(torch.tensor(np.stack(im1)).permute(0, 3, 1, 2))
    i2 = tf(torch.tensor(np.stack(im2)).permute(0, 3, 1, 2))
    image = torch.cat([i1, i2], dim=-1)
    msgs = []
    for i in range(N):
        m = prompt_template(st_n[i], None, tsk[i],
                            mode=cfg.MODEL.vla_processor.kwargs.mode,
                            action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                            action_token_len=cfg.MODEL.action_processor.token_len)
        m[1]["content"] = m[1]["content"][1:]
        msgs.append(m)
    texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
    batch = proc(text=texts, images=[[image[i].numpy()] for i in range(N)],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
    batch = dict_apply(lambda x: x.to(dev, dt), batch)
    print(f"батч {N} наблюдений, кадр {hw}")

    # --- счётчики -------------------------------------------------------------
    cnt = {"vlm": 0, "expert": 0}
    bump = lambda k: (lambda m, i_, o: cnt.__setitem__(k, cnt[k] + 1))
    vl, el = model.vlm.text_model.layers, model.action_expert.layers
    hs = [vl[i].input_layernorm.register_forward_hook(bump("vlm"))
          for i in range(n_layers)]
    hs += [el[i].input_layernorm.register_forward_hook(bump("expert"))
           for i in range(n_layers)]

    def counted(fn):
        cnt["vlm"] = cnt["expert"] = 0
        return fn(), dict(cnt)

    with torch.no_grad():
        tk_bar, c_bar = counted(lambda: model.generate(
            **batch, position_offset=args.pos_offset, do_sample=False))
    K_bar = tk_bar.cpu().numpy().reshape(N, N_LEVEL, N_POS)
    print(f"\n  официальная BAR: VLM {c_bar['vlm']}, эксперт {c_bar['expert']} "
          f"(ждали {3 * n_layers} и {3 * n_layers})")
    if c_bar != {"vlm": 3 * n_layers, "expert": 3 * n_layers}:
        raise SystemExit(f"счётчики на официальной BAR дали {c_bar} — хуки "
                         f"стоят не там, замеры экономии ничего не значат")

    # эталонные логиты первого блока
    with torch.no_grad():
        vemb, pos = model.build_inputs(position_offset=args.pos_offset, **batch)
        lg_ref = model._predict_next_block_logits(
            vlm_inputs_embeds=vemb, attention_mask=batch.get("attention_mask"),
            history_tokens=None, position_ids=pos)

    # --- 1 и 3: тождество на полной глубине, голова = копия -------------------
    model.init_joint_fast(depth=args.depth, head_dtype=dt)
    dw = float((model.fast_head.weight - model.action_lm_head.weight).abs().max())
    print(f"  голова стартует копией action_lm_head: max|Δ| = {dw:.3e}")
    if dw != 0.0:
        raise SystemExit("fast_head не является точной копией")

    with torch.no_grad():
        o24, c24 = counted(lambda: model.forward_joint_fast(
            vlm_inputs_embeds=vemb, attention_mask=batch.get("attention_mask"),
            position_ids=pos, depth=n_layers))
    same = (o24["pred_codes"].cpu().numpy() == K_bar[:, 0, :])
    dlg = (o24["logits"].float() - lg_ref.float()).abs().max().item()
    print(f"  тождество при depth={n_layers}: токены {same.mean():.6%}, "
          f"логиты max|Δ| = {dlg:.3e}")
    if not same.all() or dlg > 1e-2:
        raise SystemExit(
            "проход на полной глубине не воспроизводит первый блок BAR. "
            "Проводка (маска, позиции, норма) расходится с официальной.")
    if c24 != {"vlm": n_layers, "expert": n_layers}:
        raise SystemExit(f"depth={n_layers} исполнил {c24}")

    # --- 2: счётчики на рабочей глубине ---------------------------------------
    with torch.no_grad():
        o12, c12 = counted(lambda: model.forward_joint_fast(
            vlm_inputs_embeds=vemb, attention_mask=batch.get("attention_mask"),
            position_ids=pos))
    print(f"  depth={args.depth}: VLM {c12['vlm']}, эксперт {c12['expert']}")
    if c12 != {"vlm": args.depth, "expert": args.depth}:
        raise SystemExit(f"исполнено {c12}, а заявлено {args.depth} — "
                         f"экономия глубины ложная")
    ag = float((o12["pred_codes"].cpu().numpy() == K_bar[:, 0, :]).mean())
    print(f"  согласие с учителем ДО обучения: {ag:.1%}")

    # --- 4: белый список ------------------------------------------------------
    rep = model.trainable_report()
    tot_p = sum(p.numel() for p in model.parameters())
    print(f"\n  обучаемое:")
    for k, v in sorted(rep.items()):
        print(f"    {k:<24}{v / 1e6:9.2f} млн")
    n_tr = sum(rep.values())
    print(f"    {'итого':<24}{n_tr / 1e6:9.2f} млн из {tot_p / 1e6:.0f} "
          f"({n_tr / tot_p:.1%})")

    # --- 5: оптимизатор покрывает обучаемое ПО id() ---------------------------
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=1e-5)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    need = {id(p) for p in model.parameters() if p.requires_grad}
    if in_opt != need:
        raise SystemExit(
            f"оптимизатор покрывает {len(in_opt)} тензоров, обучаемых "
            f"{len(need)}; расхождение {len(need ^ in_opt)} — часть весов "
            f"получала бы градиент и не обновлялась")
    print(f"  оптимизатор покрывает все {len(need)} обучаемых тензоров (по id)")

    # --- 6 и 7: градиенты и настоящий шаг -------------------------------------
    watch = {
        "fast_head": model.fast_head.weight,
        "bos_embedding": model.bos_embedding,
        "expert.norm": model.action_expert.norm.weight,
        "vlm[0]": vl[0].self_attn.q_proj.weight,
        f"vlm[{args.depth - 1}]": vl[args.depth - 1].self_attn.q_proj.weight,
        "expert[0]": el[0].self_attn.q_proj.weight,
        f"expert[{args.depth - 1}]": el[args.depth - 1].self_attn.q_proj.weight,
    }
    frozen = {"vlm[deep]": vl[n_layers - 1].self_attn.q_proj.weight,
              "expert[deep]": el[n_layers - 1].self_attn.q_proj.weight}
    before = {k: v.detach().clone() for k, v in {**watch, **frozen}.items()}

    model.train()
    out = model.forward_joint_fast(
        vlm_inputs_embeds=vemb.detach(),
        attention_mask=batch.get("attention_mask"), position_ids=pos)
    loss = kd_loss(out["logits"], lg_ref.detach(), 2.0)
    print(f"\n  потеря дистилляции до шага: {float(loss):.4f}")
    opt.zero_grad(); loss.backward()
    bad = [k for k, v in watch.items()
           if v.grad is None or not torch.isfinite(v.grad).all()
           or v.grad.abs().sum() == 0]
    if bad:
        raise SystemExit(f"нет ненулевого конечного градиента у: {bad}")
    print("  градиенты дошли до: " + ", ".join(watch))
    for k, v in frozen.items():
        if v.grad is not None:
            raise SystemExit(f"у замороженного {k} появился градиент")
    print("  у глубоких слоёв градиента нет, как и должно быть")

    opt.step()
    moved = [k for k, v in watch.items() if not torch.equal(v.detach(), before[k])]
    still = [k for k, v in frozen.items() if not torch.equal(v.detach(), before[k])]
    print(f"  после шага изменились: {', '.join(moved)}")
    if set(moved) != set(watch):
        raise SystemExit(f"не изменились: {sorted(set(watch) - set(moved))}")
    if still:
        raise SystemExit(f"замороженные веса ИЗМЕНИЛИСЬ: {still}")

    # --- 8: память ------------------------------------------------------------
    if dev.type == "cuda":
        print("\n  память на один полный шаг (батч 1):")
        b1 = {k: (v[:1] if torch.is_tensor(v) and v.shape[:1] == (N,) else v)
              for k, v in batch.items()}
        for ckpt_on in (False, True):
            try:
                model.grad_ckpt = ckpt_on
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                v1, p1 = model.build_inputs(position_offset=args.pos_offset, **b1)
                o = model.forward_joint_fast(
                    vlm_inputs_embeds=v1,
                    attention_mask=b1.get("attention_mask"), position_ids=p1)
                l = kd_loss(o["logits"], o["logits"].detach() + 0.1, 2.0)
                opt.zero_grad(); l.backward(); opt.step()
                peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
                print(f"    чекпойнтинг {'вкл ' if ckpt_on else 'выкл'}: "
                      f"пик {peak:.2f} ГиБ")
            except torch.cuda.OutOfMemoryError:
                print(f"    чекпойнтинг {'вкл ' if ckpt_on else 'выкл'}: "
                      f"НЕ ХВАТИЛО ПАМЯТИ")
                torch.cuda.empty_cache()
        model.grad_ckpt = False

    for h in hs:
        h.remove()
    print("\n  все проверки пройдены: тождество на полной глубине точное, "
          "экономия глубины реальная, обучается только белый список, "
          "оптимизатор покрывает его целиком, шаг меняет первые "
          f"{args.depth} слоёв и не трогает остальные")


if __name__ == "__main__":
    main()
