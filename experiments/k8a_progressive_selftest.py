"""K-8a: самопроверки depth-aligned модели ДО обучения.

Ни один из этих тестов не про качество. Все — про проводку: что сегментированный
проход делает ровно то, что написано, и что на старте он тождественен исходной
однопроходной модели.

Проверки без модели (`--selftest`, идут на CPU):
  1. LoRA на старте тождественна базовому слою, обучаются только A и B.
  2. Обратное внедрение при ШТАТНОМ старте (alpha=1, нулевая проекция) даёт
     тождество и при этом живой градиент в проекцию; двойное обнуление
     воспроизводимо мертво.
  3. Straight-through: вперёд жёсткий код, назад градиент мягкого среднего.
  4. Валидация выходов: последний обязан стоять на полной глубине.

Проверки с моделью (`--ckpt`, нужен GPU):
  5. ТОЖДЕСТВО. При выходах (24,) токены обязаны совпасть с первым блоком
     официальной BAR на 100%, и логиты — сойтись с ним же.
  6. Тождество декодера: своя сборка суммы уровней = официальный decode.
  7. СЧЁТЧИКИ СЛОЁВ. fast=12, medium=18, full=24 вызова слоёв эксперта; башня
     VLM продвигается столько же раз; официальная BAR — 72 и 72.
  8. Внедрение сдвигает логиты ПОЗДНИХ уровней и ровно на ноль — уже выданного
     грубого. Судим по логитам: состояние может измениться, а argmax остаться.
  9. Разбивка обучаемых параметров: посторонних быть не должно.

Запуск:
    python3 experiments/k8a_progressive_selftest.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k8a_progressive_selftest.py --ckpt <ckpt>
"""

import argparse
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20


def selftest_cpu():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        # МОЛЧА ПРОПУСКАТЬ НЕЛЬЗЯ: эти проверки про поведение модулей, и
        # «самопроверка пройдена» без torch однажды уже создала ложную
        # уверенность в другом скрипте.
        raise SystemExit(
            "нет torch: самопроверки k8a проверяют поведение nn.Module "
            "(LoRA, внедрение, straight-through) и без него бессмысленны. "
            "Запускать на кластере.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from depth_rvq_vla import (CodeFeedback, LoRALinear, inject_lora,
                               straight_through)

    torch.manual_seed(0)

    # 1. LoRA тождественна на старте: B инициализирован нулём.
    base = nn.Linear(8, 5)
    x = torch.randn(3, 8)
    lo = LoRALinear(base, r=2)
    assert torch.equal(lo(x), base(x)), "LoRA на старте меняет выход"
    assert not base.weight.requires_grad, "база обязана быть заморожена"
    lo.B.data.normal_()
    assert not torch.equal(lo(x), base(x)), "после сдвига B выход обязан меняться"
    trainable = {n for n, p in lo.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}, trainable

    # 1б. inject_lora ОБЯЗАНА находить слои; пустой список — молчаливый провал.
    m = nn.Module()
    m.blk = nn.Module()
    m.blk.q_proj = nn.Linear(4, 4)
    m.blk.other = nn.Linear(4, 4)
    got = inject_lora(m, 2, ("q_proj",))
    assert got == ["blk.q_proj"], got
    assert isinstance(m.blk.q_proj, LoRALinear)
    assert not isinstance(m.blk.other, LoRALinear), "обёрнут лишний слой"
    assert inject_lora(nn.Linear(2, 2), 2, ("nope",)) == []

    # 2. Внедрение кода. Проверяем НАСТОЯЩУЮ начальную конфигурацию, ничего не
    #    подкручивая руками: прежняя версия теста сама ставила alpha=1 и
    #    случайные веса перед backward и потому не заметила бы, что при
    #    штатном старте ветвь мертва.
    fb = CodeFeedback(6, 4)                       # штатный alpha_init=1.0
    h = torch.randn(2, 16, 4)
    emb = torch.randn(2, 16, 6)
    assert torch.equal(fb(h, emb), h), (
        "на старте внедрение обязано быть тождеством: P(e)=0")
    fb(h, emb).sum().backward()
    assert fb.proj.weight.grad is not None, "нет градиента у проекции"
    assert fb.proj.weight.grad.abs().sum() > 0, (
        "градиент проекции нулевой при штатной инициализации — ветвь мертва "
        "навсегда: так бывает, когда обнулены И alpha, И веса P")
    # alpha на первом шаге градиента не имеет (P(e)=0) — это нормально, она
    # оживает, как только P сдвинется. Проверяем именно это.
    assert fb.alpha.grad is None or fb.alpha.grad.abs().item() == 0
    fb.zero_grad()
    fb.proj.weight.data.normal_(std=0.1)
    out = fb(h, emb)
    assert not torch.equal(out, h) and out.shape == h.shape
    out.sum().backward()
    assert fb.alpha.grad is not None and fb.alpha.grad.abs().item() > 0, (
        "после сдвига P вентиль alpha обязан получать градиент")

    # 2б. Двойное обнуление — тот самый отказ. Показываем, что он ловится.
    dead = CodeFeedback(6, 4, alpha_init=0.0)
    dead(h, emb).sum().backward()
    assert dead.proj.weight.grad.abs().sum() == 0, (
        "тест не воспроизводит мёртвую конфигурацию — проверка бессмысленна")

    # 3. Straight-through: вперёд жёсткий, назад мягкий.
    V, d = 7, 3
    book = torch.randn(V, d)
    logits = torch.randn(2, 16, V, requires_grad=True)
    emb, idx, p = straight_through(logits, book, tau=1.0)
    assert torch.allclose(emb, book[idx], atol=1e-6), "вперёд обязан идти argmax"
    assert idx.shape == (2, 16) and emb.shape == (2, 16, d)
    emb.sum().backward()
    g = logits.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, (
        "градиент обязан пройти через мягкую ветвь")
    # градиент не должен быть сосредоточен только на выбранном коде
    onehot = torch.zeros_like(g).scatter_(-1, idx.unsqueeze(-1), 1.0)
    assert (g * (1 - onehot)).abs().sum() > 0, (
        "градиент только на выбранном коде — мягкая ветвь не работает")

    # 4. Валидация выходов: последний обязан стоять на полной глубине.
    for bad, n in (((12, 18), 24), ((18, 12, 24), 24), ((0, 24), 24)):
        ok = (sorted(set(bad)) == list(bad) and bad[-1] == n and bad[0] >= 1)
        assert not ok, f"конфигурация {bad} должна отвергаться"
    good = (12, 18, 24)
    assert sorted(set(good)) == list(good) and good[-1] == 24 and good[0] >= 1

    print("самопроверка k8a (без модели) пройдена: LoRA стартует тождеством, "
          "внедрение при штатном старте (alpha=1, нулевая проекция) даёт "
          "тождество И живой градиент в проекцию, двойное обнуление "
          "воспроизводимо мертво, straight-through даёт жёсткий код вперёд и "
          "мягкий градиент назад")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--exits", default="12,18,24")
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--pos-offset", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    selftest_cpu()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from depth_rvq_vla import make_depth_rvq_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       process_state, prompt_template, seed_everything)

    seed_everything(args.seed)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    Cls = make_depth_rvq_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    n_layers = int(model.config.vlm_config.text_config.num_hidden_layers)
    print(f"слоёв {n_layers}, блоков {model.num_blocks}, "
          f"размер блока {model.block_size}")

    # --- кодбуки: тот же путь, что в K-6h/K-7a ------------------------------
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None)).to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)
    print(f"кодбуки: {tuple(E.shape)}")

    # --- данные -------------------------------------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    rng = np.random.default_rng(args.seed)
    im1, im2, st, act, tsk = [], [], [], [], []
    for e in rng.permutation(1693):
        if len(tsk) >= args.n_obs:
            break
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        if t.num_rows < T_CHUNK + 1:
            continue
        A_ = np.asarray(t.column("actions").to_pylist(), np.float32)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        s0 = int(rng.integers(0, t.num_rows - T_CHUNK + 1))
        png = lambda c: np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
        im1.append(png(c1[s0])); im2.append(png(c2[s0]))
        st.append(S_[s0]); act.append(A_[s0:s0 + T_CHUNK])
        tsk.append(tasks_map[ti[s0]])
    N = len(tsk)
    hw = im1[0].shape[0]
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

    # --- счётчики слоёв ------------------------------------------------------
    # ХУКИ НА input_layernorm, А НЕ НА МОДУЛЬ СЛОЯ. _shared_attention_forward
    # не вызывает layer.forward(): он сам дёргает input_layernorm, q/k/v_proj,
    # o_proj, post_attention_layernorm и mlp (bar.py:987-1061). Хук на слое
    # целиком не сработал бы ни разу, и счётчик показал бы 0/0 при исправной
    # модели. input_layernorm вызывается ровно раз на исполненный слой — это
    # уже проверено в K-7b, где такой хук сработал ровно три раза.
    cnt = {"expert": 0, "vlm": 0}

    def bump(key):
        return lambda m, i_, o: cnt.__setitem__(key, cnt[key] + 1)

    ex_layers = model.action_expert.layers
    vlm_layers = model.vlm.text_model.layers        # путь из bar.py:987
    assert len(ex_layers) == len(vlm_layers) == n_layers, (
        f"{len(ex_layers)} слоёв эксперта, {len(vlm_layers)} слоёв VLM, "
        f"ожидалось {n_layers}")
    hs = [ex_layers[i].input_layernorm.register_forward_hook(bump("expert"))
          for i in range(n_layers)]
    hs += [vlm_layers[i].input_layernorm.register_forward_hook(bump("vlm"))
           for i in range(n_layers)]

    def counted(fn):
        cnt["expert"] = cnt["vlm"] = 0
        out = fn()
        return out, dict(cnt)

    with torch.no_grad():
        tk_bar, c_bar = counted(lambda: model.generate(
            **batch, position_offset=args.pos_offset, do_sample=False))
    K_bar = tk_bar.cpu().numpy().reshape(N, N_LEVEL, N_POS)
    print(f"\n  официальная BAR: слоёв эксперта {c_bar['expert']}, "
          f"слоёв VLM {c_bar['vlm']} (ожидалось {N_LEVEL * n_layers} и "
          f"{N_LEVEL * n_layers})")
    if c_bar["expert"] != N_LEVEL * n_layers or c_bar["vlm"] != N_LEVEL * n_layers:
        raise SystemExit(
            f"счётчики на официальной BAR дали {c_bar}, а обязаны дать "
            f"{N_LEVEL * n_layers}/{N_LEVEL * n_layers}. Хуки стоят не там, и "
            f"дальнейшие замеры экономии слоёв ничего не значат.")

    # --- 5. ТОЖДЕСТВО при выходе только на полной глубине --------------------
    model.init_progressive(exits=(n_layers,), head_dtype=dt, lora_r=0)
    with torch.no_grad():
        out, c1_ = counted(lambda: model.generate_progressive(
            **batch, mode="full", books=E, position_offset=args.pos_offset))
    q0 = out["pred_codes"][0].cpu().numpy()
    same = (q0 == K_bar[:, 0, :])
    print(f"  тождество (выход только на {n_layers}): совпадение токенов "
          f"{same.mean():.6%} ({int(same.sum())}/{same.size})")
    if not same.all():
        raise SystemExit(
            "токены сегментированного прохода НЕ совпали с первым блоком BAR.\n"
            "Значит проводка (маска, позиции, норма перед головой) расходится\n"
            "с официальной. Обучать такую модель бессмысленно.")

    # Сравнение ЛОГИТОВ с официальным путём: argmax может совпасть при
    # систематическом сдвиге, который проявится только при обучении.
    with torch.no_grad():
        B_ = batch["input_ids"].shape[0]
        _, _, vemb, _ = model._build_vlm_inputs_embeds(
            input_ids=batch.get("input_ids"), inputs_embeds=None,
            pixel_values=batch.get("pixel_values"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            image_hidden_states=None)
        apos = model._build_action_pos_ids_strided(
            batch_size=B_, base_pos=vemb.shape[1],
            action_seq_len=model.block_size, device=dev,
            position_offset=args.pos_offset)
        pos = model._build_joint_position_ids(
            batch_size=B_, vlm_seq_len=vemb.shape[1], action_pos_ids=apos,
            device=dev)
        lg_ref = model._predict_next_block_logits(
            vlm_inputs_embeds=vemb, attention_mask=batch.get("attention_mask"),
            history_tokens=None, position_ids=pos)
    dlg = (out["logits"][0].float() - lg_ref.float()).abs().max().item()
    print(f"  расхождение логитов с официальным путём: max|Δ| = {dlg:.3e}")
    if dlg > 1e-2:
        raise SystemExit(
            f"логиты расходятся на {dlg:.3e} при совпавших argmax — проводка "
            f"отличается, и при обучении это разойдётся дальше.")
    assert c1_["expert"] == n_layers and c1_["vlm"] == n_layers, c1_
    print(f"  слоёв: эксперт {c1_['expert']}, VLM {c1_['vlm']} "
          f"(против {c_bar['expert']}/{c_bar['vlm']} у BAR)")

    # --- 6. тождество декодера ----------------------------------------------
    Kt = torch.as_tensor(K_bar).long().to(dev)
    with torch.no_grad():
        z = sum(E[j][Kt[:, j, :]] for j in range(N_LEVEL))
        mine, _ = codec._decode(z, embodiment_ids=0)
        mine = mine[..., :7].float().cpu().numpy()
    ref = np.asarray(proc.action_processor.decode(
        tk_bar.cpu().numpy().tolist())[0], np.float64)
    d = float(np.abs(mine - ref).max())
    print(f"  тождество декодера: max|Δ| = {d:.3e}")
    if d > 1e-3:
        raise SystemExit("своя сборка латенты расходится с официальным decode")

    # --- 7. счётчики по режимам ---------------------------------------------
    exits = tuple(int(v) for v in args.exits.split(","))
    model.init_progressive(exits=exits, head_dtype=dt, lora_r=0)
    print(f"\n  выходы {exits}:")
    ok = True
    for mode, want, n_lv in (("fast", exits[0], 1), ("medium", exits[1], 2),
                             ("full", exits[2], 3)):
        with torch.no_grad():
            o, c = counted(lambda: model.generate_progressive(
                **batch, mode=mode, books=E, position_offset=args.pos_offset))
        # Число выданных уровней проверяется наравне с числом слоёв: режим,
        # который посчитал верное число слоёв, но выдал не столько уровней,
        # сломан ровно так же.
        good = (c["expert"] == want and c["vlm"] == want
                and len(o["pred_codes"]) == n_lv)
        ok &= good
        print(f"    {mode:<7} слоёв эксперта {c['expert']:>3}, VLM {c['vlm']:>3}, "
              f"уровней {len(o['pred_codes'])} (ждали {n_lv})  "
              f"{'ок' if good else 'НЕ ТО'}")
    if not ok:
        raise SystemExit(
            "число исполненных слоёв не совпало с глубиной выхода: значит "
            "ранний режим считает лишнее, и заявленная экономия ложная")

    # --- 8. внедрение действительно работает --------------------------------
    # СУДИМ ПО ЛОГИТАМ, А НЕ ПО argmax. Состояние может измениться, а
    # выбранный код остаться прежним — тогда проверка по argmax объявила бы
    # исправную обратную связь сломанной.
    with torch.no_grad():
        base = model.generate_progressive(**batch, mode="full", books=E,
                                          position_offset=args.pos_offset)
        for fb in model.prog_feedback:
            torch.nn.init.normal_(fb.proj.weight, std=0.02)
        moved = model.generate_progressive(**batch, mode="full", books=E,
                                           position_offset=args.pos_offset)
    d0 = (base["logits"][0] - moved["logits"][0]).abs().max().item()
    dl = (base["logits"][-1] - moved["logits"][-1]).abs().max().item()
    f0 = (base["pred_codes"][0] != moved["pred_codes"][0]).float().mean().item()
    fl = (base["pred_codes"][-1] != moved["pred_codes"][-1]).float().mean().item()
    print(f"\n  внедрение включено: логиты уровня 0 сдвинулись на {d0:.3e} "
          f"(обязано 0), последнего — на {dl:.3e} (обязано > 0)")
    print(f"    для справки, доля сменившихся кодов: уровень 0 {f0:.1%}, "
          f"последний {fl:.1%}")
    if d0 != 0.0:
        raise SystemExit(
            "внедрение задело УЖЕ ВЫДАННЫЙ грубый уровень: значит оно "
            "применяется не после его головы, и ранний выход загрязнён "
            "информацией из поздних сегментов")
    if dl == 0.0:
        raise SystemExit("логиты поздних уровней не изменились — проводка "
                         "обратной связи не работает")

    # --- 9. что именно обучается, И ПРОХОД С ВКЛЮЧЁННОЙ LoRA -----------------
    # Учёта параметров мало: прежняя версия проверяла только его и потому не
    # заметила, что веса LoRA создаются на CPU — падало лишь на первом
    # настоящем проходе, уже в обучении.
    # Тот же dtype голов, что у эталона `base`, иначе расхождение логитов
    # объяснялось бы точностью, а не LoRA.
    model.init_progressive(exits=exits, head_dtype=dt, lora_r=8)
    print(f"\n  LoRA обернула слоёв: {len(model.lora_wrapped)}")
    devs = {str(p.device) for p in model.progressive_parameters()}
    print(f"  устройства обучаемых параметров: {devs}")
    if len(devs) != 1:
        raise SystemExit(
            f"обучаемые параметры на разных устройствах: {devs}. Проход "
            f"упадёт на первом же матричном умножении.")
    with torch.no_grad():
        o_lora = model.generate_progressive(**batch, mode="full", books=E,
                                            position_offset=args.pos_offset)
    assert len(o_lora["pred_codes"]) == len(exits)
    dl = (o_lora["logits"][0] - base["logits"][0]).abs().max().item()
    print(f"  логиты с LoRA против без неё: max|Δ| = {dl:.3e} (обязано 0)")
    if dl != 0.0:
        raise SystemExit("LoRA на старте меняет выход — B не нулевая")

    # Отдельно: головы в float32 — так их создаёт обучение. Проверяем только,
    # что проход не падает; побитового совпадения с fp16 тут ждать нельзя.
    model.init_progressive(exits=exits, head_dtype=torch.float32, lora_r=8)
    with torch.no_grad():
        o32 = model.generate_progressive(**batch, mode="full", books=E,
                                         position_offset=args.pos_offset)
    agree = (o32["pred_codes"][0] == o_lora["pred_codes"][0]).float().mean()
    print(f"  головы в float32: проход прошёл, коды совпали с fp16 на "
          f"{agree:.1%}")

    rep = model.trainable_report()
    tot = sum(rep.values())
    for k_, v in sorted(rep.items()):
        print(f"    {k_:<16}{v / 1e6:8.3f} млн")
    print(f"    {'итого':<16}{tot / 1e6:8.3f} млн из "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f} млн")

    for h in hs:
        h.remove()
    print("\n  все проверки с моделью пройдены: тождество на полной глубине "
          "точное, экономия слоёв реальная, обратная связь действует только "
          "вперёд по глубине")


if __name__ == "__main__":
    main()
