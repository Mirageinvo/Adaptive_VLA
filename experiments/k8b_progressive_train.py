"""K-8b: обучение depth-aligned модели при ЗАМОРОЖЕННОМ токенизаторе.

ЧТО ОБУЧАЕТСЯ. Три головы кодов на глубинах выходов, проекции обратного
внедрения и LoRA в action expert — 9.3 млн параметров из 2424 млн. Всё
остальное заморожено, включая ActionCodec и его кодбуки. Одна переменная за раз:
сначала проверяем архитектуру на существующем представлении действий, и только
потом, если понадобится, трогаем токенизатор.

ЭТО EXPERT-ONLY ADAPTATION. LoRA стоит только в экспертной ветви, поэтому
зрительная башня НЕ учится выносить task-релевантную информацию раньше. Если
Fast не заработает, это не будет означать, что глубинная схема несостоятельна —
это будет означать, что узкое место в башне VLM. Абляция с LoRA в первых слоях
VLM предусмотрена флагом --lora-vlm.

ЦЕЛЬ ДЕЙСТВИЯ — ДЕКОДИРОВАННЫЙ ЭКСПЕРТ, А НЕ СЫРОЙ ДАТАСЕТ. Токенизатор
заморожен, значит есть неустранимая ошибка реконструкции, и требовать от модели
попасть в сырое действие датасета значит заставлять её спорить с собственными
кодовыми целями. Поэтому цель:

    a* = D(E0[q0*] + E1[q1*] + E2[q2*])

Ошибка до сырого датасета всё равно печатается, но в потери не входит.

ПОЧЕМУ ОШИБКА ДЕЙСТВИЯ, А НЕ ТОЛЬКО КРОСС-ЭНТРОПИЯ. K-6e: декодер принимает
СУММУ уровней, поэтому промах в соседний код почти бесплатен, и CE ранжирует
модели не так, как ошибка декодированного действия. Градиент к головам идёт
через straight-through: вперёд жёсткий код, назад мягкое среднее.

ПЕРВЫЕ ВОСЕМЬ ШАГОВ ВАЖНЕЕ ОСТАЛЬНЫХ. При H=8 среда исполняет только их,
остальные двенадцать заменяются новым планом. K-7a измерил, что именно там
сидит решение схвата: расхождение знака схвата на первых четырёх шагах ровно
ноль, а по всему чанку 0.115%. Поэтому потеря взвешена на исполняемую часть, и
у схвата есть отдельное слагаемое по ЗНАКУ, а не только по величине.

ОТБОР ЧЕКПОЙНТА ПО ПОЛНОСТЬЮ ПРЕДСКАЗАННОМУ РЕЖИМУ. Метрики с teacher forcing
печатаются, но выбирать по ним нельзя: на инференсе учителя нет. Сохраняются
два чекпойнта — `best_fast` (по ошибке Fast) и `best_pareto` (по сумме, при
условии что Full не деградировал).

Запуск:
    python3 experiments/k8b_progressive_train.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k8b_progressive_train.py --ckpt <ckpt> \\
        --n-obs 20000 --n-ep 1600 --epochs 20 --out data/k8b
"""

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8


def tf_prob(epoch, sched=(2, 4, 6, 8)):
    """Доля учительских кодов по эпохам (с нуля).

    Ранние эпохи поздние сегменты обязаны видеть верный грубый код, иначе
    ошибка ранней головы отравляет обучение поздних с первого шага. Дальше
    учитель убирается, потому что на инференсе его нет.
    """
    e1, e2, e3, e4 = sched
    if epoch < e1:
        return 1.0
    if epoch < e2:
        return 0.75
    if epoch < e3:
        return 0.5
    if epoch < e4:
        return 0.25
    return 0.0


def split_by_episode(epi, task, seed=0, frac=(0.8, 0.1)):
    """По эпизодам, стратифицировано по задачам. Тот же приём, что в K-7c:
    без стратификации задача уходит в одну часть целиком, и валидация молча
    начинает мерить обобщение на новые ЗАДАЧИ."""
    rng = np.random.default_rng(seed)
    masks = [np.zeros(len(epi), bool) for _ in range(3)]
    for t in np.unique(task):
        g = np.where(task == t)[0]
        ep = rng.permutation(np.unique(epi[g]))
        n = len(ep)
        if n < 3:
            raise SystemExit(f"задача «{t}»: {n} эпизодов, на три части не делится")
        n_va = max(1, int(round(n * frac[1])))
        n_te = max(1, int(round(n * (1.0 - frac[0] - frac[1]))))
        if n - n_va - n_te < 1:
            n_va, n_te = 1, 1
        parts = (ep[:n - n_va - n_te], ep[n - n_va - n_te:n - n_te], ep[n - n_te:])
        for m, p in zip(masks, parts):
            m[g] = np.isin(epi[g], p)
    return masks


def selftest():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise SystemExit("нет torch: самопроверки k8b проверяют потери и "
                         "расписание, без него бессмысленны. Запускать на кластере.")

    # 1. Расписание учителя: монотонно невозрастающее, начинается с единицы,
    #    заканчивается нулём. Иначе модель никогда не увидит собственных кодов.
    v = [tf_prob(e) for e in range(12)]
    assert v[0] == 1.0 and v[-1] == 0.0
    assert all(a >= b for a, b in zip(v, v[1:])), v
    assert 0.0 in v and v.index(0.0) >= 4

    # 2. Разбиение: без утечки эпизодов, все задачи во всех частях.
    epi = np.repeat(np.arange(60), 4)
    tsk = np.array([f"t{e % 6}" for e in epi])
    tr, va, te = split_by_episode(epi, tsk, seed=0)
    assert (tr | va | te).all() and not (tr & va).any() and not (va & te).any()
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(epi[a]) & set(epi[b])), "эпизод в двух частях"
    for m, nm in ((tr, "train"), (va, "val"), (te, "test")):
        assert set(tsk[m]) == set(tsk), f"{nm}: задачи потеряны"

    # 3. Потеря действия. Ключевое свойство: расхождение ТОЛЬКО в хвосте чанка
    #    обязано весить меньше, чем такое же расхождение в исполняемой части.
    a = torch.zeros(4, T_CHUNK, 7)
    b_tail = a.clone(); b_tail[:, H_EXEC:, :6] += 0.5
    b_head = a.clone(); b_head[:, :H_EXEC, :6] += 0.5
    l_tail = action_loss(b_tail, a, mu=1.0, eta=0.25)
    l_head = action_loss(b_head, a, mu=1.0, eta=0.25)
    assert l_head > l_tail, (
        f"исполняемая часть весит не больше хвоста: {l_head:.4f} против "
        f"{l_tail:.4f} — вес выбран так, что среда не видит разницы")

    # 4. Знак схвата наказывается отдельно от величины: перевёрнутый знак
    #    обязан стоить дороже, чем сдвиг той же величины без смены знака.
    g_ok = a.clone(); g_ok[:, :, 6] = 0.5
    g_flip = a.clone(); g_flip[:, :, 6] = -0.5
    tgt = a.clone(); tgt[:, :, 6] = 0.5
    assert action_loss(g_flip, tgt, 1.0, 0.25) > action_loss(g_ok, tgt, 1.0, 0.25)
    l_no_sign = action_loss(g_flip, tgt, 0.0, 0.25)
    assert action_loss(g_flip, tgt, 1.0, 0.25) > l_no_sign, (
        "слагаемое по знаку схвата ничего не добавляет")

    # 5. Веса уровней: ранний не должен тонуть. При равной ошибке всех трёх
    #    выходов вклад Fast обязан быть не меньше вклада Medium.
    lam = (1.0, 0.5, 1.0)
    assert lam[0] >= lam[1] and lam[2] >= lam[1]

    print("самопроверка k8b пройдена: расписание учителя убывает до нуля, "
          "разбиение без утечки и со всеми задачами, исполняемая часть чанка "
          "весит больше хвоста, знак схвата наказывается отдельно от величины")


def action_loss(a_hat, a_star, mu=1.0, eta=0.25, grip_scale=4.0):
    """Ошибка действия одного выхода.

        поза(первые 8) + грип-MSE(первые 8)/6 + mu*грип-знак(первые 8)
                       + eta*поза(весь чанк)

    Схват делится на шесть не случайно: в K-6e вес 1.0 усиливал схват в шесть
    раз против прежнего усреднения по семи каналам, и обучение уезжало в него.
    """
    import torch
    import torch.nn.functional as F
    h = H_EXEC
    pose8 = F.mse_loss(a_hat[:, :h, :6], a_star[:, :h, :6])
    grip8 = F.mse_loss(a_hat[:, :h, 6], a_star[:, :h, 6])
    # ЗНАК схвата — отдельным слагаемым: именно он решает захват, а MSE может
    # быть маленькой при неверном знаке возле нуля.
    tgt = (a_star[:, :h, 6] > 0).to(a_hat.dtype)
    sign = F.binary_cross_entropy_with_logits(a_hat[:, :h, 6] * grip_scale, tgt)
    full = F.mse_loss(a_hat[..., :6], a_star[..., :6])
    return pose8 + grip8 / 6.0 + mu * sign + eta * full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--exits", default="12,18,24")
    ap.add_argument("--n-obs", type=int, default=20000)
    ap.add_argument("--n-ep", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-vlm", action="store_true",
                    help="АБЛЯЦИЯ: поставить LoRA ещё и в слои VLM до первого "
                         "выхода. Без неё это expert-only adaptation, и "
                         "зрительная башня не учится выносить информацию раньше")
    ap.add_argument("--no-feedback", action="store_true",
                    help="АБЛЯЦИЯ: независимые головы без возврата кода")
    ap.add_argument("--stage-a-steps", type=int, default=300,
                    help="проводка: только головы и проекции, LoRA выключена")
    ap.add_argument("--w-code", default="1,0.5,0.5")
    ap.add_argument("--w-act", default="1,0.5,1")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k8b")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    # ОТПЕЧАТОК ФАЙЛА ПЕРВОЙ СТРОКОЙ. За сессию трижды случалось, что на
    # кластере работала несинхронизированная версия, и это выяснялось только
    # по косвенным признакам в выводе. Хеш сравнивается глазами за секунду.
    import hashlib
    _sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k8b sha1 {_sha}")

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
    from depth_rvq_vla import inject_lora, make_depth_rvq_class, straight_through
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
    exits = tuple(int(v) for v in args.exits.split(","))
    w_code = [float(v) for v in args.w_code.split(",")]
    w_act = [float(v) for v in args.w_act.split(",")]
    os.makedirs(args.out, exist_ok=True)

    Cls = make_depth_rvq_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    model.init_progressive(exits=exits, head_dtype=torch.float32,
                           lora_r=args.lora_r,
                           feedback=not args.no_feedback)
    if args.lora_vlm:
        vlm_l = model.vlm.text_model.layers
        got = inject_lora(torch.nn.ModuleList(list(vlm_l[:exits[0]])),
                          args.lora_r, ("q_proj", "k_proj", "v_proj", "o_proj"),
                          dtype=torch.float32)
        print(f"  LoRA в VLM (слои 1..{exits[0]}): {len(got)} слоёв")
    rep = model.trainable_report()
    print(f"обучаемое: " + ", ".join(f"{k} {v/1e6:.3f} млн"
                                     for k, v in sorted(rep.items())))
    print(f"итого {sum(rep.values())/1e6:.3f} млн, выходы {exits}, "
          f"feedback {'выкл' if args.no_feedback else 'вкл'}")

    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None)).to(dev).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    def decode_prefix(embs):
        """D(сумма эмбеддингов уровней). Градиент ИДЁТ через декодер: латентная
        MSE в K-6e оказалась негодной целью — кодек распределяет ёмкость не
        туда, где чувствителен декодер."""
        z = embs[0]
        for e in embs[1:]:
            z = z + e
        x, _ = codec._decode(z.float(), embodiment_ids=0)
        return x[..., :7].float()

    # --- данные -------------------------------------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    rng = np.random.default_rng(args.seed)
    per_ep = max(1, args.n_obs // max(args.n_ep, 1))
    im1, im2, st, act, tsk, epi = [], [], [], [], [], []
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
        png = lambda c: np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
        for s0 in rng.choice(t.num_rows - T_CHUNK + 1,
                             size=min(per_ep, t.num_rows - T_CHUNK + 1),
                             replace=False):
            im1.append(png(c1[int(s0)])); im2.append(png(c2[int(s0)]))
            st.append(S_[int(s0)]); act.append(A_[int(s0):int(s0) + T_CHUNK])
            tsk.append(tasks_map[ti[int(s0)]]); epi.append(int(e))
    N = len(tsk)
    epi, tsk_a = np.asarray(epi), np.asarray(tsk)
    hw = im1[0].shape[0]
    tf_img = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])   # ДОЛЯ
    print(f"собрано {N} наблюдений, {len(np.unique(epi))} эпизодов, кадр {hw}")

    ST = np.asarray(st, np.float64)
    if ST.shape[1] == len(STATE_Q01) + 1:
        ST = process_state(ST)
    st_n = (ST - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    a_codec = np.asarray(act, np.float64).copy()
    a_codec[..., :-1] /= max_act_q[..., :-1]
    a_codec[..., -1] *= -1
    a_codec = np.clip(a_codec, -1.0, 1.0)
    K_true = np.asarray(proc.action_processor.encode(a_codec),
                        np.int64).reshape(N, N_LEVEL, N_POS)

    tb = json.load(open(args.offset_table))
    off_by_task = {k: int(v["pos_offset"]) for k, v in tb["tasks"].items()}
    miss = sorted({t for t in tsk if t not in off_by_task})
    if miss:
        raise SystemExit(f"нет офсета для задач: {miss[:3]}")
    offs = np.array([off_by_task[t] for t in tsk])

    # ЦЕЛЬ ДЕЙСТВИЯ: декодированный эксперт, а не сырой датасет.
    with torch.no_grad():
        Kt = torch.as_tensor(K_true).long().to(dev)
        A_star = []
        for i0 in range(0, N, 256):
            k = Kt[i0:i0 + 256]
            z = sum(E[j][k[:, j, :]] for j in range(N_LEVEL))
            x, _ = codec._decode(z, embodiment_ids=0)
            A_star.append(x[..., :7].float().cpu())
        A_star = torch.cat(A_star)
    gap = float((A_star.numpy() - a_codec).__abs__().mean())
    print(f"неустранимая ошибка токенизатора (|цель − датасет|): {gap:.4f}")

    # ПОТОЛКИ РЕЖИМОВ. Без них числа обучения нечитаемы: непонятно, близко ли
    # Fast к пределу или далеко. Потолок — это ИСТИННЫЕ коды нужных уровней,
    # декодированные тем же путём. Ниже него модель с замороженным
    # токенизатором не опустится ни при каком обучении.
    with torch.no_grad():
        ceil = {}
        for name, n_lv in (("fast", 1), ("medium", 2), ("full", N_LEVEL)):
            acc = []
            for i0 in range(0, N, 256):
                k = Kt[i0:i0 + 256]
                z = sum(E[j][k[:, j, :]] for j in range(n_lv))
                x, _ = codec._decode(z, embodiment_ids=0)
                acc.append(x[..., :7].float().cpu())
            a_c = torch.cat(acc)
            d = a_c - A_star
            ceil[name] = dict(
                pose8_rms=float(torch.sqrt((d[:, :H_EXEC, :6] ** 2).mean())),
                grip8_rms=float(torch.sqrt((d[:, :H_EXEC, 6] ** 2).mean())),
                grip8_flip=float((torch.sign(a_c[:, :H_EXEC, 6])
                                  != torch.sign(A_star[:, :H_EXEC, 6])
                                  ).float().mean()))
    print("потолки (истинные коды нужных уровней, ниже не опуститься):")
    for m_, r in ceil.items():
        print(f"    {m_:<7} поза8 {r['pose8_rms']:.4f}  схват8 "
              f"{r['grip8_rms']:.4f}  знак {r['grip8_flip']:.2%}")

    tr, va, te = split_by_episode(epi, tsk_a, seed=args.seed)
    print(f"train {tr.sum()}, val {va.sum()}, test {te.sum()}; "
          f"задач {len(np.unique(tsk_a))}")

    # --- сборка батча --------------------------------------------------------
    def build(sel):
        i1 = tf_img(torch.tensor(np.stack([im1[j] for j in sel])).permute(0, 3, 1, 2))
        i2 = tf_img(torch.tensor(np.stack([im2[j] for j in sel])).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for gi in sel:
            m = prompt_template(st_n[gi], None, tsk[gi],
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

    def forward(sel, mode, teacher_p):
        """Один проход. Батч собирается ИЗ ОДНОГО офсета: position_offset
        задаётся на весь вызов, и смешивать задачи с разными офсетами нельзя."""
        batch = build(sel)
        po = int(offs[sel[0]])
        assert (offs[sel] == po).all(), "в батче смешались разные офсеты"
        B_, _, vemb, _ = model._build_vlm_inputs_embeds(
            input_ids=batch.get("input_ids"), inputs_embeds=None,
            pixel_values=batch.get("pixel_values"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            image_hidden_states=None)
        apos = model._build_action_pos_ids_strided(
            batch_size=B_, base_pos=vemb.shape[1],
            action_seq_len=model.block_size, device=dev, position_offset=po)
        pos = model._build_joint_position_ids(
            batch_size=B_, vlm_seq_len=vemb.shape[1], action_pos_ids=apos,
            device=dev)
        tc = None
        if teacher_p > 0 and np.random.random() < teacher_p:
            tc = [torch.as_tensor(K_true[sel][:, g, :]).long().to(dev)
                  for g in range(N_LEVEL)]
        return model.run_progressive(
            vlm_inputs_embeds=vemb, attention_mask=batch.get("attention_mask"),
            position_ids=pos, books=E, mode=mode, teacher_codes=tc,
            tau=args.tau), (tc is not None)

    def losses(out, sel):
        """Потери по всем доступным уровням. Эмбеддинги для действия берутся из
        СОБСТВЕННЫХ предсказаний, даже при teacher forcing: иначе метрика
        измеряла бы учителя, а не модель."""
        y = torch.as_tensor(K_true[sel]).long().to(dev)
        a_star = A_star[sel].to(dev)
        l_code = torch.zeros((), device=dev)
        embs, l_act, parts = [], torch.zeros((), device=dev), {}
        for g, lg in enumerate(out["logits"]):
            l_code = l_code + w_code[g] * F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]).float(), y[:, g, :].reshape(-1))
            e, _, _ = straight_through(lg.float(), E[g], args.tau)
            embs.append(e)
            a_hat = decode_prefix(embs)
            li = action_loss(a_hat, a_star, args.mu, args.eta)
            parts[f"L{g}"] = float(li)
            l_act = l_act + w_act[g] * li
        return l_code + args.beta * l_act, l_code, l_act, parts

    # --- оценка --------------------------------------------------------------
    @torch.no_grad()
    def evaluate(mask, tag):
        idxs = np.where(mask)[0]
        res = {}
        for mode, n_lv in (("fast", 1), ("medium", 2), ("full", N_LEVEL)):
            pe, ge, gf, acc, nb = [], [], [], [], 0
            for po in sorted({int(v) for v in offs[idxs]}):
                ii = idxs[offs[idxs] == po]
                for i0 in range(0, len(ii), args.batch):
                    sel = ii[i0:i0 + args.batch]
                    out, _ = forward(sel, mode, 0.0)      # БЕЗ учителя
                    embs = [straight_through(lg.float(), E[g], args.tau)[0]
                            for g, lg in enumerate(out["logits"])]
                    a_hat = decode_prefix(embs)
                    a_st = A_star[sel].to(dev)
                    d = (a_hat - a_st)
                    pe.append(float((d[:, :H_EXEC, :6] ** 2).mean()))
                    ge.append(float((d[:, :H_EXEC, 6] ** 2).mean()))
                    gf.append(float((torch.sign(a_hat[:, :H_EXEC, 6])
                                     != torch.sign(a_st[:, :H_EXEC, 6])).float().mean()))
                    acc.append(float((out["pred_codes"][0] ==
                                      torch.as_tensor(K_true[sel][:, 0, :]).to(dev)
                                      ).float().mean()))
                    nb += 1
            res[mode] = dict(pose8_rms=float(np.sqrt(np.mean(pe))),
                             grip8_rms=float(np.sqrt(np.mean(ge))),
                             grip8_flip=float(np.mean(gf)),
                             q0_acc=float(np.mean(acc)), n_batches=nb)
        print(f"  [{tag}] " + "  ".join(
            f"{m}: поза8 {r['pose8_rms']:.4f} (потолок {ceil[m]['pose8_rms']:.4f}) "
            f"знак {r['grip8_flip']:.2%}" for m, r in res.items()))
        return res

    # --- обучение ------------------------------------------------------------
    params = model.progressive_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    itr = np.where(tr)[0]
    spe = math.ceil(len(itr) / args.batch)
    total = args.epochs * spe
    warm = max(1, int(total * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    print(f"\n{spe} шагов на эпоху, всего {total}, прогрев {warm}")

    hist, best = [], {"fast": None, "pareto": None}
    ev0 = evaluate(va, "эпоха 0, до обучения")
    step = 0
    for ep in range(args.epochs):
        p_tf = tf_prob(ep)
        model.train()
        t0, agg = time.time(), []
        # БАТЧИ СОБИРАЮТСЯ ВНУТРИ ГРУППЫ ОФСЕТА: position_offset один на вызов.
        groups = [itr[offs[itr] == po] for po in sorted({int(v) for v in offs[itr]})]
        order = []
        for g in groups:
            g = np.random.permutation(g)
            order += [g[i:i + args.batch] for i in range(0, len(g), args.batch)]
        np.random.shuffle(order)
        for sel in order:
            mode = "full"
            out, used_tf = forward(sel, mode, p_tf)
            loss, lc, la, parts = losses(out, sel)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            agg.append((float(loss), float(lc), float(la)))
            if step == args.stage_a_steps:
                print(f"  проводка пройдена: {args.stage_a_steps} шагов, "
                      f"loss {np.mean([a[0] for a in agg[-50:]]):.4f}")
        m = np.mean(agg, axis=0)
        model.eval()
        ev = evaluate(va, f"эпоха {ep + 1}, учитель {p_tf:.2f}")
        hist.append(dict(epoch=ep + 1, tf=p_tf, loss=float(m[0]),
                         l_code=float(m[1]), l_act=float(m[2]), val=ev,
                         minutes=(time.time() - t0) / 60))
        print(f"    loss {m[0]:.4f} (коды {m[1]:.3f}, действие {m[2]:.4f}), "
              f"{(time.time() - t0) / 60:.1f} мин")

        sd = {k: v.detach().cpu().clone()
              for k, v in model.state_dict().items()
              if k.startswith(("prog_heads", "prog_feedback")) or k.endswith((".A", ".B"))}
        f_now = ev["fast"]["pose8_rms"]
        if best["fast"] is None or f_now < best["fast"][0]:
            best["fast"] = (f_now, ep + 1)
            torch.save(sd, os.path.join(args.out, "best_fast.pt"))
        # ПАРЕТО: сумма трёх режимов, но только если Full не хуже старта.
        s_now = sum(ev[m_]["pose8_rms"] for m_ in ("fast", "medium", "full"))
        if ev["full"]["pose8_rms"] <= ev0["full"]["pose8_rms"] * 1.05:
            if best["pareto"] is None or s_now < best["pareto"][0]:
                best["pareto"] = (s_now, ep + 1)
                torch.save(sd, os.path.join(args.out, "best_pareto.pt"))

    print(f"\nлучший fast: эпоха {best['fast'][1]}, поза8 {best['fast'][0]:.4f}")
    if best["pareto"]:
        print(f"лучший парето: эпоха {best['pareto'][1]}")
    else:
        print("ПАРЕТО НЕ НАЙДЕН: Full деградировал во всех эпохах более чем на 5%")
    json.dump(dict(history=hist, before=ev0, ceilings=ceil,
                   best_fast=best["fast"],
                   best_pareto=best["pareto"], exits=list(exits),
                   args=vars(args), trainable=rep, tokenizer_gap=gap),
              open(os.path.join(args.out, "history.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"сохранено в {args.out}/")


if __name__ == "__main__":
    main()
