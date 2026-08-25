"""K-6d: один проход модели — все признаки для обучения уточнителей.

ЗАЧЕМ. Этап 1 это не один уточнитель, а кривая: глубины 2/4/6/12/24, с
перекрёстным вниманием к префиксу и без, разное число слоёв с доступом к кэшу.
Каждому нужны одни и те же входы, а считает их 2.2B модель двадцать минут.
Пересчитывать одни и те же детерминированные числа десяток раз бессмысленно:

    без экстрактора: 12 вариантов x (20 мин прогона + 3 мин обучения) ~ 4.6 ч
    с экстрактором:  20 мин один раз + 12 x 3 мин                    ~ 1 ч

После него ни модель, ни LIBERO, ни симулятор не нужны — обучение уточнителя
это numpy и маленькая сеть.

ЧТО СОХРАНЯЕТСЯ
    h         (N, 16, d_act)   скрытые состояния ПЕРВОГО блока: вход
                               action_lm_head (bar.py:1248), то есть всё, чем
                               располагает однопроходная схема
    ctx       (N, L, d_vlm)    состояния префикса VLM для перекрёстного
                               внимания, fp16
    ctx_len   (N,)             сколько позиций префикса значимы (паддинг слева)
    K_true    (N, 3, 16)       коды токенизатора по действиям демонстраций
    K_bar     (N, 3, 16)       коды, предсказанные самой BAR
    act       (N, 20, 7)       действия датасета в пространстве кодека
    episode, task_id, pos_offset

ОТКУДА БЕРЁТСЯ ctx. `_run_action_sequence` идёт по слоям, обновляя
vlm_hidden_states и action_hidden_states совместно (bar.py:1231-1248), и первое
наружу не возвращает. Снимаем ВХОД input_layernorm ПОСЛЕДНЕГО слоя VLM — это
состояние префикса после L-1 слоёв. На один слой меньше полного выхода; для
контекста перекрёстного внимания это несущественно, но назвать надо честно.

ПОЛНЫЙ KV-КЭШ НЕ СОХРАНЯЕМ. По 24 слоям это ~75 ГБ. Уточнитель будет делать
ОДНОУРОВНЕВОЕ перекрёстное внимание к сохранённым состояниям, а не послойное,
как блоки 2-3. Это слабее, и в статье это ограничение придётся указать.

ПРЕДОБРАБОТКА ДАТАСЕТНАЯ, НЕ СРЕДОВАЯ. Три входа обрабатываются иначе, чем в
цикле оценки, и на каждом я уже ошибался (см. FINDINGS §10):
  изображение — БЕЗ разворота каналов, PIL.convert("RGB") уже даёт RGB;
  кроп        — 87.5% от ФАКТИЧЕСКОГО размера (кадр 256, не 224);
  состояние   — уже 8-мерное, process_state НЕ применяется.

Запуск:
    python3 experiments/k6d_extract.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6d_extract.py --ckpt <ckpt> --n-obs 6000 --n-ep 600 \\
        --out data/k6d_features.npz
"""

import argparse
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20


def find_vlm_layers(vlm):
    """Найти список слоёв башни VLM, не угадывая путь.

    Устройство обёрток HF меняется между версиями, поэтому вместо
    `vlm.text_model.layers` ищем самый длинный ModuleList, элементы которого
    имеют input_layernorm — признак слоя декодера.
    """
    import torch.nn as nn
    best = None
    for name, mod in vlm.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1 and \
                hasattr(mod[0], "input_layernorm"):
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise SystemExit(
            "не нашёл слои VLM: ни один ModuleList не содержит модулей с "
            "input_layernorm. Посмотрите model.vlm и укажите путь вручную.")
    return best


def pad_left(mats, lens, d, dtype=np.float16):
    """Собрать батчи разной длины в один массив с ЛЕВЫМ паддингом.

    Процессор дополняет слева (padding_side="left"), значит значимая часть
    последовательности прижата вправо. Сохраняем в том же виде: так позиция
    -1 всегда последний реальный токен, независимо от длины подсказки.
    """
    L = max(m.shape[1] for m in mats)
    out = np.zeros((sum(m.shape[0] for m in mats), L, d), dtype)
    i = 0
    for m in mats:
        b, l = m.shape[0], m.shape[1]
        out[i:i + b, L - l:, :] = m.astype(dtype)
        i += b
    return out, np.asarray(lens)


def selftest():
    # 1. ЛЕВЫЙ ПАДДИНГ. Значимая часть обязана прижиматься вправо, а нули —
    #    уходить влево, иначе перекрёстное внимание будет смотреть в пустоту.
    a = np.ones((2, 3, 4)); b = np.full((1, 5, 4), 2.0)
    out, lens = pad_left([a, b], [3, 3, 5], 4)
    assert out.shape == (3, 5, 4), out.shape
    assert (out[0, :2] == 0).all() and (out[0, 2:] == 1).all(), \
        "короткая последовательность обязана быть прижата вправо"
    assert (out[2] == 2).all(), "длинная не должна паддиться вовсе"
    assert lens.tolist() == [3, 3, 5]

    # 2. РАСКЛАДКА КОДОВ поуровневая, а не перемежением по времени: BAR берёт
    #    блоками по block_size подряд (bar.py:1500-1503).
    toks = np.arange(N_POS * N_LEVEL)[None]
    K = toks.reshape(1, N_LEVEL, N_POS)
    assert (K[0, 0] == np.arange(0, 16)).all(), "первые 16 — уровень 0"
    assert (K[0, 2] == np.arange(32, 48)).all(), "последние 16 — уровень 2"

    print("самопроверка пройдена: левый паддинг прижимает значимое вправо, "
          "раскладка кодов поуровневая")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-obs", type=int, default=6000)
    ap.add_argument("--n-ep", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--no-ctx", action="store_true",
                    help="не сохранять состояния префикса (экономит ~2 ГБ)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k6d_features.npz")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01,  # noqa: E402
                       STATE_Q99, VisionLanguageActionProcessor, dict_apply,
                       get_cfg, process_state, prompt_template, seed_everything)

    seed_everything(args.seed)
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    n_codes = int(cfg.MODEL.action_processor.vocab_size)

    off_by_task = None
    if args.pos_offset is None:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; постройте "
                             f"k4b0_offset_table.py или задайте --pos-offset "
                             f"(единый офсет — АБЛЯЦИЯ, а не протокол)")
        tb = json.load(open(args.offset_table))
        off_by_task = {k: int(v["pos_offset"]) for k, v in tb["tasks"].items()}

    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    grab_h, grab_ctx = [], []
    model.action_lm_head.register_forward_hook(
        lambda m, i, o: grab_h.append(i[0].detach().float().cpu()))
    if not args.no_ctx:
        lname, layers = find_vlm_layers(model.vlm)
        print(f"слои VLM: {lname}, всего {len(layers)}; снимаю вход "
              f"input_layernorm последнего")
        layers[-1].input_layernorm.register_forward_hook(
            lambda m, i, o: grab_ctx.append(i[0].detach().to(torch.float16).cpu()))

    # --- данные ---------------------------------------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(1693)
    per_ep = max(1, args.n_obs // max(args.n_ep, 1))

    def png(cell):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))

    im1, im2, st, act, tsk, epi = [], [], [], [], [], []
    for e in order:
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
        n_st = t.num_rows - T_CHUNK + 1
        for s0 in rng.choice(n_st, size=min(per_ep, n_st), replace=False):
            im1.append(png(c1[int(s0)])); im2.append(png(c2[int(s0)]))
            st.append(S_[int(s0)]); act.append(A_[int(s0):int(s0) + T_CHUNK])
            tsk.append(tasks_map[ti[int(s0)]]); epi.append(int(e))
    N = len(tsk)
    epi = np.asarray(epi)
    hw = im1[0].shape[0]
    print(f"собрано {N} наблюдений, {len(np.unique(epi))} эпизодов, "
          f"кадр {hw}x{hw}")

    tf = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])   # доля, не число

    ST_RAW = np.asarray(st, np.float64)
    if ST_RAW.shape[1] == len(STATE_Q01) + 1:
        ST_RAW = process_state(ST_RAW)
    assert ST_RAW.shape[1] == len(STATE_Q01), \
        f"состояние {ST_RAW.shape[1]} измерений, нормировка ждёт {len(STATE_Q01)}"
    st_n = (ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0

    a_codec = np.asarray(act, np.float64).copy()
    a_codec[..., :-1] /= max_act_q[..., :-1]
    a_codec[..., -1] *= -1
    a_codec = np.clip(a_codec, -1.0, 1.0)
    K_true = np.asarray(proc.action_processor.encode(a_codec),
                        np.int64).reshape(N, N_LEVEL, N_POS)

    # --- проход модели --------------------------------------------------------
    # НИКАКОГО МОЛЧАЛИВОГО ОТКАТА НА 4: промах в сопоставлении названий
    # незаметно превратился бы в неверный офсет, а он МЕНЯЕТ план
    # (k4b0_padding_probe: оракул 0.941 против 0.872).
    if args.pos_offset is None:
        miss = sorted({t for t in tsk if t not in off_by_task})
        if miss:
            raise SystemExit(
                f"в таблице офсетов нет {len(miss)} задач, например: "
                f"{miss[:3]}. Пересоберите k4b0_offset_table.py или задайте "
                f"--pos-offset явно (это АБЛЯЦИЯ).")
    offs = np.array([args.pos_offset if args.pos_offset is not None
                     else off_by_task[tsk[i]] for i in range(N)])
    Hm = None
    K_bar = np.zeros((N, N_LEVEL, N_POS), np.int64)
    ctx_parts, ctx_idx, ctx_lens = [], [], np.zeros(N, np.int32)
    done_cnt = 0
    for po in sorted({int(v) for v in offs}):
        idx_po = np.where(offs == po)[0]
        for i0 in range(0, len(idx_po), args.batch):
            sel = idx_po[i0:i0 + args.batch]
            b = len(sel)
            done_cnt += b
            i1 = tf(torch.tensor(np.stack([im1[j] for j in sel])).permute(0, 3, 1, 2))
            i2 = tf(torch.tensor(np.stack([im2[j] for j in sel])).permute(0, 3, 1, 2))
            image = torch.cat([i1, i2], dim=-1)
            msgs = []
            for gi in sel:
                m = prompt_template(
                    st_n[gi], None, tsk[gi],
                    mode=cfg.MODEL.vla_processor.kwargs.mode,
                    action_vocab_size=n_codes,
                    action_token_len=cfg.MODEL.action_processor.token_len)
                m[1]["content"] = m[1]["content"][1:]
                msgs.append(m)
            texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
            batch = proc(text=texts,
                         images=[[image[j].numpy()] for j in range(b)],
                         return_tensors="pt", padding=True, padding_side="left",
                         action_processor_kwargs={"embodiment_ids": 0})
            # ДЛИНЫ БЕРЁМ ДО ПЕРЕНОСА НА GPU. Раньше в ctx_len писалась
            # длина ВСЕГО БАТЧА после паддинга, одинаковая для всех, и
            # перекрёстное внимание считало бы паддинг настоящим контекстом.
            real_lens = batch["attention_mask"].sum(dim=1).numpy().astype(np.int32)
            batch = dict_apply(lambda x: x.to(dev, dtype), batch)
            grab_h.clear(); grab_ctx.clear()
            with torch.no_grad():
                tk = model.generate(**batch, position_offset=po, do_sample=False,
                                    initial_position_shift=1)
            assert len(grab_h) == N_LEVEL, f"action_lm_head сработала {len(grab_h)} раз"
            g0 = grab_h[0]
            assert g0.shape[1] == N_POS, (
                f"на первом блоке ждали {N_POS} позиций на входе "
                f"action_lm_head, получено {g0.shape[1]}")
            if Hm is None:
                Hm = np.zeros((N, N_POS, g0.shape[-1]), np.float32)
            Hm[sel] = g0.numpy()
            K_bar[sel] = tk.cpu().numpy().reshape(b, N_LEVEL, N_POS)
            if not args.no_ctx:
                assert len(grab_ctx) >= 1, "хук на слой VLM не сработал"
                c = grab_ctx[0].numpy()          # ПЕРВЫЙ блок, как и h
                assert c.shape[1] >= real_lens.max(), (
                    f"контекст короче маски: {c.shape[1]} против "
                    f"{real_lens.max()}")
                ctx_parts.append(c); ctx_idx.append(sel)
                ctx_lens[sel] = real_lens
            if done_cnt % (args.batch * 50) < args.batch:
                print(f"  {done_cnt}/{N} (офсет {po})", flush=True)
    assert done_cnt == N, f"обработано {done_cnt} из {N}"

    out = dict(h=Hm.astype(np.float16), K_true=K_true, K_bar=K_bar,
               act=a_codec.astype(np.float32), episode=epi,
               task=np.asarray(tsk), pos_offset=offs)
    # ctx_len ПИШЕТСЯ ТОЛЬКО ВМЕСТЕ С ctx. При --no-ctx он остался бы целиком
    # нулевым, и всякий, кто посчитает по нему маску паддинга, закроет ВСЁ.
    # Лучше отсутствие ключа, чем правдоподобный ноль.
    if not args.no_ctx:
        out["ctx_len"] = ctx_lens
    if not args.no_ctx:
        d_vlm = ctx_parts[0].shape[-1]
        L = max(c.shape[1] for c in ctx_parts)
        ctx = np.zeros((N, L, d_vlm), np.float16)
        for c, sel in zip(ctx_parts, ctx_idx):
            ctx[sel, L - c.shape[1]:, :] = c      # левый паддинг, как в процессоре
        out["ctx"] = ctx
        # ПРОВЕРКА, ЧТО ДЛИНЫ НАСТОЯЩИЕ, А НЕ ПАДДИНГ. Если бы писалась
        # длина батча, разброс схлопнулся бы почти в точку.
        assert ctx_lens.max() > ctx_lens.min(), (
            "все длины контекста одинаковы — почти наверняка записана длина "
            "батча, а не число значимых токенов")
        print(f"префикс: {ctx.shape}, {ctx.nbytes / 2**30:.2f} ГиБ; "
              f"значимых токенов {ctx_lens.min()}–{ctx_lens.max()} "
              f"(медиана {int(np.median(ctx_lens))})")
    print(f"h: {Hm.shape}; совпадение кодов BAR с истинными: "
          f"{(K_bar == K_true).mean():.1%}")

    out["meta"] = json.dumps(dict(
        ckpt=args.ckpt, n_obs=N, n_episodes=int(len(np.unique(epi))),
        n_codes=n_codes, image_hw=int(hw), dtype=args.dtype,
        offsets=sorted({int(v) for v in offs}),
        ctx_source="вход input_layernorm последнего слоя VLM (на слой меньше "
                   "полного выхода); одноуровневый, не послойный KV",
        bar_code_match=float((K_bar == K_true).mean())), ensure_ascii=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"\n  сохранено: {args.out} "
          f"({os.path.getsize(args.out) / 2**30:.2f} ГиБ)")


if __name__ == "__main__":
    main()
