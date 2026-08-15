"""K-3 (Gate B0): умеет ли BAR сам чинить fine-суффикс после правки coarse.

ЗАЧЕМ. Механизм RC-HDF предполагает, что при пересмотре грубого кода надо явно
отозвать и перестроить зависимый fine-суффикс. Но обученная модель может делать
это сама: BAR генерирует уровни блоками, и блок 1 предсказывается ПО блоку 0
(token_budget 48, num_blocks 3, block_size 16 = один уровень RVQ). Если модель
за один условный проход восстанавливает почти весь достижимый ремонт, явное
связывание экономит в лучшем случае один проход.

Проверяется на ВЫЛОЖЕННОМ чекпойнте, без обучения потока: ворота перед Phase 2,
которая стоит месяцы.

ПОЧЕМУ ОДНОЙ ЧУВСТВИТЕЛЬНОСТИ ЛОГИТОВ НЕДОСТАТОЧНО. Слабая зависимость fine от
coarse может означать не «суффикс останется устаревшим», а «BAR игнорирует
coarse-префикс и предсказывает fine прямо из наблюдения» — тогда устаревший
суффикс и не был проблемой. Решают ДВЕ величины вместе: сколько ремонта вообще
ДОСТУПНО (оракул) и какую долю его забирает BAR:

    R_BAR = (e_stale - e_BAR) / (e_stale - e_oracle)

ВЫБОР ОПОРЫ. Оракул можно определить как «вернуть прежнее действие» или «лучше
всего выразить ИСТИННОЕ действие при новом префиксе». В потоке правка грубого
кода — намеренное исправление, а не порча, поэтому суффикс должен обслуживать
новое решение. Опора — ДАТАСЕТНОЕ действие.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  оракул почти не улучшает stale -> чинить нечего, механизм беспредметен;
  оракульный выигрыш крупный, R_BAR > 0.7 -> модель чинит сама, явное
      связывание экономит не более одного прохода (6% при 16 NFE, 25% при 4);
  оракульный выигрыш крупный, R_BAR < 0.3 -> сильный сигнал за явное
      связывание, можно идти в Phase 2;
  промежуточное -> нужен маленький обучаемый refiner, но не полный поток.

САНИТАРНАЯ ПРОВЕРКА ОБЯЗАТЕЛЬНА и падает громко. Формат наблюдений должен
совпадать с тем, на котором BAR обучался: порядок каналов, кроп, шаблон
промпта. При несовпадении модель выдаст мусор, а таблицы будут выглядеть
правдоподобно. У политики OAT в аналогичной проверке вышло 0.0106 размаха.

Запуск:
    python3 experiments/k3_bar_suffix_repair.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
        --zarr <путь>/libero10_N500.zarr
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_residual_cost import latent_from_codes, load_codec, projected_codebooks  # noqa: E402



def js_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Дивергенция Йенсена-Шеннона по последней оси, в битах."""
    m = 0.5 * (p + q)

    def kl(x, y):
        return (x * (x.clamp_min(1e-30).log() - y.clamp_min(1e-30).log())).sum(-1)

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / float(np.log(2.0))


def greedy_suffix(E, codes, target, p, level):
    """Жадно переквантовать уровни > level в позиции p к target[:, p].
    Жадно — не оптимально; это процедура кодировщика RVQ."""
    out = codes.clone()
    r = target[:, p] - sum(E[j][out[:, p, j]] for j in range(level + 1))
    for j in range(level + 1, E.shape[0]):
        c = torch.cdist(r.unsqueeze(1), E[j]).squeeze(1).argmin(-1)
        out[:, p, j] = c
        r = r - E[j][c]
    return out


# Константы нормировки из scripts/utils.py:163. Состояние 8-мерное:
# process_state даёт [pos(3), axis-angle(3), gripper(2)].
STATE_Q99 = np.array([0.13556506, 0.33566484, 1.27066591, 3.27734607,
                      2.4061097, 0.59776972, 0.04031316, -0.00177811])
STATE_Q01 = np.array([-0.39912487, -0.26883513, 0.03826696, 1.50895805,
                      -2.71979114, -1.08050857, 0.00174237, -0.04002561])
# MAX_ACTION_Q = max(|Q99|, |Q01|), utils.py:199
MAX_ACTION_Q = np.array([0.9375, 0.9107142686843872, 0.9375,
                         0.20357142388820648, 0.26357144117355347, 0.375, 1.0])


def prompt_template(state, task: str) -> list:
    """Копия discrete-ветки scripts/utils.py:91. Перенесена, а не
    импортирована: scripts/utils.py тянет gym, robosuite и libero, которые
    нужны только симулятору. Два плейсхолдера изображения — как у них; при
    склейке видов первый удаляется вызывающей стороной."""
    return [
        {"role": "system",
         "content": "Analyze the input image and predict robot actions."},
        {"role": "user",
         "content": [
             {"type": "image"},
             {"type": "image"},
             {"type": "text",
              "text": f"**State**: {[round(v, 3) for v in state.tolist()]}, "
                      f"**Task**: {task}."},
         ]},
    ]


def quat2axisangle(q):
    """q в порядке (x,y,z,w). Повторяет robosuite T.quat2axisangle."""
    q = np.asarray(q, np.float64).copy()
    q[..., 3] = np.clip(q[..., 3], -1.0, 1.0)
    den = np.sqrt(np.clip(1.0 - q[..., 3] ** 2, 0.0, None))
    small = den < 1e-8
    ang = 2.0 * np.arccos(q[..., 3])
    out = q[..., :3] * (ang / np.where(small, 1.0, den))[..., None]
    return np.where(small[..., None], np.zeros_like(q[..., :3]), out)


def load_lerobot(n_obs: int, T: int, n_ep: int = 24, seed: int = 0):
    """Родной обучающий датасет BAR — physical-intelligence/libero, ветка v2.0.

    Читаем parquet НАПРЯМУЮ, без библиотеки lerobot: она версии 0.4.4 требует
    формат новее v2.0 и отказывается открывать датасет (BackwardCompatibility-
    Error). Формат при этом простой, и прямое чтение полностью под контролем.

    Из meta/info.json: 1693 эпизода, 273465 кадров, fps 10, картинки 256x256
    закодированными байтами прямо в parquet (видео нет), состояние 8-мерное —
    ровно под константы STATE_Q99/Q01, без process_state.

    Обучение брало delta_timestamps [i/10 for i in range(20)] при fps 10, то
    есть ДВАДЦАТЬ ПОДРЯД идущих кадров.
    """
    import io
    import json

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        d = json.loads(line)
        tasks_map[d["task_index"]] = d["task"]

    rng = np.random.default_rng(seed)
    # Эпизодов берём много: замеры внутри эпизода сильно коррелированы, и
    # кластерный бутстрап по ним требует достаточного числа кластеров.
    eps = rng.choice(1693, size=min(n_ep, n_obs), replace=False)
    per_ep = int(np.ceil(n_obs / len(eps)))
    im1, im2, st, act, prev, tasks, epi = [], [], [], [], [], [], []

    def png(cell):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))

    for e in eps:
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        n = t.num_rows
        if n <= T:
            continue
        starts = rng.choice(n - T, size=min(per_ep, n - T), replace=False)
        A_ = np.asarray(t.column("actions").to_pylist(), np.float32)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        for s0 in starts:
            im1.append(png(c1[s0]))
            im2.append(png(c2[s0]))
            st.append(S_[s0])
            act.append(A_[s0:s0 + T])
            prev.append(A_[max(s0 - 1, 0)])      # действие ДО чанка, для базы
            tasks.append(tasks_map[ti[s0]])
            epi.append(int(e))                   # кластер для бутстрапа
        if len(tasks) >= n_obs:
            break

    k = min(n_obs, len(tasks))
    print(f"LeRobot v2.0: {len(eps)} эпизодов, {k} наблюдений, "
          f"картинки {im1[0].shape}")
    to_t = lambda a: torch.from_numpy(np.stack(a[:k])).permute(0, 3, 1, 2)  # noqa: E731
    print(f"  эпизодов в выборке: {len(set(epi[:k]))}")
    return (to_t(im1), to_t(im2), np.stack(st[:k]), np.stack(act[:k]),
            np.stack(prev[:k]), tasks[:k], np.array(epi[:k]))


def build_state(z, t_idx, quat_wxyz: bool):
    """[pos(3), axis-angle(3), gripper(2)], нормировано в [-1, 1].

    В зарре кватернион записан с w ПЕРВЫМ (первая компонента ~0.999 при почти
    единичном повороте), а robosuite ждёт (x,y,z,w) — переставляем."""
    d = z["data"]
    pos = np.asarray(d["robot0_eef_pos"])[t_idx]
    q = np.asarray(d["robot0_eef_quat"])[t_idx]
    if quat_wxyz:
        q = np.concatenate([q[:, 1:], q[:, :1]], 1)
    grip = np.asarray(d["robot0_gripper_qpos"])[t_idx]
    st = np.concatenate([pos, quat2axisangle(q), grip], 1)
    return ((st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0).astype(np.float32)


def build_batch(im1, im2, tasks, state, proc, args, device):
    """Наблюдения в формате BAR. im1/im2 — тензоры (B,C,H,W) uint8.

    Обучение (scripts/utils.py:205): RandomCrop(int(256*0.875)) + Resize(224),
    то есть 87.5% поля зрения. На инференсе берём CenterCrop той же долей —
    детерминированно. Виды склеиваются в ОДНУ картинку по ширине, и из
    сообщения удаляется первый плейсхолдер изображения."""
    from torchvision.transforms import CenterCrop, Compose, Resize

    n = len(tasks)
    hw = int(im1.shape[-2])
    tf = (Compose([CenterCrop(int(hw * 0.875)), Resize((224, 224))])
          if args.center_crop else Compose([Resize((224, 224))]))
    print(f"картинки {tuple(im1.shape)} {im1.dtype}, "
          f"диапазон [{int(im1.min())}, {int(im1.max())}], "
          f"кроп {int(hw * 0.875) if args.center_crop else hw} -> 224")

    # Развороты нужны только источнику zarr: там ориентация кадров чужая
    # (перебор показал, что нужно зеркало по ширине). У родного датасета
    # формат уже тот, на котором модель обучалась.
    f = (args.flip or "w") if args.source == "zarr" else ""
    if "h" in f:
        im1, im2 = im1.flip(-2), im2.flip(-2)
    if "w" in f:
        im1, im2 = im1.flip(-1), im2.flip(-1)
    if "c" in f:
        im1, im2 = im1.flip(-3), im2.flip(-3)

    t1, t2 = tf(im1), tf(im2)
    msgs = [prompt_template(state[i], tasks[i]) for i in range(n)]
    if args.tiled:
        im = torch.cat([t1, t2], dim=-1)
        images = [[im[i].numpy()] for i in range(n)]
        for m in msgs:
            m[1]["content"] = m[1]["content"][1:]
    else:
        images = [[t1[i].numpy(), t2[i].numpy()] for i in range(n)]
    texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
    b = proc(text=texts, images=images, return_tensors="pt", padding=True)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--source", choices=["lerobot", "zarr"], default="lerobot",
                    help="lerobot — родной обучающий датасет BAR (правильное "
                         "разрешение и формат); zarr — данные OAT, 128x128")
    ap.add_argument("--zarr", default=None, help="нужен только при --source zarr")
    ap.add_argument("--n-obs", type=int, default=48)
    ap.add_argument("--n-pos", type=int, default=6, help="позиций p на наблюдение")
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--exec-window", type=int, default=8,
                    help="сколько первых шагов чанка реально исполняется")
    ap.add_argument("--stride", type=int, default=1,
                    help="шаг по кадрам, только для --source zarr")
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--flip", default="",
                    help="какие оси разворачивать: h (высота), w (ширина), "
                         "c (каналы); можно сочетать, например 'hc'")
    ap.add_argument("--quat-wxyz", action="store_true", default=False,
                    help="переставить кватернион из wxyz в xyzw. По умолчанию "
                         "ВЫКЛ: константы STATE_Q01/Q99[3] лежат в [1.51, 3.28] "
                         "(около pi), то есть схват развёрнут на 180 градусов, "
                         "и первая компонента 0.9988 в зарре — это x, а не w")
    ap.add_argument("--tiled", action="store_true", default=True,
                    help="склеить два вида в одну картинку, как при обучении")
    ap.add_argument("--sanity-ratio", type=float, default=0.7,
                    help="генерация должна быть лучше лучшей тривиальной базы "
                         "хотя бы в 1/ratio раз")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.root))
    import zarr

    # ТОКЕНИЗАТОР БЕРЁМ ИЗ ЧЕКПОЙНТА, а не со стороны. Выложенный BAR обучен
    # с 16-токенным кодеком (их путь называется vq_v2_16tokens, block_size*
    # num_blocks = 16), тогда как ActionCodec-Base-RVQft даёт 48 токенов
    # (3 уровня x 16 позиций). Подстановка чужого кодека дала бы правдоподобный
    # мусор.
    # Регистрация типа `action_codec` в AutoConfig/AutoModel происходит как
    # побочный эффект импорта их пакета (configuration_actioncodec.py:225,
    # modeling_actioncodec.py:658). Без неё AutoModel.from_pretrained внутри
    # процессора не узнает архитектуру.
    import actioncodec  # noqa: F401

    import importlib.util
    _sp = importlib.util.spec_from_file_location(
        "ac_vla_tokenizer",
        os.path.join(os.path.abspath(args.root), "utils", "vla_tokenizer.py"))
    _m = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    proc = _m.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tok = proc.action_processor.to(args.device).eval()
    L, P = tok.num_quantizers, tok.n_tokens_per_quantizer
    print(f"кодек ИЗ ЧЕКПОЙНТА: словарь {tok.vocab_size}, уровней {L}, "
          f"позиций {P}, токенов {L * P}")
    ecfg = tok.config.embodiment_config[
        list(tok.config.embodiment_config.keys())[args.embodiment]]
    T, D_act = int(ecfg["freq"] * ecfg["duration"]), ecfg["action_dim"]
    E = projected_codebooks(tok, args.device)

    if args.source == "lerobot":
        IM1, IM2, ST_RAW, A, PREV, tasks, EPI = load_lerobot(args.n_obs, T)
        zz = t_idx = None
    else:
        zz = zarr.open(os.path.abspath(args.zarr), mode="r")
        print("ключи данных:", list(zz["data"].keys()))
        acts = np.asarray(zz["data"]["action"])
        ends = np.asarray(zz["meta"]["episode_ends"])
        starts = np.concatenate([[0], ends[:-1]])
        span = T * args.stride
        cand = [i for s_, e_ in zip(starts, ends) for i in range(s_, e_ - span)]
        t_idx = np.sort(np.random.default_rng(0).choice(cand, args.n_obs,
                                                        replace=False))
        raw = np.asarray(zz["data"]["prompt"])[t_idx]
        tasks = [(v.decode() if isinstance(v, bytes) else str(v)).strip()
                 for v in raw.ravel()]
        A = np.stack([acts[t:t + span:args.stride] for t in t_idx]).astype(np.float32)
        ST_RAW = PREV = None
        EPI = np.zeros(args.n_obs, int)
        IM1 = torch.from_numpy(np.stack(
            [np.asarray(zz["data"]["agentview_rgb"][t]) for t in t_idx])
        ).permute(0, 3, 1, 2)
        IM2 = torch.from_numpy(np.stack(
            [np.asarray(zz["data"]["robot0_eye_in_hand_rgb"][t]) for t in t_idx])
        ).permute(0, 3, 1, 2)

    # Подготовка действий РОВНО как при обучении (scripts/utils.py:246):
    # поканальное деление на MAX_ACTION_Q, НЕГАЦИЯ захвата, обрезка в [-1,1].
    assert A.shape[1] == T, f"чанк {A.shape[1]} шагов, ожидалось {T}"
    A = np.asarray(A, np.float32).copy()
    A[..., :-1] = A[..., :-1] / MAX_ACTION_Q[:-1]
    A[..., -1] = -A[..., -1]
    A = np.clip(A, -1.0, 1.0)
    if PREV is not None:
        PREV = np.asarray(PREV, np.float32).copy()
        PREV[..., :-1] = PREV[..., :-1] / MAX_ACTION_Q[:-1]
        PREV[..., -1] = -PREV[..., -1]
        PREV = torch.from_numpy(np.clip(PREV, -1.0, 1.0)).to(args.device)
    a_true = torch.from_numpy(A).to(args.device)
    scale = float(a_true.max() - a_true.min())
    B = len(A)
    print(f"наблюдений {B}, инструкция: {tasks[0]!r}")

    from smolvla.bar import SmolVLABlockwiseAR

    # token_budget и num_blocks НАДО ПЕРЕДАВАТЬ ЯВНО: в config.json чекпойнта
    # их нет, и from_pretrained берёт умолчания класса (4 блока по 4 = 16
    # токенов), тогда как кодек даёт 48. В их config/eval/bar.yaml они
    # задаются из конфига: token_budget = token_len, num_blocks = 3.
    model = SmolVLABlockwiseAR.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
        token_budget=P * L, num_blocks=L,
        action_vocab_size=tok.vocab_size).to(args.device).eval()
    bs, nb = model.block_size, model.num_blocks
    assert bs * nb == P * L, (
        f"модель ждёт {bs}x{nb}={bs*nb} токенов, кодек даёт {P}x{L}={P*L}. "
        "Значит это не тот кодек.")
    bpl = P // bs                      # блоков на один уровень RVQ
    assert P % bs == 0, "уровень не делится на целое число блоков"
    print(f"блоков {nb}, block_size {bs}, на уровень {bpl} блоков.\n"
          f"Раскладка поуровневая, поэтому блоки 0..{bpl-1} покрывают уровень 0\n"
          f"целиком, и уровень 1 предсказывается ПОСЛЕ него — обусловленность,\n"
          f"на которой стоит замер, сохраняется.\n")

    if args.source == "lerobot":
        # В их LiberoAllDataset состояние из датасета нормируется напрямую
        # 8-мерными константами, без process_state — значит оно уже приведено.
        assert ST_RAW.shape[1] == len(STATE_Q99), (
            f"состояние {ST_RAW.shape[1]}-мерное, константы {len(STATE_Q99)}-мерные")
        st = ((ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
              ).astype(np.float32)
    else:
        st = build_state(zz, t_idx, args.quat_wxyz)
    print(f"состояние: {st.shape[1]} чисел, диапазон после нормировки "
          f"[{st.min():.2f}, {st.max():.2f}]")
    batch = build_batch(IM1, IM2, tasks, st, proc, args, args.device)

    def dec(codes):
        """codes (B,P,L) -> действие. Декодируем через латенту, минуя токены."""
        h = latent_from_codes(E, codes)
        return tok._decode(h, args.embodiment, None)[0][..., :D_act]

    def err(x, win=None):
        d = (x - a_true).abs()
        if win:
            d = d[:, :win]
        return (d[..., :D_act - 1].flatten(1).amax(-1) / scale)

    # Эмбеддинги VLM (текст + картинки) считаются ОДИН раз: от истории токенов
    # действия они не зависят, а стоят дорого. Процессор отдаёт input_ids и
    # pixel_values, эмбеддинги строит сама модель.
    with torch.no_grad():
        _, _, VLM_EMB, _ = model._build_vlm_inputs_embeds(
            input_ids=batch["input_ids"],
            inputs_embeds=None,
            pixel_values=batch.get("pixel_values"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            image_hidden_states=None)
    print(f"эмбеддинги VLM: {tuple(VLM_EMB.shape)}")

    def blk(hist):
        """Логиты следующего блока при заданной истории (B, k*P) или None."""
        return model._predict_next_block_logits(
            vlm_inputs_embeds=VLM_EMB,
            attention_mask=batch.get("attention_mask"),
            history_tokens=hist).float()

    with torch.no_grad():
        def gen_from(hist, n_blocks):
            """Достроить n_blocks блоков жадно, начиная с истории hist."""
            for _ in range(n_blocks):
                c = blk(hist).argmax(-1)
                hist = c if hist is None else torch.cat([hist, c], 1)
            return hist

        def to_levels(flat_tok):
            """(B, P*L) -> (B, P, L). Раскладка поуровневая, как в decode."""
            return flat_tok.reshape(-1, L, P).transpose(1, 2)

        # ---------- обычная генерация BAR, блок за блоком ----------
        flat_gen = gen_from(None, nb)                 # (B, P*L)
        codes = to_levels(flat_gen)                   # (B, P, L)

        base = dec(codes)
        e_base = err(base)


        # ПОЧЕМУ НЕ АБСОЛЮТНЫЙ ПОРОГ. Расхождение стохастической политики с
        # КОНКРЕТНОЙ демонстрацией — не ошибка формата, а нормальный остаток
        # клонирования: валидных продолжений много, сравниваем с одним. Поэтому
        # критерий — бьёт ли модель тривиальные базы. Если наблюдения поданы
        # неверно, предсказание не будет лучше нулевого.
        zero = torch.zeros_like(a_true)
        zero[..., -1] = a_true[..., -1]              # захват оставляем истинный
        mean = a_true.mean(0, keepdim=True).expand_as(a_true)
        # ЧЕСТНОЕ УДЕРЖАНИЕ: повтор последнего действия ДО чанка, одно и то же
        # на все шаги. Прежний вариант (сдвиг истинной последовательности на
        # шаг) подглядывал будущее: политика видит одно наблюдение в начале и
        # предсказывает все T шагов, истинного действия шага t-1 у неё нет.
        prev = (PREV.unsqueeze(1).expand_as(a_true) if PREV is not None
                else torch.zeros_like(a_true))
        e_zero, e_mean, e_prev = err(zero), err(mean), err(prev)
        m = float(e_base.median())
        print("\nСАНИТАРНАЯ ПРОВЕРКА, медианы ошибки против датасетного действия:")
        print(f"  генерация BAR      {m:.4f}")
        print(f"  нулевое действие   {float(e_zero.median()):.4f}")
        print(f"  среднее по выборке {float(e_mean.median()):.4f}")
        print(f"  удержание прошлого {float(e_prev.median()):.4f}")
        best_base = min(float(e_zero.median()), float(e_mean.median()),
                        float(e_prev.median()))
        print(f"  модель лучше лучшей базы в {best_base / max(m, 1e-9):.2f} раза")
        assert m < best_base * args.sanity_ratio, (
            "модель не бьёт тривиальную базу — наблюдения поданы неверно. "
            "Перебрать формат: --probe")

        h_true = latent_from_codes(E, torch.stack(
            [torch.as_tensor(np.asarray(tok.encode(a_true, embodiment_ids=args.embodiment)),
                             device=args.device, dtype=torch.long)], 0)[0]
            .reshape(B, L, P).transpose(1, 2))       # (B,P,L): раскладка поуровневая

        rng = torch.Generator(device=args.device).manual_seed(1)
        KEYS = ("base", "stale", "loc", "glob", "orc")
        rows = {m: {k: [] for k in KEYS} for m in ("local", "on-policy")}
        rows_w = {m: {k: [] for k in KEYS} for m in ("local", "on-policy")}
        # ВЫДЕЛЕННЫЙ ЭФФЕКТ ПОДМЕНЫ: отклонение от НЕВОЗМУЩЁННОГО декодирования,
        # как в K-1. Против датасетного действия эффект тонет: собственная
        # ошибка клонирования ~0.069, а подмена одного кода из 16 позиций даёт
        # ~0.001. Здесь ошибка модели сокращается и остаётся только сдвиг.
        rows_d = {m: {k: [] for k in ("stale", "loc", "glob", "orc")}
                  for m in ("local", "on-policy")}
        js = {m: [] for m in ("local", "on-policy")}      # расхождение логитов
        top1 = {m: [] for m in ("local", "on-policy")}    # доля смен top-1
        infl = torch.zeros(P, P)                          # влияние p -> q
        n_infl = 0

        lg0 = blk(None)                                   # coarse-логиты
        lg1_base = blk(codes[:, :, 0])                    # логиты уровня 1 как есть
        p1_base = lg1_base.softmax(-1)

        for mode in ("local", "on-policy"):
            for _ in range(args.n_pos):
                pp = int(torch.randint(P, (1,), generator=rng, device=args.device))
                cur = codes[:, pp, 0]
                ar = torch.arange(B, device=args.device)
                if mode == "local":
                    # сосед в словаре: как в K-1
                    nbr = torch.cdist(E[0][cur], E[0]).topk(
                        args.knn + 1, largest=False).indices[:, 1:]
                    v = nbr[ar, torch.randint(args.knn, (B,), generator=rng,
                                              device=args.device)]
                else:
                    # альтернатива из собственных top-k логитов модели
                    tk = lg0[:, pp].topk(args.topk + 1, -1).indices[:, 1:]
                    v = tk[ar, torch.randint(args.topk, (B,), generator=rng,
                                             device=args.device)]

                c0 = codes[:, :, 0].clone()
                c0[:, pp] = v

                # BAR пересчитывает уровни 1 и 2 при новом уровне 0
                lg1n = blk(c0)
                c1n = lg1n.argmax(-1)
                c2n = to_levels(gen_from(torch.cat([c0, c1n], 1),
                                         nb - 2 * bpl))[:, :, 2]

                # насколько сместились распределения уровня 1 и где сменился top-1
                js[mode].append(js_div(p1_base, lg1n.softmax(-1)).mean(0).cpu())
                chg = (c1n != codes[:, :, 1]).float()
                top1[mode].append(float(chg.mean()))
                infl[pp] += chg.mean(0).cpu()
                n_infl += 1

                stale = codes.clone(); stale[:, pp, 0] = v
                loc = stale.clone()
                loc[:, pp, 1], loc[:, pp, 2] = c1n[:, pp], c2n[:, pp]
                glob = torch.stack([c0, c1n, c2n], -1)
                orc = greedy_suffix(E, stale, h_true, pp, 0)

                a_base = dec(codes)
                for k, cc in (("base", codes), ("stale", stale), ("loc", loc),
                              ("glob", glob), ("orc", orc)):
                    a_ = dec(cc)
                    rows[mode][k].append(err(a_).cpu().numpy())
                    rows_w[mode][k].append(err(a_, args.exec_window).cpu().numpy())
                    if k != "base":
                        d_ = (a_ - a_base).abs()[..., :D_act - 1]
                        rows_d[mode][k].append(
                            (d_.flatten(1).amax(-1) / scale).cpu().numpy())

    # ---------- отчёт ----------
    def med(d):
        return {k: float(np.median(np.concatenate(v))) for k, v in d.items()}

    def boot_ci(d, n_boot=400, seed=0):
        """Кластерный бутстрап ПО ЭПИЗОДАМ: наблюдения внутри эпизода сильно
        коррелированы, и бутстрап по строкам дал бы неправдоподобно узкий
        интервал. Возвращает интервалы для доступного ремонта и R_BAR."""
        rg = np.random.default_rng(seed)
        M = {k: np.stack(v) for k, v in d.items()}      # (n_pos, B)
        eps_u = np.unique(EPI)
        gaps, rs = [], []
        for _ in range(n_boot):
            pick = np.concatenate([np.where(EPI == e)[0] for e in
                                   rg.choice(eps_u, len(eps_u), replace=True)])
            g = {k: float(np.median(v[:, pick])) for k, v in M.items()}
            gap = g["stale"] - g["orc"]
            gaps.append(gap)
            if abs(gap) > 1e-9:
                rs.append((g["stale"] - g["loc"]) / gap)
        q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))  # noqa: E731
        return q(gaps), (q(rs) if len(rs) > n_boot // 2 else (float("nan"),) * 2)

    print("\n" + "=" * 78)
    print("ВЫДЕЛЕННЫЙ ЭФФЕКТ ПОДМЕНЫ (отклонение от невозмущённого декодирования)")
    print("=" * 78)
    print("Ошибка клонирования здесь сокращается: остаётся только сдвиг от правки.\n")
    print(f"{'режим':>11}{'stale':>9}{'BAR лок.':>10}{'BAR глоб.':>11}"
          f"{'оракул':>9}{'дост. ремонт':>14}{'95% ДИ':>18}"
          f"{'R_BAR':>8}{'95% ДИ':>18}")
    for mode in ("local", "on-policy"):
        m = med(rows_d[mode])
        gap = m["stale"] - m["orc"]
        r = (m["stale"] - m["loc"]) / gap if abs(gap) > 1e-9 else float("nan")
        cg, cr = boot_ci(rows_d[mode])
        print(f"{mode:>11}{m['stale']:>9.4f}{m['loc']:>10.4f}{m['glob']:>11.4f}"
              f"{m['orc']:>9.4f}{gap:>14.4f}"
              f"{f'[{cg[0]:+.4f}, {cg[1]:+.4f}]':>18}{r:>8.2f}"
              f"{f'[{cr[0]:+.2f}, {cr[1]:+.2f}]':>18}")
    print("""
Интервалы кластерные ПО ЭПИЗОДАМ. Если ДИ доступного ремонта накрывает ноль,
чинить нечего и R_BAR смысла не имеет. Если ДИ R_BAR шире, чем расстояние
между порогами 0.3 и 0.7, решения принимать нельзя — нужна выборка крупнее.""")

    for tag, data, note in (("ВЕСЬ ЧАНК", rows, ""),
                            (f"ПЕРВЫЕ {args.exec_window} ШАГОВ", rows_w,
                             " (только они и исполняются до перепланирования)")):
        print("\n" + "=" * 78)
        print(f"РЕМОНТ СУФФИКСА, {tag}{note}")
        print("=" * 78)
        print(f"{'режим':>11}{'база':>9}{'stale':>9}{'BAR лок.':>10}"
              f"{'BAR глоб.':>11}{'оракул':>9}{'дост. ремонт':>14}{'R_BAR':>8}")
        for mode in ("local", "on-policy"):
            m = med(data[mode])
            gap = m["stale"] - m["orc"]
            r = (m["stale"] - m["loc"]) / gap if abs(gap) > 1e-9 else float("nan")
            print(f"{mode:>11}{m['base']:>9.4f}{m['stale']:>9.4f}{m['loc']:>10.4f}"
                  f"{m['glob']:>11.4f}{m['orc']:>9.4f}{gap:>14.4f}{r:>8.2f}")

    print("\n" + "=" * 78)
    print("ЧУВСТВИТЕЛЬНОСТЬ УРОВНЯ 1 К ПОДМЕНЕ КОДА УРОВНЯ 0")
    print("=" * 78)
    print(f"{'режим':>11}{'JS, бит':>10}{'JS в позиции p':>16}{'смен top-1':>13}")
    for mode in ("local", "on-policy"):
        J = torch.stack(js[mode])                 # (n_pos, P)
        print(f"{mode:>11}{float(J.mean()):>10.4f}{float(J.max()):>16.4f}"
              f"{float(np.mean(top1[mode])):>12.1%}")
    print("""
JS считается между распределениями уровня 1 до и после подмены. Малая величина
означает, что BAR почти не смотрит на грубый код и предсказывает тонкие прямо
из наблюдения — тогда устаревший суффикс и не был проблемой, а не то что
механизм нужен.""")

    print("\n" + "=" * 76)
    print("МАТРИЦА ВЛИЯНИЯ: доля смен top-1 на уровне 1 в позиции q после правки p")
    print("=" * 76)
    infl /= max(n_infl, 1)
    diag = float(np.mean([infl[i, i] for i in range(P) if infl[i].sum() > 0]))
    off = float((infl.sum() - sum(infl[i, i] for i in range(P)))
                / max((infl > 0).sum().item() - P, 1))
    print(f"в той же позиции (диагональ): {diag:.3f}")
    print(f"в остальных позициях:         {off:.3f}")
    print(f"отношение:                    {diag / max(off, 1e-9):.1f}")
    print("""
Если влияние заметно и вне диагонали, фиксированный локальный откат суффикса
неверен по построению, и нужен обучаемый граф зависимостей — это уже сильнее
простой комбинации ResGen с self-correction.""")

    print("""
КАК ЧИТАТЬ ГЛАВНУЮ ТАБЛИЦУ.
  «дост. ремонт» = stale - оракул. Мал -> чинить нечего, механизм беспредметен.
  R_BAR = (stale - BAR лок.) / (stale - оракул) — доля доступного ремонта,
  которую BAR берёт за ОДИН условный проход.
    > 0.7  модель чинит сама; явное связывание экономит не более прохода;
    < 0.3  сильный сигнал за явное связывание;
    иначе  нужен маленький обучаемый refiner, но не полный поток.""")


if __name__ == "__main__":
    main()
