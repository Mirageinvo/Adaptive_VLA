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

LIBERO10 = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]


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


def build_batch(z, t_idx, tasks, state, proc, args, device):
    """Наблюдения в формате BAR. Повторяет scripts/eval_libero.py."""
    from torchvision.transforms import CenterCrop, Compose, Resize

    # Обучение: RandomCrop(224) из 256 -> 87.5% поля зрения. Их оценка:
    # CenterCrop(196) из 224 -> те же 87.5%. У нас картинки 128, поэтому кроп
    # берём тем же долей от фактического размера, потом ресайз в 224.
    hw = int(np.asarray(z["data"]["agentview_rgb"][0]).shape[0])
    tf = (Compose([CenterCrop(int(hw * 0.875)), Resize((224, 224))])
          if args.center_crop else Compose([Resize((224, 224))]))
    k1 = "agentview_rgb" if "agentview_rgb" in z["data"] else "agentview_image"
    k2 = ("robot0_eye_in_hand_rgb" if "robot0_eye_in_hand_rgb" in z["data"]
          else "robot0_eye_in_hand_image")
    im1 = np.stack([np.asarray(z["data"][k1][t]) for t in t_idx])
    im2 = np.stack([np.asarray(z["data"][k2][t]) for t in t_idx])
    print(f"картинки: {k1} {im1.shape} {im1.dtype}, диапазон "
          f"[{im1.min()}, {im1.max()}]")
    # ВНИМАНИЕ. В eval_libero.py стоит obs[...][:, :, ::-1] на массиве
    # (B,H,W,C) — это разворот оси 2, то есть ЗЕРКАЛО ПО ШИРИНЕ, а не
    # перестановка каналов, как читается с первого взгляда. Плюс robosuite
    # обычно отдаёт кадры перевёрнутыми по высоте. Что именно нужно нашему
    # зарру — решает перебор (--probe).
    f = args.flip
    if "h" in f:
        im1, im2 = im1[:, ::-1].copy(), im2[:, ::-1].copy()
    if "w" in f:
        im1, im2 = im1[:, :, ::-1].copy(), im2[:, :, ::-1].copy()
    if "c" in f:
        im1, im2 = im1[..., ::-1].copy(), im2[..., ::-1].copy()
    t1 = tf(torch.from_numpy(im1).permute(0, 3, 1, 2))
    t2 = tf(torch.from_numpy(im2).permute(0, 3, 1, 2))
    # Обучение склеивало виды в ОДНУ картинку по ширине и удаляло первый
    # плейсхолдер изображения из сообщения (utils.py: messages[1]["content"][1:]).
    msgs = [prompt_template(state[i], tasks[i]) for i in range(len(t_idx))]
    if args.tiled:
        im = torch.cat([t1, t2], dim=-1)
        images = [[im[i].numpy()] for i in range(len(t_idx))]
        for m in msgs:
            m[1]["content"] = m[1]["content"][1:]
    else:
        images = [[t1[i].numpy(), t2[i].numpy()] for i in range(len(t_idx))]
    texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
    b = proc(text=texts, images=images, return_tensors="pt", padding=True)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-obs", type=int, default=48)
    ap.add_argument("--n-pos", type=int, default=6, help="позиций p на наблюдение")
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--exec-window", type=int, default=8)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--flip", default="",
                    help="какие оси разворачивать: h (высота), w (ширина), "
                         "c (каналы); можно сочетать, например 'hc'")
    ap.add_argument("--stride", type=int, default=1,
                    help="шаг по кадрам при наборе чанка. Их обучение брало "
                         "delta_timestamps [i/10 for i in range(20)] — шаг 0.1 с, "
                         "то есть чанк на ДВЕ секунды. Если зарр 20 Гц, нужен 2")
    ap.add_argument("--probe-stride", action="store_true",
                    help="перебрать шаг 1..3 при найденном формате картинок")
    ap.add_argument("--probe", action="store_true",
                    help="перебрать формат картинок и напечатать ошибку "
                         "санитарной проверки для каждой комбинации")
    ap.add_argument("--quat-wxyz", action="store_true", default=False,
                    help="переставить кватернион из wxyz в xyzw. По умолчанию "
                         "ВЫКЛ: константы STATE_Q01/Q99[3] лежат в [1.51, 3.28] "
                         "(около pi), то есть схват развёрнут на 180 градусов, "
                         "и первая компонента 0.9988 в зарре — это x, а не w")
    ap.add_argument("--tiled", action="store_true", default=True,
                    help="склеить два вида в одну картинку, как при обучении")
    ap.add_argument("--sanity-max", type=float, default=0.15)
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

    zz = zarr.open(os.path.abspath(args.zarr), mode="r")
    print("ключи данных:", list(zz["data"].keys()))
    acts = np.asarray(zz["data"]["action"])
    ends = np.asarray(zz["meta"]["episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    span = T * args.stride                      # сколько кадров покрывает чанк
    cand = [i for s, e in zip(starts, ends) for i in range(s, e - span)]
    t_idx = np.sort(np.random.default_rng(0).choice(cand, args.n_obs, replace=False))

    # Инструкция лежит прямо в данных — угадывать по task_uid не нужно.
    if "prompt" in zz["data"]:
        raw = np.asarray(zz["data"]["prompt"])[t_idx]
        tasks = [(v.decode() if isinstance(v, bytes) else str(v)).strip()
                 for v in raw.ravel()]
        print(f"инструкции из зарра, пример: {tasks[0]!r}")
    else:
        uid = np.asarray(zz["data"]["task_uid"])[t_idx].astype(int).ravel()
        tasks = [LIBERO10[u % len(LIBERO10)] for u in uid]
        print("ВНИМАНИЕ: ключа 'prompt' нет, инструкции взяты по хардкоду —\n"
              "  соответствие индексов их порядку НЕ проверено.")

    # Подготовка действий РОВНО как при обучении (scripts/utils.py:246):
    # поканальное деление на MAX_ACTION_Q, НЕГАЦИЯ захвата, обрезка в [-1,1].
    # Прежние замеры (K-1, K-2) шли на сырых действиях: каналы 3 и 4 там были
    # примерно впятеро мельче обучающих. Их надо переснять.
    A = np.stack([acts[t:t + span:args.stride] for t in t_idx]).astype(np.float32)
    assert A.shape[1] == T, f"чанк {A.shape[1]} шагов при шаге {args.stride}"
    A[..., :-1] = A[..., :-1] / MAX_ACTION_Q[:-1]
    A[..., -1] = -A[..., -1]
    A = np.clip(A, -1.0, 1.0)
    a_true = torch.from_numpy(A).to(args.device)
    scale = float(a_true.max() - a_true.min())
    B = len(t_idx)

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

    st = build_state(zz, t_idx, args.quat_wxyz)
    print(f"состояние: {st.shape[1]} чисел, диапазон после нормировки "
          f"[{st.min():.2f}, {st.max():.2f}]")
    batch = build_batch(zz, t_idx, tasks, st, proc, args, args.device)

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

        def to_flat(c):
            return c.transpose(1, 2).reshape(len(c), -1)

        # ---------- обычная генерация BAR, блок за блоком ----------
        flat_gen = gen_from(None, nb)                 # (B, P*L)
        codes = to_levels(flat_gen)                   # (B, P, L)

        base = dec(codes)
        e_base = err(base)
        if args.probe_stride:
            print("\n" + "=" * 52)
            print("ПЕРЕБОР ШАГА ПО КАДРАМ (формат картинок фиксирован)")
            print("=" * 52)
            print(f"{'шаг':>5}{'секунд в чанке':>17}{'ошибка':>10}")
            for stv in (1, 2, 3, 4):
                sp = T * stv
                ti = np.sort(np.random.default_rng(0).choice(
                    [i for s_, e_ in zip(starts, ends) for i in range(s_, e_ - sp)],
                    args.n_obs, replace=False))
                Av = np.stack([acts[t:t + sp:stv] for t in ti]).astype(np.float32)
                Av[..., :-1] = Av[..., :-1] / MAX_ACTION_Q[:-1]
                Av[..., -1] = -Av[..., -1]
                Av = np.clip(Av, -1.0, 1.0)
                at = torch.from_numpy(Av).to(args.device)
                sc = float(at.max() - at.min())
                stv_ = build_state(zz, ti, args.quat_wxyz)
                bb = build_batch(zz, ti, [tasks[0]] * len(ti), stv_, proc, args,
                                 args.device)
                _, _, ve, _ = model._build_vlm_inputs_embeds(
                    input_ids=bb["input_ids"], inputs_embeds=None,
                    pixel_values=bb.get("pixel_values"),
                    pixel_attention_mask=bb.get("pixel_attention_mask"),
                    image_hidden_states=None)
                hist = None
                for _ in range(nb):
                    c = model._predict_next_block_logits(
                        vlm_inputs_embeds=ve,
                        attention_mask=bb.get("attention_mask"),
                        history_tokens=hist).float().argmax(-1)
                    hist = c if hist is None else torch.cat([hist, c], 1)
                d = dec(to_levels(hist))
                e = float(((d - at).abs()[..., :D_act - 1].flatten(1).amax(-1)
                           / sc).median())
                print(f"{stv:>5}{T * stv / 20.0:>17.1f}{e:>10.4f}")
            print("\n(секунды посчитаны в предположении, что зарр записан на 20 Гц)")
            return

        if args.probe:
            print("\n" + "=" * 60)
            print("ПЕРЕБОР ФОРМАТА: ошибка генерации против датасетной")
            print("=" * 60)
            print(f"{'flip':>6}{'кроп':>7}{'склейка':>9}{'ошибка':>10}")
            best = (1e9, None)
            for fl in ("", "h", "w", "c", "hw", "hc", "wc", "hwc"):
                for cr in (True, False):
                    for ti in (True, False):
                        args.flip, args.center_crop, args.tiled = fl, cr, ti
                        bb = build_batch(zz, t_idx, tasks, st, proc, args,
                                         args.device)
                        _, _, ve, _ = model._build_vlm_inputs_embeds(
                            input_ids=bb["input_ids"], inputs_embeds=None,
                            pixel_values=bb.get("pixel_values"),
                            pixel_attention_mask=bb.get("pixel_attention_mask"),
                            image_hidden_states=None)
                        h_, am = ve, bb.get("attention_mask")
                        hist = None
                        for _ in range(nb):
                            lg = model._predict_next_block_logits(
                                vlm_inputs_embeds=h_, attention_mask=am,
                                history_tokens=hist).float()
                            c = lg.argmax(-1)
                            hist = c if hist is None else torch.cat([hist, c], 1)
                        e = float(err(dec(to_levels(hist))).median())
                        print(f"{fl or '-':>6}{str(cr):>7}{str(ti):>9}{e:>10.4f}")
                        if e < best[0]:
                            best = (e, (fl, cr, ti))
            print(f"\nлучшее: flip={best[1][0] or '-'} кроп={best[1][1]} "
                  f"склейка={best[1][2]}, ошибка {best[0]:.4f}")
            print("Ниже 0.05 — формат найден. Около 0.19 у всех — дело не в\n"
                  "формате, а в разрешении 128 против обучающих 256.")
            return

        print(f"САНИТАРНАЯ ПРОВЕРКА — |ошибка| генерации против датасетной: "
              f"медиана {float(e_base.median()):.4f} размаха")
        assert float(e_base.median()) < args.sanity_max, (
            "формат наблюдений не совпал с обучающим: проверить порядок каналов "
            "(--bgr), кроп (--center-crop), разрешение и текст инструкции")

        h_true = latent_from_codes(E, torch.stack(
            [torch.as_tensor(np.asarray(tok.encode(a_true, embodiment_ids=args.embodiment)),
                             device=args.device, dtype=torch.long)], 0)[0]
            .reshape(B, L, P).transpose(1, 2))       # (B,P,L): раскладка поуровневая

        rng = torch.Generator(device=args.device).manual_seed(1)
        rows = {m: {k: [] for k in ("stale", "loc", "glob", "orc", "base")}
                for m in ("local", "on-policy")}
        infl = torch.zeros(P, P)                      # матрица влияния p -> q
        n_infl = 0

        lg0 = blk(None)                               # coarse-логиты
        for mode in ("local", "on-policy"):
            for _ in range(args.n_pos):
                p = int(torch.randint(P, (1,), generator=rng, device=args.device))
                cur = codes[:, p, 0]
                if mode == "local":
                    d = torch.cdist(E[0][cur], E[0])
                    nb = d.topk(args.knn + 1, largest=False).indices[:, 1:]
                    v = nb[torch.arange(B, device=args.device),
                           torch.randint(args.knn, (B,), generator=rng,
                                         device=args.device)]
                else:
                    tk = lg0[:, p].topk(args.topk + 1, -1).indices[:, 1:]
                    v = tk[torch.arange(B, device=args.device),
                           torch.randint(args.topk, (B,), generator=rng,
                                         device=args.device)]

                c0 = codes[:, :, 0].clone()
                c0[:, p] = v
                # BAR достраивает ВСЕ оставшиеся блоки при новом уровне 0
                rest = gen_from(c0, nb - bpl)
                cn = to_levels(rest)
                c1n, c2n = cn[:, :, 1], cn[:, :, 2]

                # влияние: где сменился top-1 первого fine-уровня
                infl[p] += (c1n != codes[:, :, 1]).float().mean(0).cpu()
                n_infl += 1

                stale = codes.clone(); stale[:, p, 0] = v
                loc = stale.clone(); loc[:, p, 1] = c1n[:, p]; loc[:, p, 2] = c2n[:, p]
                glob = torch.stack([c0, c1n, c2n], -1)
                orc = greedy_suffix(E, stale, h_true, p, 0)

                for k, cc in (("stale", stale), ("loc", loc), ("glob", glob),
                              ("orc", orc), ("base", codes)):
                    rows[mode][k].append(err(cc if k == "base" else cc).cpu().numpy()
                                         if k == "base" else
                                         err(dec(cc)).cpu().numpy())

    # ---------- отчёт ----------
    print("\n" + "=" * 76)
    print("РЕМОНТ СУФФИКСА: ОШИБКА ДЕЙСТВИЯ ОТНОСИТЕЛЬНО ДАТАСЕТНОГО")
    print("=" * 76)
    print(f"{'режим':>12}{'база':>9}{'stale':>9}{'BAR лок.':>10}{'BAR глоб.':>11}"
          f"{'оракул':>9}{'дост. ремонт':>14}{'R_BAR':>8}")
    for mode in ("local", "on-policy"):
        m = {k: float(np.median(np.concatenate(v))) for k, v in rows[mode].items()}
        gap = m["stale"] - m["orc"]
        r = (m["stale"] - m["loc"]) / gap if abs(gap) > 1e-9 else float("nan")
        print(f"{mode:>12}{float(np.median(np.concatenate(rows[mode]['base']))):>9.4f}"
              f"{m['stale']:>9.4f}{m['loc']:>10.4f}{m['glob']:>11.4f}{m['orc']:>9.4f}"
              f"{gap:>14.4f}{r:>8.2f}")

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
