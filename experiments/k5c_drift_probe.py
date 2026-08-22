"""K-5c: прямое измерение устаревания сгенерированного чанка действий.

ЗАЧЕМ. Существующие методы адаптивного горизонта выбирают H по КОСВЕННЫМ
признакам: AAC (CVPR 2026, arXiv 2604.04161) — по энтропии действий,
AutoHorizon (arXiv 2602.21445) — по весам self-attention, причём явно без
ground truth («does not use an explicit ground-truth signal… no supervised
training phase»). Обе валидируются только конечным успехом. Величина, которую
они якобы оценивают — насколько долго уже сгенерированный план остаётся
пригодным, — никем не измерена напрямую. Этот скрипт её меряет.

ЧТО ИМЕННО МЕРЯЕТСЯ. В момент t политика выдаёт чанк C_t длиной 20. Пусть
план исполняется открытым циклом. Тогда для абсолютного момента T = t + j
есть два действия: устаревшее C_t[j] и свежее C_T[0], полученное политикой из
фактически достигнутого состояния. Их расхождение

    D(j) = d( C_t[j], C_T[0] )

и есть устаревание. Оно ЧИСТОЕ ПО ПОСТРОЕНИЮ: при j = 0 обе величины
получены из одного состояния тем же жадным декодером (eval_libero.py:207,
do_sample=False), поэтому D(0) ≡ 0 тождественно и никакой ошибки генерации в
D не подмешано. Обратная сторона: разделить «плохо сгенерировал» и «устарело»
этим инструментом НЕЛЬЗЯ — для связи с исходом нужна отдельная калибровка
ветвлением.

ПОЗИЦИОННЫЙ КОНФАУНД — ГЛАВНАЯ ЛОВУШКА. Сравнивая C_t[j] с C_T[0], мы
сравниваем позицию j одного чанка с позицией 0 другого. Если модель
систематически ведёт себя на разных позициях по-разному (а профиль декодера у
нас по позициям неоднороден), рост D(j) окажется артефактом позиции, а не
устареванием. Поэтому вызов политики делается КАЖДЫЙ ШАГ, все чанки остаются
живыми, и на каждый момент T копится ПОЛНАЯ ПОПАРНАЯ матрица расхождений между
всеми живыми происхождениями, а не только против позиции 0.

Различающая статистика — тёплицевость. Если расхождение вызвано только
устареванием, оно зависит лишь от РАЗНОСТИ свежестей |j_a - j_b|, и матрица
постоянна вдоль диагоналей. Позиционные эффекты дают структуру сверх разности:
d(0,5) перестаёт равняться d(10,15). См. toeplitz_deviation.

ТРАЕКТОРИЯ СБОРА. Внутри чанка исполнение всегда открытоцикловое, поэтому
смещения меряются на правильных состояниях при любом H_exec. Смещено другое —
РАСПРЕДЕЛЕНИЕ ПРОИСХОЖДЕНИЙ: при H_exec=20 точки генерации лежат на траектории
H=20, а разворачивать мы собираемся при H≈8. Поэтому прогонов должно быть два,
--exec-horizon 20 и 8, и кривые надо сравнить. Совпали — вопрос закрыт.

УСРЕДНЕНИЕ ВЫКЛЮЧЕНО ВСЕГДА. ActionEnsembler смешал бы планы разной свежести
именно в той величине, которую мы меряем.

МЕТРИКИ. Шесть дельт позы и схват в одну норму не складываются: схват
практически бинарен. Считаются раздельно — L2 по позе, L_inf по позе,
несогласие схвата — и отдельно НАКОПЛЕННОЕ расхождение позы вдоль плана,
потому что ошибка в приращениях интегрируется в ошибку положения, и ломает
задачу именно она.

Запуск:
    python3 experiments/k5c_drift_probe.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k5c_drift_probe.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --task-suite 10 --task-id 0 --n-envs 10 --k-set 1 \\
        --exec-horizon 20 --out data/k5c/10_t0_H20.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

CHUNK = 20          # длина чанка политики
POSE = 6            # первые шесть измерений — приращения позы
GRIP = 6            # индекс схвата


class DriftRecorder:
    """Бухгалтерия устаревания. Вынесена из цикла раскатки, чтобы её можно
    было проверить на синтетике с известным ответом без симулятора.

    Живёт так: на каждом шаге принимает свежий чанк, отдаёт строки для всех
    живых происхождений и копит попарную матрицу.
    """

    def __init__(self, n_envs, chunk=CHUNK):
        self.n_envs = n_envs
        self.chunk = chunk
        self.chunks = {}                      # origin_T -> (n_envs, chunk, 7)
        self.cum = {}                         # origin_T -> (n_envs, POSE)
        self.pair_sum = np.zeros((chunk, chunk))
        self.pair_cnt = np.zeros((chunk, chunk))
        self.rows = {k: [] for k in (
            "T", "origin", "j", "env", "pose_l2", "pose_linf",
            "grip_absdiff", "cum_pose_l2", "done")}

    def step(self, T, chunk_now, done_mask):
        """Принять чанк, порождённый в момент T, и записать всё, что он даёт.

        chunk_now: (n_envs, chunk, 7) в НОРМАЛИЗОВАННОМ пространстве политики,
        то есть до умножения на max_act_q. Нормировка на дисперсию делается
        offline, поэтому здесь сохраняются сырые расхождения.
        """
        self.chunks[T] = chunk_now
        self.cum[T] = np.zeros((self.n_envs, POSE))
        fresh = chunk_now[:, 0, :]                        # (n_envs, 7)

        # ОДИН список живых происхождений на оба прохода. Раньше он строился
        # дважды, и совпадение порядка держалось на честном слове.
        origins = [t for t in sorted(self.chunks) if T - t < self.chunk]
        offs = np.asarray([T - t for t in origins])
        live = np.stack([self.chunks[t][:, T - t, :] for t in origins], axis=0)

        # --- строки против свежего (позиция 0) --------------------------------
        for a, t in enumerate(origins):
            j = T - t
            d = live[a][:, :POSE] - fresh[:, :POSE]       # (n_envs, POSE)
            self.cum[t] += d
            self.rows["T"].append(np.full(self.n_envs, T))
            self.rows["origin"].append(np.full(self.n_envs, t))
            self.rows["j"].append(np.full(self.n_envs, j))
            self.rows["env"].append(np.arange(self.n_envs))
            self.rows["pose_l2"].append(np.linalg.norm(d, axis=1))
            self.rows["pose_linf"].append(np.abs(d).max(axis=1))
            self.rows["grip_absdiff"].append(
                np.abs(live[a][:, GRIP] - fresh[:, GRIP]))
            self.rows["cum_pose_l2"].append(
                np.linalg.norm(self.cum[t], axis=1))
            self.rows["done"].append(done_mask.astype(np.int8))

        # --- полная попарная матрица (контроль позиционного конфаунда) --------
        # Считаются только НЕзавершённые среды: после done траектория не имеет
        # смысла, а LIBERO продолжает шагать.
        #
        # ТОЛЬКО КОГДА ЖИВЫ ВСЕ ПРОИСХОЖДЕНИЯ. Иначе у разных пар оказываются
        # РАЗНЫЕ множества наблюдений: пара (0,19) видна лишь с T >= 19, а
        # (0,1) — с T >= 1. Тогда зависимость дрейфа от фазы задачи попадёт в
        # матрицу как ложная непёплицевость, и тест начнёт срабатывать не на
        # позицию, а на неоднородность эпизода. При полном наборе живых чанков
        # множество (T, среда) у всех пар одно и то же, и любая зависимость
        # вида g_T(лаг) сокращается при усреднении.
        keep = ~done_mask
        if keep.any() and len(origins) == self.chunk:
            p = live[:, keep, :POSE]                      # (n_live, n_keep, P)
            diff = p[:, None, :, :] - p[None, :, :, :]
            dist = np.linalg.norm(diff, axis=-1).sum(axis=-1)   # (n_live,n_live)
            cnt = int(keep.sum())
            ia, ib = np.meshgrid(offs, offs, indexing="ij")
            np.add.at(self.pair_sum, (ia.ravel(), ib.ravel()), dist.ravel())
            np.add.at(self.pair_cnt, (ia.ravel(), ib.ravel()),
                      np.full(ia.size, cnt, dtype=float))

        # чанки старше длины чанка больше никогда не понадобятся
        for t in [k for k in self.chunks if T - k >= self.chunk - 1]:
            self.chunks.pop(t, None)
            self.cum.pop(t, None)

    def arrays(self):
        out = {k: (np.concatenate(v) if v else np.zeros(0))
               for k, v in self.rows.items()}
        with np.errstate(invalid="ignore", divide="ignore"):
            out["pair_mean"] = np.where(self.pair_cnt > 0,
                                        self.pair_sum / self.pair_cnt, np.nan)
        out["pair_cnt"] = self.pair_cnt
        return out


def toeplitz_deviation(pair_mean):
    """Насколько попарная матрица НЕ объясняется одной лишь разностью свежестей.

    Чистое устаревание зависит только от |j_a - j_b|, то есть матрица постоянна
    вдоль диагоналей. Позиционные эффекты ломают это. Возвращается доля
    дисперсии, не объяснённая наилучшим тёплицевым приближением: 0 — конфаунда
    нет, ближе к 1 — расхождение определяется позициями, а не свежестью.
    """
    m = np.asarray(pair_mean, float)
    n = m.shape[0]
    ia, ib = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lag = np.abs(ia - ib)
    ok = np.isfinite(m) & (lag > 0)          # диагональ нулевая по построению
    if ok.sum() < 2:
        return float("nan")
    fit = np.zeros_like(m)
    for L in range(1, n):
        sel = ok & (lag == L)
        if sel.any():
            fit[sel] = m[sel].mean()
    resid = ((m[ok] - fit[ok]) ** 2).sum()
    total = ((m[ok] - m[ok].mean()) ** 2).sum()
    return float(resid / total) if total > 0 else 0.0


def action_trigger_horizon(chunk, tau):
    """Базлайн Action Trigger из AutoHorizon, arXiv 2602.21445.

    «sets the execution horizon as the first index where the difference between
    consecutive actions exceeds tau_a», tau_a ∈ {0.01, 0.05, 0.1, 0.3}. Метрика
    и нормировка в статье не указаны — берём L2 по позе в том же нормализованном
    пространстве, в котором считается D. Это НАША интерпретация, и она должна
    быть названа таковой при сравнении.

    chunk: (..., CHUNK, 7). Возвращает горизонт в [1, CHUNK].
    """
    d = np.linalg.norm(np.diff(chunk[..., :POSE], axis=-2), axis=-1)
    over = d > tau
    first = np.where(over.any(axis=-1), over.argmax(axis=-1) + 1, chunk.shape[-2])
    return np.clip(first, 1, chunk.shape[-2])


def curve(arr, key="pose_l2", drop_done=True):
    """Средняя кривая устаревания по смещению."""
    m = np.ones(len(arr["j"]), bool)
    if drop_done:
        m &= arr["done"] == 0
    out = []
    for j in range(CHUNK):
        s = m & (arr["j"] == j)
        out.append(float(arr[key][s].mean()) if s.any() else float("nan"))
    return np.array(out)


# ----------------------------------------------------------------------------
# самопроверки: каждая на синтетике с ЗАРАНЕЕ ИЗВЕСТНЫМ ответом
# ----------------------------------------------------------------------------

def _drive(gen, n_steps, n_envs=3):
    rec = DriftRecorder(n_envs)
    done = np.zeros(n_envs, bool)
    for T in range(n_steps):
        rec.step(T, gen(T), done)
    return rec.arrays()


def selftest():
    rng = np.random.default_rng(0)

    # 1. ТОЖДЕСТВО D(0) = 0. Свежее сравнивается само с собой.
    arr = _drive(lambda T: rng.normal(size=(3, CHUNK, 7)), 30)
    z = arr["pose_l2"][arr["j"] == 0]
    assert z.size and np.abs(z).max() < 1e-12, \
        f"D(0) обязано быть нулём тождественно, получено {np.abs(z).max():.3e}"

    # 2. ВОССТАНОВЛЕНИЕ ИЗВЕСТНОЙ КРИВОЙ. Чанк, порождённый в t, на позиции j
    #    предсказывает истину для t+j плюс устаревание alpha*j. Свежий (j=0)
    #    попадает в истину точно, значит D(j) обязано выйти alpha*j*sqrt(POSE).
    alpha = 0.01
    truth = rng.normal(size=(400, 3, 7))

    def gen_stale(T):
        c = np.stack([truth[T + j] for j in range(CHUNK)], axis=1)
        c[:, :, :POSE] += alpha * np.arange(CHUNK)[None, :, None]
        return c

    arr = _drive(gen_stale, 200)
    got = curve(arr)
    want = alpha * np.arange(CHUNK) * np.sqrt(POSE)
    assert np.allclose(got, want, atol=1e-9), \
        f"кривая устаревания не восстановлена:\n  получено {got[:5]}\n  ждали {want[:5]}"

    # 3. ЛИНЕЙНОЕ устаревание тёплицево: матрица зависит только от разности.
    dev = toeplitz_deviation(arr["pair_mean"])
    assert dev < 1e-9, f"чистое устаревание не должно ломать тёплицевость, {dev:.3e}"

    # 4. ПОЗИЦИОННЫЙ КОНФАУНД. Устаревания НЕТ вовсе, но у позиции 0 своё
    #    смещение. Наивная кривая против позиции 0 обязана показать ложный
    #    ненулевой уровень, а тёплицевость — сломаться. Это и есть тот
    #    артефакт, ради которого считается попарная матрица.
    bias = np.zeros(CHUNK)
    bias[0] = 0.5                      # позиция 0 особенная, остальные равны

    def gen_pos(T):
        c = np.stack([truth[T + j] for j in range(CHUNK)], axis=1)
        c[:, :, :POSE] += bias[None, :, None]
        return c

    arr_p = _drive(gen_pos, 200)
    got_p = curve(arr_p)
    assert got_p[0] == 0.0, "j=0 всё равно обязан быть нулём"
    assert np.all(got_p[1:] > 0.4), \
        ("наивная кривая обязана поймать позиционное смещение как ложное "
         f"устаревание, получено {got_p[1:5]}")
    dev_p = toeplitz_deviation(arr_p["pair_mean"])
    assert dev_p > 0.5, \
        (f"тёплицевость обязана сломаться на позиционном конфаунде, "
         f"получено {dev_p:.3f} — контроль не работает")
    # и решающее: попарная матрица между позициями >0 расхождения не видит
    sub = arr_p["pair_mean"][1:, 1:]
    assert np.nanmax(np.abs(sub)) < 1e-9, \
        "между непривилегированными позициями расхождения быть не должно"

    # 4b. МНОЖЕСТВА НАБЛЮДЕНИЙ У ПАР ОБЯЗАНЫ СОВПАДАТЬ. Иначе неоднородность
    #     эпизода протекает в тест как ложная непёплицевость.
    cnt = arr_p["pair_cnt"]
    off_diag = cnt[~np.eye(CHUNK, dtype=bool)]
    assert off_diag.min() == off_diag.max() and off_diag.min() > 0, \
        (f"у пар разные множества наблюдений: от {off_diag.min()} до "
         f"{off_diag.max()} — состояние-зависимость протечёт в тёплицевость")

    # 4c. ПРЕДЕЛ МЕТОДА, НАЗВАННЫЙ ЯВНО. ЛИНЕЙНОЕ позиционное смещение даёт
    #     ‖b(j_a) − b(j_b)‖ = |j_a − j_b|·‖наклон‖, то есть ровно тёплицеву
    #     матрицу. Такой артефакт НЕОТЛИЧИМ от линейного устаревания, а
    #     линейный рост — именно то, что мы ожидаем увидеть. Закрывается
    #     только контролем со статической сценой (--static-steps), где
    #     истинное устаревание нулевое по построению.
    slope = 0.02

    def gen_linpos(T):
        c = np.stack([truth[T + j] for j in range(CHUNK)], axis=1)
        c[:, :, :POSE] += slope * np.arange(CHUNK)[None, :, None]
        return c

    arr_l = _drive(gen_linpos, 200)
    assert toeplitz_deviation(arr_l["pair_mean"]) < 1e-9, \
        "линейное позиционное смещение обязано выглядеть тёплицевым — если " \
        "тест его ловит, значит он ловит что-то другое, и его надо разобрать"
    assert np.allclose(curve(arr_l), slope * np.arange(CHUNK) * np.sqrt(POSE)), \
        "линейный артефакт обязан имитировать устаревание один в один"

    # 5. НАКОПЛЕННОЕ расхождение. При постоянном сдвиге на одно измерение
    #    накопление обязано расти линейно, а подельтная метрика — стоять.
    def gen_const(T):
        c = np.stack([truth[T + j] for j in range(CHUNK)], axis=1)
        c[:, 1:, 0] += 0.1
        return c

    arr_c = _drive(gen_const, 200)
    cum = curve(arr_c, "cum_pose_l2")
    assert abs(cum[1] - 0.1) < 1e-9 and abs(cum[5] - 0.5) < 1e-9, \
        f"накопление не линейно: {cum[:6]}"
    per = curve(arr_c, "pose_l2")
    assert abs(per[3] - 0.1) < 1e-9, f"подельтная метрика поплыла: {per[:6]}"

    # 6. ACTION TRIGGER. Чанк, у которого скачок ровно на переходе 3->4,
    #    обязан дать горизонт 4 при пороге ниже скачка и CHUNK при пороге выше.
    c = np.zeros((2, CHUNK, 7))
    c[:, 4:, 0] = 1.0
    assert (action_trigger_horizon(c, 0.5) == 4).all(), \
        f"Action Trigger промахнулся: {action_trigger_horizon(c, 0.5)}"
    assert (action_trigger_horizon(c, 2.0) == CHUNK).all(), \
        "при пороге выше любого скачка горизонт обязан быть максимальным"

    print("самопроверка пройдена:")
    print("  D(0) = 0 тождественно; известная кривая alpha*j восстановлена точно")
    print("  чистое устаревание тёплицево (dev < 1e-9)")
    print(f"  позиционный конфаунд ловится: наивная кривая лжёт "
          f"{got_p[1]:.2f}, тёплицевость {dev_p:.2f}")
    print("  накопленное расхождение растёт линейно, Action Trigger точен")


# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--exec-horizon", type=int, default=20,
                    help="горизонт ИСПОЛНЕНИЯ; политика всё равно вызывается "
                         "каждый шаг. Нужны оба прогона, 20 и 8: они дают "
                         "разное распределение точек генерации")
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--k-set", type=int, default=1)
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--static-steps", type=int, default=0,
                    help="шагов холостого хода с записью ДО раскатки. Даёт "
                         "позиционный пол D(j) при почти нулевом устаревании — "
                         "единственный контроль на ЛИНЕЙНЫЙ позиционный "
                         "артефакт, которого тёплицевость не видит. Сдвигает "
                         "эпизод, поэтому в основной развёртке держать 0 и "
                         "гонять отдельным диагностическим прогоном")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")
    if not args.out:
        raise SystemExit("нужен --out: сырые строки — главный продукт запуска")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)          # только корень, см. k5b: utils/ vs scripts/utils.py

    import torch
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       get_envs, process_state, prompt_template, seed_everything)

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])

    if args.pos_offset is not None:
        pos_off = args.pos_offset
    else:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; задайте --pos-offset")
        tb = json.load(open(args.offset_table))
        by = tb.get("offsets_by_suite", {})
        pos_off = int(by[args.task_suite][args.task_id])

    # СРЕДЫ ДО МОДЕЛИ: fork после инициализации CUDA вешает процесс.
    seed_everything(args.seed)
    envs, task_desc = get_envs(args.task_suite,
                               {"task_id": args.task_id, "image_size": 224},
                               args.n_envs)
    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    processor = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    print(f"=== suite {args.task_suite}, задача {args.task_id}, офсет {pos_off}")
    print(f"    «{task_desc}»   H_exec={args.exec_horizon}")

    def policy(obs):
        state = ((process_state(obs["state"]) - STATE_Q01)
                 / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0)
        im1 = tf(torch.tensor(
            obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2))
        im2 = tf(torch.tensor(
            obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
        ).permute(0, 3, 1, 2))
        image = torch.cat([im1, im2], dim=-1)
        msgs = []
        for i in range(args.n_envs):
            m = prompt_template(
                state[i], None, task_desc,
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = processor.apply_chat_template(msgs, add_generation_prompt=True)
        batch = processor(text=texts,
                          images=[[image[i].numpy()] for i in range(args.n_envs)],
                          return_tensors="pt", padding=True, padding_side="left",
                          action_processor_kwargs={"embodiment_ids": 0})
        batch = dict_apply(lambda x: x.to(dev, dtype), batch)
        with torch.no_grad():
            toks = model.generate(**batch, position_offset=pos_off,
                                  do_sample=False, initial_position_shift=1)
            return np.asarray(processor.action_processor.decode(toks.tolist())[0])

    all_arr, static_arr, meta_rounds = [], [], []
    t_start = time.time()
    try:
        for k in range(args.k_set):
            seed_everything(args.seed + 1000 * k)
            rec = DriftRecorder(args.n_envs)
            obs = envs.reset(options=[{"init_state_id": i + k * args.n_envs}
                                      for i in range(args.n_envs)])
            reward = np.zeros(args.n_envs)
            done = np.zeros(args.n_envs, bool)
            dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * args.n_envs)
            for _ in range(args.waiting_steps):
                obs, r_, done, _ = envs.step(dummy)
                reward = np.clip(reward + r_, 0, 1)

            # --- КОНТРОЛЬ СО СТАТИЧЕСКОЙ СЦЕНОЙ ------------------------------
            # Единственная проверка, закрывающая ЛИНЕЙНЫЙ позиционный артефакт:
            # тёплицевость его не ловит (самотест 4c), потому что он выглядит
            # ровно как линейное устаревание. Пока рука стоит на месте от
            # холостых действий, наблюдение почти не меняется, значит истинное
            # устаревание около нуля, и любое ненулевое D(j) здесь — это
            # позиционный пол, который надо вычесть из основной кривой.
            #
            # Сцена не идеально статична: схват и физика доседают. Поэтому это
            # ВЕРХНЯЯ оценка пола, а не точное его значение.
            if args.static_steps:
                rec_s = DriftRecorder(args.n_envs)
                for Ts in range(args.static_steps):
                    rec_s.step(Ts, policy(obs), done)
                    obs, r_, done, _ = envs.step(dummy)
                    reward = np.clip(reward + r_, 0, 1)
                a_s = rec_s.arrays()
                a_s["round"] = np.full(len(a_s["T"]), k)
                static_arr.append(a_s)
                cs = curve(a_s)
                print(f"  раунд {k} статический пол: D(1)={cs[1]:.4f} "
                      f"D(8)={cs[8]:.4f} D(19)={cs[19]:.4f}", flush=True)

            exec_chunk, exec_origin, T = None, None, 0
            while not np.all(done) and T < args.max_steps:
                chunk_now = policy(obs)                   # (n_envs, CHUNK, 7)
                rec.step(T, chunk_now, done)
                # исполнитель обновляется раз в exec_horizon шагов; между
                # обновлениями план исполняется ОТКРЫТЫМ ЦИКЛОМ, иначе
                # устаревание нечего мерить
                if exec_chunk is None or T - exec_origin >= args.exec_horizon:
                    exec_chunk, exec_origin = chunk_now, T
                a = np.copy(exec_chunk[:, T - exec_origin, :])
                a[..., :-1] = a[..., :-1] * max_act_q[..., :-1]
                a[..., -1] = -a[..., -1]
                obs, r_, done, _ = envs.step(a)
                reward = np.clip(reward + r_, 0, 1)
                T += 1

            arr = rec.arrays()
            arr["round"] = np.full(len(arr["T"]), k)
            arr["success_by_env"] = (reward >= 1.0).astype(np.int8)
            all_arr.append(arr)
            meta_rounds.append(dict(round=k, steps=T,
                                    success=int((reward >= 1.0).sum())))
            c = curve(arr)
            print(f"  раунд {k}: шагов {T}, успех "
                  f"{int((reward >= 1.0).sum())}/{args.n_envs}, "
                  f"D(1)={c[1]:.4f} D(4)={c[4]:.4f} D(8)={c[8]:.4f} "
                  f"D(19)={c[19]:.4f}", flush=True)
    finally:
        try:
            envs.close()
        except Exception:
            pass

    keys = [k for k in all_arr[0] if k not in ("pair_mean", "pair_cnt",
                                               "success_by_env")]
    out = {k: np.concatenate([a[k] for a in all_arr]) for k in keys}
    out["pair_sum"] = sum(np.nan_to_num(a["pair_mean"]) * a["pair_cnt"]
                          for a in all_arr)
    out["pair_cnt"] = sum(a["pair_cnt"] for a in all_arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = np.where(out["pair_cnt"] > 0, out["pair_sum"] / out["pair_cnt"],
                      np.nan)
    out["success_by_env"] = np.stack([a["success_by_env"] for a in all_arr])

    c, ci = curve(out), curve(out, "cum_pose_l2")
    dev = toeplitz_deviation(pm)
    floor = None
    if static_arr:
        st = {k: np.concatenate([a[k] for a in static_arr])
              for k in ("T", "j", "done", "pose_l2", "cum_pose_l2",
                        "grip_absdiff", "origin", "env")}
        floor = curve(st)
        out["static_pose_l2_curve"] = floor
        out["static_cum_curve"] = curve(st, "cum_pose_l2")
    print("\n" + "=" * 74)
    hdr = f"  {'j':>3}{'D_поза':>11}{'D_накопл':>12}{'D_схват':>11}{'строк':>10}"
    print(hdr + (f"{'пол':>10}{'за вычетом':>12}" if floor is not None else ""))
    cg = curve(out, "grip_absdiff")
    for j in range(CHUNK):
        n = int(((out["j"] == j) & (out["done"] == 0)).sum())
        line = f"  {j:>3}{c[j]:>11.4f}{ci[j]:>12.4f}{cg[j]:>11.4f}{n:>10}"
        if floor is not None:
            line += f"{floor[j]:>10.4f}{c[j] - floor[j]:>12.4f}"
        print(line)
    print(f"\n  отклонение от тёплицевости: {dev:.3f}")
    if floor is not None:
        share = float(np.nanmean(floor[1:] / np.maximum(c[1:], 1e-12)))
        print(f"  доля позиционного пола в кривой: {share:.1%}")
        print("  Пол — ВЕРХНЯЯ оценка артефакта: сцена не идеально статична.")
    else:
        print("  ПОЛ НЕ ИЗМЕРЕН. Линейный позиционный артефакт неотличим от\n"
              "  линейного устаревания (самотест 4c), и тёплицевость его не\n"
              "  видит. Без --static-steps рост D(j) НЕЛЬЗЯ называть\n"
              "  устареванием: нужен хотя бы один диагностический прогон.")
    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    print("  D(0) обязано быть ровно нулём — иначе стенд недетерминирован.")
    print("  Если D растёт с j и отклонение от тёплицевости мало (< ~0.15),")
    print("  кривая отражает устаревание, и её можно предсказывать.")
    print("  Если отклонение велико, рост D — артефакт позиции внутри чанка,")
    print("  и никакого 'срока годности плана' мы не измерили.")
    print("  Если D почти плоская, устаревания на этом горизонте нет вовсе,")
    print("  и адаптивность обосновать нечем — ветку закрывать.")

    import hashlib
    out["meta"] = json.dumps(dict(
        ckpt=args.ckpt, suite=args.task_suite, task_id=args.task_id,
        task_description=task_desc, exec_horizon=args.exec_horizon,
        n_envs=args.n_envs, k_set=args.k_set, seed=args.seed,
        pos_offset=pos_off, dtype=args.dtype, rounds=meta_rounds,
        toeplitz_deviation=dev, minutes=(time.time() - t_start) / 60,
        self_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
    ), ensure_ascii=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\n  сохранено: {args.out} ({len(out['T'])} строк), "
          f"{(time.time() - t_start) / 60:.1f} мин")


if __name__ == "__main__":
    main()
