"""K-6b: нужна ли BAR последовательность блоков, или уровни предсказуемы сразу.

ВОПРОС. BAR тратит три полных прохода по башне (147 из 195 мс, 75% вызова),
потому что RVQ последовательна: код уровня 2 квантует остаток уровня 1. Это
модельное решение BAR, а НЕ математическая необходимость — коды суть функция
действий, значит в принципе предсказуемы из наблюдения напрямую. Меряем, во
сколько обходится отказ от обусловливания.

БАЗОВАЯ ЛИНИЯ — NULL, А НЕ ПЕРЕМЕШИВАНИЕ. Перемешанные коды предыдущих уровней
это не отсутствие информации, а ШУМ, который голова обязана научиться
игнорировать, тратя на это ёмкость. Тогда «параллельный» режим выглядит хуже,
чем есть. Поэтому основная величина —

    dCE_g = CE(k_g | h, NULL) - CE(k_g | h, настоящие k_<g)

где NULL — выделенный постоянный код. Перемешивание остаётся ВТОРИЧНЫМ
контролем (перемешиваем внутри задачи, иначе вход заведомо неестественный).
Обе головы одинаковы по архитектуре и по инициализации: перед каждой ставится
один и тот же сид, иначе разница окажется отчасти разницей случая.

Строго говоря это НЕ условная взаимная информация: головы не являются
оптимальными оценщиками энтропии. Правильное название — прирост кросс-энтропии
зонда.

РАЗРЕЖЕННОСТЬ СЧИТАЕТСЯ ПОЭКЗЕМПЛЯРНО. FINDINGS §A0: правка грубого кода
меняет тонкие уровни в среднем в 4.79 позиции из 16. Но НАБОР этих позиций у
разных наблюдений СВОЙ. Усреднение dCE по абсолютной позиции сделало бы
поэкземплярно разреженную зависимость равномерной, и вывод «размазано, нужен
новый токенизатор» оказался бы ложным. Поэтому dCE сохраняется по каждому
(наблюдение, позиция), а концентрация меряется ВНУТРИ наблюдения.

ГЛАВНАЯ МЕТРИКА — ПОЛНАЯ СБОРКА, А НЕ ПОДМЕНА ОДНОГО УРОВНЯ. Заменять один
уровень при истинных остальных значит предполагать идеальный грубый код,
которого на инференсе нет. Считаются целиком:
    эксперт          истинные k0,k1,k2 из токенизатора
    BAR-последов.    все три уровня, предсказанные самой BAR
    один проход NULL k0 из первого блока BAR, k1,k2 из h-only голов
    дешёвая условн.  k0 из первого блока, k1|k0, k2|k0,k1 из дешёвых голов
Сравнение — с действием ДАТАСЕТА и с экспертным декодом; поза как RMS в долях
размаха, схват отдельно как доля неверных шагов.

РАЗБИЕНИЕ ПО ЭПИЗОДАМ. С одного эпизода берётся около десяти наблюдений, и
случайное разбиение наблюдений пустило бы соседние состояния в train и в
validation. Делим по ЭПИЗОДАМ: train/val/test, val только для отбора эпохи,
test только для итоговых чисел.

ОФСЕТ ПО ЗАДАЧАМ. Единый pos_offset — абляция, а не протокол: официальный
скрипт задаёт 3 или 4 отдельно для каждой из сорока задач, и k4b0_padding_probe
показал, что выбор МЕНЯЕТ план (оракул 0.941 против 0.872). Берём из таблицы.

ПОЧЕМУ generate, А НЕ ПРЯМОЙ ВЫЗОВ ПЕРВОГО БЛОКА. Прямой вызов втрое дешевле,
но требует самим собрать аргументы внутреннего метода — место, где легко
незаметно разойтись с настоящим путём вывода. Экономия минут на двадцатиминутной
задаче того не стоит; форма входа проверяется assert-ом.

Запуск:
    python3 experiments/k6b_delta_h.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6b_delta_h.py --ckpt <ckpt> --n-obs 128 --n-ep 32 \\
        --seeds 3 --out data/k6b_smoke.json          # сначала дым
"""

import argparse
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20


# ---------------------------------------------------------------------------
# головы-зонды
# ---------------------------------------------------------------------------

def _make_head(d, n_prev, n_codes, n_pos, hid, seed, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)          # ОДИНАКОВАЯ инициализация у всех режимов
    emb = 64

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            # СВОЯ ТАБЛИЦА НА КАЖДЫЙ УРОВЕНЬ: индекс 17 в книге 0 и в книге 1 —
            # разные векторы, общая таблица искусственно связала бы их и
            # ЗАНИЗИЛА полезность предыдущих уровней. Лишняя строка — NULL.
            self.code_emb = nn.ModuleList(
                [nn.Embedding(n_codes + 1, emb) for _ in range(n_prev)])
            self.pos_emb = nn.Embedding(n_pos, emb)
            self.net = nn.Sequential(
                nn.Linear(d + emb * n_prev + emb, hid), nn.GELU(),
                nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, n_codes))

        def forward(self, x, pr):
            b, p, _ = x.shape
            parts = [x, self.pos_emb(torch.arange(p, device=x.device))
                     .unsqueeze(0).expand(b, -1, -1)]
            for g, e in enumerate(self.code_emb):
                parts.append(e(pr[..., g]))
            return self.net(torch.cat(parts, dim=-1))

    return Head().to(device)


def train_head(X, prev, y, n_codes, splits, seed=0, epochs=30, hid=512,
               device="cpu"):
    """Обучить p(k | X, prev). Возвращает CE по (пример, позиция) на TEST.

    splits: (idx_train, idx_val, idx_test) — индексы, разбитые ПО ЭПИЗОДАМ.
    Эпоха отбирается по val, итоговые числа считаются на test, который не
    участвовал ни в обучении, ни в отборе.
    """
    import torch
    import torch.nn as nn

    dev = torch.device(device)
    X = torch.as_tensor(X, dtype=torch.float32)
    prev = torch.as_tensor(prev, dtype=torch.long)
    y = torch.as_tensor(y, dtype=torch.long)
    itr, iva, ite = (torch.as_tensor(s, dtype=torch.long) for s in splits)

    m = _make_head(X.shape[-1], prev.shape[-1], n_codes, X.shape[1], hid,
                   seed, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(reduction="none")
    Xtr, Ptr, Ytr = X[itr].to(dev), prev[itr].to(dev), y[itr].to(dev)
    Xva, Pva, Yva = X[iva].to(dev), prev[iva].to(dev), y[iva].to(dev)
    Xte, Pte, Yte = X[ite].to(dev), prev[ite].to(dev), y[ite].to(dev)
    g = torch.Generator().manual_seed(seed)
    bs, best = 256, None

    for _ in range(epochs):
        m.train()
        order = torch.randperm(Xtr.shape[0], generator=g)
        for i in range(0, len(order), bs):
            j = order[i:i + bs]
            opt.zero_grad()
            lg = m(Xtr[j], Ptr[j])
            lossf(lg.reshape(-1, n_codes), Ytr[j].reshape(-1)).mean().backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = lossf(m(Xva, Pva).reshape(-1, n_codes),
                      Yva.reshape(-1)).mean().item()
        if best is None or v < best[0]:
            with torch.no_grad():
                lg = m(Xte, Pte)
                ce = lossf(lg.reshape(-1, n_codes),
                           Yte.reshape(-1)).reshape(Yte.shape)
            best = (v, ce.cpu().numpy(), lg.argmax(-1).cpu().numpy(),
                    {k: p.detach().clone() for k, p in m.state_dict().items()})
    return dict(val=best[0], ce=best[1], pred=best[2], state=best[3], model=m)


def prev_variant(prev, mode, n_codes, group, seed):
    """TRUE / NULL / SHUFFLED при одинаковой форме входа."""
    if mode == "true":
        return prev
    if mode == "null":
        return np.full_like(prev, n_codes)     # выделенный постоянный код
    if mode == "shuffled":
        # ПЕРЕМЕШИВАЕМ ВНУТРИ ГРУППЫ (задачи): код от другой задачи дал бы
        # заведомо неестественный вход. НО при маленьких группах перестановка
        # почти тождественна (группа из одного элемента — вовсе тождество), и
        # контроль вырождается: на дымовом прогоне SHUF-TRUE вышло +0.004 при
        # NULL-TRUE +0.127, то есть перемешанные коды работали КАК НАСТОЯЩИЕ.
        # Поэтому доля реально сдвинутых записей ИЗМЕРЯЕТСЯ, и при вырождении
        # контроль расширяется на всю выборку с явным предупреждением.
        rng = np.random.default_rng(seed)

        def _perm_within(g):
            out = prev.copy()
            moved = 0
            for gv in np.unique(g):
                m = np.where(g == gv)[0]
                pm = m[rng.permutation(len(m))]
                out[m] = prev[pm]
                moved += int((pm != m).sum())
            return out, moved / max(len(g), 1)

        out, frac = _perm_within(group)
        if frac < 0.5:
            out2, frac2 = _perm_within(np.zeros(len(group), int))
            print(f"    ВНИМАНИЕ: перестановка внутри задач сдвинула лишь "
                  f"{frac:.0%} записей (группы слишком малы) — контроль "
                  f"расширен на всю выборку, сдвинуто {frac2:.0%}. Вход стал "
                  f"менее естественным, читать контроль с оговоркой.")
            return out2
        return out
    raise ValueError(mode)


def concentration(dce):
    """Насколько зависимость сосредоточена ВНУТРИ наблюдения.

    dce: (N, POS). Считается по каждому наблюдению отдельно, потому что набор
    зависимых позиций у разных наблюдений СВОЙ, и среднее по абсолютной
    позиции его размазало бы.
    """
    pos = np.clip(dce, 0, None)
    tot = pos.sum(axis=1)
    ok = tot > 1e-9
    if not ok.any():
        return dict(top5_share=float("nan"), n80=float("nan"), frac_active=0.0)
    srt = np.sort(pos[ok], axis=1)[:, ::-1]
    top5 = srt[:, :5].sum(axis=1) / tot[ok]
    cum = np.cumsum(srt, axis=1) / tot[ok][:, None]
    n80 = (cum < 0.8).sum(axis=1) + 1
    return dict(top5_share=float(np.median(top5)),
                n80=float(np.median(n80)),
                frac_active=float(ok.mean()))


# ---------------------------------------------------------------------------
# самопроверки
# ---------------------------------------------------------------------------

def selftest(epochs=25):
    rng = np.random.default_rng(0)
    N, d, C = 1800, 32, 16
    h = rng.normal(size=(N, N_POS, d)).astype(np.float32)
    k1 = rng.integers(0, C, size=(N, N_POS, 1))
    grp = rng.integers(0, 4, size=N)
    ep = np.arange(N) // 10                      # по десять наблюдений на «эпизод»
    sp = split_by_episode(ep, seed=0)

    def run(y, mode, seed=0):
        return train_head(h, prev_variant(k1, mode, C, grp, 7), y, C, sp,
                          seed=seed, epochs=epochs)["ce"]

    W = rng.normal(size=(d, C))
    y_indep = (h @ W).argmax(-1)

    # 1. ЗАВИСИМОСТИ НЕТ -> dCE около нуля.
    d0 = float(run(y_indep, "null").mean() - run(y_indep, "true").mean())
    assert abs(d0) < 0.15, f"без зависимости dCE ~ 0, получено {d0:+.3f}"

    # 2. ЗАВИСИМОСТЬ В ФИКСИРОВАННЫХ ПЯТИ ПОЗИЦИЯХ -> видна и локализована.
    fixed = np.array([2, 5, 7, 11, 14])
    y_fix = y_indep.copy()
    y_fix[:, fixed] = k1[:, fixed, 0]
    gap_fix = run(y_fix, "null") - run(y_fix, "true")
    assert gap_fix[:, fixed].mean() > 1.0, "фиксированная зависимость не видна"
    other = np.setdiff1d(np.arange(N_POS), fixed)
    assert abs(gap_fix[:, other].mean()) < 0.25, "ложный разрыв вне зависимых"

    # 3. ГЛАВНЫЙ ТЕСТ: зависимость в ПЯТИ СЛУЧАЙНЫХ позициях У КАЖДОГО примера.
    #    Средняя карта по абсолютной позиции обязана стать ПЛОСКОЙ, а
    #    поэкземплярная концентрация — по-прежнему находить зависимость.
    #    Именно этот случай отличает «зависимость размазана» от «её support
    #    перемещается», и именно его старая версия зонда не различала.
    y_var = y_indep.copy()
    r2 = np.random.default_rng(3)
    for i in range(N):
        p = r2.choice(N_POS, 5, replace=False)
        y_var[i, p] = k1[i, p, 0]
    gap_var = run(y_var, "null") - run(y_var, "true")
    per_pos = gap_var.mean(axis=0)
    flat = per_pos.std() / max(per_pos.mean(), 1e-9)
    con = concentration(gap_var)
    assert flat < 0.35, \
        f"средняя карта обязана быть плоской при плавающем support, CV={flat:.2f}"
    assert con["top5_share"] > 0.55, \
        (f"поэкземплярная концентрация обязана находить зависимость, "
         f"top5={con['top5_share']:.2f} — иначе плавающий support неотличим "
         f"от равномерного")
    con_fix = concentration(gap_fix)

    print("самопроверка пройдена:")
    print(f"  без зависимости dCE = {d0:+.3f}")
    print(f"  фиксированные 5 позиций: dCE там {gap_fix[:, fixed].mean():.2f}, "
          f"вне {gap_fix[:, other].mean():+.3f}, top5 {con_fix['top5_share']:.2f}")
    print(f"  ПЛАВАЮЩИЕ 5 позиций: средняя карта плоская (CV {flat:.2f}), "
          f"но поэкземплярно top5 = {con['top5_share']:.2f}, "
          f"позиций до 80% разрыва: {con['n80']:.0f}")
    print("  контроль NULL, а не перемешивание; у каждого уровня своя таблица;")
    print("  разбиение по эпизодам, test не участвует в отборе эпохи")


class _Tee:
    """Дублировать stdout в файл. Печать остаётся на экране, но прогон
    полностью сохраняется: числа зонда рождаются один раз и переигрываются
    дорого, поэтому терять вывод нельзя. Пишем с немедленным сбросом, чтобы
    лог был читаем и у оборванного процесса."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()


def split_by_episode(ep, seed=0, fr=(0.7, 0.15)):
    """train/val/test ПО ЭПИЗОДАМ. С одного эпизода берётся ~10 наблюдений,
    и случайное разбиение наблюдений пустило бы соседние состояния в обе
    части."""
    u = np.unique(ep)
    r = np.random.default_rng(seed).permutation(len(u))
    n1, n2 = int(len(u) * fr[0]), int(len(u) * (fr[0] + fr[1]))
    s = [set(u[r[:n1]]), set(u[r[n1:n2]]), set(u[r[n2:]])]
    return tuple(np.where(np.isin(ep, list(x)))[0] for x in s)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-obs", type=int, default=3000)
    ap.add_argument("--n-ep", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--pos-offset", type=int, default=None,
                    help="единый офсет — АБЛЯЦИЯ, а не протокол")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None,
                    help="куда дублировать вывод; по умолчанию рядом с --out "
                         "с расширением .log")
    args = ap.parse_args()

    log_path = args.log or (os.path.splitext(args.out)[0] + ".log"
                            if args.out else None)
    if log_path:
        import datetime
        sys.stdout = _Tee(log_path)
        print(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====")
        print("$ " + " ".join(sys.argv))

    if args.selftest:
        selftest()
        return
    selftest()
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

    # ОФСЕТ ПО ЗАДАЧАМ, а не единый
    off_by_task = None
    if args.pos_offset is None:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; постройте "
                             f"k4b0_offset_table.py или задайте --pos-offset "
                             f"(это АБЛЯЦИЯ, а не протокол)")
        tb = json.load(open(args.offset_table))
        # ключ таблицы — ОПИСАНИЕ задачи (k4b0_offset_table.py строит
        # {описание: {suite, task_id, pos_offset}}), а по нему мы и ищем.
        off_by_task = {k: int(v["pos_offset"]) for k, v in tb["tasks"].items()}

    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    grabbed = []
    model.action_lm_head.register_forward_hook(
        lambda m, i, o: grabbed.append(i[0].detach().float().cpu()))

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
    assert im1[0].shape[0] == im1[0].shape[1], f"неквадратная картинка {im1[0].shape}"
    print(f"собрано {N} наблюдений, {len(np.unique(epi))} эпизодов, "
          f"картинка {hw}x{hw}")

    # ПРЕОБРАЗОВАНИЕ ОТ ФАКТИЧЕСКОГО РАЗМЕРА. Обучающий протокол берёт 87.5%
    # поля зрения (k3_bar_suffix_repair.py:210). Жёсткое int(224*0.875)=196
    # верно только для картинок 224 из среды; на кадрах 256 из датасета оно
    # обрезало бы 76.6% вместо 87.5%.
    tf = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])
    # И НИКАКОГО РАЗВОРОТА КАНАЛОВ: `[:, :, ::-1]` в цикле оценки нужен
    # картинкам из robosuite (BGR). PIL.convert("RGB") уже даёт RGB.

    # --- истинные коды --------------------------------------------------------
    a_codec = np.asarray(act, np.float64).copy()
    a_codec[..., :-1] /= max_act_q[..., :-1]
    a_codec[..., -1] *= -1
    a_codec = np.clip(a_codec, -1.0, 1.0)        # как в обучающем preprocessing
    toks = np.asarray(proc.action_processor.encode(a_codec), np.int64)
    assert toks.shape[1] == N_POS * N_LEVEL, f"токенов {toks.shape[1]}"
    K_true = toks.reshape(N, N_LEVEL, N_POS)

    # --- скрытые состояния и коды самой BAR ----------------------------------
    H, K_bar = [], []
    # СОСТОЯНИЕ ИЗ ДАТАСЕТА УЖЕ ОБРАБОТАНО. process_state нужен только сырым
    # наблюдениям среды: он ждёт 9 измерений (схват 2, позиция 3, кватернион 4)
    # и переводит кватернион в ось-угол. В LeRobot состояние уже в целевом
    # 8-мерном виде, и k4b0_build_router_dataset.py:575 нормирует его напрямую.
    ST_RAW = np.asarray(st, np.float64)
    if ST_RAW.shape[1] == len(STATE_Q01) + 1:
        ST_RAW = process_state(ST_RAW)           # на случай сырого формата
    assert ST_RAW.shape[1] == len(STATE_Q01), (
        f"состояние имеет {ST_RAW.shape[1]} измерений, нормировка ждёт "
        f"{len(STATE_Q01)}")
    st_n = (ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    # ГРУППИРОВКА ПО ОФСЕТУ: разные задачи требуют разного pos_offset, а он
    # передаётся на весь батч.
    offs = np.array([args.pos_offset if args.pos_offset is not None
                     else off_by_task.get(tsk[i], 4) for i in range(N)])
    # ОБХОД ПО ГРУППАМ ОФСЕТА, а не срезами общего порядка. Прежняя версия
    # брала срез длиной batch и отбрасывала из него чужой офсет, но индекс
    # всё равно двигался на batch — отброшенные наблюдения не обрабатывались
    # НИКОГДА и оставались нулями в Hm.
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
          for j, gi in enumerate(sel):
            m = prompt_template(
                st_n[gi], None, tsk[gi],
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=n_codes,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
          texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
          batch = proc(text=texts, images=[[image[j].numpy()] for j in range(b)],
                     return_tensors="pt", padding=True, padding_side="left",
                     action_processor_kwargs={"embodiment_ids": 0})
          batch = dict_apply(lambda x: x.to(dev, dtype), batch)
          grabbed.clear()
          with torch.no_grad():
            tk = model.generate(**batch, position_offset=po, do_sample=False,
                                initial_position_shift=1)
          assert len(grabbed) == N_LEVEL, f"голова сработала {len(grabbed)} раз"
          g0 = grabbed[0]
          assert g0.shape[1] == N_POS, (
            f"на первом блоке ждали {N_POS} позиций на входе action_lm_head, "
            f"получено {g0.shape[1]}")
          H.append((sel, g0.numpy(), tk.cpu().numpy().reshape(b, N_LEVEL, N_POS)))
          if done_cnt % (args.batch * 50) < args.batch:
            print(f"  {done_cnt}/{N} (офсет {po})", flush=True)
    assert done_cnt == N, f"обработано {done_cnt} из {N} наблюдений"
    Hm = np.zeros((N, N_POS, H[0][1].shape[-1]), np.float32)
    K_bar = np.zeros((N, N_LEVEL, N_POS), np.int64)
    for sel, hh, kk in H:
        Hm[sel] = hh
        K_bar[sel] = kk
    print(f"скрытые состояния: {Hm.shape}; совпадение кодов BAR с истинными: "
          f"{(K_bar == K_true).mean():.1%}")

    splits = split_by_episode(epi, seed=args.seed)
    print(f"разбиение по эпизодам: train {len(splits[0])}, val {len(splits[1])}, "
          f"test {len(splits[2])}")
    grp = np.array([hash(t) % 997 for t in tsk])

    # --- dCE по уровням, усреднение по сидам ---------------------------------
    res, heads = {}, {}
    for lvl in (1, 2):
        prev = np.transpose(K_true[:, :lvl, :], (0, 2, 1))
        y = K_true[:, lvl, :]
        acc = {}
        for mode in ("true", "null", "shuffled"):
            ces, preds = [], []
            for s in range(args.seeds):
                r = train_head(Hm, prev_variant(prev, mode, n_codes, grp, 7 + s),
                               y, n_codes, splits, seed=s, epochs=args.epochs,
                               device=args.device)
                ces.append(r["ce"]); preds.append(r["pred"])
                if mode == "true" and s == 0:
                    heads[(lvl, "true")] = r
                if mode == "null" and s == 0:
                    heads[(lvl, "null")] = r
            acc[mode] = (np.mean(ces, axis=0), preds[0])
        dce = acc["null"][0] - acc["true"][0]
        dsh = acc["shuffled"][0] - acc["true"][0]
        con = concentration(dce)
        res[f"level{lvl}"] = dict(
            dce_mean=float(dce.mean()), dce_shuffled_mean=float(dsh.mean()),
            per_position=dce.mean(axis=0).tolist(), **con)
        print(f"\n=== уровень {lvl}")
        print(f"  dCE (NULL−TRUE)      {dce.mean():+.3f}")
        print(f"  контроль (SHUF−TRUE) {dsh.mean():+.3f}")
        print(f"  по абсолютным позициям: " +
              " ".join(f"{v:5.2f}" for v in dce.mean(axis=0)))
        print(f"  ПОЭКЗЕМПЛЯРНО: медианная доля топ-5 позиций {con['top5_share']:.2f}, "
              f"позиций до 80% разрыва {con['n80']:.0f}, "
              f"наблюдений с зависимостью {con['frac_active']:.0%}")

    # --- полная сборка: главная метрика --------------------------------------
    ite = splits[2]

    def decode(codes):
        d = proc.action_processor.decode(codes.reshape(len(codes), -1).tolist())
        return np.asarray(d if isinstance(d, np.ndarray) else d[0], np.float64)

    a_ref = a_codec[ite]
    variants = {"эксперт (истинные коды)": K_true[ite],
                "BAR последовательная": K_bar[ite]}
    for name, mode in (("один проход, NULL-головы", "null"),
                       ("дешёвые условные головы", "true")):
        Kx = K_bar[ite].copy()                   # k0 из первого блока — он есть
        for lvl in (1, 2):
            Kx[:, lvl, :] = heads[(lvl, mode)]["pred"]
        variants[name] = Kx
    print("\n" + "=" * 74)
    print(f"  {'вариант':<28}{'поза RMS':>11}{'схват, доля':>13}{'к эксперту':>12}")
    dec_ref = decode(K_true[ite])
    rng_pose = float(a_ref[..., :6].max() - a_ref[..., :6].min())
    for name, Kx in variants.items():
        d = decode(Kx)
        pose = float(np.sqrt(((d[..., :6] - a_ref[..., :6]) ** 2).mean())) / rng_pose
        grip = float((np.sign(d[..., 6]) != np.sign(a_ref[..., 6])).mean())
        vs = float(np.sqrt(((d[..., :6] - dec_ref[..., :6]) ** 2).mean())) / rng_pose
        res[name] = dict(pose_rms=pose, gripper_frac=grip, vs_expert=vs)
        print(f"  {name:<28}{pose:>11.4f}{grip:>13.1%}{vs:>12.4f}")

    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    print("  «один проход, NULL» ≈ «дешёвые условные» — блоки не нужны.")
    print("  Хуже, но поэкземплярная концентрация высока (топ-5 > ~0.6) —")
    print("  нужна дешёвая голова на мягком coarse, а не проход башни.")
    print("  Хуже и концентрация низка — оправдан новый токенизатор.")
    print("  Сравнивать надо с «BAR последовательная», а не с экспертом:")
    print("  эксперт недостижим, он видит истинные действия.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        res["meta"] = dict(ckpt=args.ckpt, n_obs=N, n_episodes=int(len(np.unique(epi))),
                           n_codes=n_codes, seeds=args.seeds, epochs=args.epochs,
                           image_hw=int(hw), offsets=sorted(set(offs.tolist())))
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
