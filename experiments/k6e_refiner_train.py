"""K-6e: этап 1 — догоняет ли однопроходный уточнитель блочную авторегрессию.

ВОПРОС. K-6b показал, что двуслойный MLP на скрытых состояниях первого блока
отстаёт от BAR на 30% по ошибке позы (0.0315 против 0.0242), а обусловливание
на предыдущих уровнях RVQ даёт лишь +0.4%. K-6c показал, что латентность больше
не ограничение: в бюджет ускорения 1.4x влезает уточнитель ГЛУБЖЕ и ВДВОЕ ШИРЕ
полной экспертной башни, да ещё с доступом к префиксу (24 слоя x 1536, кэш в
трёх слоях = 8.76 мс).

Осталось ровно две гипотезы, и этот скрипт их различает:
    ёмкость   — MLP просто слаб, глубокий уточнитель разрыв закроет;
    контекст  — в h нет нужного, и помогает только доступ к префиксу VLM;
    ни то ни другое — упор в САМО h, и вот это было бы мандатом на новый
                      токенизатор.

СХЕМА РАЗВЁРТЫВАНИЯ, КОТОРУЮ ВОСПРОИЗВОДИМ. Тяжёлый проход даёт h и логиты
ПЕРВОГО блока, то есть уровень 0 достаётся бесплатно. Уточнитель предсказывает
уровни 1 и 2 ПАРАЛЛЕЛЬНО, не заглядывая друг в друга. Поэтому по умолчанию
уровень 0 берётся из K_bar, а не предсказывается: так считается ровно то, что
будет работать на инференсе.

ГЛАВНЫЕ ПОКАЗАТЕЛИ. Прежняя «доля закрытого разрыва» R удалена: с базой
«случайные тонкие коды» она насыщается около 0.94-0.96 и не различает хорошие
модели от плохих. Решают:

    target=bar      ошибка ИМИТАЦИИ учителя, нормированная на собственную
                    ошибку BAR относительно датасета (одна шкала у числителя
                    и знаменателя — размах действий ДАТАСЕТА);
    target=dataset  относительное отставание от BAR; отрицательное = обгон;
    и то и другое   отдельно train / val / test, чтобы отличать
                    непредставимость от переобучения.

ЛЕСТНИЦА ПО ВЫХОДУ (--head). Четыре варианта одного ствола, различающиеся
только формой ответа; каждая ступень снимает одно ограничение:
    free    свободные классификаторы на n_codes;
    tied    вектор в латенте, логиты из геометрии ЗАМОРОЖЕННЫХ книг уровней
            1 и 2 (именно их, не грубой);
    cont    одна непрерывная поправка к латенте, без квантования;
    direct  поправка сразу к действию, минуя декодер.
Сравнение соседних ступеней локализует помеху: параметризация, дискретизация
или латентное пространство с декодером.

РАЗБИЕНИЕ ПО ЭПИЗОДАМ, не по наблюдениям: с одного эпизода взято около десяти
состояний, и случайное деление пустило бы соседние кадры в train и в test.

Запуск:
    python3 experiments/k6e_refiner_train.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6e_refiner_train.py --feats data/k6d_features.npz \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --out data/k6e.json
"""

import argparse
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL = 16, 3
# БАЗЫ БОЛЬШЕ НЕ ЗАШИТЫ. Прежние 0.0315 и 0.0242 получены на ДРУГОМ разбиении;
# подставлять их к новому test — значит сравнивать несравнимое. Теперь E_BAR
# считается на текущем test, а роль слабой базы играет самый маленький из
# обученных в этом же прогоне вариантов.
E_MLP_LEGACY, E_BAR_LEGACY = 0.0315, 0.0242


class _Tee:
    """Дублировать stdout в файл. Прогон длинный, идёт в tmux, и терять вывод
    нельзя: числа рождаются один раз и переигрываются дорого. Пишем с
    немедленным сбросом, чтобы лог был читаем и у прерванного процесса."""

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
    u = np.unique(ep)
    r = np.random.default_rng(seed).permutation(len(u))
    n1, n2 = int(len(u) * fr[0]), int(len(u) * (fr[0] + fr[1]))
    parts = [set(u[r[:n1]]), set(u[r[n1:n2]]), set(u[r[n2:]])]
    return tuple(np.where(np.isin(ep, list(x)))[0] for x in parts)


def closed_fraction(e_ref, e_weak, e_bar):
    """Доля разрыва между слабой базой и BAR, закрытая уточнителем.

    Обе базы передаются явно и считаются на ТОМ ЖЕ разбиении, что и e_ref.
    """
    den = e_weak - e_bar
    return float("nan") if abs(den) < 1e-12 else (e_weak - e_ref) / den


def build_refiner(layers, d, d_in, d_ctx, xa_at, n_codes, n_out, heads=8, ff=4,
                  head="free", E=None, z_dim=512, tau=1.0, coarse_vec=None,
                  n_steps=20, n_dim=7):
    """head:
      free   — свободные классификаторы на n_codes (как было);
      tied   — предсказать вектор в латенте, логиты из геометрии ЗАМОРОЖЕННОГО
               кодбука: -||r - E_g[k]||^2 / tau. Отрезает возможность выучить
               произвольное отображение в 2048 меток;
      cont   — непрерывная поправка к латенте, без квантования вовсе;
      direct — поправка прямо к действию, минуя декодер.
    Диагностическая лестница: если cont догоняет BAR, а tied нет, мешает
    дискретизация; если tied догоняет — новый токенизатор не нужен.
    """
    import torch
    import torch.nn as nn

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_norm = nn.LayerNorm(d_in)
            self.inp = nn.Linear(d_in, d)
            self.head = head
            self.n_out = n_out
            # ГРУБЫЙ КОД ПОДАЁТСЯ ЯВНО. K_bar[:,0,:] известен бесплатно после
            # первого прохода; заставлять уточнитель выводить его заново из h
            # — значит тратить ограниченные данные на переоткрытие
            # 2048-классового отображения. Без этого вывод «информации нет в
            # h» преждевременен.
            # ЗАМОРОЖЕННЫЙ вектор кода вместо обучаемой таблицы: минус
            # 1.57 млн параметров и правильная геометрия вместо случайной.
            if coarse_vec is not None:
                self.register_buffer("cvec", coarse_vec)
                self.coarse_proj = nn.Linear(coarse_vec.shape[-1], d)
                self.coarse_emb = None
            else:
                self.coarse_emb = nn.Embedding(n_codes, d)
            self.pos = nn.Embedding(N_POS, d)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(d, heads, d * ff, batch_first=True,
                                           norm_first=True, dropout=0.0)
                for _ in range(layers)])
            self.xa_at = set(xa_at)
            if self.xa_at:
                # НОРМИРОВКА КОНТЕКСТА ОБЯЗАТЕЛЬНА. ctx снят как ВХОД
                # input_layernorm последнего слоя VLM, то есть остаточный
                # поток ДО нормализации: у языковых моделей там выбросы в
                # сотни единиц. Без LayerNorm проекция взрывается, и вариант
                # с доступом к префиксу оказывается ХУЖЕ варианта без него —
                # ровно это и наблюдалось (0.0449 против 0.0320).
                self.ctx_norm = nn.LayerNorm(d_ctx)
                self.ctx_proj = nn.Linear(d_ctx, d)
                self.xa = nn.ModuleDict({
                    str(i): nn.MultiheadAttention(d, heads, batch_first=True)
                    for i in self.xa_at})
                self.xa_norm = nn.ModuleDict({str(i): nn.LayerNorm(d)
                                              for i in self.xa_at})
            self.norm = nn.LayerNorm(d)
            if head == "free":
                self.out = nn.ModuleList([nn.Linear(d, n_codes)
                                          for _ in range(n_out)])
            elif head in ("tied", "cont"):
                # 2 x d x z_dim вместо 2 x d x n_codes: вчетверо меньше
                # cont — ОДНА общая поправка: две складывающиеся
                # неидентифицируемы, одна голова может дать +v, другая -v.
                self.out = nn.ModuleList([
                    nn.Linear(d, z_dim)
                    for _ in range(n_out if head == "tied" else 1)])
                if head == "tied":
                    # E ЗДЕСЬ — ТОЛЬКО КНИГИ ПРЕДСКАЗЫВАЕМЫХ УРОВНЕЙ. Раньше
                    # передавался весь набор из трёх, а цикл шёл по g=0,1:
                    # fine-1 сравнивался с ГРУБОЙ книгой E0, fine-2 с E1,
                    # тогда как декодирование шло через E1 и E2. Код
                    # выбирался по одной книге, а трактовался по другой.
                    assert E is not None and E.shape[0] == n_out, (
                        f"tied: книг {None if E is None else E.shape[0]}, "
                        f"выходов {n_out} — должны совпадать")
                    self.register_buffer("Ebook", E)
                    self.tau = tau
            elif head == "direct":
                self.out = nn.ModuleList([nn.Linear(d * N_POS, n_steps * n_dim)])
            else:
                raise ValueError(head)

        def forward(self, x, mem=None, mem_mask=None, k0=None):
            b = x.shape[0]
            x = self.inp(self.in_norm(x)) + self.pos(
                torch.arange(N_POS, device=x.device)).unsqueeze(0).expand(b, -1, -1)
            if k0 is not None:
                x = x + (self.coarse_emb(k0) if self.coarse_emb is not None
                         else self.coarse_proj(self.cvec[k0]))
            m = self.ctx_proj(self.ctx_norm(mem)) if self.xa_at else None
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.xa_at:
                    a, _ = self.xa[str(i)](self.xa_norm[str(i)](x), m, m,
                                           key_padding_mask=mem_mask,
                                           need_weights=False)
                    x = x + a
            x = self.norm(x)
            if self.head == "free":
                return [o(x) for o in self.out]
            if self.head == "tied":
                out = []
                for g, o in enumerate(self.out):
                    r = o(x)                                    # (b, P, z)
                    # РАСКРЫТЫЙ КВАДРАТ, А НЕ ШИРОКОВЕЩАНИЕ. Разность
                    # (b,P,1,z)-(V,z) материализует (b,P,V,z): при батче 256
                    # это 17 ГБ и мгновенный OOM. Тождество
                    # ||r-e||^2 = ||r||^2 - 2<r,e> + ||e||^2 даёт (b,P,V).
                    Eb = self.Ebook[g]                          # (V, z)
                    dist = ((r * r).sum(-1, keepdim=True)
                            - 2.0 * (r @ Eb.t())
                            + (Eb * Eb).sum(-1))
                    out.append(-dist / self.tau)
                return out
            if self.head == "cont":
                return [o(x) for o in self.out]                 # латенты
            return [self.out[0](x.flatten(1)).view(x.shape[0], n_steps, n_dim)]

    return Refiner()


def selftest():
    ep = np.repeat(np.arange(60), 10)
    tr, va, te = split_by_episode(ep, seed=0)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(ep[a]) & set(ep[b])), "эпизоды протекают между частями"
    assert len(tr) + len(va) + len(te) == len(ep)

    # Доля закрытого разрыва на известных точках
    w, b = E_MLP_LEGACY, E_BAR_LEGACY
    assert abs(closed_fraction(w, w, b) - 0.0) < 1e-12, "слабая база -> R=0"
    assert abs(closed_fraction(b, w, b) - 1.0) < 1e-12, "уровень BAR -> R=1"
    assert closed_fraction(0.0200, w, b) > 1.0, "лучше BAR -> R>1"
    assert closed_fraction(0.0400, w, b) < 0.0, "хуже слабой базы -> R<0"
    assert abs(closed_fraction((w + b) / 2, w, b) - 0.5) < 1e-12, "середина"
    import math
    assert math.isnan(closed_fraction(0.03, 0.02, 0.02)), \
        "вырожденные базы обязаны давать NaN, а не деление на ноль"

    # ДЫМОВАЯ ПРОВЕРКА ЧЕТЫРЁХ ГОЛОВ. Пишутся они вслепую (torch есть не
    # везде), а ошибка формы вскрылась бы только через час обучения.
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "torch отсутствует: формы голов и прохождение градиента НЕ "
            "проверены. Считать это успехом нельзя — главная часть "
            "самопроверки пропускается.")
    B, P, Z, C, D = 3, N_POS, 8, 16, 32
    Efake = torch.randn(N_LEVEL, C, Z)
    x = torch.randn(B, P, D)
    k0v = torch.randint(0, C, (B, P))
    want = {"free": (B, P, C), "tied": (B, P, C), "cont": (B, P, Z),
            "direct": (B, 20, 7)}
    for hd in ("free", "tied", "cont", "direct"):
        # книг СТОЛЬКО ЖЕ, сколько выходов — иначе tied сравнивает с чужими
        mm = build_refiner(2, 16, D, 4, (), C, 2, heads=2, head=hd,
                           E=Efake[[1, 2]], z_dim=Z, coarse_vec=Efake[0])
        out = mm(x, None, None, k0v)
        n_out = 1 if hd in ("direct", "cont") else 2
        assert len(out) == n_out, f"{hd}: выходов {len(out)}, ждали {n_out}"
        assert tuple(out[0].shape) == want[hd], \
            f"{hd}: форма {tuple(out[0].shape)}, ждали {want[hd]}"
        out[0].sum().backward()          # градиент обязан пройти
        assert mm.inp.weight.grad is not None, f"{hd}: градиент не дошёл"
        assert not hasattr(mm, "coarse_emb") or mm.coarse_emb is None, \
            "при замороженном векторе обучаемой таблицы быть не должно"
    print("самопроверка пройдена: разбиение по эпизодам без протечек; "
          "R корректен на известных точках; все четыре головы дают верные "
          "формы и пропускают градиент")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--feats", default="data/k6d_features.npz")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--grip-weight", type=float, default=1.0 / 6.0,
                    help="вес схвата. Умолчание 1/6 воспроизводит прежнее "
                         "усреднение по семи каналам. При весе 1.0 схват "
                         "вносил в 5-20 раз больше позы (замерено на дыме: "
                         "поза 0.08-0.29 против схвата 1.25-1.87), то есть "
                         "оптимизировался в основном он, тогда как метрика и "
                         "отбор эпохи идут по позе")
    ap.add_argument("--ce-weight", type=float, default=0.0,
                    help="вес кросс-энтропии. ПО УМОЛЧАНИЮ НОЛЬ: при 0.01 её "
                         "вклад был ~0.043 против ~0.0038 у ошибки действия, "
                         "то есть она ДОМИНИРОВАЛА вдесятеро. И cont/direct "
                         "её не используют вовсе — при ненулевом весе "
                         "лестница была бы нечестной")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predict-level0", action="store_true",
                    help="предсказывать и уровень 0 тоже; по умолчанию он "
                         "берётся из первого блока BAR, как на инференсе")
    ap.add_argument("--head", default="free",
                    choices=["free", "tied", "cont", "direct"],
                    help="лестница по параметризации выхода: free — свободные "
                         "классификаторы; tied — латент плюс геометрия "
                         "замороженного кодбука; cont — непрерывная поправка "
                         "без квантования; direct — поправка прямо к действию")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="температура расстояний в режиме tied")
    ap.add_argument("--train-sizes", default="0",
                    help="ВЛОЖЕННЫЕ подвыборки обучения при ОДНИХ И ТЕХ ЖЕ "
                         "val/test, через запятую; 0 = вся. Только так рост "
                         "данных отделяется от роста разнообразия эпизодов")
    ap.add_argument("--target", choices=["dataset", "bar"], default="dataset",
                    help="dataset — действие демонстрации (можно ли ПРЕВЗОЙТИ "
                         "BAR); bar — декод её собственных кодов (можно ли её "
                         "хотя бы ВОСПРОИЗВЕСТИ)")
    ap.add_argument("--variants", default="4x768x0,12x768x0,12x768x2",
                    help="слои x ширина x число слоёв с доступом к префиксу")
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

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt (для декодера действий) или --selftest")

    sys.path.insert(0, os.path.abspath(args.root))
    import torch
    import torch.nn as nn

    import actioncodec  # noqa: F401
    from utils import VisionLanguageActionProcessor  # noqa: E402

    z = np.load(args.feats, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    h = z["h"]
    has_ctx = "ctx" in z.files
    ctx_len = z["ctx_len"] if "ctx_len" in z.files else None
    ctx = z["ctx"] if has_ctx else None
    if not has_ctx:
        print("  ВНИМАНИЕ: в признаках нет ctx (извлечено с --no-ctx) — "
              "варианты с перекрёстным вниманием будут пропущены")
    K_true, K_bar, act, epi = z["K_true"], z["K_bar"], z["act"], z["episode"]
    N, d_act = h.shape[0], h.shape[-1]
    d_vlm = ctx.shape[-1] if has_ctx else 1
    L_ctx = ctx.shape[1] if has_ctx else 1
    n_codes = int(meta["n_codes"])
    print(f"признаки: h {h.shape}, "
          f"ctx {ctx.shape if has_ctx else 'нет'}, {N} наблюдений, "
          f"{len(np.unique(epi))} эпизодов")

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    def decode(codes):
        d = proc.action_processor.decode(codes.reshape(len(codes), -1).tolist())
        return np.asarray(d if isinstance(d, np.ndarray) else d[0], np.float64)

    # ДИАГНОСТИКА МАСШТАБОВ. Если ctx на порядки больше h, сырая проекция
    # без нормировки заведомо неустойчива — это и была причина провала.
    for nm, arr in ((("h", h), ("ctx", ctx)) if has_ctx else (("h", h),)):
        a = np.asarray(arr[:200], np.float32)
        print(f"  {nm:>4}: |avg| {np.abs(a).mean():8.3f}  max {np.abs(a).max():9.1f}  "
              f"sd {a.std():8.3f}")

    dev = torch.device(args.device)

    # --- КОДБУКИ: вектор, который код ДОБАВЛЯЕТ к сумме -----------------------
    # k1_residual_cost.projected_codebooks: from_codes складывает
    # out_project(decode_code(c)), и остаток живёт именно там.
    ac = proc.action_processor
    codec = getattr(ac, "vq", None) and ac or getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit(
            "не нашёл квантователь в action_processor: нужен объект с .vq."
            " Посмотрите dir(proc.action_processor) и подставьте путь.")
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size)).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float()
    assert E.shape[0] == N_LEVEL, f"уровней в кодбуке {E.shape[0]}"
    print(f"  кодбуки: {tuple(E.shape)}  (уровни, коды, размерность латенты)")
    E = E.to(dev)
    # ДЕКОДЕР НУЖЕН ЦЕЛИКОМ И ДИФФЕРЕНЦИРУЕМЫМ. Латентный MSE оказался слеп:
    # он стоял на 0.0006 с нулевой эпохи во всех вариантах. Причина — в нашем
    # же замере: кодек размещает ёмкость НЕ там, где чувствителен декодер
    # (§1), а декодер чувствителен к смещениям ВНУТРИ кодового подпространства
    # (§2в). Остаточные уровни малы по норме, но бьют в чувствительные
    # направления, поэтому равновзвешенный MSE их не видит. Единственная
    # честная цель — ошибка ДЕКОДИРОВАННОГО действия, а _decode это обычный
    # PerceiverDecoder, через него градиент идёт.
    codec_t = codec.to(dev).eval()
    for prm in codec_t.parameters():
        prm.requires_grad_(False)

    def decode_soft(z_q):
        x, _ = codec_t._decode(z_q, embodiment_ids=0)
        return x[..., :7]

    def k0(idx):
        """Грубый код от BAR для этих наблюдений — вход уточнителя."""
        return torch.as_tensor(K_bar[idx, 0, :]).long().to(dev)

    def lat(codes_lp):
        """Сумма латентов по кодам (N, L, P) -> (N, P, D)."""
        c = torch.as_tensor(codes_lp).long().to(dev)
        return sum(E[j][c[:, j, :]] for j in range(N_LEVEL))

    splits = split_by_episode(epi, seed=args.seed)
    itr, iva, ite = splits
    print(f"разбиение по эпизодам: {len(itr)}/{len(iva)}/{len(ite)}")

    # МАСКА ПАДДИНГА ОБЯЗАТЕЛЬНА. Паддинг слева, значимая часть прижата
    # вправо; без маски перекрёстное внимание смотрело бы в нули.
    pad_mask = ((np.arange(L_ctx)[None, :] < (L_ctx - ctx_len[:, None]))
                if has_ctx else None)
    if args.predict_level0:
        raise SystemExit(
            "--predict-level0 временно запрещён: lat0 уже содержит вклад "
            "грубого уровня от BAR, и предсказанный уровень 0 добавился бы "
            "ВТОРОЙ раз. Нужна отдельная ветка сборки латенты.")
    levels = [1, 2]
    # ВСЕ ТРИ МЕСТА СОГЛАСОВАНЫ С --target: лосс действия, цель CE и отбор
    # чекпойнта. Прежде лосс имитировал BAR, CE учила коды ТОКЕНИЗАТОРА, а
    # чекпойнт выбирался по расстоянию до действия ДАТАСЕТА — три разные
    # задачи в одном прогоне, и он не отвечал ни на одну чисто.
    if args.target == "dataset":
        tgt_act, y_ce_full = act, K_true
    else:
        tgt_act = np.concatenate(
            [decode(K_bar[i:i + 512]) for i in range(0, N, 512)])[..., :7]
        y_ce_full = K_bar
    # ЦЕЛЬ — сумма по ИСТИННЫМ кодам всех трёх уровней; БАЗА — вклад уровня 0
    # от BAR, который на инференсе достаётся бесплатно из первого блока.
    with torch.no_grad():
        # ХРАНИМ НА CPU. Два таких тензора по 0.92 ГиБ занимали на карте
        # 1.8 ГиБ всё время прогона, хотя нужны построчно по батчам.
        # lat_t к тому же после разделения лосса вообще не используется в
        # обучении — оставлен только как диагностика.
        lat_t = torch.cat([lat(K_true[i:i + 512]).cpu()
                           for i in range(0, N, 512)])
        lat0 = torch.cat([(E[0][torch.as_tensor(K_bar[i:i + 512, 0, :]).long().to(dev)]).cpu()
                          for i in range(0, N, 512)])
    print(f"  целевая латента: {tuple(lat_t.shape)}, "
          f"вклад уровня 0 от BAR: {tuple(lat0.shape)}")
    a_ref = tgt_act[ite]
    # РАЗМАХ ВСЕГДА ОТ ДЕЙСТВИЙ ДАТАСЕТА. При target=bar числитель нормировался
    # размахом decode(K_bar), а знаменатель — размахом датасета, и делились
    # две по-разному нормированные величины.
    rng_pose = float(act[..., :6].max() - act[..., :6].min())
    dec_ref = decode(K_true[ite])

    def score(Kx):
        d = decode(Kx)
        return (float(np.sqrt(((d[..., :6] - a_ref[..., :6]) ** 2).mean())) / rng_pose,
                float((np.sign(d[..., 6]) != np.sign(a_ref[..., 6])).mean()),
                float(np.sqrt(((d[..., :6] - dec_ref[..., :6]) ** 2).mean())) / rng_pose)

    print("\n" + "=" * 84)
    print(f"  {'вариант':<26}{'поза RMS':>10}{'разброс':>10}"
          f"{('к BAR' if args.target == 'dataset' else 'имит./BAR'):>10}"
          f"{'парам.':>9}")
    # БАЗЫ НА ТЕКУЩЕМ TEST. Слабая база — «уровень 0 от BAR, тонкие уровни
    # СЛУЧАЙНЫЕ»: она измеряется здесь же и не зависит от чужого прогона.
    # ЦЕЛЬ ОБУЧЕНИЯ. Разделение существенно: «воспроизвести BAR» и
    # «превзойти BAR» — разные вопросы. Если уточнитель не воспроизводит даже
    # её действие, дело в представлении или архитектуре. Если воспроизводит,
    # но датасетное действие не улучшает, дело уже не в параллельном
    # декодировании.
    _tn = "действия датасета" if args.target == "dataset" else "декода кодов BAR"
    print(f"  цель обучения: {args.target}; все ошибки считаются относительно {_tn}")

    res = {}
    # ЗНАМЕНАТЕЛЬ ВСЕГДА ОДИН: ошибка BAR относительно ДЕЙСТВИЯ ДАТАСЕТА.
    # При --target bar целью служит decode(K_bar), поэтому ошибка BAR
    # относительно СВОЕЙ ЖЕ цели равна ровно нулю, и деление на неё уронило бы
    # прогон после нескольких часов обучения.
    _aa = act[ite]
    _rp = float(_aa[..., :6].max() - _aa[..., :6].min())
    _dbar = np.concatenate([decode(K_bar[ite][i:i + 512])
                            for i in range(0, len(ite), 512)])
    e_bar_ds = float(np.sqrt(((_dbar[..., :6] - _aa[..., :6]) ** 2).mean())) / _rp
    tsk_all = z["task"] if "task" in z.files else np.zeros(N, int)
    rngb = np.random.default_rng(args.seed)
    Krnd = K_bar[ite].copy()
    Krnd[:, 1:, :] = rngb.integers(0, n_codes, size=Krnd[:, 1:, :].shape)
    e_bar = score(K_bar[ite])[0]
    e_weak = score(Krnd)[0]
    # БАЗЫ ОТДЕЛЬНО ПО ЧАСТЯМ. Сравнивать обучающую ошибку уточнителя с BAR,
    # посчитанной на test, формально нельзя — части могут различаться по
    # трудности. Считаем BAR и эксперта на каждой части.
    print("\n  базы по частям (относительно цели обучения):")
    print(f"    {'часть':<8}{'BAR':>9}{'эксперт':>10}{'эпизодов':>10}{'задач':>8}")
    per_split = {}
    for nm, ix in (("train", itr), ("val", iva), ("test", ite)):
        aa = tgt_act[ix]
        rp = float(aa[..., :6].max() - aa[..., :6].min())
        def e_of(K):
            d = decode(K)
            return float(np.sqrt(((d[..., :6] - aa[..., :6]) ** 2).mean())) / rp
        per_split[nm] = dict(bar=e_of(K_bar[ix]), expert=e_of(K_true[ix]),
                             n_ep=int(len(np.unique(epi[ix]))),
                             n_task=int(len(set(np.asarray(tsk_all)[ix]))))
        v = per_split[nm]
        print(f"    {nm:<8}{v['bar']:>9.4f}{v['expert']:>10.4f}"
              f"{v['n_ep']:>10}{v['n_task']:>8}")
    miss = (set(np.asarray(tsk_all)[itr]) | set(np.asarray(tsk_all)[iva])
            | set(np.asarray(tsk_all)[ite]))
    for nm, ix in (("train", itr), ("val", iva), ("test", ite)):
        absent = miss - set(np.asarray(tsk_all)[ix])
        if absent:
            print(f"    ВНИМАНИЕ: в {nm} отсутствуют задачи: {len(absent)} шт.")
    res["per_split_baselines"] = per_split

    print(f"\n  базы текущего test: BAR {e_bar:.4f}, случайные тонкие "
          f"{e_weak:.4f}  (прежний прогон: {E_BAR_LEGACY:.4f} / "
          f"{E_MLP_LEGACY:.4f} на ДРУГОМ разбиении)")
    ref_rows = [("эксперт (истинные)", K_true[ite]),
                ("BAR последовательная", K_bar[ite]),
                ("случайные тонкие (база)", Krnd)]
    for name, Kx in ref_rows:
        pp, g, v = score(Kx)
        R = closed_fraction(pp, e_weak, e_bar)
        res[name] = dict(pose_rms=pp, gripper=g, R=R)
        print(f"  {name:<22}{pp:>10.4f}{g:>9.1%}{R:>8.2f}")

    def to_action(mm_, idx):
        """Признаки -> предсказанное действие. Одна точка ветвления по режиму
        головы: раньше сборка латенты была вшита в цикл обучения, и добавить
        режим без квантования было негде."""
        xb = torch.as_tensor(h[idx], dtype=torch.float32).to(dev)
        memb = (torch.as_tensor(ctx[idx], dtype=torch.float32).to(dev)
                if mm_.xa_at else None)
        mk = (torch.as_tensor(pad_mask[idx]).to(dev) if mm_.xa_at else None)
        lg = mm_(xb, memb, mk, k0(idx))
        base = lat0[idx].to(dev)          # на карту только текущий батч
        if args.head == "direct":
            return decode_soft(base) + lg[0], lg
        if args.head == "cont":
            return decode_soft(base + lg[0]), lg
        # free и tied: straight-through по кодам
        z = base.clone()
        for k, lv in enumerate(levels):
            ps = torch.softmax(lg[k], -1)
            ph = torch.zeros_like(ps).scatter_(-1, ps.argmax(-1, keepdim=True), 1.0)
            z = z + (ph + ps - ps.detach()) @ E[lv]
        return decode_soft(z), lg

    itr_full = itr
    sizes = [int(v) for v in args.train_sizes.split(",")]
    for n_tr in sizes:
      # ВЛОЖЕННЫЕ ПОДВЫБОРКИ при неизменных val/test: иначе рост числа
      # наблюдений смешивается с ростом числа эпизодов, и «данные помогли»
      # нельзя отличить от «разнообразие помогло».
      itr = itr_full[:n_tr] if n_tr else itr_full
      for spec in args.variants.split(","):
          L, d, nxa = (int(v) for v in spec.strip().split("x"))
          if nxa and not has_ctx:
              print(f"  пропуск {spec}: нет ctx в признаках")
              continue
          step = max(1, L // nxa) if nxa else 1
          xa_at = tuple(range(0, L, step))[:nxa]
          scores, curves = [], []
          for s_i in range(args.seeds):
              torch.manual_seed(s_i)
              m = build_refiner(L, d, d_act, d_vlm, xa_at, n_codes,
                                len(levels), head=args.head, E=E[levels],
                                z_dim=E.shape[-1], tau=args.tau,
                                coarse_vec=E[0]).to(dev)
              n_par = sum(pp.numel() for pp in m.parameters())
              opt = torch.optim.AdamW(m.parameters(), lr=args.lr,
                                      weight_decay=args.wd)
              # ПРОГРЕВ И КОСИНУС: без прогрева глубокий трансформер с
              # norm_first расходится на первых шагах.
              steps = args.epochs * max(1, len(itr) // args.batch)
              warm = max(1, steps // 20)
              sched = torch.optim.lr_scheduler.LambdaLR(
                  opt, lambda t: (t + 1) / warm if t < warm
                  else 0.5 * (1 + np.cos(np.pi * (t - warm) / max(1, steps - warm))))
              lossf = nn.CrossEntropyLoss()
              g = torch.Generator().manual_seed(s_i)
              best, curve = (1e9, None, -1), []

              def err_on(idx, bs=256):
                  """Ошибка действия на наборе, через ту же точку ветвления,
                  что и обучение — иначе train и eval разойдутся по режиму."""
                  m.eval()
                  acc = []
                  with torch.no_grad():
                      for i2 in range(0, len(idx), bs):
                          jj = idx[i2:i2 + bs]
                          a_hat, _ = to_action(m, jj)
                          tt = torch.as_tensor(tgt_act[jj],
                                               dtype=torch.float32).to(dev)
                          acc.append(((a_hat[..., :6] - tt[..., :6]) ** 2)
                                     .mean().item() * len(jj))
                  return float(np.sqrt(sum(acc) / len(idx))) / rng_pose

              for ep_i in range(args.epochs):
                  m.train()
                  order = itr[torch.randperm(len(itr), generator=g).numpy()]
                  for i in range(0, len(order), args.batch):
                      j = order[i:i + args.batch]
                      opt.zero_grad()
                      a_hat, lg = to_action(m, j)
                      a_tgt = torch.as_tensor(tgt_act[j],
                                              dtype=torch.float32).to(dev)
                      # ПОЗА И СХВАТ РАЗДЕЛЕНЫ. Раньше лосс усреднялся по
                      # всем семи каналам, а лучшая эпоха выбиралась только по
                      # шести позных — цель и критерий отбора расходились.
                      l_pose = ((a_hat[..., :6] - a_tgt[..., :6]) ** 2).mean()
                      l_grip = ((a_hat[..., 6] - a_tgt[..., 6]) ** 2).mean()
                      l_act = l_pose + args.grip_weight * l_grip
                      loss = l_act
                      assert torch.isfinite(loss), (
                          f"лосс не конечен на эпохе {ep_i}: поза "
                          f"{l_pose.item()}, схват {l_grip.item()}")
                      l_ce = None
                      if args.ce_weight and args.head in ("free", "tied"):
                          y = torch.as_tensor(
                              y_ce_full[j][:, levels, :]).long().to(dev)
                          l_ce = sum(lossf(lg[k].reshape(-1, n_codes),
                                           y[:, k, :].reshape(-1))
                                     for k in range(len(levels))) / len(levels)
                          loss = loss + args.ce_weight * l_ce
                      loss.backward()
                      nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                      opt.step()
                      sched.step()
                      if i == 0 and ep_i % 10 == 0:
                          # ВКЛАДЫ РАЗДЕЛЬНО: при ce_weight=0.01 CE давала
                          # ~0.043 против ~0.0038 у действия, то есть
                          # доминировала вдесятеро.
                          ce_s = (f", CE·λ {args.ce_weight * l_ce.item():.5f}"
                                  if l_ce is not None else "")
                          print(f"        вклады: поза {l_pose.item():.5f}, "
                                f"схват·w {args.grip_weight * l_grip.item():.5f}"
                                f"{ce_s}", flush=True)
                  v = err_on(iva)
                  v_tr = err_on(itr[:len(iva)])
                  curve.append(dict(epoch=ep_i, train=v_tr, val=v))
                  if v < best[0]:
                      best = (v, {k: pp.detach().clone()
                                  for k, pp in m.state_dict().items()}, ep_i)
                  if ep_i % 10 == 0 or ep_i == args.epochs - 1:
                      print(f"      эпоха {ep_i:>3}: действие train {v_tr:.4f}"
                            f" / val {v:.4f}", flush=True)
              m.load_state_dict(best[1])
              scores.append(err_on(ite))
              curves.append(dict(seed=s_i, best_epoch=best[2],
                                 best_val=best[0], last_val=curve[-1]["val"],
                                 curve=curve, n_params=n_par))
              del m
              if dev.type == "cuda":
                  torch.cuda.empty_cache()
          sc = np.array(scores)
          p_ = float(sc.mean())
          tag = (f"{args.head} {L}сл x{d}"
                 + (f" кэш x{nxa}" if nxa else "")
                 + (f" n={len(itr)}" if len(sizes) > 1 else ""))
          # ОТСТАВАНИЕ ОТ BAR НАПРЯМУЮ — главное число. R с базой «случайные
          # тонкие» насыщался на 0.94-0.96 и был бесполезен.
          res[tag] = dict(pose_rms=p_, sd=float(sc.std()),
                          vs_bar_dataset=float(p_ / e_bar_ds),
                          R=closed_fraction(p_, e_weak, e_bar),
                          layers=L, d_model=d, xa=nxa, head=args.head,
                          n_train=int(len(itr)), curves=curves)
          # ПРИ target=bar ЭТО ОШИБКА ИМИТАЦИИ, и «+X% к BAR» тут
          # бессмысленно: цель и есть выход BAR. Нормируем на ошибку BAR
          # относительно датасета.
          rel = (p_ / e_bar_ds - 1.0 if args.target == "dataset"
                 else p_ / e_bar_ds)
          print(f"  {tag:<26}{p_:>10.4f}{sc.std():>10.4f}"
                f"{rel:>10.1%}{curves[0]['n_params'] / 1e6:>9.2f}M")

    # ВОРОТА ПО R УДАЛЕНЫ. С базой «случайные тонкие коды» он насыщается на
    # 0.94-0.96 и не различает хорошие модели от плохих. Решают абсолютные
    # величины и разрыв train/val.
    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    if args.target == "bar":
        print("  Главное — ОШИБКА ИМИТАЦИИ как доля собственной ошибки BAR")
        print("  относительно датасета. Ниже ~0.3 — уточнитель воспроизводит")
        print("  учителя; выше ~0.6 — не воспроизводит, и разговор об обгоне")
        print("  беспредметен.")
    else:
        print("  Главное — отставание от BAR. Отрицательное значит ОБГОН.")
    print("  И отдельно train против val: сходятся — вопрос выразительности,")
    print("  расходятся — вопрос переноса. Новый токенизатор следует ТОЛЬКО")
    print("  из первого, из второго не следует.")
    print("\n  Лестница читается так:")
    print("    tied догоняет            -> новый токенизатор пока не нужен")
    print("    cont догоняет, tied нет  -> непрерывное легче, но сперва")
    print("                                проверить температуру tau")
    print("    direct догоняет, прочие нет -> ограничивает латента/декодер")
    print("    все учатся на train, но не на val -> перенос, не токенизатор")
    print("    даже direct не учится на train -> вход h, оптимизация или")
    print("                                       проводка первого прохода")
    print("\n  Сравнение метода — cached BAR при H=8 против однопроходного при")
    print("  H=8, то есть около 1.47x. Множители 1.31x (кэш) и 2x (горизонт)")
    print("  архитектуре НЕ принадлежат.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        res["meta"] = dict(feats=args.feats, n_obs=int(N), epochs=args.epochs,
                           seeds=args.seeds, levels=levels,
                           e_bar_split=float(e_bar), e_weak_split=float(e_weak),
                           d_action=int(d_act), d_vlm=int(d_vlm))
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
