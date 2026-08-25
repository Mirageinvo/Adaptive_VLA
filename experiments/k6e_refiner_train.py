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

ДОЛЯ ЗАКРЫТОГО РАЗРЫВА — ГЛАВНОЕ ЧИСЛО:

    R = (E_слабая - E_уточнитель) / (E_слабая - E_BAR)

Обе базы считаются НА ТЕКУЩЕМ test: слабая — «уровень 0 от BAR, тонкие
случайные», сильная — сама BAR. Прежние 0.0315 и 0.0242 получены на другом
разбиении и годятся только для справки.

ПРАВИЛО ЧТЕНИЯ, записано до запуска:
    R >= 0.8      уточнитель практически решает задачу — в симулятор;
    0.5 <= R < 0.8 одно сквозное дообучение, потом решать;
    R < 0.5       существующие коды плохо ложатся на однопроходную схему,
                  это основание для нового токенизатора;
    R <= 0        сломана постановка обучения, а не архитектура.

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


def build_refiner(layers, d, d_in, d_ctx, xa_at, n_codes, n_out, heads=8, ff=4):
    import torch
    import torch.nn as nn

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_norm = nn.LayerNorm(d_in)
            self.inp = nn.Linear(d_in, d)
            # ГРУБЫЙ КОД ПОДАЁТСЯ ЯВНО. K_bar[:,0,:] известен бесплатно после
            # первого прохода; заставлять уточнитель выводить его заново из h
            # — значит тратить ограниченные данные на переоткрытие
            # 2048-классового отображения. Без этого вывод «информации нет в
            # h» преждевременен.
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
            self.out = nn.ModuleList([nn.Linear(d, n_codes)
                                      for _ in range(n_out)])

        def forward(self, x, mem=None, mem_mask=None, k0=None):
            b = x.shape[0]
            x = self.inp(self.in_norm(x)) + self.pos(
                torch.arange(N_POS, device=x.device)).unsqueeze(0).expand(b, -1, -1)
            if k0 is not None:
                x = x + self.coarse_emb(k0)
            m = self.ctx_proj(self.ctx_norm(mem)) if self.xa_at else None
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.xa_at:
                    a, _ = self.xa[str(i)](self.xa_norm[str(i)](x), m, m,
                                           key_padding_mask=mem_mask,
                                           need_weights=False)
                    x = x + a
            x = self.norm(x)
            return [o(x) for o in self.out]

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

    print("самопроверка пройдена: разбиение по эпизодам без протечек; "
          "R=0 при уровне MLP, R=1 при уровне BAR, R=0.5 посередине")


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
    ap.add_argument("--ce-weight", type=float, default=0.01,
                    help="вес кросс-энтропии как регуляризатора; основная "
                         "цель — реконструкция суммы латентов")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predict-level0", action="store_true",
                    help="предсказывать и уровень 0 тоже; по умолчанию он "
                         "берётся из первого блока BAR, как на инференсе")
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
    args = ap.parse_args()

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
    levels = list(range(N_LEVEL)) if args.predict_level0 else [1, 2]
    # ЦЕЛЬ — сумма по ИСТИННЫМ кодам всех трёх уровней; БАЗА — вклад уровня 0
    # от BAR, который на инференсе достаётся бесплатно из первого блока.
    with torch.no_grad():
        lat_t = torch.cat([lat(K_true[i:i + 512]).cpu()
                           for i in range(0, N, 512)]).to(dev)
        lat0 = torch.cat([(E[0][torch.as_tensor(K_bar[i:i + 512, 0, :]).long().to(dev)]).cpu()
                          for i in range(0, N, 512)]).to(dev)
    print(f"  целевая латента: {tuple(lat_t.shape)}, "
          f"вклад уровня 0 от BAR: {tuple(lat0.shape)}")
    a_ref = act[ite]
    rng_pose = float(a_ref[..., :6].max() - a_ref[..., :6].min())
    dec_ref = decode(K_true[ite])

    def score(Kx):
        d = decode(Kx)
        return (float(np.sqrt(((d[..., :6] - a_ref[..., :6]) ** 2).mean())) / rng_pose,
                float((np.sign(d[..., 6]) != np.sign(a_ref[..., 6])).mean()),
                float(np.sqrt(((d[..., :6] - dec_ref[..., :6]) ** 2).mean())) / rng_pose)

    print("\n" + "=" * 84)
    print(f"  {'вариант':<22}{'поза RMS':>10}{'схват':>9}{'R':>8}"
          f"{'разброс':>10}{'мс':>8}{'ускор.':>9}")
    # БАЗЫ НА ТЕКУЩЕМ TEST. Слабая база — «уровень 0 от BAR, тонкие уровни
    # СЛУЧАЙНЫЕ»: она измеряется здесь же и не зависит от чужого прогона.
    # ЦЕЛЬ ОБУЧЕНИЯ. Разделение существенно: «воспроизвести BAR» и
    # «превзойти BAR» — разные вопросы. Если уточнитель не воспроизводит даже
    # её действие, дело в представлении или архитектуре. Если воспроизводит,
    # но датасетное действие не улучшает, дело уже не в параллельном
    # декодировании.
    tgt_act = act if args.target == "dataset" else np.concatenate(
        [decode(K_bar[i:i + 512]) for i in range(0, N, 512)])[..., :7]
    print(f"  цель обучения: {args.target}")

    rngb = np.random.default_rng(args.seed)
    Krnd = K_bar[ite].copy()
    Krnd[:, 1:, :] = rngb.integers(0, n_codes, size=Krnd[:, 1:, :].shape)
    e_bar = score(K_bar[ite])[0]
    e_weak = score(Krnd)[0]
    print(f"\n  базы текущего test: BAR {e_bar:.4f}, случайные тонкие "
          f"{e_weak:.4f}  (прежний прогон: {E_BAR_LEGACY:.4f} / "
          f"{E_MLP_LEGACY:.4f} на ДРУГОМ разбиении)")
    ref_rows = [("эксперт (истинные)", K_true[ite]),
                ("BAR последовательная", K_bar[ite]),
                ("случайные тонкие (база)", Krnd)]
    res = {}
    for name, Kx in ref_rows:
        pp, g, v = score(Kx)
        R = closed_fraction(pp, e_weak, e_bar)
        res[name] = dict(pose_rms=pp, gripper=g, R=R)
        print(f"  {name:<22}{pp:>10.4f}{g:>9.1%}{R:>8.2f}")

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
          scores = []
          for s_i in range(args.seeds):
              torch.manual_seed(s_i)
              m = build_refiner(L, d, d_act, d_vlm, xa_at, n_codes,
                                len(levels)).to(dev)
              opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
              # ПРОГРЕВ И КОСИНУС. Без прогрева глубокий трансформер с
              # norm_first расходится на первых шагах, и глубина начинает
              # ВРЕДИТЬ — что и наблюдалось (12 слоёв хуже 4).
              steps = args.epochs * max(1, len(itr) // args.batch)
              warm = max(1, steps // 20)
              sched = torch.optim.lr_scheduler.LambdaLR(
                  opt, lambda t: (t + 1) / warm if t < warm
                  else 0.5 * (1 + np.cos(np.pi * (t - warm) / max(1, steps - warm))))
              lossf = nn.CrossEntropyLoss()
              g = torch.Generator().manual_seed(s_i)
              best = (1e9, None)
              for ep_i in range(args.epochs):
                  m.train()
                  order = itr[torch.randperm(len(itr), generator=g).numpy()]
                  for i in range(0, len(order), args.batch):
                      j = order[i:i + args.batch]
                      x = torch.as_tensor(h[j], dtype=torch.float32).to(dev)
                      mem = (torch.as_tensor(ctx[j], dtype=torch.float32).to(dev)
                             if xa_at else None)
                      mm = (torch.as_tensor(pad_mask[j]).to(dev) if xa_at else None)
                      y = torch.as_tensor(K_true[j][:, levels, :]).long().to(dev)
                      opt.zero_grad()
                      lg = m(x, mem, mm, k0(j))
                      # ЛАТЕНТНЫЙ ЛОСС, А НЕ КРОСС-ЭНТРОПИЯ ПО КОДАМ.
                      # Декодер принимает СУММУ латентов (§1): промах на
                      # соседний по эмбеддингу код почти бесплатен для действия,
                      # но кросс-энтропией штрафуется полностью. Отсюда и
                      # наблюдавшееся CE выше равномерного (10.6-12.7 против
                      # ln(2048)=7.62) при вполне приличном действии.
                      # Мягкое ожидание по кодбуку дифференцируемо, argmax
                      # обходится, цель совпадает с тем, что видит декодер.
                      # ВАЖНО: в сумму входит и вклад уровня 0 ОТ BAR, поэтому
                      # уточнитель может научиться компенсировать её грубые
                      # промахи, а не только угадывать свои уровни.
                      # STRAIGHT-THROUGH. Обучение на softmax-смеси даёт
                      # латенту, НЕДОСТИЖИМУЮ ни одним дискретным кодом, а
                      # инференс берёт argmax. Прямой проход теперь идёт по
                      # жёсткому выбору, градиент — по мягкому.
                      pred_lat = lat0[j].clone()
                      for k, lv in enumerate(levels):
                          ps = torch.softmax(lg[k], -1)
                          ph = torch.zeros_like(ps).scatter_(
                              -1, ps.argmax(-1, keepdim=True), 1.0)
                          pst = ph + ps - ps.detach()
                          pred_lat = pred_lat + pst @ E[lv]
                      # ОШИБКА ДЕЙСТВИЯ ЧЕРЕЗ ДЕКОДЕР, а не MSE в латенте.
                      a_hat = decode_soft(pred_lat)
                      a_tgt = torch.as_tensor(tgt_act[j], dtype=torch.float32).to(dev)
                      loss = ((a_hat - a_tgt) ** 2).mean()
                      ce = sum(lossf(lg[k].reshape(-1, n_codes),
                                     y[:, k, :].reshape(-1))
                               for k in range(len(levels))) / len(levels)
                      loss = loss + args.ce_weight * ce
                      loss.backward()
                      nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                      opt.step()
                      sched.step()
                  m.eval()
                  with torch.no_grad():
                      vs = []
                      for i in range(0, len(iva), 256):
                          j = iva[i:i + 256]
                          x = torch.as_tensor(h[j], dtype=torch.float32).to(dev)
                          mem = (torch.as_tensor(ctx[j], dtype=torch.float32).to(dev)
                                 if xa_at else None)
                          mm = (torch.as_tensor(pad_mask[j]).to(dev) if xa_at else None)
                          y = torch.as_tensor(K_true[j][:, levels, :]).long().to(dev)
                          lg = m(x, mem, mm, k0(j))
                          # НА УРОВЕНЬ, А НЕ СУММА. Сумма по двум уровням
                          # сравнивалась с ln(2048)=7.62, и 10.6 читалось как
                          # «хуже равномерного», хотя это 5.3 на уровень, то
                          # есть ЛУЧШЕ. В обучении деление было, в валидации нет.
                          vs.append(sum(lossf(lg[k].reshape(-1, n_codes),
                                              y[:, k, :].reshape(-1)).item()
                                        for k in range(len(levels)))
                                    / len(levels))
                      v_ce = float(np.mean(vs))
                      # ОТБОР ПО ОШИБКЕ ДЕЙСТВИЯ, а не по кросс-энтропии.
                      # Декодер принимает СУММУ латентов (§1), поэтому промах на
                      # соседний по эмбеддингу код почти бесплатен, и токенная
                      # энтропия с ошибкой действия расходятся.
                      Kv = K_bar[iva].copy()
                      for i in range(0, len(iva), 256):
                          j = iva[i:i + 256]
                          x = torch.as_tensor(h[j], dtype=torch.float32).to(dev)
                          mem = (torch.as_tensor(ctx[j], dtype=torch.float32).to(dev)
                                 if xa_at else None)
                          mm = (torch.as_tensor(pad_mask[j]).to(dev) if xa_at else None)
                          lg = m(x, mem, mm, k0(j))
                          for k, lv in enumerate(levels):
                              Kv[i:i + len(j), lv, :] = lg[k].argmax(-1).cpu().numpy()
                      dv = decode(Kv)
                      v = float(np.sqrt(((dv[..., :6] - act[iva][..., :6]) ** 2).mean()))
                  if v < best[0]:
                      best = (v, {k: p.detach().clone()
                                  for k, p in m.state_dict().items()})
                  if ep_i % 10 == 0 or ep_i == args.epochs - 1:
                      # ОБЕ величины рядом намеренно. Их расхождение — измеренное
                      # свойство кодека: декодер видит сумму, поэтому попадание в
                      # код и попадание в действие суть разные цели.
                      with torch.no_grad():
                          pl = lat0[iva].clone()
                          for i in range(0, len(iva), 256):
                              jj = iva[i:i + 256]
                              xx = torch.as_tensor(h[jj], dtype=torch.float32).to(dev)
                              mmx = (torch.as_tensor(ctx[jj], dtype=torch.float32).to(dev)
                                     if xa_at else None)
                              mmk = (torch.as_tensor(pad_mask[jj]).to(dev)
                                     if xa_at else None)
                              lgx = m(xx, mmx, mmk, k0(jj))
                              acc_l = lat0[jj].clone()
                              for k, lv in enumerate(levels):
                                  acc_l = acc_l + torch.softmax(lgx[k], -1) @ E[lv]
                              pl[i:i + len(jj)] = acc_l
                          lat_err = float(((pl - lat_t[iva]) ** 2).mean())
                      # ОБУЧАЮЩАЯ ОШИБКА РЯДОМ С ВАЛИДАЦИОННОЙ. Без неё
                      # утверждение «переобучение» ничем не подкреплено: расти
                      # на валидации величина может и от расходящейся
                      # оптимизации. Считается на подвыборке того же размера,
                      # что валидация, чтобы числа были сравнимы.
                      with torch.no_grad():
                          sub = itr[:len(iva)]
                          Kt = K_bar[sub].copy()
                          for i in range(0, len(sub), 256):
                              jj = sub[i:i + 256]
                              xx = torch.as_tensor(h[jj], dtype=torch.float32).to(dev)
                              mmx = (torch.as_tensor(ctx[jj], dtype=torch.float32).to(dev)
                                     if xa_at else None)
                              mmk = (torch.as_tensor(pad_mask[jj]).to(dev)
                                     if xa_at else None)
                              lgx = m(xx, mmx, mmk, k0(jj))
                              for k, lv in enumerate(levels):
                                  Kt[i:i + len(jj), lv, :] = lgx[k].argmax(-1).cpu().numpy()
                          dt = decode(Kt)
                          v_tr = float(np.sqrt(
                              ((dt[..., :6] - act[sub][..., :6]) ** 2).mean()))
                      print(f"      эпоха {ep_i:>3}: действие train "
                            f"{v_tr / rng_pose:.4f} / val {v / rng_pose:.4f}"
                            f"   (CE {v_ce:.2f}, латента {lat_err:.4f})",
                            flush=True)
              m.load_state_dict(best[1])
              m.eval()
              Kx = K_bar[ite].copy()
              with torch.no_grad():
                  for i in range(0, len(ite), 256):
                      j = ite[i:i + 256]
                      x = torch.as_tensor(h[j], dtype=torch.float32).to(dev)
                      mem = (torch.as_tensor(ctx[j], dtype=torch.float32).to(dev)
                             if xa_at else None)
                      mm = (torch.as_tensor(pad_mask[j]).to(dev) if xa_at else None)
                      lg = m(x, mem, mm, k0(j))
                      for k, lv in enumerate(levels):
                          Kx[i:i + len(j), lv, :] = lg[k].argmax(-1).cpu().numpy()
              scores.append(score(Kx))
              del m
              if dev.type == "cuda":
                  torch.cuda.empty_cache()
          sc = np.array(scores)
          p, g_, v_ = sc.mean(axis=0)
          R = closed_fraction(p, e_weak, e_bar)
          tag = (f"{L}сл x{d}" + (f" кэш x{nxa}" if nxa else " только h")
                 + (f" n={len(itr)}" if len(sizes) > 1 else ""))
          res[tag] = dict(pose_rms=float(p), gripper=float(g_), R=float(R),
                          sd=float(sc[:, 0].std()), layers=L, d_model=d, xa=nxa)
          print(f"  {tag:<22}{p:>10.4f}{g_:>9.1%}{R:>8.2f}{sc[:, 0].std():>10.4f}")

    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    print("  R >= 0.8      уточнитель решает задачу — интегрировать и в симулятор")
    print("  0.5 <= R < 0.8 одно сквозное дообучение, потом решать")
    print("  R < 0.5       коды плохо ложатся на однопроходную схему — основание")
    print("                для нового токенизатора")
    print("  R <= 0        сломана постановка обучения, а не архитектура")
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
