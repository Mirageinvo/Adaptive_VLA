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

    R = (E_MLP - E_уточнитель) / (E_MLP - E_BAR),  E_MLP=0.0315, E_BAR=0.0242

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
E_MLP, E_BAR = 0.0315, 0.0242          # из K-6b, отложенные 900 наблюдений


def split_by_episode(ep, seed=0, fr=(0.7, 0.15)):
    u = np.unique(ep)
    r = np.random.default_rng(seed).permutation(len(u))
    n1, n2 = int(len(u) * fr[0]), int(len(u) * (fr[0] + fr[1]))
    parts = [set(u[r[:n1]]), set(u[r[n1:n2]]), set(u[r[n2:]])]
    return tuple(np.where(np.isin(ep, list(x)))[0] for x in parts)


def closed_fraction(e_ref):
    """Доля разрыва между слабой головой и BAR, закрытая уточнителем."""
    return (E_MLP - e_ref) / (E_MLP - E_BAR)


def build_refiner(layers, d, d_in, d_ctx, xa_at, n_codes, n_out, heads=8, ff=4):
    import torch
    import torch.nn as nn

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d_in, d)
            self.pos = nn.Embedding(N_POS, d)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(d, heads, d * ff, batch_first=True,
                                           norm_first=True, dropout=0.0)
                for _ in range(layers)])
            self.xa_at = set(xa_at)
            if self.xa_at:
                self.ctx_proj = nn.Linear(d_ctx, d)
                self.xa = nn.ModuleDict({
                    str(i): nn.MultiheadAttention(d, heads, batch_first=True)
                    for i in self.xa_at})
                self.xa_norm = nn.ModuleDict({str(i): nn.LayerNorm(d)
                                              for i in self.xa_at})
            self.norm = nn.LayerNorm(d)
            self.out = nn.ModuleList([nn.Linear(d, n_codes)
                                      for _ in range(n_out)])

        def forward(self, x, mem=None, mem_mask=None):
            b = x.shape[0]
            x = self.inp(x) + self.pos(
                torch.arange(N_POS, device=x.device)).unsqueeze(0).expand(b, -1, -1)
            m = self.ctx_proj(mem) if self.xa_at else None
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
    assert abs(closed_fraction(E_MLP) - 0.0) < 1e-12, "слабая голова -> R=0"
    assert abs(closed_fraction(E_BAR) - 1.0) < 1e-12, "уровень BAR -> R=1"
    assert closed_fraction(0.0200) > 1.0, "лучше BAR -> R>1"
    assert closed_fraction(0.0400) < 0.0, "хуже слабой головы -> R<0"
    mid = (E_MLP + E_BAR) / 2
    assert abs(closed_fraction(mid) - 0.5) < 1e-12, "середина -> R=0.5"

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
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predict-level0", action="store_true",
                    help="предсказывать и уровень 0 тоже; по умолчанию он "
                         "берётся из первого блока BAR, как на инференсе")
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
    h, ctx, ctx_len = z["h"], z["ctx"], z["ctx_len"]
    K_true, K_bar, act, epi = z["K_true"], z["K_bar"], z["act"], z["episode"]
    N, d_act = h.shape[0], h.shape[-1]
    d_vlm, L_ctx = ctx.shape[-1], ctx.shape[1]
    n_codes = int(meta["n_codes"])
    print(f"признаки: h {h.shape}, ctx {ctx.shape}, {N} наблюдений, "
          f"{len(np.unique(epi))} эпизодов")

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    def decode(codes):
        d = proc.action_processor.decode(codes.reshape(len(codes), -1).tolist())
        return np.asarray(d if isinstance(d, np.ndarray) else d[0], np.float64)

    splits = split_by_episode(epi, seed=args.seed)
    itr, iva, ite = splits
    print(f"разбиение по эпизодам: {len(itr)}/{len(iva)}/{len(ite)}")

    dev = torch.device(args.device)
    # МАСКА ПАДДИНГА ОБЯЗАТЕЛЬНА. Паддинг слева, значимая часть прижата
    # вправо; без маски перекрёстное внимание смотрело бы в нули.
    pad_mask = (np.arange(L_ctx)[None, :] < (L_ctx - ctx_len[:, None]))
    levels = list(range(N_LEVEL)) if args.predict_level0 else [1, 2]
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
    ref_rows = [("эксперт (истинные)", K_true[ite]), ("BAR последовательная", K_bar[ite])]
    res = {}
    for name, Kx in ref_rows:
        p, g, v = score(Kx)
        res[name] = dict(pose_rms=p, gripper=g, R=closed_fraction(p))
        print(f"  {name:<22}{p:>10.4f}{g:>9.1%}{closed_fraction(p):>8.2f}"
              f"{'':>10}{'':>8}{'':>9}")
    print(f"  {'MLP из K-6b':<22}{E_MLP:>10.4f}{'0.7%':>9}{0.0:>8.2f}")

    for spec in args.variants.split(","):
        L, d, nxa = (int(v) for v in spec.strip().split("x"))
        step = max(1, L // nxa) if nxa else 1
        xa_at = tuple(range(0, L, step))[:nxa]
        scores = []
        for s_i in range(args.seeds):
            torch.manual_seed(s_i)
            m = build_refiner(L, d, d_act, d_vlm, xa_at, n_codes,
                              len(levels)).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
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
                    lg = m(x, mem, mm)
                    loss = sum(lossf(lg[k].reshape(-1, n_codes),
                                     y[:, k, :].reshape(-1))
                               for k in range(len(levels))) / len(levels)
                    loss.backward()
                    opt.step()
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
                        lg = m(x, mem, mm)
                        vs.append(sum(lossf(lg[k].reshape(-1, n_codes),
                                            y[:, k, :].reshape(-1)).item()
                                      for k in range(len(levels))))
                    v = float(np.mean(vs))
                if v < best[0]:
                    best = (v, {k: p.detach().clone()
                                for k, p in m.state_dict().items()})
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
                    lg = m(x, mem, mm)
                    for k, lv in enumerate(levels):
                        Kx[i:i + len(j), lv, :] = lg[k].argmax(-1).cpu().numpy()
            scores.append(score(Kx))
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        sc = np.array(scores)
        p, g_, v_ = sc.mean(axis=0)
        R = closed_fraction(p)
        tag = f"{L}сл x{d}" + (f" кэш x{nxa}" if nxa else " только h")
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
                           seeds=args.seeds, levels=levels, E_MLP=E_MLP,
                           E_BAR=E_BAR, d_action=int(d_act), d_vlm=int(d_vlm))
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
