"""K-7c: с какой глубины эксперта грубые коды уже определены.

ВОПРОС. Если голова на состоянии слоя 12 воспроизводит то, что полная башня
выдаёт на слое 24, то башню можно останавливать на двенадцатом — это ещё
примерно вдвое поверх уже измеренного отказа от двух проходов.

ЦЕЛЬ — K_bar[:, 0, :], грубые коды САМОЙ BAR на полной глубине. Вопрос ранней
остановки формулируется как «воспроизвести раньше то, что даёт полная башня», а
не «угадать токенизатор». Совпадение с K_true считается вторичной метрикой.

ГЛАВНАЯ МЕТРИКА — ОШИБКА ДЕЙСТВИЯ, НЕ ТОЧНОСТЬ КОДОВ. Урок K-6e: декодер
принимает СУММУ уровней, поэтому промахнуться в соседний код почти бесплатно, а
кросс-энтропия по кодам ранжирует модели не так, как ошибка декодированного
действия. Точность кодов печатается, но решение принимается по ошибке позы.

ОБЯЗАТЕЛЬНЫЕ ОПОРЫ В ТАБЛИЦЕ, без них цифры не читаются:
  BAR direct   — сами коды BAR: нулевая ошибка по построению цели.
  БЕЗ ОБУЧЕНИЯ — настоящая замороженная action_lm_head, применённая прямо к
                 состоянию каждой глубины. На полной глубине это тождество и
                 ОБЯЗАНО дать 100%; если нет, признаки сняты не с того места и
                 зонд недействителен — скрипт падает. Ни одного шага
                 оптимизации не требует.
  after_<N>    — обучаемый зонд на состоянии после N слоёв.
  случайные    — коды из равномерного распределения: пол ошибки.

ГОЛОВА ЗОНДА ПРИВЯЗАНА К НАСТОЯЩЕЙ (--head tied-lm, по умолчанию): ствол
отображает состояние в размерность головы, дальше идёт замороженная
action_lm_head. Причина: первый прогон с собственным линейным слоем на 2048
классов (--head free) дал на ПОЛНОЙ глубине лишь 69.2% точности, хотя настоящая
голова — линейная функция ровно этого входа. Зонду приходилось заново выучивать
768x2048 = 1.57 млн параметров по 45 тысячам примеров, и он просаживал
собственный потолок, а вместе с ним и все промежуточные строки. Режим `free`
оставлен, чтобы этот эффект можно было воспроизвести.

ДВЕ ШКАЛЫ. Основная — расстояние до coarse-действия самой BAR (имитация). Но
печатается и расстояние до действий ДАТАСЕТА: ранняя голова может не
воспроизводить коды BAR и при этом давать не худшее действие. После K-6h, где
+18% ошибки реконструкции не стоили ни одного пункта успеха, это ожидаемый, а
не экзотический исход.

РАЗБИЕНИЕ ПО ЭПИЗОДАМ И СО СТРАТИФИКАЦИЕЙ ПО ЗАДАЧАМ. По строкам была бы
утечка: соседние наблюдения эпизода почти одинаковы. Без стратификации задача
целиком уходила бы в одну часть, и тест молча мерил бы обобщение на новые
ЗАДАЧИ — другой и куда более трудный вопрос.

ПРАВИЛО ЧТЕНИЯ, записано до запуска:
  ошибка на глубине N в пределах ~10% от последней строки -> ранний выход
      берётся из существующего чекпойнта даром;
  ошибка падает вплоть до последнего слоя -> post-hoc ранний выход не работает.

ЧЕГО ЭТОТ ЗОНД НЕ РЕШАЕТ. Он спрашивает про грубый код УЖЕ ОБУЧЕННОЙ BAR в
ЗАМОРОЖЕННОМ backbone. Отрицательный ответ не закрывает токенизатор, у которого
ранний уровень обучается быть самостоятельным: там другие коды, промежуточные
головы дают ранним слоям собственный сигнал, и backbone дообучается. Но цена
такого варианта — совместное обучение 2.2B, то есть выход за рамки, которые
план сам себе поставил. Отрицательный исход — сигнал о стоимости, не запрет.

Запуск:
    python3 experiments/k7c_depth_probe.py --selftest
    python3 experiments/k7c_depth_probe.py --feats data/k7b_depth_4k.npz \\
        --ckpt <ckpt> --epochs 60 --out data/k7c_depth_probe.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

N_POS, N_LEVEL = 16, 3


def split_by_episode(epi, task=None, seed=0, frac=(0.7, 0.15)):
    """Разбиение ПО ЭПИЗОДАМ, стратифицированное ПО ЗАДАЧАМ.

    Без стратификации задача может целиком попасть в одну часть, и тогда
    тест меряет не «обобщение на новые эпизоды», а «обобщение на новые
    задачи» — совсем другой и куда более трудный вопрос, причём молча.
    Разбиение делается внутри каждой задачи отдельно.
    """
    rng = np.random.default_rng(seed)
    masks = [np.zeros(len(epi), bool) for _ in range(3)]
    groups = ([np.arange(len(epi))] if task is None
              else [np.where(task == t)[0] for t in np.unique(task)])
    for g in groups:
        ep = rng.permutation(np.unique(epi[g]))
        n = len(ep)
        if n < 3:
            raise SystemExit(
                f"задача с {n} эпизодами: на три части не делится. Увеличьте "
                f"--n-ep при извлечении.")
        # Доли считаются ОТ КОНЦА: сначала гарантируем непустые val и test, а
        # остаток отдаём train. Иначе при малом числе эпизодов на задачу
        # округление съедало тест целиком, и он оказывался пустым молча.
        n_va = max(1, int(round(n * frac[1])))
        n_te = max(1, int(round(n * (1.0 - frac[0] - frac[1]))))
        if n - n_va - n_te < 1:
            n_va, n_te = 1, 1
        n_tr = n - n_va - n_te
        parts = (ep[:n_tr], ep[n_tr:n_tr + n_va], ep[n_tr + n_va:])
        assert all(len(p) for p in parts), (n, [len(p) for p in parts])
        for m, p in zip(masks, parts):
            m[g] = np.isin(epi[g], p)
    return masks


def selftest():
    # 1. Разбиение по эпизодам: ни один эпизод не попадает в две части.
    epi = np.repeat(np.arange(50), 8)
    tsk = np.array(["t%d" % (e // 5) for e in epi])
    tr, va, te = split_by_episode(epi, tsk, seed=0)
    assert (tr | va | te).all() and not (tr & va).any() and not (tr & te).any()
    assert not (va & te).any()
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(epi[a]) & set(epi[b])), "эпизод попал в две части"
    assert 0.55 < tr.mean() < 0.85, tr.mean()

    # 1б. СТРАТИФИКАЦИЯ: каждая задача обязана быть во всех трёх частях.
    #     Без неё задача уходит в одну часть целиком, и тест молча меряет
    #     обобщение на НОВЫЕ ЗАДАЧИ — другой и куда более трудный вопрос.
    for m, nm in ((tr, "train"), (va, "val"), (te, "test")):
        assert set(tsk[m]) == set(tsk), f"{nm}: задачи потеряны"
    tr2, va2, te2 = split_by_episode(epi, None, seed=0)
    lost = max(len(set(tsk) - set(tsk[m])) for m in (tr2, va2, te2))
    assert lost > 0, "без стратификации задачи должны теряться, иначе тест пуст"

    # 2. Разбиение по СТРОКАМ дало бы утечку — показываем, что это другое.
    rng = np.random.default_rng(0)
    rows = rng.random(len(epi)) < 0.7
    leak = len(set(epi[rows]) & set(epi[~rows])) / len(np.unique(epi))
    assert leak > 0.9, f"утечка при построчном разбиении {leak:.2f}"

    # 3. Точность кодов и ошибка действия — РАЗНЫЕ величины, и ранжируют
    #    по-разному. Урок K-6e: декодер берёт сумму, соседний код почти
    #    бесплатен. Модель A угадывает коды чаще, но промахивается далеко;
    #    модель B ошибается в кодах чаще, но всегда рядом.
    book = np.arange(8).astype(float)[:, None] * 10.0       # далеко разнесены
    true = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    A = np.array([0, 1, 2, 3, 4, 5, 7, 6])                  # 6 из 8 верно
    B = np.array([1, 0, 3, 2, 5, 4, 7, 6])                  # 0 из 8 верно
    accA = (A == true).mean(); accB = (B == true).mean()
    errA = np.abs(book[A] - book[true]).mean()
    errB = np.abs(book[B] - book[true]).mean()
    assert accA > accB and errA < errB
    A2 = np.array([0, 1, 2, 3, 4, 5, 6, 0])                 # 7 из 8, но далеко
    err2 = np.abs(book[A2] - book[true]).mean()
    assert (A2 == true).mean() > accA and err2 > errA, (
        "точность выросла, а ошибка тоже — именно поэтому решаем по ошибке")

    print("самопроверка k7c пройдена: разбиение по эпизодам стратифицировано "
          "по задачам и без утечки, без стратификации задачи теряются, "
          "точность кодов и ошибка действия ранжируют по-разному")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--feats", default="data/k7b_depth_4k.npz")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normed", choices=["auto", "yes", "no"], default="auto",
                    help="брать состояния ПОСЛЕ action_expert.norm. Голова "
                         "кодов принимает вход после неё (bar.py:1246); "
                         "промежуточные глубины снимаются до неё, и без "
                         "нормировки полная глубина выигрывает просто потому, "
                         "что уже нормирована. auto = взять нормированные, "
                         "если они есть в файле")
    ap.add_argument("--head", choices=["tied-lm", "free"], default="tied-lm",
                    help="tied-lm: ствол отображает состояние в размерность "
                         "головы, дальше НАСТОЯЩАЯ замороженная action_lm_head. "
                         "free: собственный линейный слой на 2048 классов — так "
                         "зонд обязан заново выучить 1.57 млн параметров по 45к "
                         "примеров и просаживает собственный потолок")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="data/k7c_depth_probe.json")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    sys.path.insert(0, os.path.abspath(args.root))
    import torch
    import torch.nn as nn

    import actioncodec  # noqa: F401
    from utils import VisionLanguageActionProcessor  # noqa: E402

    z = np.load(args.feats, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    keys = meta["keys"]
    K_bar, K_true, epi = z["K_bar"], z["K_true"], z["episode"]
    act = z["act"]
    N, n_codes = len(epi), int(meta["n_codes"])
    dev = torch.device(args.device)
    print(f"  {N} наблюдений, {len(np.unique(epi))} эпизодов, глубины {keys}")

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None)).to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(n_codes, device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    # НАСТОЯЩАЯ ГОЛОВА КОДОВ. bar.py:1247-1248 нормирует состояние экспертной
    # башни и подаёт в action_lm_head, а наш after_<полная глубина> — это её
    # ВХОД. Значит голову можно взять отдельно, без всей модели.
    from huggingface_hub import snapshot_download
    local = (args.ckpt if os.path.isdir(args.ckpt)
             else snapshot_download(args.ckpt))
    comp = torch.load(os.path.join(local, "action_components.bin"),
                      map_location="cpu")
    if "action_lm_head" not in comp:
        raise SystemExit(f"в action_components.bin нет action_lm_head: {list(comp)}")
    sd_lm = comp["action_lm_head"]
    W = sd_lm["weight"]
    d_lm = int(W.shape[1])
    lm_real = nn.Linear(d_lm, int(W.shape[0]), bias="bias" in sd_lm)
    lm_real.load_state_dict(sd_lm)
    lm_real = lm_real.to(dev).float().eval()
    for p in lm_real.parameters():
        p.requires_grad_(False)
    print(f"  настоящая голова кодов: {tuple(W.shape)}")

    def decode_coarse(codes):
        """Действие ТОЛЬКО из грубого уровня — тот режим, ради которого всё."""
        outs = []
        for i0 in range(0, len(codes), 256):
            k = torch.as_tensor(codes[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                x, _ = codec._decode(E[0][k], embodiment_ids=0)
            outs.append(x[..., :7].float().cpu().numpy())
        return np.concatenate(outs)

    tsk = z["task"]
    tr, va, te = split_by_episode(epi, tsk, seed=0)
    print(f"  train {tr.sum()}, val {va.sum()}, test {te.sum()}; "
          f"задач {len(np.unique(tsk))} / "
          f"{len(np.unique(tsk[tr]))} / {len(np.unique(tsk[va]))} / "
          f"{len(np.unique(tsk[te]))} (всего/tr/va/te)")
    for m, nm in ((tr, "train"), (va, "val"), (te, "test")):
        miss = sorted(set(np.unique(tsk)) - set(np.unique(tsk[m])))
        if miss:
            raise SystemExit(
                f"в {nm} нет задач {miss[:3]}... — тогда зонд молча меряет "
                f"обобщение на НОВЫЕ ЗАДАЧИ, а не на новые эпизоды")

    tgt = K_bar[:, 0, :]                       # цель: грубые коды самой BAR
    a_ref = decode_coarse(tgt)                 # эталон: coarse-only от BAR
    rng_pose = float(act[..., :6].max() - act[..., :6].min())

    def rms_vs(codes, mask, ref):
        d = decode_coarse(codes[mask] if codes.ndim > 1 else codes) - ref[mask]
        return float(np.sqrt((d[..., :6] ** 2).mean())) / rng_pose

    rnd = np.random.default_rng(0).integers(0, n_codes, size=tgt.shape)
    floor = rms_vs(rnd, te, a_ref)
    # ВТОРАЯ ШКАЛА: расстояние до ДЕЙСТВИЙ ДАТАСЕТА. Ранняя голова может не
    # воспроизводить коды BAR и при этом давать не худшее действие — после
    # K-6h это не умозрительная возможность, а ожидаемая.
    ds_bar = rms_vs(tgt, te, act)
    ds_floor = rms_vs(rnd, te, act)
    print(f"\n  шкала 1, имитация BAR (цель зонда):")
    print(f"    пол, случайные коды:        {floor:.4f}")
    print(f"    сами коды BAR:              0.0000 по построению")
    print(f"  шкала 2, расстояние до действий датасета:")
    print(f"    coarse-only от BAR:         {ds_bar:.4f}")
    print(f"    случайные коды:             {ds_floor:.4f}")

    class ResidualAdapter(nn.Module):
        """x + MLP(x), последний слой инициализирован нулём.

        ПОЧЕМУ ИМЕННО ТАК. Прежний ствол начинался с LayerNorm, а тот вычитает
        среднее по каналам, и величина среднего у каждого примера своя —
        последующие линейные слои такую поправку не восстанавливают. То есть
        зонд НЕ МОГ выразить тождество, хотя на полной глубине именно тождество
        и есть верное решение: настоящая голова на нетронутом состоянии даёт
        100%, а обученный зонд давал 73.2%, то есть обучение было хуже
        бездействия.

        Здесь на старте выход РОВНО равен входу, поэтому зонд стартует из
        строки «без обучения» и может только улучшать её. LayerNorm остался, но
        внутри ветви, а не на пути сигнала.
        """

        def __init__(self, d, hidden, n_layers):
            super().__init__()
            mods, cur = [nn.LayerNorm(d)], d
            for _ in range(n_layers):
                mods += [nn.Linear(cur, hidden), nn.GELU()]
                cur = hidden
            last = nn.Linear(cur, d)
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
            self.body = nn.Sequential(*mods, last)

        def forward(self, x):
            return x + self.body(x)

    def train_probe(X, seed):
        torch.manual_seed(seed)
        d_in = X.shape[-1]
        if args.head == "tied-lm":
            assert d_in == d_lm, (
                f"состояние {d_in} каналов, голова ждёт {d_lm}: остаточный "
                f"адаптер требует совпадения размерностей")
            trunk = ResidualAdapter(d_in, args.hidden, args.layers).to(dev)
            head = lm_real                       # заморожена, обучению не подлежит
        else:
            mods, d = [nn.LayerNorm(d_in)], d_in
            for _ in range(args.layers):
                mods += [nn.Linear(d, args.hidden), nn.GELU()]
                d = args.hidden
            trunk = nn.Sequential(*mods).to(dev)
            head = nn.Linear(d, n_codes).to(dev)
        train_p = [p for p in list(trunk.parameters()) + list(head.parameters())
                   if p.requires_grad]
        opt = torch.optim.AdamW(train_p, lr=args.lr, weight_decay=0.01)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        Yt = torch.as_tensor(tgt, dtype=torch.long)
        itr = np.where(tr)[0]
        # ceil, а не //: цикл обрабатывает и последний неполный батч, и при
        # делении нацело косинус доходил бы до нуля раньше конца обучения.
        steps = args.epochs * max(1, math.ceil(len(itr) / args.batch))
        # ПРОГРЕВ ОБЯЗАТЕЛЕН: без него в K-6e глубокий вариант расходился, и
        # глубина ошибочно выглядела вредной.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, s / max(1, int(steps * args.warmup_frac)))
            * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        def snapshot():
            return ({k: v.detach().clone() for k, v in trunk.state_dict().items()},
                    {k: v.detach().clone() for k, v in head.state_dict().items()})

        if args.head == "tied-lm":
            with torch.no_grad():
                xb = torch.as_tensor(X[:64], dtype=torch.float32).to(dev)
                d0 = float((trunk(xb) - xb).abs().max())
            assert d0 == 0.0, f"адаптер на старте не тождество: max|Δ| = {d0}"

        # СТАРТОВОЕ СОСТОЯНИЕ УЧАСТВУЕТ В ОТБОРЕ. Иначе обучение, которое всё
        # только портит, всё равно вернуло бы обученный чекпойнт — ровно то, что
        # произошло в прошлом прогоне, где зонд оказался хуже бездействия.
        trunk.eval(); head.eval()
        best = pose_rms_pred(predict(trunk, head, Xt, va), va)
        best_state = snapshot()
        rg = np.random.default_rng(seed)
        for ep in range(args.epochs):
            trunk.train(); head.train()
            perm = rg.permutation(itr)          # перемешивание НА КАЖДУЮ эпоху
            for i0 in range(0, len(perm), args.batch):
                sel = perm[i0:i0 + args.batch]
                xb = Xt[sel].to(dev); yb = Yt[sel].to(dev)
                lg = head(trunk(xb))
                loss = nn.functional.cross_entropy(
                    lg.reshape(-1, n_codes), yb.reshape(-1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(train_p, 1.0)
                opt.step(); sched.step()
            trunk.eval(); head.eval()
            pv = predict(trunk, head, Xt, va)
            # ОТБОР ПО ТОЙ ЖЕ ВЕЛИЧИНЕ, ПО КОТОРОЙ ОТЧИТЫВАЕМСЯ. В K-6e отбор
            # шёл по CE, а отчёт по ошибке действия — разные чекпойнты.
            m = pose_rms_pred(pv, va)
            if m < best:
                best, best_state = m, snapshot()
        trunk.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        return trunk, head

    def predict(trunk, head, Xt, mask):
        out = []
        ii = np.where(mask)[0]
        with torch.no_grad():
            for i0 in range(0, len(ii), 512):
                xb = Xt[ii[i0:i0 + 512]].to(dev)
                out.append(head(trunk(xb)).argmax(-1).cpu().numpy())
        return np.concatenate(out)

    def pose_rms_pred(pred, mask):
        d = decode_coarse(pred) - a_ref[mask]
        return float(np.sqrt((d[..., :6] ** 2).mean())) / rng_pose

    def pose_rms_ds(pred, mask):
        d = decode_coarse(pred) - act[mask]
        return float(np.sqrt((d[..., :6] ** 2).mean())) / rng_pose

    # СТРОКА БЕЗ ОБУЧЕНИЯ: настоящая голова прямо на состоянии глубины. Для
    # полной глубины это тождество и обязано дать 100% — тогда признаки заведомо
    # верны, и любой недобор обучаемого зонда есть недостаток ЗОНДА, а не
    # глубины. Считается за секунды и не требует ни одного шага оптимизации.
    def lm_direct(X, mask):
        ii = np.where(mask)[0]
        out = []
        with torch.no_grad():
            for i0 in range(0, len(ii), 512):
                xb = torch.as_tensor(X[ii[i0:i0 + 512]], dtype=torch.float32).to(dev)
                if xb.shape[-1] != d_lm:
                    return None
                out.append(lm_real(xb).argmax(-1).cpu().numpy())
        return np.concatenate(out)

    have_n = f"hn_{keys[0]}" in z.files
    if args.normed == "yes" and not have_n:
        raise SystemExit("в файле нет нормированных состояний (hn_*): "
                         "перезапустите k7b новой версией")
    use_n = have_n if args.normed == "auto" else (args.normed == "yes")

    def feat_key(k_):
        return f"hn_{k_}" if use_n else f"h_{k_}"

    print(f"\n  состояния: {'ПОСЛЕ action_expert.norm' if use_n else 'СЫРЫЕ'}"
          f"{'' if have_n else ' (нормированных в файле нет)'}")
    print("  настоящая голова БЕЗ обучения, прямо на состоянии глубины:")
    direct = {}
    for k_ in keys:
        p = lm_direct(z[feat_key(k_)].astype(np.float32), te)
        if p is None:
            continue
        direct[str(k_)] = dict(code_acc=float((p == tgt[te]).mean()),
                               pose_rms=float(np.sqrt(
                                   ((decode_coarse(p) - a_ref[te])[..., :6] ** 2
                                    ).mean())) / rng_pose)
        print(f"    {str(k_):>10}: точность кодов {direct[str(k_)]['code_acc']:.1%}, "
              f"ошибка {direct[str(k_)]['pose_rms']:.4f}")
    full_direct = direct.get(str(keys[-1]), {}).get("code_acc", 0.0)
    if full_direct < 0.999:
        raise SystemExit(
            f"настоящая голова на полной глубине даёт {full_direct:.1%}, а обязана\n"
            f"дать 100%: это её собственный вход. Значит признаки сняты не с того\n"
            f"места — зонд недействителен целиком.")
    print(f"    полная глубина даёт {full_direct:.1%} — признаки верны, и любой "
          f"недобор ниже\n    есть недостаток ЗОНДА, а не глубины")

    res = {}
    print(f"\n{'глубина':>10}{'имит. BAR':>12}{'от пола':>10}"
          f"{'до датасета':>13}{'точн. кодов':>13}{'vs токенайз.':>14}")
    print(f"{'BAR direct':>10}{0.0:>12.4f}{0.0:>10.2f}{ds_bar:>13.4f}"
          f"{1.0:>12.1%}{(K_bar[te][:, 0, :] == K_true[te][:, 0, :]).mean():>13.1%}")
    for k_ in keys:
        X = z[feat_key(k_)].astype(np.float32)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        errs, errs_ds, accs, accs_t = [], [], [], []
        for s in range(args.seeds):
            trunk, head = train_probe(X, s)
            p = predict(trunk, head, Xt, te)
            errs.append(pose_rms_pred(p, te))
            errs_ds.append(pose_rms_ds(p, te))
            accs.append(float((p == tgt[te]).mean()))
            accs_t.append(float((p == K_true[te][:, 0, :]).mean()))
        e = float(np.mean(errs))
        res[str(k_)] = dict(pose_rms=e, pose_rms_seeds=errs,
                            pose_rms_vs_dataset=float(np.mean(errs_ds)),
                            code_acc=float(np.mean(accs)),
                            code_acc_vs_true=float(np.mean(accs_t)),
                            frac_of_floor=e / floor)
        print(f"{str(k_):>10}{e:>12.4f}{e / floor:>10.2f}"
              f"{np.mean(errs_ds):>13.4f}{np.mean(accs):>12.1%}"
              f"{np.mean(accs_t):>13.1%}")

    full = str(keys[-1])
    fin = res[full]["pose_rms"]
    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    if fin > 0.25 * floor:
        print(f"  Строка {full} — ОБУЧАЕМАЯ голова на входе action_lm_head, и")
        print(f"  она дала {fin:.4f} при поле {floor:.4f}. Это НЕ означает, что")
        print("  сломано извлечение: точную сверку делает k7b, сравнивая argmax")
        print("  настоящей головы с K_bar. Значит не справилась ЭТА голова —")
        print("  параметризация, объём данных или обучение. Промежуточные")
        print("  глубины сравнивать не с чем, пока верхняя опора не работает.")
    else:
        ok = [k_ for k_ in keys if k_ != full
              and res[str(k_)]["pose_rms"] <= 1.1 * fin]
        if ok:
            print(f"  В пределах 10% от {full}: {ok}. Самая ранняя — {ok[0]}.")
            print("  Ранний выход берётся из СУЩЕСТВУЮЩЕГО чекпойнта даром.")
        else:
            print(f"  Ни одна промежуточная глубина не подошла к {full} ближе 10%.")
        print()
        print("  ЧЕГО ЭТОТ ЗОНД НЕ РЕШАЕТ. Он спрашивает только, лежит ли грубый")
        print("  код УЖЕ ОБУЧЕННОЙ BAR в ранних слоях замороженного backbone.")
        print("  Отрицательный ответ не закрывает токенизатор с ранними")
        print("  уровнями: там код другой, головы дают промежуточным слоям")
        print("  собственный сигнал обучения, и backbone дообучается. Но цена")
        print("  такого варианта — совместное обучение 2.2B, а это выходит за")
        print("  рамки, которые план сам себе поставил. Отрицательный исход —")
        print("  сигнал о стоимости, а не запрет.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(dict(depths=res, floor_random=floor, n_obs=int(N),
                   normed=bool(use_n), direct_no_training=direct,
                   feats=args.feats, epochs=args.epochs, seeds=args.seeds,
                   split="по эпизодам 70/15/15",
                   target="K_bar[:,0,:], грубые коды BAR на полной глубине"),
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
