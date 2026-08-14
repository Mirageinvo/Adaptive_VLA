"""Головы политики, знающие геометрию решётки FSQ, и проверка их смысла.

ЗАЧЕМ. Политика OAT предсказывает токен плоским софтмаксом по 1000 кодов.
Коды FSQ — точки регулярной решётки [8,5,5,5] в [-1,1]^4, то есть геометрия
там есть по построению, но голова о ней ничего не знает: для неё код 573 и
код 574 такие же чужие друг другу, как 573 и 12.

Следствие, которое в имитации почти незаметно, а в RL существенно:
когда градиент понижает вероятность плохого чанка, он понижает её у ОДНОГО
кода из тысячи. Физические соседи, дающие практически то же самое действие,
остаются нетронутыми. Политика узнаёт «код 573 плохой», а не «в этой области
действий плохо». Это и есть цена незнания геометрии.

ТРИ ГОЛОВЫ.
  FlatHead        — как у OAT: Linear(n_emb -> K). База.
  FactorizedHead  — 4 независимых софтмакса по координатам решётки (8+5+5+5
                    = 23 выхода вместо 1000). Проверяет, достаточно ли одной
                    произведённой структуры без всякой метрики. Корреляции
                    между координатами представить не может.
  MixtureLatentHead — смесь гауссиан В ЛАТЕНТЕ решётки:
                    p(i) ~ sum_m pi_m * N(e_i; mu_m, diag(sigma_m^2)),
                    где e_i — точка решётки для кода i. Метрика и корреляции
                    есть, число выходов M*(1+2d) вместо K.

ПОЧЕМУ sigma ДИАГОНАЛЬНАЯ. Шаг решётки по координатам разный: он равен
1/(L//2), то есть 0.25 при 8 уровнях и 0.5 при 5 — разница вдвое. Скалярная
sigma заставила бы одну ширину обслуживать обе.

ПОЧЕМУ У sigma ЕСТЬ ПОЛ. При sigma -> 0 распределение схлопывается в
ближайший узел, градиент по остальным кодам исчезает, и голова вырождается в
ту же плоскую. Пол не даёт обучению уйти в это вырождение.

Запуск проверки (ни данных, ни OAT не нужно):
    python3 experiments/head_geometric.py
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# решётка FSQ
# --------------------------------------------------------------------------
def fsq_grid(levels=(8, 5, 5, 5)) -> torch.Tensor:
    """Точки решётки FSQ, (K, d). Повторяет indices_to_embedding из
    oat/tokenizer/oat/quantizer/fsq.py:

        codes = (indices // basis) % levels
        emb   = (codes - levels//2) / (levels//2)

    Заметим, что при ЧЁТНОМ числе уровней решётка несимметрична: при L=8
    координата пробегает -1.00 .. 0.75, а не -1 .. 1. Это не опечатка, а их
    формула, и голова обязана видеть ту же решётку, что квантователь."""
    lv = torch.tensor(levels, dtype=torch.long)
    basis = torch.cat([torch.ones(1, dtype=torch.long), lv[:-1].cumprod(0)])
    K = int(lv.prod())
    idx = torch.arange(K).unsqueeze(1)
    codes = (idx // basis) % lv
    half = lv // 2
    return (codes - half).float() / half.float()


def verify_grid_against_oat(oat_root: str, ckpt: str) -> None:
    """Сверка нашей решётки с настоящей побитово. Если формула разойдётся,
    вся голова будет учиться не на той геометрии, а по потерям это никак
    не проявится — только по результату. Поэтому проверка обязательна."""
    import sys

    sys.path.insert(0, oat_root)
    import dill
    import hydra

    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu")
    tok = hydra.utils.instantiate(payload["cfg"].tokenizer)
    Q = tok.quantizer
    ours = fsq_grid(tuple(Q._levels.tolist()))
    theirs = Q.indices_to_embedding(torch.arange(Q.codebook_size)).float()
    d = (ours - theirs).abs().max().item()
    print(f"сверка решётки: макс. расхождение {d:.2e} "
          f"({'совпадает' if d < 1e-6 else 'РАЗОШЛОСЬ'})")
    assert d < 1e-6, "наша решётка не совпала с решёткой квантователя"


# --------------------------------------------------------------------------
# головы
# --------------------------------------------------------------------------
class FlatHead(nn.Module):
    """Как у OAT."""

    def __init__(self, n_emb: int, K: int, n_extra: int = 1):
        super().__init__()
        self.K, self.n_extra = K, n_extra
        self.lin = nn.Linear(n_emb, K + n_extra)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class FactorizedHead(nn.Module):
    """Независимые софтмаксы по координатам решётки.

    log p(i) = sum_d log p_d(code_d(i)) — то есть решётка учтена как
    произведение, но метрика внутри координаты по-прежнему не задана:
    соседние уровни одной координаты остаются несвязанными."""

    def __init__(self, n_emb: int, levels=(8, 5, 5, 5), n_extra: int = 1):
        super().__init__()
        self.levels, self.n_extra = tuple(levels), n_extra
        lv = torch.tensor(self.levels, dtype=torch.long)
        basis = torch.cat([torch.ones(1, dtype=torch.long), lv[:-1].cumprod(0)])
        K = int(lv.prod())
        self.K = K
        idx = torch.arange(K).unsqueeze(1)
        self.register_buffer("codes", ((idx // basis) % lv).long())   # (K, d)
        self.lin = nn.Linear(n_emb, sum(self.levels))
        self.extra = nn.Parameter(torch.zeros(n_extra)) if n_extra else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.lin(x)                                    # (..., sum L)
        out, off = 0.0, 0
        for d, L in enumerate(self.levels):
            lp = F.log_softmax(raw[..., off:off + L], dim=-1)  # (..., L)
            out = out + lp[..., self.codes[:, d]]              # (..., K)
            off += L
        if self.n_extra:
            ex = self.extra.expand(*out.shape[:-1], self.n_extra)
            out = torch.cat([out, ex], dim=-1)
        return out


class MixtureLatentHead(nn.Module):
    """Смесь гауссиан в латенте решётки.

    Ненормированный балл кода i:
        s_i = logsumexp_m [ log pi_m - 0.5*||(e_i - mu_m)/sigma_m||^2
                            - sum_d log sigma_m,d ]
    затем обычный софтмакс по всем K. Нормировка получается сама, потому что
    решётка конечна и одна и та же для всех позиций.

    mu не ограничиваем: смесь должна уметь ставить моду и за краем решётки,
    иначе крайние коды систематически недооценивались бы."""

    def __init__(self, n_emb: int, levels=(8, 5, 5, 5), n_mix: int = 4,
                 sigma_min: float = 0.15, n_extra: int = 1):
        super().__init__()
        grid = fsq_grid(levels)
        self.register_buffer("grid", grid)                   # (K, d)
        self.K, self.d = grid.shape
        self.M, self.sigma_min, self.n_extra = n_mix, sigma_min, n_extra
        # Диагностический рычаг: подменить sigma фиксированной величиной,
        # чтобы мерить дальность обобщения как функцию ширины, а не как
        # свойство случайной инициализации. В обучении всегда None.
        self.sigma_override = None
        self.lin = nn.Linear(n_emb, n_mix * (1 + 2 * self.d))
        # старт от разумного: широкие компоненты, покрывающие решётку
        nn.init.zeros_(self.lin.bias)
        nn.init.normal_(self.lin.weight, std=0.02)
        self.extra = nn.Parameter(torch.zeros(n_extra)) if n_extra else None

    def params(self, x: torch.Tensor):
        M, d = self.M, self.d
        raw = self.lin(x)
        logpi, mu, s = raw.split([M, M * d, M * d], dim=-1)
        mu = mu.reshape(*x.shape[:-1], M, d)
        if self.sigma_override is not None:      # только для диагностики
            sigma = torch.full_like(mu, self.sigma_override)
        else:
            sigma = self.sigma_min + F.softplus(s.reshape(*x.shape[:-1], M, d))
        return F.log_softmax(logpi, dim=-1), mu, sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logpi, mu, sigma = self.params(x)                    # (..., M), (..., M, d) x2
        e = self.grid.view(*([1] * (x.dim() - 1)), 1, self.K, self.d)
        z = (e - mu.unsqueeze(-2)) / sigma.unsqueeze(-2)     # (..., M, K, d)
        comp = (-0.5 * z.pow(2).sum(-1)
                - torch.log(sigma).sum(-1, keepdim=True))    # (..., M, K)
        out = torch.logsumexp(comp + logpi.unsqueeze(-1), dim=-2)   # (..., K)
        if self.n_extra:
            ex = self.extra.expand(*out.shape[:-1], self.n_extra)
            out = torch.cat([out, ex], dim=-1)
        return out


class ScaledHead(nn.Module):
    """Обёртка, множащая логиты на константу.

    Нужна, чтобы сравнивать головы ПРИ ОДИНАКОВОЙ УВЕРЕННОСТИ. Случайно
    инициализированные головы дают почти равномерное распределение
    (p(моды) ~ 1/K), а настоящая политика OAT держит 0.22-0.31. В почти
    равномерном режиме статистика дальности вырождается: освободившаяся
    масса -dp(i*) исчезающе мала, и отношение делит на почти ноль.

    Множитель меняет только остроту распределения, но не геометрию, поэтому
    у смеси sigma остаётся отвечать за РАДИУС обобщения, а множитель — за
    уверенность, и эти две вещи перестают путаться."""

    def __init__(self, head: nn.Module, scale: float):
        super().__init__()
        self.head, self.scale = head, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x) * self.scale


def fit_scale(head: nn.Module, n_emb: int, K: int, target: float,
              n: int = 32, seed: int = 7) -> float:
    """Множитель, при котором медианная p(моды) равна target.
    Заострение софтмакса монотонно повышает максимум, поэтому двоичный."""
    def p_max(s: float) -> float:
        rng = torch.Generator().manual_seed(seed)
        h = ScaledHead(head, s).double()
        vals = []
        with torch.no_grad():
            for _ in range(n):
                xi = torch.randn(1, n_emb, generator=rng, dtype=torch.float64)
                vals.append(float(torch.softmax(h(xi)[0, :K], -1).max()))
        vals.sort()
        return vals[len(vals) // 2]

    lo, hi = 1e-3, 1e5
    if p_max(hi) < target:
        return hi
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        if p_max(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


# --------------------------------------------------------------------------
# проверка центрального утверждения
# --------------------------------------------------------------------------
def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(rx @ ry / (rx.norm() * ry.norm() + 1e-12))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--oat", default=None, help="корень OAT — сверить решётку")
    ap.add_argument("--ckpt", default=None, help="чекпойнт токенизатора")
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.oat and args.ckpt:
        verify_grid_against_oat(args.oat, args.ckpt)
        print()
    levels = (8, 5, 5, 5)
    grid = fsq_grid(levels)
    K, d = grid.shape
    n_emb = 256
    print(f"решётка {levels}: {K} кодов в R^{d}")
    print(f"  диапазон по координатам: "
          f"{[f'[{grid[:, j].min():.2f},{grid[:, j].max():.2f}]' for j in range(d)]}")
    step = [1.0 / (l // 2) for l in levels]
    print(f"  шаг решётки по координатам: {[f'{s:.3f}' for s in step]}\n")

    # Зерно ИМЕННО ЗДЕСЬ: сверка решётки выше создаёт токенизатор и тратит
    # глобальный генератор, из-за чего головы инициализировались бы по-разному
    # с флагом --oat и без него.
    torch.manual_seed(0)
    heads = {
        "плоская (OAT)": FlatHead(n_emb, K),
        "факторизованная": FactorizedHead(n_emb, levels),
        "смесь в латенте M=4": MixtureLatentHead(n_emb, levels, n_mix=4),
        "смесь в латенте M=8": MixtureLatentHead(n_emb, levels, n_mix=8),
    }

    print("=" * 78)
    print("1. РАЗМЕР ГОЛОВЫ И КОРРЕКТНОСТЬ")
    print("=" * 78)
    print(f"{'голова':<24}{'параметров':>13}{'к плоской':>12}{'sum p':>10}")
    x = torch.randn(16, n_emb)
    for name, h in heads.items():
        p = sum(q.numel() for q in h.parameters())
        lo = h(x)
        s = float(torch.softmax(lo, -1).sum(-1).mean())
        ratio = p / sum(q.numel() for q in heads["плоская (OAT)"].parameters())
        print(f"{name:<24}{p:>13,}{ratio:>11.2f}x{s:>10.4f}")
        assert abs(s - 1.0) < 1e-4, f"{name}: распределение не нормировано"

    print("\n" + "=" * 78)
    print("2. ГЛАВНОЕ: ОБОБЩАЕТСЯ ЛИ ГРАДИЕНТ ПО ОКРЕСТНОСТИ")
    print("=" * 78)
    print("""Делаем шаг градиента, понижающий вероятность кода i* (как поступил бы
PG с плохим чанком), и смотрим, куда разошлась поправка.

ЦЕЛЬ СЭМПЛИРУЕТСЯ ИЗ САМОЙ ГОЛОВЫ. PG наказывает тот код, который политика
выбрала сама, то есть код из моды. Если брать i* равномерно, он попадает в
хвост, где и цель, и её соседи, и далёкие коды лежат в одной пренебрежимой
области — их логарифмы двигаются заодно, перетекание выходит около единицы,
и меряется режим, которого в обучении не бывает. Ниже это показано отдельной
таблицей.

ШАГ НЕ ПОДБИРАЕТСЯ. Отношение dlogp(соседей)/dlogp(цели) при малом шаге от
величины шага не зависит, поэтому шаг просто дробится до устойчивости. Идея
подбирать его под равную просадку не годится: у смеси просадка на цели
НЕМОНОТОННА по lr, и двоичный поиск сходится к мусору.

Знак корреляции: соседи должны падать СИЛЬНЕЕ далёких, то есть -dlogp велика
вблизи и мала вдали, а расстояние наоборот -> ожидаем ОТРИЦАТЕЛЬНУЮ связь.
У плоской головы связи быть не должно (корреляция ~ 0), потому что прочие
логиты сдвигаются пропорционально текущему p_j, безотносительно к тому, где
код лежит на решётке.

дальность — на сколько дальше от цели ушла освободившаяся вероятность по
сравнению с пропорциональным перераспределением. Больше 1 — поправка
выбралась за пределы окрестности, и это ровно то, что нужно: иначе политика
пересядет на неотличимое действие и ничего не выучит. У плоской головы
ожидается около 1, но НЕ ТОЧНО 1: для софтмакса dp_j ~ p_j(p_j - C), то
есть пропорциональность лишь приближённая, при малых p_j.
p(i*), ранг — где сидела цель до шага. Все головы приведены множителем на
логиты к одной уверенности, иначе сравнивались бы разные режимы: у случайно
инициализированной головы p(моды) ~ 1/K, а у настоящей политики OAT 0.22-0.31
(замер oat1), и при p(i*) ~ 0 статистика вырождается делением на почти ноль.
""")
    def one_step(h, xi, i_star, lr):
        h2 = copy.deepcopy(h)
        with torch.no_grad():
            lp0 = F.log_softmax(h2(xi)[0, :K], dim=-1)
        opt = torch.optim.SGD(h2.parameters(), lr=lr)
        opt.zero_grad()
        F.log_softmax(h2(xi)[0, :K], dim=-1)[i_star].backward()   # понижаем p(i*)
        opt.step()
        with torch.no_grad():
            lp1 = F.log_softmax(h2(xi)[0, :K], dim=-1)
        return lp0, lp1

    def linear_flow(h, xi, i_star, dist, m):
        """ДАЛЬНОСТЬ ПЕРЕТОКА массы, в долях от пропорционального.

        Что мы хотим от геометричной головы. PG говорит «этот чанк плох».
        У плоской головы освободившаяся вероятность расходится строго
        пропорционально текущему p_j, то есть в основном возвращается на
        почти такие же действия: политика переключилась на неотличимое и
        ничего не выучила. Геометричная должна опустить всю окрестность
        сразу, и тогда масса уходит ЗА её пределы.

            d_move = sum_j dp_j * dist_j / (-dp_i*)   куда масса ушла
            d_flat = sum_j p_j  * dist_j / sum_j p_j  куда ушла бы пропорционально

        Отношение > 1 — поправка выбралась из окрестности (то, что нужно).
        Ровно 1.000 у плоской головы ТОЖДЕСТВЕННО, поскольку для софтмакса
        dp_j ∝ p_j; это встроенная проверка реализации, а не совпадение.

        Отношение масштабно-инвариантно, поэтому шаг не подбирается под
        какую-то просадку (у смеси она немонотонна по lr и вилка сходится к
        мусору), а просто дробится до устойчивости. Не устоялось за 30
        дроблений — возвращаем NaN: несошедшееся число хуже отсутствующего."""
        lr, prev, last = 1e-2, None, None
        for _ in range(30):
            lp0, lp1 = one_step(h, xi, i_star, lr)
            p0, p1 = lp0.exp(), lp1.exp()
            dp = p1 - p0
            R = -float(dp[i_star])
            last = (lp0, lp1, float(p0[i_star]))
            if R <= 0:                    # шаг ушёл не туда либо утонул в нуле
                lr = lr / 3.0
                continue
            d_move = float((dp[m] * dist[m]).sum()) / R
            d_flat = float((p0[m] * dist[m]).sum() / p0[m].sum())
            r = d_move / d_flat
            if prev is not None and abs(r - prev) < 1e-3 * max(abs(prev), 1e-12):
                return r, lp1 - lp0, float(p0[i_star])
            prev, lr = r, lr / 3.0
        return float("nan"), last[1] - last[0], last[2]

    # Усредняем по состояниям и целевым кодам: у плоской головы прочие логиты
    # сдвигаются пропорционально p_j, а p_j при случайной инициализации
    # случайно, и по одной затравке вышел бы шум, а не ноль.
    n_trials = 48
    grid_d = grid.double()

    def sweep(h, tag, mode="sampled"):
        rng = torch.Generator().manual_seed(1)
        h = copy.deepcopy(h).double()
        rhos, ratios, ranks, pstars = [], [], [], []
        for _ in range(n_trials):
            xi = torch.randn(1, n_emb, generator=rng, dtype=torch.float64)
            with torch.no_grad():
                p = torch.softmax(h(xi)[0, :K], dim=-1)
            i_star = (int(torch.multinomial(p, 1, generator=rng))
                      if mode == "sampled"
                      else int(torch.randint(K, (1,), generator=rng)))
            ranks.append(int((p > p[i_star]).sum()) + 1)
            dist = (grid_d - grid_d[i_star]).norm(dim=-1)
            m = torch.ones(K, dtype=torch.bool)
            m[i_star] = False
            r, d, ps = linear_flow(h, xi, i_star, dist, m)
            pstars.append(ps)
            if r == r:                                   # не NaN
                ratios.append(r)
                rhos.append(spearman(-d[m], dist[m]))
        if not ratios:
            print(f"{tag:<30}{'НЕ СОШЛОСЬ':>12}")
            return
        # МЕДИАНА, не среднее: при малых p(i*) отдельные испытания уходят на
        # 1e+200 из-за деления на почти ноль, и среднее становится мусором.
        def med(v):
            s = sorted(v)
            return s[len(s) // 2]
        print(f"{tag:<30}{med(rhos):>12.3f}{med(ratios):>14.3f}"
              f"{med(pstars):>12.3f}{med(ranks):>11.0f}{len(ratios):>10d}")

    target_p = 0.25          # уверенность настоящей политики OAT (oat1)
    print(f"Все головы приведены к одной уверенности: медианная p(моды) = "
          f"{target_p:.2f}.\n")
    print(f"{'голова':<30}{'корр.':>12}{'дальность':>14}"
          f"{'p(i*)':>12}{'ранг':>11}{'сошлось':>10}")
    for name, h in heads.items():
        s = fit_scale(h, n_emb, K, target_p)
        sweep(ScaledHead(h, s), f"{name}  (x{s:.1f})")

    print(f"""
Развёртка по ширине компоненты. sigma и есть та ручка, что задаёт ДАЛЬНОСТЬ
обобщения поправки, и при инициализации она равна 0.84 — три-четыре шага
решётки ({step[0]:.2f}-{step[1]:.2f}), то есть величина, взятая с потолка. Ниже sigma задана
явно, чтобы выбирать sigma_min и инициализацию по кривой, а не наугад.
""")
    print(f"{'sigma':>8}{'в шагах решётки':>22}{'корр.':>12}{'дальность':>14}"
          f"{'p(i*)':>12}{'ранг':>11}{'сошлось':>10}")
    base = MixtureLatentHead(n_emb, levels, n_mix=4)
    sigmas = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00)
    for sg in sigmas:
        h = copy.deepcopy(base)
        h.sigma_override = sg
        s = fit_scale(h, n_emb, K, target_p)
        sweep(ScaledHead(h, s), f"{sg:>8.2f}{sg / step[0]:>22.1f}")

    print(f"""
Уверенность здесь удержана постоянной ({target_p:.2f}) множителем на логиты, поэтому
столбец дальности отражает ИМЕННО РАДИУС ГЕОМЕТРИИ, а не остроту
распределения. Ожидание: дальность растёт с sigma — чем шире компонента, тем
дальше выбрасывается масса. Если она окажется плоской, значит sigma на
обобщение не влияет и подбирать её незачем.""")

    print("""
ЧЕГО ЭТА ПРОВЕРКА НЕ ГОВОРИТ. Что это повысит успех. Обобщение градиента
полезно ровно в той мере, в какой соседние по решётке коды и правда
равноценны. Если рядом проходит граница мод — обход препятствия с другой
стороны, переключение захвата в другой момент — то размазывание поправки по
окрестности не помогает, а портит. Для того в голове и СМЕСЬ, а не одна
гауссиана: компоненты держат моды раздельно, гладкость только внутри моды.
Проверяется это отдельным замером, не здесь.""")


if __name__ == "__main__":
    main()
