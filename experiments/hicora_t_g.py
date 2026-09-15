#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HiCoRA-T-G: гауссова траекторная голова для RL.

ЧТО ИМЕННО СЛУЧАЙНО

    u ~ N(mu(h24, Z0), sigma_T^2 I_64),  c = rho * tanh(u),
    dZ = reshape(B_T^T c),  Z_T = Z0 + dZ.

ОДИН 64-МЕРНЫЙ ВЕКТОР ШУМА ВЫБИРАЕТ СОГЛАСОВАННОЕ ИЗМЕНЕНИЕ ВСЕЙ ТРАЕКТОРИИ.
В гауссовой HiCoRA шум живёт в 16 * rank координатах, и шестнадцать позиций
чанка дёргаются независимо друг от друга. Здесь возмущение одно на чанк, и
именно это отличие проверяется.

sigma_T НЕЛЬЗЯ ПЕРЕНЕСТИ ИЗ HiCoRA. Размерность другая (64 против 512) и базис
другой, поэтому одно и то же число даёт другое возмущение действий. Масштаб
калибруется офлайн по RMS изменения ДЕКОДИРОВАННЫХ действий — см.
`calibrate_sigma`.

ПОЧЕМУ PPO ВЕДЁТСЯ В ПРОСТРАНСТВЕ u, А НЕ ДЕЙСТВИЙ. Плотности конечного
действия здесь нет: B_T прямоугольная (64 x 8192), декодер ActionCodec
отображает 512 измерений латента в 7 измерений действия и необратим. Поэтому
хранится сэмпл u и его правдоподобие, а якобиан не нужен — всё, что после u,
детерминированный контроллер.

log_std — ОБУЧАЕМЫЙ ВЕКТОР ДЛИНЫ rank, но на первой лестнице ЗАМОРОЖЕН: sigma
фиксируется калибровкой, иначе выбор sigma смешался бы с обучением.
"""
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


def make_gaussian_trajectory_head():
    """Гауссова голова поверх детерминированной. Наследованием, не копией.

    Имена `proj_h.*`, `proj_z.*`, `net.*` обязаны совпасть, иначе чекпойнт
    HiCoRA-T D1 не загрузится в среднюю ветвь и сравнение с проверенным
    результатом станет невозможным. Базис, rho и предел амплитуды тоже
    переиспользуются: дублировать их значило бы разойтись в ограничении, ради
    которого всё и строилось.
    """
    import torch
    import torch.nn as nn
    from hicora_t_vla import make_trajectory_head

    Base = make_trajectory_head()

    class GaussianTrajectoryHead(Base):

        LOG_STD_MIN, LOG_STD_MAX = LOG_STD_MIN, LOG_STD_MAX

        def __init__(self, d_hidden, d_latent, n_pos=16, rank=64, proj=64,
                     hidden=512, init_log_std=-2.3):
            super().__init__(d_hidden, d_latent, n_pos=n_pos, rank=rank,
                             proj=proj, hidden=hidden)
            v = float(init_log_std)
            if not (self.LOG_STD_MIN <= v <= self.LOG_STD_MAX):
                raise ValueError(
                    f"init_log_std={v} вне [{self.LOG_STD_MIN}, "
                    f"{self.LOG_STD_MAX}]: начальная sigma уже была бы у "
                    f"границы, и обучение начиналось с обрезанного градиента")
            self.log_std = nn.Parameter(torch.full((self.rank,), v))

        def freeze_log_std(self):
            """Заморозить sigma ДЕЙСТВИЕМ, а не комментарием.

            На первой лестнице sigma фиксируется калибровкой. Если оставить
            log_std обучаемым «по договорённости», один недосмотр в списке
            параметров оптимизатора сделает sigma обучаемой, и выбор sigma
            смешается с обучением — а заметить это будет нечем.
            """
            self.log_std.requires_grad_(False)
            return self

        def trainable_prefixes(self):
            """Что обучается на первой лестнице. log_std СЮДА НЕ ВХОДИТ."""
            return ("proj_h.", "proj_z.", "net.")

        def std(self):
            return torch.exp(self.log_std.clamp(self.LOG_STD_MIN,
                                                self.LOG_STD_MAX))

        def log_prob_u(self, u, mu, std):
            """log N(u; mu, sigma), просуммированный по rank.

            Суммировать по всей оси координат обязательно: одна поправка на
            чанк — это ОДНО решение, а не 64 независимых.
            """
            var = std * std
            lp = (-0.5 * ((u - mu) ** 2) / var - torch.log(std)
                  - 0.5 * float(np.log(2.0 * np.pi)))
            return lp.flatten(1).sum(-1)

        def forward(self, h, z0, deterministic=False, u=None, generator=None):
            """Поправка, сэмпл и его правдоподобие.

            `u` передаётся НА ОБНОВЛЕНИИ PPO: политика пересчитывает
            log_prob для ТОГО ЖЕ действия, а не сэмплирует новое. Пересэмпл
            сделал бы отношение правдоподобий бессмысленным.
            """
            if not (self._basis_ready and self._rho_ready):
                raise RuntimeError(
                    "не заданы базис или rho: нужны set_basis и set_rho")
            if deterministic and u is not None:
                raise ValueError(
                    "одновременно deterministic=True и заданный u: непонятно, "
                    "что исполнять")
            mu = self.mean_coeffs(h, z0)
            std = self.std().to(mu.dtype)
            if u is None:
                if deterministic:
                    u = mu
                else:
                    eps = torch.empty_like(mu).normal_(generator=generator)
                    u = mu + std * eps
            else:
                # DETACH ОБЯЗАТЕЛЕН, А НЕ НА СОВЕСТИ ВЫЗЫВАЮЩЕГО: на обновлении
                # сохранённое действие — КОНСТАНТА, и градиент должен идти
                # только через mu. Тензор с историей молча подменил бы
                # score-function репараметризацией.
                u = u.detach().to(device=mu.device, dtype=mu.dtype)
                if u.shape != mu.shape:
                    raise ValueError(
                        f"сохранённый u формы {tuple(u.shape)}, ожидалась "
                        f"{tuple(mu.shape)}: это действие другого состояния")
            c = torch.tanh(u)
            dz = (self.rho * c) @ self.basis
            return dict(mu=mu, log_std=self.log_std, std=std, u=u, coeffs=c,
                        dz=dz.unflatten(-1, (self.n_pos, self.d_latent)),
                        log_prob_u=self.log_prob_u(u, mu, std))

    return GaussianTrajectoryHead


def fixed_eps(n_rows, rank, seed=0):
    """ОДИН И ТОТ ЖЕ набор eps для всех sigma при калибровке.

    Со свежим шумом на каждом измерении RMS гуляет от выборки, и двоичный
    поиск гоняется за собственным шумом вместо величины. Общий набор делает
    измерение детерминированной функцией sigma.
    """
    return np.random.default_rng(seed).normal(
        size=(int(n_rows), int(rank))).astype(np.float32)


def rms_action_change(decode, z0, dz_a, dz_b, max_act_q=None):
    """RMS изменения ДЕКОДИРОВАННЫХ действий между двумя поправками.

    Сравнение ведётся в пространстве действий, а не латента: одно и то же
    возмущение латента даёт разное изменение действия в зависимости от базиса,
    а робот исполняет действия.
    """
    a = decode(z0 + dz_a)
    b = decode(z0 + dz_b)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if max_act_q is not None:
        d = d * np.asarray(max_act_q, np.float64)
    return float(np.sqrt((d ** 2).mean()))


def check_monotone(measure, lo=1e-3, hi=1.0, n=7, log=print):
    """Монотонность RMS по sigma на логарифмической сетке.

    Проверки двух концов мало: декодер нелинеен, tanh насыщается, и величина
    могла бы иметь плато или излом внутри. Двоичный поиск при этом вернул бы
    произвольное число, не сообщив об этом.
    """
    grid = np.geomspace(lo, hi, n)
    vals = [measure(float(s)) for s in grid]
    for s_, v in zip(grid, vals):
        log(f"    сетка: sigma {s_:.5f} -> RMS {v:.5f}")
    bad = [(float(grid[i]), vals[i], float(grid[i + 1]), vals[i + 1])
           for i in range(len(vals) - 1) if vals[i + 1] < vals[i]]
    if bad:
        raise SystemExit(
            "RMS не монотонен по sigma: "
            + "; ".join(f"{a:.4f}->{b:.5f} больше {c:.4f}->{d:.5f}"
                        for a, b, c, d in bad[:3])
            + ". Двоичный поиск дал бы произвольное значение")
    return list(zip([float(x) for x in grid], vals))


def calibrate_sigma(target_rms, measure, lo=1e-3, hi=1.0, tol=0.02,
                    max_iter=30, log=print, grid_n=7):
    """Подобрать sigma_T, дающую заданный RMS изменения действий.

    ДЕЛЕНИЕМ ОТРЕЗКА, А НЕ ПЕРЕБОРОМ СЕТКИ: величина монотонна по sigma (шум
    масштабируется, tanh монотонен), поэтому двоичный поиск даёт ответ за
    десяток измерений вместо полного sweep. Полноценный перебор здесь и не
    нужен — это офлайновая калибровка, а не выбор гиперпараметра по успеху.

    Монотонность ПРОВЕРЯЕТСЯ на концах: если её нет, двоичный поиск вернул бы
    произвольное число.
    """
    grid = check_monotone(measure, lo, hi, grid_n, log) if grid_n else None
    r_lo, r_hi = (grid[0][1], grid[-1][1]) if grid else (measure(lo),
                                                         measure(hi))
    if not (r_lo < r_hi):
        raise SystemExit(
            f"RMS не растёт с sigma: {r_lo:.5f} при {lo} и {r_hi:.5f} при "
            f"{hi}. Двоичный поиск дал бы произвольное значение")
    if not (r_lo <= target_rms <= r_hi):
        raise SystemExit(
            f"цель {target_rms:.5f} вне достижимого [{r_lo:.5f}, {r_hi:.5f}] "
            f"на отрезке [{lo}, {hi}]")
    hist = []
    for i in range(max_iter):
        mid = float(np.sqrt(lo * hi))          # геометрическая середина
        r = measure(mid)
        hist.append(dict(sigma=mid, rms=r))
        log(f"    sigma {mid:.5f} -> RMS {r:.5f} (цель {target_rms:.5f})")
        if abs(r - target_rms) <= tol * target_rms:
            return mid, r, dict(search=hist, grid=grid)
        if r < target_rms:
            lo = mid
        else:
            hi = mid
    raise SystemExit(f"за {max_iter} шагов не сошлось: последний RMS "
                     f"{hist[-1]['rms']:.5f} против цели {target_rms:.5f}")


def selftest():
    import torch

    D_H, D_L, NP_, RK = 16, 12, 4, 6
    torch.manual_seed(0)
    Cls = make_gaussian_trajectory_head()
    q, _ = torch.linalg.qr(torch.randn(NP_ * D_L, RK, dtype=torch.float64))
    B = q.T.float()
    RHO = torch.tensor([0.7, 1.3, 0.4, 2.1, 0.9, 1.1])

    def mk(**kw):
        h = Cls(D_H, D_L, n_pos=NP_, rank=RK, proj=5, hidden=16, **kw)
        return h.set_basis(B).set_rho(RHO).eval()

    g = mk()
    h = torch.randn(5, NP_, D_H)
    z = torch.randn(5, NP_, D_L)

    # --- 1. ДЕТЕРМИНИРОВАННОЕ СРЕДНЕЕ СОВПАДАЕТ С D1 ----------------------
    from hicora_t_vla import make_trajectory_head
    d1 = make_trajectory_head()(D_H, D_L, n_pos=NP_, rank=RK, proj=5,
                                hidden=16)
    d1.set_basis(B).set_rho(RHO).eval()
    with torch.no_grad():
        for p_ in list(d1.proj_h.parameters()) + list(d1.proj_z.parameters()) \
                + list(d1.net.parameters()):
            p_.normal_(0.0, 0.3)
    sd = {k: v for k, v in d1.state_dict().items()
          if k.startswith(("proj_h.", "proj_z.", "net."))}
    miss, extra = g.load_state_dict(sd, strict=False)
    assert not [k for k in miss if k.startswith(("proj_h.", "proj_z.", "net."))]
    assert not extra, extra
    with torch.no_grad():
        o = g(h, z, deterministic=True)
        dz_d1, c_d1 = d1(h, z)
    assert torch.allclose(o["coeffs"], c_d1, atol=1e-6), "коэффициенты разошлись"
    assert torch.allclose(o["dz"], dz_d1, atol=1e-6), "поправка разошлась"
    assert torch.equal(o["u"], o["mu"]), "deterministic обязан брать u=mu"

    # --- 2. ШУМ ДОХОДИТ ДО ПОПРАВКИ ----------------------------------------
    g0 = mk()
    with torch.no_grad():
        assert float(g0(h, z, deterministic=True)["dz"].abs().max()) == 0.0
        assert float(g0(h, z)["dz"].abs().max()) > 0.0, "шум не дошёл"

    # --- 3. ОДНО ВОЗМУЩЕНИЕ ДВИГАЕТ ВСЕ ПОЗИЦИИ ---------------------------
    # Ради этого всё и делается: в гауссовой HiCoRA шум независим по позициям.
    with torch.no_grad():
        o3 = g0(h, z)
    per_pos = o3["dz"][0].abs().max(dim=-1).values
    assert float(per_pos.min()) > 0, "часть позиций не затронута шумом"

    # --- 4. ПРЕДЕЛ ДЕРЖИТСЯ И ДЛЯ СЭМПЛОВ ---------------------------------
    lim = g0.bound()
    assert abs(lim - float(torch.linalg.norm(RHO))) < 1e-6
    big = mk(init_log_std=1.0)
    worst = 0.0
    with torch.no_grad():
        for _ in range(40):
            oo = big(torch.randn(64, NP_, D_H), torch.randn(64, NP_, D_L))
            worst = max(worst, float(torch.linalg.norm(
                oo["dz"].flatten(1), dim=-1).max()))
    assert worst <= lim + 1e-4, f"||dZ|| {worst:.4f} превысила {lim:.4f}"

    # --- 5. ПРАВДОПОДОБИЕ ПРОТИВ torch.distributions ----------------------
    with torch.no_grad():
        o5 = g(h, z)
        ref = torch.distributions.Normal(o5["mu"], o5["std"]).log_prob(
            o5["u"]).flatten(1).sum(-1)
    assert o5["log_prob_u"].shape == (5,), o5["log_prob_u"].shape
    assert torch.allclose(o5["log_prob_u"], ref, atol=1e-5)

    # --- 6. ПЕРЕДАННЫЙ u НЕ ПЕРЕСЭМПЛИРУЕТСЯ ------------------------------
    with torch.no_grad():
        again = g(h, z, u=o5["u"])
    assert torch.equal(again["u"], o5["u"]), "u пересэмплирован"
    assert torch.allclose(again["log_prob_u"], o5["log_prob_u"], atol=1e-6)
    assert torch.allclose(again["dz"], o5["dz"], atol=1e-6)
    for kw, needle in ((dict(deterministic=True, u=o5["u"]), "непонятно"),
                       (dict(u=o5["u"][:, :1]), "другого состояния")):
        try:
            g(h, z, **kw)
        except ValueError as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"принято: {kw}")

    # --- 7. ГРАДИЕНТЫ: КУДА ИДУТ И КУДА НЕТ -------------------------------
    zg = z.clone().requires_grad_(True)
    og = g(h, zg)
    (og["log_prob_u"].sum() + og["dz"].sum()).backward()
    assert g.log_std.grad is not None and torch.isfinite(g.log_std.grad).all()
    for nm, p_ in g.named_parameters():
        if nm.startswith(("proj_h.", "proj_z.", "net.")):
            assert p_.grad is not None and torch.isfinite(p_.grad).all(), nm
    assert zg.grad is None or float(zg.grad.abs().max()) == 0.0, \
        "градиент прошёл в черновик"
    assert g.basis.grad is None and g.rho.grad is None

    # --- 8. КАЛИБРОВКА sigma ----------------------------------------------
    # монотонная модель: RMS растёт как sqrt(sigma)
    got, rms, hist = calibrate_sigma(0.02, lambda s: 0.1 * np.sqrt(s),
                                     log=lambda *_: None)
    assert abs(rms - 0.02) <= 0.02 * 0.02, (got, rms)
    assert abs(got - 0.04) < 0.01, got
    assert len(hist["grid"]) == 7 and hist["search"], hist
    # НЕМОНОТОННОСТЬ ВНУТРИ отрезка ловится, хотя концы в порядке
    def humped(s_):
        return 0.1 * np.sqrt(s_) * (0.3 if 0.01 < s_ < 0.1 else 1.0)
    try:
        calibrate_sigma(0.02, humped, log=lambda *_: None)
    except SystemExit as e:
        assert "не монотонен" in str(e), e
    else:
        raise AssertionError("немонотонность внутри отрезка пропущена")
    # ОДИН НАБОР eps: измерение детерминировано по sigma
    e1, e2 = fixed_eps(8, RK, 0), fixed_eps(8, RK, 0)
    assert np.array_equal(e1, e2) and e1.shape == (8, RK)
    assert not np.array_equal(e1, fixed_eps(8, RK, 1))
    # цель ВНЕ отрезка: при 10*s минимум на sigma=1e-3 равен 0.01, и 0.001
    # недостижимо ни при каком sigma из [1e-3, 1]
    # убывающую величину теперь ловит проверка сетки — раньше и подробнее,
    # чем прежняя проверка двух концов
    for tgt, meas, needle in ((0.02, lambda s: 0.5 - s, "не монотонен"),
                              (0.001, lambda s: 10.0 * s, "вне достижимого"),
                              (99.0, lambda s: 10.0 * s, "вне достижимого")):
        try:
            calibrate_sigma(tgt, meas, log=lambda *_: None)
        except SystemExit as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"принято: цель {tgt}, «{needle}»")
    # достижимая цель на том же измерении сходится
    got2, rms2, _ = calibrate_sigma(0.02, lambda s: 10.0 * s,
                                    log=lambda *_: None)
    assert abs(got2 - 0.002) < 1e-4, got2

    # --- 8b. ГРАДИЕНТ ТОЛЬКО ЧЕРЕЗ log_prob С СОХРАНЁННЫМ u ---------------
    # Изоляция score-function пути: прежний тест пускал градиент и через dz, и
    # не отличил бы репараметризацию от правильного пути.
    gg = mk()
    with torch.no_grad():
        for p_ in list(gg.net.parameters()):
            p_.normal_(0, 0.1)
    with torch.no_grad():
        u_saved = gg(h, z)["u"]
    o_up = gg(h, z, u=u_saved)
    assert not o_up["u"].requires_grad, "сохранённый u не отцеплен"
    o_up["log_prob_u"].sum().backward()
    for nm, p_ in gg.named_parameters():
        if nm.startswith("net."):
            assert p_.grad is not None and float(p_.grad.abs().max()) > 0, nm
    # dz при сохранённом u ВООБЩЕ не требует градиента: он зависит только от
    # u, а u — константа. Это и отличает score-function путь от
    # репараметризованного: там градиент шёл бы в net через dz.
    gg.zero_grad(set_to_none=True)
    o_up2 = gg(h, z, u=u_saved)
    assert not o_up2["dz"].requires_grad, \
        "dz при сохранённом u требует градиента: путь репараметризованный"
    assert not o_up2["coeffs"].requires_grad
    # а при СЭМПЛИРОВАНИИ градиент через dz есть — иначе тест был бы пуст
    assert gg(h, z)["dz"].requires_grad

    # --- 8c. ЗАМОРОЗКА log_std ДЕЙСТВИЕМ ----------------------------------
    gf = mk().freeze_log_std()
    assert not gf.log_std.requires_grad
    before = gf.log_std.detach().clone()
    opt = torch.optim.Adam([p_ for n_, p_ in gf.named_parameters()
                            if n_.startswith(gf.trainable_prefixes())], lr=0.1)
    o = gf(h, z)
    (o["log_prob_u"].sum() + o["dz"].sum()).backward()
    opt.step()
    assert torch.equal(gf.log_std.detach(), before), "log_std изменилась"
    assert "log_std" not in " ".join(gf.trainable_prefixes())

    # --- 9. ПРЕДЕЛЫ log_std КОНЕЧНЫ ---------------------------------------
    for bad in (-9.0, 5.0):
        try:
            mk(init_log_std=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"init_log_std={bad} принят")
    with torch.no_grad():
        g.log_std.fill_(100.0)
        assert float(g.std().max()) <= float(np.exp(LOG_STD_MAX)) + 1e-6
        g.log_std.fill_(-100.0)
        assert float(g.std().min()) >= float(np.exp(LOG_STD_MIN)) - 1e-12
    # РАНГ HiCoRA НА ПОЗИЦИЮ — 32, А НЕ 64: сравнение 64 против 16*32 = 512,
    # то есть восьмикратное сокращение. Цифра 1024 вышла бы при сравнении с
    # позиционной головой ранга 64, какой в прогонах не было.
    print(f"самопроверка hicora_t_g пройдена: шум в 64 измерениях на чанк "
          f"против 16*32 = 512 у фактически использованной HiCoRA — "
          f"сокращение в 8 раз")


if __name__ == "__main__":
    selftest()
