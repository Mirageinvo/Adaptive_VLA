"""HiCoRA-G: гауссова голова поправки. ОТДЕЛЬНЫМ МОДУЛЕМ, и это существенно.

ПОЧЕМУ НЕ В hicora_vla.py. Sha того модуля записан в мета-файле кэша головы и
сверяется в `check_hicora_meta` при каждом запуске симуляторного гейта и
стенда латентности. Любая правка — даже чисто добавляющая — сделала бы K-11e
и K-11h невоспроизводимыми: они отказались бы стартовать, сообщив, что модуль
не тот, на котором обучалась голова. Тот же запрет действует для
`joint12_vla.py` со времён K-9. Поэтому детерминированная ветвь берётся
импортом, а не копией: расхождение базиса, предела или stop-gradient между
двумя копиями было бы незаметным и обесценило бы сравнение с D1.

ЧТО ЗДЕСЬ ЕСТЬ. Фабрика `make_gaussian_residual_head()` и восемь проверок,
обязательных ДО PPO. Сама политика и обучение — в следующих скриптах.

Запуск:
    python3 experiments/hicora_g.py          # самопроверка
"""

import numpy as np

from hicora_vla import make_residual_head


def make_gaussian_residual_head():
    """HiCoRA-G: гауссова голова поправки. Рядом с детерминированной, не вместо.

    ЗАЧЕМ НАСЛЕДОВАНИЕМ ОТ ResidualHead. Имена `proj.*` и `net.*` обязаны
    совпасть, иначе чекпойнт D1 не загрузится в среднюю ветвь и сравнение с
    проверенным результатом станет невозможным. Базис, rho, их проверки и
    предел амплитуды тоже переиспользуются: дублировать их значило бы
    разойтись в ограничении, ради которого всё и строилось.

    ЧТО ИМЕННО СЛУЧАЙНО:

        u ~ N(mu, sigma),  c = tanh(u),  dz = B (rho .* c)

    `u` — ВНУТРЕННЕЕ СТОХАСТИЧЕСКОЕ ДЕЙСТВИЕ, и PPO ведётся в его
    пространстве. Всё, что после, — `tanh`, масштаб `rho`, базис `B` и
    декодер ActionCodec — фиксированный детерминированный контроллер.

    ПОЧЕМУ НЕ log pi(a|s) ДЛЯ ДЕЙСТВИЯ РОБОТА. Такой величины здесь просто
    нет: `B` прямоугольная и задаёт подпространство, декодер отображает 512
    измерений латента в 7 измерений действия и необратим. Попытка считать
    плотность конечного действия была бы не «сложной», а бессмысленной.
    Поэтому хранится сэмпл `u`, а

        log pi(u|s) = sum_i log N(u_i; mu_i, sigma_i)

    и никакого якобиана не требуется. Для полноты есть и `log_prob_c` с
    поправкой tanh — но в отношении правдоподобий PPO при ОДНОМ И ТОМ ЖЕ
    сохранённом `u` эта поправка сокращается.

    log_std — ОБУЧАЕМЫЙ ВЕКТОР РАЗМЕРА rank, а не отдельная сеть от
    состояния. Так устойчивее на старте RL; state-dependent sigma остаётся
    абляцией, а не первой версией.

    ЧТО ОБУЧАЕТСЯ НА ЭТОМ ЭТАПЕ: только `mu`-ветвь (`proj.*`, `net.*`),
    `log_std` и голова ценности. Ствол, голова черновика, `B`, `rho` и
    декодер заморожены — тогда отображение латентного действия в действие
    робота не зависит от обучаемых параметров, и PPO корректен.
    """
    import torch
    import torch.nn as nn

    Base = make_residual_head()

    class GaussianResidualHead(Base):
        # ПРЕДЕЛЫ log_std КОНЕЧНЫЕ. Без них градиент уводит sigma либо в нуль
        # (политика перестаёт исследовать и PPO стоит), либо в бесконечность
        # (каждый сэмпл в насыщении tanh, и поправка становится шумом).
        LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0

        def __init__(self, d_hidden, d_latent, rank=16, hidden=512, proj=64,
                     init_log_std=-2.3):
            super().__init__(d_hidden, d_latent, rank=rank, hidden=hidden,
                             proj=proj)
            v = float(init_log_std)
            if not (self.LOG_STD_MIN <= v <= self.LOG_STD_MAX):
                raise ValueError(
                    f"init_log_std={v} вне [{self.LOG_STD_MIN}, "
                    f"{self.LOG_STD_MAX}]: начальная sigma уже была бы у "
                    f"границы, и обучение начиналось с обрезанного градиента")
            self.log_std = nn.Parameter(torch.full((self.rank,), v))

        def log_std_eff(self):
            """ФАКТИЧЕСКИ ИСПОЛЬЗУЕМЫЙ log_std, то есть обрезанный.

            Возвращать сырой параметр было бы ловушкой: sigma считается из
            обрезанного значения, и диагностика противоречила бы реально
            исполнявшемуся распределению — например, показывала бы log_std=12
            там, где политика шла с exp(1).
            """
            return self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

        def std(self):
            return torch.exp(self.log_std_eff())

        def mean_coeffs(self, h, z0):
            """Средние ДО tanh. Именно это и есть `mu`.

            Родительский `coeffs` возвращает tanh(net(x)); здесь нужно
            предсжатое значение, иначе при sigma -> 0 голова не сводилась бы к
            D1 в точности.
            """
            x = torch.cat([h, self.proj(z0.detach())], dim=-1)
            return self.net(x)

        def log_prob_u(self, u, mu, std):
            """log N(u; mu, sigma), просуммированный по позициям И по rank.

            Суммировать обязательно по всем осям кроме батча: `u` имеет форму
            [batch, n_pos, rank], и одна поправка на чанк — это ОДНО решение,
            а не n_pos*rank независимых.
            """
            # СЧИТАЕТСЯ В fp32 ВСЕГДА. Под autocast fp16 сумма по 512
            # координатам теряет точность, а отношение правдоподобий PPO
            # чувствительно к разнице двух больших почти равных чисел.
            u32, mu32 = u.float(), mu.float()
            std32 = std.float()
            lp = (-0.5 * ((u32 - mu32) ** 2) / (std32 * std32)
                  - torch.log(std32)
                  - 0.5 * float(np.log(2.0 * np.pi)))
            return lp.flatten(1).sum(-1)

        def log_prob_c(self, u, mu, std, eps=1e-6):
            """То же в координатах c = tanh(u), с поправкой tanh.

            НУЖНА НЕ ДЛЯ PPO. В отношении правдоподобий при одном и том же
            сохранённом `u` поправка сокращается. Приводится, чтобы величина
            была определена явно, а не «как-нибудь».
            """
            c = torch.tanh(u)
            corr = torch.log1p(-(c ** 2) + eps)
            return self.log_prob_u(u, mu, std) - corr.flatten(1).sum(-1)

        def forward(self, h, z0, deterministic=False, u=None, generator=None):
            """Поправка, сэмпл и его правдоподобие.

            `u` передаётся НА ОБНОВЛЕНИИ PPO: политика пересчитывает
            log_prob для того же действия, а не сэмплирует новое. Пересэмпл на
            обновлении сделал бы отношение правдоподобий бессмысленным.
            """
            if not (self._basis_ready and self._rho_ready):
                raise RuntimeError(
                    "не заданы базис или rho: нужны set_basis и set_rho. "
                    "Нулевой базис дал бы тождественно нулевую поправку, и "
                    "обучение «сошлось» бы, ничего не выучив")
            if deterministic and u is not None:
                raise ValueError(
                    "одновременно deterministic=True и заданный u: "
                    "непонятно, что исполнять. Для пересчёта правдоподобия "
                    "сохранённого действия deterministic не нужен")
            mu = self.mean_coeffs(h, z0)
            std = self.std().to(mu.dtype)
            if u is None:
                if deterministic:
                    u = mu
                else:
                    eps = torch.empty_like(mu).normal_(generator=generator)
                    u = mu + std * eps
            else:
                # И УСТРОЙСТВО, НЕ ТОЛЬКО dtype: сохранённый сэмпл
                # приходит из буфера rollout, где он лежит на CPU, а `mu`
                # считается на карте. Без переноса обновление PPO падало бы.
                u = u.to(device=mu.device, dtype=mu.dtype)
                if u.shape != mu.shape:
                    raise ValueError(
                        f"сохранённый u формы {tuple(u.shape)}, ожидалась "
                        f"{tuple(mu.shape)}: это действие от другого состояния")
            c = torch.tanh(u)
            dz = (self.rho * c) @ self.basis.T
            return dict(mu=mu, log_std=self.log_std_eff(), std=std, u=u,
                        coeffs=c, dz=dz,
                        log_prob_u=self.log_prob_u(u, mu, std))

    return GaussianResidualHead




def selftest():
    import torch

    # ======================= HiCoRA-G: ГАУССОВА ГОЛОВА ======================
    # Восемь обязательных проверок ДО PPO. Смысл каждой — в её тексте: без
    # них можно потратить долгий rollout на режим, где шум либо ничего не
    # меняет, либо сразу разрушает стратегию, и не понять, что сломалось.
    torch.manual_seed(0)
    D_H, D_L, RK, NP_ = 24, 32, 4, 6
    Gh = make_gaussian_residual_head()
    Dh = make_residual_head()

    def _basis(d, r):
        q, _ = torch.linalg.qr(torch.randn(d, r, dtype=torch.float64))
        return q.to(torch.float32)

    B_ = _basis(D_L, RK)
    RHO_ = torch.tensor([0.7, 1.3, 0.4, 2.1])

    def _mk(cls, **kw):
        m = cls(D_H, D_L, rank=RK, hidden=16, proj=8, **kw)
        m.set_basis(B_)
        m.set_rho(RHO_)
        return m.eval()

    g = _mk(Gh)
    d1 = _mk(Dh)
    h_ = torch.randn(5, NP_, D_H)
    z_ = torch.randn(5, NP_, D_L)

    # --- 1. ЗАГРУЗКА D1 В СРЕДНЮЮ ВЕТВЬ ДАЁТ ТО ЖЕ САМОЕ -------------------
    # Если имена или порядок слоёв разойдутся, сравнение с проверенным
    # результатом K-11c/K-11e станет невозможным, а расхождение выглядело бы
    # как «гауссова голова хуже».
    for p_ in d1.net.parameters():
        torch.nn.init.normal_(p_, 0.0, 0.3)
    for p_ in d1.proj.parameters():
        torch.nn.init.normal_(p_, 0.0, 0.3)
    sd = {k: v for k, v in d1.state_dict().items()
          if k.startswith(("proj.", "net."))}
    miss, extra = g.load_state_dict(sd, strict=False)
    assert not [k for k in miss if k.startswith(("proj.", "net."))], miss
    assert not extra, extra
    with torch.no_grad():
        o = g(h_, z_, deterministic=True)
        dz_d1, c_d1 = d1(h_, z_)
    assert torch.allclose(o["coeffs"], c_d1, atol=1e-6), "коэффициенты разошлись"
    assert torch.allclose(o["dz"], dz_d1, atol=1e-6), "поправка разошлась"
    assert torch.allclose(o["u"], o["mu"]), "deterministic обязан брать u=mu"

    # --- 2. mu=0 И ДЕТЕРМИНИЗМ -> СТРОГО НУЛЕВАЯ ПОПРАВКА ------------------
    g0 = _mk(Gh)
    with torch.no_grad():
        o0 = g0(h_, z_, deterministic=True)
    assert float(o0["dz"].abs().max()) == 0.0, "нулевое тождество нарушено"
    assert float(o0["coeffs"].abs().max()) == 0.0

    # --- 3. mu=0, sigma>0, СЭМПЛИРОВАНИЕ -> ПОПРАВКА НЕНУЛЕВАЯ -------------
    # Это НЕ противоречит пункту 2: нулевое тождество выполняется для
    # детерминированного среднего, а не для сэмплированной политики, где
    # c = tanh(sigma*eps) != 0. Путать их значило бы считать, что RL стартует
    # из точной копии D1.
    with torch.no_grad():
        o3 = g0(h_, z_)
    assert float(o3["dz"].abs().max()) > 0.0, "шум не дошёл до поправки"
    assert float(o3["std"].min()) > 0.0

    # --- 4. ПРЕДЕЛ АМПЛИТУДЫ ДЕРЖИТСЯ И ДЛЯ СЭМПЛОВ ------------------------
    # Ради этого предела базис и делался ортонормированным и замороженным.
    lim = g0.bound()
    assert abs(lim - float(torch.linalg.norm(RHO_))) < 1e-6
    gbig = _mk(Gh, init_log_std=1.0)        # sigma = e, почти насыщение
    with torch.no_grad():
        worst = 0.0
        for _ in range(40):
            oo = gbig(torch.randn(64, NP_, D_H), torch.randn(64, NP_, D_L))
            worst = max(worst, float(
                torch.linalg.norm(oo["dz"], dim=-1).max()))
    assert worst <= lim + 1e-5, f"||dz||={worst:.4f} превысила предел {lim:.4f}"

    # --- 5. log_prob_u СОВПАДАЕТ С torch.distributions --------------------
    with torch.no_grad():
        o5 = g(h_, z_)
        ref = torch.distributions.Normal(o5["mu"], o5["std"]).log_prob(
            o5["u"]).flatten(1).sum(-1)
    assert o5["log_prob_u"].shape == (5,), o5["log_prob_u"].shape
    assert torch.allclose(o5["log_prob_u"], ref, atol=1e-5), "правдоподобие"

    # --- 6. ПЕРЕДАННЫЙ u НЕ ПЕРЕСЭМПЛИРУЕТСЯ ------------------------------
    # На обновлении PPO политика обязана пересчитать правдоподобие ТОГО ЖЕ
    # действия. Пересэмпл сделал бы отношение правдоподобий бессмысленным.
    with torch.no_grad():
        again = g(h_, z_, u=o5["u"])
    assert torch.equal(again["u"], o5["u"]), "u пересэмплирован"
    assert torch.allclose(again["log_prob_u"], o5["log_prob_u"], atol=1e-6)
    assert torch.allclose(again["dz"], o5["dz"], atol=1e-6)
    try:
        g(h_, z_, deterministic=True, u=o5["u"])
    except ValueError:
        pass
    else:
        raise AssertionError("deterministic вместе с заданным u принят")
    try:
        g(h_, z_, u=o5["u"][:, :1])
    except ValueError:
        pass
    else:
        raise AssertionError("u чужой формы принят")

    # --- 7. ГРАДИЕНТЫ: КУДА ИДУТ И КУДА НЕ ИДУТ ---------------------------
    zg = z_.clone().requires_grad_(True)
    og = g(h_, zg)
    (og["log_prob_u"].sum() + og["dz"].sum()).backward()
    assert g.log_std.grad is not None and torch.isfinite(g.log_std.grad).all()
    for nm_, p_ in g.named_parameters():
        if nm_.startswith(("proj.", "net.")):
            assert p_.grad is not None and torch.isfinite(p_.grad).all(), nm_
    # STOP-GRADIENT НА ЧЕРНОВИКЕ: голова исправляет то, что модель предсказала
    # сама, и не переучивает q0 через себя.
    assert zg.grad is None or float(zg.grad.abs().max()) == 0.0, \
        "градиент прошёл в z0"
    assert g.basis.grad is None and g.rho.grad is None, "базис или rho учатся"

    # --- 7b. ГРАДИЕНТ PPO-ОБНОВЛЕНИЯ: ПУТЬ score-function -----------------
    # Пункт 7 смешивал log_prob с dz.sum() и брал репараметризованный `u`,
    # поэтому проходил бы и при сломанном score-function пути: градиент
    # дотекал бы через сам сэмпл. На обновлении PPO `u` — КОНСТАНТА из
    # буфера, и градиент обязан идти только через mu и log_std.
    g7 = _mk(Gh)
    for p_ in list(g7.net.parameters()) + list(g7.proj.parameters()):
        torch.nn.init.normal_(p_, 0.0, 0.3)
    with torch.no_grad():
        u_buf = g7(h_, z_)["u"].detach().clone()
    assert not u_buf.requires_grad
    g7.zero_grad(set_to_none=True)
    o7 = g7(h_, z_, u=u_buf)
    assert o7["u"].grad_fn is None, "переданный u попал в граф"
    (-o7["log_prob_u"].mean()).backward()
    assert g7.log_std.grad is not None
    assert torch.isfinite(g7.log_std.grad).all()
    assert float(g7.log_std.grad.abs().max()) > 0.0, "log_std не учится"
    for nm_, p_ in g7.named_parameters():
        if nm_.startswith(("proj.", "net.")):
            assert p_.grad is not None and torch.isfinite(p_.grad).all(), nm_
    assert float(g7.net[-1].weight.grad.abs().max()) > 0.0, \
        "средняя ветвь не учится от log_prob"
    # КОНТРОЛЬ: без градиента по mu путь был бы мёртв. Отцепив mu, требуем
    # нуля — иначе тест не различал бы рабочий и сломанный случай.
    g7.zero_grad(set_to_none=True)
    o7b = g7(h_, z_, u=u_buf)
    (-g7.log_prob_u(u_buf, o7b["mu"].detach(), o7b["std"]).mean()).backward()
    assert g7.net[-1].weight.grad is None or \
        float(g7.net[-1].weight.grad.abs().max()) == 0.0

    # --- 8. log_prob_c ПРОТИВ TransformedDistribution ---------------------
    with torch.no_grad():
        o8 = g(h_, z_)
        td = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(o8["mu"], o8["std"]),
            torch.distributions.TanhTransform(cache_size=1))
        ref_c = td.log_prob(torch.tanh(o8["u"])).flatten(1).sum(-1)
        got_c = g.log_prob_c(o8["u"], o8["mu"], o8["std"], eps=0.0)
    assert torch.allclose(got_c, ref_c, atol=1e-3), (got_c - ref_c).abs().max()
    # В ОТНОШЕНИИ ПРАВДОПОДОБИЙ PPO ПОПРАВКА СОКРАЩАЕТСЯ при одном и том же u.
    g2 = _mk(Gh)
    with torch.no_grad():
        a = g(h_, z_, u=o8["u"])
        b = g2(h_, z_, u=o8["u"])
        r_u = a["log_prob_u"] - b["log_prob_u"]
        r_c = (g.log_prob_c(o8["u"], a["mu"], a["std"])
               - g2.log_prob_c(o8["u"], b["mu"], b["std"]))
    assert torch.allclose(r_u, r_c, atol=1e-5), "поправка не сократилась"

    # --- предел log_std конечен -------------------------------------------
    for bad in (-9.0, 5.0):
        try:
            _mk(Gh, init_log_std=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"init_log_std={bad} принят")
    with torch.no_grad():
        g.log_std.fill_(100.0)
        assert float(g.std().max()) <= float(np.exp(Gh.LOG_STD_MAX)) + 1e-6
        # ВОЗВРАЩАЕТСЯ ОБРЕЗАННЫЙ log_std, А НЕ СЫРОЙ ПАРАМЕТР. Иначе
        # диагностика показывала бы log_std=100 там, где политика шла с
        # exp(1), и записанное в лог распределение не было бы исполнявшимся.
        ocl = g(h_, z_, deterministic=True)
        assert float(ocl["log_std"].max()) <= Gh.LOG_STD_MAX + 1e-6, \
            "вернулся необрезанный log_std"
        assert torch.allclose(torch.exp(ocl["log_std"]), ocl["std"],
                              atol=1e-6), "log_std и std не согласованы"
        g.log_std.fill_(-100.0)
        assert float(g.std().min()) >= float(np.exp(Gh.LOG_STD_MIN)) - 1e-12
        ocl = g(h_, z_, deterministic=True)
        assert float(ocl["log_std"].min()) >= Gh.LOG_STD_MIN - 1e-6
        assert torch.allclose(torch.exp(ocl["log_std"]), ocl["std"],
                              atol=1e-9)

    print("самопроверка hicora_g пройдена: D1 грузится в среднюю ветвь и "
          "даёт\n  побитово то же в детерминированном режиме; нулевое "
          "тождество держится для\n  среднего, а сэмплы ненулевые; предел "
          "||dz|| <= ||rho|| выполняется и на\n  сэмплах при насыщающей "
          "sigma; log pi(u) сходится с Normal и с\n  TanhTransform; "
          "переданный u не пересэмплируется; в z0 градиент не идёт;\n  "
          "log_std ограничен с обеих сторон и возвращается обрезанным;\n  "
          "градиент обновления PPO идёт через mu и log_std при постоянном u")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    selftest()
