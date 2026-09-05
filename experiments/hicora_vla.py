"""HiCoRA-VLA: грубый дискретный черновик на слое 12 плюс непрерывная поправка.

ЧТО ЭТО. Один проход по двадцати четырём слоям. На слое 12 голова q0 выдаёт
дискретный черновик; оставшиеся слои того же прохода вычисляют непрерывную
поправку к латенту ЭТОГО черновика; декодер ActionCodec вызывается ровно один
раз для суммы. Это не ранний выход и не три запуска модели.

    q0_hat = argmax H12(norm(h12))
    z0     = C0[q0_hat]
    dz     = Basis(rho * tanh(Hres(norm(h24), P(sg(z0)))))
    A      = D(z0 + dz)

ПОЧЕМУ sg(z0). Поправка обязана исправлять черновик, а не переучивать голову
q0 через себя: без stop-gradient градиент поправки потёк бы в кодовую книгу и
в голову двенадцатого слоя, и «черновик» перестал бы быть тем, что модель
предсказывает самостоятельно. Кодовые книги при этом заморожены и так, но
sg делает независимость свойством архитектуры, а не режима обучения.

ПОЧЕМУ tanh, rho И ЗАМОРОЖЕННЫЙ БАЗИС. Ограничение коэффициентов само по
себе НИЧЕГО НЕ ОГРАНИЧИВАЕТ, если базис обучаемый: |c| <= 1 и |rho*c| <= rho,
но линейный слой волен увеличить веса во сто раз и обойти любой предел.
Поэтому базис здесь ФИКСИРОВАН и ОРТОНОРМИРОВАН: строится на обучающей части,
загружается через `set_basis`, замораживается. Тогда ||dz|| = ||rho .* c|| <=
||rho|| — это настоящая граница, а не тавтология. Обучаются только
коэффициенты. rho считается на обучающей части в координатах ЭТОГО базиса и
кладётся в чекпойнт; доля насыщенных коэффициентов печатается при каждом
разборе — постоянное насыщение есть стоп-условие ветки, а не деталь.

ПОСЛЕДНИЙ ЛИНЕЙНЫЙ СЛОЙ ИНИЦИАЛИЗИРОВАН НУЛЯМИ. Это делает нулевое тождество
проверяемым: до обучения модель обязана побитово совпадать с базовым coarse
выходом головы q0. Тождество проверяет K-11b.

СХВАТ МЕНЯТЬСЯ МОЖЕТ, И ЭТО ИЗМЕРЯЕТСЯ. Ранее здесь стояло «схват берётся из
q0» — утверждение неверное: декодер ActionCodec отображает латент СОВМЕСТНО во
все семь каналов, поэтому любое изменение z меняет и схват. Механизма,
удерживающего схват от D(z0), в латентном варианте нет. Принято: HiCoRA-D
остаётся ЛАТЕНТНОЙ поправкой, схват считается изменяемым, его согласие с
D(z0) измеряется отдельной величиной и входит в стоп-условия. Смешивать
ошибку схвата с позиционной в одно число по-прежнему нельзя — ровно из-за
этого офлайновый гейт K-10d оказался неспособен подтверждать (K-10g).

`joint12_vla.py` НЕ ИЗМЕНЯЕТСЯ. Его sha записан в каждой ячейке симуляторного
гейта K-9; правка расколола бы развёртку. Здесь отдельный класс с собственным
проходом на полную глубину и отводами.
"""

import numpy as np

TAPS = (12, 18, 24)
N_POS = 16


def make_residual_head():
    """Класс головы поправки.

    ОТДЕЛЬНОЙ ФАБРИКОЙ, а не вложением в модель: вложенный класс нельзя ни
    создать в самопроверке, ни импортировать в обучающий скрипт, и его
    пришлось бы дублировать — ровно тот путь, каким разошлись разметчики
    событий в K-10. Torch импортируется внутри, чтобы модуль оставался
    импортируемым там, где его нет.
    """
    import torch
    import torch.nn as nn

    class ResidualHead(nn.Module):
        """Непрерывная поправка к латенту предсказанного черновика.

        БАЗИС ОБЩИЙ ДЛЯ ШЕСТНАДЦАТИ ПОЗИЦИЙ, коэффициенты — свои у каждой.
        Позиции чанка описывают один и тот же тип величины в разные моменты;
        отдельный базис на позицию утроил бы параметры ради структуры,
        которой в данных, скорее всего, нет. Если диагностика K-11a покажет
        разные подпространства по позициям, это придётся пересмотреть.
        """

        def __init__(self, d_hidden, d_latent, rank=16, hidden=512, proj=64):
            super().__init__()
            self.rank, self.d_latent = int(rank), int(d_latent)
            self.proj = nn.Linear(d_latent, proj)
            self.net = nn.Sequential(
                nn.Linear(d_hidden + proj, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, self.rank))
            # БАЗИС — БУФЕР, А НЕ СЛОЙ. Обучаемый базис обесценил бы rho:
            # предел на коэффициенты обходится масштабированием столбцов.
            self.register_buffer("basis", torch.zeros(d_latent, self.rank))
            self.register_buffer("basis_set", torch.zeros(1))
            # НУЛЕВАЯ ИНИЦИАЛИЗАЦИЯ ПОСЛЕДНЕГО СЛОЯ: до обучения поправка
            # строго нулевая, и нулевое тождество проверяемо.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            # rho — БУФЕР, а не параметр: предел амплитуды задаётся данными в
            # K-11a и не должен подстраиваться градиентом под потерю.
            self.register_buffer("rho", torch.ones(self.rank))

        def set_basis(self, B):
            """Ортонормированный базис поправок, построенный на train.

            ОРТОНОРМИРОВАННОСТЬ ПРОВЕРЯЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ: только при
            B^T B = I выполняется ||dz|| = ||rho .* c||, то есть граница
            амплитуды становится настоящей.
            """
            b = torch.as_tensor(B, dtype=self.basis.dtype)
            if tuple(b.shape) != tuple(self.basis.shape):
                raise ValueError(f"базис формы {tuple(b.shape)}, ожидалась "
                                 f"{tuple(self.basis.shape)}")
            if not torch.isfinite(b).all():
                raise ValueError("в базисе есть nan или inf")
            g = b.T @ b
            off = float((g - torch.eye(self.rank, dtype=g.dtype)).abs().max())
            if off > 1e-4:
                raise ValueError(f"базис не ортонормирован: max|B^T B - I| = "
                                 f"{off:.2e}")
            self.basis.data = b.to(self.basis.device)
            self.basis_set.data = torch.ones_like(self.basis_set)

        def bound(self):
            """Гарантированный предел ||dz||_2. Настоящий, а не тавтология."""
            return float(torch.linalg.norm(self.rho))

        def set_rho(self, rho):
            r = torch.as_tensor(rho, dtype=self.rho.dtype)
            if tuple(r.shape) != tuple(self.rho.shape):
                raise ValueError(f"rho формы {tuple(r.shape)}, ожидалась "
                                 f"{tuple(self.rho.shape)}")
            if not torch.isfinite(r).all() or bool((r <= 0).any()):
                raise ValueError("rho обязана быть конечной и положительной")
            self.rho.data = r.to(self.rho.device)

        def coeffs(self, h, z0):
            # STOP-GRADIENT НА ЧЕРНОВИКЕ: голова исправляет то, что модель
            # предсказала сама, и не переучивает q0 через себя.
            x = torch.cat([h, self.proj(z0.detach())], dim=-1)
            return torch.tanh(self.net(x))

        def forward(self, h, z0):
            """Поправка и коэффициенты — вторые нужны для учёта насыщения."""
            if float(self.basis_set) == 0.0:
                raise RuntimeError(
                    "базис не задан: сначала set_basis по обучающей части. "
                    "Нулевой базис дал бы тождественно нулевую поправку и "
                    "обучение «сошлось» бы, ничего не выучив")
            c = self.coeffs(h, z0)
            return (self.rho * c) @ self.basis.T, c

    return ResidualHead


def make_hicora_class(base_cls):
    import torch
    import torch.nn as nn

    class _HiCoRAVLA(base_cls):
        """Полная глубина за один проход, отводы на 12/18/24."""

        def init_hicora(self, q0_depth=12, rank=16, hidden=512, proj=64,
                        taps=TAPS):
            if int(q0_depth) not in taps:
                raise ValueError(f"глубина q0 {q0_depth} не входит в отводы "
                                 f"{taps}")
            self.q0_depth = int(q0_depth)
            self.taps = tuple(int(t) for t in taps)
            self.n_layers_total = len(self.action_expert.layers)
            if max(self.taps) != self.n_layers_total:
                raise ValueError(
                    f"последний отвод {max(self.taps)} не равен числу слоёв "
                    f"{self.n_layers_total}: проход был бы неполным")
            if not hasattr(self, "fast_head"):
                raise RuntimeError(
                    "нет головы q0: сначала init_joint_fast и загрузка весов "
                    "K-9c, иначе черновик брать неоткуда")
            if not hasattr(self, "hicora_d_latent"):
                raise RuntimeError(
                    "сначала set_codebooks: размерность латента берётся из "
                    "кодовых книг, а не задаётся отдельно — иначе базис можно "
                    "собрать под чужую размерность и заметить это только при "
                    "загрузке весов")
            d_hidden = int(self.fast_head.in_features)
            self.hicora_head = make_residual_head()(
                d_hidden, self.hicora_d_latent, rank=rank, hidden=hidden,
                proj=proj)
            self.hicora_head.to(next(self.parameters()).device)
            return self

        def set_codebooks(self, E):
            """Кодовые книги как БУФЕР: замороженные и едущие с моделью."""
            e = torch.as_tensor(E).float()
            if e.dim() != 3:
                raise ValueError(f"книги формы {tuple(e.shape)}, ждали "
                                 f"(уровней, кодов, размерность)")
            self.register_buffer("codebooks", e)
            self.hicora_d_latent = int(e.shape[-1])
            return self

        def forward_taps(self, *, vlm_inputs_embeds, attention_mask,
                         position_ids):
            """Полная глубина, состояние потока действий на каждом отводе.

            Цикл дословно повторяет `forward_joint_fast`, кроме глубины и
            съёма отводов: расхождение здесь означало бы, что q0 этой ветки
            не тот, что у Joint12, и вся конструкция сравнивалась бы не с
            тем черновиком. K-11b проверяет совпадение побитово.
            """
            B = vlm_inputs_embeds.shape[0]
            dev, dt = vlm_inputs_embeds.device, vlm_inputs_embeds.dtype
            n = self.block_size
            bos = self.bos_embedding.expand(B, n, -1).to(dev, dt)
            empty = torch.empty((B, 0, bos.shape[-1]), device=dev, dtype=dt)
            action_hidden = torch.cat([bos, empty], dim=1)
            vlm_hidden = vlm_inputs_embeds
            mask4d = self._build_joint_attention_mask_blockwise_ar(
                attention_mask=attention_mask,
                vlm_seq_len=vlm_inputs_embeds.shape[1], action_seq_len=n,
                device=dev,
                action_key_mask=torch.ones((B, n), device=dev,
                                           dtype=torch.long))
            out, seen = {}, 0
            for layer_idx in range(self.n_layers_total):
                vlm_hidden, action_hidden = self._shared_attention_forward(
                    vlm_hidden_states=vlm_hidden,
                    action_hidden_states=action_hidden,
                    layer_idx=layer_idx, attention_mask=mask4d,
                    position_ids=position_ids, past_key_values=None,
                    use_cache=False, cache_position=None)
                seen += 1
                if seen in self.taps:
                    out[seen] = action_hidden
            if seen != self.n_layers_total:
                raise RuntimeError(f"исполнено {seen} слоёв из "
                                   f"{self.n_layers_total}")
            # СЧЁТЧИК ИСПОЛНЕННЫХ СЛОЁВ ВОЗВРАЩАЕТСЯ НАРУЖУ: заявление «один
            # проход» обязано быть измеримым, а не декларируемым.
            out["layers_run"] = seen
            return out

        def set_res_norm(self, module):
            """Норма для ПОЗДНЕЙ ветви, отдельная от нормы черновика.

            Чекпойнт Joint12 перезаписывает `action_expert.norm`, а она
            обучалась читать h12. Позняя ветвь должна читать h24 исходной
            финальной нормой, иначе вход головы поправки нормируется
            статистикой другой глубины. Здесь исходная норма сохраняется до
            наложения весов и подставляется явно.
            """
            self.res_norm = module
            return self

        def q0_from(self, h):
            """Черновик из отвода: та же норма и голова, что у Joint12."""
            normed = self.action_expert.norm(h)
            logits = self.fast_head(normed.to(self.fast_head.weight.dtype))
            return logits, logits.argmax(-1)

        def forward_hicora(self, *, vlm_inputs_embeds, attention_mask,
                           position_ids):
            taps = self.forward_taps(vlm_inputs_embeds=vlm_inputs_embeds,
                                     attention_mask=attention_mask,
                                     position_ids=position_ids)
            logits, q0 = self.q0_from(taps[self.q0_depth])
            z0 = self.codebooks[0][q0]
            if not hasattr(self, "res_norm"):
                raise RuntimeError(
                    "не задана res_norm: поздняя ветвь читала бы h24 нормой "
                    "Joint12, обученной на h12. Вызовите set_res_norm с "
                    "ИСХОДНОЙ финальной нормой, снятой до наложения весов")
            h24 = self.res_norm(taps[max(self.taps)]).float()
            dz, c = self.hicora_head(h24, z0)
            return dict(q0=q0, q0_logits=logits, z0=z0, dz=dz, z=z0 + dz,
                        coeffs=c, layers_run=taps["layers_run"],
                        saturated=float((c.abs() > 0.99).float().mean()))

        def hicora_trainable_prefixes(self):
            """Что обучается на этапе D1. Всё прочее заморожено."""
            return ("hicora_head.",)

    return _HiCoRAVLA


def saturation_report(c, thr=0.99):
    """Доля коэффициентов у границы. Печатается всегда, не по запросу."""
    a = np.abs(np.asarray(c, np.float64))
    return dict(frac=float((a > thr).mean()), mean_abs=float(a.mean()),
                p95=float(np.percentile(a, 95)))


def rho_from_residual(coef, q=95.0):
    """Предел амплитуды по РЕАЛЬНОМУ остатку обучающей части.

    Процентиль, а не максимум: единичный выброс задрал бы предел так, что
    ограничение перестало бы ограничивать. Считается только на train — подбор
    по test запрещён планом и здесь просто нечем воспользоваться.
    """
    a = np.abs(np.asarray(coef, np.float64))
    if a.ndim != 2:
        raise ValueError(f"ожидалась матрица (наблюдения, ранг), дано "
                         f"{a.shape}")
    r = np.percentile(a, q, axis=0)
    if not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError("процентиль дал ноль или бесконечность: остаток "
                         "вырожден хотя бы по одной координате базиса")
    return r


def selftest():
    # --- насыщение ---------------------------------------------------------
    c = np.concatenate([np.full(90, 0.5), np.full(10, 0.999)])
    assert abs(saturation_report(c)["frac"] - 0.10) < 1e-12

    # --- предел амплитуды --------------------------------------------------
    rng = np.random.default_rng(0)
    coef = rng.normal(0, 1.0, size=(10000, 4))
    coef[0] = 1e6                       # выброс
    r = rho_from_residual(coef, q=95.0)
    # ВЫБРОС НЕ ЗАДИРАЕТ ПРЕДЕЛ: иначе ограничение перестало бы ограничивать.
    assert r.max() < 3.0 and (r > 1.0).all(), r
    try:
        rho_from_residual(np.zeros((10, 3)))
        raise AssertionError("вырожденный остаток прошёл")
    except ValueError:
        pass

    try:
        import torch
    except ImportError:
        print("самопроверка hicora_vla пройдена ЧАСТИЧНО (torch нет): "
              "насыщение и предел амплитуды проверены, СЕТЬ НЕ ПРОВЕРЕНА")
        return

    RH = make_residual_head()
    d_h, d_z, rank = 24, 32, 8
    head = RH(d_h, d_z, rank=rank, hidden=64, proj=16)
    h = torch.randn(3, N_POS, d_h)
    z0 = torch.randn(3, N_POS, d_z, requires_grad=True)

    # --- БЕЗ БАЗИСА ГОЛОВА ОТКАЗЫВАЕТ ---------------------------------------
    # Нулевой базис дал бы тождественно нулевую поправку, и обучение
    # «сошлось» бы, не выучив ничего.
    try:
        head(h, z0)
        raise AssertionError("голова без базиса не отказала")
    except RuntimeError:
        pass
    B = torch.linalg.qr(torch.randn(d_z, rank))[0]
    head.set_basis(B)
    for bad in (torch.randn(d_z, rank), B[:, :rank - 1], B * 2.0):
        try:
            head.set_basis(bad)
            raise AssertionError("неортонормированный базис принят")
        except ValueError:
            pass
    head.set_basis(B)

    # --- НУЛЕВОЕ ТОЖДЕСТВО --------------------------------------------------
    dz, c0 = head(h, z0)
    assert float(dz.detach().abs().max()) == 0.0, "поправка не нулевая"
    assert float(c0.abs().max()) == 0.0

    # --- ГОЛОВА ЖИВАЯ -------------------------------------------------------
    # ПРЕЖНЯЯ ВЕРСИЯ ЭТОГО ТЕСТА БЫЛА ПУСТОЙ: потеря вида sum(y^2) при
    # нулевом выходе даёт нулевой градиент по построению, и тест проходил бы
    # и на мёртвой голове. Здесь линейная потеря.
    head.zero_grad()
    head.net(torch.cat([h, head.proj(z0.detach())], -1)).sum().backward()
    g = head.net[-1].weight.grad
    assert g is not None and float(g.abs().sum()) > 0, "голова мертва"

    # --- ГРАДИЕНТ НЕ ТЕЧЁТ В ЧЕРНОВИК ---------------------------------------
    # ПРЕЖНЯЯ ВЕРСИЯ БЫЛА ВЫРОЖДЕННОЙ: при нулевом последнем слое градиент по
    # z0 равен нулю и с detach, и без него. Сначала делаем голову ненулевой,
    # затем требуем ОДНОВРЕМЕННО: по h градиент есть, по параметрам есть, по
    # черновику нет.
    torch.nn.init.normal_(head.net[-1].weight, std=0.5)
    torch.nn.init.normal_(head.net[-1].bias, std=0.5)
    head.zero_grad()
    h1 = torch.randn(3, N_POS, d_h, requires_grad=True)
    z1 = torch.randn(3, N_POS, d_z, requires_grad=True)
    head(h1, z1)[0].sum().backward()
    assert h1.grad is not None and float(h1.grad.abs().sum()) > 0, \
        "градиент по представлению не идёт — голова ничего не читает"
    assert float(head.net[-1].weight.grad.abs().sum()) > 0
    assert z1.grad is None or float(z1.grad.abs().sum()) == 0.0, \
        "stop-gradient не работает: поправка переучивает черновик"
    # Контроль самого теста: без detach градиент по черновику ОБЯЗАН появиться,
    # иначе проверка снова ничего не различала бы.
    head.zero_grad()
    z2 = torch.randn(3, N_POS, d_z, requires_grad=True)
    ((head.rho * torch.tanh(head.net(torch.cat([h1, head.proj(z2)], -1))))
     @ head.basis.T).sum().backward()
    assert z2.grad is not None and float(z2.grad.abs().sum()) > 0, \
        "тест на stop-gradient вырожден: без detach градиента тоже нет"

    # --- ГРАНИЦА АМПЛИТУДЫ НАСТОЯЩАЯ ----------------------------------------
    # При ортонормированном базисе ||dz|| = ||rho .* c|| <= ||rho||, и это
    # не зависит от того, чему обучились коэффициенты.
    torch.nn.init.normal_(head.net[-1].weight, std=50.0)
    torch.nn.init.normal_(head.net[-1].bias, std=50.0)
    rho = torch.full((rank,), 0.5)
    head.set_rho(rho)
    with torch.no_grad():
        dz2, c2 = head(h, z0)
        nrm = torch.linalg.norm(dz2, dim=-1).max()
        assert float(nrm) <= head.bound() + 1e-4, (float(nrm), head.bound())
        assert float(c2.abs().max()) <= 1.0, "tanh не ограничил коэффициенты"
    assert saturation_report(c2.numpy())["frac"] > 0.5, \
        saturation_report(c2.numpy())

    # --- rho ПРОВЕРЯЕТСЯ, А НЕ ПРИНИМАЕТСЯ ----------------------------------
    for bad in (torch.zeros(rank), torch.full((rank,), -1.0),
                torch.ones(rank + 1)):
        try:
            head.set_rho(bad)
            raise AssertionError("негодная rho принята")
        except ValueError:
            pass

    print("самопроверка hicora_vla пройдена (версия «замороженный базис»): доля насыщения, процентиль устойчив к выбросу, "
          "нулевая инициализация даёт строго нулевую поправку, голова живая "
          "при линейной потере, градиент по h есть а по черновику нет и тест "
          "это различает, базис ортонормирован и заморожен, ||dz|| <= ||rho|| "
          "при любых весах")


if __name__ == "__main__":
    selftest()
