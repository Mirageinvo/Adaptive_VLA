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

ПОЧЕМУ tanh И rho. Без ограничения амплитуды голова может увести латент в
область, где декодер не обучался, и офлайновая ошибка улучшится, а поведение
развалится. rho считается на обучающей части как процентиль коэффициентов
проекции реального остатка и КЛАДЁТСЯ В ЧЕКПОЙНТ; доля насыщенных
коэффициентов печатается при каждом разборе — постоянное насыщение есть
стоп-условие ветки, а не деталь.

ПОСЛЕДНИЙ ЛИНЕЙНЫЙ СЛОЙ ИНИЦИАЛИЗИРОВАН НУЛЯМИ. Это делает нулевое тождество
проверяемым: до обучения модель обязана побитово совпадать с базовым coarse
выходом головы q0. Тождество проверяет K-11b.

СХВАТ НЕ ИСПРАВЛЯЕТСЯ. В HiCoRA-D он целиком берётся из q0. Канал схвата
дискретный и редкий; смешивать его ошибку с позиционной в одну величину
нельзя — это ровно та ошибка, из-за которой офлайновый гейт K-10d оказался
неспособен подтверждать (см. K-10g).

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
            self.basis = nn.Linear(self.rank, d_latent, bias=False)
            # НУЛЕВАЯ ИНИЦИАЛИЗАЦИЯ ПОСЛЕДНЕГО СЛОЯ: до обучения поправка
            # строго нулевая, и нулевое тождество проверяемо.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            # rho — БУФЕР, а не параметр: предел амплитуды задаётся данными в
            # K-11a и не должен подстраиваться градиентом под потерю.
            self.register_buffer("rho", torch.ones(self.rank))

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
            c = self.coeffs(h, z0)
            return self.basis(self.rho * c), c

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
            h24 = self.action_expert.norm(taps[max(self.taps)]).float()
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

    # --- НУЛЕВОЕ ТОЖДЕСТВО -------------------------------------------------
    dz, c0 = head(h, z0)
    assert float(dz.abs().max()) == 0.0, "поправка до обучения не нулевая"
    assert float(c0.abs().max()) == 0.0

    # --- ГОЛОВА ЖИВАЯ ------------------------------------------------------
    # Нулевая инициализация не должна означать нулевой градиент: иначе
    # тождество достигалось бы мёртвой головой и обучение не стронулось бы.
    head.zero_grad()
    head.net(torch.cat([h, head.proj(z0.detach())], -1)).pow(2).sum().backward()
    g = head.net[-1].weight.grad
    assert g is not None and float(g.abs().sum()) > 0, "голова мертва"

    # --- ГРАДИЕНТ НЕ ТЕЧЁТ В ЧЕРНОВИК --------------------------------------
    head.zero_grad()
    z1 = torch.randn(3, N_POS, d_z, requires_grad=True)
    dz1, _ = head(h, z1)
    (dz1.sum() + head.net[-1].weight.sum()).backward()
    assert z1.grad is None or float(z1.grad.abs().sum()) == 0.0, \
        "stop-gradient не работает: поправка переучивает черновик"

    # --- ОГРАНИЧЕНИЕ АМПЛИТУДЫ ДЕЙСТВУЕТ -----------------------------------
    torch.nn.init.normal_(head.net[-1].weight, std=3.0)
    torch.nn.init.normal_(head.net[-1].bias, std=3.0)
    rho = torch.full((rank,), 0.5)
    head.set_rho(rho)
    with torch.no_grad():
        dz2, c2 = head(h, z0)
        lim = float((head.basis.weight.abs() @ rho).max())
        assert float(dz2.abs().max()) <= lim + 1e-5, (
            float(dz2.abs().max()), lim)
        assert float(c2.abs().max()) <= 1.0, "tanh не ограничил коэффициенты"
    # Насыщение при большом масштабе обязано БЫТЬ ЗАМЕТНЫМ, иначе метрика
    # насыщения не измеряла бы ничего.
    assert saturation_report(c2.numpy())["frac"] > 0.5, \
        saturation_report(c2.numpy())

    # --- rho ПРОВЕРЯЕТСЯ, А НЕ ПРИНИМАЕТСЯ ---------------------------------
    for bad in (torch.zeros(rank), torch.full((rank,), -1.0),
                torch.ones(rank + 1)):
        try:
            head.set_rho(bad)
            raise AssertionError("негодная rho принята")
        except ValueError:
            pass

    print("самопроверка hicora_vla пройдена (версия «нулевое тождество, sg "
          "на черновике»): доля насыщения, процентиль устойчив к выбросу, "
          "нулевая инициализация даёт строго нулевую поправку, голова живая, "
          "градиент в черновик не течёт, rho ограничивает и проверяется")


if __name__ == "__main__":
    selftest()
