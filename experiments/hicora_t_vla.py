#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HiCoRA-T: одна поправка целой траектории вместо шестнадцати независимых.

ЧЕМ ОТЛИЧАЕТСЯ ОТ HiCoRA. Там каждая из 16 латентных позиций получает свои
коэффициенты в общем базисе позиции: 16 * rank степеней свободы, и ничто не
связывает поправки соседних моментов чанка. Здесь предсказывается ОДИН вектор
u из 64 чисел на весь чанк, а базис живёт в пространстве целого остатка
(16 * 512 = 8192). Один коэффициент двигает всю траекторию согласованно.

    u = f(h24, Z0) in R^64,   c = rho * tanh(u),
    dZ = reshape_16x512(B_T^T c),   Z_T = Z0 + dZ.

ЧТО ЭТО МЕНЯЕТ ПО СУЩЕСТВУ. Предел амплитуды остаётся тем же по форме
(||dZ|| <= ||rho|| при ортонормированном базисе), но теперь он ограничивает
норму ВСЕЙ поправки чанка, а не каждой позиции по отдельности. И шум при RL
задаётся в 64 измерениях вместо 512: одна реализация шума выбирает
согласованное изменение траектории, а не шестнадцать независимых рывков.

ПОЧЕМУ НАСЛЕДОВАНИЕ ОТ КЛАССА HiCoRA. Путь получения черновика — отводы,
`q0_from`, `res_norm`, кодовые книги — должен совпадать дословно, иначе
сравнение HiCoRA и HiCoRA-T окажется сравнением двух разных черновиков.
Наследование гарантирует это конструкцией, а не сверкой кода.

ЕДИНСТВЕННЫЙ ПРОХОД. Черновик снимается с 12-го слоя того же прохода, что даёт
h24; обратно в слои 13..24 он не вставляется и второго прохода не запускает.
Уровни q1 и q2 не генерируются вовсе — их заменяет непрерывная поправка.
"""
import numpy as np

N_POS_DEFAULT, RANK_DEFAULT = 16, 64


def make_trajectory_head():
    """Класс головы траекторной поправки.

    Отдельной фабрикой по той же причине, что и в hicora_vla: вложенный класс
    нельзя ни создать в самопроверке, ни импортировать в обучающий скрипт.
    """
    import torch
    import torch.nn as nn

    class TrajectoryHead(nn.Module):
        """Один вектор коэффициентов на весь action chunk."""

        def __init__(self, d_hidden, d_latent, n_pos=N_POS_DEFAULT,
                     rank=RANK_DEFAULT, proj=64, hidden=512):
            super().__init__()
            self.d_hidden, self.d_latent = int(d_hidden), int(d_latent)
            self.n_pos, self.rank = int(n_pos), int(rank)
            self.d_flat = self.n_pos * self.d_latent
            self.proj_h = nn.Linear(self.d_hidden, int(proj))
            self.proj_z = nn.Linear(self.d_latent, int(proj))
            self.net = nn.Sequential(
                nn.LayerNorm(self.n_pos * 2 * int(proj)),
                nn.Linear(self.n_pos * 2 * int(proj), int(hidden)),
                nn.GELU(),
                nn.Linear(int(hidden), self.rank))
            # НУЛЕВАЯ ИНИЦИАЛИЗАЦИЯ ПОСЛЕДНЕГО СЛОЯ: до обучения поправка
            # строго нулевая, и HiCoRA-T в начальной точке совпадает с
            # fast12 (Joint12), а НЕ с coarse24: черновик снимается головой
            # Joint12 на 12-м слое, тогда как coarse24 — это авторегрессионный
            # generate на 24 слоя, из которого берётся нулевой уровень.
            # Тождество проверяется в пространстве действий в K-13c.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            # БАЗИС И ПРЕДЕЛ — БУФЕРЫ, А НЕ ПАРАМЕТРЫ. Обучаемый базис
            # обесценил бы rho: предел на коэффициенты обходится
            # масштабированием строк базиса.
            self.register_buffer("basis", torch.zeros(self.rank, self.d_flat))
            self.register_buffer("basis_set", torch.zeros(1))
            self.register_buffer("rho", torch.ones(self.rank))
            self.register_buffer("rho_set", torch.zeros(1))
            self._basis_ready = False
            self._rho_ready = False

        def set_basis(self, B):
            """Базис траекторного многообразия, построенный на train.

            ОРТОНОРМИРОВАННОСТЬ ПРОВЕРЯЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ: только при
            B B^T = I выполняется ||dZ|| = ||rho * c||, то есть предел
            амплитуды становится настоящим, а не декоративным.
            """
            b = torch.as_tensor(B, dtype=self.basis.dtype)
            if tuple(b.shape) != tuple(self.basis.shape):
                raise ValueError(f"базис формы {tuple(b.shape)}, ожидалась "
                                 f"{tuple(self.basis.shape)}")
            if not torch.isfinite(b).all():
                raise ValueError("в базисе есть nan или inf")
            g = b @ b.T
            off = float((g - torch.eye(self.rank, dtype=g.dtype)).abs().max())
            if off > 1e-4:
                raise ValueError(f"базис не ортонормирован: max|B B^T - I| = "
                                 f"{off:.2e}")
            self.basis.data = b.to(self.basis.device)
            self.basis_set.data = torch.ones_like(self.basis_set)
            self._basis_ready = True
            return self

        def set_rho(self, rho):
            r = torch.as_tensor(rho, dtype=self.rho.dtype)
            if tuple(r.shape) != tuple(self.rho.shape):
                raise ValueError(f"rho формы {tuple(r.shape)}, ожидалась "
                                 f"{tuple(self.rho.shape)}")
            if not torch.isfinite(r).all() or bool((r <= 0).any()):
                raise ValueError("rho обязана быть конечной и положительной")
            with torch.no_grad():
                self.rho.copy_(r.to(self.rho.device))
            self.rho_set.data = torch.ones_like(self.rho_set)
            self._rho_ready = True
            return self

        def bound(self):
            """Гарантированный предел ||dZ||_2 по всему чанку."""
            return float(torch.linalg.norm(self.rho))

        def check_ready(self):
            """Готовность ОДИН РАЗ, при сборке, а не в forward."""
            if not (self._basis_ready or bool(
                    self.basis_set.detach().cpu().item())):
                raise RuntimeError(
                    "базис не задан: сначала set_basis по обучающей части. "
                    "Нулевой базис дал бы тождественно нулевую поправку, и "
                    "обучение «сошлось» бы, ничего не выучив")
            if not (self._rho_ready or bool(
                    self.rho_set.detach().cpu().item())):
                raise RuntimeError(
                    "rho не задана: без set_rho предел равен единицам, то "
                    "есть ||dZ|| <= sqrt(rank) вместо измеренного на train "
                    "значения")
            self._basis_ready = self._rho_ready = True
            return self

        def features(self, h, z0):
            """Признаки чанка: обе проекции, конкатенация, развёртка.

            STOP-GRADIENT НА ЧЕРНОВИКЕ — как в HiCoRA: голова исправляет то,
            что модель предсказала сама, и не переучивает q0 через себя.
            """
            if h.shape[-2] != self.n_pos or z0.shape[-2] != self.n_pos:
                raise ValueError(
                    f"позиций {h.shape[-2]} и {z0.shape[-2]}, ожидалось "
                    f"{self.n_pos}: базис построен на чанк фиксированной длины")
            a = self.proj_h(h)
            b = self.proj_z(z0.detach())
            return torch.cat([a, b], dim=-1).flatten(-2)

        def mean_coeffs(self, h, z0):
            """Средние ДО tanh: это и есть mu для гауссовой версии."""
            return self.net(self.features(h, z0))

        def forward(self, h, z0):
            """Поправка целого чанка и её коэффициенты.

            Возвращает dZ формы (..., n_pos, d_latent) и c формы (..., rank).
            """
            if not (self._basis_ready and self._rho_ready):
                raise RuntimeError(
                    "не заданы базис или rho: нужны set_basis и set_rho. "
                    "После загрузки state_dict вызовите check_ready")
            c = torch.tanh(self.mean_coeffs(h, z0))
            dz = (self.rho * c) @ self.basis
            return dz.unflatten(-1, (self.n_pos, self.d_latent)), c

        def trainable_prefixes(self):
            """Что обучается. БАЗИС И rho СЮДА НЕ ВХОДЯТ."""
            return ("proj_h.", "proj_z.", "net.")

    return TrajectoryHead


def make_hicora_t_class(base_cls):
    """Модель HiCoRA-T поверх КЛАССА HiCoRA.

    Наследование, а не копия: отводы, `q0_from`, `res_norm` и кодовые книги
    должны совпадать с HiCoRA дословно, иначе сравнение двух архитектур
    окажется сравнением двух разных черновиков.
    """
    import torch  # noqa: F401
    from hicora_vla import make_hicora_class

    class _HiCoRATVLA(make_hicora_class(base_cls)):

        def init_hicora_t(self, q0_depth=12, rank=RANK_DEFAULT, proj=64,
                          hidden=512, n_pos=N_POS_DEFAULT, taps=(12, 18, 24)):
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
                raise RuntimeError("нет головы q0: сначала init_joint_fast")
            if not hasattr(self, "hicora_d_latent"):
                raise RuntimeError("сначала set_codebooks")
            d_hidden = int(self.fast_head.in_features)
            self.hicora_t_head = make_trajectory_head()(
                d_hidden, self.hicora_d_latent, n_pos=n_pos, rank=rank,
                proj=proj, hidden=hidden)
            self.hicora_t_head.to(next(self.parameters()).device)
            return self

        def forward_hicora_t(self, *, vlm_inputs_embeds, attention_mask,
                             position_ids):
            """Один проход VLA, один черновик, одна поправка всей траектории.

            Слой 18 не используется: он остаётся возможным местом для гейта
            «нужна ли коррекция», но в v0 его нет.
            """
            taps = self.forward_taps(vlm_inputs_embeds=vlm_inputs_embeds,
                                     attention_mask=attention_mask,
                                     position_ids=position_ids)
            logits, q0 = self.q0_from(taps[self.q0_depth])
            z0 = self.codebooks[0][q0]
            if not hasattr(self, "res_norm"):
                raise RuntimeError(
                    "не задана res_norm: поздняя ветвь читала бы h24 нормой "
                    "Joint12, обученной на h12")
            h24 = self.res_norm(taps[max(self.taps)]).float()
            dz, c = self.hicora_t_head(h24, z0)
            return dict(q0=q0, q0_logits=logits, z0=z0, dz=dz, z=z0 + dz,
                        coeffs=c, layers_run=taps["layers_run"])

        def hicora_t_trainable_prefixes(self):
            return tuple("hicora_t_head." + p
                         for p in self.hicora_t_head.trainable_prefixes())

        def configure_hicora_t(self, verbose=True):
            """Заморозить всё, кроме траекторной головы.

            Возвращать список префиксов недостаточно: `init_joint_fast`
            оставляет обучаемыми первые двенадцать слоёв, норму q0 и голову
            q0, а `res_norm` копируется после него и наследует
            requires_grad=True. Без явной заморозки обучался бы ствол.
            """
            pref = self.hicora_t_trainable_prefixes()
            for p_ in self.parameters():
                p_.requires_grad_(False)
            n_tr = n_par = 0
            for name, p_ in self.named_parameters():
                if any(name.startswith(x) for x in pref):
                    p_.requires_grad_(True)
                    n_tr += 1
                    n_par += p_.numel()
            if n_tr == 0:
                raise RuntimeError(
                    f"ни один параметр не подошёл под {pref}: голова не "
                    f"создана или названа иначе, и обучение шло бы вхолостую")
            if verbose:
                print(f"  обучаемых тензоров {n_tr}, параметров {n_par}")
            return n_par

    return _HiCoRATVLA


# ------------------------------ самопроверка ------------------------------

def selftest():
    import torch

    D_H, D_L, NP_, RK = 24, 32, 6, 8
    torch.manual_seed(0)
    Cls = make_trajectory_head()

    def basis(rank, d):
        q, _ = torch.linalg.qr(torch.randn(d, rank, dtype=torch.float64))
        return q.T.float()

    B = basis(RK, NP_ * D_L)
    RHO = torch.tensor([0.7, 1.3, 0.4, 2.1, 0.9, 1.1, 0.5, 1.7])
    head = Cls(D_H, D_L, n_pos=NP_, rank=RK, proj=5, hidden=16)
    head.set_basis(B).set_rho(RHO)
    head.eval()
    h = torch.randn(4, NP_, D_H)
    z = torch.randn(4, NP_, D_L)

    # --- 1. ФОРМЫ ---------------------------------------------------------
    with torch.no_grad():
        dz, c = head(h, z)
    assert dz.shape == (4, NP_, D_L), dz.shape
    assert c.shape == (4, RK), c.shape
    assert head.features(h, z).shape == (4, NP_ * 2 * 5)

    # --- 2. ТОЖДЕСТВО ПРИ НУЛЕВОЙ ИНИЦИАЛИЗАЦИИ ---------------------------
    # Последний слой нулевой -> коэффициенты нулевые -> поправка ТОЧНО нулевая,
    # то есть Z_T = Z_0 и действия совпадают с fast12 бит в бит. Проверка в
    # пространстве действий — в K-13c: здесь декодера нет.
    assert float(c.abs().max()) == 0.0, "коэффициенты не нулевые"
    assert float(dz.abs().max()) == 0.0, "поправка не нулевая"

    # --- 3. ОДИН КОЭФФИЦИЕНТ ДВИГАЕТ ВСЕ ПОЗИЦИИ --------------------------
    # В этом и состоит архитектурная разница с HiCoRA: там коэффициент влияет
    # на СВОЮ позицию, здесь — на весь чанк. Если бы базис был блочным по
    # позициям, тронулась бы одна позиция.
    u = torch.zeros(1, RK)
    u[0, 0] = 2.0
    c1 = RHO * torch.tanh(u)
    dz1 = (c1 @ head.basis).unflatten(-1, (NP_, D_L))
    per_pos = dz1[0].abs().max(dim=-1).values
    assert float(per_pos.min()) > 0, \
        f"позиции {torch.flatnonzero(per_pos == 0).tolist()} не затронуты"

    # --- 4. ПРЕДЕЛ АМПЛИТУДЫ ДЕРЖИТСЯ НА ВСЕЙ ТРАЕКТОРИИ ------------------
    lim = head.bound()
    assert abs(lim - float(torch.linalg.norm(RHO))) < 1e-6
    worst = 0.0
    with torch.no_grad():
        for _ in range(50):
            uu = torch.randn(32, RK) * 5.0        # почти насыщение tanh
            cc = RHO * torch.tanh(uu)
            dd = (cc @ head.basis)
            worst = max(worst, float(torch.linalg.norm(dd, dim=-1).max()))
    assert worst <= lim + 1e-4, f"||dZ|| = {worst:.4f} превысила {lim:.4f}"

    # --- 5. ОТКАЗЫ --------------------------------------------------------
    bad = Cls(D_H, D_L, n_pos=NP_, rank=RK, proj=5, hidden=16)
    for fn, needle in ((lambda: bad(h, z), "не заданы базис или rho"),
                       (lambda: bad.set_basis(torch.randn(RK, NP_ * D_L)),
                        "не ортонормирован"),
                       (lambda: bad.set_basis(B[:, :3]), "формы"),
                       (lambda: bad.set_rho(torch.zeros(RK)),
                        "положительной")):
        try:
            fn()
        except (RuntimeError, ValueError) as e:
            assert needle in str(e), f"ожидал «{needle}», получил: {e}"
        else:
            raise AssertionError(f"отказа «{needle}» не было")
    try:
        head(torch.randn(2, NP_ + 1, D_H), torch.randn(2, NP_ + 1, D_L))
    except ValueError as e:
        assert "позиций" in str(e), e
    else:
        raise AssertionError("чужая длина чанка принята")

    # --- 6. ИЗОЛЯЦИЯ ГРАДИЕНТОВ -------------------------------------------
    h2 = Cls(D_H, D_L, n_pos=NP_, rank=RK, proj=5, hidden=16)
    h2.set_basis(B).set_rho(RHO)
    with torch.no_grad():                 # снять нулевую инициализацию
        h2.net[-1].weight.normal_(0, 0.1)
        h2.net[-1].bias.normal_(0, 0.1)
    zg = z.clone().requires_grad_(True)
    dzz, _cc = h2(h, zg)
    dzz.sum().backward()
    for nm, p_ in h2.named_parameters():
        assert p_.grad is not None and torch.isfinite(p_.grad).all(), nm
    assert h2.basis.grad is None and h2.rho.grad is None, "базис или rho учатся"
    assert zg.grad is None or float(zg.grad.abs().max()) == 0.0, \
        "градиент прошёл в черновик: stop-gradient не работает"

    # --- 7. МИНИ-ПЕРЕОБУЧЕНИЕ: голова способна выучить цель ---------------
    tgt = torch.randn(8, RK) * 0.3
    hh, zz = torch.randn(8, NP_, D_H), torch.randn(8, NP_, D_L)
    h3 = Cls(D_H, D_L, n_pos=NP_, rank=RK, proj=5, hidden=32)
    h3.set_basis(B).set_rho(RHO)
    opt = torch.optim.Adam(h3.parameters(), lr=0.02)
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = ((torch.tanh(h3.mean_coeffs(hh, zz)) - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
    assert float(loss) < 0.25 * first, (first, float(loss))

    # --- 8. ЧИСЛО СТЕПЕНЕЙ СВОБОДЫ ПРОТИВ HiCoRA --------------------------
    # У HiCoRA это n_pos * rank коэффициентов на чанк, здесь rank
    assert c.shape[-1] == RK and c.dim() == 2, c.shape
    print(f"самопроверка hicora_t_vla пройдена: {RK} коэффициентов на чанк "
          f"против {NP_ * RK} у HiCoRA при том же ранге на позицию")


if __name__ == "__main__":
    selftest()
