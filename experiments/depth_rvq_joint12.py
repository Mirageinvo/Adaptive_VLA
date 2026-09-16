"""Depth-aligned RVQ V2 поверх уже обученного Joint12.

K-8 уже реализовал правильную общую топологию::

    h12 -> q0 -> feedback -> h18 -> q1 -> feedback -> h24 -> q2

но проверял её до появления хорошего Joint12: ранняя голова одновременно
переучивалась с поздними целями, а все выходы читались одной нормой. Этот
модуль сохраняет работающую политику Joint12 как фиксированный черновик и
обучает только уточнение:

* q0 и слои 1..12 заморожены;
* h18 и h24 имеют отдельные нормы и головы;
* выбранные q0/q1 возвращаются в поток действий;
* q1/q2 можно размечать относительно ФАКТИЧЕСКИ предсказанного prefix, а не
  относительно истинных предыдущих RVQ-кодов;
* один проход backbone и один последующий decode остаются проверяемыми.

Старые ``depth_rvq_vla.py`` и K-8 не меняются: они являются воспроизводимым
отрицательным контролем. Здесь новая ветка с другим экспериментальным
вопросом — способны ли поздние слои улучшить уже пригодный q0 Joint12.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # запуск и как ``python experiments/...``, и как пакетный импорт
    from .depth_rvq_vla import CodeFeedback, straight_through
except ImportError:
    from depth_rvq_vla import CodeFeedback, straight_through


DEFAULT_EXITS = (12, 18, 24)


def _finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name}: есть nan или inf")


def _project_fp32(module: nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
    """Применить Linear/Identity в fp32, не меняя dtype весов на месте.

    ``VectorQuantize.forward`` сам переводит квантователь в fp32. Здесь
    мутировать переданный codec нельзя: conditional relabeling не должен
    менять исполняемую модель. У ActionCodec проекции имеют ровно два
    допустимых типа; неизвестный тип лучше отвергнуть, чем получить другую
    метрику расстояния молча.
    """
    if isinstance(module, nn.Identity):
        return x.float()
    if isinstance(module, nn.Linear):
        bias = None if module.bias is None else module.bias.float()
        return F.linear(x.float(), module.weight.float(), bias)
    raise TypeError(f"{name}: ожидался Linear или Identity, дано {type(module).__name__}")


def code_contribution(quantizer: nn.Module, codes: torch.Tensor) -> torch.Tensor:
    """Вклад одного RVQ-уровня в общем 512-мерном latent space.

    Используется ровно тот путь, что ``ResidualVectorQuantize.from_codes``:
    lookup во внутренней книге, затем ``out_project``. Нельзя считать, что
    ``quantizer.codebook`` уже живёт в общем latent space: при
    ``codebook_dim != input_dim`` это неверно.
    """
    if codes.dtype != torch.long:
        codes = codes.long()
    if codes.ndim != 2:
        raise ValueError(f"codes должны иметь форму (B,P), дано {tuple(codes.shape)}")
    z = quantizer.decode_code(codes).float()
    return _project_fp32(quantizer.out_project, z, "out_project")


@torch.no_grad()
def nearest_code(
    residual: torch.Tensor,
    quantizer: nn.Module,
    *,
    chunk_rows: int = 4096,
) -> torch.Tensor:
    """Точный argmin данного ActionCodec-квантователя без EMA-обновления.

    Расстояние считается ПОСЛЕ ``in_project`` — именно так работает
    ``actioncodec.rvq.VectorQuantize.forward``. Поиск разбит по строкам,
    чтобы не создавать матрицу ``(B*P, vocab)`` произвольного размера.
    """
    if residual.ndim != 3:
        raise ValueError(
            f"residual должен иметь форму (B,P,D), дано {tuple(residual.shape)}")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows должен быть положительным, дано {chunk_rows}")
    _finite(residual, "residual")
    enc = _project_fp32(quantizer.in_project, residual, "in_project")
    book = quantizer.codebook.float().to(enc.device)
    if enc.shape[-1] != book.shape[-1]:
        raise ValueError(
            f"in_project дал {enc.shape[-1]}, книга имеет {book.shape[-1]}")
    _finite(enc, "projected residual")
    _finite(book, "codebook")
    flat = enc.reshape(-1, enc.shape[-1])
    b2 = book.square().sum(-1).unsqueeze(0)
    ans = []
    for i in range(0, flat.shape[0], chunk_rows):
        x = flat[i:i + chunk_rows]
        dist = x.square().sum(-1, keepdim=True) - 2.0 * x @ book.T + b2
        ans.append(dist.argmin(-1))
    return torch.cat(ans).reshape(enc.shape[:-1])


@torch.no_grad()
def conditional_target(
    target_latent: torch.Tensor,
    prefix_codes: Sequence[torch.Tensor],
    quantizers: Sequence[nn.Module],
    level: int,
    *,
    chunk_rows: int = 4096,
) -> torch.Tensor:
    """Цель уровня ``level`` после реально предсказанного prefix.

    ``target_latent`` лучше брать из ``codec._encode(actions)``. Сумма
    истинных code contributions допустима как более дешёвая аппроксимация,
    но тогда модель исправляет ошибку относительно квантованного, а не
    исходного encoder latent.

    Для q2 передаются ФАКТИЧЕСКИЕ предсказания q0 и q1. Подставлять целевой q1
    нельзя: это вернуло бы teacher-forcing mismatch, который функция создана
    устранить.
    """
    if target_latent.ndim != 3:
        raise ValueError(
            f"target_latent должен иметь форму (B,P,D), дано "
            f"{tuple(target_latent.shape)}")
    if not 0 <= level < len(quantizers):
        raise ValueError(f"уровень {level} вне 0..{len(quantizers) - 1}")
    if len(prefix_codes) != level:
        raise ValueError(
            f"для уровня {level} нужен prefix длины {level}, дано "
            f"{len(prefix_codes)}")
    residual = target_latent.float().clone()
    for g, codes in enumerate(prefix_codes):
        if tuple(codes.shape) != tuple(target_latent.shape[:2]):
            raise ValueError(
                f"prefix[{g}] формы {tuple(codes.shape)}, ожидалась "
                f"{tuple(target_latent.shape[:2])}")
        residual.sub_(code_contribution(quantizers[g], codes).to(residual.device))
    return nearest_code(residual, quantizers[level], chunk_rows=chunk_rows)


@torch.no_grad()
def conditional_targets(
    target_latent: torch.Tensor,
    predicted_codes: Sequence[torch.Tensor],
    quantizers: Sequence[nn.Module],
    *,
    first_level: int = 1,
    chunk_rows: int = 4096,
) -> list[torch.Tensor]:
    """Все поздние цели, каждая относительно собственного predicted prefix."""
    if len(predicted_codes) < len(quantizers) - 1:
        raise ValueError(
            f"предсказано {len(predicted_codes)} уровней, для целей до "
            f"{len(quantizers) - 1} нужно хотя бы {len(quantizers) - 1}")
    if not 0 <= first_level < len(quantizers):
        raise ValueError(f"first_level {first_level} вне диапазона")
    return [
        conditional_target(
            target_latent, predicted_codes[:g], quantizers, g,
            chunk_rows=chunk_rows)
        for g in range(first_level, len(quantizers))
    ]


@torch.no_grad()
def latent_from_codes(
    codes: torch.Tensor,
    quantizers: Sequence[nn.Module],
) -> torch.Tensor:
    """Восстановить общий latent из кодов формы ``(B,L,P)``."""
    if codes.ndim != 3:
        raise ValueError(f"codes должны иметь форму (B,L,P), дано {tuple(codes.shape)}")
    if codes.shape[1] != len(quantizers):
        raise ValueError(
            f"в codes {codes.shape[1]} уровней, квантователей {len(quantizers)}")
    z = None
    for g, q in enumerate(quantizers):
        part = code_contribution(q, codes[:, g, :])
        z = part if z is None else z + part
    assert z is not None
    return z


def make_joint_depth_rvq_class(base_cls):
    """Добавить глубинное уточнение к уже собранному классу Joint12."""

    class _JointDepthRVQ(base_cls):

        def init_joint_depth_rvq(
            self,
            *,
            refine_norm: nn.Module,
            books: torch.Tensor,
            exits: Iterable[int] = DEFAULT_EXITS,
            head_dtype: torch.dtype = torch.float32,
            feedback: bool = True,
        ):
            exits = tuple(int(x) for x in exits)
            n_layers = len(self.action_expert.layers)
            if exits != tuple(sorted(set(exits))):
                raise ValueError(f"выходы должны строго возрастать: {exits}")
            if len(exits) != 3:
                raise ValueError(f"V2 сейчас требует ровно q0/q1/q2, дано {exits}")
            if exits[-1] != n_layers:
                raise ValueError(
                    f"последний выход {exits[-1]} против {n_layers} слоёв")
            if not hasattr(self, "fast_head"):
                raise RuntimeError("сначала соберите и загрузите Joint12 fast_head")
            if hasattr(self, "depth_rvq_books"):
                raise RuntimeError(
                    "depth-RVQ V2 уже инициализирован: повторная сборка "
                    "могла бы смешать нормы и головы двух конфигураций")
            if int(getattr(self, "fast_depth", -1)) != exits[0]:
                raise RuntimeError(
                    f"Joint12 глубины {getattr(self, 'fast_depth', None)}, "
                    f"первый выход задан на {exits[0]}")

            b = torch.as_tensor(books).float()
            if b.ndim != 3 or b.shape[0] != len(exits):
                raise ValueError(
                    f"books должны иметь форму (3,V,D), дано {tuple(b.shape)}")
            if b.shape[1] != self.fast_head.out_features:
                raise ValueError(
                    f"словарь {b.shape[1]} против fast_head "
                    f"{self.fast_head.out_features}")
            _finite(b, "books")

            # Joint12 — уже проверенная политика. Замораживаем её и весь
            # backbone ДО создания новых модулей, чтобы whitelist был точным.
            for p in self.parameters():
                p.requires_grad_(False)

            dev = self.fast_head.weight.device
            d_model = int(self.fast_head.in_features)
            vocab = int(self.fast_head.out_features)
            self.depth_rvq_exits = exits
            self.depth_rvq_use_feedback = bool(feedback)
            self.register_buffer("depth_rvq_books", b.to(dev))

            # Норма Joint12 обучена читать h12. Её применение к h18/h24 было
            # бы тем же precision/provenance классом ошибки, который уже
            # ловился в HiCoRA. Поздним выходам даём отдельные копии исходной
            # финальной нормы; дальше они обучаются вместе с головами.
            self.depth_rvq_norms = nn.ModuleList([
                copy.deepcopy(refine_norm).to(device=dev, dtype=head_dtype)
                for _ in range(2)
            ])
            self.depth_rvq_heads = nn.ModuleList()
            for _ in range(2):
                h = nn.Linear(
                    d_model, vocab,
                    bias=self.action_lm_head.bias is not None,
                    device=dev, dtype=head_dtype)
                with torch.no_grad():
                    h.weight.copy_(self.action_lm_head.weight.to(dev, head_dtype))
                    if h.bias is not None:
                        h.bias.copy_(self.action_lm_head.bias.to(dev, head_dtype))
                self.depth_rvq_heads.append(h)
            self.depth_rvq_feedback = nn.ModuleList([
                CodeFeedback(
                    int(b.shape[-1]), d_model, alpha_init=1.0,
                    dtype=head_dtype).to(dev)
                for _ in range(2)
            ])
            self.configure_joint_depth_rvq(verbose=False)
            return self

        def joint_depth_rvq_trainable_prefixes(self) -> tuple[str, ...]:
            return (
                "depth_rvq_norms.",
                "depth_rvq_heads.",
                "depth_rvq_feedback.",
            )

        def configure_joint_depth_rvq(self, *, verbose: bool = True) -> int:
            """Заморозить всё кроме поздних норм, голов и feedback."""
            pref = self.joint_depth_rvq_trainable_prefixes()
            for p in self.parameters():
                p.requires_grad_(False)
            for name, p in self.named_parameters():
                if any(name.startswith(x) for x in pref):
                    p.requires_grad_(True)
            trainable = {n: p for n, p in self.named_parameters() if p.requires_grad}
            stray = [n for n in trainable if not any(n.startswith(x) for x in pref)]
            if stray:
                raise RuntimeError(f"обучаемое вне whitelist: {stray[:5]}")
            missing_groups = [x for x in pref if not any(n.startswith(x) for n in trainable)]
            if missing_groups:
                raise RuntimeError(f"пустые группы обучаемых весов: {missing_groups}")
            if isinstance(self.depth_rvq_books, nn.Parameter):
                raise RuntimeError("кодовые книги стали параметром")
            n = sum(p.numel() for p in trainable.values())
            if verbose:
                print(
                    f"  depth-RVQ V2: {len(trainable)} обучаемых тензоров, "
                    f"{n / 1e6:.3f} млн параметров; Joint12 и backbone "
                    f"заморожены")
            return n

        def _joint_depth_logits(self, action_hidden: torch.Tensor, level: int):
            if level == 0:
                normed = self.action_expert.norm(action_hidden)
                head = self.fast_head
            else:
                norm = self.depth_rvq_norms[level - 1]
                head = self.depth_rvq_heads[level - 1]
                # Явный dtype: поздние головы должны работать в fp32 и не
                # зависеть от внешней границы autocast.
                w = next(norm.parameters(), None)
                if w is None:
                    raise RuntimeError("поздняя норма не имеет параметров")
                normed = norm(action_hidden.to(w.dtype))
            return head(normed.to(head.weight.dtype))

        def forward_joint_depth_rvq(
            self,
            *,
            vlm_inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            mode: str = "full",
            teacher_codes: torch.Tensor | None = None,
            teacher_levels: Sequence[int] = (),
            tau: float = 1.0,
        ):
            """Один segmented forward с отдельными predicted/injected codes.

            ``teacher_codes`` имеет форму ``(B,3,P)``. Учитель используется
            только на явно перечисленных ``teacher_levels``; по умолчанию его
            нет. Для основного V2 q0 всегда собственный, потому что именно его
            ошибки поздние уровни должны научиться исправлять.
            """
            stop = {"fast": 0, "medium": 1, "full": 2}
            if mode not in stop:
                raise ValueError(f"неизвестный mode {mode}")
            stop_level = stop[mode]
            teach = set(int(x) for x in teacher_levels)
            if any(x not in (0, 1) for x in teach):
                raise ValueError(f"teacher применяется только перед продолжением: {teach}")

            B = vlm_inputs_embeds.shape[0]
            dev, dt = vlm_inputs_embeds.device, vlm_inputs_embeds.dtype
            n = self.block_size
            bos = self.bos_embedding.expand(B, n, -1).to(dev, dt)
            empty = torch.empty((B, 0, bos.shape[-1]), device=dev, dtype=dt)
            action_hidden = torch.cat([bos, empty], dim=1)
            vlm_hidden = vlm_inputs_embeds
            mask4d = self._build_joint_attention_mask_blockwise_ar(
                attention_mask=attention_mask,
                vlm_seq_len=vlm_inputs_embeds.shape[1],
                action_seq_len=n,
                device=dev,
                action_key_mask=torch.ones((B, n), device=dev, dtype=torch.long))

            if teacher_codes is not None:
                want = (B, 3, n)
                if tuple(teacher_codes.shape) != want:
                    raise ValueError(
                        f"teacher_codes формы {tuple(teacher_codes.shape)}, "
                        f"ожидалась {want}")

            logits, pred, injected, embs = [], [], [], []
            layers_run = 0
            for layer_idx in range(len(self.action_expert.layers)):
                vlm_hidden, action_hidden = self._shared_attention_forward(
                    vlm_hidden_states=vlm_hidden,
                    action_hidden_states=action_hidden,
                    layer_idx=layer_idx,
                    attention_mask=mask4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=None)
                layers_run += 1
                depth = layer_idx + 1
                if depth not in self.depth_rvq_exits:
                    continue
                g = self.depth_rvq_exits.index(depth)
                lg = self._joint_depth_logits(action_hidden, g)
                idx = lg.argmax(-1)
                logits.append(lg)
                pred.append(idx)
                if g >= stop_level:
                    break

                # q0 фиксирован и внедряется hard. Для q1 сохраняем ST-путь,
                # чтобы ошибка q2/action могла обучать среднюю голову.
                if g == 0:
                    emb = self.depth_rvq_books[g][idx]
                else:
                    emb, _, _ = straight_through(
                        lg.float(), self.depth_rvq_books[g], tau=tau)
                inj_idx = idx
                if g in teach:
                    if teacher_codes is None:
                        raise ValueError(
                            f"запрошен teacher уровня {g}, но teacher_codes нет")
                    inj_idx = teacher_codes[:, g, :].to(dev).long()
                    hard = self.depth_rvq_books[g][inj_idx].to(emb.dtype)
                    # Для q1 сохраняем soft backward, меняем только forward.
                    emb = hard if g == 0 else emb + (hard - emb).detach()
                injected.append(inj_idx)
                embs.append(emb)
                if self.depth_rvq_use_feedback:
                    action_hidden = self.depth_rvq_feedback[g](action_hidden, emb)

            if layers_run != self.depth_rvq_exits[stop_level]:
                raise RuntimeError(
                    f"mode={mode}: исполнено {layers_run} слоёв, ожидалось "
                    f"{self.depth_rvq_exits[stop_level]}")
            return dict(
                logits=logits,
                pred_codes=pred,
                injected_codes=injected,
                embeddings=embs,
                layers_run=layers_run,
            )

    return _JointDepthRVQ


class _FakeQuantizer(nn.Module):
    """Минимальный точный двойник интерфейса ActionCodec для selftest."""

    def __init__(self, book: torch.Tensor):
        super().__init__()
        self.in_project = nn.Identity()
        self.out_project = nn.Identity()
        self.register_buffer("codebook", book.float())

    def decode_code(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.codebook)


def selftest() -> None:
    torch.manual_seed(0)

    # --- conditional targets действительно исправляют predicted prefix ---
    books_1d = [
        torch.tensor([[0.0], [10.0], [20.0]]),
        torch.tensor([[-10.0], [0.0], [10.0]]),
        torch.tensor([[-10.0], [0.0], [10.0]]),
    ]
    qs = [_FakeQuantizer(x) for x in books_1d]
    true = torch.tensor([[[0], [1], [1]]])       # target latent = 0
    z = latent_from_codes(true, qs)
    assert torch.equal(z, torch.zeros_like(z))
    pred = [torch.tensor([[1]]), torch.tensor([[1]]), torch.tensor([[1]])]
    dyn = conditional_targets(z, pred, qs)
    # Ошибочный q0=+10 требует q1=-10 вместо статической истинной метки 0.
    assert int(dyn[0]) == 0 and int(true[0, 1, 0]) == 1
    # q2 тоже видит фактически предсказанный q1=0, а не динамическую цель q1.
    assert int(dyn[1]) == 0
    e_static = (z - code_contribution(qs[0], pred[0])
                - code_contribution(qs[1], true[:, 1, :])).abs().max()
    e_dynamic = (z - code_contribution(qs[0], pred[0])
                 - code_contribution(qs[1], dyn[0])).abs().max()
    assert float(e_dynamic) == 0.0 and float(e_static) == 10.0

    # При правильном prefix условная разметка воспроизводит исходный RVQ.
    true_prefix = [true[:, 0, :], true[:, 1, :], true[:, 2, :]]
    same = conditional_targets(z, true_prefix, qs)
    assert torch.equal(same[0], true[:, 1, :])
    assert torch.equal(same[1], true[:, 2, :])

    # in_project нельзя игнорировать: меняем метрику так, чтобы argmin
    # отличался от поиска в общем latent space.
    q_proj = _FakeQuantizer(torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
    q_proj.in_project = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        q_proj.in_project.weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 1.0]]))
    # После проекции обе точки книги равноудалены и argmin обязан быть 0;
    # наивный поиск в исходном space выбрал бы 1 для residual [2,0].
    got = nearest_code(torch.tensor([[[2.0, 0.0]]]), q_proj)
    assert int(got) == 0

    # Fail-closed формы и NaN.
    for bad in (torch.tensor([1.0]), torch.full((1, 1, 1), float("nan"))):
        try:
            nearest_code(bad, qs[0])
            raise AssertionError("неверный residual принят")
        except ValueError:
            pass

    # --- один проход и точное сохранение Joint12 ---------------------------
    D, V, Z, P, L = 8, 7, 4, 3, 24

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(D, D, bias=False) for _ in range(L)
            ])
            self.norm = nn.LayerNorm(D)

    class FakeBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_expert = FakeExpert()
            self.action_lm_head = nn.Linear(D, V)
            self.fast_head = nn.Linear(D, V)
            self.fast_depth = 12
            self.block_size = P
            self.bos_embedding = nn.Parameter(torch.randn(1, P, D))
            self.unrelated = nn.Parameter(torch.randn(2))
            self.config = SimpleNamespace(
                vlm_config=SimpleNamespace(
                    text_config=SimpleNamespace(num_hidden_layers=L)))

        def _build_joint_attention_mask_blockwise_ar(self, **_):
            return None

        def _shared_attention_forward(
            self, *, vlm_hidden_states, action_hidden_states, layer_idx, **_
        ):
            # Небольшой residual, чтобы все сегменты реально влияли на выход.
            ah = action_hidden_states + 0.05 * torch.tanh(
                self.action_expert.layers[layer_idx](action_hidden_states))
            return vlm_hidden_states, ah

    M = make_joint_depth_rvq_class(FakeBase)
    m = M()
    late_norm = copy.deepcopy(m.action_expert.norm)
    books = torch.randn(3, V, Z)
    m.init_joint_depth_rvq(refine_norm=late_norm, books=books)
    x = torch.randn(2, 5, D)

    # Эталон fast вручную: та же непрерывная раскладка bos и те же 12 слоёв.
    ah = torch.cat([
        m.bos_embedding.expand(2, P, -1),
        torch.empty(2, 0, D),
    ], dim=1)
    vh = x
    for li in range(12):
        vh, ah = m._shared_attention_forward(
            vlm_hidden_states=vh, action_hidden_states=ah, layer_idx=li)
    ref = m.fast_head(m.action_expert.norm(ah).to(m.fast_head.weight.dtype))
    out_fast = m.forward_joint_depth_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="fast")
    assert out_fast["layers_run"] == 12
    assert torch.equal(out_fast["logits"][0], ref)

    out0 = m.forward_joint_depth_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    assert out0["layers_run"] == 24 and len(out0["logits"]) == 3
    q0_before = out0["pred_codes"][0].clone()
    # Нулевая feedback-проекция обязана быть точным тождеством.
    m.depth_rvq_use_feedback = False
    nofb = m.forward_joint_depth_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    for a, b in zip(out0["logits"], nofb["logits"]):
        assert torch.equal(a, b)

    # После ненулевого feedback меняются только поздние уровни, q0 остаётся.
    m.depth_rvq_use_feedback = True
    for fb in m.depth_rvq_feedback:
        nn.init.normal_(fb.proj.weight, std=0.1)
        nn.init.normal_(fb.proj.bias, std=0.1)
    moved = m.forward_joint_depth_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    assert torch.equal(moved["pred_codes"][0], q0_before)
    assert not torch.equal(moved["logits"][1], nofb["logits"][1])

    # Whitelist точный: q0, backbone, bos и посторонний параметр заморожены.
    n = m.configure_joint_depth_rvq(verbose=False)
    tr = {name for name, p in m.named_parameters() if p.requires_grad}
    assert n > 0 and tr
    assert all(name.startswith(m.joint_depth_rvq_trainable_prefixes()) for name in tr)
    for name in ("fast_head.weight", "bos_embedding", "unrelated",
                 "action_expert.layers.0.weight"):
        assert not dict(m.named_parameters())[name].requires_grad, name

    # Все разрешённые тензоры участвуют в настоящем полном пути.
    m.zero_grad(set_to_none=True)
    out = m.forward_joint_depth_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    loss = sum(v.float().square().mean() for v in out["logits"][1:])
    loss.backward()
    missing = [name for name, p in m.named_parameters()
               if p.requires_grad and p.grad is None]
    nonfinite = [name for name, p in m.named_parameters()
                 if p.requires_grad and p.grad is not None
                 and not torch.isfinite(p.grad).all()]
    assert not missing, missing
    assert not nonfinite, nonfinite
    assert any(float(p.grad.abs().max()) > 0 for p in m.parameters()
               if p.requires_grad and p.grad is not None)

    print(
        "самопроверка depth_rvq_joint12 пройдена: conditional targets "
        "исправляют predicted prefix и используют in_project; правильный "
        "prefix воспроизводит исходные RVQ-метки; fast-путь побитово равен "
        "Joint12; full исполняет 24 слоя; нулевая feedback-проекция — "
        "тождество; q0 заморожен; whitelist и градиенты поздней ветви полны")


if __name__ == "__main__":
    selftest()
