#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14h: draft-conditioned re-attention после раннего выхода (§49).

ЗАЧЕМ. Три попытки улучшить ЧИТАТЕЛЯ — мягкое правило вывода (§45.1),
двухслойная голова (§46.1), другой бюджет (§47) — дали по удвоению захвата и
упёрлись около 0.05...0.09 при пороге 0.20. Общее у них одно: слои после
раннего выхода получают о черновике ровно один вектор E0[q0], прибавленный к
состоянию. Новой информации о сцене у них не появляется, они могут лишь иначе
переработать уже имеющуюся.

ЧТО ДЕЛАЕТ ЭТОТ МОДУЛЬ. Даёт поздним слоям возможность СХОДИТЬ ЗА НОВОЙ
информацией: после получения q0 формируется запрос, которого до раннего
выхода не существовало (модель ещё не знала, какой черновик выберет), и по
нему из уже вычисленного VLM-префикса извлекается то, что нужно именно для
исправления ЭТОГО черновика.

ПРЕФИКС, А НЕ КАМЕРЫ. К моменту прохода изображения уже слиты в общий
vlm_hidden вместе с текстом, и маски позиций изображения туда не передаётся.
Это re-attention к VLM-ПРЕФИКСУ. Image-only — отдельный контроль и только с
явной маской, сверенной с выходом inputs_merger.

ПОЧЕМУ ОТДЕЛЬНЫЙ ФАЙЛ, А НЕ ПРАВКА depth_rvq_joint12.py. Семантический
отпечаток того файла входит в `code_version` чекпойнта q1_main_s0.pt, на
котором стоят кэш h18 и все сравнения K-14f/K-14g. Правка рассорила бы их
между собой. Расширение подклассом — та же схема, какой сам depth-RVQ
расширяет Joint12.

ОДИН ПРОХОД VLA СОХРАНЯЕТСЯ: изображение кодируется один раз, слои 1-12 не
перезапускаются, q0 не меняется, работа добавляется только в режимах medium
и full.
"""
import sys

try:
    from .depth_rvq_vla import straight_through
except ImportError:                                   # запуск как скрипта
    from depth_rvq_vla import straight_through


def make_draft_reattn_block():
    """Блок re-attention. Возвращается фабрикой, чтобы torch грузился лениво."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DraftReattn(nn.Module):
        """h' = h + alpha * W_o( Attention(Q(h, D0), K(ctx), V(ctx)) ).

        ЗАПРОС СОБИРАЕТСЯ КОНКАТЕНАЦИЕЙ, А НЕ СУММОЙ. Сумма лишила бы блок
        возможности взвешивать два источника запроса — состояние и черновик, —
        и различие двух моделей смешалось бы с различием способов смешивания.
        Выбор зафиксирован в §49 до прогона.

        ВЫХОДНАЯ ПРОЕКЦИЯ ОБНУЛЕНА, alpha = 1. Обнулять оба нельзя: градиент
        по alpha шёл бы через нулевой выход и по W_o через нулевую alpha —
        блок остался бы мёртвым. При W_o = 0 модель функционально тождественна
        нынешней, и сравнение архитектур начинается с общей точки.
        """

        def __init__(self, d_model, d_code, n_heads=8, use_draft=True,
                     dtype=torch.float32):
            super().__init__()
            if d_model % n_heads:
                raise ValueError(f"d_model {d_model} не делится на {n_heads}")
            self.d_model, self.n_heads = int(d_model), int(n_heads)
            self.use_draft = bool(use_draft)
            kw = dict(dtype=dtype)
            self.ln_h = nn.LayerNorm(d_model, **kw)
            self.p_d = nn.Linear(d_code, d_model, **kw)
            self.ln_d = nn.LayerNorm(d_model, **kw)
            self.ln_ctx = nn.LayerNorm(d_model, **kw)
            q_in = 2 * d_model if self.use_draft else d_model
            self.w_q = nn.Linear(q_in, d_model, **kw)
            self.w_k = nn.Linear(d_model, d_model, **kw)
            self.w_v = nn.Linear(d_model, d_model, **kw)
            self.w_o = nn.Linear(d_model, d_model, **kw)
            self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))
            with torch.no_grad():
                self.w_o.weight.zero_()
                if self.w_o.bias is not None:
                    self.w_o.bias.zero_()

        def _heads(self, x):
            b, n, _ = x.shape
            return x.view(b, n, self.n_heads, -1).transpose(1, 2)

        def forward(self, h, d0, ctx, ctx_mask=None):
            """h: (B,P,D) действия; d0: (B,P,Dc); ctx: (B,S,D) префикс.

            `ctx_mask`: (B,S), True — валидный токен. ПАДДИНГ ОБЯЗАН БЫТЬ
            ЗАКРЫТ: без маски внимание распределялось бы и на дополненные
            позиции, а их число зависит от состава батча — то самое, из-за
            чего q0 когда-то оказался невоспроизводимым.
            """
            dt = self.w_q.weight.dtype
            hq = self.ln_h(h.to(dt))
            if self.use_draft:
                q = self.w_q(torch.cat([hq, self.ln_d(self.p_d(d0.to(dt)))],
                                       dim=-1))
            else:
                q = self.w_q(hq)
            c = self.ln_ctx(ctx.to(dt))
            k, v = self.w_k(c), self.w_v(c)
            am = None
            if ctx_mask is not None:
                am = ctx_mask.to(torch.bool)[:, None, None, :]
            out = F.scaled_dot_product_attention(
                self._heads(q), self._heads(k), self._heads(v),
                attn_mask=am)
            out = out.transpose(1, 2).reshape(h.shape[0], h.shape[1], -1)
            return self.alpha * self.w_o(out)

    return DraftReattn


TRAIN_PREFIXES = ("depth_rvq_norms.0.", "depth_rvq_heads.0.",
                  "depth_rvq_feedback.0.", "draft_reattn.0.")


def make_draft_reattn_class(base_cls):
    """Подкласс depth-RVQ с блоком re-attention после раннего выхода."""
    import torch
    import torch.nn as nn
    Block = make_draft_reattn_block()

    class JointDraftReattn(base_cls):

        def init_draft_reattn(self, *, n_heads=8, use_draft=True,
                              head_dtype=torch.float32, verbose=True):
            if not hasattr(self, "depth_rvq_books"):
                raise RuntimeError("сначала init_joint_depth_rvq")
            if hasattr(self, "draft_reattn"):
                raise RuntimeError("блок уже собран")
            d_model = int(self.fast_head.in_features)
            d_code = int(self.depth_rvq_books.shape[-1])
            dev = self.fast_head.weight.device
            self.draft_reattn = nn.ModuleList(
                [Block(d_model, d_code, n_heads=n_heads,
                       use_draft=use_draft, dtype=head_dtype).to(dev)])
            self.draft_reattn_enabled = True
            self.draft_reattn_uses_draft = bool(use_draft)
            # ПОСЛЕ СБОРКИ НИЧЕГО НЕ ОБУЧАЕТСЯ, как и в depth-RVQ: этап
            # выбирается явно, умолчания нет.
            for p in self.draft_reattn.parameters():
                p.requires_grad_(False)
            if verbose:
                n = sum(p.numel() for p in self.draft_reattn.parameters())
                print(f"  re-attention собран: голов {n_heads}, запрос "
                      f"{'с черновиком' if use_draft else 'БЕЗ черновика'}, "
                      f"{n / 1e6:.3f} млн параметров, W_o обнулена")

        def configure_draft_reattn(self, *, stage="q1", variant="main",
                                   verbose=True):
            """Белый список этапа плюс параметры блока. Точное множество."""
            info = self.configure_joint_depth_rvq(stage=stage, variant=variant,
                                                  verbose=False)
            if not hasattr(self, "draft_reattn"):
                raise RuntimeError("блок не собран")
            for n, p in self.named_parameters():
                if n.startswith("draft_reattn.0."):
                    p.requires_grad_(True)
            names = sorted(n for n, p in self.named_parameters()
                           if p.requires_grad)
            want = sorted(n for n, _ in self.named_parameters()
                          if n.startswith(TRAIN_PREFIXES))
            if variant == "no_feedback":
                want = [n for n in want
                        if not n.startswith("depth_rvq_feedback.")]
            if names != want:
                raise RuntimeError(
                    f"обучаемые не совпали с белым списком: лишние "
                    f"{sorted(set(names) - set(want))[:5]}, нет "
                    f"{sorted(set(want) - set(names))[:5]}")
            out = dict(info)
            out.update(names=names, n_tensors=len(names),
                       n_params=int(sum(p.numel()
                                        for n, p in self.named_parameters()
                                        if n in set(names))),
                       reattn_uses_draft=bool(self.draft_reattn_uses_draft))
            if verbose:
                print(f"  этап {stage}/{variant} с re-attention: "
                      f"{out['n_tensors']} тензоров, "
                      f"{out['n_params'] / 1e6:.3f} млн параметров")
            return out

        def forward_joint_depth_rvq(self, *, vlm_inputs_embeds, attention_mask,
                                    position_ids, mode="full",
                                    teacher_codes=None, teacher_levels=(),
                                    tau=1.0):
            """Тот же проход, что у базового класса, плюс блок после q0.

            ПОЧЕМУ ЦИКЛ ПОВТОРЁН, А НЕ ВЫЗВАН ЧЕРЕЗ super(). Блоку нужны
            vlm_hidden, маска и состояние действий в один и тот же момент —
            между выходом q0 и слоями 13-18, — а базовый проход этих величин
            наружу не отдаёт. Расхождение с базовым классом ловится
            обязательной проверкой тождественности при обнулённой W_o.
            """
            stop = {"fast": 0, "medium": 1, "full": 2}
            if mode not in stop:
                raise ValueError(f"неизвестный mode {mode}")
            stop_level = stop[mode]
            teach = set(int(x) for x in teacher_levels)
            if any(x not in (0, 1) for x in teach):
                raise ValueError(f"teacher только перед продолжением: {teach}")
            if getattr(self, "depth_rvq_feedback_mask", None) is None:
                raise RuntimeError("маска feedback не задана")

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
            if teacher_codes is not None and \
                    tuple(teacher_codes.shape) != (B, 3, n):
                raise ValueError(
                    f"teacher_codes формы {tuple(teacher_codes.shape)}")

            logits, pred, injected, embs = [], [], [], []
            layers_run, n_reattn = 0, 0
            for layer_idx in range(len(self.action_expert.layers)):
                vlm_hidden, action_hidden = self._shared_attention_forward(
                    vlm_hidden_states=vlm_hidden,
                    action_hidden_states=action_hidden, layer_idx=layer_idx,
                    attention_mask=mask4d, position_ids=position_ids,
                    past_key_values=None, use_cache=False, cache_position=None)
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
                if g == 0:
                    emb = self.depth_rvq_books[g][idx]
                else:
                    emb, _, _ = straight_through(
                        lg.float(), self.depth_rvq_books[g], tau=tau)
                inj_idx = idx
                if g in teach:
                    if teacher_codes is None:
                        raise ValueError(f"нет teacher_codes для уровня {g}")
                    inj_idx = teacher_codes[:, g, :].to(dev).long()
                    hard = self.depth_rvq_books[g][inj_idx].to(emb.dtype)
                    emb = hard if g == 0 else emb + (hard - emb).detach()
                injected.append(inj_idx)
                embs.append(emb)
                if self.depth_rvq_feedback_built \
                        and self.depth_rvq_feedback_mask[g]:
                    action_hidden = self.depth_rvq_feedback[g](action_hidden,
                                                               emb)
                # --- НОВОЕ: обращение к префиксу с учётом черновика --------
                if g == 0 and getattr(self, "draft_reattn_enabled", False):
                    c = self.draft_reattn[0](
                        action_hidden, emb, vlm_hidden,
                        ctx_mask=attention_mask)
                    action_hidden = action_hidden + c.to(action_hidden.dtype)
                    n_reattn += 1

            if layers_run != self.depth_rvq_exits[stop_level]:
                raise RuntimeError(
                    f"mode={mode}: исполнено {layers_run} слоёв, ожидалось "
                    f"{self.depth_rvq_exits[stop_level]}")
            want_re = 0 if mode == "fast" else (
                1 if getattr(self, "draft_reattn_enabled", False) else 0)
            if n_reattn != want_re:
                raise RuntimeError(
                    f"блок исполнен {n_reattn} раз, ожидалось {want_re}: "
                    f"лишний вызов означал бы лишний проход по префиксу")
            return dict(logits=logits, pred_codes=pred,
                        injected_codes=injected, embs=embs,
                        layers_run=layers_run, reattn_calls=n_reattn)

    return JointDraftReattn


def selftest():
    import torch
    torch.manual_seed(0)
    Block = make_draft_reattn_block()
    B, P, S, D, Dc = 3, 4, 7, 16, 8

    blk = Block(D, Dc, n_heads=4)
    h = torch.randn(B, P, D)
    d0 = torch.randn(B, P, Dc)
    ctx = torch.randn(B, S, D)

    # --- ПРИ ОБНУЛЁННОЙ W_o ВЫХОД СТРОГО НУЛЕВОЙ -------------------------
    out = blk(h, d0, ctx)
    assert out.shape == (B, P, D)
    assert float(out.abs().max()) == 0.0, "нулевая инициализация нарушена"
    assert float(blk.alpha.item()) == 1.0, "alpha обязана быть единицей"

    # --- ПОСЛЕ ИЗМЕНЕНИЯ W_o ВЫХОД НЕНУЛЕВОЙ -----------------------------
    with torch.no_grad():
        blk.w_o.weight.normal_(0, 0.5)
    assert float(blk(h, d0, ctx).abs().max()) > 0

    # --- МАСКА ДЕЙСТВУЕТ: закрытые позиции не влияют ----------------------
    m = torch.ones(B, S, dtype=torch.bool)
    m[:, -3:] = False
    o1 = blk(h, d0, ctx, ctx_mask=m)
    ctx2 = ctx.clone()
    ctx2[:, -3:] = torch.randn(B, 3, D) * 100      # мусор в закрытых позициях
    o2 = blk(h, d0, ctx2, ctx_mask=m)
    assert torch.allclose(o1, o2, atol=1e-6), \
        "закрытые маской позиции влияют на выход"
    # без маски — влияют, иначе проверка выше ничего не значила бы
    assert not torch.allclose(blk(h, d0, ctx), blk(h, d0, ctx2), atol=1e-6)

    # --- ЧЕРНОВИК ВХОДИТ В ЗАПРОС ----------------------------------------
    o3 = blk(h, torch.randn(B, P, Dc), ctx)
    assert not torch.allclose(blk(h, d0, ctx), o3, atol=1e-6), \
        "черновик не влияет на выход, хотя заявлен в запросе"
    nod = Block(D, Dc, n_heads=4, use_draft=False)
    with torch.no_grad():
        nod.w_o.weight.normal_(0, 0.5)
    assert torch.allclose(nod(h, d0, ctx), nod(h, torch.randn(B, P, Dc), ctx),
                          atol=0), "вариант без черновика от него зависит"
    assert nod.w_q.in_features == D and blk.w_q.in_features == 2 * D

    # --- ГРАДИЕНТ ДОХОДИТ ДО ВСЕХ ПАРАМЕТРОВ ------------------------------
    blk.zero_grad()
    blk(h, d0, ctx).sum().backward()
    nog = [n for n, p in blk.named_parameters() if p.grad is None]
    assert not nog, f"нет градиента у {nog}"
    # и при нулевой W_o тоже: alpha не должна обнулять путь к W_o
    b2 = Block(D, Dc, n_heads=4)
    b2(h, d0, ctx).sum().backward()
    assert b2.w_o.weight.grad is not None \
        and float(b2.w_o.weight.grad.abs().max()) > 0, \
        "при нулевой W_o градиент до неё не доходит — блок мёртв"

    # --- ФОРМА ГОЛОВ -------------------------------------------------------
    try:
        Block(D, Dc, n_heads=5)
    except ValueError:
        pass
    else:
        raise AssertionError("принято число голов, не делящее d_model")
    print("самопроверка k14h_reattn пройдена")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
