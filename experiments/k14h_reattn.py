#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14h: draft-conditioned re-attention после раннего выхода (§49).

ЗАЧЕМ. Три попытки улучшить ЧИТАТЕЛЯ — мягкое правило вывода (§45.1),
двухслойная голова (§46.1), другой бюджет (§47) — дали по удвоению захвата и
упёрлись около 0.05...0.09 при пороге 0.20. Общее у них одно: о черновике
слои после раннего выхода получают ровно один вектор E0[q0], прибавленный к
состоянию.

ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ — и это важнее того, что делает. Он НЕ даёт
поздним слоям доступ к сцене впервые. Слои 13-18 продолжают совместное
внимание, токены действий обращаются к VLM-префиксу и там, а после
аддитивной ветви их запросы уже косвенно обусловлены q0. Утверждать, будто
новой информации о сцене у поздних слоёв не появляется, неверно.

ЧТО ОН ПРОВЕРЯЕТ НА САМОМ ДЕЛЕ — гипотезу у́же: помогает ли ОТДЕЛЬНЫЙ ЯВНЫЙ
запрос к префиксу на самой границе, сразу после получения q0, СВЕРХ того
доступа, который слои 13-18 уже реализуют сами. Запрос строится из
состояния и черновика напрямую, а не через прибавку к состоянию, и читает
H_vlm ровно того слоя, на котором черновик получен. Это специализированное
повторное обращение к сцене, а не первое.

ЧТО ЭТО ЗНАЧИТ ДЛЯ ТРЁХ ВАРИАНТОВ.
    baseline      — существующие слои сами пользуются префиксом после
                    аддитивной ветви;
    reattn_state  — на границе добавлен отдельный прямой проход внимания;
    reattn_draft  — тот же проход, но в запросе явный q0.
Первая пара отвечает, помогает ли сам дополнительный проход; вторая — даёт
ли что-то ЯВНАЯ зависимость запроса от черновика сверх уже имеющейся
косвенной.

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
import os
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

        def __init__(self, d_model, d_code, d_ctx=None, n_heads=8,
                     use_draft=True, dtype=torch.float32):
            """`d_ctx` — ширина ЧИТАЕМОГО префикса, она же ширина башни VLM.

            ОНА НЕ РАВНА `d_model`. Эксперт действий у́же башни (768 против
            2048), а блок читает из башни и пишет в поток действий. Значение
            по умолчанию `d_ctx = d_model` оставлено только ради
            самопроверок на игрушечных размерах; настоящая сборка обязана
            передавать ширину явно.
            """
            super().__init__()
            if d_model % n_heads:
                raise ValueError(f"d_model {d_model} не делится на {n_heads}")
            d_ctx = int(d_model if d_ctx is None else d_ctx)
            self.d_model, self.n_heads = int(d_model), int(n_heads)
            self.d_code, self.d_ctx = int(d_code), d_ctx
            self.use_draft = bool(use_draft)
            kw = dict(dtype=dtype)
            self.ln_h = nn.LayerNorm(d_model, **kw)
            self.p_d = nn.Linear(d_code, d_model, **kw)
            self.ln_d = nn.LayerNorm(d_model, **kw)
            self.ln_ctx = nn.LayerNorm(d_ctx, **kw)
            # ФОРМА ЗАПРОСА ОДНА У ОБОИХ ВАРИАНТОВ. Если бы у контроля
            # W_q была вдвое у́же, он имел бы меньше параметров, и
            # превосходство варианта с черновиком нельзя было бы приписать
            # информации q0 — оно объяснялось бы лишней ёмкостью. Поэтому
            # контроль подаёт во второй слот ФИКСИРОВАННЫЙ нулевой черновик;
            # через LN и P_D он превращается в константу, одинаковую для всех
            # позиций, и никакой информации о q0 не несёт.
            self.w_q = nn.Linear(2 * d_model, d_model, **kw)
            self.register_buffer("null_draft",
                                 torch.zeros(1, 1, d_code, dtype=dtype))
            self.w_k = nn.Linear(d_ctx, d_model, **kw)
            self.w_v = nn.Linear(d_ctx, d_model, **kw)
            self.w_o = nn.Linear(d_model, d_model, **kw)
            self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))
            with torch.no_grad():
                self.w_o.weight.zero_()
                if self.w_o.bias is not None:
                    self.w_o.bias.zero_()

        def _heads(self, x):
            b, n, _ = x.shape
            return x.view(b, n, self.n_heads, -1).transpose(1, 2)

        def set_use_draft(self, flag):
            """Переключение архитектуры БЕЗ пересборки весов.

            Обе архитектуры имеют одну форму запроса и один набор
            параметров, поэтому различаются только тем, что подаётся во
            второй слот. Один объект обслуживает оба варианта — это и делает
            проверку тождественности честной: сравниваются не две разные
            сборки, а один блок в двух режимах.
            """
            self.use_draft = bool(flag)

        def forward(self, h, d0, ctx, ctx_mask=None):
            """h: (B,P,D) действия; d0: (B,P,Dc); ctx: (B,S,D) префикс.

            `ctx_mask`: (B,S), True — валидный токен. ПАДДИНГ ОБЯЗАН БЫТЬ
            ЗАКРЫТ: без маски внимание распределялось бы и на дополненные
            позиции, а их число зависит от состава батча — то самое, из-за
            чего q0 когда-то оказался невоспроизводимым.
            """
            for nm_, got_, want_ in (("h", h.shape[-1], self.d_model),
                                     ("d0", d0.shape[-1], self.d_code),
                                     ("ctx", ctx.shape[-1], self.d_ctx)):
                if int(got_) != int(want_):
                    raise ValueError(
                        f"{nm_} шириной {int(got_)}, блок собран под "
                        f"{int(want_)} (d_model={self.d_model}, "
                        f"d_code={self.d_code}, d_ctx={self.d_ctx})")
            if ctx_mask is not None and \
                    int(ctx_mask.shape[-1]) != int(ctx.shape[1]):
                raise ValueError(
                    f"маска длины {int(ctx_mask.shape[-1])} при префиксе "
                    f"{int(ctx.shape[1])}")
            dt = self.w_q.weight.dtype
            hq = self.ln_h(h.to(dt))
            d_in = (d0.to(dt) if self.use_draft
                    else self.null_draft.to(dt).expand(h.shape[0],
                                                       h.shape[1], -1))
            q = self.w_q(torch.cat([hq, self.ln_d(self.p_d(d_in))], dim=-1))
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


ARCHITECTURES = ("baseline", "reattn_draft", "reattn_state")
FEEDBACK = ("on", "off")


def gate_case(architecture, feedback):
    if architecture not in ARCHITECTURES:
        raise SystemExit(f"неизвестная архитектура {architecture}")
    if feedback not in FEEDBACK:
        raise SystemExit(f"неизвестное состояние feedback {feedback}")
    return f"{architecture}/{feedback}"


GATE_SAME = ("heads", "head_dtype",
             "joint_ckpt", "joint_sha1", "ckpt", "q0_npz", "q0_npz_sha1",
             "q0_manifest_sha1", "plan_sha1", "plan_batch", "gate_r_sha1",
             "device", "gpu_uuid", "compute_dtype", "torch_version",
             "cuda_version", "tf32_matmul", "tf32_cudnn",
             "cudnn_deterministic", "cudnn_benchmark",
             "reattn_sha1", "depth_rvq_joint12_sha1", "depth_rvq_vla_sha1",
             "joint12_vla_sha1", "bar_sha1")


def check_identity_gate(path, *, architecture, feedback, expect, file_sha):
    """Требовать пройденный архитектурный гейт ДЛЯ ЭТОЙ КОНФИГУРАЦИИ.

    ОТСУТСТВИЕ ФАЙЛА — ОТКАЗ, как и с Gate R. Но одного `passed` мало:
    гейт обязан относиться к тому же коду, тем же весам, тому же
    каноническому черновику, тому же режиму вычислений И к той самой паре
    «архитектура + аддитивная обратная связь», которая сейчас запускается.
    Общий флаг «что-то проверялось» проверкой не является.

    `expect` — словарь величин текущего прогона, `file_sha` — функция
    отпечатка файла: модуль не решает, как их считать.
    """
    import json
    import os
    if not path:
        raise SystemExit(
            "не указан артефакт гейта тождественности. Обучать архитектуру, "
            "про которую не доказано, что до обучения она совпадает с "
            "прежней, значит сравнивать её с неизвестно чем")
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}: гейт тождественности не проводился")
    g = json.load(open(path))
    if g.get("kind") != "k14h_identity_gate":
        raise SystemExit(f"{path} описывает {g.get('kind')}")
    if g.get("passed") is not True:
        raise SystemExit(f"гейт не пройден: {g.get('passed')!r}")
    if not g.get("run_id"):
        raise SystemExit("в гейте нет run_id")
    if g.get("git_dirty"):
        raise SystemExit("гейт снят при незакоммиченном коде")
    miss = [k for k in GATE_SAME if g.get(k) is None]
    if miss:
        raise SystemExit(f"в гейте нет полей {miss}")
    bad = [f"{k}: гейт {g[k]}, сейчас {expect.get(k)}"
           for k in GATE_SAME if str(g[k]) != str(expect.get(k))]
    if bad:
        raise SystemExit("гейт снят в другой обстановке: " + "; ".join(bad))
    case = gate_case(architecture, feedback)
    cases = g.get("cases") or {}
    if case not in cases:
        raise SystemExit(
            f"в гейте нет случая {case}; есть {sorted(cases)}. Проверялась "
            f"другая конфигурация, а не запускаемая")
    c = cases[case]
    if c.get("passed") is not True:
        raise SystemExit(f"случай {case} не пройден")
    # ЧИСЛО ВЫЗОВОВ ЗАВИСИТ ОТ АРХИТЕКТУРЫ. У baseline блок выключен
    # целиком, и правильное ожидание — ноль во всех режимах. Требовать 0/1/1
    # от любого случая значило бы либо никогда не пропускать baseline, либо
    # считать вход в оболочку вместо фактического вычисления внимания.
    on = 0 if architecture == "baseline" else 1
    for mode, want in (("fast", 0), ("medium", on), ("full", on)):
        if mode not in (c.get("modes") or {}):
            raise SystemExit(f"случай {case}: режим {mode} не проверялся")
        got = (c.get("reattn_calls") or {}).get(mode)
        if int(got if got is not None else -1) != want:
            raise SystemExit(f"случай {case}, режим {mode}: блок вызван "
                             f"{got} раз, ожидалось {want}")
    if c.get("q0_matches_canonical") is not True:
        raise SystemExit(f"случай {case}: q0 не сверялся с каноническим")
    return dict(identity_gate=path, identity_gate_sha1=file_sha(path),
                identity_gate_run_id=g["run_id"], identity_case=case,
                identity_rows_sha1=g.get("rows_sha1"), identity_passed=True)


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
            # ЧИТАЕМЫЙ ПРЕФИКС ШИРЕ ПОТОКА ДЕЙСТВИЙ. Эксперт действий у́же
            # башни VLM, и блок ходит за информацией именно в башню. Ширина
            # берётся из конфига, а не приравнивается к d_model; несовпадение
            # с фактическим тензором отвергается в forward.
            d_ctx = getattr(self.config.vlm_config.text_config,
                            "hidden_size", None)
            if not d_ctx:
                raise RuntimeError(
                    "в конфиге нет vlm_config.text_config.hidden_size — "
                    "ширину префикса взять неоткуда")
            d_ctx = int(d_ctx)
            dev = self.fast_head.weight.device
            self.draft_reattn = nn.ModuleList(
                [Block(d_model, d_code, d_ctx=d_ctx, n_heads=n_heads,
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
                      f"действия {d_model}, префикс {d_ctx}, код {d_code}, "
                      f"{n / 1e6:.3f} млн параметров, W_o обнулена")

        def configure_draft_reattn(self, *, stage="q1", variant="main",
                                   include_block=None, verbose=True):
            """Белый список этапа плюс параметры блока. Точное множество.

            `variant="no_additive_feedback"` выключает аддитивную ветвь
            CodeFeedback; базовый класс знает её под именем `no_feedback`, и
            имя переводится здесь, а не в вызывающем коде.

            `include_block` — входят ли параметры блока в обучаемые. У
            baseline блок НЕ ВЫЗЫВАЕТСЯ, значит градиента до него не
            доходит; держать его в белом списке означало бы объявить
            обучаемым то, что не обучается, и тренер справедливо отказывает
            на проверке «градиент есть у всех обучаемых». Но главное не
            техническое: baseline обязан обучать ровно то же, что обучала
            прежняя модель, иначе это не та точка отсчёта, с которой
            сравнивают. По умолчанию берётся `draft_reattn_enabled`, но
            вызывающему лучше сказать явно — тогда результат не зависит от
            порядка установки флагов.
            """
            base_variant = ("no_feedback" if variant == "no_additive_feedback"
                            else variant)
            # СНАЧАЛА ЗАМОРАЖИВАЕТСЯ ВСЁ, ПОТОМ ПРИМЕНЯЕТСЯ СПИСОК. При
            # переключении между конфигурациями `requires_grad=True` мог бы
            # уцелеть от предыдущей — и обучалось бы объединение двух белых
            # списков, а не заявленный. Проверяется точным сравнением
            # множеств ниже, но полагаться на проверку вместо сброса нельзя.
            for p in self.parameters():
                p.requires_grad_(False)
            info = self.configure_joint_depth_rvq(stage=stage,
                                                  variant=base_variant,
                                                  verbose=False)
            if not hasattr(self, "draft_reattn"):
                raise RuntimeError("блок не собран")
            if include_block is None:
                include_block = bool(getattr(self, "draft_reattn_enabled",
                                             False))
            include_block = bool(include_block)
            if include_block and not getattr(self, "draft_reattn_enabled",
                                             False):
                raise RuntimeError(
                    "блок просят обучать, но он выключен: градиент до него "
                    "не дойдёт")
            if include_block:
                for n, p in self.named_parameters():
                    if n.startswith("draft_reattn.0."):
                        p.requires_grad_(True)
            names = sorted(n for n, p in self.named_parameters()
                           if p.requires_grad)
            prefixes = (TRAIN_PREFIXES if include_block
                        else tuple(x for x in TRAIN_PREFIXES
                                   if not x.startswith("draft_reattn.")))
            want = sorted(n for n, _ in self.named_parameters()
                          if n.startswith(prefixes))
            # ИМЯ ВАРИАНТА ТОЧНОЕ. При включённом блоке q0 продолжает
            # входить в вычисление через запрос внимания, поэтому «без
            # обратной связи» было бы неверно: выключается только АДДИТИВНАЯ
            # ветвь CodeFeedback.
            if variant in ("no_feedback", "no_additive_feedback"):
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
                       reattn_uses_draft=bool(self.draft_reattn_uses_draft),
                       reattn_trainable=include_block)
            if verbose:
                print(f"  этап {stage}/{variant}, блок "
                      f"{'обучается' if include_block else 'ВНЕ обучаемых'}: "
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
                # ЗАПРОС СТРОИТСЯ ИЗ СОСТОЯНИЯ ДО additive feedback, и обе
                # добавки ПАРАЛЛЕЛЬНЫ:
                #     h' = h + F(E0[q0]) + ReAttn(h, E0[q0], H_vlm)
                # Если бы запрос собирался из состояния ПОСЛЕ feedback,
                # черновик входил бы дважды — и это была бы другая
                # архитектура, а сравнение с базовой перестало бы отвечать на
                # заданный вопрос.
                h_pre = action_hidden
                if self.depth_rvq_feedback_built \
                        and self.depth_rvq_feedback_mask[g]:
                    action_hidden = self.depth_rvq_feedback[g](action_hidden,
                                                               emb)
                if g == 0 and getattr(self, "draft_reattn_enabled", False):
                    # БЕЗ МАСКИ БЛОК НЕ РАБОТАЕТ. Внимание разошлось бы и на
                    # дополненные позиции, а их число зависит от состава
                    # батча — ровно та причина, по которой q0 когда-то
                    # оказался невоспроизводимым. Молча брать полный префикс
                    # нельзя.
                    if attention_mask is None:
                        raise RuntimeError(
                            "re-attention включён, а маски префикса нет: "
                            "паддинг остался бы открытым")
                    c = self.draft_reattn[0](h_pre, emb, vlm_hidden,
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
    # ТРИ ШИРИНЫ РАЗНЫЕ, И ЭТО НЕ ПРИДИРКА. Первый прогон гейта упал именно
    # потому, что самопроверка гоняла блок на ctx той же ширины, что и поток
    # действий, — и не могла увидеть, что K и V проецируются не из той
    # размерности. Равные размеры в тесте прячут ровно тот класс ошибки,
    # ради которого тест написан.
    B, P, S, D, Dc, Dx = 3, 4, 7, 16, 8, 24

    blk = Block(D, Dc, d_ctx=Dx, n_heads=4)
    h = torch.randn(B, P, D)
    d0 = torch.randn(B, P, Dc)
    ctx = torch.randn(B, S, Dx)
    assert blk.w_k.in_features == Dx and blk.w_v.in_features == Dx, \
        "K и V проецируются не из ширины префикса"
    assert blk.w_k.out_features == D and blk.ln_ctx.normalized_shape == (Dx,)

    # --- НЕСОВПАДЕНИЕ ШИРИН ОТВЕРГАЕТСЯ С ВНЯТНЫМ СООБЩЕНИЕМ -------------
    for bad_h, bad_d, bad_c, why in (
            (torch.randn(B, P, D + 1), d0, ctx, "h шириной"),
            (h, torch.randn(B, P, Dc + 1), ctx, "d0 шириной"),
            (h, d0, torch.randn(B, S, Dx + 1), "ctx шириной")):
        try:
            blk(bad_h, bad_d, bad_c)
        except ValueError as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(f"принят вход неверной ширины: {why}")
    try:
        blk(h, d0, ctx, ctx_mask=torch.ones(B, S + 1, dtype=torch.bool))
    except ValueError as e:
        assert "маска длины" in str(e), e
    else:
        raise AssertionError("принята маска не по длине префикса")
    assert Block(D, Dc, n_heads=4).d_ctx == D, \
        "по умолчанию ширина префикса обязана совпадать с d_model"

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
    ctx2[:, -3:] = torch.randn(B, 3, Dx) * 100      # мусор в закрытых позициях
    o2 = blk(h, d0, ctx2, ctx_mask=m)
    assert torch.allclose(o1, o2, atol=1e-6), \
        "закрытые маской позиции влияют на выход"
    # без маски — влияют, иначе проверка выше ничего не значила бы
    assert not torch.allclose(blk(h, d0, ctx), blk(h, d0, ctx2), atol=1e-6)

    # --- ЧЕРНОВИК ВХОДИТ В ЗАПРОС ----------------------------------------
    o3 = blk(h, torch.randn(B, P, Dc), ctx)
    assert not torch.allclose(blk(h, d0, ctx), o3, atol=1e-6), \
        "черновик не влияет на выход, хотя заявлен в запросе"
    nod = Block(D, Dc, d_ctx=Dx, n_heads=4, use_draft=False)
    with torch.no_grad():
        nod.w_o.weight.normal_(0, 0.5)
    assert torch.equal(nod(h, d0, ctx), nod(h, torch.randn(B, P, Dc), ctx)), \
        "вариант без черновика от него зависит"
    # КОНТРОЛЬ УРАВНЕН ПО ПАРАМЕТРАМ: форма запроса одна и та же, во втором
    # слоте фиксированный нулевой черновик.
    assert nod.w_q.in_features == 2 * D and blk.w_q.in_features == 2 * D
    n_blk = sum(p.numel() for p in blk.parameters())
    n_nod = sum(p.numel() for p in nod.parameters())
    assert n_blk == n_nod, (f"число параметров различается: {n_blk} против "
                            f"{n_nod} — превосходство нельзя будет приписать "
                            f"информации q0")
    assert float(nod.null_draft.abs().max()) == 0.0

    # --- ГРАДИЕНТЫ: ЧТО ОБЯЗАНО БЫТЬ НЕНУЛЕВЫМ И КОГДА --------------------
    # При ТОЧНОМ W_o = 0 выход не зависит от Q, K, V, норм и alpha, поэтому
    # их градиенты закономерно нулевые, и требовать обратного значило бы
    # требовать математически невозможного. Ненулевым обязан быть ровно один
    # градиент — по самой W_o: иначе блок мёртв и не тронется с места.
    b2 = Block(D, Dc, d_ctx=Dx, n_heads=4)
    b2(h, d0, ctx).sum().backward()
    assert b2.w_o.weight.grad is not None \
        and float(b2.w_o.weight.grad.abs().max()) > 0, \
        "при нулевой W_o градиент до неё не доходит — блок мёртв"
    for nm_ in ("w_q.weight", "w_k.weight", "w_v.weight", "alpha"):
        g_ = dict(b2.named_parameters())[nm_].grad
        assert g_ is None or float(g_.abs().max()) == 0.0, \
            f"при нулевой W_o градиент по {nm_} обязан быть нулевым"
    # ПОСЛЕ ненулевой W_o он доходит до всех
    blk.zero_grad()
    blk(h, d0, ctx).sum().backward()
    nog = [n for n, p in blk.named_parameters() if p.grad is None
           or float(p.grad.abs().max()) == 0.0]
    assert not nog, f"после ненулевой W_o нет градиента у {nog}"

    # --- ФОРМА ГОЛОВ -------------------------------------------------------
    try:
        Block(D, Dc, d_ctx=Dx, n_heads=5)
    except ValueError:
        pass
    else:
        raise AssertionError("принято число голов, не делящее d_model")
    # --- БЕЛЫЙ СПИСОК: BASELINE НЕ ОБУЧАЕТ ТО, ЧТО НЕ ВЫЗЫВАЕТСЯ ----------
    # Эта проверка написана после того, как смоук baseline упал на «градиента
    # нет у draft_reattn.*»: блок безусловно попадал в обучаемые, хотя при
    # выключенной архитектуре он не вызывается ни разу. Игрушечная база
    # нужна именно затем, чтобы логика белого списка проверялась без модели
    # на 2.2 млрд параметров.
    import torch.nn as nn
    from types import SimpleNamespace

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_rvq_norms = nn.ModuleList([nn.Linear(D, D)])
            self.depth_rvq_heads = nn.ModuleList([nn.Linear(D, D)])
            self.depth_rvq_feedback = nn.ModuleList([nn.Linear(D, D)])
            self.deep = nn.Linear(D, D)
            self.fast_head = nn.Linear(D, 9)
            self.register_buffer("depth_rvq_books", torch.zeros(3, 5, Dc))
            self.config = SimpleNamespace(vlm_config=SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=Dx)))

        def configure_joint_depth_rvq(self, *, stage, variant, verbose=True):
            for q in self.parameters():
                q.requires_grad_(False)
            pref = ["depth_rvq_norms.0.", "depth_rvq_heads.0."]
            if variant != "no_feedback":
                pref.append("depth_rvq_feedback.0.")
            for nm2, q in self.named_parameters():
                if nm2.startswith(tuple(pref)):
                    q.requires_grad_(True)
            nms = sorted(nm2 for nm2, q in self.named_parameters()
                         if q.requires_grad)
            return dict(names=nms, n_tensors=len(nms),
                        n_params=sum(q.numel() for nm2, q
                                     in self.named_parameters()
                                     if nm2 in set(nms)))

    Cls = make_draft_reattn_class(_Base)
    m = Cls()
    m.init_draft_reattn(n_heads=4, verbose=False)
    base_names = sorted(_Base.configure_joint_depth_rvq(
        m, stage="q1", variant="main")["names"])

    m.draft_reattn_enabled = False
    i_base = m.configure_draft_reattn(stage="q1", variant="main",
                                      include_block=False, verbose=False)
    assert sorted(i_base["names"]) == base_names, \
        "baseline обучает не то же, что прежняя модель"
    assert not any(n_.startswith("draft_reattn.") for n_ in i_base["names"])
    assert i_base["reattn_trainable"] is False
    # умолчание следует флагу, а не наоборот
    assert sorted(m.configure_draft_reattn(
        stage="q1", variant="main", verbose=False)["names"]) == base_names
    try:
        m.configure_draft_reattn(stage="q1", variant="main",
                                 include_block=True, verbose=False)
    except RuntimeError as e:
        assert "выключен" in str(e), e
    else:
        raise AssertionError("блок объявлен обучаемым при выключенной ветви")

    m.draft_reattn_enabled = True
    i_on = m.configure_draft_reattn(stage="q1", variant="main",
                                    include_block=True, verbose=False)
    added = sorted(set(i_on["names"]) - set(base_names))
    assert added and all(n_.startswith("draft_reattn.0.") for n_ in added)
    assert len(added) == len([n_ for n_, _ in m.named_parameters()
                              if n_.startswith("draft_reattn.0.")])
    assert i_on["reattn_trainable"] is True
    # ВАРИАНТ БЕЗ АДДИТИВНОЙ ВЕТВИ УБИРАЕТ ТОЛЬКО ЕЁ
    i_nf = m.configure_draft_reattn(stage="q1",
                                    variant="no_additive_feedback",
                                    include_block=True, verbose=False)
    assert not any(n_.startswith("depth_rvq_feedback.")
                   for n_ in i_nf["names"])
    assert any(n_.startswith("draft_reattn.0.") for n_ in i_nf["names"])

    # --- ПРОВЕРКА ГЕЙТА: ОТКАЗ НА КАЖДОМ НЕСОВПАДЕНИИ ---------------------
    import json as _json
    import tempfile
    exp = {k: f"<{k}>" for k in GATE_SAME}
    good = dict(kind="k14h_identity_gate", passed=True, run_id="R1",
                git_dirty=False, rows_sha1="RS",
                cases={"reattn_draft/on": dict(
                    passed=True, modes={"fast": {}, "medium": {}, "full": {}},
                    reattn_calls={"fast": 0, "medium": 1, "full": 1},
                    q0_matches_canonical=True)}, **exp)
    with tempfile.TemporaryDirectory() as td:
        def w(obj):
            q = os.path.join(td, "g.json")
            _json.dump(obj, open(q, "w"))
            return q
        info = check_identity_gate(w(good), architecture="reattn_draft",
                                   feedback="on", expect=exp,
                                   file_sha=lambda _p: "SH")
        assert info["identity_case"] == "reattn_draft/on"
        assert info["identity_gate_run_id"] == "R1"
        for patch, why in (
                ({"passed": False}, "не пройден"),
                ({"run_id": ""}, "run_id"),
                ({"git_dirty": True}, "незакоммиченном"),
                ({"kind": "x"}, "описывает"),
                ({"bar_sha1": "ДРУГОЕ"}, "в другой обстановке"),
                ({"gpu_uuid": None}, "нет полей")):
            try:
                check_identity_gate(w(dict(good, **patch)),
                                    architecture="reattn_draft",
                                    feedback="on", expect=exp,
                                    file_sha=lambda _p: "SH")
            except SystemExit as e:
                assert why in str(e), (why, e)
            else:
                raise AssertionError(f"гейт принят при {patch}")
        # ЧУЖОЙ СЛУЧАЙ НЕ ГОДИТСЯ
        for arch, fb, why in (("reattn_state", "on", "нет случая"),
                              ("reattn_draft", "off", "нет случая")):
            try:
                check_identity_gate(w(good), architecture=arch, feedback=fb,
                                    expect=exp, file_sha=lambda _p: "SH")
            except SystemExit as e:
                assert why in str(e), (arch, fb, e)
            else:
                raise AssertionError(f"принят чужой случай {arch}/{fb}")
        # BASELINE: ожидание ноль во всех режимах
        base_ok = _json.loads(_json.dumps(good))
        base_ok["cases"] = {"baseline/on": dict(
            passed=True, modes={"fast": {}, "medium": {}, "full": {}},
            reattn_calls={"fast": 0, "medium": 0, "full": 0},
            q0_matches_canonical=True)}
        assert check_identity_gate(w(base_ok), architecture="baseline",
                                   feedback="on", expect=exp,
                                   file_sha=lambda _p: "SH")["identity_case"] \
            == "baseline/on"
        bad_b = _json.loads(_json.dumps(base_ok))
        bad_b["cases"]["baseline/on"]["reattn_calls"]["medium"] = 1
        try:
            check_identity_gate(w(bad_b), architecture="baseline",
                                feedback="on", expect=exp,
                                file_sha=lambda _p: "SH")
        except SystemExit as e:
            assert "medium" in str(e), e
        else:
            raise AssertionError("baseline с вызовом блока принят")

        # ЧИСЛО ВЫЗОВОВ БЛОКА У reattn
        for mode, val, why in (("fast", 1, "fast"), ("medium", 0, "medium"),
                               ("full", 2, "full")):
            bad_c = _json.loads(_json.dumps(good))
            bad_c["cases"]["reattn_draft/on"]["reattn_calls"][mode] = val
            try:
                check_identity_gate(w(bad_c), architecture="reattn_draft",
                                    feedback="on", expect=exp,
                                    file_sha=lambda _p: "SH")
            except SystemExit as e:
                assert why in str(e), (mode, e)
            else:
                raise AssertionError(f"принято {val} вызовов в {mode}")
        no_q0 = _json.loads(_json.dumps(good))
        no_q0["cases"]["reattn_draft/on"]["q0_matches_canonical"] = False
        try:
            check_identity_gate(w(no_q0), architecture="reattn_draft",
                                feedback="on", expect=exp,
                                file_sha=lambda _p: "SH")
        except SystemExit as e:
            assert "q0 не сверялся" in str(e), e
        else:
            raise AssertionError("принят гейт без сверки q0")
        for bad in ("", os.path.join(td, "нет.json")):
            try:
                check_identity_gate(bad, architecture="reattn_draft",
                                    feedback="on", expect=exp,
                                    file_sha=lambda _p: "SH")
            except SystemExit:
                pass
            else:
                raise AssertionError("принято отсутствие гейта")
    assert gate_case("baseline", "off") == "baseline/off"
    for bad in (("нет", "on"), ("baseline", "нет")):
        try:
            gate_case(*bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"принято {bad}")

    print("самопроверка k14h_reattn пройдена")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
