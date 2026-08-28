"""Depth-Aligned Single-Pass RVQ-VLA: уровни RVQ выходят на разных глубинах.

ИДЕЯ. BAR тратит три полных прохода башни — по одному на уровень RVQ. Здесь
уровни выпускаются на разных глубинах ОДНОГО прохода:

    слои  1-12 -> q0 -> уже исполнимый чанк      (Fast)
    слои 13-18 -> q1 -> уточнение                (Medium)
    слои 19-24 -> q2 -> полное действие          (Full)

Выбранный код возвращается в поток скрытых состояний, поэтому поздние сегменты
уточняют уже принятое грубое решение, а не гадают независимо.

ЧТО ЗДЕСЬ ПРОВЕРЕНО ПО КОДУ, А НЕ УГАДАНО (bar.py:1131-1247):
  * `_run_action_sequence` держит ОДИН цикл `for layer_idx in range(num_layers)`,
    и `_shared_attention_forward` продвигает башню VLM и эксперта ВМЕСТЕ,
    возвращая пару (vlm_hidden, action_hidden). Значит остановка цикла экономит
    обе ветви, а не только эксперта;
  * `_shared_attention_forward` НЕ вызывает `layer.forward()`: он сам дёргает
    `layer.input_layernorm`, `layer.self_attn.q_proj`, `layer.mlp` и так далее
    (bar.py:987-1061). Поэтому хук на модуле слоя не сработает — считать
    исполненные слои надо по `input_layernorm`;
  * пути к башням: `self.vlm.text_model.layers` и `self.action_expert.layers`
    (bar.py:987-988), угадывать не требуется;
  * `num_layers` берётся из `config.vlm_config.text_config.num_hidden_layers`;
  * в конце идёт `self.action_expert.norm(...)`, и только потом
    `action_lm_head`. Поэтому на КАЖДОМ раннем выходе норму надо применять тоже,
    иначе голова получит вход не той природы, под который обучена;
  * блок 0 — это `bos_len=block_size` при ПУСТОЙ истории, и берутся первые
    `n` позиций (bar.py:1288-1296).

Последнее важно отдельно: в зонде K-7c промежуточные глубины снимались БЕЗ этой
нормы, а `after_24` — с ней (вход `action_lm_head` уже нормирован). То есть
строка «без обучения» недооценивала ранние глубины. Здесь норма применяется
единообразно на всех выходах.

БЕЗОПАСНЫЙ СТАРТ. Проекции обратного внедрения инициализированы нулём, вентили
`alpha` тоже. При выходах `(24,)` и нулевых вентилях модель обязана выдать
ТОЧНО те же токены, что первый блок официальной BAR. Это проверяется, а не
предполагается.

Файлы third_party не изменяются: класс наследуется.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Минимальная LoRA-обёртка. Своя, чтобы не зависеть от peft на кластере.

    Базовый слой заморожен; обучаются только A и B. B инициализирован нулём,
    поэтому на старте обёртка тождественна базовому слою.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0,
                 dtype=torch.float32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B остаётся нулём

    def forward(self, x):
        out = self.base(x)
        lo = (x.to(self.A.dtype) @ self.A.t()) @ self.B.t()
        return out + (self.scale * lo).to(out.dtype)


def inject_lora(module, r, targets, alpha=1.0, dtype=torch.float32):
    """Обернуть в LoRA все nn.Linear, чьё имя оканчивается на один из targets.

    Возвращает список обёрнутых имён — его надо ПЕЧАТАТЬ и проверять глазами:
    пустой список означает, что имена не совпали, и обучение пойдёт вхолостую.
    """
    wrapped = []
    for name, child in list(module.named_modules()):
        for tgt in targets:
            if name.endswith(tgt) and isinstance(child, nn.Linear):
                parent = module
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(child, r, alpha, dtype))
                wrapped.append(name)
                break
    return wrapped


class CodeFeedback(nn.Module):
    """Возврат эмбеддинга выбранного кода в поток скрытых состояний.

        h <- h + alpha * P(LayerNorm(E_g[q_g]))

    ВНИМАНИЕ, alpha СТАРТУЕТ С ЕДИНИЦЫ, А НЕ С НУЛЯ. Обнулять надо ровно одну
    вещь — веса проекции. Если обнулить и alpha, и P, ветвь умирает навсегда:

        h' = h + alpha * P(e)
        dh'/dP     = alpha * e = 0   (alpha нулевая)
        dh'/dalpha = P(e)      = 0   (P нулевая)

    Ни один из двух параметров не получит градиента ни на одном шаге. Прямой
    проход при alpha=1 и нулевой P всё равно остаётся точным тождеством, потому
    что P(e)=0, но производная по весам P уже ненулевая, и ветвь оживает с
    первого шага. Сама alpha получит градиент, как только P сдвинется.
    """

    def __init__(self, d_code, d_model, alpha_init=1.0, dtype=torch.float32):
        super().__init__()
        self.norm = nn.LayerNorm(d_code, dtype=dtype)
        self.proj = nn.Linear(d_code, d_model, dtype=dtype)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=dtype))

    def forward(self, h, emb):
        d = self.alpha * self.proj(self.norm(emb.to(self.alpha.dtype)))
        return h + d.to(h.dtype)


def straight_through(logits, book, tau=1.0):
    """Эмбеддинг кода: вперёд жёсткий argmax, назад градиент мягкого среднего.

    ЗАЧЕМ. Через argmax градиент не течёт, а обучать голову надо по ошибке
    ДЕЙСТВИЯ: K-6e показал, что кросс-энтропия по кодам ранжирует иначе, чем
    ошибка декодированного действия — декодер берёт сумму уровней, и промах в
    соседний код почти бесплатен.

    Возвращает (emb_st, hard_idx, probs).
    """
    p = torch.softmax(logits / tau, dim=-1)
    soft = p @ book.to(p.dtype)
    idx = logits.argmax(dim=-1)
    hard = book.to(p.dtype)[idx]
    return soft + (hard - soft).detach(), idx, p


class DepthRVQVLA:
    """Примесь к SmolVLABlockwiseAR. Собирается через `make_depth_rvq_class`,
    чтобы не импортировать third_party на уровне модуля (самопроверки должны
    работать без модели и без GPU)."""


def make_depth_rvq_class(base_cls):
    """Построить класс поверх SmolVLABlockwiseAR."""

    class _DepthRVQVLA(base_cls):

        # ---- настройка ----------------------------------------------------
        def init_progressive(self, exits=(12, 18, 24), d_code=512,
                             alpha_init=1.0, head_dtype=torch.float32,
                             lora_r=0, lora_targets=("q_proj", "k_proj",
                                                     "v_proj", "o_proj",
                                                     "gate_proj", "up_proj",
                                                     "down_proj"),
                             feedback=True):
            n_layers = int(self.config.vlm_config.text_config.num_hidden_layers)
            exits = tuple(int(e) for e in exits)
            if sorted(set(exits)) != list(exits):
                raise ValueError(f"выходы должны строго возрастать: {exits}")
            if exits[-1] != n_layers:
                raise ValueError(
                    f"последний выход обязан стоять на полной глубине "
                    f"{n_layers}, задано {exits[-1]}")
            if exits[0] < 1:
                raise ValueError(f"выход {exits[0]} меньше первого слоя")
            self.exits = exits
            self.n_levels = len(exits)
            self.use_feedback = bool(feedback)

            # ЗАМОРОЗКА ВСЕГО ДО СОЗДАНИЯ НОВЫХ МОДУЛЕЙ. eval() не выключает
            # градиенты, а inject_lora замораживает только те Linear, которые
            # сама обернула. Без этой строки у 2.2B параметров остаётся
            # requires_grad=True: autograd напрасно копит промежуточные
            # значения, а случайный `model.parameters()` в оптимизаторе начал
            # бы учить всю модель. На V100 это отказ по памяти в лучшем случае
            # и молчаливое переобучение бэкбона в худшем.
            for p in self.parameters():
                p.requires_grad_(False)

            d_model = self.action_lm_head.in_features
            V = self.action_lm_head.out_features
            # ГОЛОВЫ ИНИЦИАЛИЗИРУЮТСЯ ИЗ action_lm_head: официальная BAR
            # применяет одну и ту же голову ко всем трём блокам, поэтому это
            # не эвристика, а точка, из которой модель начинает как исходная.
            self.prog_heads = nn.ModuleList()
            for _ in range(self.n_levels):
                h = nn.Linear(d_model, V, bias=self.action_lm_head.bias is not None,
                              dtype=head_dtype)
                with torch.no_grad():
                    h.weight.copy_(self.action_lm_head.weight.to(head_dtype))
                    if h.bias is not None:
                        h.bias.copy_(self.action_lm_head.bias.to(head_dtype))
                self.prog_heads.append(h)
            self.prog_heads = self.prog_heads.to(self.action_lm_head.weight.device)

            self.prog_feedback = nn.ModuleList([
                CodeFeedback(d_code, d_model, alpha_init, head_dtype)
                for _ in range(self.n_levels - 1)
            ]).to(self.action_lm_head.weight.device)

            self.lora_wrapped = []
            if lora_r > 0:
                self.lora_wrapped = inject_lora(
                    self.action_expert, lora_r, lora_targets, dtype=head_dtype)
                if not self.lora_wrapped:
                    raise RuntimeError(
                        f"LoRA не обернула ни одного слоя по именам "
                        f"{lora_targets} — обучение шло бы вхолостую")
            return self

        def progressive_parameters(self):
            ps = list(self.prog_heads.parameters())
            ps += list(self.prog_feedback.parameters())
            for m in self.action_expert.modules():
                if isinstance(m, LoRALinear):
                    ps += [m.A, m.B]
            return ps

        def trainable_report(self):
            """Что именно обучается. Печатать перед каждым запуском обучения:
            это единственный способ заметить, что бэкбон разморожен."""
            groups = {}
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                key = ("prog_heads" if name.startswith("prog_heads") else
                       "prog_feedback" if name.startswith("prog_feedback") else
                       "lora" if name.endswith((".A", ".B")) else
                       "ПОСТОРОННЕЕ:" + name)
                groups[key] = groups.get(key, 0) + p.numel()
            stray = [k for k in groups if k.startswith("ПОСТОРОННЕЕ")]
            if stray:
                raise RuntimeError(
                    f"обучаемыми оказались параметры вне новых веток: "
                    f"{stray[:5]} — бэкбон не заморожен")
            return groups

        # ---- сегментированный проход ---------------------------------------
        def run_progressive(self, *, vlm_inputs_embeds, attention_mask,
                            position_ids, books=None, mode="full",
                            teacher_codes=None, tau=1.0):
            """Один проход с выпуском уровней на глубинах self.exits.

            `books`: (n_levels, V, d_code) — вклады кодбуков в латенту, те же,
            что суммирует официальный декодер. Нужны только при feedback.
            `teacher_codes`: (n_levels, B, n) истинные коды; если заданы, в
            поток внедряются ОНИ, а не собственные предсказания.

            ВОЗВРАЩАЕТ ПРЕДСКАЗАННЫЕ И ВНЕДРЁННЫЕ КОДЫ ОТДЕЛЬНО. При teacher
            forcing это разные вещи, и склеивать их в одно поле опасно:
            обучающий код посчитал бы точность по учительским кодам, собрал бы
            действие из них же и не заметил бы, что ранние головы не учатся.
            `pred_codes` — то, что выдала модель; `injected_codes` — то, что
            ушло в следующий сегмент.
            """
            if mode not in ("fast", "medium", "full"):
                raise ValueError(f"неизвестный режим {mode}")
            stop_level = {"fast": 0, "medium": 1, "full": self.n_levels - 1}[mode]
            if stop_level >= self.n_levels:
                stop_level = self.n_levels - 1

            B = vlm_inputs_embeds.shape[0]
            dev, dt = vlm_inputs_embeds.device, vlm_inputs_embeds.dtype
            n = self.block_size
            bos = self.bos_embedding.expand(B, n, -1).to(device=dev, dtype=dt)
            action_hidden = bos
            vlm_hidden = vlm_inputs_embeds
            vlm_len, act_len = vlm_inputs_embeds.shape[1], n

            mask4d = self._build_joint_attention_mask_blockwise_ar(
                attention_mask=attention_mask, vlm_seq_len=vlm_len,
                action_seq_len=act_len, device=dev,
                action_key_mask=torch.ones((B, act_len), device=dev,
                                           dtype=torch.long))

            n_layers = int(self.config.vlm_config.text_config.num_hidden_layers)
            logits, pred_codes, inj_codes, embs = [], [], [], []
            layers_run = 0
            for layer_idx in range(n_layers):
                vlm_hidden, action_hidden = self._shared_attention_forward(
                    vlm_hidden_states=vlm_hidden,
                    action_hidden_states=action_hidden,
                    layer_idx=layer_idx, attention_mask=mask4d,
                    position_ids=position_ids, past_key_values=None,
                    use_cache=False, cache_position=None)
                layers_run += 1
                depth = layer_idx + 1
                if depth not in self.exits:
                    continue
                g = self.exits.index(depth)
                # НОРМА ПЕРЕД ГОЛОВОЙ на КАЖДОМ выходе: bar.py:1246 применяет
                # action_expert.norm перед action_lm_head, и головы взяты из
                # неё, поэтому вход обязан быть той же природы.
                normed = self.action_expert.norm(action_hidden)
                hd = self.prog_heads[g].weight.dtype
                lg = self.prog_heads[g](normed.to(hd))
                logits.append(lg)
                pred_codes.append(lg.argmax(-1))
                if g >= stop_level:
                    break
                if books is None:
                    raise ValueError("для продолжения нужны books")
                # ЭМБЕДДИНГ ВСЕГДА ЧЕРЕЗ straight-through, даже при teacher
                # forcing: жёсткий индекс подменяется учительским, но мягкая
                # ветвь остаётся своей, иначе к ранней голове не придёт
                # градиент от ошибки поздних уровней.
                emb, idx, _ = straight_through(lg, books[g], tau)
                if teacher_codes is not None:
                    t_idx = teacher_codes[g].to(dev)
                    t_hard = books[g].to(emb.dtype)[t_idx]
                    emb = emb + (t_hard - emb).detach()
                    idx = t_idx
                inj_codes.append(idx)
                embs.append(emb)
                if self.use_feedback:
                    action_hidden = self.prog_feedback[g](action_hidden, emb)
            return dict(logits=logits, pred_codes=pred_codes,
                        injected_codes=inj_codes, embeddings=embs,
                        layers_run=layers_run)

        # ---- удобная обёртка над generate ----------------------------------
        @torch.no_grad()
        def generate_progressive(self, *, mode="full", books=None,
                                 position_offset=0, initial_position_shift=1,
                                 **kw):
            B, _, vlm_embeds, _ = self._build_vlm_inputs_embeds(
                input_ids=kw.get("input_ids"),
                inputs_embeds=kw.get("inputs_embeds"),
                pixel_values=kw.get("pixel_values"),
                pixel_attention_mask=kw.get("pixel_attention_mask"),
                image_hidden_states=kw.get("image_hidden_states"))
            dev = vlm_embeds.device
            vlm_len = vlm_embeds.shape[1]
            act_pos = self._build_action_pos_ids_strided(
                batch_size=B, base_pos=vlm_len, action_seq_len=self.block_size,
                device=dev, position_offset=position_offset)
            pos = self._build_joint_position_ids(
                batch_size=B, vlm_seq_len=vlm_len, action_pos_ids=act_pos,
                device=dev)
            return self.run_progressive(
                vlm_inputs_embeds=vlm_embeds,
                attention_mask=kw.get("attention_mask"),
                position_ids=pos, books=books, mode=mode)

    return _DepthRVQVLA
