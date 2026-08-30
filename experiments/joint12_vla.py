"""Joint-12: 12-слойный Fast-выход с СОВМЕСТНЫМ обучением первых слоёв.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ K-8b. Там обучались головы и LoRA поверх замороженного
представления, и четыре конфигурации сошлись к 26% согласия против 87% у полной
глубины. Здесь первые двенадцать слоёв ОБЕИХ башен размораживаются целиком:
проверяется не «лежит ли решение на слое 12», а «можно ли его туда перенести».

ОДИН ВЫХОД И НИЧЕГО БОЛЬШЕ. Ни Medium/Full, ни обратного внедрения кода, ни
уровней 1-2, ни нового токенизатора. K-8b показал, что подключение поздних
целей обваливает согласие ранней головы с 26.5% до 17.6% — поэтому сначала
доказывается перенос, и только потом что-либо добавляется.

ЧТО ПРОВЕРЕНО ПО КОДУ (bar.py):
  * `_run_action_sequence` держит один цикл, `_shared_attention_forward`
    продвигает башню VLM и эксперта вместе (1236-1246) — остановка на 12
    экономит обе;
  * этот метод НЕ вызывает `layer.forward()`, а дёргает `input_layernorm`,
    `self_attn.q_proj`, `mlp` вручную (987-1061); считать исполненные слои надо
    по `input_layernorm`;
  * пути к башням: `self.vlm.text_model.layers`, `self.action_expert.layers`;
  * перед головой стоит `action_expert.norm` (1246-1247);
  * блок 0 — `bos_len=block_size` при пустой истории (1288-1296).

БЕЗОПАСНЫЙ СТАРТ. `fast_head` — копия `action_lm_head`, поэтому при depth=24 и
до первого шага модель обязана побитово воспроизвести первый блок официальной
BAR. Проверяется в k9b, а не предполагается.

ГРАДИЕНТНЫЙ ЧЕКПОЙНТИНГ СВОЙ. Автоматический из Hugging Face здесь не
применится: слои вызываются не через свой forward. Поэтому оборачивается
каждый шаг общего внимания, и эффект обязан быть ИЗМЕРЕН, а не предположен.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def make_joint12_class(base_cls):
    """Построить класс поверх SmolVLABlockwiseAR. third_party не изменяется."""

    class _Joint12VLA(base_cls):

        # ---- настройка -----------------------------------------------------
        def init_joint_fast(self, depth=12, head_dtype=None, grad_ckpt=False):
            n_layers = int(self.config.vlm_config.text_config.num_hidden_layers)
            if not 1 <= depth <= n_layers:
                raise ValueError(f"глубина {depth} вне 1..{n_layers}")
            self.fast_depth = depth
            self.grad_ckpt = bool(grad_ckpt)

            # ЗАМОРОЗКА ВСЕГО ДО СОЗДАНИЯ НОВОГО. eval() градиенты не
            # выключает, и без этой строки у 2.2B остаётся requires_grad=True.
            for p in self.parameters():
                p.requires_grad_(False)

            hd = head_dtype or self.action_lm_head.weight.dtype
            self.fast_head = nn.Linear(
                self.action_lm_head.in_features,
                self.action_lm_head.out_features,
                bias=self.action_lm_head.bias is not None, dtype=hd,
                device=self.action_lm_head.weight.device)
            with torch.no_grad():
                self.fast_head.weight.copy_(self.action_lm_head.weight.to(hd))
                if self.fast_head.bias is not None:
                    self.fast_head.bias.copy_(self.action_lm_head.bias.to(hd))

            # БЕЛЫЙ СПИСОК. Всё, что вне него, обязано остаться замороженным, и
            # trainable_report это проверяет отказом, а не предупреждением.
            #
            # НА ПОСЛЕДНЕМ ИСПОЛНЯЕМОМ СЛОЕ БАШНЯ VLM ЖИВА ЛИШЬ ЧАСТИЧНО.
            # `_shared_attention_forward` возвращает пару, но выход башни после
            # слоя depth-1 никуда не идёт: дальше берётся только поток
            # действий. Запросы VLM дают `attn_output[:, :vlm_seq_len]`, то
            # есть исключительно вывод башни, а действия смотрят на КЛЮЧИ и
            # ЗНАЧЕНИЯ. Поэтому на последнем слое влияют только
            # input_layernorm, k_proj и v_proj, а q_proj, o_proj,
            # post_attention_layernorm и mlp не имеют пути к потере вообще.
            # Измерено: норма градиента у vlm[depth-1].q_proj ровно нулевая при
            # ненулевых у всех прочих. Держать их обучаемыми — это около 60 млн
            # параметров впустую вместе с состояниями Adam.
            last = depth - 1
            self.trainable_prefixes = tuple(
                [f"vlm.text_model.layers.{i}." for i in range(last)]
                + [f"vlm.text_model.layers.{last}.input_layernorm.",
                   f"vlm.text_model.layers.{last}.self_attn.k_proj.",
                   f"vlm.text_model.layers.{last}.self_attn.v_proj."]
                + [f"action_expert.layers.{i}." for i in range(depth)]
                + ["action_expert.norm.", "bos_embedding", "fast_head."])
            for name, p in self.named_parameters():
                if any(name.startswith(pf) or name == pf.rstrip(".")
                       for pf in self.trainable_prefixes):
                    p.requires_grad_(True)
            return self

        def trainable_report(self):
            """Точный отчёт, что обучается. Печатать перед каждым запуском."""
            groups, stray = {}, []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if not any(name.startswith(pf) or name == pf.rstrip(".")
                           for pf in self.trainable_prefixes):
                    stray.append(name)
                    continue
                if name.startswith(f"vlm.text_model.layers.{self.fast_depth - 1}."):
                    k = "vlm.layers[%d] (только k/v)" % (self.fast_depth - 1)
                elif name.startswith("vlm."):
                    k = "vlm.layers[0:%d]" % (self.fast_depth - 1)
                elif name.startswith("action_expert.layers"):
                    k = "expert.layers[0:%d]" % self.fast_depth
                elif name.startswith("action_expert.norm"):
                    k = "expert.norm"
                elif name.startswith("fast_head"):
                    k = "fast_head"
                else:
                    k = "bos_embedding"
                groups[k] = groups.get(k, 0) + p.numel()
            if stray:
                raise RuntimeError(
                    f"обучаемыми оказались параметры вне белого списка: "
                    f"{stray[:5]} (всего {len(stray)})")
            # Обратная проверка: всё, что ДОЛЖНО быть заморожено, заморожено.
            for name, p in self.named_parameters():
                deep = (".layers." in name and any(
                    name.startswith(f"{t}.layers.{i}.")
                    for t in ("vlm.text_model", "action_expert")
                    for i in range(self.fast_depth,
                                   int(self.config.vlm_config.text_config
                                       .num_hidden_layers))))
                if deep and p.requires_grad:
                    raise RuntimeError(f"глубокий слой не заморожен: {name}")
            return groups

        def trainable_parameters(self):
            return [p for p in self.parameters() if p.requires_grad]

        def to_fp32_trainable(self):
            """Перевести ОБУЧАЕМЫЕ веса в fp32, замороженные оставить в fp16.

            ЗАЧЕМ. Модель грузится в fp16, и AdamW на fp16-параметрах держит
            состояния тоже в fp16. При lr=1e-5 и весах порядка 1e-2
            относительный шаг около 1e-3 — ровно на границе разрешения fp16,
            то есть часть обновлений просто не представима. Проверка «веса
            изменились после шага» это не поймает: первый шаг Adam по величине
            близок к lr и представим, а последующие мелкие — уже нет.

            Правильный режим: обучаемые веса и состояния оптимизатора в fp32,
            прямой проход под autocast fp16, масштабирование градиента через
            GradScaler. Замороженная часть остаётся в fp16 и памяти не ест.

            Вызывать ПОСЛЕ проверки структурного тождества: вне autocast
            fp32-слой на fp16-входе упадёт по типам.
            """
            n = 0
            for p in self.parameters():
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.float()
                    n += 1
            return n

        def memory_estimate(self):
            """Сколько потребует конфигурация. Печатать ДО длинного запуска."""
            tr = [p for p in self.parameters() if p.requires_grad]
            fr = [p for p in self.parameters() if not p.requires_grad]
            n_tr = sum(p.numel() for p in tr)
            n_fr = sum(p.numel() for p in fr)
            bt = 4 if (tr and tr[0].dtype == torch.float32) else 2
            g = 2 ** 30
            return dict(
                trainable=n_tr, frozen=n_fr,
                weights_trainable_gib=n_tr * bt / g,
                weights_frozen_gib=n_fr * 2 / g,
                grads_gib=n_tr * bt / g,
                adam_states_gib=n_tr * 8 / g,      # exp_avg + exp_avg_sq, fp32
                total_static_gib=(n_tr * bt * 2 + n_fr * 2 + n_tr * 8) / g)

        # ---- проход --------------------------------------------------------
        def forward_joint_fast(self, *, vlm_inputs_embeds, attention_mask,
                               position_ids, depth=None):
            """Блок 0 на заданной глубине. Ни истории, ни поздних уровней."""
            d = int(depth or self.fast_depth)
            B = vlm_inputs_embeds.shape[0]
            dev, dt = vlm_inputs_embeds.device, vlm_inputs_embeds.dtype
            n = self.block_size
            action_hidden = self.bos_embedding.expand(B, n, -1).to(dev, dt)
            vlm_hidden = vlm_inputs_embeds
            mask4d = self._build_joint_attention_mask_blockwise_ar(
                attention_mask=attention_mask,
                vlm_seq_len=vlm_inputs_embeds.shape[1], action_seq_len=n,
                device=dev,
                action_key_mask=torch.ones((B, n), device=dev, dtype=torch.long))

            for layer_idx in range(d):
                if self.grad_ckpt and self.training:
                    def step(vh, ah, li=layer_idx):
                        return self._shared_attention_forward(
                            vlm_hidden_states=vh, action_hidden_states=ah,
                            layer_idx=li, attention_mask=mask4d,
                            position_ids=position_ids, past_key_values=None,
                            use_cache=False, cache_position=None)
                    vlm_hidden, action_hidden = checkpoint(
                        step, vlm_hidden, action_hidden, use_reentrant=False)
                else:
                    vlm_hidden, action_hidden = self._shared_attention_forward(
                        vlm_hidden_states=vlm_hidden,
                        action_hidden_states=action_hidden,
                        layer_idx=layer_idx, attention_mask=mask4d,
                        position_ids=position_ids, past_key_values=None,
                        use_cache=False, cache_position=None)

            normed = self.action_expert.norm(action_hidden)
            hd = self.fast_head.weight.dtype
            logits = self.fast_head(normed.to(hd))
            return dict(logits=logits, pred_codes=logits.argmax(-1),
                        layers_run=d)

        def build_inputs(self, *, position_offset, **kw):
            """Общая подготовка префикса и позиций — та же, что у generate."""
            B, _, vemb, _ = self._build_vlm_inputs_embeds(
                input_ids=kw.get("input_ids"),
                inputs_embeds=kw.get("inputs_embeds"),
                pixel_values=kw.get("pixel_values"),
                pixel_attention_mask=kw.get("pixel_attention_mask"),
                image_hidden_states=kw.get("image_hidden_states"))
            apos = self._build_action_pos_ids_strided(
                batch_size=B, base_pos=vemb.shape[1],
                action_seq_len=self.block_size, device=vemb.device,
                position_offset=position_offset)
            pos = self._build_joint_position_ids(
                batch_size=B, vlm_seq_len=vemb.shape[1], action_pos_ids=apos,
                device=vemb.device)
            return vemb, pos

    return _Joint12VLA


def kd_loss(student_logits, teacher_logits, temperature=2.0):
    """Дистилляция: KL(учитель || ученик) с масштабом T^2.

    T^2 нужен, чтобы величина градиента не зависела от температуры — при
    делении логитов на T градиенты уменьшаются как 1/T^2.
    """
    import torch.nn.functional as F
    T = temperature
    lp_s = F.log_softmax(student_logits.float() / T, dim=-1)
    p_t = F.softmax(teacher_logits.float() / T, dim=-1)
    return (T * T) * F.kl_div(lp_s, p_t, reduction="batchmean") / lp_s.shape[1]
