"""
Unified VLA Training Script

Supports four training strategies:
- ar: Auto-Regressive (Identity) - uses AutoModelForImageTextToText
- bar: Blockwise Auto-Regressive - uses SmolVLABlockwiseAR
- pd: Parallel Decoding - uses SmolVLAParallelDecoding
- ki: Knowledge Isolation - uses SmolVLAKnowledgeIsolation

Usage:
    python train_vla.py --config config/train/ar.yaml
    python train_vla.py --config config/train/bar.yaml
    python train_vla.py --config config/train/pd.yaml
    python train_vla.py --config config/train/ki.yaml
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/dzb/github/libero")

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model

import actioncodec  # noqa: F401 - ensure actioncodec is registered
from utils import (
    VLADataCollator,  # noqa: F401
    VLDataCollator,  # noqa: F401
    create_dataloaders,
    dict_apply,
    get_cfg,
    get_obj_from_str,
)
from utils.vla_tokenizer import VisionLanguageActionProcessor


class VLATrainingWrapper(L.LightningModule):
    """
    Unified VLA Training Wrapper supporting multiple strategies.

    Strategies:
    - ar: Auto-Regressive - uses prompt mask-based labels
    - bar: Blockwise Auto-Regressive - encodes action tokens with random position offset
    - pd: Parallel Decoding - encodes action tokens, standard forward
    - ki: Knowledge Isolation - uses action as labels directly
    """

    VALID_STRATEGIES = {"ar", "bar", "pd", "ki"}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.strategy = cfg.get("strategy", "ar")

        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {self.strategy}. Must be one of {self.VALID_STRATEGIES}"
            )

        self.processor = self._init_processor(cfg)
        self.model = self._init_model(cfg)

        # may be different across different modes, 'assistant\n' for most of the cases
        self._answer_start = self.processor.tokenizer(cfg.MODEL.answer_start_token)["input_ids"]

    def _init_processor(self, cfg) -> VisionLanguageActionProcessor:
        action_processor_cls = get_obj_from_str(cfg.MODEL.action_processor._target_)
        vl_processor_cls = get_obj_from_str(cfg.MODEL.vl_processor._target_)

        action_processor = action_processor_cls.from_pretrained(**cfg.MODEL.action_processor.kwargs)
        try:
            action_processor = action_processor.to(self.device).eval()
        except Exception:
            pass
        vl_processor = vl_processor_cls.from_pretrained(**cfg.MODEL.vl_processor.kwargs)
        vl_processor.image_processor.do_image_splitting = False
        vl_processor.image_processor.do_resize = False

        return VisionLanguageActionProcessor(
            action_processor, vl_processor, **cfg.MODEL.vla_processor.kwargs
        )

    def _init_model(self, cfg):
        model_cls = get_obj_from_str(cfg.MODEL.vlm._target_)
        model = model_cls.from_pretrained(**cfg.MODEL.vlm.kwargs).train()

        # AR strategy requires resizing token embeddings
        if self.strategy == "ar":
            model.resize_token_embeddings(len(self.processor.tokenizer))

        if cfg.TRAINING.use_lora:
            peft_config = OmegaConf.to_container(cfg.TRAINING.lora_kwargs, resolve=True)
            peft_config = LoraConfig(**peft_config)
            model = get_peft_model(model, peft_config)

        return model

    # =========================================================================
    # Training Steps (Strategy-specific)
    # =========================================================================

    def training_step(self, batch, batch_idx=None):
        """Dispatch to strategy-specific training step."""
        return getattr(self, f"_training_step_{self.strategy}")(batch, batch_idx)

    def _training_step_ar(self, batch, batch_idx=None):
        """AR strategy: Prompt mask-based labels, next-token prediction."""
        del batch["action"]
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)

        # Create labels with prompt masking
        labels = processed_batch["input_ids"].clone()
        answer_start_tensor = torch.tensor(self._answer_start, device=self.device, dtype=torch.long)
        unfolded_input_ids = labels.unfold(dimension=1, size=len(self._answer_start), step=1)
        matches = (unfolded_input_ids == answer_start_tensor).all(dim=2)
        match_start_indices = torch.argmax(matches.int(), dim=1)
        match_end_indices = match_start_indices + len(self._answer_start) - 1
        sequence_indices = torch.arange(labels.shape[1], device=self.device).expand(
            labels.shape[0], -1
        )
        prompt_mask = sequence_indices <= match_end_indices.unsqueeze(1)
        padding_mask = processed_batch["attention_mask"] == 0
        labels[prompt_mask | padding_mask] = -100

        outputs = self.model(
            **processed_batch,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            labels=labels,
        )
        loss = outputs.loss

        # Logging
        with torch.no_grad():
            logits_for_loss = outputs.logits[:, :-1, :]
            labels_for_loss = labels[:, 1:]
            active_loss_mask = labels_for_loss.reshape(-1) != -100
            active_logits = logits_for_loss.reshape(-1, logits_for_loss.size(-1))[active_loss_mask]
            active_labels = labels_for_loss.reshape(-1)[active_loss_mask]
            if active_labels.numel() > 0:
                acc = (
                    (active_labels.unsqueeze(1) == active_logits.topk(1, dim=-1).indices)
                    .float()
                    .mean()
                )
            else:
                acc = torch.zeros(1, device=self.device)

        self.log(
            "acc/acc@1", acc.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _training_step_bar(self, batch, batch_idx=None):
        """BAR strategy: Encodes action tokens, random_position_offset=True."""
        self.processor.action_processor.to(self.device, torch.float32)
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)

        action = torch.stack(batch["action"], dim=0).to(device=self.device, dtype=torch.float32)
        del batch["action"]

        action_tokens = torch.tensor(
            self.processor.action_processor.encode(action), device=self.device, dtype=torch.long
        )

        outputs = self.model(
            **processed_batch,
            return_dict=True,
            labels=action_tokens,
            random_position_offset=True,
        )
        loss = outputs.loss

        with torch.no_grad():
            acc = (outputs.logits.argmax(dim=-1) == action_tokens).float().mean()

        self.log(
            "acc/acc@1", acc.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _training_step_pd(self, batch, batch_idx=None):
        """PD strategy: Encodes action tokens, standard forward."""
        self.processor.action_processor.to(self.device, torch.float32)
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)

        action = torch.stack(batch["action"], dim=0).to(device=self.device, dtype=torch.float32)
        del batch["action"]

        action_tokens = torch.tensor(
            self.processor.action_processor.encode(action), device=self.device, dtype=torch.long
        )

        outputs = self.model(
            **processed_batch,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            labels=action_tokens,
        )
        loss = outputs.loss

        with torch.no_grad():
            acc = (outputs.logits.argmax(dim=-1) == action_tokens).float().mean()

        self.log(
            "acc/acc@1", acc.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _training_step_ki(self, batch, batch_idx=None):
        """KI strategy: Uses action as labels directly."""
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)

        action = torch.stack(batch["action"], dim=0).to(device=self.device, dtype=self.dtype)
        del batch["action"]

        outputs = self.model(
            **processed_batch,
            use_cache=True,
            return_dict=True,
            labels=action,
        )
        loss = outputs.loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # =========================================================================
    # Validation Steps (Strategy-specific)
    # =========================================================================

    def validation_step(self, batch, batch_idx=None):
        """Dispatch to strategy-specific validation step."""
        return getattr(self, f"_validation_step_{self.strategy}")(batch, batch_idx)

    def _validation_step_ar(self, batch, batch_idx=None):
        """AR strategy: Uses model.generate(), decodes with processor."""
        action = torch.stack(batch["action"], dim=0).to(self.device, dtype=torch.float32)
        del batch["action"]
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)

        with torch.no_grad():
            outputs = self.model.generate(**processed_batch, max_new_tokens=200)

        pred_action = self.processor.batch_decode(
            outputs,
            decode_actions=True,
            actions_only=True,
            skip_special_tokens=False,
            action_processor_kwargs={"embodiment_ids": 0},
        )
        pred_action = torch.stack(
            [
                torch.zeros_like(action[0])
                if each is None
                else torch.tensor(each, device=action[0].device, dtype=torch.float32)
                for each in pred_action
            ],
            dim=0,
        ).to(self.device, dtype=torch.float32)

        loss = (action - pred_action).abs().mean()
        self.log("val/l1_dist", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def _validation_step_bar(self, batch, batch_idx=None):
        """BAR strategy: Uses model.generate() with position_offset=3."""
        self.processor.action_processor.to(self.device, torch.float32)
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)

        action = torch.stack(batch["action"], dim=0).to(device=self.device, dtype=torch.float32)
        del batch["action"]

        with torch.no_grad():
            pred_action_tokens = self.model.generate(
                **processed_batch, return_dict=True, position_offset=3
            )

        pred_action = torch.tensor(
            self.processor.action_processor.decode(pred_action_tokens.tolist())[0],
            device=self.device,
            dtype=torch.float32,
        )
        loss = (action - pred_action).abs().mean()
        self.log("val/l1_dist", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def _validation_step_pd(self, batch, batch_idx=None):
        """PD strategy: Forward pass + argmax."""
        self.processor.action_processor.to(self.device, torch.float32)
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)

        action = torch.stack(batch["action"], dim=0).to(device=self.device, dtype=torch.float32)
        del batch["action"]

        with torch.no_grad():
            outputs = self.model(
                **processed_batch,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            pred_action_tokens = outputs.logits.argmax(dim=-1)

        pred_action = torch.tensor(
            self.processor.action_processor.decode(pred_action_tokens.tolist())[0],
            device=self.device,
            dtype=torch.float32,
        )
        loss = (action - pred_action).abs().mean()
        self.log("val/l1_dist", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def _validation_step_ki(self, batch, batch_idx=None):
        """KI strategy: Uses model.sample() with sampling_steps=10."""
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)

        action = torch.stack(batch["action"], dim=0).to(device=self.device, dtype=self.dtype)
        del batch["action"]

        with torch.no_grad():
            pred_action = self.model.sample(
                **processed_batch,
                use_cache=True,
                return_dict=True,
                sampling_steps=10,
            )

        loss = (action - pred_action).abs().mean()
        self.log("val/l1_dist", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    # =========================================================================
    # Optimizer Configuration (Shared)
    # =========================================================================

    def configure_optimizers(self):
        optimizer = get_obj_from_str(self.cfg.TRAINING.optimizer._target_)(
            self.parameters(), **self.cfg.TRAINING.optimizer.kwargs
        )

        def lr_lambda(current_step: int):
            if current_step < self.cfg.TRAINING.warmup_steps:
                start_factor = 1e-6
                return start_factor + (1.0 - start_factor) * (
                    current_step / self.cfg.TRAINING.warmup_steps
                )
            else:
                decay_steps = self.cfg.TRAINING.training_steps - self.cfg.TRAINING.warmup_steps
                current_decay_step = current_step - self.cfg.TRAINING.warmup_steps
                if decay_steps <= 0:
                    return 1e-1
                end_factor = 1e-1
                progress = min(1.0, current_decay_step / decay_steps)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                return end_factor + (1.0 - end_factor) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    cfg = get_cfg()

    training_model = VLATrainingWrapper(cfg)
    train_dataloader, val_dataloader = create_dataloaders(
        cfg, training_model.processor, training_model.strategy
    )

    # Test mode: 10 steps, no logging, no checkpoints
    test_mode = cfg.get("test_mode", False)

    callbacks = []
    if not test_mode:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        ckpt_callback = ModelCheckpoint(
            dirpath=cfg.TRAINING.ckpt_dir,
            filename="{epoch}-{step}",
            save_top_k=-1,
            every_n_train_steps=5000,
        )
        callbacks.append(ckpt_callback)

    trainer = L.Trainer(
        accelerator="cuda",
        max_steps=10 if test_mode else cfg.TRAINING.training_steps,
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=False if test_mode else True,
        val_check_interval=cfg.TRAINING.get("val_check_interval", 1000),
        **cfg.TRAINING.trainer,
    )

    trainer.fit(training_model, train_dataloader, val_dataloader)
