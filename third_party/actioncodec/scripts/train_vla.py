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
from typing import Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/dzb/github/libero")

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader, IterableDataset
import shutil
from pathlib import Path

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


DEBUG_OVERFIT_STEPS = int(os.environ.get("DEBUG_OVERFIT_STEPS", "80"))
DEBUG_GRAD_MIN_NORM = 1e-5
DEBUG_FINAL_LOSS_RATIO = 0.20
DEBUG_FINAL_LOSS_MAX = 0.15
# Short VRAM-profile runs (e.g. 5–10 steps) only need Grad/FSDP gates.
DEBUG_OVERFIT_SKIP_PLATEAU = os.environ.get("DEBUG_OVERFIT_SKIP_PLATEAU", "").lower() in {
    "1",
    "true",
    "yes",
} or DEBUG_OVERFIT_STEPS < 20


class RollingDiskCheckpoint(Callback):
    """Disk-bounded checkpoints: only ``latest/``, ``latest_backup/``, ``best/``.

    Forbidden: unbounded ``epoch-step`` trees. Every periodic save atomically
    rotates ``latest`` → ``latest_backup`` and writes a fresh ``latest``, so
    on-disk footprint stays at most two rolling snapshots plus one best.
    """

    LATEST = "latest"
    BACKUP = "latest_backup"
    BEST = "best"
    STAGING = ".latest_staging"
    CKPT_NAME = "last.ckpt"

    def __init__(
        self,
        root: str | Path,
        *,
        every_n_train_steps: int = 1000,
        monitor: str | None = "val/l1_dist",
        mode: str = "min",
    ) -> None:
        super().__init__()
        if every_n_train_steps < 1:
            raise ValueError("every_n_train_steps must be >= 1")
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be min|max, got {mode!r}")
        self.root = Path(root).expanduser().resolve()
        self.every_n_train_steps = int(every_n_train_steps)
        self.monitor = monitor
        self.mode = mode
        self.best_score: float | None = None
        self._last_rolled_step = -1

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if trainer.is_global_zero:
            self.root.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if step <= 0 or step % self.every_n_train_steps != 0:
            return
        if step == self._last_rolled_step:
            return
        self._last_rolled_step = step
        self._atomic_roll(trainer, tag=f"step={step}")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.monitor is None or trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        score = float(metrics[self.monitor].detach().cpu())
        improved = (
            self.best_score is None
            or (self.mode == "min" and score < self.best_score)
            or (self.mode == "max" and score > self.best_score)
        )
        if not improved:
            return
        self.best_score = score
        self._save_best(trainer, score=score)

    def _atomic_roll(self, trainer: L.Trainer, *, tag: str) -> None:
        staging_dir = self.root / self.STAGING
        latest_dir = self.root / self.LATEST
        backup_dir = self.root / self.BACKUP
        staging_path = staging_dir / self.CKPT_NAME

        if trainer.is_global_zero:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)

        trainer.strategy.barrier()
        trainer.save_checkpoint(str(staging_path))
        trainer.strategy.barrier()

        if not trainer.is_global_zero:
            return

        if latest_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            latest_dir.rename(backup_dir)
        staging_dir.rename(latest_dir)
        print(
            f"[RollingCheckpoint] rotated {tag} → {latest_dir} "
            f"(backup={backup_dir.exists()})",
            flush=True,
        )

    def _save_best(self, trainer: L.Trainer, *, score: float) -> None:
        best_dir = self.root / self.BEST
        staging = self.root / ".best_staging"
        ckpt_path = staging / self.CKPT_NAME

        if trainer.is_global_zero:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True, exist_ok=True)

        trainer.strategy.barrier()
        trainer.save_checkpoint(str(ckpt_path))
        trainer.strategy.barrier()

        if not trainer.is_global_zero:
            return

        if best_dir.exists():
            shutil.rmtree(best_dir)
        staging.rename(best_dir)
        print(
            f"[RollingCheckpoint] best/{self.CKPT_NAME} "
            f"monitor={self.monitor} score={score:.6g}",
            flush=True,
        )


def _clone_batch(value: Any) -> Any:
    """Clone a collated CPU batch so training_step may mutate it safely."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch(item) for item in value)
    return value


class _RepeatedBatchDataset(IterableDataset):
    """Yield byte-identical clones of one already-collated real batch."""

    def __init__(self, batch: dict[str, Any], repeats: int) -> None:
        super().__init__()
        self.batch = _clone_batch(batch)
        self.repeats = int(repeats)

    def __iter__(self):
        for _ in range(self.repeats):
            yield _clone_batch(self.batch)

    def __len__(self) -> int:
        return self.repeats


def make_debug_overfit_loader(
    source: DataLoader,
    *,
    optimizer_steps: int,
    accumulate_grad_batches: int,
) -> DataLoader:
    """Materialize one *real collated* batch and repeat it exactly.

    Caching happens after dataset I/O, image augmentation, tokenization, and
    collation. Consequently input_ids, pixel_values, and actions are identical
    on every micro-step, rather than merely referring to the same sample index.
    """
    if not bool(getattr(source, "_calvin_real_rgb_cache", False)):
        raise RuntimeError(
            "--debug-overfit is a production gate and refuses synthetic/Libero "
            "data. Set CALVIN_DATA_ROOT (or DATASET.root) to the completed "
            "clean task_D_D RGB cache."
        )
    try:
        real_batch = next(iter(source))
    except StopIteration as exc:
        raise RuntimeError("debug-overfit source dataloader is empty") from exc
    required = {"input_ids", "pixel_values", "action"}
    missing = required.difference(real_batch)
    if missing:
        raise RuntimeError(
            f"debug-overfit requires a real VLA batch with {sorted(required)}; "
            f"missing={sorted(missing)}"
        )
    print(
        "[Debug Overfit][Data] cached one real collated batch from "
        f"{getattr(source, '_calvin_dataset_root', '<unknown>')}",
        flush=True,
    )
    repeats = int(optimizer_steps) * max(1, int(accumulate_grad_batches))
    return DataLoader(
        _RepeatedBatchDataset(real_batch, repeats),
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )


def _shape_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_tree(item) for item in value]
    return type(value).__name__


def assert_bar_action_only_supervision(
    processed_batch: dict[str, Any],
    action_labels: torch.Tensor,
    *,
    token_budget: int,
    action_vocab_size: int,
) -> None:
    """Fail closed unless BAR supervision contains action codes only.

    BAR does not insert action codes into ``input_ids``. The action expert takes
    a separate ``action_labels=[B, token_budget]`` tensor. Any LM-backbone
    ``labels`` tensor (if present) must be fully masked with ``-100`` so text /
    image-placeholder tokens never contribute to the loss.
    """
    text_labels = processed_batch.get("labels", None)
    if text_labels is not None:
        if not torch.is_tensor(text_labels):
            raise AssertionError(
                "CRITICAL FAILURE: LLM labels present but not a tensor; "
                "BAR protocol forbids active text supervision."
            )
        if not torch.all(text_labels == -100):
            raise AssertionError(
                "CRITICAL FAILURE: LLM labels detected active tokens! "
                "BAR protocol forces text-supervision to be completely masked with -100."
            )
    if action_labels.ndim != 2 or action_labels.shape[-1] != int(token_budget):
        raise AssertionError(
            "CRITICAL: Violated BAR action_labels contract! "
            f"Expected [B, {token_budget}], got {tuple(action_labels.shape)}"
        )
    if action_labels.dtype != torch.long:
        raise AssertionError(f"action labels must be torch.long, got {action_labels.dtype}")
    if action_labels.numel() == 0:
        raise AssertionError("action labels are empty")
    lo = int(action_labels.min().item())
    hi = int(action_labels.max().item())
    if lo < 0 or hi >= int(action_vocab_size):
        raise AssertionError(
            f"action labels outside [0,{action_vocab_size}): min={lo}, max={hi}"
        )


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
        self.debug_overfit = bool(cfg.get("debug_overfit", False))
        self._debug_shapes_checked = False
        self._debug_grad_checked = False
        self._action_only_gate_done = False
        self._debug_optimizer_losses: list[float] = []

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
        action_items = batch.pop("action")
        processed_batch = dict_apply(lambda x: x.to(self.device), batch)
        action = torch.stack(action_items, dim=0).to(device=self.device, dtype=torch.float32)

        action_tokens = torch.tensor(
            self.processor.action_processor.encode(action), device=self.device, dtype=torch.long
        )

        # Production fail-closed: every FSDP rank inspects supervision before
        # any dirty gradient can be written (first working step, step == 0).
        if int(self.global_step) == 0 and not self._action_only_gate_done:
            assert_bar_action_only_supervision(
                processed_batch,
                action_tokens,
                token_budget=int(self.cfg.MODEL.action_processor.token_len),
                action_vocab_size=int(self.cfg.MODEL.action_processor.vocab_size),
            )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if self.global_rank == 0:
                print(
                    "[ActionOnlyGate] PASS step=0: LM labels fully masked or absent; "
                    f"action_labels={tuple(action_tokens.shape)} "
                    f"(token_budget={int(self.cfg.MODEL.action_processor.token_len)})",
                    flush=True,
                )
                if self.debug_overfit:
                    print(
                        "[Debug Overfit][Shapes] "
                        f"inputs={_shape_tree(processed_batch)} "
                        f"labels={_shape_tree(action_tokens)}",
                        flush=True,
                    )
                    print(
                        "[Debug Overfit][Mask Check] PASS: BAR action-only "
                        "supervision invariant held on step 0.",
                        flush=True,
                    )
            self._action_only_gate_done = True
            self._debug_shapes_checked = True

        outputs = self.model(
            **processed_batch,
            return_dict=True,
            labels=action_tokens,
            # Random offsets make exact single-batch convergence nondeterministic.
            random_position_offset=not self.debug_overfit,
        )
        loss = outputs.loss

        with torch.no_grad():
            acc = (outputs.logits.argmax(dim=-1) == action_tokens).float().mean()

        self.log(
            "acc/acc@1", acc.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Check unscaled gradients and synchronize debug logging.

        Lightning invokes this hook after mixed-precision unscale and before
        ``scaler.step(optimizer)``. This is the only reliable point to inspect
        true gradients (checking after zero_grad would produce a false failure).
        Every rank enters the barriers; rank 0 alone prints.
        """
        if not self.debug_overfit:
            return
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if not self._debug_grad_checked:
            selected_name = ""
            local_norm = torch.zeros((), device=self.device, dtype=torch.float32)
            fallback: tuple[str, torch.Tensor] | None = None
            for name, param in self.model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                norm = param.grad.detach().float().norm()
                if fallback is None:
                    fallback = (name, norm)
                if "action_lm_head" in name or name.endswith("lm_head.weight"):
                    selected_name, local_norm = name, norm
                    break
            if not selected_name and fallback is not None:
                selected_name, local_norm = fallback
            if not selected_name:
                raise AssertionError("CRITICAL: no trainable parameter has a gradient")

            # FSDP gradients are sharded. MAX catches a non-zero shard while
            # still rejecting a globally detached head.
            global_norm = local_norm.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(global_norm, op=dist.ReduceOp.MAX)
            value = float(global_norm.item())
            if self.global_rank == 0:
                print(
                    f"[Grad Check] Layer: {selected_name} | "
                    f"Gradient Norm = {value:.5f}",
                    flush=True,
                )
            if not np.isfinite(value) or value <= DEBUG_GRAD_MIN_NORM:
                raise AssertionError(
                    f"CRITICAL: gradient is detached/non-finite for {selected_name}: {value}"
                )
            self._debug_grad_checked = True

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def on_fit_start(self) -> None:
        if not self.debug_overfit:
            return
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if not isinstance(scaler, ShardedGradScaler):
            raise AssertionError(
                "debug-overfit requires torch.distributed.fsdp.ShardedGradScaler; "
                f"got {type(scaler).__name__}"
            )
        if self.global_rank == 0:
            print(
                "[Debug Overfit][FSDP] ShardedGradScaler active; "
                "all debug log points use dist.barrier().",
                flush=True,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if not self.debug_overfit:
            return
        # global_step changes only after an optimizer step. Record one scalar
        # per optimizer step, not every accumulation micro-step.
        if outputs is None:
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if not torch.is_tensor(loss):
            return
        step = int(self.global_step)
        if step <= len(self._debug_optimizer_losses):
            return
        value = float(loss.detach().float().cpu().item())
        self._debug_optimizer_losses.append(value)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.global_rank == 0:
            print(
                f"[Debug Overfit] optimizer_step={step:02d}/{DEBUG_OVERFIT_STEPS} "
                f"loss={value:.8f}",
                flush=True,
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def on_train_end(self) -> None:
        if not self.debug_overfit:
            return
        losses = self._debug_optimizer_losses
        if len(losses) < DEBUG_OVERFIT_STEPS:
            raise AssertionError(
                f"debug-overfit recorded {len(losses)} optimizer losses; "
                f"expected {DEBUG_OVERFIT_STEPS}"
            )
        first, final = losses[0], losses[-1]
        increases = sum(b > a + max(1e-6, abs(a) * 1e-3) for a, b in zip(losses, losses[1:]))
        if DEBUG_OVERFIT_SKIP_PLATEAU:
            if self.global_rank == 0:
                print(
                    f"[Debug Overfit] VRAM-profile complete first={first:.8f} "
                    f"final={final:.8f} steps={len(losses)} "
                    f"(plateau assert skipped)",
                    flush=True,
                )
            return
        if (
            not np.isfinite(final)
            or final > first * DEBUG_FINAL_LOSS_RATIO
            or final > DEBUG_FINAL_LOSS_MAX
        ):
            raise AssertionError(
                f"debug-overfit plateau: first={first:.6g}, final={final:.6g}; "
                f"required final <= {DEBUG_FINAL_LOSS_RATIO:.0%} of initial "
                f"and <= {DEBUG_FINAL_LOSS_MAX}"
            )
        # Treat sub-0.1% movement as fp16 noise; all material steps must descend.
        if increases:
            raise AssertionError(
                f"debug-overfit is not monotone within fp16 tolerance: "
                f"{increases} material loss increases"
            )
        if self.global_rank == 0:
            print(
                f"[Debug Overfit] PASS first={first:.8f} final={final:.8f} "
                f"ratio={final / first:.4f} increases={increases}/{len(losses)-1}",
                flush=True,
            )

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
    debug_overfit = bool(cfg.get("debug_overfit", False))
    if debug_overfit and training_model.strategy != "bar":
        raise ValueError("--debug-overfit reliability gate is frozen for strategy=bar")

    trainer_cfg = OmegaConf.to_container(cfg.TRAINING.trainer, resolve=True)
    accumulate = int(trainer_cfg.get("accumulate_grad_batches", 1))
    if debug_overfit:
        train_dataloader = make_debug_overfit_loader(
            train_dataloader,
            optimizer_steps=DEBUG_OVERFIT_STEPS,
            accumulate_grad_batches=accumulate,
        )

    callbacks = []
    if not test_mode and not debug_overfit:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        ckpt_interval = int(cfg.TRAINING.get("checkpoint_interval", 1000))
        # Hard fail-closed disk contract: only latest/ + latest_backup/ + best/.
        # Never emit unbounded epoch-step trees under ckpt_dir.
        callbacks.append(
            RollingDiskCheckpoint(
                cfg.TRAINING.ckpt_dir,
                every_n_train_steps=ckpt_interval,
                monitor=cfg.TRAINING.get("checkpoint_monitor", "val/l1_dist"),
                mode=str(cfg.TRAINING.get("checkpoint_mode", "min")),
            )
        )

    trainer_kwargs = dict(trainer_cfg)
    # Production FSDP + 16-mixed needs ShardedGradScaler via FSDPPrecision
    # (same contract as --debug-overfit). Explicit scaler kwargs are broken on
    # Lightning 2.6.x; let the plugin own precision.
    precision = str(trainer_kwargs.get("precision", ""))
    strategy = str(trainer_kwargs.get("strategy", ""))
    use_fsdp_mixed = (
        not test_mode
        and precision in {"16-mixed", "16"}
        and "fsdp" in strategy.lower()
    )
    if use_fsdp_mixed:
        try:
            from lightning.pytorch.plugins.precision import FSDPPrecision
        except ImportError:
            from lightning.pytorch.plugins import FSDPPrecision
        trainer_kwargs["plugins"] = [FSDPPrecision("16-mixed")]
        trainer_kwargs.pop("precision", None)
        trainer_kwargs.pop("gradient_clip_val", None)
        trainer_kwargs.pop("gradient_clip_algorithm", None)

    if debug_overfit:
        trainer_kwargs.update(
            {
                "limit_val_batches": 0,
                "num_sanity_val_steps": 0,
                "use_distributed_sampler": False,
                "deterministic": True,
                "log_every_n_steps": 1,
            }
        )
        if not use_fsdp_mixed:
            raise RuntimeError(
                "--debug-overfit requires FSDP + fp16 mixed precision so the "
                "ShardedGradScaler/barrier gate is real; got "
                f"strategy={strategy!r}, precision={precision!r}"
            )

    trainer = L.Trainer(
        accelerator="cuda",
        max_steps=(
            DEBUG_OVERFIT_STEPS
            if debug_overfit
            else (10 if test_mode else cfg.TRAINING.training_steps)
        ),
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=False if (test_mode or debug_overfit) else True,
        val_check_interval=cfg.TRAINING.get("val_check_interval", 1000),
        **trainer_kwargs,
    )

    trainer.fit(training_model, train_dataloader, val_dataloader)
