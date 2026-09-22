#!/usr/bin/env python3
"""FSDP memory probe for SmolVLM2-2.2B on dual Tesla V100 (sm_70).

Closes the Block-4 risk early: measure peak VRAM under the frozen CALVIN
contract (fp16 16-mixed, SDPA, FSDP FULL_SHARD, micro-batch 1) **without**
the actions cache.

Important: a disconnected dummy ``loss = ones()`` does **not** exercise
activations and under-reports VRAM. This probe runs a real
forward → GradScaler → backward → AdamW step on fake vision/text batches.

Launch on ccmplanner (both GPUs must be visible)::

    export PYTHONUNBUFFERED=1
    torchrun --standalone --nproc_per_node=2 \\
      calvin_hicora/scripts/vla_memory_probe.py \\
      --steps 200 --output-json outputs/vla_fsdp_memory_probe.json

If the container only mounts GPU 0, restart it with both devices or run on
the host env that sees ``nvidia-smi`` GPUs 0 and 1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def configure_calvin_processor(processor: Any, *, image_size: int) -> Any:
    """Force CALVIN-like single-frame vision (no HF default image splitting)."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("processor.image_processor is missing")
    # Default SmolVLM splitting explodes one frame into dozens of crops and
    # inflates the text stream with matching <image> tokens (seen: 34 crops,
    # seq_len≈2852). That is forbidden for the V100 memory gate / CALVIN path.
    if hasattr(image_processor, "do_image_splitting"):
        image_processor.do_image_splitting = False
    # SmolVLM uses longest_edge dicts, not {height,width}.
    longest = {"longest_edge": int(image_size)}
    for attr, value in (
        ("size", longest),
        ("max_image_size", longest),
        ("do_resize", True),
        ("do_center_crop", False),
    ):
        if hasattr(image_processor, attr):
            setattr(image_processor, attr, value)
    return processor


def truncate_text_context(
    batch: dict[str, torch.Tensor],
    *,
    max_text_tokens: int,
    image_token_id: int | None,
) -> dict[str, torch.Tensor]:
    """Clip only the text tail; never cut mid-<image> token blocks.

    SmolVLM requires ``#image_tokens % image_seq_len == 0`` (81 for SmolVLM2).
    Hard-clipping a global budget of 128 was corrupting the second camera's
    placeholders and raised ``<image> tokens not divisible by patch_size``.
    """
    if "input_ids" not in batch:
        return batch
    input_ids = batch["input_ids"]
    if input_ids.shape[-1] <= max_text_tokens:
        return batch

    ids = input_ids[0]
    if image_token_id is not None:
        image_positions = (ids == int(image_token_id)).nonzero(as_tuple=False).flatten()
        if len(image_positions) > 0:
            keep_prefix = int(image_positions[-1].item()) + 1
        else:
            keep_prefix = 0
    else:
        keep_prefix = 0

    if keep_prefix > max_text_tokens:
        raise RuntimeError(
            f"image placeholders alone need {keep_prefix} tokens but "
            f"max_text_tokens={max_text_tokens}; raise the cap (do not clip "
            f"<image> blocks — SmolVLM requires full image_seq_len groups)"
        )

    end = max_text_tokens
    sliced: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key in {"input_ids", "attention_mask", "labels"} and value.ndim >= 2:
            sliced[key] = value[..., :end].contiguous()
        else:
            sliced[key] = value
    return sliced


def build_smolvlm_batch(
    processor: Any,
    *,
    batch_size: int,
    num_images: int,
    image_size: int,
    max_text_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a processor-aligned fake batch with CALVIN-scale sequence length."""
    from PIL import Image

    configure_calvin_processor(processor, image_size=image_size)
    images = [
        Image.fromarray(
            (torch.rand(image_size, image_size, 3).numpy() * 255).astype("uint8")
        )
        for _ in range(num_images)
    ]
    content: list[dict[str, str]] = [{"type": "image"} for _ in range(num_images)]
    content.append(
        {
            "type": "text",
            "text": "open the drawer",
        }
    )
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    encoded = processor(text=text, images=images, return_tensors="pt")
    batch: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if not torch.is_tensor(value):
            continue
        tensor = value
        if tensor.ndim >= 1 and tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1))
        batch[key] = tensor

    image_token_id = getattr(getattr(processor, "tokenizer", None), "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_id", None)
    if "labels" not in batch and "input_ids" in batch:
        labels = batch["input_ids"].clone()
        if image_token_id is not None:
            labels = labels.masked_fill(labels == int(image_token_id), -100)
        batch["labels"] = labels

    batch = truncate_text_context(
        batch,
        max_text_tokens=max_text_tokens,
        image_token_id=int(image_token_id) if image_token_id is not None else None,
    )

    # Fail closed if splitting somehow survived.
    if "pixel_values" in batch and batch["pixel_values"].ndim == 5:
        num_crops = int(batch["pixel_values"].shape[1])
        if num_crops > num_images:
            raise RuntimeError(
                f"image splitting still active: pixel_values has {num_crops} crops "
                f"for num_images={num_images}; expected do_image_splitting=False"
            )
    if "input_ids" in batch and batch["input_ids"].shape[-1] > max_text_tokens:
        raise RuntimeError(
            f"input_ids length {batch['input_ids'].shape[-1]} exceeds "
            f"max_text_tokens={max_text_tokens}"
        )

    for key, value in list(batch.items()):
        batch[key] = value.to(device, non_blocking=True)
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=256,
        help=(
            "Hard cap on total input_ids length including vision placeholders. "
            "SmolVLM2 uses image_seq_len=81 per frame; 2 cams need ≥162 + short text. "
            "Still << the dirty 2852-token splitting blow-up."
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument(
        "--require-exclusive-gpus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort unless no foreign compute processes hold VRAM (protocol).",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--nvidia-smi-every",
        type=int,
        default=50,
        help="Sample host nvidia-smi this often (0 disables). Cheap; keeps OS-level VRAM honest.",
    )
    parser.add_argument(
        "--from-config-only",
        action="store_true",
        help="Build randomly-initialized weights from config (no HF weight download).",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match baseline_protocol.json v100_execution.activation_checkpointing.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def setup_dist() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun so LOCAL_RANK / RANK / WORLD_SIZE are set")
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def assert_exclusive_gpus(local_rank: int) -> None:
    """Fail closed if foreign CUDA processes share the node."""
    if local_rank != 0:
        return
    rows = query_nvidia_smi()
    apps_raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=5,
    ).strip()
    foreign = []
    our_markers = ("vla_memory_probe", "torchrun")
    for line in apps_raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        pid, name, used = parts[0], parts[1], parts[2]
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            cmdline = name
        if any(marker in cmdline for marker in our_markers):
            continue
        foreign.append({"pid": pid, "name": name, "used_mib": used, "cmd": cmdline[:160]})
    if foreign:
        raise RuntimeError(
            "Exclusive GPU gate failed; foreign compute apps present: "
            + json.dumps(foreign)
            + f"; smi={rows}"
        )
    print(f"[FSDP Probe] exclusive GPU gate passed; smi={rows}", flush=True)


def query_nvidia_smi() -> list[dict[str, Any]]:
    """OS-level VRAM sample. Safe to call from rank 0 every N steps."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in raw.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": float(parts[2]),
                "memory_total_mib": float(parts[3]),
                "utilization_gpu": float(parts[4]),
            }
        )
    return rows


def build_model(args: argparse.Namespace, local_rank: int) -> torch.nn.Module:
    if local_rank == 0:
        print(f"[FSDP Probe] loading config for {args.model_id}", flush=True)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    # V100: forbid FlashAttention-2; force SDPA (frozen in FINDINGS / protocol).
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "sdpa"
    # Keep master weights in fp32 so GradScaler can unscale; FSDP MixedPrecision
    # + autocast provide the fp16 compute path (casting the whole module to
    # fp16 before wrap makes scaler crash: "Attempting to unscale FP16 gradients").
    kwargs = {"trust_remote_code": True, "torch_dtype": torch.float32, "attn_implementation": "sdpa"}
    if args.from_config_only:
        if local_rank == 0:
            print("[FSDP Probe] from_config (random init; no weight download)", flush=True)
        model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)
        model = model.to(dtype=torch.float32)
    else:
        if local_rank == 0:
            print("[FSDP Probe] from_pretrained (downloads weights on first use)", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(args.model_id, **kwargs)
    model.train()
    return model


def maybe_enable_activation_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    # Best-effort; transformer block class differs across SmolVLM revisions.
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        def _check_fn(module: torch.nn.Module) -> bool:
            name = module.__class__.__name__.lower()
            return any(key in name for key in ("decoderlayer", "transformerblock", "llamadecoderlayer"))

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=_check_fn,
        )
    except Exception as exc:  # noqa: BLE001 — probe must continue even if wrap misses
        print(f"[FSDP Probe] activation checkpointing skipped: {exc}", flush=True)


def forward_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Real LM loss so activations + grads allocate like training."""
    allowed = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_attention_mask",
        "labels",
    }
    model_batch = {key: value for key, value in batch.items() if key in allowed}
    outputs = model(**model_batch)
    if getattr(outputs, "loss", None) is None:
        logits = outputs.logits
        labels = model_batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    else:
        loss = outputs.loss
    return loss


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    args = parse_args()
    local_rank, world_size = setup_dist()
    if args.require_exclusive_gpus:
        # Only rank0 inspects; other ranks wait so we fail before heavy alloc.
        if local_rank == 0:
            assert_exclusive_gpus(local_rank)
        dist.barrier()
    if world_size < 2 and local_rank == 0:
        print(
            "[WARN] world_size < 2; dual-V100 FSDP headroom is not being tested",
            flush=True,
        )
    if local_rank == 0:
        print(
            f"[FSDP Probe] world_size={world_size} model={args.model_id} "
            f"steps={args.steps} micro_batch={args.micro_batch_size} "
            f"precision=fp16-mixed attn=sdpa",
            flush=True,
        )
        print(f"[FSDP Probe] visible cuda devices={torch.cuda.device_count()}", flush=True)
        for row in query_nvidia_smi():
            print(f"[nvidia-smi before] {row}", flush=True)

    model = build_model(args, local_rank)
    maybe_enable_activation_checkpointing(model, args.activation_checkpointing)

    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    # Wrap after moving to the local CUDA device so FSDP can shard eagerly.
    model = model.to(torch.device("cuda", local_rank))
    auto_wrap = None
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        auto_wrap = transformer_auto_wrap_policy({LlamaDecoderLayer})
    except Exception:
        auto_wrap = None

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=auto_wrap,
        use_orig_params=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-10,
    )
    scaler = ShardedGradScaler(enabled=True)
    if local_rank == 0:
        print(f"[FSDP Probe] loading processor for {args.model_id}", flush=True)
        print("[FSDP Probe] using ShardedGradScaler (fp16 + FSDP on Volta)", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    # Protocol / CALVIN: fixed square frame, not the model default 384 with splitting.
    vision_size = int(args.image_size)
    device = torch.device("cuda", local_rank)
    # Build one representative batch and reuse it — VRAM peak is what we measure.
    probe_batch = build_smolvlm_batch(
        processor,
        batch_size=args.micro_batch_size,
        num_images=args.num_images,
        image_size=vision_size,
        max_text_tokens=int(args.max_text_tokens),
        device=device,
    )
    if local_rank == 0:
        shapes = {k: tuple(v.shape) for k, v in probe_batch.items()}
        print(f"[FSDP Probe] batch shapes={shapes}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    allocated0 = torch.cuda.memory_allocated() / (1024**2)
    if local_rank == 0:
        print(f"[GPU:{local_rank}] post-FSDP allocated={allocated0:.1f} MiB", flush=True)

    history: list[dict[str, Any]] = []
    smi_history: list[dict[str, Any]] = []
    loss_finite = True
    grad_finite = True
    t0 = time.perf_counter()
    oom = False

    model.train()
    try:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            # Volta/V100: fp16-mixed under FSDP requires ShardedGradScaler so
            # Inf/NaN overflow masks sync across ranks before the optimizer step.
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = forward_loss(model, probe_batch)
            if not torch.isfinite(loss):
                loss_finite = False
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                grad_finite = False
            scaler.step(optimizer)
            scaler.update()

            if step % args.log_every == 0 or step + 1 == args.steps:
                peak = torch.cuda.max_memory_allocated() / (1024**2)
                entry = {
                    "step": step,
                    "loss": float(loss.detach().float().cpu()),
                    "grad_norm": float(grad_norm.detach().float().cpu())
                    if torch.is_tensor(grad_norm)
                    else float(grad_norm),
                    "peak_vram_torch_mib": peak,
                }
                history.append(entry)
                if local_rank == 0:
                    print(
                        f"[Step {step}/{args.steps}] loss={entry['loss']:.4f} "
                        f"grad={entry['grad_norm']:.4f} "
                        f"peak_torch={peak:.1f} MiB",
                        flush=True,
                    )
            if (
                args.nvidia_smi_every > 0
                and local_rank == 0
                and (step % args.nvidia_smi_every == 0 or step + 1 == args.steps)
            ):
                sample = {"step": step, "gpus": query_nvidia_smi()}
                smi_history.append(sample)
                print(f"[nvidia-smi step={step}] {sample['gpus']}", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        oom = True
        if local_rank == 0:
            print(f"[OOM] {exc}", flush=True)

    elapsed = time.perf_counter() - t0
    peak_final = torch.cuda.max_memory_allocated() / (1024**2)
    report = {
        "model_id": args.model_id,
        "world_size": world_size,
        "steps_requested": args.steps,
        "steps_ran": len(history) and (history[-1]["step"] + 1) or 0,
        "micro_batch_size_per_gpu": args.micro_batch_size,
        "precision": "fp16_mixed",
        "attention": "sdpa",
        "fsdp": "full_shard",
        "activation_checkpointing": bool(args.activation_checkpointing),
        "from_config_only": bool(args.from_config_only),
        "peak_vram_torch_mib": peak_final,
        "elapsed_seconds": elapsed,
        "micro_steps_per_second": (args.steps / elapsed) if elapsed > 0 and not oom else None,
        "loss_finite": loss_finite,
        "gradient_norm_finite": grad_finite,
        "oom": oom,
        "history": history,
        "nvidia_smi_samples": smi_history,
        "pass_no_oom": not oom,
    }

    dist.barrier()
    if local_rank == 0:
        print(json.dumps({k: v for k, v in report.items() if k not in {"history", "nvidia_smi_samples"}}, indent=2), flush=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[FSDP Probe] wrote {args.output_json}", flush=True)
        if oom:
            print("[FAIL] OOM under 16-mixed + SDPA + FSDP — move 2.2B off this host", flush=True)
        else:
            print("[SUCCESS] 200-step memory probe finished without OOM", flush=True)

    dist.destroy_process_group()
    return 1 if oom else 0


if __name__ == "__main__":
    raise SystemExit(main())
