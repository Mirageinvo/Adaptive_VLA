#!/usr/bin/env python3
"""Train the primary CALVIN-native ActionCodec stages.

Primary stages:
  - base_vq: single-layer VQ
  - rvq_posttrain: Appendix C RVQ with frozen encoder + primary codebook

There is no primary pretrained-transfer training arm. A pretrained transfer run
is descriptive-only and, if ever added, must train only the new CALVIN
embodiment soft-prompt while freezing the inherited trunk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from codec_common import (
    ActionCodec,
    assert_dataset_matches_protocol,
    build_base_vq_codec,
    encode_quantize_decode,
    enforce_rvq_frozen_primary,
    grouped_reconstruction_loss,
    initialize_rvq_from_base,
    load_json,
    seed_everything,
    sha1_file,
    warm_residual_codebook_init,
)
from codec_data import CalvinActionChunkDataset


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("base_vq", "rvq_posttrain"), required=True)
    parser.add_argument(
        "--base-vq-checkpoint",
        type=Path,
        help="Required for rvq_posttrain; directory containing the base VQ model.",
    )
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--protocol", type=Path, default=root / "protocols" / "codec_protocol.json"
    )
    parser.add_argument("--normalization", choices=("native", "quantile"), default="native")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=None,
        help="Max val batches. Default: full loader for scientific/benchmark; "
        "smoke uses a small cap.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--memory-smoke",
        action="store_true",
        help="Two optimizer steps at the registered micro-batch size (default 256).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the registered 200-step hardware benchmark; not a scientific result.",
    )
    return parser.parse_args()


def build_model(
    args: argparse.Namespace, protocol: dict[str, Any], device: torch.device
) -> tuple[ActionCodec, int, dict[str, Any] | None]:
    if args.resume:
        model = ActionCodec.from_pretrained(args.resume / "model").to(device)
        keys = list(model.config.embodiment_config)
        return model, keys.index("franka_calvin_30hz"), None
    if args.stage == "base_vq":
        model, embodiment_id = build_base_vq_codec(protocol)
        return model.to(device), embodiment_id, None
    if args.base_vq_checkpoint is None:
        raise ValueError("--base-vq-checkpoint is required for rvq_posttrain")
    base = ActionCodec.from_pretrained(args.base_vq_checkpoint)
    model, embodiment_id, inheritance_report = initialize_rvq_from_base(base, protocol)
    return model.to(device), embodiment_id, inheritance_report


@torch.no_grad()
def validate(
    model: ActionCodec,
    loader: DataLoader,
    embodiment_id: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "reconstruction": 0.0, "quantization": 0.0}
    groups = {"position": 0.0, "rotation": 0.0, "gripper": 0.0}
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        action = batch["action"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        prediction, recon_mask, _, quantization_loss = encode_quantize_decode(
            model, action, embodiment_id, padding_mask
        )
        reconstruction_loss, grouped = grouped_reconstruction_loss(
            prediction, action, recon_mask
        )
        total_loss = reconstruction_loss + quantization_loss
        totals["loss"] += float(total_loss)
        totals["reconstruction"] += float(reconstruction_loss)
        totals["quantization"] += float(quantization_loss)
        for key in groups:
            groups[key] += float(grouped[key])
        count += 1
    model.train()
    enforce_rvq_frozen_primary(model)
    if count == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {key: value / count for key, value in (totals | groups).items()}


def save_checkpoint(
    directory: Path,
    model: ActionCodec,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    best_val: float,
    metadata: dict[str, Any],
    bad_validations: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "model", safe_serialization=True)
    torch.save(
        {
            "step": step,
            "stage": metadata.get("stage"),
            "best_val": best_val,
            "bad_validations": bad_validations,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metadata": metadata,
        },
        directory / "trainer_state.pt",
    )


def main() -> int:
    args = parse_args()
    protocol = load_json(args.protocol)
    registered = protocol["training"]["codec"]
    stage_config = registered[args.stage]
    if sum(bool(flag) for flag in (args.smoke, args.benchmark, args.memory_smoke)) > 1:
        raise ValueError("--smoke, --benchmark, and --memory-smoke are mutually exclusive")
    if (
        not args.smoke
        and not args.benchmark
        and not args.memory_smoke
        and (
            registered["betas_status"].startswith("not_disclosed")
            or registered["weight_decay_status"].startswith("not_disclosed")
            or (
                args.stage == "rvq_posttrain"
                and str(stage_config.get("learning_rate_status", "")).startswith(
                    "not_disclosed"
                )
            )
            or (
                args.stage == "rvq_posttrain"
                and stage_config.get("max_steps") is None
                and args.max_steps is None
            )
        )
    ):
        raise RuntimeError(
            "Scientific codec training is blocked: tokenizer optimizer "
            "betas/weight_decay and/or RVQ post-train lr/steps are not disclosed "
            "and have not been resolved in the frozen protocol"
        )
    if registered["precision"] != "fp32":
        raise RuntimeError(
            f"Registered codec precision is {registered['precision']!r}; "
            "fp16/bf16 autocast is forbidden for the primary tokenizer path"
        )
    # stage_config already bound above for the disclosure gate
    registered_max_steps = stage_config["max_steps"]
    early_stopping_patience = int(registered["early_stopping_validation_patience"])
    if args.benchmark:
        max_steps = int(
            args.max_steps or protocol["budget"]["codec_benchmark_steps_per_stage"]
        )
    elif args.memory_smoke:
        max_steps = int(args.max_steps or 2)
    elif args.max_steps is None and registered_max_steps is None:
        raise ValueError(
            "RVQ post-training step count is undisclosed and unfrozen; "
            "pass --max-steps only for --smoke until it is registered"
        )
    else:
        max_steps = int(args.max_steps or registered_max_steps)
    batch_size = int(
        args.batch_size or registered["micro_batch_size_candidate_for_benchmark"]
    )
    accumulation_steps = int(
        registered["gradient_accumulation_steps_candidate_for_benchmark"]
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else registered["num_workers"]
    )
    validation_interval = int(registered["validation_interval"])
    checkpoint_interval = int(registered["checkpoint_interval"])
    if args.smoke:
        max_steps = min(max_steps, 10)
        batch_size = min(batch_size, 4)
        accumulation_steps = 1
        num_workers = 0
        validation_interval = 5
        checkpoint_interval = 10
        args.validation_batches = (
            2 if args.validation_batches is None else min(args.validation_batches, 2)
        )
    if args.memory_smoke:
        accumulation_steps = 1
        num_workers = min(num_workers, 2)
        validation_interval = max_steps + 1
        checkpoint_interval = max_steps
        args.validation_batches = 1
    global_batch = int(registered["global_batch_size"])
    if registered.get("require_micro_batch_times_accumulation_equals_global_batch"):
        if not args.smoke and not args.memory_smoke:
            if batch_size * accumulation_steps != global_batch:
                raise RuntimeError(
                    f"Effective batch {batch_size}*{accumulation_steps}="
                    f"{batch_size * accumulation_steps} != registered global "
                    f"batch {global_batch}"
                )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CalvinActionChunkDataset(
        args.data_root,
        split="train",
        normalization=args.normalization,
        protocol_path=args.protocol,
    )
    assert_dataset_matches_protocol(train_dataset.manifest, protocol)
    selection_split = str(registered.get("selection_split", "dev"))
    if selection_split == "val" and not args.smoke and not args.memory_smoke:
        raise RuntimeError(
            "Official CALVIN validation/ is forbidden for codec model selection; "
            "use the carved development split"
        )
    # Fallback for pipeline trees that disabled the carve.
    available_codes = train_dataset.episode_index["split_codes"]
    if selection_split not in available_codes:
        raise RuntimeError(f"Selection split {selection_split!r} missing from dataset")
    if (
        selection_split == "dev"
        and not train_dataset.manifest.get("development_split", {}).get("enabled", False)
    ):
        if args.smoke or args.memory_smoke or args.benchmark:
            selection_split = "val"
        else:
            raise RuntimeError(
                "Development split is disabled in the converted dataset; "
                "reconversion with an enabled carve is required"
            )
    val_dataset = CalvinActionChunkDataset(
        args.data_root,
        split=selection_split,  # type: ignore[arg-type]
        normalization=args.normalization,
        protocol_path=args.protocol,
    )
    if len(val_dataset) == 0:
        raise RuntimeError(
            f"Selection split {selection_split!r} has zero chunks; "
            "reconversion or a larger source tree is required"
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training dataset is smaller than one drop-last batch")
    if args.validation_batches is None:
        args.validation_batches = max(len(val_loader), 1)

    model, embodiment_id, expansion_report = build_model(args, protocol, device)
    residual_init_report = None
    if args.stage == "rvq_posttrain" and args.resume is None:
        warm_batch = next(iter(train_loader))["action"].to(device)
        residual_init_report = warm_residual_codebook_init(
            model, warm_batch, embodiment_id
        )
    model.train()
    enforce_rvq_frozen_primary(model)
    learning_rate = stage_config.get("learning_rate")
    if learning_rate is None:
        learning_rate = stage_config.get(
            "learning_rate_implementation_candidate",
            registered["reported_actioncodec_recipe"]["learning_rate"],
        )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(learning_rate),
        betas=tuple(registered["betas_implementation_candidate"]),
        weight_decay=float(registered["weight_decay_implementation_candidate"]),
        eps=float(registered["eps"]),
    )
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    start_step, best_val, bad_validations = 0, float("inf"), 0
    if args.resume:
        state = torch.load(args.resume / "trainer_state.pt", map_location="cpu")
        if state.get("stage", args.stage) != args.stage:
            raise RuntimeError(
                f"Resume stage {state.get('stage')!r} does not match --stage {args.stage!r}"
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        best_val = float(state["best_val"])
        bad_validations = int(state.get("bad_validations", 0))

    metadata = {
        "stage": args.stage,
        "seed": args.seed,
        "normalization": args.normalization,
        "protocol_sha1": sha1_file(args.protocol),
        "data_manifest": {
            "generated_sha256": train_dataset.manifest["generated_sha256"],
            "is_pipeline_validation": train_dataset.manifest["is_pipeline_validation"],
            "dataset_root": train_dataset.manifest["dataset_root"],
            "num_frames": train_dataset.manifest["num_frames"],
            "num_chunks": train_dataset.manifest["num_chunks"],
        },
        "embodiment_id": embodiment_id,
        "vq_to_rvq_inheritance_gate": expansion_report,
        "residual_codebook_init": residual_init_report,
        "smoke": args.smoke,
        "memory_smoke": args.memory_smoke,
        "benchmark": args.benchmark,
        "precision": registered["precision"],
        "autocast": False,
        "early_stopping_validation_patience": early_stopping_patience,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "registered_global_batch_size": global_batch,
        "validation_batches": args.validation_batches,
        "selection_split": selection_split,
        "learning_rate": float(learning_rate),
        "scheduler": "constant",
        "scheduler_status": "paper reports peak lr only; constant schedule is an implementation assumption",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    history_path = args.output_dir / "history.jsonl"
    if history_path.exists() and args.resume is None:
        history_path.write_text("")
    train_iterator = iter(train_loader)
    timing_start = time.perf_counter()
    stopped_early = False

    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        aggregate = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "quantization": 0.0,
            "position": 0.0,
            "rotation": 0.0,
            "gripper": 0.0,
        }
        for _micro_step in range(accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            action = batch["action"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            # Explicit fp32 path: no autocast.
            prediction, recon_mask, _, quantization_loss = encode_quantize_decode(
                model, action, embodiment_id, padding_mask
            )
            reconstruction_loss, grouped = grouped_reconstruction_loss(
                prediction, action, recon_mask
            )
            loss = reconstruction_loss + quantization_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
            (loss / accumulation_steps).backward()
            aggregate["loss"] += float(loss.detach()) / accumulation_steps
            aggregate["reconstruction"] += (
                float(reconstruction_loss.detach()) / accumulation_steps
            )
            aggregate["quantization"] += (
                float(quantization_loss.detach()) / accumulation_steps
            )
            for key in grouped:
                aggregate[key] += float(grouped[key].detach()) / accumulation_steps
        gradient_norm = clip_grad_norm_(
            model.parameters(), float(registered["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        enforce_rvq_frozen_primary(model)

        record = {
            "step": step,
            "train_loss": aggregate["loss"],
            "train_reconstruction": aggregate["reconstruction"],
            "train_quantization": aggregate["quantization"],
            "train_position_mse": aggregate["position"],
            "train_rotation_mse": aggregate["rotation"],
            "train_gripper_mse": aggregate["gripper"],
            "gradient_norm": float(gradient_norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if step % validation_interval == 0 or step == max_steps:
            record["validation"] = validate(
                model, val_loader, embodiment_id, device, args.validation_batches
            )
            val_loss = record["validation"]["reconstruction"]
            if val_loss < best_val:
                best_val = val_loss
                bad_validations = 0
                save_checkpoint(
                    args.output_dir / "best",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_val,
                    metadata,
                    bad_validations,
                )
            else:
                bad_validations += 1
            record["bad_validations"] = bad_validations
            if (
                not args.smoke
                and not args.memory_smoke
                and bad_validations >= early_stopping_patience
            ):
                record["early_stopped"] = True
                stopped_early = True
        with history_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step % 10 == 0 or step == max_steps or stopped_early:
            elapsed = time.perf_counter() - timing_start
            print(
                f"step={step}/{max_steps} loss={record['train_loss']:.6f} "
                f"lr={record['learning_rate']:.3e} steps/s={(step-start_step)/elapsed:.3f}"
                + (
                    f" early_stop={bad_validations}/{early_stopping_patience}"
                    if "bad_validations" in record
                    else ""
                ),
                flush=True,
            )
        if step % checkpoint_interval == 0 or step == max_steps or stopped_early:
            save_checkpoint(
                args.output_dir / f"step_{step:08d}",
                model,
                optimizer,
                scheduler,
                step,
                best_val,
                metadata,
                bad_validations,
            )
        if stopped_early:
            break

    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"peak_cuda_memory_mib={peak_mib:.1f}", flush=True)
        (args.output_dir / "hardware_probe.json").write_text(
            json.dumps(
                {
                    "peak_cuda_memory_mib": peak_mib,
                    "batch_size": batch_size,
                    "device": str(device),
                    "memory_smoke": args.memory_smoke,
                    "benchmark": args.benchmark,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
