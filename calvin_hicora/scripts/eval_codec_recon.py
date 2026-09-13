#!/usr/bin/env python3
"""Evaluate CALVIN ActionCodec reconstruction and registered RVQ gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from codec_common import ActionCodec, load_json, seed_everything, sha1_file
from codec_data import CalvinActionChunkDataset


LEVEL_NAMES = {1: "level0", 2: "level0+1", 3: "full"}


def assert_vendored_codec_signatures(model: ActionCodec) -> None:
    """Fail closed if the vendored ActionCodec API drifts."""
    import inspect

    from_codes_params = tuple(inspect.signature(model.vq.from_codes).parameters)
    decode_params = tuple(inspect.signature(model._decode).parameters)
    if from_codes_params != ("codes",):
        raise RuntimeError(
            f"Unexpected ResidualVectorQuantize.from_codes signature: {from_codes_params}"
        )
    # Vendored rvq.py returns (z_q, projected_latents, codes).
    probe_codes = torch.zeros(1, model.n_tokens_per_quantizer, 1, dtype=torch.long)
    from_codes_out = model.vq.from_codes(probe_codes)
    if not isinstance(from_codes_out, tuple) or len(from_codes_out) != 3:
        raise RuntimeError(
            f"from_codes must return 3 values; got {type(from_codes_out)} "
            f"len={getattr(from_codes_out, '__len__', None)}"
        )
    if decode_params != ("z_q", "embodiment_ids", "durations"):
        raise RuntimeError(
            f"Unexpected ActionCodec._decode signature: {decode_params}; "
            "padding_mask is not accepted and must be derived from embodiment ids"
        )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=root / "protocols" / "codec_protocol.json"
    )
    parser.add_argument("--normalization", choices=("native", "quantile"), default="native")
    parser.add_argument("--split", choices=("train", "val", "dev", "all"), default="dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enforce-gate", action="store_true")
    parser.add_argument("--include-code-distribution", action="store_true")
    return parser.parse_args()


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = np.zeros(7, dtype=np.float64)
        self.sse = np.zeros(7, dtype=np.float64)
        self.sae = np.zeros(7, dtype=np.float64)
        self.target_sum = np.zeros(7, dtype=np.float64)
        self.target_sq_sum = np.zeros(7, dtype=np.float64)
        self.gripper_correct = 0
        self.gripper_count = 0
        self.start_sse = np.zeros(7, dtype=np.float64)
        self.end_sse = np.zeros(7, dtype=np.float64)
        self.boundary_count = 0
        self.temporal_sse = np.zeros(7, dtype=np.float64)
        self.temporal_count = 0

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> None:
        valid = mask.bool().unsqueeze(-1).expand_as(target)
        error = prediction - target
        for values, destination in (
            (error.square(), self.sse),
            (error.abs(), self.sae),
            (target, self.target_sum),
            (target.square(), self.target_sq_sum),
        ):
            destination += (
                torch.where(valid, values, torch.zeros_like(values))
                .sum(dim=(0, 1))
                .double()
                .cpu()
                .numpy()
            )
        self.count += valid.sum(dim=(0, 1)).double().cpu().numpy()
        gripper_valid = valid[..., 6]
        predicted_sign = torch.where(prediction[..., 6] >= 0, 1, -1)
        target_sign = torch.where(target[..., 6] >= 0, 1, -1)
        self.gripper_correct += int(
            ((predicted_sign == target_sign) & gripper_valid).sum().item()
        )
        self.gripper_count += int(gripper_valid.sum().item())

        self.start_sse += error[:, 0].square().sum(0).double().cpu().numpy()
        self.end_sse += error[:, -1].square().sum(0).double().cpu().numpy()
        self.boundary_count += target.shape[0]
        delta_valid = mask[:, 1:] & mask[:, :-1]
        delta_error = (prediction[:, 1:] - prediction[:, :-1]) - (
            target[:, 1:] - target[:, :-1]
        )
        self.temporal_sse += (
            torch.where(
                delta_valid.unsqueeze(-1),
                delta_error.square(),
                torch.zeros_like(delta_error),
            )
            .sum(dim=(0, 1))
            .double()
            .cpu()
            .numpy()
        )
        self.temporal_count += int(delta_valid.sum().item())

    def result(self) -> dict[str, Any]:
        mse = self.sse / np.maximum(self.count, 1)
        mae = self.sae / np.maximum(self.count, 1)
        centered_sst = self.target_sq_sum - np.square(self.target_sum) / np.maximum(
            self.count, 1
        )
        r2 = 1 - self.sse / np.maximum(centered_sst, np.finfo(np.float64).eps)

        def group(values: np.ndarray, start: int, stop: int) -> float:
            return float(values[start:stop].mean())

        group_sse = {
            "position": float(self.sse[:3].sum()),
            "rotation": float(self.sse[3:6].sum()),
        }
        group_sst = {
            "position": float(centered_sst[:3].sum()),
            "rotation": float(centered_sst[3:6].sum()),
        }
        return {
            "per_channel": {
                str(index): {
                    "mse": float(mse[index]),
                    "mae": float(mae[index]),
                    "r2": float(r2[index]),
                }
                for index in range(7)
            },
            "position_mse": group(mse, 0, 3),
            "rotation_mse": group(mse, 3, 6),
            "gripper_mse": float(mse[6]),
            "position_rotation_mse": float(mse[:6].mean()),
            "position_r2": 1
            - group_sse["position"] / max(group_sst["position"], np.finfo(float).eps),
            "rotation_r2": 1
            - group_sse["rotation"] / max(group_sst["rotation"], np.finfo(float).eps),
            "gripper_sign_accuracy": self.gripper_correct / max(self.gripper_count, 1),
            "start_mse_per_channel": (
                self.start_sse / max(self.boundary_count, 1)
            ).tolist(),
            "end_mse_per_channel": (
                self.end_sse / max(self.boundary_count, 1)
            ).tolist(),
            "temporal_delta_mse_per_channel": (
                self.temporal_sse / max(self.temporal_count, 1)
            ).tolist(),
            "valid_scalar_count": int(self.count.sum()),
        }


def codebook_usage(counts: np.ndarray) -> dict[str, Any]:
    total = counts.sum()
    probabilities = counts[counts > 0] / max(total, 1)
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        "used_codes": int((counts > 0).sum()),
        "dead_codes": int((counts == 0).sum()),
        "dead_fraction": float((counts == 0).mean()),
        "perplexity": float(math.exp(entropy)),
        "observations": int(total),
    }


def apply_gate(levels: dict[str, Any], usage: list[dict[str, Any]], protocol: dict) -> dict:
    gate = protocol["codec_gate"]
    frozen = gate.get("frozen_checks", gate)
    provisional = gate.get(
        "provisional_numeric_candidates_not_paper_or_spec_sourced", {}
    )
    level0, level1, full = levels["level0"], levels["level0+1"], levels["full"]
    ratio = level0["position_rotation_mse"] / max(
        full["position_rotation_mse"], np.finfo(float).eps
    )
    checks = {
        "decode_path_parity": bool(levels["decode_path_parity"]),
        "monotonic_mse": (
            level0["position_rotation_mse"]
            > level1["position_rotation_mse"]
            > full["position_rotation_mse"]
        ),
    }
    if not frozen.get("require_decode_path_parity", True):
        checks.pop("decode_path_parity")
    if not frozen.get("require_monotonic_mse", True):
        checks.pop("monotonic_mse")

    provisional_checks = {}
    if provisional:
        provisional_checks = {
            "minimum_level0_to_full_mse_ratio": ratio
            >= float(provisional["minimum_level0_to_full_mse_ratio"]),
            "level0_position_r2_range": float(provisional["minimum_level0_position_r2"])
            <= level0["position_r2"]
            <= float(provisional["maximum_level0_position_r2"]),
            "level0_rotation_r2_range": float(provisional["minimum_level0_rotation_r2"])
            <= level0["rotation_r2"]
            <= float(provisional["maximum_level0_rotation_r2"]),
            "minimum_level0_gripper_sign_accuracy": level0["gripper_sign_accuracy"]
            >= float(provisional["minimum_level0_gripper_sign_accuracy"]),
            "maximum_dead_code_fraction": all(
                item["dead_fraction"]
                <= float(provisional["maximum_dead_code_fraction_per_level"])
                for item in usage
            ),
        }
    scientific_passed = all(checks.values()) if checks else True
    return {
        "passed": scientific_passed,
        "checks": checks,
        "provisional_numeric_checks": provisional_checks,
        "provisional_numeric_status": provisional.get(
            "status", "absent"
        ),
        "level0_to_full_position_rotation_mse_ratio": ratio,
        "note": (
            "Numeric R2/ratio/dead-code thresholds are provisional candidates and "
            "do not decide scientific go/no-go until frozen with provenance."
        ),
    }


@torch.no_grad()
def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    protocol = load_json(args.protocol)
    device = torch.device(args.device)
    model = ActionCodec.from_pretrained(args.model).to(device).eval()
    assert_vendored_codec_signatures(model)
    keys = list(model.config.embodiment_config)
    if "franka_calvin_30hz" not in keys:
        raise ValueError("Checkpoint has no franka_calvin_30hz embodiment")
    embodiment_id = keys.index("franka_calvin_30hz")
    dataset = CalvinActionChunkDataset(
        args.data_root,
        split=args.split,
        normalization=args.normalization,
        protocol_path=args.protocol,
    )
    if args.enforce_gate and dataset.manifest["is_pipeline_validation"]:
        raise RuntimeError("Cannot enforce gate on pipeline validation dataset")
    if args.split == "val":
        print(
            "WARNING: evaluating on official CALVIN validation/; "
            "model selection must use the carved development split",
            flush=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    accumulators = {name: MetricAccumulator() for name in LEVEL_NAMES.values()}
    code_counts = np.zeros(
        (int(protocol["geometry"]["rvq_levels"]), int(protocol["geometry"]["codebook_size"])),
        dtype=np.int64,
    )
    decode_path_code_parity = True
    decode_path_latent_parity = True
    max_decode_path_delta = 0.0
    batches = 0
    evaluated_chunks = 0

    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        action = batch["action"].to(device, non_blocking=True)
        evaluated_chunks += action.shape[0]
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        ids = torch.full(
            (action.shape[0],), embodiment_id, dtype=torch.long, device=device
        )
        z_e = model._encode(action, ids, padding_mask)
        _, full_codes, _, _ = model._quantize(z_e, return_perplexity=False)
        for level in range(full_codes.shape[-1]):
            bincount = torch.bincount(
                full_codes[..., level].reshape(-1),
                minlength=code_counts.shape[1],
            )
            code_counts[level] += bincount.cpu().numpy()

        for n_quantizers, name in LEVEL_NAMES.items():
            # Vendored rvq.py returns (z_q, projected_latents, codes).
            z_from_codes, _, _ = model.vq.from_codes(full_codes[..., :n_quantizers])
            z_direct, direct_codes, _, _, _ = model.vq(
                z_e, n_quantizers=n_quantizers
            )
            delta = float((z_from_codes - z_direct).abs().max().item())
            max_decode_path_delta = max(max_decode_path_delta, delta)
            decode_path_code_parity &= bool(
                torch.equal(full_codes[..., :n_quantizers], direct_codes)
            )
            decode_path_latent_parity &= bool(
                torch.allclose(
                    z_from_codes,
                    z_direct,
                    rtol=0.0,
                    atol=float(
                        protocol["codec_gate"]["frozen_checks"][
                            "maximum_decode_path_latent_absolute_delta"
                        ]
                        if "frozen_checks" in protocol["codec_gate"]
                        else protocol["codec_gate"][
                            "maximum_decode_path_latent_absolute_delta"
                        ]
                    ),
                )
            )
            # ActionCodec._decode accepts (z_q, embodiment_ids, durations);
            # its decoder derives the reconstruction mask from embodiment IDs.
            prediction, recon_mask = model._decode(z_from_codes, ids)
            accumulators[name].update(prediction[..., :7], action, recon_mask)
        batches += 1

    if batches == 0:
        raise RuntimeError("No validation batches evaluated")
    level_results = {name: accumulator.result() for name, accumulator in accumulators.items()}
    level_results["decode_path_code_parity"] = decode_path_code_parity
    level_results["decode_path_latent_parity"] = decode_path_latent_parity
    level_results["decode_path_parity"] = (
        decode_path_code_parity and decode_path_latent_parity
    )
    level_results["maximum_decode_path_absolute_delta"] = max_decode_path_delta
    usage = [codebook_usage(row) for row in code_counts]
    gate_result = apply_gate(level_results, usage, protocol)
    report = {
        "format_version": 1,
        "is_pipeline_validation": bool(dataset.manifest["is_pipeline_validation"]),
        "warning": (
            "DEBUG DATASET: PIPELINE VALIDATION ONLY; NOT A SCIENTIFIC RESULT"
            if dataset.manifest["is_pipeline_validation"]
            else None
        ),
        "protocol_sha1": sha1_file(args.protocol),
        "model": str(args.model.resolve()),
        "data_manifest": dataset.manifest,
        "normalization": args.normalization,
        "split": args.split,
        "seed": args.seed,
        "evaluated_batches": batches,
        "evaluated_chunks": evaluated_chunks,
        "levels": level_results,
        "codebook_usage": usage,
        "code_distribution": code_counts.tolist() if args.include_code_distribution else None,
        "codec_gate": gate_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"codec_gate": gate_result, "output": str(args.output)}, indent=2))
    return 2 if args.enforce_gate and not gate_result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
