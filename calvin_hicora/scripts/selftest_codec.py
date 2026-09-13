#!/usr/bin/env python3
"""CPU self-tests for conversion, ActionCodec geometry, and training paths."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from codec_common import (
    ActionCodec,
    ActionCodecConfig,
    enforce_rvq_frozen_primary,
    expand_pretrained_with_gate,
    grouped_reconstruction_loss,
    initialize_rvq_from_base,
    load_json,
    seed_everything,
    warm_residual_codebook_init,
)
from codec_data import CalvinActionChunkDataset


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=root / "protocols" / "codec_protocol.json"
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        help="Run the mandatory real expand_embodiment gate (downloads weights if needed).",
    )
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument(
        "--trainer-smoke",
        action="store_true",
        help="Run two steps for production base VQ and RVQ post-training, then eval.",
    )
    return parser.parse_args()


def write_fake_split(path: Path, split: str, start: int, n_frames: int) -> None:
    path.mkdir(parents=True)
    midpoint = start + n_frames // 2
    bounds = np.asarray([[start, midpoint - 1], [midpoint, start + n_frames - 1]])
    np.save(path / "ep_start_end_ids.npy", bounds)
    np.save(
        path / "scene_info.npy",
        {"calvin_scene_D": [start, start + n_frames - 1]},
        allow_pickle=True,
    )
    language_dir = path / "lang_annotations"
    language_dir.mkdir()
    np.save(
        language_dir / "auto_lang_ann.npy",
        {
            "language": {
                "ann": [f"fake {split} instruction"],
                "task": ["fake_task"],
                "emb": np.zeros((1, 4), dtype=np.float32),
            },
            "info": {"indx": [[start + 2, start + 8]]},
        },
        allow_pickle=True,
    )
    rng = np.random.default_rng(start)
    base = rng.uniform(-0.8, 0.8, size=(n_frames, 6)).astype(np.float32)
    base = np.cumsum(base * 0.05, axis=0)
    base /= max(float(np.abs(base).max()), 1.0)
    gripper = np.where(np.arange(n_frames) % 11 < 6, 1.0, -1.0)[:, None]
    actions = np.concatenate([base, gripper], axis=1).astype(np.float32)
    for offset, frame_id in enumerate(range(start, start + n_frames)):
        np.savez_compressed(
            path / f"episode_{frame_id:07d}.npz",
            rel_actions=actions[offset],
            actions=actions[offset],
            robot_obs=np.zeros(15, dtype=np.float32),
            scene_obs=np.zeros(24, dtype=np.float32),
            rgb_static=np.zeros((200, 200, 3), dtype=np.uint8),
            rgb_gripper=np.zeros((84, 84, 3), dtype=np.uint8),
        )


def converter_dataset_test(protocol_path: Path, temporary: Path) -> Path:
    source = temporary / "calvin_debug_dataset"
    write_fake_split(source / "training", "train", 100, 80)
    write_fake_split(source / "validation", "val", 300, 80)
    converted = temporary / "converted"
    converter = Path(__file__).with_name("convert_calvin.py")
    subprocess.run(
        [
            "python3",
            str(converter),
            "--dataset-root",
            str(source),
            "--output-root",
            str(converted),
            "--pipeline-validation",
        ],
        check=True,
    )
    manifest = json.loads((converted / "data_manifest.json").read_text())
    assert manifest["is_pipeline_validation"] is True
    assert manifest["format_version"] == 2
    assert manifest["num_frames"] == 160
    assert manifest["development_split"]["enabled"] is True
    assert manifest["development_split"]["dev_episode_count"] >= 1
    episodes = json.loads((converted / "episode_index.json").read_text())["episodes"]
    assert all(record["scene"] == "D" for record in episodes)
    assert all(record["n_frames"] == 40 for record in episodes)
    assert any(record["has_lang_ann"] for record in episodes)
    assert any(record["split"] == "dev" for record in episodes)
    assert any(record["split"] == "train" for record in episodes)

    dataset = CalvinActionChunkDataset(
        converted, split="train", protocol_path=protocol_path, verify_hashes=True
    )
    dev_dataset = CalvinActionChunkDataset(
        converted, split="dev", protocol_path=protocol_path, verify_hashes=True
    )
    assert len(dataset) + len(dev_dataset) == 22
    assert len(dev_dataset) >= 1
    sample = dataset[0]
    assert sample["action"].shape == (30, 7)
    assert sample["padding_mask"].all()
    chunks = np.load(converted / "chunk_index.npy")
    for chunk in chunks:
        episode = episodes[int(chunk["episode_id"])]
        start = int(chunk["action_start"])
        assert episode["action_start"] <= start
        assert start + 30 <= episode["action_end_exclusive"]
    return converted


def tiny_codec(codebook_size: int = 64) -> ActionCodec:
    config = ActionCodecConfig(
        embodiment_config={
            "franka_calvin_30hz": {
                "action_dim": 7,
                "freq": 30,
                "duration": 1,
                "description": "self-test",
            }
        },
        n_tokens=48,
        n_quantizers=3,
        z_dim=32,
        vq_type="rvq",
        vq_codebook_size=codebook_size,
        vq_kmeans_init=False,
        vq_quantizer_dropout=0.25,
        encoder_dim=64,
        encoder_n_layers=1,
        encoder_n_heads=4,
        decoder_dim=64,
        decoder_n_layers=1,
        decoder_n_heads=4,
    )
    return ActionCodec(config)


def tiny_base_vq(codebook_size: int = 64) -> ActionCodec:
    config = ActionCodecConfig(
        embodiment_config={
            "franka_calvin_30hz": {
                "action_dim": 7,
                "freq": 30,
                "duration": 1,
                "description": "self-test",
            }
        },
        n_tokens=16,
        n_quantizers=1,
        z_dim=32,
        vq_type="vq",
        vq_codebook_size=codebook_size,
        vq_kmeans_init=False,
        encoder_dim=64,
        encoder_n_layers=1,
        encoder_n_heads=4,
        decoder_dim=64,
        decoder_n_layers=1,
        decoder_n_heads=4,
    )
    return ActionCodec(config)


def staged_inheritance_test(protocol: dict, actions: torch.Tensor) -> None:
    test_protocol = copy.deepcopy(protocol)
    test_protocol["geometry"]["latent_dim"] = 32
    test_protocol["geometry"]["codebook_size"] = 64
    base = tiny_base_vq()
    base.eval()
    _ = base.encode(actions, embodiment_ids=0)
    rvq, _, report = initialize_rvq_from_base(base, test_protocol)
    assert report["passed"]
    assert report["encoder_trainable_parameters"] == 0
    assert not rvq.vq.quantizers[0].training
    warm = warm_residual_codebook_init(rvq, actions, 0)
    assert warm["passed"]
    rvq.train()
    enforce_rvq_frozen_primary(rvq)
    frozen = rvq.vq.quantizers[0].codebook.clone()
    ids = torch.zeros(actions.shape[0], dtype=torch.long)
    _ = rvq(actions, embodiment_ids=ids)
    assert torch.equal(rvq.vq.quantizers[0].codebook, frozen)


@torch.no_grad()
def geometry_test(model: ActionCodec, actions: torch.Tensor) -> None:
    model.eval()
    ids = torch.zeros(actions.shape[0], dtype=torch.long)
    z_e = model._encode(actions, ids)
    assert z_e.shape == (actions.shape[0], 16, 32)
    z_full, codes, _, _, _ = model.vq(z_e, n_quantizers=3)
    assert codes.shape == (actions.shape[0], 16, 3)
    decoded, mask = model._decode(z_full, ids)
    assert decoded.shape == (actions.shape[0], 30, 7)
    assert mask.shape == (actions.shape[0], 30)
    assert torch.isfinite(decoded).all()
    for levels in (1, 2, 3):
        z_codes, _, _ = model.vq.from_codes(codes[..., :levels])
        z_direct, direct_codes, _, _, _ = model.vq(z_e, n_quantizers=levels)
        assert torch.equal(codes[..., :levels], direct_codes)
        assert torch.equal(z_codes, z_direct)


def overfit_once(
    actions: torch.Tensor, steps: int, bypass_quantizer: bool
) -> tuple[float, float]:
    seed_everything(7 if bypass_quantizer else 11)
    model = tiny_codec().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    ids = torch.zeros(actions.shape[0], dtype=torch.long)
    mask = torch.ones(actions.shape[:2], dtype=torch.bool)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        z_e = model._encode(actions, ids, mask)
        if bypass_quantizer:
            z = z_e
            quantization_loss = torch.zeros(())
        else:
            z, _, _, quantization_loss = model._quantize(z_e, return_perplexity=False)
        prediction, recon_mask = model._decode(z, ids)
        reconstruction, _ = grouped_reconstruction_loss(
            prediction[..., :7], actions, recon_mask
        )
        loss = reconstruction + (0.1 * quantization_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite overfit loss")
        loss.backward()
        optimizer.step()
        losses.append(float(reconstruction.detach()))
    return losses[0], min(losses[-10:])


def overfit_test(actions: torch.Tensor, protocol: dict) -> dict[str, float]:
    gate = protocol["overfit_gate"]
    steps = int(gate["steps"])
    bypass_initial, bypass_final = overfit_once(actions, steps, True)
    quantized_initial, quantized_final = overfit_once(actions, steps, False)
    bypass_orders = float(np.log10(bypass_initial / max(bypass_final, 1e-12)))
    quantized_orders = float(
        np.log10(quantized_initial / max(quantized_final, 1e-12))
    )
    if bypass_orders < float(gate["minimum_bypass_mse_reduction_orders"]):
        raise AssertionError(f"Bypass overfit gate failed: {bypass_orders:.3f} orders")
    if quantized_orders < float(gate["minimum_quantized_mse_reduction_orders"]):
        raise AssertionError(
            f"Quantized overfit gate failed: {quantized_orders:.3f} orders"
        )
    return {
        "bypass_initial_mse": bypass_initial,
        "bypass_final_mse": bypass_final,
        "bypass_reduction_orders": bypass_orders,
        "quantized_initial_mse": quantized_initial,
        "quantized_final_mse": quantized_final,
        "quantized_reduction_orders": quantized_orders,
    }


def production_trainer_smoke(
    converted: Path, protocol_path: Path, temporary: Path
) -> dict[str, str]:
    trainer = Path(__file__).with_name("train_codec.py")
    evaluator = Path(__file__).with_name("eval_codec_recon.py")
    base_output = temporary / "base_vq_smoke"
    subprocess.run(
        [
            "python3",
            str(trainer),
            "--data-root",
            str(converted),
            "--output-dir",
            str(base_output),
            "--stage",
            "base_vq",
            "--seed",
            "0",
            "--protocol",
            str(protocol_path),
            "--device",
            "cpu",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--smoke",
        ],
        check=True,
    )
    base_model = base_output / "step_00000002" / "model"
    rvq_output = temporary / "rvq_smoke"
    subprocess.run(
        [
            "python3",
            str(trainer),
            "--data-root",
            str(converted),
            "--output-dir",
            str(rvq_output),
            "--stage",
            "rvq_posttrain",
            "--base-vq-checkpoint",
            str(base_model),
            "--seed",
            "0",
            "--protocol",
            str(protocol_path),
            "--device",
            "cpu",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--smoke",
        ],
        check=True,
    )
    model = rvq_output / "step_00000002" / "model"
    report = temporary / "trainer_smoke_eval.json"
    subprocess.run(
        [
            "python3",
            str(evaluator),
            "--model",
            str(model),
            "--data-root",
            str(converted),
            "--output",
            str(report),
            "--protocol",
            str(protocol_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--max-batches",
            "1",
        ],
        check=True,
    )
    payload = json.loads(report.read_text())
    assert payload["is_pipeline_validation"] is True
    assert payload["evaluated_batches"] == 1
    return {"training": "passed", "evaluation": "passed"}


def main() -> int:
    args = parse_args()
    protocol = load_json(args.protocol)
    seed_everything(0)
    with tempfile.TemporaryDirectory(prefix="calvin_codec_selftest_") as directory:
        converted = converter_dataset_test(args.protocol, Path(directory))
        dataset = CalvinActionChunkDataset(converted, split="train")
        actions = torch.stack([dataset[index]["action"] for index in range(8)])
        model = tiny_codec(int(protocol["overfit_gate"]["test_codebook_size"]))
        geometry_test(model, actions)
        staged_inheritance_test(protocol, actions)
        report: dict[str, object] = {
            "converter_dataset": "passed",
            "geometry_decode_path_parity": "passed",
            "vq_to_rvq_inheritance": "passed",
        }
        if not args.skip_overfit:
            report["overfit"] = overfit_test(actions, protocol)
        if args.trainer_smoke:
            report["production_trainer"] = production_trainer_smoke(
                converted, args.protocol, Path(directory)
            )

    if args.pretrained_checkpoint:
        pretrained = ActionCodec.from_pretrained(args.pretrained_checkpoint)
        _, expansion_report = expand_pretrained_with_gate(
            pretrained, protocol, torch.device("cpu")
        )
        report["expand_embodiment"] = expansion_report
    else:
        report["expand_embodiment"] = (
            "not_run; required only for the descriptive pretrained-transfer ablation"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
