# CALVIN HiCoRA port

This directory contains the CALVIN-native ActionCodec and HiCoRA pipeline. It
does not modify the frozen LIBERO implementation under `experiments/` or
`third_party/actioncodec/`.

## Scope

1. Convert official CALVIN transitions without changing action semantics.
2. Train the primary CALVIN-native ActionCodec staged path (single-layer VQ,
   then Appendix C RVQ with frozen encoder + primary codebook) with two seeds.
   Pretrained LIBERO transfer is descriptive/secondary only.
3. Evaluate encode/quantize/decode reconstruction at RVQ levels 0, 0+1, and
   0+1+2 using the frozen protocol.
4. Train a full-parameter SmolVLM2-2.2B-Instruct ActionCodec-BAR policy with
   two seeds.
5. Add `coarse_full`, `joint_early`, and deterministic HiCoRA-D after the
   codec and full BAR gates pass.

## Registered protocol

`protocols/codec_protocol.json` is the normative source for normalization,
geometry, arm selection, go/no-go thresholds, permitted ablations, and compute
budget. Evaluation must fail closed when a required field is absent.

CALVIN `rel_actions` are consumed directly in their native `[-1, 1]` scale:

- position channels are already scaled by 50;
- rotation channels are already scaled by 20;
- gripper is `-1` closed and `+1` open;
- no LIBERO gripper-sign flip is applied.

## Dataset provenance

Before a real run, fill and archive:

- CALVIN repository commit;
- Adaptive_VLA commit;
- ActionCodec and SmolVLM checkpoint revisions;
- official archive SHA256;
- corrected language annotations and `scene_info.npy` provenance;
- Python, PyTorch, CUDA, MuJoCo, driver, GPU and host details;
- exact dataset paths and generated manifest SHA256.

The converter emits:

- compact `actions.npy`;
- `episode_index.json` with `scene`, `split`, `n_frames`, and `has_lang_ann`;
- `chunk_index.npy`, asserted not to cross episode or scene boundaries;
- train-only `norm_stats.json`;
- `data_manifest.json`.

## Result validity

Results from `calvin_debug_dataset` validate only the pipeline. They are not
scientific results and must carry `"is_pipeline_validation": true`.

Hyperparameter selection uses the carved episode-disjoint `dev` split taken
from official `training/` (5%, seed 0, scene-stratified). Official CALVIN
`validation/` is evaluation-domain only and must not drive model selection.
CALVIN test and long-horizon rollouts remain closed until the relevant choices
are frozen.

## Hardware plan

ActionCodec uses one V100 in fp32. Full SmolVLM2-2.2B training requires both
32 GiB V100s exclusively, fp16 mixed precision, SDPA, FSDP full sharding, and
activation checkpointing. LoRA and smaller backbones are not substitutes for
the registered primary experiment.
