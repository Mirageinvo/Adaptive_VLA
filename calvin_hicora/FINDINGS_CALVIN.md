# CALVIN findings and deviations

## Reporting rules

- Record every deviation from the registered plan as a separate entry.
- Each entry must include the date, the planned behavior, the actual behavior,
  the reason, the decision, and links or hashes for supporting artifacts.
- Record negative results with the same level of detail as positive results.
- Write an entry when the decision is made, not retrospectively during report
  preparation.
- Never present debug-dataset measurements as scientific results.

## Entry template

### YYYY-MM-DD — Short title

- **Status:** deviation / negative result / positive result / infrastructure
- **Planned:**
- **Observed:**
- **Reason:**
- **Decision:**
- **Artifacts:** paths, hashes, logs, checkpoints

## Entries

### 2026-09-13 — Initial cluster capacity audit

- **Status:** infrastructure
- **Planned:** Run the full pipeline on the initial MIPT host.
- **Observed:** The host has two Tesla V100-SXM2 32 GiB GPUs connected by
  NVLink, but they are shared without a scheduler. Only 164 GiB disk space was
  free, less than the safe 500 GiB requirement for `task_D_D`, extracted data,
  caches, and checkpoints.
- **Reason:** Shared host storage was 92% full; one GPU was actively occupied.
- **Decision:** Implement and validate the debug pipeline first. Run
  ActionCodec when one GPU is available. Move the full 2.2B stage to another
  cluster if two exclusive GPUs and sufficient storage cannot be secured.
- **Artifacts:** `protocols/codec_protocol.json`; cluster audit performed via
  `nvidia-smi`, `df`, and Docker inspection.

### 2026-09-13 — V100-compatible precision and attention

- **Status:** infrastructure
- **Planned:** The upstream BAR recipe uses `bf16-mixed` and
  `flash_attention_2`.
- **Observed:** Tesla V100 (Volta, sm_70) does not support bf16 or
  FlashAttention-2.
- **Reason:** These features require newer GPU architectures; retaining them
  would make the registered run impossible on `ccmplanner.mipt.ru`.
- **Decision:** Freeze `16-mixed` (fp16 with loss scaling) and `sdpa` before
  observing VLA results. These settings may not be changed in response to
  validation or test outcomes.
- **Artifacts:** `configs/fullbar_calvin_2b.yaml`;
  `protocols/codec_protocol.json`.

### 2026-09-13 — Primary VLA backbone and learning rate

- **Status:** protocol decision
- **Planned:** Resolve the mismatch between the upstream 256M default and
  CALVIN specification section 12, which requests SmolVLM2-2.2B.
- **Observed:** Although the released `bar.yaml` defaults to 256M, the
  ActionCodec paper Appendix A.3 Table 7 explicitly evaluates all tokenizers
  with full-parameter SmolVLM2-2.2B using peak `lr=1e-4`, 1k warmup steps,
  cosine decay to `1e-5`, global batch 128, and 30k training steps.
- **Reason:** The project requires the full 2.2B model and its 24-layer
  `layer 12 -> layer 24` HiCoRA contract.
- **Decision:** Freeze
  `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, full-parameter FSDP training,
  and the Table 7 schedule (`lr=1e-4`, global batch 128, 30k steps). The VLA
  GPU-hour budget remains explicitly unfrozen until a required 200-step
  benchmark on two exclusive V100s; the benchmark may set the budget but may
  not change backbone, precision, attention, metrics, or test policy.
- **Artifacts:** `configs/fullbar_calvin_2b.yaml`;
  `protocols/codec_protocol.json`;
  upstream `third_party/actioncodec/config/train/bar.yaml`.

### 2026-09-13 — CALVIN-native staged ActionCodec is primary

- **Status:** protocol decision
- **Planned:** Train the tokenizer according to CALVIN specification section
  11.3 and the ActionCodec RVQ post-training procedure.
- **Observed:** ActionCodec section 5 and Appendix C specify a single-layer VQ
  first, followed by an RVQ depth-3 model that inherits and freezes the
  encoder and primary codebook while training the decoder and residual
  codebooks. Appendix A.3 reports batch 8192, `lr=2e-4`, and 100k tokenizer
  steps, but does not disclose a separate RVQ post-training step count.
- **Reason:** Training a three-level RVQ from scratch in one phase would not
  preserve the primary-token topology and would violate the published method.
- **Decision:** Make staged CALVIN-native VQ -> frozen-primary RVQ the only
  primary codec path. A pretrained transfer run is descriptive only and may
  not rescue a failed primary. Leave the undisclosed RVQ post-training step
  count explicitly unresolved rather than inventing it.
- **Artifacts:** ActionCodec arXiv 2602.15397 sections 5, A.3, and Appendix C;
  `protocols/codec_protocol.json`; `configs/actioncodec_calvin.yaml`.

### 2026-09-13 — VLA and codec optimizer settings have different provenance

- **Status:** protocol decision
- **Planned:** Avoid silently sharing optimizer defaults between tokenizer and
  VLA training.
- **Observed:** The released VLA `bar.yaml` uses AdamW betas `(0.9, 0.999)`
  and `weight_decay=1e-10`; Table 7 reports the VLA learning-rate schedule but
  does not report betas or weight decay. A full-text search of arXiv 2602.15397
  likewise finds no tokenizer AdamW betas or weight decay. The tokenizer
  training script was not released, so its complete optimizer settings cannot
  be recovered from the repository.
- **Reason:** Treating VLA and codec optimizer settings as interchangeable
  would claim unsupported methodological equivalence.
- **Decision:** VLA uses the released-code betas and effectively zero weight
  decay, clearly labelled with their source. Codec optimizer provenance is
  tracked separately; the paper-reported batch 8192, `lr=2e-4`, and 100k
  steps are authoritative, while undisclosed fields remain identified as
  implementation assumptions rather than paper facts.
- **Artifacts:** `protocols/baseline_protocol.json`;
  `protocols/codec_protocol.json`; `third_party/actioncodec/config/train/bar.yaml`.

### 2026-09-13 — Real CALVIN debug pipeline validation

- **Status:** infrastructure
- **Planned:** Validate fetch, conversion, staged VQ -> RVQ training, and
  reconstruction evaluation on the 1.3 GB debug dataset before using GPU time.
- **Observed:** The official archive SHA256 matched. Conversion produced 4,446
  frames, two episodes, and 4,388 stride-1 chunks at approximately 1,922
  frames/s; strict scene containment passed, both episodes intersect language
  annotations, and zero frames required clipping. Production base-VQ and
  frozen-primary RVQ smoke runs completed with finite losses; evaluator
  decode-path parity passed.
- **Reason:** This is the registered pipeline validation stage.
- **Decision:** Proceed to a one-GPU tokenizer hardware benchmark. Do not
  interpret reconstruction or codec-gate values after ten smoke steps as
  scientific results; gate enforcement is programmatically forbidden for
  debug manifests.
- **Artifacts:** `data/calvin_converted/debug/data_manifest.json`;
  `data/calvin_runs/debug_base_vq_s0`;
  `data/calvin_runs/debug_rvq_s0/eval_debug.json`.

### 2026-09-13 — Architecture source of truth and pretrained freeze policy

- **Status:** protocol decision
- **Planned:** Avoid silent YAML/code divergence for ActionCodec geometry and
  architecture, and avoid treating pretrained transfer as a primary CALVIN
  training arm.
- **Observed:** `ActionCodecConfig` defaults are `encoder_dim=256` /
  `encoder_n_layers=6`, while the released ActionCodec-Base geometry uses 384 /
  12. The HF RVQft checkpoint embodiments are named `a_franka_libero_20hz`, not
  the code-default `franka_libero_20hz`.
- **Reason:** Hardcoded builder fields and YAML copies can diverge; pretrained
  transfer is not the Appendix C / §11.3 primary path.
- **Decision:** `protocols/codec_protocol.json` `architecture` + `geometry` are
  the source of truth for builder construction. Primary training remains staged
  VQ then frozen-primary RVQ. Pretrained transfer stays descriptive-only; if
  run later, freeze the inherited trunk and train only the new CALVIN
  embodiment soft-prompt. Codec precision remains explicit fp32 with no
  autocast.
- **Artifacts:** `protocols/codec_protocol.json`;
  `scripts/codec_common.py`; `scripts/train_codec.py`.

### 2026-09-13 — Council audit: SPEC compliance and A* readiness

- **Status:** negative result / protocol decision
- **Planned:** Multi-model council review of `calvin_hicora/` against
  `HiCoRA_CALVIN_SPEC.md` for plan fidelity and A*-grade representativeness.
- **Observed:** Primary scientific choices (2.2B, staged Appendix C codec,
  Table 7 VLA schedule, pretrained demoted) are correctly registered. Blocking
  issues: wrong Hugging Face backbone ID (`…-Video-Instruct` does not exist;
  real ID is `SmolVLM2-2.2B-Instruct`); invented numeric codec-gate thresholds
  without paper/SPEC provenance; no development split carved from train;
  Stage 0/1 evaluator/HULC artifacts absent; VLA trainer not implemented; A*
  reviewers would demand ABC→D, stronger one-pass controls, and a frozen
  statistical gate. Implementation bugs: vendored single-process residual
  codebook zero-init; primary freeze only via `.eval()` undone by
  `model.train()`; global batch 8192 not enforced in code.
- **Reason:** Early infrastructure focus without fail-closed enforcement of
  registered invariants, plus silent invention of operational thresholds.
- **Decision:** Fix backbone ID and residual/primary freeze before any
  scientific codec run. Demote numeric gate thresholds to provisional
  candidates. Keep scientific training blocked on undisclosed optimizer and
  RVQ-step fields. Do not treat 256M as a scientific substitute. Record A*
  gaps as open design work after Stage 1–3, not as silent protocol changes.
- **Artifacts:** council reviews; `protocols/*.json`; this entry.

### 2026-09-13 — Single-GPU residual codebook init and primary hard-freeze

- **Status:** deviation / bugfix
- **Planned:** Appendix C RVQ inheritance with live residual codebooks and a
  frozen primary book on 1×V100 without editing vendored ActionCodec SHA.
- **Observed:** Vendored `VectorQuantize.init_codebook` /
  `replace_dead_codes` write zeros when `torch.distributed` is not
  initialized. Primary `.eval()` is cleared by `model.train()`, allowing EMA
  to mutate the inherited book.
- **Reason:** Upstream assumes DDP; CALVIN codec path is single-process.
- **Decision:** Runtime-patch single-process k-means/dead-code ops in
  `codec_common.py`; suppress residual init during the inheritance probe;
  warm residual books from a real train batch; hard-freeze primary via
  no-op EMA/dead-code plus snapshot assert. Log as forced infrastructure
  workaround; do not mutate `third_party/actioncodec/`.
- **Artifacts:** `scripts/codec_common.py`; `scripts/train_codec.py`;
  `scripts/selftest_codec.py`.

### 2026-09-13 — Episode-disjoint development split registered

- **Status:** protocol decision
- **Planned:** Separate model-selection data from the official CALVIN
  validation domain before any scientific codec or VLA run.
- **Observed:** Converter previously mapped official `training/` → `train` and
  `validation/` → `val`, and early stopping used `val`.
- **Reason:** SPEC requires development-split selection; official validation
  scenes/start states feed the long-horizon evaluation domain.
- **Decision:** Carve `dev` as an episode-disjoint 5% sample from official
  training episodes (seed 0, scene-stratified). Norm stats remain train-only
  after the carve. Official `validation/` stays evaluation-domain only.
  Converted format_version becomes 2. Trainer/eval selection defaults to
  `dev`. HULC reproduction is scaffolded under `evaluator/` and remains a
  closed-loop blocker only.
- **Artifacts:** `scripts/convert_calvin.py`; `scripts/codec_data.py`;
  `scripts/train_codec.py`; `scripts/eval_codec_recon.py`;
  `protocols/codec_protocol.json`; `protocols/baseline_protocol.json`;
  `evaluator/README.md`.

### 2026-09-13 — Codec GPU smoke / memory-smoke on V100-0

- **Status:** infrastructure / positive result
- **Planned:** Measure VRAM and throughput for the registered micro-batch
  candidate before a 200-step budget benchmark.
- **Observed:** Inside `avla_hicora_askhabaliev_gs` on Tesla V100-SXM2-32GB
  (GPU 0), torch 2.4.1+cu124:
  - `--smoke` (batch capped to 4): peak **641.7 MiB**, **3.218** optimizer
    steps/s, finite loss;
  - `--memory-smoke --batch-size 256` (accum forced to 1): peak
    **3522.4 MiB**, **0.909** optimizer steps/s, finite loss;
  - DataLoader `num_workers=8` + `persistent_workers` + `pin_memory` passed
    on the debug converted set.
- **Reason:** Action-only fp32 codec has large headroom under 32 GiB; the
  registered global batch 8192 is therefore achievable with a much larger
  micro-batch and small accumulation, not only 256×32.
- **Decision:** Keep global batch 8192. Treat 256×32 as the conservative
  candidate already used by the running 200-step benchmark. After that
  benchmark, re-measure peak VRAM for micro-batch candidates 2048/4096/8192
  and register the chosen `(micro, accum)` pair in protocol metadata without
  changing the global batch. `--smoke` and `--memory-smoke` correctly skip
  the `micro × accum == 8192` assert; `--benchmark` enforces it.
- **Artifacts:** container `avla_hicora_askhabaliev_gs`;
  `outputs/calvin_codec_gpu_smoke/base_vq_smoke/`;
  `outputs/calvin_codec_gpu_smoke/base_vq_memory/hardware_probe.json`.

### 2026-09-13 — Codec 200-step benchmark and micro-batch selection

- **Status:** infrastructure / protocol decision
- **Planned:** 200-step hardware benchmark at global batch 8192; then choose
  `(micro, accum)` without changing the global batch.
- **Observed:** On V100-0 / `avla_hicora_askhabaliev_gs`:
  - 200-step `256×32`: **0.213** opt-steps/s, peak **3652 MiB**, loss/grad
    finite → ~**130.6** GPU-h / 100k steps;
  - memory-smoke `2048×1`: peak **24928 MiB**, **0.415** steps/s;
  - timed 50-step `2048×4`: **0.230** opt-steps/s, peak **25053 MiB**,
    effective batch 8192 → ~**120.8** GPU-h / 100k steps;
  - `4096`/`8192` single-micro not VRAM-tested on debug (only 2742 train
    chunks with `drop_last`); linear scaling from 2048 suggests OOM.
  - `--smoke` / `--memory-smoke` skip `micro×accum==8192`; `--benchmark`
    enforces it (confirmed).
- **Reason:** Larger micro-batch uses the spare VRAM and is slightly faster
  per optimizer step than 256×32 while preserving the paper global batch.
- **Decision:** Register execution as **micro 2048, accum 4, global 8192**.
  Freeze base-VQ GPU budget at ~120.8 h/seed on this host. RVQ post-train
  budget remains open until step count is disclosed or sensitivity-frozen.
- **Artifacts:** `results/codec/benchmark.json`;
  `outputs/calvin_codec_gpu_smoke/base_vq_benchmark/`;
  `outputs/calvin_codec_gpu_smoke/base_vq_benchmark_2048x4/`;
  `protocols/codec_protocol.json`.
