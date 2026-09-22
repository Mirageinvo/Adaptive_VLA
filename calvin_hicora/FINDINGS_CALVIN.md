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

### 2026-09-22 — Veto: do not extrapolate 256M VRAM to 2.2B production

- **Status:** deviation / negative result (fail-closed)
- **Planned:** Launch Stage-3 SmolVLM2-2.2B FSDP production (30k steps) on
  aicenter1 after the 256M `--debug-overfit` VRAM profile (+~7.9 / +7.5 GiB
  incremental) looked to leave large A100 headroom.
- **Observed:** aicenter1 was not exclusive: 6× foreign `root` PPO jobs held
  ~21 / ~13 GiB; load average ~199; disk 100% with only ~122 GiB free. The
  256M overfit delta is not a linear forecast for 2.2B weights + VLM
  activations. Protocol (`vla_memory_probe --require-exclusive-gpus`) and
  AGENT_PLAN require exclusive GPUs before 2.2B.
- **Reason:** Shared-node launch risks OOM against foreign jobs, host-wide
  disk fill from unbounded step trees, and an invalid hardware gate.
- **Decision:** Hard veto on production train while PPO lives. Implement
  atomic rolling checkpoints (`latest/` + `latest_backup/` + `best/` only),
  prepare an honest exclusive 2.2B memory probe, and auto-start a 10-step
  probe only when each GPU reports `<2000 MiB` used with no compute apps.
- **Artifacts:** `third_party/actioncodec/scripts/train_vla.py`
  (`RollingDiskCheckpoint`); `calvin_hicora/scripts/run_aicenter_vla_2b_memory_probe.sh`;
  `calvin_hicora/scripts/wait_exclusive_then_probe.sh`;
  `config/train/bar_calvin_production_s1.yaml`.

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

### 2026-09-18 — CALVIN loss, short-episode padding, and rollout contracts

- **Status:** pre-scientific protocol deviation / implementation hardening
- **Observed:** The vendored decoder emits unconstrained linear outputs.
  Therefore a binary gripper objective must consume channel 6 as a logit, not
  as a probability. The old converter omitted episodes shorter than 30
  frames, and training/evaluation used the decoder-generated all-valid mask
  rather than intersecting it with the input data mask. Closed-loop horizon
  values existed only as metadata.
- **Decision:** Before any scientific CALVIN run, freeze continuous-channel
  MSE plus `0.1 * BCEWithLogits(gripper, (target+1)/2)` as a CALVIN-specific
  primary deviation. Convert short episodes into one 30-step chunk by
  repeating the final legitimate action and store `valid_length`; padded
  timesteps are masked out of encoder attention, loss, and evaluation.
  Add a fail-closed rollout core that plans 30, executes 12, discards 18, and
  asserts simulator `dt == 1/30 s`.
- **Caveat:** The claim that the previous equally per-channel MSE was
  necessarily gripper-dominated was not established empirically. This change
  is an explicit CALVIN protocol decision, not attributed to the ActionCodec
  paper. Existing debug caches are legacy format v2; scientific caches must
  be regenerated as format v3. The earlier throughput benchmark remains
  hardware evidence, but its observed loss values are not comparable to the
  new objective.
- **Artifacts:** `scripts/codec_common.py`, `scripts/convert_calvin.py`,
  `scripts/codec_data.py`, `scripts/train_codec.py`,
  `scripts/eval_codec_recon.py`, `evaluator/receding_horizon.py`,
  `protocols/codec_protocol.json`.

### 2026-09-18 — Variant B: stream zip → actions-only cache (no full unzip)

- **Status:** infrastructure / protocol decision
- **Planned:** Scientific `task_D_D` conversion previously assumed unzip onto
  disk (~166 GiB extracted + archive), which no shared MIPT host currently
  has free.
- **Observed:** Cluster disk free space is far below the old 500 GiB unzip
  budget (see companion audit entry). ActionCodec training only needs
  relative actions; RGB would only be required later for VLA.
- **Reason:** Full unzip doubles storage and is unnecessary for the codec
  stage. Streaming members via `zipfile.ZipFile.open` → `np.load` keeps peak
  disk near `sizeof(zip) + sizeof(actions.npy≈70MiB) + indices`.
- **Decision:** Adopt variant B. New converter
  `scripts/convert_from_zip_actions_only.py`:
  1. hash source zip SHA-256 before any deletion;
  2. stream each `episode_XXXXXXX.npz`, retain only `--action-key`
     (default `rel_actions`);
  3. write contiguous `actions.npy` memmap plus `episode_index.json` /
     `episode_index.npz` so stride-1 length-30 windows never cross episode
     or scene boundaries;
  4. document `modalities.actions_only=true` and
     `rgb_present_in_cache=false` in `data_manifest.json` so future VLA
     code cannot silently read images from this cache.
  Directory-based `convert_calvin.py` remains valid for already-extracted
  trees. Memory plan for training stays mmap-only (below).
- **Artifacts:** `scripts/convert_from_zip_actions_only.py`.

### 2026-09-18 — Memory-tight load plan (conversion + training)

- **Status:** infrastructure
- **Planned:** Avoid RAM/disk inflation while preparing and training on
  CALVIN actions.
- **Decision (conversion):** Never materialize RGB. One transition at a time
  from the zip stream; write `actions.npy` through `np.lib.format.open_memmap`.
  Episode metadata (`ep_start_end_ids`, `scene_info`, lang index) is tiny and
  may be loaded whole. Optional `--delete-zip-after` only after SHA-256 is in
  the manifest.
- **Decision (training / eval):** `CalvinActionChunkDataset` already opens
  `actions.npy` and `chunk_index.npy` with `mmap_mode="r"`. Keep that.
  Do not concatenate full-split actions into RAM outside of one-time
  `norm_stats` computation at convert time. DataLoader workers share the
  mmap; prefer moderate `num_workers` over caching decoded chunks.
  Collate only the current micro-batch of `(B, 30, 7)` float32.
- **Artifacts:** `scripts/codec_data.py`; `scripts/convert_from_zip_actions_only.py`.

### 2026-09-18 — Three-host capacity re-audit (ccmplanner / cds2 / aicenter1)

- **Status:** infrastructure
- **Planned:** Pick a host for zip download + actions-only conversion and
  later codec runs.
- **Observed (2026-09-18):**
  - **ccmplanner** (`100.98.148.137`): RAM 62 GiB total / **~47 GiB
    available**; disk **~77 GiB** free; 2×V100; GPU util **0%** (only small
    idle eval/API processes ~0.3–1.7 GiB each). Least compute activity.
  - **cds2** (`100.98.2.11`): RAM 125 GiB total / **~73 GiB available**;
    disk **~136 GiB** free; 2×V100 both **~93–98%** busy with foreign PPO /
    RL runs. Most contested GPUs.
  - **aicenter1** (`100.98.208.203`): RAM 157 GiB total / **~123 GiB
    available** (most RAM); disk **~14 GiB** free (tightest disk); 2×A100
    partially busy (PPO on GPU0, vLLM ~72 GiB on GPU1).
- **Reason:** Disk, not RAM, remains the conversion bottleneck. Even with
  variant B, `task_D_D.zip` itself is ~166 GiB, so none of the three hosts
  currently has enough free disk to hold the archive.
- **Decision:** Prefer **ccmplanner** for low-activity codec work once disk
  is freed; **aicenter1** has the most spare RAM but cannot store the zip
  today. Do not start `task_D_D` download until ≥~180 GiB free on the chosen
  host (zip + actions cache + headroom). Variant B removes the old 500 GiB
  unzip requirement but does not remove the zip-sized download requirement.
- **Artifacts:** live `free` / `df` / `nvidia-smi` on the three SSH targets.

### 2026-09-18 — Network stream (no zip on disk): Range CD + sequential body

- **Status:** infrastructure / protocol decision
- **Planned:** Avoid storing `task_D_D.zip` (~177 GiB) on ccmplanner/aicenter1.
- **Observed:** Freiburg serves `Accept-Ranges: bytes` and HTTP 206. A giant
  `BytesIO` of the archive is forbidden (would require ~177 GiB RAM).
- **Decision:** `convert_from_zip_actions_only.py --source-url`:
  1. `HEAD` + `HttpRangeFile` + `zipfile` reads EOCD/central directory and
     sparse metadata;
  2. one sequential body download via `ResumableHttpStream` with byte-offset
     resume/retry and periodic memmap `flush()` + `.stream_checkpoint.json`;
  3. inflate only `episode_*.npz`, keep `rel_actions`, discard RGB;
  4. provenance from official `sha256sum.txt` (stream digest when started at
     byte 0 without mid-resume).
  `fetch_data.sh` now launches this path (disk budget ~5 GiB for D). Selftest
  covers the HTTP path. Long runs: `tmux new -s calvin_download` with
  `PYTHONUNBUFFERED=1`.
- **Artifacts:** `scripts/zip_http_stream.py`;
  `scripts/convert_from_zip_actions_only.py`; `scripts/fetch_data.sh`;
  `scripts/selftest_codec.py`.

### 2026-09-18 — VLA FSDP 200-step memory probe skeleton

- **Status:** infrastructure
- **Planned:** Close Block-4 risk (2.2B fp16+SDPA+FSDP on 2×V100) before the
  actions cache finishes, so an OOM does not waste a week.
- **Decision:** Add `scripts/vla_memory_probe.py`: fake vision/text batches,
  real forward/backward under GradScaler, FSDP FULL_SHARD, optional
  activation checkpointing, torch peak VRAM + periodic `nvidia-smi` samples.
  Dummy disconnected losses are forbidden (under-report activations). Launch
  with `torchrun --standalone --nproc_per_node=2`. Container must expose
  **both** GPUs (current `avla_hicora_askhabaliev_gs` DeviceRequests showed
  only GPU 0 — fix before the probe).
- **Artifacts:** `scripts/vla_memory_probe.py`; `protocols/baseline_protocol.json`.

### 2026-09-18 — Dirty FSDP OOM at step 0 (invalid hardware gate)

- **Status:** negative result / infrastructure / protocol hardening
- **Planned:** Clean 200-step dual-V100 FSDP probe for SmolVLM2-2.2B under
  fp16-mixed + SDPA.
- **Observed:** First probe OOMed on step 0 with
  `peak_vram_torch_mib ≈ 24843.6`. Batch shapes were
  `pixel_values (1, 34, 3, 384, 384)` and `input_ids (1, 2852)`. GPU0 also
  hosted foreign compute (earlier `replica_rgbd` ~6 GiB; later other idle
  eval processes), violating `require_exclusive_gpus`.
- **Reason:** (1) HuggingFace SmolVLM processor default
  `do_image_splitting=True` expands each camera frame into dozens of crops
  and matching `<image>` tokens — fatal activation blow-up on V100 for
  robot VLA. (2) The resulting token stream (~2852) is not a CALVIN
  instruction length. (3) Shared-node VRAM made the MiB reading unclean.
- **Decision:** Mark that run **invalid** for the hardware gate. Before any
  re-probe: force `processor.image_processor.do_image_splitting = False`,
  fixed square resize (default 224), hard-cap `input_ids` ≤ 128, and abort
  unless `nvidia-smi` shows no foreign compute apps. Re-run only on an
  exclusive 2×V100 window; record OS-level samples at steps 0/50/100/150/200.
  Note: total `input_ids` must still fit full SmolVLM `image_seq_len` groups
  (81/frame); a global clip of 128 corrupted two-camera placeholders — raise
  the cap to ≥256 while keeping splitting off (still ≪ 2852). Use
  `ShardedGradScaler` (not `torch.cuda.amp.GradScaler`) under FSDP+fp16 on
  Volta so Inf/NaN overflow masks sync across ranks.
- **Artifacts:** `outputs/vla_fsdp_memory_probe.json` (dirty);
  `scripts/vla_memory_probe.py` (contract updated).

### 2026-09-18 — Freeze codec AdamW betas/wd candidates to unlock base_vq

- **Status:** protocol decision / pre-scientific
- **Planned:** Start Stage-2 base VQ on the streamed task_D_D actions cache.
- **Observed:** `train_codec.py` fail-closes scientific runs while
  `betas_status` / `weight_decay_status` start with `not_disclosed`. Paper does
  not disclose tokenizer AdamW betas/wd.
- **Decision:** Freeze labelled implementation candidates
  `betas=(0.9, 0.95)`, `weight_decay=0.01` (not paper facts). Launch
  `base_vq` seed 0, micro 2048 × accum 4 = global 8192, `max_steps=100000`,
  `lr=2e-4`, data
  `/home/askhabaliev_gs/calvin_stream/converted_task_D_D`.
- **Artifacts:** `protocols/codec_protocol.json`;
  `configs/actioncodec_calvin.yaml`; tmux `codec_base_vq_s0`.

### 2026-09-18 — Rolling codec checkpoints (latest / latest_backup)

- **Status:** infrastructure / protocol hardening
- **Planned:** Protect the ~120 GPU-h base_vq seed from crash/OOM loss and
  from unbounded `step_XXXXXXXX/` disk growth on ccmplanner (~68 GiB free).
- **Decision:** Before step 1000, switch codec saves to rolling slots only:
  `best/` (val improve), `latest/` (resume), transient `latest_backup/` during
  atomic write. `checkpoint_interval` 5000 → **1000** (~1 h max loss at
  ~0.26 steps/s). No hot-reload — graceful restart from step 0 (~minutes).
- **Artifacts:** `scripts/train_codec.py`; `protocols/codec_protocol.json`;
  `configs/actioncodec_calvin.yaml`.

### 2026-09-21 — Freeze RVQ post-train lr/steps (close B2) and launch Stage-2

- **Status:** protocol decision / scientific Stage-2 start
- **Planned:** Stage-2 RVQ post-train from base_vq `best/` (s0 stop@52k,
  s1 stop@56k, early_stop 5/5).
- **Observed:** ActionCodec Appendix C does not disclose RVQ post-train lr or
  step count; `train_codec.py` fail-closes while statuses start with
  `not_disclosed`.
- **Decision:** Freeze labelled sensitivity candidates before scientific runs:
  `learning_rate=1e-4`, `max_steps=50000`, patience 5, checkpoint_interval 1000.
  Warm-start AdamW (no Stage-1 optimizer resume). Keep reconstruction selection
  metric (= MSE[:6] + 0.1·BCE gripper). Launch `rvq_posttrain` seeds 0/1 from
  matching `base_vq_s{seed}/best/model`.
- **Cluster note:** ccmplanner GPU0 occupied by foreign job → seed0 on
  `cuda:1`. aicenter1 seed1 container remapped to host GPU1 (`--gpus device=1`,
  visible as `cuda:0` inside).
- **Artifacts:** `protocols/codec_protocol.json`;
  `configs/actioncodec_calvin.yaml`; tmux `codec_rvq_s0` / `codec_rvq_s1`.
