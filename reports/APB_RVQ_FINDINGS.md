# APB-RVQ Gate-1 Findings

## Hypothesis
Adaptive per-position RVQ depth allocation improves reconstruction quality at matched token budgets versus non-adaptive baselines.

## Preregistered Thresholds
- Oracle must beat matched-random by >=15% on at least one key budget (`B=24`, `32`, or `40`).
- Oracle should also beat best non-adaptive baseline (uniform/static/global mix) by >=10% on key budgets.
- Cluster bootstrap 95% CI for oracle−random delta should exclude zero on key budgets.
- Greedy retained gain versus stronger search should be >=95% on validation subset.
- Episode-selection stability: >=3 seeds with the same qualitative conclusion.

## Procedure
- Smoke/invariants: `python experiments/rate0_smoke.py --n-chunks 32 --device cuda`
- Gate-1 oracle: `python experiments/rate1_oracle_greedy.py --n-chunks 2048 --budgets 16,20,24,28,32,36,40,44,48 --device cuda --output artifacts/apb_rvq/oracle_full`
- Validate search: `python experiments/rate2_validate_search.py --oracle-dir artifacts/apb_rvq/oracle_full --beam-width 128 --n-beam-chunks 128 --budget 32 --device cuda`
- Corrected summary: `python experiments/rate1_recompute_summary.py --oracle-dir artifacts/apb_rvq/oracle_*`
- Seed stability: `bash scripts/run_apb_rvq_seed_stability.sh` (`seed∈{1,2}`, `n_chunks=512`; seed 0 = full run)

## Results

### Smoke (`n_chunks=4`, CPU)
| Budget | Improvement vs Random |
|---|---:|
| 16 | 0.0% |
| 20 | 27.9% |
| 24 | 34.7% |

- Beam retention smoke (`budget=20`, `beam_width=16`): `0.985` (passes 0.95).

### Medium (`n_chunks=256`, CUDA)
| Budget | vs Random | Notes |
|---|---:|---|
| 20 | 20.2% | |
| 24 | 26.4% | |
| 28 | 29.0% | peak |
| 32 | 28.8% | |
| 36 | 27.3% | |
| 40 | 23.0% | |

- Beam retention medium (`budget=32`, `beam_width=64`, 32 chunks): **0.923** (below 0.95).

### Full seed=0 (`n_chunks=2048`, CUDA)

Primary numbers from `artifacts/apb_rvq/oracle_full/metrics_corrected.json`:

| Budget | vs Random | vs Best non-adaptive | 95% CI on (oracle−random) RMS delta | CI excludes 0 |
|---|---:|---:|---|---|
| 20 | 21.2% | 15.3% | [-0.0125, -0.0119] | yes |
| 24 | **27.7%** | **21.0%** | [-0.0156, -0.0149] | yes |
| 28 | **29.9%** | **22.7%** | [-0.0161, -0.0154] | yes |
| 32 | **29.6%** | **22.8%** | [-0.0153, -0.0147] | yes |
| 36 | 27.6% | 20.7% | [-0.0137, -0.0131] | yes |
| 40 | **23.3%** | **17.0%** | [-0.0111, -0.0105] | yes |

- Peak at **B=28 (29.9% vs random)**.
- Gates A/B on key budgets `24/32/40`: **PASS**.

### Multi-seed stability (episode-selection seeds 0/1/2)

| Seed | n_chunks | B=24 | B=28 | B=32 | B=40 | Gates A/B+CI |
|---|---:|---:|---:|---:|---:|---|
| 0 (`oracle_full`) | 2048 | 27.7% | 29.9% | 29.6% | 23.3% | PASS |
| 1 (`oracle_seed1`) | 512 | 27.7% | 30.3% | 30.4% | 24.4% | PASS |
| 2 (`oracle_seed2`) | 512 | 28.0% | 30.1% | 29.7% | 23.6% | PASS |
| **mean ± std** | — | **27.8 ± 0.14** | **30.1 ± 0.17** | **29.9 ± 0.38** | **23.8 ± 0.44** | — |

- Across seeds, key-budget improvements stay in a tight band (~0.1–0.4 pp std).
- No seed collapses below the 15% / 10% thresholds.

### Rate2 validation (full, seed=0)
From `artifacts/apb_rvq/oracle_full/validate_search.json`:

| Metric | Value | Threshold | Status |
|---|---:|---|---|
| `beam_retained_mean` | **0.935** | >=0.95 | FAIL |
| `beam_retained_median` | **0.959** | >=0.95 | PASS |
| `passes_95pct_exact_subset` | true | >=0.95 | PASS* |

\*Exact-subset mean/median can exceed 1.0 because the reduced-position exact search is not a true global upper bound; treat beam retention as the primary greedy-quality check.

**Plan §9.4 implication:** report **beam** as the conservative search-quality reference; use **greedy** as a fast label generator for rate3+. The matched-budget oracle signal remains strong under greedy and is stable across seeds.

### Per-task stability
- Full seed=0 (`per_task.json`): 40 tasks; at `B=24/32/40` → **40/40 positive**, **40/40 ≥15%**.
- Seed1: `B=24/32/40` → 40/40 ≥15%.
- Seed2: `B=24/32` → 40/40 ≥15%; `B=40` → **39/40 ≥15%** (still all positive).

## Sanity Checks
- `vocab=2048`, `rvq_levels=3`, `positions=16`.
- Full-depth manual decode matches native decode.
- Level-major token layout verified.
- Gripper convention: `invert`.
- Trivial budgets B=16 and B=48 show ~0% vs random (expected).

## Interpretation
### What the evidence supports
- At matched token budgets, **per-position allocation substantially beats random/uniform/static/global-mix** on LIBERO v2.0 action reconstruction RMS.
- Effect is stable across **medium→full scale** and across **3 episode-selection seeds**.
- Corrected episode-clustered bootstrap CIs exclude zero on key budgets for seed 0/1/2.
- Task-level coverage is broad (not a 1–2 task artifact).

### What it does not support
- That greedy is ≥95% of beam on the full validation mean (0.935 < 0.95).
- Any claim about LIBERO **success rate**, BAR router quality, wall-clock sparse speedup, or causal observability of the oracle map (rate3+).

## GO/KILL Decision
- **Decision:** `GO` — **Gate-1 CLOSED**.
- **Reasoning:**
  - Thresholds A/B + CI pass on full seed=0 and on seeds 1/2.
  - Multi-seed mean peak ≈ **30.1% at B=28**, with low seed-to-seed variance.
  - Greedy-vs-beam mean retention remains slightly below 0.95 → downstream labels may use greedy maps, but papers/reports should note beam as the tighter search reference.
- **Next stage:** Phase A rate3 (`coarse_codes` + `depth_map` labels from existing oracle artifacts; no rate1 recompute), then tiny router eval before optional Phase B BAR features.

## Revisions and Environment
- Working commit used for full/seed runs: `cafd70e4ba22416ce5ab62ff91effd633835439e`
- Model: `ZibinDong/ActionCodec-Base-RVQft`
- Dataset: `physical-intelligence/libero@v2.0`
- Cluster: `ccmplanner.mipt.ru`, CUDA V100
- Artifacts:
  - `artifacts/apb_rvq/oracle_full/`
  - `artifacts/apb_rvq/oracle_medium/`
  - `artifacts/apb_rvq/oracle_seed1/`
  - `artifacts/apb_rvq/oracle_seed2/`
  - Prefer `metrics_corrected.json` over raw `metrics.json` for CI / baseline comparisons
- Process notes:
  - Rate2 first attempt died on SSH (no tmux); relaunched in tmux with checkpointing.
  - Seed pipeline briefly stalled after seed1 when recompute required missing `pandas`; fixed to pyarrow-only and seed2 completed.
