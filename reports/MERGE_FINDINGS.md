# Adaptive Action-Token Merging (AATM) — ICRA Research Log

Branch: `feature/aatm-vla`  
Paper target: **ICRA** (robotics; emphasis on measurable inference gains + LIBERO evaluation)

---

## Working title

**Adaptive Action-Token Merging: State-Dependent Temporal Compression of VLA Action Representations**

---

## Hypothesis

Neighboring temporal action tokens in a fixed 16-position ActionCodec chunk are often redundant. An **observation-dependent merge** can reduce the effective number of temporal tokens with small reconstruction / control degradation, and—if integrated **before** VLA generation—reduce inference cost.

## Positioning vs APB-RVQ

| | APB-RVQ | AATM (this work) |
|---|---------|------------------|
| Axis | RVQ **depth** per position | **Temporal** merge across positions |
| Operation | Add/remove fine codes | Merge contiguous spans |
| Oracle object | depth map | segment partition |
| APB result | Oracle strong; router weak | TBD |

These are **orthogonal**. Combined allocation (merge then depth) is future work, not Stage 1.

---

## Claims policy (ICRA)

We will **not** claim wall-clock speedup unless:

1. Stage 0 identifies a viable integration path with measured fractions.
2. Stage 4 reports **batch=1** end-to-end latency on target GPU.
3. Merger overhead is reported separately from VLA and codec decode.

We **may** claim reconstruction / control quality vs token budget after Stage 1 oracle GO.

---

## Preregistered gates

### Stage 0 — Architectural integration (merge0_integration)

**Procedure:** `python experiments/merge0_integration.py --device cuda`

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Sparse BAR pass (Path B) viable for e2e speedup | action-position share ≥ 10% | **NO-GO** (k4a3: 5.2%) |
| Merge-after decode (Path A) saves VLA compute | must change VLA token count | **NO-GO** (by design) |
| Oracle redundancy experiments | always run | **GO** |
| Latency paper claims | require Path C and/or E + e2e bench | **CONDITIONAL** |

**Implication:** Stage 1 oracle measures **compression headroom**; latency story requires retrain (Path C) or action-only refiner (Path E).

### Stage 1 — Oracle compression curve (merge1_oracle)

**Procedure:** `bash scripts/run_aatm_oracle.sh medium|full`

| Criterion | Threshold |
|-----------|-----------|
| Headroom @ k∈{8,10,12} | median RMS increase ≤ 5% vs no-merge |
| Adaptive vs fixed @ k=8 | oracle RMS < fixed_pair RMS; bootstrap CI excludes 0 |
| Chunks with ≥25% token reduction | ≥50% chunks at ≤5% RMS increase |
| Kill | compression <20% at ≤5% error **or** adaptive ≈ fixed |

### Stage 2 — Temporal locality + causal predictability (merge2, merge3)

| Criterion | Threshold |
|-----------|-----------|
| Pair merge correlates with neighbor similarity | report ρ; not required for GO |
| Causal heuristics vs oracle | must beat random; target ≥50% oracle gap closed |
| Kill (APB lesson) | heuristics ≈ random → **NO learned merger** |

### Stage 3 — Learned merger (merge4)

Only if Stage 1 **and** Stage 2 GO.

| Criterion | Threshold |
|-----------|-----------|
| Learned vs best fixed @ matched budget | statistically significant on held-out episodes |
| Oracle gap closed | ≥50% of (oracle − best_fixed) |

### Stage 4 — Quality vs compute (merge5, merge6)

| Criterion | Threshold |
|-----------|-----------|
| LIBERO success | ≤2 pp drop vs baseline at matched latency **or** |
| Latency | ≥15% e2e reduction at matched success |
| Hardware | V100 batch=1 minimum; A100/FP16 preferred for final table |

---

## Experiment registry

| Script | Stage | Purpose |
|--------|-------|---------|
| `merge0_smoke.py` | 0 | Pipeline sanity |
| `merge0_integration.py` | 0 | **Architectural gate + latency fractions** |
| `merge1_oracle_compression.py` | 1 | Oracle curve, fixed vs adaptive |
| `merge1_posthoc_plan_gaps.py` | 1 | **§3 scheme C + §1 span 2–4** from saved `eval_rows` |
| `merge2_locality.py` | 2 | Pair heatmaps, span stats |
| `merge3_labels.py` | 2 | Episode-disjoint oracle labels |
| `merge4_train.py` | 3 | Causal MLP merger |
| `merge5_latency.py` | 4 | V100 batch=1 breakdown |
| `merge6_rollout.py` | 4 | LIBERO success rate |

Launcher: `bash scripts/run_aatm_oracle.sh {smoke|medium|full}`

---

## Planned paper figures

1. **Oracle curve:** #segments → RMS (16→4), fixed / random / adaptive.
2. **Adaptive gain:** oracle − best fixed vs segment budget.
3. **Temporal locality heatmap:** pair (i,i+1) merge frequency & error.
4. **Main result:** success (or RMS) vs latency / #tokens (batch=1).
5. **Latency breakdown:** VLM prefix | BAR blocks | merger | codec decode.

---

## Integration paths (Stage 0)

| Path | Speedup viable? | Paper role |
|------|-----------------|------------|
| A: merge-after VLA | No | Oracle / ablation only |
| B: sparse BAR pass | No (integrated) | Negative result / motivation |
| C: retrain K<P tokens | Conditional | Primary latency path |
| D: adaptive NFE | Orthogonal | Related work / future |
| E: action-only refiner | Open | Secondary latency path |

---

## Artifacts

- `artifacts/merge/stage0_integration.json`
- `artifacts/merge/merge0_smoke_*.json`
- `artifacts/merge/oracle_*/summary.json`
- `artifacts/merge/oracle_*/eval_rows.parquet`
- `artifacts/merge/oracle_*/plan_gaps.json`

After the live medium job writes rows:

```bash
# Cheap: span, locality heatmap, quantiles, heavy chunks (no GPU)
python experiments/merge1_posthoc_plan_gaps.py \
  --oracle-dir artifacts/merge/oracle_medium --rows-only

# Scheme C, span≤4 proxy, similarity, uniform at all k
python experiments/merge1_posthoc_plan_gaps.py \
  --oracle-dir artifacts/merge/oracle_medium --device cuda

# Optional greedy vs oracle (plan §8), extra decodes
python experiments/merge1_posthoc_plan_gaps.py \
  --oracle-dir artifacts/merge/oracle_medium --device cuda --compute-greedy
```

Как читать цифры: **`reports/AATM_POSTHOC_INTERPRETATION.md`** (канон).

**Science decision (after medium + post-hoc cuda):** use `by_budget["8"].scheme_c` keys `*_val` only (`adaptive_gain_vs_scheme_c_val`, `adaptive_gain_vs_best_fixed_val`, `adaptive_gain_bootstrap_ci_val`, `bootstrap_p_oracle_better_than_*_val`). Fields `*_all` / `train_rms` are transparency, not the gate.

**Proxies / what is not the science decision:**
- Scheme C is a **proxy**: train-oracle winners (+ pair at k=8), not a global argmin over all partitions. Gain vs C may be optimistic.
- Span≤4 oracle is a **proxy**: exact only when the unrestricted winner is already legal; otherwise min over observed legal schemes, not a full re-enumeration.
- `stage1_gate` in merge1 is **pair-only** and is **not** the science decision. Do not treat merge1 GO as §3.

Latency is not decided post-hoc (Stage 0 CONDITIONAL).

**Binding lock (мультиагентка 2026-08-22):** не стопать live medium; лишних ранов нет. Что делать при GO / OPEN / NO-GO — `reports/AATM_POSTHOC_INTERPRETATION.md` § Binding lock. Коротко: NO-GO на optimistic bound финальный; GO/OPEN → один next = истинная C на том же split, не рестарт и не full.

---

## Status

| Stage | Status |
|-------|--------|
| 0 integration gate | CONDITIONAL (BAR 5.2%, merge-after no speedup) |
| 1 oracle curve | medium done; post-hoc cuda done |
| 2–4 | not started; no merger (AUROC 0.61, latency CONDITIONAL) |

Human write-up for the advisor: **`reports/AATM_MEDIUM_RESULTS.md`**.

---

## GO/KILL decision

**Decision:** `OPEN` — headroom and adaptive vs best_fixed pass on val; AUROC 0.61 does not pass. Do not train a merger.

**Next:** Do not launch full / stratified / max_span restart. True global C only if we need a non-proxy adaptive-gain number; it does not fix AUROC.
