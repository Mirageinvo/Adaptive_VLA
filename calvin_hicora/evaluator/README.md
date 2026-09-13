# CALVIN evaluator / HULC reproduction

This path is required before **closed-loop** evaluation. It does **not** block
codec scientific training or VLA offline training.

## Goal

Reproduce the official CALVIN long-horizon evaluator (HULC-compatible chain
success, SR1…SR5, AvgLen) with:

- the standard 1000 instruction chains;
- fixed initial states and pairing keys;
- the same success oracle used by the CALVIN benchmark;
- fail-closed provenance (env commit, hashes, seeds).

## Status

| Item | Status |
|---|---|
| Codec / VLA offline training | independent; may proceed |
| HULC / `calvin_env` install lock | TODO |
| Official chain set binding | TODO |
| Success oracle parity check | TODO |
| Rollout harness for BAR / HiCoRA arms | TODO |

## Next concrete steps

1. Lock `calvin` / `calvin_env` / MuJoCo versions in `calvin_hicora/env/`.
2. Run the official HULC or CALVIN baseline evaluation once and archive raw
   chain outcomes plus SR1…SR5.
3. Only then wire HiCoRA / BAR policies into the same harness.

Do not invent numeric success thresholds here; register them after the
reproduction and power analysis required by the SPEC.
