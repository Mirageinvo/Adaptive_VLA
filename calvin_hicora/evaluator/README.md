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
| Generic 30 Hz receding-horizon core | Implemented + contract-tested |
| HULC adapter for BAR / HiCoRA arms | TODO |

`receding_horizon.py` enforces the offline/online contract:

- every policy call returns exactly `(30, 7)` native-scale actions;
- exactly the first 12 actions are executed, the remaining 18 are discarded;
- the next plan observes the state after action 12;
- the supplied simulator timestep must equal `1/30` seconds;
- there is no overlap averaging;
- termination/truncation stops execution immediately.

The timestep assertion checks the simulator's configured control timestep, not
wall-clock latency. A simulator may run faster or slower than real time while
still representing 30 Hz physics correctly.

## Next concrete steps

1. Lock `calvin` / `calvin_env` / MuJoCo versions in `calvin_hicora/env/`.
2. Run the official HULC or CALVIN baseline evaluation once and archive raw
   chain outcomes plus SR1…SR5.
3. Bind the policy and environment APIs to `receding_horizon.py`; do not
   duplicate horizon logic in each policy arm.

Do not invent numeric success thresholds here; register them after the
reproduction and power analysis required by the SPEC.
