# K-32: обучаемая коррекция поддерева с ограниченным RL

План: K32_Routed_Subtree_Residual_RL_Agent_Plan.md (v1.0, 2026-11-05). Источник —
замороженный parents_cond24 K-25 (seeds 0 и 1). Учатся только новые actor и
critic; actor на каждом вызове выбирает узел r ∈ {root*, mid*, leaf*} и
латентную поправку η; замороженный оператор переводит (r, η) в команды.

| Рука | Оператор |
|---|---|
| R `parents_ref32` | свежий FP32 reference |
| T `parents_subtree_rl32` | 6D, правка узла с пересчётом потомков (k31_readout.edited) |
| P `parents_profile_rl32` | 6D, тот же профиль W узла, без пересчёта |
| F `parents_direct_rl32` | 48D, AR(1) по 8×6 |

Общий ограничитель: clip ±0.08, RMS-cap 0.04, рука = mid + half·tanh(Z0/half + Du);
схват — всегда reference; η ≡ 0 — рука reference побитово.

| Пакет | Файлы |
|---|---|
| A | `k32_dist.py` (плотность, joint KL), `k32_operator.py` (T/P/F, ограничитель), `k32_model.py` (признаки 4H+67 / 6H+72, нормировка, actor/critic) |
| B | `k32_policy.py` (рука B = 1), `k32_collect.py` (волны, переходы) |
| C | `k32_ppo.py` (PPO с откатом шага по KL_old ≤ 0.03, тренер, возобновление) |
| D | `k32_protocol.py`, `k32_harness.py`, `k32_analyze.py`, `k32_latency.py`, `k32_run.sh` |

Запуск — после K-31; запланированная большая проверка LIBERO имеет приоритет:

```bash
cd /home/malinin_aa/Adaptive_VLA
bash experiments/k32_run.sh selftest cpu
bash experiments/k32_run.sh prepare
bash experiments/k32_run.sh normalizer cuda:0
bash experiments/k32_run.sh real-check cuda:0
bash experiments/k32_run.sh collector-smoke cuda:0
nohup bash experiments/k32_run.sh run cuda:0 > /dev/null 2>&1 &
```

Бюджет (data/k32/ledger.json): обучение 6 × 200 000, валидация 6 × 27 450, SR
732 000, проверки 50 000 — всего 2 146 700 физических шагов.
