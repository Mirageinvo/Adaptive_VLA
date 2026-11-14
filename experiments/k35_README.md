# K-35 (K35-A): адаптивное ветвление временного дерева при фиксированном бюджете

План агента v1.0 (2026-11-14). Один источник (K-34 T0, best main 3000), один
сид, общий декодер для A/C/N, плотный оракул, два селектора, ограниченный
SR-пилот.

| Файл | Что |
|---|---|
| `k35_topology.py` | 36 срезов c = 6a + b: 2 корня + 2 средних + 4 листа, канонические ID, родители (локальные и канонические), слоты K-22, Q = 192, router-Q = 48 |
| `k35_model.py` | класс `K35(K34Parents)`: компактный проход 4 → 8 → 12, mapped-router с явными родителями, только выбранные потомки, смешанная сборка partial y трёх голов (FP32, tanh один раз); `source_forward` = K-34 forward 24 |
| `k35_train.py` | warmup 500 (три головы, 3e−4) + main 4000 (expert 2e−5, LoRA 1e−5, new 1e−4, heads 1e−4), `u_abs = 300000 + u`, срезы по группам из 36 логических строк; best по 6 срезам на checkpoint-val; `real-check`, `fitcheck`, `check-fresh`, `train-valid` |
| `k35_oracle.py` | панели (oracle-train 64 / oracle-val 16 / checkpoint-val 8 на задачу), 36 настоящих forward на строку, c_const, гейт потенциала |
| `k35_selector.py` | PCA-32, OOF по эпизодам (5 folds), селекторы корней (136 → 6) и средних (282 → 6), гейт обучаемости; сохранение набора с отпечатком |
| `k35_policy.py` | руки A/C/N (один K-35, контроллер; сети выбора всегда вычисляются) и S (K-34 T0); журнал вызовов, снимок первого вызова, пересчёт голов и маршрута |
| `k35_harness.py` | адаптер K-9h (план, ячейка, один блок, контекст N) |
| `k35_protocol.py` | prepare + панели, oracle-шарды, гейты оракула и селекторов, планы smoke/pilot, транзакции (`data/k35/ledger.json`: checks 20 000, pilot 122 000), smoke-check, решение пилота |
| `k35_analyze.py` | McNemar A−C / A−N + Holm-2, S описательно, bootstrap 100 000 (seed 35011) |
| `k35_latency.py` | A/C/N/C_fast в одном экземпляре, S отдельно; B = 1 и 5; только свободная карта |

Порядок (отдельная копия на ccmplanner):
```
bash experiments/k35_run.sh selftest
bash experiments/k35_run.sh run-a cuda:0   # prepare -> real-check -> fitcheck -> smoke -> train -> oracle -> гейты -> пилот -> latency -> analyze
```
`run-a` останавливается на технических отказах (код 3) и на непройденных
офлайн-гейтах (`stopped_low_oracle`, `stopped_selector`: информационный стоп,
код 0). Пилот сам проверяет свежесть гейтов (`offline-verify`). Задержка при
занятой карте пишется как `latency_pending` и не блокирует SR.

Нужны в копии: источник K-34 T0 (`data/k34/runs/s0/parents_attn24/best.pt`,
`reports/k34/train_s0_parents_attn24.json`, `reports/k34/prepare.json`),
K-26-источник и его план (`data/k26`, `reports/k26`), подготовка K-21
(`reports/k21`, `data/k21`, кадры), `data/k15f/basis_s0.pt`, снимок BAR.
Перенесённые файлы сверяются по SHA256; prepare записывает их SHA256 в
таблицу размещения и сверяет её при каждом плане.
