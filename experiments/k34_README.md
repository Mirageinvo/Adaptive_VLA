# K-34: исполнимые выходы дерева после слоёв 8, 16 и 24

План агента v1.0 (2026-11-10). Один checkpoint — три режима исполнения.

| Файл | Что |
|---|---|
| `k34_model.py` | выходы 8/16/24 поверх K-26: `forward(exit_depth=d)` с настоящим усечением, `forward_all` (один проход, три головы), путь источника K-26 побитово; головы `readout8/16` — глубокие копии `readout24` после загрузки источника |
| `k34_train.py` | warmup 500 (только новые головы, lr 3e−4) + main 4000 (expert/LoRA/new/heads, разогрев 200, косинус до 0.1), очередь `u_abs = 200000 + u`, единый best по (L8+L16+L24)/3, продолжение; `fitcheck`, `real-check` |
| `k34_policy.py` | рука K-9h: выбранная глубина обученного checkpoint (или init в smoke) либо R = источник K-26; журнал глубины и счётчиков, отладочный снимок головы |
| `k34_harness.py` | адаптер K-9h (план, ячейка, один блок) |
| `k34_protocol.py` | prepare (источники K-26), планы smoke/pilot/eval, транзакции блоков (`data/k34/ledger.json`: checks 20 000, pilot 122 000, eval 1 464 000), проверки блоков, smoke-check (пересчёт головы), гейт пилота |
| `k34_latency.py` | p50/p95 выходов 8/16/24 (B = 1 и 5), только свободная карта |
| `k34_analyze.py` | McNemar + Holm-4 (T8/T16 против T24), Holm-6 (T против F), joint bootstrap (100 000, seed 34011) |

Порядок (одна машина, отдельная от рабочих копий K-32/K-33):
```
bash experiments/k34_run.sh selftest
bash experiments/k34_run.sh run cuda:0     # prepare -> real-check -> fitcheck -> smoke -> pilot -> (passed) eval
```
Нужны на машине: `reports/k26` (планы) и `data/k26` (best.pt источников),
подготовка K-21 (`reports/k21`, `data/k21`, кадры), допуск не нужен (BC в
K-34 нет), `data/k15f/basis_s0.pt`, снимок BAR.
