# K-31: выбор поддерева для коррекции по исходам развилок

План: K31_Selective_Subtree_Repair_Agent_Plan.md (v1.0, 2026-11-03). Источник —
замороженный parents_cond24 K-25 (seeds 0 и 1); одна коррекция одного
восьмишагового чанка за эпизод; селектор учится по исходам проверенных
продолжений.

| Пакет | Файлы | Что делает |
|---|---|---|
| A1 | `k31_readout.py`, `k31_candidates.py` | Правки корня/среднего/листа с причинным пересчётом потомков (вне поддерева — reference побитово, схват — reference); путь по argmax W2; локальный RNG; плоский AR(1); нормировка эффекта (RMS 0.02, λ ≤ 8, max ≤ 0.08); общий m; 149 признаков; правило выбора (порог 0.10) |
| A2 | `k31_policy.py`, `k31_collect.py` | Рука reference / repair / flatrepair (B = 1, право на ремонт — на эпизод); эпизод с защёлкой done/успеха и остатком горизонта 600 − tau; reservoir-развилка; ветви через свежую среду + побитовый повтор исполненного префикса; обязательный noop; журнал бюджета (резерв 610, авария списывает резерв, повтор не начисляется) |
| A3 | `k31_protocol.py`, `k31_data.py` | prepare, real-check, fork-smoke, пилот A и гейт, добор B, набор (PCA 16+16 только на train, стандартизация), план SR |
| B | `k31_train.py` | Селектор 149→64→32→3 (11 779 параметров), CPU, AdamW, 1000 обновлений, best по val CE, fitcheck, возобновление = непрерывному |
| C | `k31_harness.py` | SR B = 1: блоки по 5 последовательных эпизодов, независимая CPU-проверка журнала и пересчёт выбора |
| D | `k31_analyze.py`, `k31_latency.py`, `k31_run.sh` | Два вывода (project_result, hierarchy_specific_evidence), Holm-4, совместный бутстрэп; задержка B = 1; оркестрация |

Запуск (после окончания K-30; до этого — только самопроверки из отдельной копии):

```bash
cd /home/malinin_aa/Adaptive_VLA
bash experiments/k31_run.sh selftest cpu
bash experiments/k31_run.sh prepare
bash experiments/k31_run.sh real-check cuda:0
bash experiments/k31_run.sh fork-smoke cuda:0
nohup bash experiments/k31_run.sh run cuda:0 > /dev/null 2>&1 &
tail -F logs/k31/run.log
```

Коды: 0 — успех или информационный стоп (stopped_no_headroom,
stopped_low_proposal_coverage, stopped_inactive_selector — причина в логе и
reports/k31/pilot.json / train.json); 3 — технический отказ (повторный run
продолжает с проверенного места); 1 — самопроверки.

Бюджет физических шагов (data/k31/ledger.json): development 400 000 на сид
(пилот A — до 160 000), проверки 20 000, SR 549 000.
