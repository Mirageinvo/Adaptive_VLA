# K-36: подтверждение и развитие SR через RL-коррекцию дерева

План агента v1.0 (2026-11-17). Источники — завершённый K-32 (cds2):
`parents_cond24` K-25 s0/s1, final actor T/P, `state.pt` T, нормировка.
Банк — K-33 v2 `ddf3f626f0b5` (space 1: A/C/H по 100 случаев на задачу).

## Первая часть реализации (до решения A)

| Файл | Что |
|---|---|
| `k36_bank.py` | разбиение банка A/C/H (PCG64 [36, 20261117, t]), B-screen, C-списки; `BankSim` — среда B1 (bank / legacy) с API k18 `Sim` |
| `k36_inputs.py` | таблица размещения (SHA1 + SHA256), аудит K-32/K-25/банка, неизменяемый снимок `state.pt` для продолжения |
| `k36_protocol.py` | prepare (идемпотентен), очередь A (360 блоков), бюджеты (9 scopes, 3 687 900) |
| `k36_harness.py` | эпизодные транзакции R/T32/P32, real-check, legacy-check, bank-smoke, cold-repeat, confirm-a, status |
| `k36_analyze.py` | анализ A (эстиманд §7.1, 97.5%/95% бутстрэп, семья A, ресурсное правило), CPU-аудит точности |

Эпизод — неизменённый `k31_collect.run_episode` (seed 101, 10 waiting, до 600
команд) с `k32_policy.Arm` (детерминированно); проверка журнала —
`k32_harness.journal_problems`. Код K-32/K-33 не меняется.

## Порядок (cds2, где лежат артефакты K-32 и копия банка)

```
bash experiments/k36_run.sh selftest
bash experiments/k36_run.sh prepare
bash experiments/k36_run.sh prepare        # должно сказать «закреплён»
bash experiments/k36_run.sh cpu-audit
bash experiments/k36_run.sh real-check cuda:0
bash experiments/k36_run.sh collector-smoke cuda:0
nohup bash experiments/k36_run.sh confirm-a cuda:0 > /dev/null 2>&1 &
bash experiments/k36_run.sh analyze-a
```

Файлы этапа A закреплены prepare (включая `k36_run.sh`) и не меняются.
Вторая часть (continue-b, screen-b, confirm-c с BAR, latency, итоговый
analyze) — новые модули и отдельный раннер; пишется, пока идёт этап A.

## Вторая часть: этапы B и C (новые файлы, отдельный раннер)

| Файл | Что |
|---|---|
| `k36b_gate.py` | точное решение A → B в целых (Σ(выигрыши − потери) ≥ 18 совместно, ≥ −6 на сид; 0 passed / 4 stopped / 3 сбой), принадлежность открытого резерва |
| `k36b_train.py` | продолжение T из снимка `state.pt` K-32 (k32_ppo.Trainer без изменений математики; scopes `train_tplus_s<S>`/`val_tplus_s<S>`, кап +200 000, mid по добавленным шагам) |
| `k36b_screen.py` | B-screen (180 эпизодов Tplus на закреплённых 30 A-сценах 0/6/8), гейт B с пересчётом по исходам |
| `k36c_bar.py` | original BAR в цикле B1 (generate + сборка кодов, контроль официального decode), предварительные проверки legacy 4 / smoke 6 / repeat 3 |
| `k36c_confirm.py` | план C (SHA Tplus и решения B до первого эпизода), очередь 616 блоков, исполнение кусками по источнику |
| `k36c_latency.py` | задержка B1: R/T32/Tplus (s0) и BAR |
| `k36c_analyze.py` | семья C (Tplus−T32, Tplus−R на 0/6/8, Tplus−BAR на all10), описательные таблицы |
| `k36bc_run.sh` | раннер второй части (общая блокировка с `k36_run.sh`) |

```
bash experiments/k36bc_run.sh selftest
bash experiments/k36bc_run.sh gate-a        # после analyze-a: точное решение A
nohup bash experiments/k36bc_run.sh run cuda:N > /dev/null 2>&1 &   # после A
```
