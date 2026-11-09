# K-33: подтверждающая оценка на свежем банке LIBERO-10

План агента v1.0 (2026-11-06) с поправками: §14 (банк v2, scope `bank_v2`)
и решение 2026-11-09 (полный объём, машины A = ccmplanner и B = cds2).

## Первая очередь — банк сцен (закрыта)

- `k33_bank.py` — рецепт сцены (native reset, снимок S0, `set_init_state(S0)`,
  exec seed 101, 10 шагов ожидания), подписи, `K33Env` для K-9h.
- `k33_protocol.py` — построение банка, bank-check, журнал, миграция,
  архив v1, сверка v1/v2.
- `k33_diag.py` — диагностика расхождения v1 и `--post-fix`.
- Банк v2 `ddf3f626f0b5`: 3000 confirmation + 50 dev, bank-check 3050/3050
  в двух порядках, dev-replay fullbar 10/10.

## Вторая очередь — шесть рук

| ID | Политика | Путь |
|---|---|---|
| T0, T1 | parents_attn24 s0/s1 | `k26_policy.build_arm`, спецификация из плана K-26 |
| F0, F1 | flat_attn24 s0/s1 | то же |
| B | fullbar | штатный generate K-9h |
| BC | fullbar_cached | `k21_cached_policy.build_arm`, допуск k21_barcache |

- `k33_policy.py` — реестр рук, источники (контентные отпечатки), подмена руки,
  проверки метаданных и журнала руки.
- `k33_harness.py` — адаптер K-9h: среда банка/legacy + рука из плана; ячейка
  процесса сверяется с планом машины.
- `k33_eval.py` — план машины, разбиение B5-групп (g чётное — A, нечётное — B;
  по 30 групп каждой задачи), очередь (перестановка задач 33010, сдвиг рук),
  транзакции блоков (reserve -> K-9h -> JSON -> finish: проверка, commit/settle,
  карантин), сверки legacy/dev/repeat/port, гейты, прогноз, прогресс.
- `k33_analyze.py` — McNemar + Holm-2 (T−F по сидам), совместный бутстрэп по
  6-битным образцам (100 000, seed 33011), NI-семейство к B/BC (хвост α/8,
  маржа 3 п.п.).
- `k33_latency.py` — T0, F0, B, BC; батчи 1 и 5; только на свободной карте.

Один процесс K-9h = один блок (холодный старт): 1800 процессов SR на машину.

## Порядок

Машина A (ccmplanner):
```
bash experiments/k33_run.sh eval A cuda:0
```
selftest -> plan A -> legacy-smoke 6 рук (t0/t8 i30, старый путь против
адаптера, побитово) -> dev 50 сцен × 6 рук -> холодные повторы dev t0/6/8 ->
прогноз -> гейты -> SR (продолжается с места при повторе).

Машина B (cds2), после завершения K-32 и переноса банка/источников:
```
bash experiments/k33_run.sh eval B cuda:0
```
plan B (контракт обязан совпасть с A) -> dev t0/6/8 × 6 рук (перенос) ->
прогноз -> гейты -> SR.

Анализ (на A, после копирования `reports/k33/plan_B.json` и `reports/k33/sr/B`):
```
bash experiments/k33_run.sh analyze
bash experiments/k33_run.sh latency cuda:0
bash experiments/k33_run.sh analyze
```
