# Adaptive Residual-Consistent Hierarchical Discrete Flow для VLA

Самодостаточный исследовательский и инженерный план для передачи другому агенту.

**Рабочее название метода:** RC-HDF / Adaptive Residual-Consistent Hierarchical
Discrete Flow.

**Статус:** обновлённая формулировка после сопоставления с DFM-VLA,
ActionCodec-BAR, ResGen и hierarchy-aware diffusion над RVQ-аудиокодеками.

---

## 0. Краткое резюме

Мы строим дискретный flow над иерархическими RVQ-кодами действий. Основная
новизна — не просто адаптивное число шагов, а новый тип перехода:

> При изменении грубого RVQ-кода flow одновременно перестраивает зависимый
> residual suffix относительно предсказанной чистой cumulative latent.

Это позволяет повторно исправлять coarse-уровни, не оставляя fine-коды,
закодированные относительно старого residual. Переход является **reversible
coarse-to-fine refinement**: coarse-уровень не фиксируется навсегда, но его
пересмотр не разрушает остаточную семантику.

Вторая часть метода — **адаптивное распределение вычислений**:

1. на каждом проходе выбирать, какой RVQ-уровень или блок уточнять;
2. оценивать ожидаемое физическое улучшение на единицу вычислений;
3. завершать refinement, когда дальнейшие изменения несущественны.

Итого заявка:

> Adaptive residual-consistent discrete flow динамически выбирает, какой
> уровень RVQ исправить, применяет согласованный блочный переход и использует
> разное число backbone-forward passes для разных наблюдений.

Главные экспериментальные вопросы:

1. Возникает ли на самом деле residual inconsistency при независимом
   параллельном refinement?
2. Улучшают ли coupled transitions качество при одинаковом числе проходов?
3. Сокращает ли adaptive scheduler среднее число проходов без потери success
   rate относительно фиксированного и matched-random бюджетов?

Если вопрос 1 получает отрицательный ответ, основной механизм теряет
мотивацию. Если вопрос 3 получает отрицательный ответ, остаётся самостоятельный
вклад residual-consistent flow, но адаптивность нельзя заявлять как результат.

---

## 1. Что уже занято

### 1.1 Геометрия действий в DFM-VLA

[DFM-VLA](https://arxiv.org/abs/2603.26320) уже вводит Metric-Aligned Action
Tokenizer (MAAT) и metric-induced probability path:

```text
p_t(x | x_1) = softmax(-beta_t * d(x, x_1))
beta_t = c * (t / (1-t))^alpha
u_t(x, z | x_1)
    = p_t(x | x_1) * dot(beta_t)
      * [d(z, x_1) - d(x, x_1)]_+
```

MAAT дискретизирует каждый нормализованный скаляр действия на общей сетке из
2001 значений и обучает embedding сохранять порядок расстояний в пространстве
действий. Поэтому заявка «добавим action geometry в discrete flow» сама по себе
уже недостаточно нова.

При этом DFM-VLA использует **target-centered geometry**: новый токен должен
быть ближе к предсказанной цели, но скорость явно не штрафует длину самого
перехода `d(z, x)`. Source-local graph kernel остаётся возможным расширением,
но не является основным вкладом текущего плана.

### 1.2 Иерархическая генерация RVQ тоже существует

Нельзя заявлять, что никто не применял diffusion к RVQ-кодам:

- [ResGen](https://arxiv.org/abs/2412.10208) применяет masked discrete
  diffusion к RVQ, генерирует coarse-to-fine и предсказывает cumulative
  embeddings;
- [SIEDD/HiCoDD](https://arxiv.org/abs/2608.06424) применяет diffusion к
  текущему RVQ-codebook, рассматривая более грубые уровни как чистый
  зафиксированный контекст;
- [HiCoDiT](https://arxiv.org/abs/2604.15923) использует hierarchy-aware codec
  diffusion для video-to-speech;
- [ActionCodec-BAR](https://arxiv.org/abs/2602.15397) генерирует RVQ-уровни
  блоками в coarse-to-fine порядке.

Общая стратегия этих методов — учитывать порядок уровней либо зафиксировать
coarse-уровень перед генерацией fine-уровня.

### 1.3 Adaptive stopping само по себе не ново

Выбор числа solver/denoising steps встречается в generative modeling. Поэтому
простой confidence threshold поверх готовой модели — слабая самостоятельная
заявка. Адаптивность должна быть связана с новым пространством согласованных
иерархических переходов и показывать выигрыш против matched-random бюджета.

---

## 2. Незанятая область и точная новизна

Существуют две крайности.

### 2.1 Joint independent refinement

Все токены обновляются параллельно и независимо:

```text
(z1, z2, z3, z4) -> (z1', z2, z3, z4)
```

Но `z2:z4` кодировались как уточнения residual после старого `z1`. После
замены `z1` комбинация остаётся допустимой алгебраически, однако может быть
редкой или отсутствовать среди комбинаций, выдаваемых RVQ-энкодером.

### 2.2 Irreversible coarse-to-fine generation

Уровни генерируются последовательно:

```text
z1 -> freeze(z1) -> z2 -> freeze(z2) -> z3 -> z4
```

Остаточная структура сохраняется, но ошибка раннего coarse-уровня уже не
исправляется полноценным iterative refinement.

### 2.3 Предлагаемый третий режим

Мы разрешаем пересматривать coarse-код, но меняем вместе с ним residual suffix:

```text
(z1, z2, z3, z4) -> (z1', z2', z3', z4')
```

Новые `z2':z4'` строятся относительно `z1'` и текущего предсказания чистой
cumulative latent. Это **residual-consistent block transition**.

Точная формулировка новизны:

> Existing hierarchy-aware RVQ generators preserve the hierarchy through
> coarse-to-fine commitment. RC-HDF instead permits repeated revision of
> coarse codes by transporting the dependent residual suffix as a coupled
> CTMC state transition.

Вторая новизна при положительном экспериментальном результате:

> An adaptive level-wise sampler allocates coupled transitions according to
> their expected action-space improvement and terminates refinement using a
> variable per-observation compute budget.

---

## 3. Формализация состояния

Пусть action tokenizer кодирует чанк действий `a` в RVQ-коды

```text
z in {1, ..., V}^{P x L},
```

где `P` — число временных latent-позиций, `L` — число RVQ-уровней. Для позиции
`p` cumulative latent равна

```text
h_p(z) = sum_{ell=1}^L e_ell(z[p, ell]),
```

а декодированный action chunk:

```text
a_hat = Dec(h_1(z), ..., h_P(z), state_metadata).
```

Условная flow-модель получает:

- visual-language context `c`;
- текущее зашумлённое состояние `z_t`;
- flow time `t`;

и предсказывает:

1. категориальное распределение чистых кодов
   `p_theta(z_1 | z_t, c, t)`;
2. либо дополнительно clean cumulative latent `h_hat_theta`.

Для MVP удобнее иметь обе головы:

```text
token head:      logits[p, ell, v]
cumulative head: h_hat[p, :]
```

`h_hat` можно также получить как ожидаемую сумму codebook embeddings, но
отдельная регрессионная голова позволяет явно обучать геометрию общей латенты.

---

## 4. Residual-consistent block transition

Рассмотрим замену уровня `ell` в позиции `p` на кандидат `v`.

### 4.1 Построение кандидата

1. Сохранить уровни до `ell`:

```text
z'[p, j] = z[p, j], j < ell.
```

2. Установить новый код:

```text
z'[p, ell] = v.
```

3. Вычислить residual относительно предсказанной чистой латенты:

```text
r = h_hat[p] - sum_{j=1}^ell e_j(z'[p, j]).
```

4. Последовательно переквантовать suffix:

```text
for j = ell+1, ..., L:
    z'[p, j] = argmin_u ||r - e_j(u)||^2
    r = r - e_j(z'[p, j])
```

5. Остальные временные позиции оставить без изменения.

Обозначим полученное состояние

```text
C_{p,ell}(z, v; h_hat).
```

Для `ell=L` это обычная замена одного токена. Для `ell=1` меняется вся
остаточная колонка RVQ.

### 4.2 Почему это не просто постобработка

Слабый MVP может сначала сделать независимый flow-переход, а затем применить
re-canonicalization suffix. Это дешёвая проверка механизма, но не финальная
формулировка статьи.

Полный метод должен определить CTMC-граф, в котором допустимые рёбра уже имеют
вид

```text
z -> C_{p,ell}(z, v; h_hat).
```

То есть единицей перехода является согласованный RVQ-блок, а не отдельный
категориальный токен.

Для генератора CTMC:

```text
Q_t(z, z') >= 0, z' != z
Q_t(z, z) = -sum_{z' != z} Q_t(z, z').
```

Поддержка off-diagonal rates ограничена coupled-кандидатами. Конкретную
flow-matching параметризацию нужно вывести так, чтобы она удовлетворяла
continuity equation; простое умножение готовой скорости DFM-VLA на Gaussian
kernel этого не гарантирует.

### 4.3 Два этапа реализации

**MVP-R (retraction).** Обучить joint DFM, затем после выбранного coarse
transition канонизировать suffix через `h_hat`. Цель — проверить существование
эффекта с минимальным изменением модели.

**Full RC-Flow.** Обучать rate/velocity head непосредственно на графе coupled
transitions. Возможные варианты:

1. conditional graph bridges между noisy state и clean RVQ tuple;
2. auxiliary velocity head, чья поддержка ограничена coupled edges;
3. graph-constrained kinetic-optimal flow с численной проверкой continuity
   equation.

Не переходить к сложному выводу Full RC-Flow, пока MVP-R не показывает
измеримого выигрыша.

---

## 5. Адаптивное распределение бюджета

Адаптивность должна быть **instance-dependent**, а не просто фиксированным
числом шагов для разных уровней.

### 5.1 Что выбирает scheduler

На каждом backbone-проходе `k` scheduler решает:

1. какой уровень или множество уровней обновить;
2. какие временные позиции активировать;
3. продолжить sampling или остановиться.

Пример траекторий:

```text
простое состояние:  L1 -> L2 -> STOP
сложное состояние:  L1 -> L1 -> L2 -> L3 -> L1 -> L4 -> STOP
```

Вторая траектория может вернуться к `L1`, но coupled transition сразу
перестроит зависимый suffix.

### 5.2 Оценка полезности перехода

Для блока `b=(p, ell)` вычислять:

```text
U_b     = uncertainty of clean-code prediction
I_b     = residual inconsistency / suffix recanonicalization magnitude
G_b     = expected decrease of action-space risk after coupled transition
C_b     = measured compute cost
S_b     = G_b / C_b
```

Основной вариант:

```text
R(z) = E_{z_1 ~ p_theta} [d_act(z, z_1)]
G_b  = R(z) - E_v[R(C_b(z, v; h_hat))].
```

`d_act` — decoder-induced расстояние между исполняемыми префиксами действий,
а не только L2 между embeddings. Для дешёвого варианта его аппроксимирует
обученная маленькая gain-head.

Scheduler выбирает

```text
b* = argmax_b S_b.
```

### 5.3 Правило остановки

Остановить refinement, если

```text
max_b G_b < tau_gain
```

или если верхняя граница физического изменения первого исполняемого окна ниже
`tau_action` несколько последовательных проходов.

Порог выбирать только на validation set. Нельзя подбирать его по test success.

### 5.4 Откуда берётся реальная экономия

Изменение меньшего числа токенов само по себе не экономит compute, если каждый
раз вызывается полный backbone. Поэтому обязательная цель — переменное число
полных проходов

```text
K = K(observation),
```

и метрики `average backbone NFE` и wall-clock latency.

Level-wise selection даёт дополнительную экономию только при наличии хотя бы
одного механизма:

- отдельных level blocks/action experts;
- caching стабильных уровней;
- пропуска fine-level heads/blocks;
- active-token computation.

В первой реализации достаточно adaptive stopping полного backbone. Selective
compute — следующий этап после подтверждения сигнала.

### 5.5 Обязательные контроли

Для каждого адаптивного метода сравнивать:

1. fixed `K`;
2. matched-random `K`, имеющий ту же эмпирическую гистограмму бюджета;
3. random level schedule с тем же числом обновлений каждого уровня;
4. oracle schedule по ground-truth action error — только как верхнюю границу.

Адаптивность считается доказанной только если она превосходит matched-random,
а не только fixed maximum budget.

---

## 6. Предварительные измерения

### 6.1 Неравномерное влияние уровней/позиций

Доля влияния позиции на декодированное действие:

| токенизатор | 1-я | 2-я | 3-я | 4-я | остальные |
|---|---:|---:|---:|---:|---:|
| VQ-VLA, RVQ | 53% | 21% | 16% | 10% | — |
| OAT, FSQ | 51% | около 7% | около 7% | около 7% | около 7% |

Это устойчивый паттерн на двух архитектурах, но пока не «инварианта». Его
нужно проверить на ActionCodec и желательно X-Tokenizer.

### 6.2 Геометрия FSQ-кодов OAT

Перебор всех 1000 кодов каждого регистра дал:

| величина | значение |
|---|---:|
| соседи по FSQ-решётке ближе по действию | 3.85x, медиана |
| то же для первого регистра | 25.9x |
| ложные локальные соседи | 0% |
| глобальная ранговая связь | 0.726 |
| ранговая связь внутри окрестности 20 | 0.414 |

На FSQ решётка уже является хорошей глобальной физической метрикой. На
нерегулярном ActionCodec-RVQ это ещё не проверено.

### 6.3 OAT prefix reconstruction ladder

Ошибка реконструкции при использовании первых `k` регистров:

```text
k=1: 1.61%
k=2: 1.23%
k=4: 1.04%
k=8: 0.88% диапазона действий
```

Чистые ступени наблюдались на степенях двойки, что согласуется с
`token_dropout_mode: pow2`. Это поддерживает идею variable depth, но не
доказывает state-dependent adaptive scheduling.

### 6.4 Предыдущий отрицательный опыт адаптивности

Ранее instance-adaptive `K/R` и convergence/PACE-подобные правила не дали
надёжного выигрыша против matched-random. Поэтому текущая работа не должна
зависеть от успеха одной stop-эвристики. Главная страховка — самостоятельный
residual-consistent transition.

---

## 7. Kill-tests до обучения flow

Все проверки выполняются на выпущенных весах ActionCodec-RVQ.

### 7.1 Decoder geometry на ActionCodec-RVQ

Для каждого уровня, temporal position и набора контекстных кодов сравнить:

1. расстояние codebook embeddings;
2. расстояние cumulative RVQ latents;
3. decoder-induced action distance.

Измерить Spearman correlation, precision@k соседей и долю false neighbors.
Особенно важна дисперсия decoder-distance одной пары `(u,v)` по residual
contexts.

**Решение:**

- высокая корреляция и малая context variance — использовать дешёвую
  embedding metric;
- высокая context variance — нужна conditional decoder-aware metric/gain-head.

### 7.2 Цена residual inconsistency — главный тест

Для реального чанка:

1. получить encoder latent `h=Enc(a)` и настоящий RVQ tuple `z`;
2. заменить coarse-код `z[p,ell]` на близкий/предсказанный кандидат `v`;
3. вариант A: оставить старый suffix;
4. вариант B: заново квантизовать residual того же `h` после нового prefix;
5. декодировать A и B.

Измерить:

- latent error относительно `h`;
- decoded action displacement;
- suffix Hamming distance;
- jerk первого исполняемого окна;
- дополнительную ошибку поверх обычного encode-decode-encode cycle error.

Проверить разные уровни, позиции и радиусы замены.

**Go:** stale suffix статистически и практически хуже canonical suffix.

**Kill:** эффект мал относительно reconstruction floor кодека.

### 7.3 Compute-quality curve ActionCodec-BAR

На выпущенном BAR-чекпойнте прогнать 1, 2, 3, 4 RVQ-блока и дополнительные
refinement passes, если они поддерживаются.

Измерить success, reconstruction/action error, NFE и latency.

Если 2 уровня почти равны 4, state-adaptive depth может быть не нужен, но
fixed truncation становится сильным efficiency baseline.

### 7.4 State-dependence влияния

Для каждого observation/action chunk измерить физический эффект изменения
каждого уровня.

```text
Var_state(impact_ell)
```

сравнить с

```text
E_state(impact_ell).
```

- малая вариативность — использовать фиксированное неравномерное расписание;
- большая вариативность и предсказуемость из hidden state — оправдание
  instance-adaptive scheduler;
- большая, но непредсказуемая вариативность — адаптивность может не обучиться.

Порядок выполнения: **7.2 -> 7.4 -> 7.3 -> 7.1**. Residual inconsistency —
самый быстрый тест основной гипотезы; decoder geometry нужна позже для
конкретной параметризации.

---

## 8. Базовые модели и абляции

### 8.1 Основные базовые линии

| модель | роль |
|---|---|
| ActionCodec-PD | one-shot нижняя граница latency |
| ActionCodec-AR | обычная последовательная генерация |
| ActionCodec-BAR | ближайший hierarchy-aware конкурент |
| ActionCodec-KI | внешний continuous flow baseline, не discrete RVQ-flow |
| Joint factorized DFM | независимый refinement всех RVQ-кодов |
| Levelwise/committed DFM | закончить coarse, затем перейти к fine |
| ResGen-style cumulative predictor | контроль влияния cumulative latent head |
| Fixed RC-HDF | coupled transitions, фиксированный budget |
| Adaptive RC-HDF | полный предлагаемый метод |
| Matched-random RC-HDF | обязательный контроль адаптивности |

DFM-VLA/MAAT использовать как внешний flat-token reference, но не как
единственную базовую линию: tokenizer, длина последовательности и backbone
отличаются.

### 8.2 Факторная абляция

Сначала провести факторный эксперимент:

| coupled suffix | adaptive scheduler | назначение |
|---:|---:|---|
| нет | нет | joint DFM baseline |
| да | нет | вклад residual consistency |
| нет | да | вклад scheduler без нового перехода |
| да | да | полный метод |

Дополнительные абляции:

- suffix retraction только на inference против обучения coupled flow;
- token-only head против token+cumulative head;
- embedding distance против decoder/action distance;
- фиксированная важность уровней против instance-dependent gain;
- adaptive stopping без level selection;
- разные максимальные бюджеты `K_max`;
- source-local graph kernel как необязательное расширение.

### 8.3 Фиксированные бюджеты

Минимальная сетка:

```text
K in {1, 2, 4, 8, 16}
```

Для адаптивного метода задавать тот же `K_max`, публиковать полную гистограмму
реализованного `K(observation)` и сравнивать при одинаковом среднем compute.

---

## 9. Метрики и статистика

### 9.1 Основные метрики

- LIBERO success rate по каждому suite и среднее;
- среднее число полных backbone forward passes;
- median/p90 wall-clock latency;
- throughput actions/s;
- task success versus compute Pareto frontier.

### 9.2 Диагностические метрики

- residual inconsistency score;
- physical length каждого CTMC jump;
- negative-progress rate;
- число возвратов к coarse-уровням;
- suffix changes после coarse transition;
- action jerk/smoothness;
- calibration predicted gain versus realized gain;
- stop precision/recall относительно oracle improvement.

### 9.3 Статистический протокол

- одинаковые initial states и task seeds для сравниваемых методов;
- task-paired bootstrap, а не только независимый Bernoulli SE;
- несколько training seeds хотя бы для главного сравнения;
- заранее заданный non-inferiority margin, например 2 п.п. LIBERO SR;
- adaptation thresholds выбирать только на validation tasks/rollouts;
- публиковать matched-random с доверительными интервалами.

Критерий успеха efficiency-заявки:

> Adaptive RC-HDF уменьшает среднее число backbone passes или wall-clock не
> менее чем на 25% относительно лучшего fixed schedule при падении SR не более
> заранее заданного margin и превосходит matched-random budget allocation.

Число 25% — рабочая инженерная цель, а не теоретически выведенная константа.

---

## 10. План реализации

### Phase 0. Проверка гипотез, без обучения

1. Реализовать загрузчик ActionCodec-RVQft.
2. Реализовать `encode -> intervene -> decode` API.
3. Провести тест residual inconsistency §7.2.
4. Провести state-dependence тест §7.4.
5. Построить BAR compute-quality curve §7.3.
6. Измерить RVQ geometry §7.1.

**Выход:** короткий отчёт с GO/KILL по основной и adaptive гипотезам.

### Phase 1. Воспроизводимая база

1. Воспроизвести ActionCodec-PD/BAR на LIBERO.
2. Реализовать joint factorized discrete flow на том же backbone/tokenizer.
3. Проверить fixed-budget curve `K={1,2,4,8,16}`.
4. Зафиксировать latency measurement protocol.

Нельзя оценивать новый метод, пока joint DFM не обучается стабильно и не
достигает разумного качества относительно PD/BAR.

### Phase 2. MVP residual retraction

1. Добавить cumulative latent head.
2. После coarse transition переквантизовать suffix относительно `h_hat`.
3. Сравнить при том же `K`:
   - no retraction;
   - inference-only retraction;
   - oracle retraction относительно ground-truth `h`.
4. Измерить inconsistency, SR, jerk и NFE.

Большой разрыв между oracle и predicted retraction означает, что механизм
верен, но `h_hat` недостаточно точна.

### Phase 3. Full coupled flow

1. Построить граф coupled transitions.
2. Реализовать rate head/conditional bridge на этом графе.
3. Проверить conservation/continuity numerically на малом словаре.
4. Обучить Full RC-HDF.
5. Сравнить с inference-only retraction.

### Phase 4. Adaptive scheduler

1. Начать с дешёвого rule-based gain score.
2. Собрать labels realized gain из rollout/model trajectories.
3. Обучить gain-head или budget policy.
4. Калибровать stopping threshold на validation.
5. Сравнить fixed, matched-random и oracle schedules.
6. Только после положительного результата добавлять selective level compute.

### Phase 5. Полный benchmark

1. LIBERO-10/четыре suites.
2. Несколько seeds главной модели.
3. Pareto curves quality/compute.
4. Поведенческие визуализации coupled trajectories.
5. По возможности перенос на второй RVQ-tokenizer, например X-Tokenizer.

---

## 11. Инженерная структура кода

Рекомендуемые компоненты:

```text
tokenizer_adapter.py
    encode_actions(actions, metadata) -> z, h
    decode_tokens(z, metadata) -> actions
    codebook_embedding(level, ids) -> e

residual_transition.py
    build_candidate(z, h_hat, position, level, new_code)
    recanonicalize_suffix(prefix, h_hat)

joint_dfm.py
    predict_clean_tokens(z_t, context, t)
    sample_independent_transition(...)

rc_flow.py
    enumerate_coupled_edges(...)
    compute_rates(...)
    sample_coupled_transition(...)

adaptive_scheduler.py
    estimate_gain(...)
    choose_active_block(...)
    should_stop(...)

metrics.py
    residual_inconsistency(...)
    decoded_action_distance(...)
    jump_statistics(...)
    compute_statistics(...)
```

Псевдокод inference:

```python
z = sample_initial_rvq_state()

for k in range(K_max):
    token_logits, h_hat, features = model(z, context, time=k / K_max)

    gains = scheduler.estimate_all_blocks(
        z=z,
        token_logits=token_logits,
        h_hat=h_hat,
        features=features,
    )

    if scheduler.should_stop(gains, k):
        break

    position, level = scheduler.choose_block(gains)
    new_code = sample_transition_code(token_logits, position, level)
    z = residual_consistent_transition(
        z=z,
        h_hat=h_hat,
        position=position,
        level=level,
        new_code=new_code,
    )

actions = tokenizer.decode(z)
```

Замечание: использование `time=k/K_max` при ранней остановке — только MVP.
Полный sampler должен корректно достигать terminal distribution; нельзя просто
выдать состояние стандартного time-dependent DFM при `t<1` без terminal
prediction/validation. Практический вариант — на остановке вернуть clean
prediction `argmax p_theta(z_1|z_t,c)` с одной residual canonicalization либо
использовать адаптивные step sizes, суммарно доводящие local/global clock до
`t=1`.

---

## 12. Основные риски и ответы

### Residual inconsistency мала

Тогда coupled transition не оправдан. Остановить направление после Phase 0,
не пытаться спасать его только adaptive stopping.

### Canonical suffix ухудшает multimodality

Жадная переквантизация относительно среднего `h_hat` может усреднять режимы.
Ответ: сэмплировать clean latent/target tuple, использовать несколько
кандидатов или mixture head.

### Coupled edge слишком большой

Замена coarse-кода может менять весь suffix и давать крупный физический jump.
Ответ: ограничить candidates ожидаемым action-space gain и сравнить с
source-local graph constraint.

### Adaptive scheduler не лучше random

Тогда не заявлять instance adaptation. Оставить fixed residual-consistent flow
и честно показать отрицательную абляцию.

### Нет реальной экономии latency

Декодирование кандидатов может быть дороже сэкономленных NFE. Использовать
gain-head и батчевую оценку; главным числом считать wall-clock, не token updates.

### DFM-VLA code unavailable

Не блокировать основной эксперимент: сравнение внутри ActionCodec с joint DFM,
BAR и PD важнее. DFM-VLA оставить внешним reference до официального кода.

### LIBERO насыщен

Фокусироваться на compute-quality Pareto, LIBERO-Long/Goal, perturbations и
trajectory metrics. Не строить статью только на улучшении среднего SR с 97–98%.

---

## 13. Критерии GO/NO-GO

### GO основной идеи

Продолжать Full RC-Flow, если одновременно:

1. stale suffix заметно хуже canonical suffix в §7.2;
2. MVP retraction снижает inconsistency и/или улучшает fixed-NFE качество;
3. накладные расходы не уничтожают latency.

### GO адаптивности

Продолжать adaptive scheduler, если:

1. влияние уровней существенно меняется между состояниями;
2. это изменение предсказывается из model features;
3. oracle schedule имеет заметный разрыв с лучшим fixed schedule;
4. learned/rule scheduler превосходит matched-random.

### Итоговый сильный результат

Лучший сценарий статьи:

1. показан новый failure mode independent RVQ refinement;
2. residual-consistent flow улучшает качество при fixed compute;
3. adaptive scheduler сохраняет качество с меньшим средним NFE/latency;
4. механизм переносится хотя бы на два иерархических action tokenizer.

---

## 14. Источники и код

- DFM-VLA: <https://arxiv.org/abs/2603.26320>, проект:
  <https://chris1220313648.github.io/DFM-VLA/>
- ActionCodec: <https://arxiv.org/abs/2602.15397>, код:
  <https://github.com/ZibinDong/actioncodec>
- ActionCodec-RVQft:
  <https://huggingface.co/ZibinDong/ActionCodec-Base-RVQft>
- ActionCodec-BAR LIBERO:
  <https://huggingface.co/ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO>
- ResGen: <https://arxiv.org/abs/2412.10208>
- SIEDD/HiCoDD: <https://arxiv.org/abs/2608.06424>
- HiCoDiT: <https://arxiv.org/abs/2604.15923>
- Discrete Diffusion VLA: <https://arxiv.org/abs/2508.20072>, код:
  <https://github.com/Liang-ZX/DiscreteDiffusionVLA>
- X-Tokenizer: <https://github.com/X-Square-Robot/X-Tokenizer>
- OAT: <https://github.com/Chaoqi-LIU/oat>
- GeCO: <https://arxiv.org/abs/2603.17834>

---

## 15. Одно предложение для презентации

> Мы вводим адаптивный дискретный flow над RVQ-кодами действий, который может
> повторно исправлять грубые коды, согласованно переносит зависимые residual
> уровни и тратит разное число refinement-проходов в зависимости от ожидаемого
> физического улучшения.
