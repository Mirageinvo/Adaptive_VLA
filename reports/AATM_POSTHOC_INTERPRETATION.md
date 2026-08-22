# Руководство: как читать AATM после medium

Канонический текст для ветки `feature/aatm-vla`.  
План: `adaptive_action_token_merging_plan.txt`.  
Код: `merge1_oracle_compression.py` + `merge1_posthoc_plan_gaps.py`.

Это не пересказ плана и не абстрактный JSON. Ниже — **реальные ключи** и пороги, по которым писать `MERGE_FINDINGS.md`.

---

## Зачем два источника, и какой главный

`summary.json` (merge1) отвечает: есть ли компрессия и лучше ли oracle **pair-merge**.  
`plan_gaps.json` (post-hoc) отвечает: лучше ли oracle **лучшей фиксированной схемы, включая C**, и видно ли merge из causal features.

Главный тест плана §3 — **не** pair-only.  
`best_fixed = min(fixed_pair, scheme_C)`.  
`stage1_gate` в merge1 схему C не знает. Если C ≈ oracle, merge1-GO завышен.

Science decision после medium: **только** ключи `*_val` в `plan_gaps.json` (held-out эпизоды).  
`*_all` / `train_rms` — прозрачность, не гейт. Пустой val → decision fields = `null` + warning, без тихого fallback на all.

Два разных решения (не смешивать):

| Решение | Вопрос | Уже известно |
|---------|--------|----------------|
| **Science GO** | Учить merger? | Нет, ждём medium + post-hoc |
| **Latency / ICRA wall-clock** | Писать speedup в статью? | Stage 0: **CONDITIONAL**. Integrated BAR, batch=1: позиции ≈ 5.2%, `t(16)/t(8)≈1.03×`. Merge-after-decode не ускоряет VLA. |

Формула `(k/16)²` всегда 0.25 при k=8. Это не эмпирический PASS. Её нельзя ставить в один ряд с gain и AUROC.

---

## Файлы и запуск

| Файл | Когда |
|------|--------|
| `artifacts/merge/oracle_medium/summary.json` | конец merge1 (2 GPU + merge shards) |
| `artifacts/merge/oracle_medium/eval_rows.parquet` | то же |
| `artifacts/merge/stage0_integration_medium.json` | уже есть |
| `artifacts/merge/oracle_medium/plan_gaps.json` | post-hoc |
| `artifacts/merge/oracle_medium/plan_gaps.md` | post-hoc, таблица % vs no-merge |

```bash
# после появления rows / shards
python experiments/merge1_posthoc_plan_gaps.py \
  --oracle-dir artifacts/merge/oracle_medium --rows-only

python experiments/merge1_posthoc_plan_gaps.py \
  --oracle-dir artifacts/merge/oracle_medium --device cuda
```

`--rows-only`: span, heatmap, квантили, heavy chunks.  
`--device cuda`: схема C, AUROC, similarity, uniform на все k.  
`--compute-greedy`: §8, не обязателен для GO.

RMS: меньше лучше.  
`adaptive_gain_*` = `метод − oracle` → **плюс = oracle лучше**.

`rel_error_increase_vs_no_merge` в `summary.json` — **доля** (0.07 = 7%).  
Колонки `plan_gaps.md` — уже **проценты**.

---

## Карта `plan_gaps.json`

```
by_budget["8"].span_stats          §1 длины / legal span
by_budget["8"].locality            §4 heatmap, длина с позиции i
by_budget["8"].scheme_c            §3C; science = *_val; *_all = transparency  [нужен cuda]
by_budget["8"].similarity_vs_merge §4/§5 по этому k
heuristic_aurocs                   §5, обычно k=8         [нужен cuda]
similarity                         сводка bins + Spearman
compute_estimate["8"]              (8/16)² и 8/16
comparison_table                   то же, что plan_gaps.md
limitations                        что не считали
```

Схема C — одна партиция, выбранная на **train-эпизодах** среди legal oracle-победителей (+ pair @ k=8). Это не 16 369 схем.

Bootstrap: чанки без `episode_id` **исключаются**; `episode_id=-1` больше не создаётся как фейковый кластер.

Greedy (`--compute-greedy`): если нет legal adjacent merge (`max_span`), поиск останавливается; метрики всегда пересчитываются на **финальной** сегментации (`n_segments = len(final segs)`).

---

## 1. Oracle headroom и span 2–4

Смотреть **k=8**, рядом k=10 и k=12.

| Ключ | Смысл |
|------|--------|
| `summary.oracle_curve["8"].rel_error_increase_vs_no_merge` | насколько хуже no-merge (доля) |
| `summary.pct_chunks_within_5pct_rms_at_budget["8"].fraction` | доля чанков с −50% токенов при ≤5% RMS |
| `by_budget["8"].span_stats.pct_chunks_plan_legal_span` | все сегменты длины 1–4 |
| `mean_segment_length`, `span_histogram` | типичный span |
| `scheme_c.oracle_span_le4_proxy_all_rms` | proxy рестриктивного oracle |
| `scheme_c.adaptive_gain_unrestricted_vs_span_proxy` | насколько full лучше legal |

| Цифра | Вывод |
|-------|--------|
| rel-increase < 0.05 | Компрессия почти бесплатна |
| 0.05–0.10 | Приемлемый headroom |
| 0.10–0.15 | Дорого; смотри k=10/12 |
| > 0.15–0.20 | План: ветку закрывать |
| legal span > 90% и proxy ≈ full | Лимит 2–4 не важен |
| legal < 70% или большой gap full vs proxy | Full oracle завышен; гейты по proxy |

При k=8 среднее длины ≈ 2 ожидаемо (16/8). Существенно больше 3 → oracle клеит длинные плато.

---

## 2. Адаптивность (главный тест плана §3)

Считать самим (в JSON абсолютный RMS-gain) **на val**:

```text
relative_gain = 100 * adaptive_gain_vs_best_fixed_val / best_fixed_val_rms
```

`best_fixed_val_rms` = `min(fixed_pair, scheme_C)` на тех же val-чанках.  
Не смешивать числитель `*_val` со знаменателем `*_all`.

| Ключ | Зачем |
|------|--------|
| `adaptive_gain_vs_best_fixed_val` | **science:** oracle vs лучший fixed (k=8, val) |
| `bootstrap_p_oracle_better_than_best_fixed_val` | **science:** H1 oracle < best_fixed, cluster по episode, val |
| `adaptive_gain_bootstrap_ci_val` | **science:** CI gain vs scheme C на val |
| `adaptive_gain_vs_scheme_c_val` | **science:** если C уже почти oracle — merger не нужен |
| `val_rms` vs `oracle_unrestricted_val_rms` | то же на held-out эпизодах |
| `adaptive_gain_vs_best_fixed_all` / `*_all` | прозрачность; не гейт |
| `n_unique_oracle_schemes` | мало схем → почти state-independent |
| `n_candidates` | из скольких winners собрали C |

| relative_gain | p | Решение по §3 |
|---------------|---|---------------|
| > 15% | < 0.05 | Адаптивность нужна |
| 10–15% | < 0.05 | OPEN, не учить merger; лучше full |
| < 10% или p > 0.05 | — | Adaptive merger не нужен |

Если pair сильно хуже C, а C ≈ oracle — merge1 «adaptive vs pair» врёт. Смотреть **best_fixed**.

---

## 3. Temporal locality

Есть в `--rows-only`: `by_budget["8"].locality`.

- Границы: `adjacent_merge_frequency` (15 чисел), не «диагональ 16×16».
- Совместный сегмент: `co_segment_heatmap[i][j]`.
- Где стартуют длинные куски: `segment_length_by_start`.

| Паттерн | Вывод |
|---------|--------|
| Частоты 0.6+ только на краях / в середине | Можно пробовать **структурированную** fixed-схему (это усиливает C, ослабляет нужду в merger) |
| Плоское облако ~0.3 | Позиция сама по себе не решает |
| Широкие блоки на heatmap | Длинные плато; смотри legal span |

Locality **не** отдельный GO-критерий. Она объясняет §3 и подсказывает, не хватит ли одной маски.

---

## 4. Causal heuristics (урок APB-RVQ)

После cuda: `heuristic_aurocs` и `similarity`.

Считать лучший среди: `cosine`, `neg_l2`, `same_coarse_code`.  
`one_minus_cosine` — контроль знака (должен быть ≈ 1 − AUROC cosine).  
`entropy` / `margin` = `null`: нет логитов VLA, это не ноль.

| Лучший AUROC | Вывод |
|--------------|--------|
| > 0.70 | Similarity heuristic (§11C) — сильный baseline |
| 0.60–0.70 | MLP возможен, не гарантирован |
| < 0.60 | Как APB router: oracle может быть сильным, выучить нельзя |

Spearman/Pearson в `similarity`: > 0.5 сильная связь, < 0.3 почти нет.  
Бины: монотонный рост `oracle_merge_frequency` с cosine — порог по похожести осмыслен.

Связка с §3:

- большой gain + низкий AUROC → **не учить merger** (APB-паттерн);
- маленький gain + высокий AUROC → все чанки одинаково гладкие, хватит C.

---

## 5. Compute — что можно сказать честно

| Источник | Что это |
|----------|---------|
| `compute_estimate["8"].self_attn_quadratic` = 0.25 | Тождество (8/16)² |
| `linear_mlp` = 0.5 | Тождество 8/16 |
| Stage 0 + k4a3 | e2e ~273 ms, доля позиций ~5%, 16→8 внутри BAR ~3% |
| Codec decode | ~7 ms, merger ~0.3 ms — не bottleneck |

**Нельзя:** 16→8 = 4× speedup.  
**Можно:** если science GO — latency только на Path C (retrain меньшего K) или Path E (action-only refiner). Это Stage 4, не post-hoc.

Quality vs «compute» на этом этапе = колонка Oracle в `plan_gaps.md` против k. Pareto по реконструкции, не по wall-clock.

---

## 6. GO / OPEN / NO-GO

Пороги зафиксированы до чтения medium. Бюджет **k=8**.

Критерии **science** (учить ли merger):

| # | Критерий | Где | PASS | FAIL |
|---|----------|-----|------|------|
| 1 | Headroom | `rel_error_increase_vs_no_merge` @8 | < 0.10 | > 0.15 |
| 2 | Adaptive > best fixed | `relative_gain` из `adaptive_gain_vs_best_fixed_val` | > 15% | < 10% |
| 3 | Значимость | `bootstrap_p_oracle_better_than_best_fixed_val` | < 0.05 | > 0.10 |
| 4 | Предсказуемость | лучший AUROC | > 0.65 | < 0.55 |
| S | Span не врёт | `pct_chunks_plan_legal_span` | > 0.70 **или** proxy всё ещё бьёт best_fixed | legal < 0.50 и только full oracle «хорош» |

Критерий «FLOPs < 0.5» **не используется** — он тождественно истинен при k=8.

```
IF 1,2,3,4,S все PASS:
    Science GO → Phase A (маленький merger).
    Latency в статье не обещать, пока нет Path C/E + batch=1 e2e.

ELSE IF 1,2,3,S PASS и 4 FAIL:
    OPEN/риск = APB. Без новых разрешённых фичей — NO-GO на merger.
    Можно оставить paper-claim «oracle redundancy», без learned method.

ELSE IF 2 в зоне 10–15% и p<0.05:
    OPEN → full (2048), не учить.

ELSE:
    Science NO-GO.
```

`stage1_gate == GO` в merge1 **недостаточно**. Пересчитать 2–3 после post-hoc по `*_val`.

Greedy (`retained_gain_vs_random`) на это решение не влияет. ≥ 0.90 — потом можно greedy inference. Метрики greedy считаются на финальных сегментах (stall по `max_span` больше не отдаёт stale trial RMS).

---

## Binding lock (мультиагентка 2026-08-22)

Три независимых судьи (ICRA science / economy / GO-flip). Живой medium **не стопать**. Параллельных ранов нет. Смотреть сюда, когда появятся цифры — не переоткрывать дизайн.

Текущий job = **оптимистичный экран**, не camera-ready таблица:
unrestricted oracle ≤ span≤4 oracle; C-proxy ≥ истинная C; action-AUROC ≥ obs-only AUROC.
Gain / p / headroom / #4 завышены. **NO-GO на этом bound финальный.**

| После post-hoc (`*_val`, k=8) | Что делать | Чего не делать |
|------------------------------|------------|----------------|
| **NO-GO** | Закрыть merger. В лог: `science_decision = NO-GO` + reason. | Restricted oracle, истинная C, full 2048, стратификация, другой seed, merger, latency/success |
| **OPEN** (gain 10–15% и p<0.05; или 1–3–S PASS и #4 FAIL) | **Один** следующий пересчёт: истинная глобальная C на **том же** medium split. Если #4 FAIL при PASS 1–3–S — это APB-ловушка, не лечится C; merger не учить. Full 2048 только если C всё ещё в зоне 10–15%. | Рестарт merge1 с `max_span=4`; стратификация; учить merger; Path C/E |
| **GO** (1,2,3,4,S все PASS) | Сначала истинная C на том же split (единственный незакрытый завышатель gain). Если C всё ещё PASS — Phase A. | Full/стратификация «на всякий случай»; LIBERO/V100 до Phase A |

Не переоткрывать:
- kill/restart live merge1;
- span≤4 enum «чтобы цифры были по §1» до решения (критерий S уже сторожит);
- observation-only фичи в этом oracle (это Stage 2, только если 1–3–S PASS);
- путать `stage1_gate` с science.

---

## 7. Примеры (гипотетические)

### Science GO

| | Пример | |
|--|--------|--|
| legal span | 0.92 | PASS |
| rel-increase @8 | 0.068 | PASS |
| relative_gain, p | 18.5%, 0.01 | PASS |
| AUROC cosine | 0.72 | PASS |

Дальше Phase A. В статье: reconstruction vs k. Не «4× faster BAR».

### Science NO-GO

| | Пример | |
|--|--------|--|
| legal span | 0.45 | WARN |
| rel-increase @8 | 0.185 | FAIL |
| relative_gain, p | 4.2%, 0.15 | FAIL |
| AUROC | 0.52 | FAIL |

Закрыть merger. Если full oracle «красивый» только за счёт span>4 — в отчёт писать proxy.

### Ловушка как APB

rel-increase 6%, gain vs **pair** 20%, но C почти равен oracle, AUROC 0.54.  
merge1-gate может сказать GO. По этому руководству: **NO-GO на merger**, adaptive не state-dependent / не предсказуем.

---

## Чего нет — не ноль

| Нет | Почему |
|-----|--------|
| entropy, margin | нет логитов VLA |
| точный span≤4 oracle | не перебирали все legal partitions |
| точная C | не все 16k схем |
| LIBERO success, V100 e2e | другие этапы |
| learned merger | §7 после Science GO |

---

## Что записать в `MERGE_FINDINGS.md`

```
k=8 legal_span_frac =
k=8 oracle_rel_increase =
k=8 relative_gain_vs_best_fixed_val =     p_val =
k=8 scheme_C_vs_pair =
k=8 best_AUROC =     Spearman =
merge1_stage1_gate =   (pair-only; not science)
science_decision = GO | OPEN | NO-GO   (from *_val)
latency_decision = CONDITIONAL (Stage 0, без изменений)
reason =
```
