# Промпт-референс для агента: HiCoRA на CALVIN

**Статус документа:** нормативный рабочий план для агента-исполнителя  
**Дата:** 2026-09-13  
**Ветка кода:** `calvin_hicora/` в репозитории Adaptive_VLA  
**Спецификация:** `HiCoRA_CALVIN_SPEC.md` (вне репо / Mattermost tmp)  
**Протоколы:** `calvin_hicora/protocols/*.json` — источник научных чисел  
**Отклонения:** `calvin_hicora/FINDINGS_CALVIN.md`

Ты — инженер-исполнитель в проекте HiCoRA-VLA. Доведи CALVIN-ветку от текущего состояния до Definition of Done по плану ниже. Этот файл — нормативный ориентир. Не выдумывай значения, не меняй архитектуру и научные критерии после наблюдения результатов, не подгоняй под итог.

После каждого этапа пиши отчёт в формате в конце документа и дополняй `FINDINGS_CALVIN.md`.

---

## Контекст

**Проект:** HiCoRA — Hierarchical Coarse-to-Residual Action Generation.

Поток:

1. ранний грубый план `q0` (ранний слой ствола);
2. поздняя ограниченная непрерывная поправка `Δz`;
3. один вызов декодера ActionCodec;
4. чанк действий робота.

**Что уже сделано на LIBERO (не переносить как готовую победу):**

- офлайн улучшение D1 ~13%;
- превосходство по closed-loop успеху **не доказано**;
- ускорение ~2.02× против fullbar на batch 1, но ~+8% против идеализированного `coarse24`.

**Задача CALVIN-ветки:** независимо проверить перенос на другом бенчмарке (30 Гц, 34 задачи, цепочки из 5).

Главные вопросы:

1. возникает ли иерархия действий на CALVIN при 30 Гц;
2. помогает ли поздняя поправка на длинных цепочках;
3. сохраняется ли преимущество одного прохода над полной BAR.

---

## Железо

| Параметр | Значение |
|---|---|
| Хост | `ccmplanner.mipt.ru` / `askhabaliev_gs@100.98.148.137` |
| GPU | 2× Tesla V100-SXM2 32 GiB, NVLink |
| Precision | **fp16-mixed для VLA; fp32 для codec** |
| Attention | **sdpa** |
| Запрещено | bf16, FlashAttention-2 (Volta) |
| Codec GPU | 1× V100 |
| VLA GPU | 2× V100 exclusive |
| Диск | variant B (stream zip, actions-only): ≥~180 GiB free for `task_D_D.zip` + cache; full unzip (~500 GiB) больше не требуется |

Обе V100 часто заняты чужим ресерчем — согласовывать exclusive access. Если 2.2B не влезает / слишком медленный на этом хосте → перенос 2.2B на другой кластер. **Не откат на 256M как scientific baseline.**

SSH с локальной машины агента может не иметь ключа; синк/запуск на кластере — через доступ пользователя.

---

## Карта артефактов (текущая)

```text
calvin_hicora/
├── AGENT_PLAN.md                 ← этот файл
├── CALVIN_PORT.md
├── FINDINGS_CALVIN.md
├── README.md
├── configs/
│   ├── actioncodec_calvin.yaml
│   └── fullbar_calvin_2b.yaml
├── protocols/
│   ├── codec_protocol.json       ← codec + geometry source of truth
│   └── baseline_protocol.json    ← VLA 2.2B Table 7
├── evaluator/README.md           ← HULC scaffold (closed-loop only)
├── scripts/
│   ├── convert_calvin.py
│   ├── convert_from_zip_actions_only.py
│   ├── codec_common.py
│   ├── codec_data.py
│   ├── train_codec.py
│   ├── eval_codec_recon.py
│   ├── selftest_codec.py
│   ├── check_dataloader_workers.py
│   ├── fetch_data.sh
│   └── run_gpu_codec_smoke.sh
└── (позже) results/, manifests/, env/, final_gate_protocol.json
```

Нормативные числа читать из JSON-протоколов, не из YAML «на глаз». YAML — зеркало; при расхождении побеждает protocol.

---

## Что уже закрыто (не переделывать)

1. **Backbone** → `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (старый `-Video-` не существует на HF). Fix опечатки до VLA-наблюдения.
2. **Residual RVQ на single-GPU** — runtime patch vendored k-means/dead-code + warm-init с train batch; SHA `third_party/` не трогать. Regression: residual `unique ≫ 1`, primary не мутирует после `train()`.
3. **Primary freeze** — hard-freeze (no-op EMA + snapshot assert), не только `.eval()`.
4. **Global batch 8192** — assert `micro × accum == global` вне smoke/memory-smoke.
5. **Codec gate** — числовые R²/ratio в provisional; scientific pass = `decode_path_parity` + `monotonic_mse`.
6. Индексы (hash chunk_index), gripper ±1, protocol↔manifest, history wipe, resume stage check, RVQ lr как `not_disclosed`.
7. DataLoader на macOS — только из файла, не stdin/heredoc.
8. Конвертер на debug: 4446 кадров, ~1922 кадров/с, `clipped_frames=0`.
9. CPU selftest + `--trainer-smoke`: staged VQ→RVQ, production train/eval — passed.
10. **Dev-split зарегистрирован и реализован:**
    - episode-disjoint carve из official `training/`;
    - **fraction = 0.05**, seed = 0, scene-stratified (см. `codec_protocol.json` → `data.development_split`);
    - `split_codes`: `train|dev|val`; scientific conversion uses
      `format_version = 3` with `valid_length` and honest padding masks;
    - `selection_split: "dev"` в protocol; early stopping на `dev`;
    - official `validation/` → evaluation-domain only;
    - debug с 1 train-эпизодом: carve disabled, smoke может падать на `val` (pipeline only).

> В раннем черновике фигурировало `--dev-fraction 0.1`. **Нормативно зафиксировано 0.05.** Не менять без новой записи в FINDINGS и до научных прогонов.

---

## Открытые блокеры

### B1. Dev split — CLOSED

См. раздел «Что уже закрыто». Перед scientific conversion полного `task_D_D` убедиться, что `data_manifest.development_split.enabled == true` и `split_counts.dev > 0`.

### B2. `not_disclosed` поля кодека — OPEN (блокирует scientific codec run)

| Поле | Статус |
|---|---|
| tokenizer AdamW betas | not disclosed |
| tokenizer weight_decay | not disclosed |
| RVQ post-train learning rate | not disclosed |
| RVQ post-train max_steps | not disclosed |

Действия:

1. Ещё раз проверить ActionCodec paper (Appendix A.3 / C) и released training code.
2. Если нет — оставить `"*_status": "not_disclosed_..."`.
3. Заморозить **sensitivity-кандидаты** в protocol **до** scientific run (2–3 значения), не выдавать их за paper facts.
4. `train_codec.py` уже fail-closed: без smoke/benchmark/memory-smoke scientific path блокируется, пока статусы `not_disclosed`.

Smoke / 200-step benchmark **разрешены** с `*_implementation_candidate`.

### B3. `final_gate_protocol.json` — OPEN (до HiCoRA closed-loop)

Создать и заморозить **до** финальных HiCoRA исходов:

- primary: HiCoRA-D улучшает `AvgLen` относительно `joint_early`;
- co-primary / gate: HiCoRA-D не хуже `fullbar` в заранее выбранном допуске;
- margin, α, multiplicity, power target;
- power simulation по baseline chains;
- treatment of two heads, checkpoint rule;
- число chains = 1000 (стандарт CALVIN), pairing keys.

Не переносить числовой non-inferiority margin из LIBERO.

### B4. HULC reproduction — OPEN, параллельно

Блокирует **только closed-loop evaluation** (этапы 6–7). **Не** блокирует codec scientific run и VLA training.

См. `calvin_hicora/evaluator/README.md`.

Минимум:

1. lock `calvin` / `calvin_env` / MuJoCo в `calvin_hicora/env/`;
2. один episode smoke;
3. HULC D→D на 1000 цепочек;
4. сверить AvgLen / SR с опубликованным (допуски регистрировать до прогона; черновой ориентир AvgLen ±0.10, SR ±3 п.п. — **подтвердить по источнику, не выдумывать**);
5. сырые outcomes + checksums.

---

## Этап 0. Синк и локальная готовность

1. Засинкать `calvin_hicora/` на кластер (SSH-ключ у пользователя).
2. Убедиться, что GPU 0 свободен для codec; для VLA — обе exclusive.
3. Legacy debug data: `data/calvin_converted/debug` (`format_version: 2`);
   regenerate as v3 before the next GPU smoke.
4. Полный `task_D_D` — только при ≥~180 GiB free (variant B): скачать zip,
   затем `scripts/convert_from_zip_actions_only.py` **без** `--no-dev-split`
   (полный unzip / старый 500 GiB budget не нужны). Directory converter
   `convert_calvin.py` остаётся для уже распакованных деревьев.

Команды проверки локально:

```bash
python3 calvin_hicora/scripts/selftest_codec.py --skip-overfit --trainer-smoke
python3 calvin_hicora/scripts/check_dataloader_workers.py data/calvin_converted/debug
```

---

## Этап 1. GPU benchmark кодека

**Цель:** измерить бюджет. Не отборочный прогон.

Скрипт: `calvin_hicora/scripts/run_gpu_codec_smoke.sh`

Мерить и записать в `FINDINGS` + `results/codec/benchmark.json`:

- micro steps/s;
- optimizer seconds/step;
- peak VRAM MiB;
- loss finite / grad norm finite на 200 шагах;
- оценка GPU-hours на base VQ 100k и на RVQ post-train (когда steps заморожены).

Параметры измерительного прогона:

- device `cuda:0`, fp32;
- micro-batch candidate 256, accum candidate 32 → global 8192 на scientific;
- `--benchmark` → 200 шагов.

Решения:

- OOM / неприемлемо медленно для 2.2B позже → другой кластер;
- **не** менять архитектуру/метрики после benchmark;
- **не** подменять scientific baseline на 256M.

**Артефакт:** `results/codec/benchmark.json`, запись в FINDINGS, заморозка codec GPU-hour budget.

---

## Этап 2. Scientific codec run

**Предусловие:** B2 закрыт или явно заморожены sensitivity-кандидаты; scientific data с enabled `dev`.

### 2.1 Base VQ

```bash
python3 calvin_hicora/scripts/train_codec.py \
  --stage base_vq --seed 0 \
  --data-root "$CALVIN_DATA_ROOT" \
  --output-dir checkpoints/codec/base_vq_s0 \
  --protocol calvin_hicora/protocols/codec_protocol.json \
  --device cuda:0
# затем --seed 1 → base_vq_s1
```

- single-layer VQ, `n_quantizers=1`, latent positions 16;
- train на `split=train`, selection/early-stop на `dev`;
- global batch 8192, lr `2e-4`, max_steps `100000` (Appendix A.3);
- источник Stage 1: ActionCodec Appendix C / A.3.

### 2.2 RVQ post-training

```bash
python3 calvin_hicora/scripts/train_codec.py \
  --stage rvq_posttrain \
  --base-vq-checkpoint checkpoints/codec/base_vq_s0/best/model \
  --seed 0 \
  --data-root "$CALVIN_DATA_ROOT" \
  --output-dir checkpoints/codec/rvq_posttrain_s0 \
  --device cuda:0
```

- Residual levels 1–2; **encoder + primary codebook frozen** (hard-freeze);
- lr / steps — из статьи или замороженный sensitivity-кандидат;
- warm residual init обязателен.

### 2.3 Eval

```bash
python3 calvin_hicora/scripts/eval_codec_recon.py \
  --model checkpoints/codec/rvq_posttrain_s0/best/model \
  --data-root "$CALVIN_DATA_ROOT" \
  --split dev \
  --enforce-gate \
  --output results/codec/rvq_s0_dev.json
```

- scientific pass = `decode_path_parity` + `monotonic_mse`;
- `val` — отдельный descriptive отчёт, не gate;
- pipeline debug (`is_pipeline_validation: true`) — `--enforce-gate` запрещён.

**Критерий go (SPEC §11.3, качественный):** `q0` исполняем, но не идеален; тонкие уровни добавляют измеримую информацию.

**Блокирует:** VLA training.

---

## Этап 3. VLA trainer 2.2B

**Ещё не реализовано.** Ориентир: `third_party/actioncodec/scripts/train_vla.py` + `configs/fullbar_calvin_2b.yaml` + `protocols/baseline_protocol.json`.

### 3.1 Dataset

Новый `calvin_vla_dataset.py` (или эквивалент):

- `rgb_static` + `rgb_gripper` → 224×448;
- язык из `auto_lang_ann.npy`;
- state из `robot_obs`;
- chunk 30; action tokens из обученного CALVIN codec;
- splits: `train` / `dev`; official `val` только финал.

### 3.2 Trainer

| Поле | Значение | Источник |
|---|---|---|
| backbone | `SmolVLM2-2.2B-Instruct` | SPEC §12 / HF |
| BAR | token_budget 48, num_blocks 3, vocab 2048 | geometry |
| precision | fp16-mixed | V100 |
| attention | sdpa | V100 |
| FSDP | full shard + activation ckpt | baseline protocol |
| peak lr | 1e-4 | Table 7 |
| warmup | 1000 | Table 7 |
| schedule | cosine → 1e-5 | Table 7 |
| global batch | 128 (= 1×2×64 accum) | Table 7 |
| max_steps | 30000 | Table 7 |
| weight_decay | 1e-10 | released `bar.yaml` (не Table 7) |
| betas | [0.9, 0.999] | released `bar.yaml` |
| seeds | 0, 1 | protocol |

Сначала D→D; основной режим ABCD→D; ABC→D — stretch / отдельный A* трек.

### 3.3 Benchmark 2.2B

200 шагов на 2×V100 exclusive → `results/baselines/vla_2b_benchmark.json` → freeze GPU-hours/seed.

OOM → move cluster, не 256M.

### 3.4 Validation

- token accuracy level0/1/2;
- decoded position/rotation MSE, gripper sign accuracy;
- raw predictions;
- полный val domain по protocol (`evaluate_complete_validation_split`).

**Стоп:** baseline слаб / нестабилен → чинить baseline, не HiCoRA.

---

## Этап 4. Depth sweep и `joint_early`

После хорошего fullbar:

1. taps на глубинах 6, 12, 18 из 24 (не догма «всегда 12»);
2. для каждой — ранняя q0-голова + допустимый участок ствола;
3. метрики: q0 agreement, codec recon, latency; closed-loop — когда evaluator готов;
4. выбрать одну глубину **только по `dev`** + заранее записанное правило;
5. baselines: `fullbar_calvin`, `coarse_full_calvin`, `joint_early_calvin`.

**Блокирует:** HiCoRA-D.

---

## Этап 5. HiCoRA-D

### 5.1 Кэш

`h_early`, `h_final`, фактический `q0`/`z0`, целевой латент, action target, маски, identifiers; SHA к модели, codec, norm, manifest, скрипту.

### 5.2 Базис и предел

- `B` только на train;
- rank ladder 16/32/64;
- выбор ранга на `dev`, правило заранее;
- joint-coverage `rho`;
- test / official long-horizon eval states не использовать для `B`/`rho`.

### 5.3 Голова

\[
\Delta z = (\rho \odot \tanh f(h_{\mathrm{final}}, \operatorname{sg}(z_0))) B^\top
\]

Две seed-головы; сначала только голова, baseline заморожен.

### 5.4 Проверки до simulator eval

- нулевая инициализация ≡ `joint_early`;
- норма поправки держится;
- нет градиентов в `z0`, codec, `B`, `rho`, замороженном стволе;
- конфиг голов одинаков кроме seed;
- real-model parity под тем же dtype/autocast;
- офлайн улучшение на held-out.

---

## Этап 6. Closed-loop evaluation

**Предусловия:** B3 + B4.

### Руки

1. `fullbar_calvin`
2. `coarse_full_calvin`
3. `joint_early_calvin`
4. `hicora_d_s0_calvin`
5. `hicora_d_s1_calvin`

### Условия

Те же 1000 цепочек, начальные состояния, инструкции, rollout seeds, horizon, oracle; хеши и pairing key в каждой ячейке.

### Метрики

SR1–SR5, AvgLen; 34 подзадачи; позиция первого сбоя; latency p50/p95; peak VRAM; число проходов/слоёв/decoder calls.

### Статистика

Парные по цепочке; bootstrap по цепочке; seeds раздельно; primary endpoint/margin/chains — до запуска; test не для выбора.

---

## Этап 7. Отчёт и передача

1. `FINDINGS_CALVIN.md` — всё, включая отрицательное и `not_disclosed`.
2. `CALVIN_PORT.md`, `manifests/`, configs, protocols, results, `AUDIT_MANIFEST.sha256`.
3. Audit archive без дублирования огромных весов — ссылки + SHA.
4. Однозначный вывод: перенос подтверждён / не подтверждён / стенд не способен ответить — **без переименования гипотезы**.

---

## Что не входит в этот minimum plan

- HiCoRA-G / PPO — только после детерминированной проверки.
- RoboTwin — отдельная ветка.
- ABC→D как primary, PD/one-pass comparator set, ≥3 end-to-end seeds — **A\* submission track**, не blocker текущего MVP (но не забывать).

---

## Жёсткие правила

1. **Не выдумывать значения.** Нет в статье → `not_disclosed` + sensitivity, зарегистрированная до прогона.
2. **Не менять архитектуру и научные критерии после наблюдения результатов.**
3. **Не использовать test split ни для какого решения.**
4. **Не откатываться на 256M как scientific baseline.** Только pipeline-smoke.
5. **Не править SHA `third_party`.** Только runtime patch / wrapper в `calvin_hicora/`.
6. **`val` не участвует в selection.** Только `dev`.
7. **Seeds не пулить.** Показывать раздельно.
8. **Fail-closed.** При сомнении — стоп + FINDINGS.
9. **Всё, что можно зарегистрировать до прогона — регистрировать до прогона.**
10. **Инфраструктурные отклонения (fp16, sdpa, HTTP download, single-GPU codebook patch) — в FINDINGS с датой и причиной.**
11. **Не мутировать frozen LIBERO `experiments/`.** Вся CALVIN-работа в `calvin_hicora/`.

---

## График (ориентир)

| Неделя | Результат |
|---|---|
| 1 | B2 `not_disclosed`/sensitivity freeze; B3 draft `final_gate`; B4 HULC параллельно; GPU sync |
| 2 | GPU benchmark кодека; freeze codec GPU budget |
| 3 | base VQ + RVQ post-train (2 seeds); codec eval на `dev` |
| 4 | VLA 2.2B benchmark; fullbar D→D |
| 5 | fullbar ABCD→D; второй seed |
| 6 | depth sweep; `joint_early`; `coarse_full` |
| 7 | базис, rho, две HiCoRA-D головы |
| 8 | paired 1000-chain gate, latency, отчёт |

HULC параллельно неделям 1–3. Если `calvin_env` не поднимается — блокирует только недели 7–8 (closed-loop).

---

## Definition of Done

1. Воспроизводимый CALVIN evaluator, подтверждённый HULC D→D.
2. CALVIN-native codec, scientific gate passed на `dev`.
3. Fullbar на ≥2 seeds, устойчивый ненулевой long-horizon результат.
4. `coarse_full` и `joint_early`.
5. Две HiCoRA-D головы.
6. Замороженный paired `final_gate_protocol.json`.
7. SR1–SR5 и AvgLen на сырых 1000 цепочках.
8. Latency и память для всех рук.
9. Полный provenance / audit.
10. Однозначный вывод без переименования гипотезы.

---

## Ближайший конкретный next step (сейчас)

1. Закрыть **B2** (найти или заморозить sensitivity для codec optimizer / RVQ steps).
2. Синк на кластер + `bash calvin_hicora/scripts/run_gpu_codec_smoke.sh` на GPU 0.
3. Записать benchmark в `results/codec/benchmark.json` и FINDINGS.
4. Параллельно: HULC env lock; черновик `final_gate_protocol.json`.
5. После codec gate на scientific data — VLA trainer 2.2B.

---

## Формат отчёта агента

После каждого этапа:

```text
[ЭТАП N] [статус: passed/failed/blocked]
- что сделано
- артефакты (пути)
- числа (если есть)
- отклонения от плана (если есть)
- блокеры (если есть)
- следующий шаг
```

Отрицательные результаты и отклонения — **обязательно**, не скрывать.
