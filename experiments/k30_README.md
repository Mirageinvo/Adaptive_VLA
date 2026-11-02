# K-30: дискретные планы корней и непрерывные уточнения детей

План: `K30_Hybrid_Root_Plans_Agent_Plan.md` (v1.0, 2026-11-02).

| Пакет | Файлы | Что делает |
|---|---|---|
| A | `k30_codec.py`, `k30_data.py` | Грубые планы корней (средние нормированной руки на 4 участках по 5 команд, маски хвостов), weighted k-means++ K=128 только на train, метки train/val/teacher по float32 P, отчёт и допуск (val MSE ≤ 0.5 · постоянный) |
| B | `k30_readout.py`, `k30_model.py`, `k30_train.py` | Фиксированная геометрия, план после слоя 8 (hard: P[argmax] + STE; soft: softmax@P), plan_proj до Expand1, readout K-25 с C0 = half·atanh(b), потери L_action + 0.02·L_code + 0.1·L_coarse, warmup 500 + main 4000, real-check (+ micro), fitcheck |
| C | `k30_policy.py`, `k30_harness.py`, `k30_protocol.py` | Рука для холодного харнесса K-9h (как K-25), журнал плана/уточнений, четыре плана SR, независимая пересборка блока на CPU, smoke-повтор классификатора |
| D | `k30_analyze.py`, `k30_latency.py`, `k30_run.sh` | H − C по сидам, Holm, совместный бутстрэп, описательные контроли; задержка H/C; цепочка |

Запуск (после окончания K-29):

```bash
cd /home/malinin_aa/Adaptive_VLA
export PYTHONPATH="$HOME/LIBERO" MUJOCO_GL=egl HF_HUB_OFFLINE=1
bash experiments/k30_run.sh selftest cpu
bash experiments/k30_run.sh prepare            # CPU: кодек + base prepare
bash experiments/k30_run.sh real-check cuda:0
bash experiments/k30_run.sh fitcheck cuda:0
nohup bash experiments/k30_run.sh run cuda:0 > /dev/null 2>&1 &
tail -F logs/k30/run.log
```

Кодек можно подготовить заранее на CPU: `python3 experiments/k30_data.py codec`
(cwd — основная копия с данными; выход только в `data/k30`, `reports/k30`).
