#!/usr/bin/env python3
"""Сравнение времени закрытия T32 и демонстратора на кадрах K-21.

Запуск из корня Adaptive_VLA с experiments в PYTHONPATH. Без --device
подсчитывает окна с закрытием. С --device cuda:N прогнозирует команду
схвата через замороженный референс K-32. T32 использует ту же команду.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import os

import numpy as np


TASKS = (0, 6, 8)
CHUNK = 8
OPEN, CLOSE = -1, 1


def expert_close(grip, valid, prev):
    for tau in range(min(CHUNK, len(grip) - 2, len(valid) - 2)):
        before = prev if tau == 0 else grip[tau - 1]
        if (before == OPEN and valid[tau:tau + 3].all() and
                (grip[tau:tau + 3] == CLOSE).all()):
            return tau
    return None


def model_close(grip, prev):
    # Тот же критерий K-38 для полного восьмишагового чанка.
    history = np.r_[prev, grip]
    for tau in range(2, CHUNK - 2):
        if ((history[tau - 2:tau + 1] == OPEN).all() and
                (history[tau + 1:tau + 4] == CLOSE).all()):
            return tau
    return None


def select_windows(data, names, split, limit):
    row_key = {"train": "train_rows", "val_sel": "val_rows"}[split]
    by_task = defaultdict(list)
    task_ids = {names[str(task)]: task for task in TASKS}
    for row in data[row_key]:
        task = task_ids.get(str(data["task"][row]))
        if task is not None:
            by_task[task].append(int(row))
    commands = -np.sign(data["act"][..., 6])
    valid = data["valid"]
    selected = {}
    for task in TASKS:
        by_episode = defaultdict(list)
        for row in by_task[task]:
            by_episode[int(data["episode"][row])].append(row)
        windows, missing_prev = [], 0
        for episode, rows in by_episode.items():
            rows.sort(key=lambda r: int(data["step"][r]))
            for previous, row in zip(rows[:-1], rows[1:]):
                step = int(data["step"][row])
                if step % CHUNK or not valid[row, :CHUNK].all():
                    continue
                prev_i = step - int(data["step"][previous]) - 1
                if not 0 <= prev_i < commands.shape[1] or not valid[previous, prev_i]:
                    missing_prev += 1
                    continue
                prev = int(commands[previous, prev_i])
                if prev == 0 or not (commands[row, :CHUNK] != 0).all():
                    continue
                tau = expert_close(commands[row], valid[row], prev)
                if tau is not None:
                    order = hashlib.sha256(
                        f"k38-demo-close-v1|{task}|{episode}".encode()).digest()
                    windows.append((order, episode, row, prev, tau))
                    break
        chosen = sorted(windows)[:limit]
        selected[task] = [(row, prev, tau) for _, _, row, prev, tau in chosen]
        print(f"t{task}: демо {len(by_episode)}, с подходящим окном {len(windows)}, "
              f"выбрано {len(chosen)}, окон без предыдущей команды {missing_prev}",
              flush=True)
    return selected


def report(task, seed, n, deltas):
    print(f"s{seed} t{task}: событие T32 применимо в {len(deltas)}/{n} окон", flush=True)
    if deltas:
        d = np.asarray(deltas)
        errors = np.abs(d[:, None] - np.array([-2, 0, 2]))
        hist = dict(sorted(Counter(deltas).items()))
        print(f"  эксперт минус T32, шаги: {hist}", flush=True)
        print("  средняя абсолютная ошибка, шаги: "
              f"early {errors[:, 0].mean():.2f}, T32 {errors[:, 1].mean():.2f}, "
              f"late {errors[:, 2].mean():.2f}, "
              f"лучший сдвиг отдельно для каждого окна {errors.min(axis=1).mean():.2f}",
              flush=True)


def predict(torch, base, data, selected, seed, device):
    import k20_data as kd20
    import k32_policy as policy
    import k32_protocol as protocol

    mod, bar, model, decoder = protocol.load_source_dec(torch, base, seed, device)
    arm = policy.Arm(torch, mod, decoder, model, kind="R", seed=seed)
    for task, windows in selected.items():
        deltas = []
        for row, prev, tau in windows:
            state = data["state8"][[row]]
            batch = kd20.prompt_batch(bar, [np.array(data["IMG"][row], copy=True)],
                                      kd20.norm_states(state), [data["task"][row]])
            arm.begin(task, 0, int(data["episode"][row]))
            a, _ = arm.decide(batch, int(data["pos_offset"][row]), state,
                              int(data["step"][row]))
            predicted = model_close(-a[:, 6], prev)
            if predicted is not None:
                deltas.append(tau - predicted)
        report(task, seed, len(windows), deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="cuda:N для прогнозов")
    parser.add_argument("--split", choices=("train", "val_sel"), default="train")
    parser.add_argument("--per-task", type=int, default=24)
    args = parser.parse_args()
    if args.per_task < 1:
        parser.error("--per-task должен быть положительным")

    import k21_data as d21
    import k32_protocol as protocol

    base = protocol.load_base()
    data = d21.load_prepared(require_teacher=False)
    print(f"Диагностика K-21, набор {args.split}, по одному окну на демо", flush=True)
    selected = select_windows(data, base["tasks"], args.split, args.per_task)
    if not args.device or not any(selected.values()):
        return

    import torch
    import k36_harness as harness
    import k37_confirm as confirm

    idx = confirm.dev_index(args.device)
    gpu = confirm.gpu_query(idx)
    lock = confirm.Lock(os.path.join(confirm.lock_dir(), gpu["uuid"] + ".lock"),
                        f"карта {gpu['uuid']}", confirm.Busy)
    try:
        confirm.check_gpu(idx, confirm.RES["min_free_gb"])
        confirm.bind_device(torch, idx, gpu)
        for seed in (0, 1):
            confirm.check_gpu(idx, confirm.RES["min_free_gb"])
            predict(torch, base, data, selected, seed, args.device)
            harness.free(torch)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
