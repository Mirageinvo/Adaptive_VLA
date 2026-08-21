"""K-5b: проверка стенда LIBERO на завышение успеха и воспроизводимость сброса.

ВОПРОС. В нашем цикле раунд идёт, пока не завершатся ВСЕ среды, и уже
финишировавшие продолжают шагать вместе с остальными. Если векторная обёртка
их автоматически перезапускает, то среда начинает НОВЫЙ эпизод, а накопление
`reward = clip(reward + r, 0, 1)` может зачесть успех оттуда. Тогда успех
завышается, и тем сильнее, чем длиннее раунд — то есть неравномерно по
горизонтам, что прямо искажает сравнение.

Отдельно проверяется воспроизводимость сброса: два вызова reset с одним
init_state_id обязаны дать побитово одинаковое наблюдение. Мы уже поймали, что
состояние протекает МЕЖДУ конфигурациями (ячейка давала 10 успехов будучи
восьмой в процессе и 9 будучи первой), и лечили это одной конфигурацией на
процесс. Здесь проверяется более узкое: воспроизводим ли сброс внутри процесса.

ЧТО ДЕЛАЕТ. Один раунд настоящей политикой с записью по каждому шагу и каждой
среде: флаг done, приращение награды, отпечаток наблюдения. Затем три проверки:

  1. МОНОТОННОСТЬ done: True не должен превращаться обратно в False;
  2. НАГРАДА ПОСЛЕ ЗАВЕРШЕНИЯ: после первого done приращений быть не должно;
  3. АВТОСБРОС: отпечаток наблюдения не должен возвращаться к начальному.

Третья — прямое обнаружение перезапуска: если среда после done снова показывает
своё стартовое наблюдение, значит она начала новый эпизод.

Политика нужна настоящая: со случайными действиями ни одна среда не завершится
раньше предела времени, и ситуация «одна закончила, другие идут» не возникнет.

Запуск:
    python3 experiments/k5b_harness_check.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k5b_harness_check.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --task-suite 10 --task-id 0 --n-envs 10
"""

import argparse
import hashlib
import json
import os
import sys


def analyze(trace, init_fp):
    """Три проверки по записанному следу.

    trace: список шагов, в каждом dict(done=[...], dr=[...], fp=[...]).
    init_fp: отпечатки НАЧАЛЬНЫХ наблюдений по средам.
    """
    n = len(init_fp)
    first_done = [None] * n
    flips, after_reward, restarts = [], [], []
    for t, st in enumerate(trace):
        for i in range(n):
            if st["done"][i] and first_done[i] is None:
                first_done[i] = t
            if first_done[i] is not None and t > first_done[i]:
                if not st["done"][i]:
                    flips.append((i, t))
                if st["dr"][i] > 0:
                    after_reward.append((i, t, st["dr"][i]))
                if st["fp"][i] == init_fp[i]:
                    restarts.append((i, t))
    return dict(first_done=first_done, flips=flips,
                after_reward=after_reward, restarts=restarts)


def selftest():
    """Синтетические следы С ИЗВЕСТНЫМ ОТВЕТОМ для разбора.

    Проверяется не среда, а сам разбор: он обязан ловить каждый из трёх
    дефектов по отдельности и молчать на чистом следе.
    """
    init = ["A", "B"]
    # чистый след: среда 0 завершилась на шаге 1 и дальше стоит
    clean = [dict(done=[False, False], dr=[0, 0], fp=["x", "y"]),
             dict(done=[True, False], dr=[1, 0], fp=["p", "y"]),
             dict(done=[True, False], dr=[0, 0], fp=["p", "y"]),
             dict(done=[True, True], dr=[0, 1], fp=["p", "q"])]
    r = analyze(clean, init)
    assert r["first_done"] == [1, 3], f"момент завершения неверен: {r}"
    assert not r["flips"] and not r["after_reward"] and not r["restarts"], \
        f"на чистом следе сработала ложная тревога: {r}"

    # 1. done переворачивается обратно
    bad = [dict(done=[True], dr=[0], fp=["x"]),
           dict(done=[False], dr=[0], fp=["x"])]
    assert analyze(bad, ["A"])["flips"], "переворот done не пойман"

    # 2. награда после завершения
    bad = [dict(done=[True], dr=[1], fp=["x"]),
           dict(done=[True], dr=[1], fp=["x"])]
    assert analyze(bad, ["A"])["after_reward"], "награда после done не поймана"

    # 3. наблюдение вернулось к начальному — автосброс
    bad = [dict(done=[True], dr=[0], fp=["x"]),
           dict(done=[True], dr=[0], fp=["A"])]
    assert analyze(bad, ["A"])["restarts"], "автосброс не пойман"

    # и обратное: возврат к начальному ДО завершения тревогой не считается
    ok = [dict(done=[False], dr=[0], fp=["A"]),
          dict(done=[True], dr=[0], fp=["x"])]
    assert not analyze(ok, ["A"])["restarts"], "ложная тревога до завершения"
    print("самопроверка пройдена: разбор ловит переворот done, награду после "
          "завершения и автосброс, и молчит на чистом следе")


def fp_of(obs, i):
    """Отпечаток наблюдения одной среды: картинка плюс состояние."""
    h = hashlib.sha1()
    for k in ("agentview_image", "robot0_eye_in_hand_image", "state"):
        if k in obs:
            h.update(bytes(obs[k][i].tobytes()))
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)

    import numpy as np
    import torch
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01,  # noqa: E402
                       STATE_Q99, VisionLanguageActionProcessor, dict_apply,
                       get_cfg, get_envs, process_state, prompt_template,
                       seed_everything)

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    pos_off = args.pos_offset
    if pos_off is None:
        tb = json.load(open(args.offset_table))
        pos_off = int(tb["offsets_by_suite"][args.task_suite][args.task_id])

    seed_everything(args.seed)
    envs, task_desc = get_envs(args.task_suite,
                               {"task_id": args.task_id, "image_size": 224},
                               args.n_envs)
    n = args.n_envs
    try:
        # ---- ПРОВЕРКА 0: воспроизводимость сброса ------------------------
        # Два вызова с одним init_state_id обязаны дать одно и то же
        # наблюдение. Если нет — «одинаковые начальные состояния» между
        # горизонтами не обеспечены, и парное сравнение невозможно.
        opts = [{"init_state_id": i} for i in range(n)]
        o1 = envs.reset(options=opts)
        f1 = [fp_of(o1, i) for i in range(n)]
        o2 = envs.reset(options=opts)
        f2 = [fp_of(o2, i) for i in range(n)]
        same = sum(a == b for a, b in zip(f1, f2))
        print("=" * 74)
        print("0. ВОСПРОИЗВОДИМОСТЬ СБРОСА")
        print("=" * 74)
        print(f"  сред {n}, совпало отпечатков после повторного reset: {same}")
        if same != n:
            print("  СБРОС НЕ ВОСПРОИЗВОДИМ — парное сравнение горизонтов "
                  "невозможно,\n  нужна непарная статистика и больше эпизодов")

        # ---- модель грузится ПОСЛЕ создания сред -------------------------
        model = SmolVLABlockwiseAR.from_pretrained(
            **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
        if next(model.parameters()).dtype != dtype:
            model = model.to(dtype)
        processor = VisionLanguageActionProcessor.from_pretrained(
            args.ckpt, trust_remote_code=True, mode="discrete")
        max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
        tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])
        print(f"\n  задача: «{task_desc}», офсет {pos_off}, H={args.horizon}")

        # ---- один раунд с записью следа ----------------------------------
        obs = envs.reset(options=opts)
        init_fp = [fp_of(obs, i) for i in range(n)]
        reward = np.zeros(n)
        done = np.zeros(n, bool)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n)
        for _ in range(args.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)

        trace, steps = [], 0
        while not np.all(done) and steps < args.max_steps:
            state = ((process_state(obs["state"]) - STATE_Q01)
                     / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0)
            im1 = tf(torch.tensor(
                obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2))
            im2 = tf(torch.tensor(
                obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
            ).permute(0, 3, 1, 2))
            image = torch.cat([im1, im2], dim=-1)
            msgs = []
            for i in range(n):
                m = prompt_template(
                    state[i], None, task_desc,
                    mode=cfg.MODEL.vla_processor.kwargs.mode,
                    action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                    action_token_len=cfg.MODEL.action_processor.token_len)
                m[1]["content"] = m[1]["content"][1:]
                msgs.append(m)
            texts = processor.apply_chat_template(msgs,
                                                  add_generation_prompt=True)
            batch = processor(text=texts,
                              images=[[image[i].numpy()] for i in range(n)],
                              return_tensors="pt", padding=True,
                              padding_side="left",
                              action_processor_kwargs={"embodiment_ids": 0})
            batch = dict_apply(lambda x: x.to(dev, dtype), batch)
            with torch.no_grad():
                toks = model.generate(**batch, position_offset=pos_off,
                                      do_sample=False, initial_position_shift=1)
                act = processor.action_processor.decode(toks.tolist())[0]
            action = np.copy(act)
            action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
            action[..., -1] = -action[..., -1]

            for t in range(args.horizon):
                if np.all(done) or steps >= args.max_steps:
                    break
                obs, r_, done, _ = envs.step(action[:, t])
                trace.append(dict(done=[bool(x) for x in done],
                                  dr=[float(x) for x in r_],
                                  fp=[fp_of(obs, i) for i in range(n)]))
                reward = np.clip(reward + r_, 0, 1)
                steps += 1
    finally:
        try:
            envs.close()
        except Exception:
            pass

    res = analyze(trace, init_fp)
    print("\n" + "=" * 74)
    print(f"СЛЕД: шагов {steps}, сред {n}")
    print("=" * 74)
    fd = res["first_done"]
    print(f"  успех по накоплению награды: {int((reward >= 1).sum())}/{n}")
    print(f"  моменты первого done: {fd}")
    spread = [x for x in fd if x is not None]
    if spread:
        print(f"  разброс завершения: {min(spread)}…{max(spread)} шаг, "
              f"то есть {max(spread) - min(spread)} шагов среды шагали "
              f"после своего завершения")

    print(f"\n  1. переворотов done (True->False): {len(res['flips'])}")
    print(f"  2. приращений награды ПОСЛЕ завершения: "
          f"{len(res['after_reward'])}")
    if res["after_reward"]:
        print(f"     первые: {res['after_reward'][:5]}")
    print(f"  3. возвратов наблюдения к начальному после завершения: "
          f"{len(res['restarts'])}")
    if res["restarts"]:
        print(f"     первые: {res['restarts'][:5]}")

    ok = not (res["flips"] or res["after_reward"] or res["restarts"])
    print("\n" + ("  СТЕНД ЧИСТ: завершившиеся среды не перезапускаются и не "
                  "добавляют награду.\n  Накопление reward по всему раунду "
                  "успех не завышает."
                  if ok else
                  "  ОБНАРУЖЕН ДЕФЕКТ. Успех может быть завышен: надо "
                  "фиксировать исход\n  в момент первого done и дальше среду "
                  "игнорировать, а не суммировать\n  награду до конца раунда."))

    if args.out:
        json.dump(dict(clean=ok, first_done=fd, flips=len(res["flips"]),
                       after_reward=len(res["after_reward"]),
                       restarts=len(res["restarts"]), steps=steps,
                       reset_reproducible=(same == n),
                       success=int((reward >= 1).sum()), n_envs=n,
                       task=task_desc, suite=args.task_suite,
                       task_id=args.task_id, horizon=args.horizon),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
