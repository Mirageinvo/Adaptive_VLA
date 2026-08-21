"""K-5b: развёртка по фиксированному горизонту исполнения в LIBERO.

ВОПРОС. Политика предсказывает 20 действий, а официальный протокол исполняет
первые ЧЕТЫРЕ и вызывает её заново (`eval_libero.py:298` --horizon=4,
`eval_libero_bar.sh:13` HORIZON=4). Один вызов стоит ~198 мс (K-5a2), то есть
частота вызовов — главная статья расходов. Вопрос фазы B: существует ли вообще
полезный размен между частотой перепланирования и успехом, и РАЗЛИЧАЕТСЯ ли
безопасный горизонт между состояниями. Без гетерогенности адаптивность не
нужна при любой форме средней кривой.

ПОЧЕМУ СВОЙ ЦИКЛ, А НЕ ВЫЗОВ eval_libero.py. Официальный скрипт (а) пишет mp4
на каждый эпизод через imageio, (б) НЕ УМЕЕТ выключать усреднение: аргумент
объявлен как `type=bool`, поэтому `--ensemble False` даёт True — проверено,
выключает только пустая строка, (в) не логирует ни времени, ни числа вызовов
политики. Настройка среды переиспользуется из `scripts/utils.py` целиком:
get_envs, get_cfg, process_state, prompt_template, ActionEnsembler. Свой здесь
только цикл.

УСРЕДНЕНИЕ — КОНФАУНД, КОТОРЫЙ НАДО РАЗВЕСТИ. ActionEnsembler копит предсказания
на каждый абсолютный момент времени и усредняет их. При H=4 и чанке 20 каждое
исполняемое действие покрыто ПЯТЬЮ планами разной свежести, при H=20 — одним.
Значит падение успеха на длинных горизонтах смешивает «устаревание наблюдения»
с «потерей усреднения». Меряем оба режима: ensemble=off отвечает вопросу
метода, ensemble=on воспроизводит официальный протокол.

ОДИНАКОВЫЕ НАЧАЛЬНЫЕ СОСТОЯНИЯ. init_state_id детерминирован (i + k*n_envs),
поэтому эпизод с данным индексом одинаков при всех горизонтах, и сравнение
парное. Сравнивать разные наборы эпизодов нельзя.

ЧТО СОХРАНЯЕТСЯ. По каждому эпизоду: успех, число шагов среды, число вызовов
политики, суммарное и поэпизодное время политики, длина исполненного префикса.
Из этого считаются вызовы на исполненное действие и миллисекунды на действие —
две главные оси кривой размена.

Запуск:
    python3 experiments/k5b_fixed_horizon_eval.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k5b_fixed_horizon_eval.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --task-suite goal --task-id 0 --n-envs 5 --k-set 1 \\
        --horizons 4 8 --ensemble both --out data/k5b_pilot.json
"""

import argparse
import json
import os
import statistics
import sys
import time


def summarize(eps):
    """Агрегаты по эпизодам одного режима.

    Вызовы на ИСПОЛНЕННОЕ действие — главная величина: именно её сокращает
    метод. Число вызовов на эпизод само по себе обманчиво, потому что эпизоды
    разной длины.
    """
    n = len(eps)
    steps = sum(e["env_steps"] for e in eps)
    calls = sum(e["policy_calls"] for e in eps)
    ms = sum(e["policy_ms"] for e in eps)
    return dict(
        episodes=n,
        success_rate=sum(e["success"] for e in eps) / max(n, 1),
        env_steps=steps, policy_calls=calls,
        calls_per_action=calls / max(steps, 1),
        policy_ms_total=ms,
        ms_per_action=ms / max(steps, 1),
        ms_per_call=ms / max(calls, 1),
        mean_episode_steps=steps / max(n, 1),
    )


def selftest():
    """Учёт вызовов и шагов на синтетике С ИЗВЕСТНЫМ ОТВЕТОМ.

    Главный риск этого скрипта — не физика, а АРИФМЕТИКА УЧЁТА: легко посчитать
    вызовы на эпизод вместо вызовов на исполненное действие и получить
    «ускорение» из разной длины эпизодов. Проверяем на случае, где ответ
    известен точно.
    """
    # Два эпизода по 40 шагов. При H=4 вызовов 10 на эпизод, при H=8 — пять.
    for H, want in ((4, 0.25), (8, 0.125), (20, 0.05)):
        eps = [dict(success=True, env_steps=40, policy_calls=40 // H,
                    policy_ms=(40 // H) * 200.0) for _ in range(2)]
        s = summarize(eps)
        assert abs(s["calls_per_action"] - want) < 1e-12, \
            f"вызовов на действие при H={H}: {s['calls_per_action']} против {want}"
        assert abs(s["ms_per_call"] - 200.0) < 1e-9, "мс на вызов поплыли"
        assert abs(s["ms_per_action"] - 200.0 * want) < 1e-9, \
            "мс на действие не согласованы с вызовами на действие"

    # Ловушка: эпизоды РАЗНОЙ длины. Вызовы на эпизод растут, а на действие —
    # нет. Если бы мы считали на эпизод, вышло бы ложное различие.
    a = [dict(success=True, env_steps=40, policy_calls=10, policy_ms=2000.0)]
    b = [dict(success=True, env_steps=80, policy_calls=20, policy_ms=4000.0)]
    assert summarize(a)["calls_per_action"] == summarize(b)["calls_per_action"], \
        "нормировка на исполненные действия не сработала"
    assert summarize(a)["policy_calls"] != summarize(b)["policy_calls"], \
        "проверка бессмысленна: число вызовов на эпизод совпало"

    s = summarize([dict(success=True, env_steps=10, policy_calls=3, policy_ms=1.0),
                   dict(success=False, env_steps=10, policy_calls=3, policy_ms=1.0)])
    assert s["success_rate"] == 0.5, "успех считается неверно"
    print("самопроверка пройдена: вызовы и время нормируются на ИСПОЛНЕННЫЕ "
          "действия, разная длина эпизодов не создаёт ложного различия")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="goal",
                    help='goal, spatial, object или "10" для long')
    ap.add_argument("--task-id", type=int, nargs="+", default=[0])
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[1, 2, 4, 6, 8, 12, 16, 20])
    ap.add_argument("--ensemble", choices=["on", "off", "both"], default="both")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--k-set", type=int, default=1,
                    help="раундов по n-envs; официальный протокол берёт 50 "
                         "эпизодов на задачу, то есть k_set*n_envs = 50")
    ap.add_argument("--pos-offset", type=int, default=None,
                    help="по умолчанию берётся из data/pos_offset_table.json")
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600,
                    help="жёсткий предел шагов среды на эпизод")
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
    sys.path.insert(0, os.path.join(root, "scripts"))

    import numpy as np
    import torch
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,  # noqa: E402
                       ActionEnsembler, dict_apply, get_cfg, get_envs,
                       process_state, prompt_template, seed_everything)
    from utils.vla_tokenizer import VisionLanguageActionProcessor

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if dev.type == "cuda":
        cc = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
        if args.dtype == "bfloat16" and cc[0] < 8:
            print("  ВНИМАНИЕ: bfloat16 без аппаратной поддержки на этом GPU")

    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    got = next(model.parameters()).dtype
    if got != dtype:
        print(f"  from_pretrained вернул {got} — привожу к {dtype}")
        model = model.to(dtype)
    print(f"  dtype модели по факту: {next(model.parameters()).dtype}")
    processor = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])

    suite_disp = "long" if args.task_suite == "10" else args.task_suite
    modes = (["off", "on"] if args.ensemble == "both" else [args.ensemble])

    def offset_for(task_id):
        if args.pos_offset is not None:
            return args.pos_offset
        if not os.path.exists(args.offset_table):
            raise SystemExit(
                f"нет {args.offset_table}; постройте k4b0_offset_table.py или "
                f"задайте --pos-offset явно. Единый офсет — АБЛЯЦИЯ, а не "
                f"официальный протокол")
        tb = json.load(open(args.offset_table))
        by = tb.get("offsets_by_suite", {})
        if args.task_suite in by and task_id < len(by[args.task_suite]):
            return int(by[args.task_suite][task_id])
        for _, rec in tb.get("tasks", {}).items():
            if (rec.get("suite") == args.task_suite
                    and rec.get("task_id") == task_id):
                return int(rec["pos_offset"])
        raise SystemExit(f"в таблице нет офсета для {args.task_suite}/{task_id}")

    def rollout(envs, task_desc, horizon, ensemble, pos_off, k):
        """Один раунд из n_envs эпизодов с общими начальными состояниями."""
        n_envs = args.n_envs
        ens = ActionEnsembler() if ensemble else None
        ts = 0
        if ens is not None:
            ens.reset()
        obs = envs.reset(options=[{"init_state_id": i + k * n_envs}
                                  for i in range(n_envs)])
        reward = np.zeros(n_envs)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n_envs)
        done = np.zeros(n_envs, bool)
        for _ in range(args.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)

        calls, steps, ms = 0, 0, 0.0
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
            for i in range(n_envs):
                m = prompt_template(
                    state[i], None, task_desc,
                    mode=cfg.MODEL.vla_processor.kwargs.mode,
                    action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                    action_token_len=cfg.MODEL.action_processor.token_len)
                m[1]["content"] = m[1]["content"][1:]      # склеенные виды
                msgs.append(m)
            texts = processor.apply_chat_template(msgs, add_generation_prompt=True)
            batch = processor(text=texts,
                              images=[[image[i].numpy()] for i in range(n_envs)],
                              return_tensors="pt", padding=True,
                              padding_side="left",
                              action_processor_kwargs={"embodiment_ids": 0})
            batch = dict_apply(lambda x: x.to(dev, dtype), batch)

            # ВРЕМЯ ПОЛИТИКИ меряется вокруг генерации и декода вместе с
            # синхронизацией: без неё замерялось бы время постановки в очередь.
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                toks = model.generate(**batch, position_offset=pos_off,
                                      do_sample=False, initial_position_shift=1)
                act = processor.action_processor.decode(toks.tolist())[0]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ms += (time.perf_counter() - t0) * 1e3
            calls += 1

            action = np.copy(act)
            action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
            action[..., -1] = -action[..., -1]
            if ens is not None:
                ens.add_actions(action, ts)

            for t in range(horizon):
                if np.all(done) or steps >= args.max_steps:
                    break
                if ens is not None:
                    a_t = ens.get_action(ts)
                    ts += 1
                else:
                    a_t = action[:, t]
                obs, r_, done, _ = envs.step(a_t)
                reward = np.clip(reward + r_, 0, 1)
                steps += 1

        return [dict(success=bool(reward[i] >= 1.0), env_steps=steps,
                     policy_calls=calls, policy_ms=ms / n_envs,
                     init_state_id=i + k * n_envs, env_index=i)
                for i in range(n_envs)]

    res = {}
    t_start = time.time()
    for task_id in args.task_id:
        pos_off = offset_for(task_id)
        seed_everything(args.seed)
        envs, task_desc = get_envs(args.task_suite,
                                   {"task_id": task_id, "image_size": 224},
                                   args.n_envs)
        print(f"\n=== suite {suite_disp}, задача {task_id}, офсет {pos_off}")
        print(f"    «{task_desc}»")
        try:
            for ens_mode in modes:
                for H in args.horizons:
                    eps = []
                    for k in range(args.k_set):
                        eps += rollout(envs, task_desc, H, ens_mode == "on",
                                       pos_off, k)
                    s = summarize(eps)
                    key = f"{suite_disp}/{task_id}/ens_{ens_mode}/H{H}"
                    res[key] = dict(summary=s, episodes=eps, horizon=H,
                                    ensemble=(ens_mode == "on"),
                                    task_id=task_id, suite=suite_disp,
                                    pos_offset=pos_off,
                                    task_description=task_desc)
                    print(f"    ens={ens_mode:<3} H={H:>2}: успех "
                          f"{s['success_rate']:.2%}, вызовов на действие "
                          f"{s['calls_per_action']:.3f}, мс на действие "
                          f"{s['ms_per_action']:.1f}, шагов "
                          f"{s['mean_episode_steps']:.0f}, "
                          f"мс на вызов {s['ms_per_call']:.0f}", flush=True)
        finally:
            try:
                envs.close()
            except Exception:
                pass

    print("\n" + "=" * 74)
    print("СВОДКА: успех против частоты вызовов")
    print("=" * 74)
    print(f"  {'режим':<8}{'H':>4}{'успех':>9}{'выз/действие':>14}"
          f"{'мс/действие':>13}{'шагов':>8}")
    for ens_mode in modes:
        for H in args.horizons:
            ks = [k for k in res if f"ens_{ens_mode}/H{H}" in k]
            if not ks:
                continue
            allep = [e for k in ks for e in res[k]["episodes"]]
            s = summarize(allep)
            print(f"  {ens_mode:<8}{H:>4}{s['success_rate']:>8.1%}"
                  f"{s['calls_per_action']:>14.3f}{s['ms_per_action']:>13.1f}"
                  f"{s['mean_episode_steps']:>8.0f}")
    print(f"\n  ВНИМАНИЕ ПРО БАТЧИРОВАНИЕ: сред {args.n_envs} в одном вызове.\n"
          "  Один вызов политики обслуживает все среды сразу, поэтому «мс на\n"
          "  действие» здесь — АМОРТИЗИРОВАННАЯ величина, а не задержка\n"
          "  реакции одного робота. Отношения между горизонтами от этого не\n"
          "  страдают, но для заявлений о latency нужен отдельный прогон с\n"
          "  --n-envs 1.")
    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.\n"
          "  Главная кривая — успех против ВЫЗОВОВ НА ИСПОЛНЕННОЕ ДЕЙСТВИЕ.\n"
          "  Если успех при H=8 не хуже, чем при H=4, адаптивность в этом\n"
          "  диапазоне не нужна: достаточно поставить фиксированный H, и это\n"
          "  инженерное наблюдение, а не метод.\n"
          "  Если успех падает плавно — есть место для адаптивного выбора.\n"
          "  Если рушится сразу после H=4 везде — гетерогенности нет, ветку\n"
          "  закрывать до обучения.\n"
          "  Разница между ens=off и ens=on показывает, какая часть падения\n"
          "  объясняется потерей усреднения, а не устареванием наблюдения.")
    print(f"\n  всего времени: {(time.time() - t_start) / 60:.1f} мин")

    if args.out:
        import hashlib
        import subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             text=True).strip()
        except Exception:
            commit = "unknown"
        res["meta"] = dict(
            commit=commit, ckpt=args.ckpt, dtype=args.dtype, seed=args.seed,
            gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
            torch=torch.__version__, suite=suite_disp, task_ids=args.task_id,
            horizons=args.horizons, ensemble=args.ensemble,
            n_envs=args.n_envs, k_set=args.k_set,
            waiting_steps=args.waiting_steps, max_steps=args.max_steps,
            self_sha256=hashlib.sha256(
                open(__file__, "rb").read()).hexdigest()[:16])
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}")


if __name__ == "__main__":
    main()
