"""K-6h: сколько стоит в УСПЕХЕ отказ от двух тонких уровней RVQ.

ВОПРОС. K-6e закрыл линию уточнения по h: пять параметризаций, сходимость,
ничтожный разброс — ни одна не обошла «только грубый уровень» (0.0277 против
0.0234 у BAR). Но «только грубый» — это один проход башни вместо трёх, то есть
97.3 мс против 148.9 у BAR с честным кэшем: 1.53x, и 4.0x на действие при H=8.

Цена — +18% ОФФЛАЙНОВОЙ ошибки позы. А сколько это в успехе задачи, неизвестно:
на этой же линейке корреляция реконструкционного зазора с ценностью в замкнутом
цикле составила 0.08. То есть +18% может не стоить ничего.

ПОЧЕМУ НЕ НУЖНА БЫСТРАЯ РЕАЛИЗАЦИЯ. Для успеха важны ДЕЙСТВИЯ, а не время их
получения. «Только грубый» отличается от BAR лишь тем, какие коды идут в
декодер. Поэтому здесь по-прежнему вызывается обычный generate (все три блока),
а уровни 1-2 отбрасываются ПЕРЕД декодированием. Действия получаются ровно те
же, что дала бы однопроходная реализация, а латентность измерена отдельно
(k5a_pipeline_bench, k6c_refiner_cost). Так снимается целый класс ошибок,
который принесла бы правка генерации.

ПРОВЕРКА ТОЖДЕСТВА НА ХОДУ. Своя сборка латенты обязана при трёх уровнях дать
ТО ЖЕ, что официальный decode. Если нет — сравнение недействительно, и лучше
узнать это на первом вызове, чем после ночи раскаток. Проверяется автоматически.

ПРОТОКОЛ БЕРЁТСЯ ИЗ K-5b БЕЗ ИЗМЕНЕНИЙ: те же начальные состояния
(init_state_id = i + k*n_envs), тот же учёт успеха (reward = clip(reward + r, 0, 1),
while not all(done)), то же масштабирование действий и знак схвата. Одна
конфигурация на процесс: среды нельзя переиспользовать (reset восстанавливает
состояние не полностью), а пересоздать нельзя после загрузки модели (fork после
CUDA вешает процесс).

ПРАВИЛО ЧТЕНИЯ, записано до запуска:
  падение успеха в пределах доверительного интервала -> есть бесплатный метод:
      1.53x против кэшированной базы без единой обученной детали;
  падение до ~10 пунктов -> «только грубый» остаётся основой, но нужна
      компенсация, и иерархический токенизатор получает мотивацию;
  падение больше ~10 пунктов -> тонкая информация действительно нужна, и
      мотивация переходит к тому, чтобы сделать её предсказуемой за один проход.

Запуск (одна задача и один режим на процесс):
    python3 experiments/k6h_coarse_gate.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6h_coarse_gate.py --ckpt <ckpt> \\
        --task-suite 10 --task-id 0 --levels 1 --horizon 8 \\
        --n-envs 10 --k-set 2 --out data/k6h/10_t0_L1.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

N_POS, N_LEVEL = 16, 3


def summarize(eps):
    n = len(eps)
    steps = sum(e["env_steps"] for e in eps)
    calls = sum(e["policy_calls"] for e in eps)
    return dict(episodes=n,
                success_rate=sum(e["success"] for e in eps) / max(n, 1),
                env_steps=steps, policy_calls=calls,
                calls_per_action=calls / max(steps, 1))


def selftest():
    # Учёт успеха и нормировка вызовов — та же арифметика, что в K-5b.
    for H, want in ((4, 0.25), (8, 0.125)):
        eps = [dict(success=True, env_steps=40, policy_calls=40 // H)
               for _ in range(2)]
        assert abs(summarize(eps)["calls_per_action"] - want) < 1e-12
    a = [dict(success=True, env_steps=40, policy_calls=10)]
    b = [dict(success=True, env_steps=80, policy_calls=20)]
    assert summarize(a)["calls_per_action"] == summarize(b)["calls_per_action"], \
        "нормировка на исполненные действия не сработала"
    s = summarize([dict(success=True, env_steps=10, policy_calls=3),
                   dict(success=False, env_steps=10, policy_calls=3)])
    assert s["success_rate"] == 0.5

    # Отбрасывание уровней обязано быть ЛОКАЛЬНЫМ: раскладка поуровневая,
    # первые 16 токенов — уровень 0 (bar.py:1500-1503).
    toks = np.arange(N_POS * N_LEVEL)[None]
    K = toks.reshape(1, N_LEVEL, N_POS)
    assert (K[0, 0] == np.arange(0, 16)).all(), "первые 16 — уровень 0"
    assert (K[0, 2] == np.arange(32, 48)).all(), "последние 16 — уровень 2"
    print("самопроверка пройдена: учёт вызовов нормирован на исполненные "
          "действия, раскладка кодов поуровневая")


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
    ap.add_argument("--levels", type=int, default=1, choices=[1, 2, 3],
                    help="сколько уровней RVQ идёт в декодер; 3 = как BAR")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--k-set", type=int, default=2)
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
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
    sys.path.insert(0, root)          # только корень, см. K-5b
    import torch
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       get_envs, process_state, prompt_template, seed_everything)

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])   # кадр среды 224

    if args.pos_offset is not None:
        pos_off = args.pos_offset
    else:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; задайте --pos-offset")
        tb = json.load(open(args.offset_table))
        pos_off = int(tb["offsets_by_suite"][args.task_suite][args.task_id])

    # СРЕДЫ ДО МОДЕЛИ: fork после инициализации CUDA вешает процесс.
    seed_everything(args.seed)
    envs, task_desc = get_envs(args.task_suite,
                               {"task_id": args.task_id, "image_size": 224},
                               args.n_envs)
    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь в action_processor")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size)).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    print(f"=== suite {args.task_suite}, задача {args.task_id}, офсет {pos_off}")
    print(f"    «{task_desc}»   H={args.horizon}, уровней={args.levels}")

    identity_checked = [False]

    def decode_levels(toks_np):
        """Собрать действие из первых `levels` уровней. При levels=3 обязано
        совпасть с официальным decode — проверяется на первом вызове."""
        K = toks_np.reshape(-1, N_LEVEL, N_POS)
        with torch.no_grad():
            zq = E[0][torch.as_tensor(K[:, 0, :]).long().to(dev)]
            for j in range(1, args.levels):
                zq = zq + E[j][torch.as_tensor(K[:, j, :]).long().to(dev)]
            x, _ = codec._decode(zq, embodiment_ids=0)
            out = x[..., :7].float().cpu().numpy()
        if not identity_checked[0]:
            identity_checked[0] = True
            with torch.no_grad():
                z3 = sum(E[j][torch.as_tensor(K[:, j, :]).long().to(dev)]
                         for j in range(N_LEVEL))
                x3, _ = codec._decode(z3, embodiment_ids=0)
                mine = x3[..., :7].float().cpu().numpy()
            ref = np.asarray(proc.action_processor.decode(toks_np.tolist())[0],
                             np.float64)
            d = float(np.abs(mine - ref).max())
            print(f"    тождество сборки при трёх уровнях: max|Δ| = {d:.3e}")
            if d > 1e-3:
                raise SystemExit(
                    f"своя сборка латенты расходится с официальным decode на "
                    f"{d:.3e} — сравнение недействительно")
        return out

    def rollout(k):
        seed_everything(args.seed + 1000 * k)
        n = args.n_envs
        obs = envs.reset(options=[{"init_state_id": i + k * n} for i in range(n)])
        reward = np.zeros(n)
        done = np.zeros(n, bool)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n)
        for _ in range(args.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)
        calls = steps = 0
        while not np.all(done) and steps < args.max_steps:
            state = ((process_state(obs["state"]) - STATE_Q01)
                     / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0)
            i1 = tf(torch.tensor(
                obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2))
            i2 = tf(torch.tensor(
                obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
            ).permute(0, 3, 1, 2))
            image = torch.cat([i1, i2], dim=-1)
            msgs = []
            for i in range(n):
                m = prompt_template(
                    state[i], None, task_desc,
                    mode=cfg.MODEL.vla_processor.kwargs.mode,
                    action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                    action_token_len=cfg.MODEL.action_processor.token_len)
                m[1]["content"] = m[1]["content"][1:]
                msgs.append(m)
            texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
            batch = proc(text=texts,
                         images=[[image[i].numpy()] for i in range(n)],
                         return_tensors="pt", padding=True, padding_side="left",
                         action_processor_kwargs={"embodiment_ids": 0})
            batch = dict_apply(lambda x: x.to(dev, dtype), batch)
            with torch.no_grad():
                toks = model.generate(**batch, position_offset=pos_off,
                                      do_sample=False, initial_position_shift=1)
            act = decode_levels(toks.cpu().numpy())
            calls += 1
            # ТО ЖЕ МАСШТАБИРОВАНИЕ И ЗНАК СХВАТА, что в eval_libero.
            action = np.copy(act)
            action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
            action[..., -1] = -action[..., -1]
            for t in range(args.horizon):
                if np.all(done) or steps >= args.max_steps:
                    break
                obs, r_, done, _ = envs.step(action[:, t])
                reward = np.clip(reward + r_, 0, 1)
                steps += 1
        return [dict(success=bool(reward[i] >= 1.0), env_steps=steps,
                     policy_calls=calls, init_state_id=i + k * args.n_envs,
                     env_index=i) for i in range(args.n_envs)]

    t0 = time.time()
    eps = []
    try:
        for k in range(args.k_set):
            r = rollout(k)
            eps += r
            print(f"  раунд {k}: успех {sum(e['success'] for e in r)}/"
                  f"{args.n_envs}, шагов {r[0]['env_steps']}", flush=True)
    finally:
        try:
            envs.close()
        except Exception:
            pass

    s = summarize(eps)
    print(f"\n  уровней {args.levels}, H={args.horizon}: успех "
          f"{s['success_rate']:.1%} ({sum(e['success'] for e in eps)}/{len(eps)}), "
          f"вызовов на действие {s['calls_per_action']:.3f}")
    print(f"  время: {(time.time() - t0) / 60:.1f} мин")
    print("\n  ЧИТАТЬ ТАК: сравнивать с тем же прогоном при --levels 3 на ТЕХ ЖЕ")
    print("  начальных состояниях. Латентность здесь НЕ меряется — она взята")
    print("  из k5a/k6c: один проход 97.3 мс против 148.9 у BAR с кэшем.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(dict(summary=s, episodes=eps, levels=args.levels,
                       horizon=args.horizon, task_id=args.task_id,
                       suite=args.task_suite, pos_offset=pos_off,
                       task_description=task_desc, seed=args.seed),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}")


if __name__ == "__main__":
    main()
