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

ОДИН РАУНД НА ПРОЦЕСС — ЖЁСТКО. Прежняя версия гоняла `--k-set 2`, то есть
переиспользовала среды между раундами. Это ровно то, что K-5b запрещает и
ИЗМЕРИЛ: ячейка ens=on,H=20 дала 10 успехов будучи восьмой в процессе и 9 будучи
первой, при полностью детерминированной среде (k5b_fixed_horizon_eval.py:306).
`reset` восстанавливает состояние не полностью, поэтому вторая десятка зависела
бы от того, какие траектории шли перед ней, а они у режимов levels=1 и levels=3
РАЗНЫЕ. Парность бы рассыпалась. И размер эффекта — один эпизод из десяти —
ровно того же порядка, что разница, которую мы пытаемся поймать.

Поэтому раунд один, а блок начальных состояний задаётся `--init-start`:
init_state_id = init_start + i. Для 20 эпизодов запускается два процесса с
--init-start 0 и 10, а не один процесс с двумя раундами.

СВЕРКА ПАРНОСТИ. После холостых шагов пишется хеш начального наблюдения. Пара
(levels=1, levels=3) на одном init_state_id обязана иметь одинаковый хеш; если
нет, эпизоды стартовали из разных состояний и сравнивать их нельзя. Проверяет
агрегатор, а не этот скрипт.

УСРЕДНЕНИЕ ДЕЙСТВИЙ — ОТДЕЛЬНАЯ ОСЬ, НЕ УМОЛЧАНИЕ. ActionEnsembler копит планы
на каждый абсолютный момент и усредняет их; официальный eval_libero.py включает
его по умолчанию, и K-5b мерил ОБА режима. Прежняя версия молча работала только
без усреднения и при этом заявляла «протокол K-5b без изменений» — неверно.
Здесь режим задаётся явно, и мерить надо оба: ens=on отвечает на вопрос «сколько
стоит отказ от тонких уровней в официальном протоколе», ens=off изолирует
механизм. Это не праздная симметрия: усреднение по перекрывающимся планам гасит
шум, а грубое квантование — это в первую очередь шум, так что coarse-only может
терять при ens=off ЗАМЕТНО БОЛЬШЕ, чем при ens=on.

Остальное из K-5b дословно: учёт успеха (reward = clip(reward + r, 0, 1),
while not all(done)), масштабирование действий, знак схвата, сброс сида перед
раундом, среды создаются ДО модели (fork после CUDA вешает процесс).

ПРАВИЛО ЧТЕНИЯ, записано до запуска. Читается по ВЫХОДУ АГРЕГАТОРА
(k6h_summarize.py), не по среднему успеху одного процесса:
  односторонняя нижняя граница парной разности выше -10 пунктов -> тонкая
      информация не нужна на этом чекпойнте, есть 1.53x без обученных деталей;
  граница ниже -10 пунктов -> тонкая информация нужна, и мотивация переходит к
      тому, чтобы сделать её предсказуемой за один проход.
Утверждать «падение в пределах доверительного интервала = эквивалентность»
нельзя: пересечение с нулём это отсутствие доказательства разницы, а не
доказательство её отсутствия. Отсюда односторонняя граница, а не двусторонний
интервал. Мощность при 10 задачах x 20 эпизодов и доле дискордантных пар 0.15:
обнаруживается падение около 7 пунктов, а с поправкой на кластеризацию по
задачам (DEFF~2) около 9-10. Заявить «не хуже 5 пунктов» на этой выборке
НЕЛЬЗЯ, для этого нужно ~700 эпизодов.

Запуск (одна задача, один режим уровней, один режим усреднения, один блок
начальных состояний на процесс):
    python3 experiments/k6h_coarse_gate.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6h_coarse_gate.py --ckpt <ckpt> \\
        --task-suite 10 --task-id 0 --levels 1 --horizon 8 \\
        --n-envs 10 --init-start 0 --ensemble on \\
        --out data/k6h/10_t0_L1_ensON_s0.json
"""

import argparse
import hashlib
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

    # Блоки начальных состояний обязаны быть НЕПЕРЕСЕКАЮЩИМИСЯ и покрывать
    # диапазон без дыр: два процесса с --init-start 0 и 10 при n_envs=10 дают
    # ровно 0..19. Смещение на единицу здесь стоило бы всей парности.
    ids = [s + i for s in (0, 10) for i in range(10)]
    assert ids == list(range(20)) and len(set(ids)) == 20, ids
    print("самопроверка пройдена: учёт вызовов нормирован на исполненные "
          "действия, раскладка кодов поуровневая, блоки начальных состояний "
          "не пересекаются")


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
    ap.add_argument("--init-start", type=int, default=0,
                    help="init_state_id = init_start + i; блок состояний. "
                         "Переиспользовать среды для второго блока НЕЛЬЗЯ — "
                         "запускайте второй процесс")
    ap.add_argument("--ensemble", choices=["on", "off"], default=None,
                    help="усреднение перекрывающихся планов; on = официальный "
                         "протокол eval_libero, off = изоляция механизма. "
                         "Умолчания нет намеренно: молчаливый выбор здесь уже "
                         "приводил к ложному заявлению о воспроизведении K-5b")
    ap.add_argument("--k-set", type=int, default=1,
                    help=argparse.SUPPRESS)
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
    if args.ensemble is None:
        raise SystemExit(
            "--ensemble on|off обязателен. Умолчания нет намеренно: "
            "официальный\neval_libero.py усредняет по умолчанию, прежняя "
            "версия этого скрипта\nмолча не усредняла и при этом заявляла "
            "«протокол K-5b без изменений».")
    if args.k_set != 1:
        raise SystemExit(
            "--k-set больше не поддерживается: переиспользование сред между\n"
            "раундами ломает парность, потому что reset восстанавливает\n"
            "состояние не полностью и результат зависит от того, какие\n"
            "траектории шли раньше — а у levels=1 и levels=3 они разные.\n"
            "Второй блок эпизодов — отдельный процесс с --init-start.")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)          # только корень, см. K-5b
    import torch
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       ActionEnsembler, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, get_envs, process_state,
                       prompt_template, seed_everything)

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

    def rollout():
        # СИД СБРАСЫВАЕТСЯ ПЕРЕД РАУНДОМ, как в K-5b: иначе расход глобального
        # ГСЧ различался бы между режимами и начальные состояния разошлись бы.
        seed_everything(args.seed + 1000 * args.init_start)
        n = args.n_envs
        ens = ActionEnsembler() if args.ensemble == "on" else None
        ts = 0
        if ens is not None:
            ens.reset()
        obs = envs.reset(options=[{"init_state_id": args.init_start + i}
                                  for i in range(n)])
        reward = np.zeros(n)
        done = np.zeros(n, bool)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n)
        for _ in range(args.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)
        # ХЕШ СТАРТА для сверки парности агрегатором: два режима на одном
        # init_state_id обязаны стартовать из одного состояния.
        init_hash = [hashlib.sha1(np.ascontiguousarray(
            np.concatenate([obs["state"][i].ravel(),
                            obs["agentview_image"][i].ravel() / 255.0])
        ).astype(np.float32).tobytes()).hexdigest()[:16] for i in range(n)]
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
            if ens is not None:
                ens.add_actions(action, ts)
            for t in range(args.horizon):
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
        # env_steps — длина РАУНДА, а не эпизода: раунд идёт, пока не
        # завершатся все среды. Для сравнения режимов этого достаточно.
        return [dict(success=bool(reward[i] >= 1.0), env_steps=steps,
                     policy_calls=calls, init_state_id=args.init_start + i,
                     env_index=i, init_hash=init_hash[i])
                for i in range(args.n_envs)]

    t0 = time.time()
    try:
        eps = rollout()
        print(f"  успех {sum(e['success'] for e in eps)}/{args.n_envs}, "
              f"шагов {eps[0]['env_steps']}", flush=True)
    finally:
        try:
            envs.close()
        except Exception:
            pass

    s = summarize(eps)
    print(f"\n  уровней {args.levels}, H={args.horizon}, ens={args.ensemble}: "
          f"успех {s['success_rate']:.1%} "
          f"({sum(e['success'] for e in eps)}/{len(eps)}), "
          f"вызовов на действие {s['calls_per_action']:.3f}")
    print(f"  время: {(time.time() - t0) / 60:.1f} мин")
    print("\n  ЧИТАТЬ ТАК: не по этому числу. Оно осмысленно только в паре с")
    print("  --levels 3 при тех же init_state_id, ensemble и seed, и только")
    print("  через k6h_summarize.py — одна ячейка из десяти эпизодов ничего не")
    print("  решает. Латентность здесь НЕ меряется: она взята из k5a/k6c —")
    print("  один проход 97.3 мс против 148.9 у BAR с кэшем.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        # ПОЛНЫЕ ПАРАМЕТРЫ И ХЕШ СКРИПТА: иначе через неделю не отличить, каким
        # кодом получена ячейка, а ячейки живут в разных файлах и процессах.
        sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
        json.dump(dict(summary=s, episodes=eps, levels=args.levels,
                       horizon=args.horizon, task_id=args.task_id,
                       suite=args.task_suite, pos_offset=pos_off,
                       ensemble=args.ensemble, init_start=args.init_start,
                       n_envs=args.n_envs, task_description=task_desc,
                       seed=args.seed, ckpt=args.ckpt, script_sha1=sha,
                       argv=vars(args)),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}  (sha {sha})")


if __name__ == "__main__":
    main()
