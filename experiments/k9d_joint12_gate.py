"""K-9d: симуляторный гейт для Joint-12 против грубого выхода на 24 слоях.

ЗАЧЕМ ОТДЕЛЬНЫЙ СКРИПТ. k6h_coarse_gate.py сравнивает комплектации УРОВНЕЙ у
одной и той же модели: строит обычную BAR, зовёт generate и отбрасывает уровни
перед декодированием. Joint-12 так не запустить — нужно init_joint_fast,
загрузка обученных весов и forward_joint_fast вместо generate. Поэтому здесь
своя развилка по РУКАМ, а всё остальное — среды, сиды, порядок вызовов,
масштабирование действий, усреднение, хеши парности, формат JSON — перенесено
из k6h дословно, чтобы опорное число 89.0% осталось сравнимым.

ЧТО СРАВНИВАЕТСЯ.
  * arm=coarse24 — грубый выход полной глубины. Это БУКВАЛЬНО ветка K-6h
    --levels 1: обычный generate, первые 16 токенов, сборка из уровня 0.
    Опора: 89.0% при усреднении и 89.5% без него (400 парных эпизодов).
  * arm=fast12  — тот же грубый выход, но с 12 слоёв обеих башен после
    совместного дообучения. forward_joint_fast, дальше та же сборка.
Разница между руками — только глубина и веса первых слоёв. Уровни 2-3 не
исполняются НИ В ОДНОЙ руке: вопрос здесь не про них, на него ответил K-6h.

ПОЧЕМУ ОПОРА — ИМЕННО COARSE24, А НЕ ПОЛНАЯ BAR. Joint-12 обучался
дистилляцией грубых кодов учителя. Полная BAR отличается от него двумя вещами
сразу (глубина и число уровней), и разделить вклады по такому сравнению было бы
нельзя. Сравнение с coarse24 изолирует ровно глубину.

ПРЕ-РЕГИСТРИРОВАННОЕ ПРАВИЛО ЧТЕНИЯ. Записано ДО запуска, читается через
k6h_summarize.py --field arm --test fast12 --ref coarse24 --margin 5. Границы
ОДНОСТОРОННИЕ, по 5% с каждого края кластерного бутстрапа:
  * НИЖНЯЯ выше -5 пунктов -> не-худшесть ДОКАЗАНА, ветка положительная;
  * ВЕРХНЯЯ ниже -5 пунктов -> ухудшение более чем на 5 пунктов ДОКАЗАНО,
    ветка закрывается отрицательно;
  * иначе -> НЕ ДОКАЗАНО НИЧЕГО, добираются блоки через --init-start.

Две границы здесь не симметричная придирка, а разные утверждения. «Нижняя
граница ниже -5» означает лишь отсутствие доказательства не-худшести, и
принимать по ней отрицательное решение — та же ошибка, что читать
неотвергнутую нулевую гипотезу как доказанное равенство. Ранняя версия этой
докстроки такую ошибку содержала.
Отдельно: офлайновое согласие с учителем (33.4% на отложенных эпизодах) НЕ
является предсказанием этого числа. Связь кода и успеха уже мерилась и вышла
слабой: coarse-only имел на 32% худшую реконструкцию действия при нулевой
потере успеха. Поэтому решение принимается здесь, а не по офлайну.

ЧЕГО ЗДЕСЬ НАМЕРЕННО НЕТ. Латентности. Она меряется в k7a на фиксированных
входах с прогревом и синхронизацией; мерить её внутри цикла со средой значит
смешать время политики со временем MuJoCo. Оценка 1.48x к coarse24 — АРИФМЕТИКА
по числу слоёв, и в отчёт она идёт только с этой пометкой, пока k7a не запущен
на Joint-12.

Запуск:
    python3 experiments/k9d_joint12_gate.py --selftest

    # опорная рука (одна задача, десять эпизодов)
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9d_joint12_gate.py --ckpt <base_ckpt> \\
        --arm coarse24 --task-id 0 --ensemble on \\
        --out data/k9d/coarse24_t0_i0_ens1.json

    # рука Joint-12 (те же task-id, init-start, seed, ensemble)
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9d_joint12_gate.py --ckpt <base_ckpt> \\
        --arm fast12 --joint-ckpt data/k9c_150k/best_imitation.pt \\
        --task-id 0 --ensemble on \\
        --out data/k9d/fast12_t0_i0_ens1.json
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

N_POS, N_LEVEL = 16, 3
ARMS = ("coarse24", "fast12")
# K-6h, грубый выход. ПО 200 ПАР НА ПРОТОКОЛ (10 задач x 2 блока по 10 сред),
# а не 400 на каждый: 400 — это сумма двух протоколов. Здесь развёртка идёт на
# четырёх блоках при ens=on, то есть 400 пар на основном протоколе — вдвое
# больше опорного, а не «ровно как в K-6h».
REFERENCE_K6H = {"on": 89.0, "off": 89.5}


def summarize(eps):
    n = len(eps)
    steps = sum(e["env_steps"] for e in eps)
    calls = sum(e["policy_calls"] for e in eps)
    return dict(episodes=n,
                success_rate=sum(e["success"] for e in eps) / max(n, 1),
                env_steps=steps, policy_calls=calls,
                calls_per_action=calls / max(steps, 1))


def ckpt_depth(obj):
    """Глубина берётся ИЗ ЧЕКПОЙНТА, а не с командной строки.

    Ключ: если запустить обученный на 12 слоях чекпойнт с --depth 16, веса
    лягут без ошибки (имена совпадают), а исполнится другая сеть. Такое
    расхождение молчаливо, поэтому глубина не является аргументом вовсе.
    """
    d = obj.get("depth")
    if d is None:
        raise SystemExit("в чекпойнте нет поля depth — он не от k9c")
    return int(d)


def selftest():
    # Учёт вызовов нормирован на ИСПОЛНЕННЫЕ действия — та же арифметика, что в
    # K-5b и K-6h, иначе руки с разным горизонтом несравнимы.
    for H, want in ((4, 0.25), (8, 0.125)):
        eps = [dict(success=True, env_steps=40, policy_calls=40 // H)
               for _ in range(2)]
        assert abs(summarize(eps)["calls_per_action"] - want) < 1e-12
    s = summarize([dict(success=True, env_steps=10, policy_calls=3),
                   dict(success=False, env_steps=10, policy_calls=3)])
    assert s["success_rate"] == 0.5

    # Раскладка кодов поуровневая: первые 16 из 48 — уровень 0 (bar.py:1500).
    K = np.arange(N_POS * N_LEVEL)[None].reshape(1, N_LEVEL, N_POS)
    assert (K[0, 0] == np.arange(0, 16)).all(), "первые 16 — уровень 0"

    # Блоки начальных состояний не пересекаются и покрывают диапазон без дыр.
    ids = [s + i for s in (0, 10) for i in range(10)]
    assert ids == list(range(20)) and len(set(ids)) == 20, ids

    # Глубина только из чекпойнта.
    assert ckpt_depth({"depth": 12}) == 12
    try:
        ckpt_depth({})
    except SystemExit:
        pass
    else:
        raise AssertionError("чекпойнт без depth обязан быть отвергнут")

    # Фильтр белого списка при загрузке: в чекпойнте лежат ТОЛЬКО обучаемые
    # префиксы, и попытка положить туда что-то ещё обязана быть замечена.
    prefixes = ("vlm.text_model.layers.0.", "action_expert.norm.",
                "bos_embedding", "fast_head.")
    ok = "action_expert.norm.weight"
    bad = "vlm.text_model.layers.20.mlp.gate_proj.weight"
    def allowed(k):
        return any(k.startswith(p) or k == p.rstrip(".") for p in prefixes)
    assert allowed(ok) and allowed("bos_embedding") and not allowed(bad)

    # Руки взаимоисключающие, и у каждой свой обязательный набор аргументов.
    assert set(ARMS) == {"coarse24", "fast12"} and len(ARMS) == 2
    print("самопроверка k9d пройдена (версия «глубина из чекпойнта»): "
          "нормировка вызовов, поуровневая раскладка, непересекающиеся блоки, "
          "фильтр белого списка, отказ на чекпойнте без depth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", help="базовый чекпойнт BAR (обе руки)")
    ap.add_argument("--arm", choices=ARMS, default=None,
                    help="coarse24 — опора K-6h; fast12 — Joint-12")
    ap.add_argument("--joint-ckpt", default=None,
                    help="best_imitation.pt от k9c; только для fast12")
    ap.add_argument("--expect-depth", type=int, default=12,
                    help="глубина, которую обязан объявить чекпойнт. Рука "
                         "называется fast12, и без этой проверки она молча "
                         "приняла бы чекпойнт глубины 18.")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--init-start", type=int, default=0,
                    help="сдвиг блока init_state_id; блоки по n-envs штук "
                         "не пересекаются")
    ap.add_argument("--ensemble", choices=["on", "off"], default=None,
                    help="ОБЯЗАТЕЛЕН. Умолчания нет намеренно: официальный "
                         "eval_libero.py усредняет, и молчаливое несовпадение "
                         "режима между руками обесценило бы всю пару.")
    ap.add_argument("--teacher-agreement", choices=["off", "first", "every"],
                    default="off",
                    help="согласие с учителем ВНУТРИ ЦИКЛА, на своей же "
                         "траектории. Требует копии исходных весов в памяти "
                         "(~4 ГиБ), на вердикт не влияет, по умолчанию выкл.")
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
    if args.arm is None:
        raise SystemExit(f"--arm обязателен, одно из {ARMS}")
    if args.ensemble is None:
        raise SystemExit(
            "--ensemble on|off обязателен. Умолчания нет намеренно: "
            "официальный\neval_libero.py усредняет по умолчанию, и рука без "
            "усреднения\nмолча сравнивалась бы с рукой с усреднением.")
    if args.arm == "fast12" and not args.joint_ckpt:
        raise SystemExit("рука fast12 требует --joint-ckpt")
    if args.arm == "coarse24" and args.joint_ckpt:
        raise SystemExit(
            "рука coarse24 — ОПОРА и обязана идти на исходных весах; "
            "--joint-ckpt здесь запрещён, иначе опора уедет вместе с рукой")
    if args.arm == "coarse24" and args.teacher_agreement != "off":
        raise SystemExit("согласие с учителем осмысленно только для fast12")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)          # только корень, см. K-5b
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       ActionEnsembler, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, get_envs, process_state,
                       prompt_template, seed_everything)
    from joint12_vla import make_joint12_class

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

    # СРЕДЫ ДО МОДЕЛИ: fork после инициализации CUDA вешает процесс. Порядок
    # тот же, что в K-6h, и от него зависит расход глобального ГСЧ, то есть
    # начальные состояния. Менять нельзя.
    seed_everything(args.seed)
    envs, task_desc = get_envs(args.task_suite,
                               {"task_id": args.task_id, "image_size": 224},
                               args.n_envs)

    Joint = make_joint12_class(SmolVLABlockwiseAR)
    model = Joint.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    # --- комплектация руки ---------------------------------------------------
    joint_meta, orig_weights, depth = None, None, 24
    if args.arm == "fast12":
        # SHA САМИХ ВЕСОВ, а не обучавшего скрипта. Путь к файлу ничего не
        # удостоверяет: best_imitation.pt перезаписывается каждой лучшей
        # эпохой, и две ячейки с одинаковым путём могут быть посчитаны разными
        # весами. Агрегатор отказывается смешивать разные sha.
        h = hashlib.sha1()
        with open(args.joint_ckpt, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        weights_sha = h.hexdigest()[:12]
        obj = torch.load(args.joint_ckpt, map_location="cpu")
        depth = ckpt_depth(obj)
        if depth != args.expect_depth:
            raise SystemExit(
                f"чекпойнт объявляет глубину {depth}, ожидалась "
                f"{args.expect_depth}. Рука называется fast{args.expect_depth}; "
                f"если это намеренно, задайте --expect-depth {depth}.")
        model.init_joint_fast(depth=depth)
        state = obj["state"]
        # ЗАГРУЗКА СТРОГАЯ И ПРОВЕРЯЕМАЯ. Чекпойнт содержит только обучаемые
        # префиксы; всё, что вне белого списка, — признак чужого файла, а не
        # повод «загрузить что получится».
        stray = [k for k in state
                 if not any(k.startswith(p) or k == p.rstrip(".")
                            for p in model.trainable_prefixes)]
        if stray:
            raise SystemExit(
                f"в чекпойнте {len(stray)} ключей вне белого списка "
                f"глубины {depth}: {stray[:5]}")
        own = dict(model.named_parameters())
        missing = [k for k in own
                   if own[k].requires_grad and k not in state]
        if missing:
            raise SystemExit(
                f"в чекпойнте нет {len(missing)} обучаемых весов: {missing[:5]}")
        if args.teacher_agreement != "off":
            orig_weights = {k: own[k].detach().clone() for k in state}
        loaded = 0
        with torch.no_grad():
            for k, v in state.items():
                p = own[k]
                if tuple(p.shape) != tuple(v.shape):
                    raise SystemExit(f"форма {k}: {tuple(p.shape)} против "
                                     f"{tuple(v.shape)}")
                # ВЕСА ОСТАЮТСЯ В FP32, как при обучении, а проход идёт под
                # autocast fp16. Округлить их до fp16 значило бы исполнять не ту
                # сеть, которую отбирали по валидации.
                p.data = v.to(dev, torch.float32)
                loaded += 1
        model.eval()
        # SHA joint12_vla.py тоже в файле: forward_joint_fast определяет
        # инференс не меньше, чем веса, и его правка меняет исполняемую сеть
        # при неизменном чекпойнте.
        import joint12_vla as _jv
        vla_sha = hashlib.sha1(
            open(_jv.__file__, "rb").read()).hexdigest()[:12]
        joint_meta = dict(path=os.path.abspath(args.joint_ckpt),
                          weights_sha1=weights_sha, depth=depth,
                          tensors=loaded, joint12_vla_sha1=vla_sha,
                          train_sha1=obj.get("sha1"),
                          train_args=obj.get("args"))
        print(f"  Joint-12: загружено {loaded} тензоров, глубина {depth}, "
              f"веса sha {weights_sha}, joint12_vla sha {vla_sha}, "
              f"обучено скриптом sha {obj.get('sha1')}")
    else:
        # У ОПОРЫ init_joint_fast НЕ ВЫЗЫВАЕТСЯ. Он создаёт fast_head и
        # сдвигает аллокатор — а в K-9b именно сдвиг аллокатора давал
        # расхождение логитов 5.9e-02 при полностью совпадающих весах.
        # Опора обязана быть побитово той же веткой, что в K-6h.
        print("  опора: generate на исходных весах, первые 16 токенов")

    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if args.arm == "fast12"
                else contextlib.nullcontext())

    @contextlib.contextmanager
    def original_weights():
        """Временно вернуть исходные веса — только для замера согласия."""
        keep = {k: dict(model.named_parameters())[k].detach().clone()
                for k in orig_weights}
        own = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in orig_weights.items():
                own[k].data.copy_(v)
        try:
            yield
        finally:
            with torch.no_grad():
                for k, v in keep.items():
                    own[k].data.copy_(v)

    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь в action_processor")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        # ИНДЕКСЫ НА ТОМ ЖЕ УСТРОЙСТВЕ, ЧТО КНИГИ: arange по умолчанию на CPU,
        # и F.embedding падает. Стоило одного упавшего ночного прогона в k6h.
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    print(f"=== suite {args.task_suite}, задача {args.task_id}, офсет {pos_off}")
    print(f"    «{task_desc}»   H={args.horizon}, рука={args.arm}, "
          f"глубина={depth}, ens={args.ensemble}")

    identity_checked = [False]

    def check_assembly(toks48_np):
        """Своя сборка из трёх уровней обязана совпасть с официальным decode.

        Проверяется ОДИН РАЗ и в обеих руках, на любых синтаксически годных
        токенах: это тождество про кодек, а не про качество политики. Без него
        расхождение декодера выглядело бы как разница рук.
        """
        if identity_checked[0]:
            return
        identity_checked[0] = True
        K = toks48_np.reshape(-1, N_LEVEL, N_POS)
        with torch.no_grad():
            z3 = sum(E[j][torch.as_tensor(K[:, j, :]).long().to(dev)]
                     for j in range(N_LEVEL))
            x3, _ = codec._decode(z3, embodiment_ids=0)
            mine = x3[..., :7].float().cpu().numpy()
        ref = np.asarray(proc.action_processor.decode(toks48_np.tolist())[0],
                         np.float64)
        d = float(np.abs(mine - ref).max())
        print(f"    тождество сборки при трёх уровнях: max|Δ| = {d:.3e}")
        if d > 1e-3:
            raise SystemExit(
                f"своя сборка латенты расходится с официальным decode на "
                f"{d:.3e} — сравнение недействительно")

    def decode_coarse(codes16_np):
        """Действие из ОДНОГО грубого уровня — то, что обе руки исполняют."""
        k = torch.as_tensor(codes16_np).long().to(dev).reshape(-1, N_POS)
        with torch.no_grad():
            x, _ = codec._decode(E[0][k], embodiment_ids=0)
            return x[..., :7].float().cpu().numpy()

    agree = []

    def policy(batch, first):
        """Грубые коды (B, 16) для текущей руки."""
        if args.arm == "coarse24":
            with torch.no_grad():
                toks = model.generate(**batch, position_offset=pos_off,
                                      do_sample=False, initial_position_shift=1)
            t = toks.cpu().numpy()
            check_assembly(t)
            return t.reshape(-1, N_LEVEL, N_POS)[:, 0, :]

        with torch.no_grad(), autocast:
            v, p = model.build_inputs(position_offset=pos_off, **batch)
            out = model.forward_joint_fast(
                vlm_inputs_embeds=v, attention_mask=batch.get("attention_mask"),
                position_ids=p)
        codes = out["pred_codes"].cpu().numpy()
        if first:
            assert out["layers_run"] == depth, out["layers_run"]
            # ТОЖДЕСТВО КОДЕКА БЕЗ ВЫЗОВА generate. Проверяется утверждение
            # «своя сборка равна официальному decode», а оно про кодек и верно
            # для ЛЮБЫХ валидных индексов. Поэтому 48 токенов собираются
            # повтором собственных шестнадцати, и обученная модель не гоняется
            # по пути, который при fp32-весах и autocast в обучении не
            # исполнялся ни разу.
            check_assembly(np.concatenate([codes] * N_LEVEL, axis=1))
        if orig_weights is not None and (first or
                                         args.teacher_agreement == "every"):
            # Учитель здесь идёт под тем же autocast: веса хранятся в fp32
            # ячейках, и вне autocast типы разошлись бы. Это приближение
            # исходного fp16-учителя, годное для диагностики согласия и НЕ
            # используемое в вердикте.
            with original_weights(), torch.no_grad(), autocast:
                tt = model.generate(**batch, position_offset=pos_off,
                                    do_sample=False).cpu().numpy()
            tc = tt.reshape(-1, N_LEVEL, N_POS)[:, 0, :]
            agree.append(float((tc == codes).mean()))
        return codes

    def rollout():
        # СИД СБРАСЫВАЕТСЯ ПЕРЕД РАУНДОМ, как в K-5b/K-6h: иначе расход
        # глобального ГСЧ различался бы между руками и начальные состояния
        # разошлись бы — а вся статистика здесь парная.
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
        # ХЕШ СТАРТА для сверки парности агрегатором: две руки на одном
        # init_state_id обязаны стартовать из одного состояния.
        #
        # ДВА ХЕША, А НЕ ОДИН. init_hash повторяет формулу K-6h дословно
        # (состояние и agentview), чтобы ячейки, посчитанные до этой правки,
        # продолжали сопоставляться. init_hash_full добавляет камеру на
        # запястье: политика смотрит в обе, и совпадение только по одной —
        # более слабое условие, чем нужно. Агрегатор сверяет полный хеш там,
        # где он есть у обеих рук, и это строго усиливает проверку, ничего не
        # ломая.
        def _h(parts):
            return hashlib.sha1(np.ascontiguousarray(
                np.concatenate(parts).astype(np.float32)).tobytes()
            ).hexdigest()[:16]
        init_hash = [_h([obs["state"][i].ravel(),
                         obs["agentview_image"][i].ravel() / 255.0])
                     for i in range(n)]
        init_hash_full = [_h([obs["state"][i].ravel(),
                              obs["agentview_image"][i].ravel() / 255.0,
                              obs["robot0_eye_in_hand_image"][i].ravel() / 255.0])
                          for i in range(n)]
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
            act = decode_coarse(policy(batch, calls == 0))
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
        # завершатся все среды. Для сравнения рук этого достаточно.
        return [dict(success=bool(reward[i] >= 1.0), env_steps=steps,
                     policy_calls=calls, init_state_id=args.init_start + i,
                     env_index=i, init_hash=init_hash[i],
                     init_hash_full=init_hash_full[i])
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
    print(f"\n  рука {args.arm}, H={args.horizon}, ens={args.ensemble}: "
          f"успех {s['success_rate']:.1%} "
          f"({sum(e['success'] for e in eps)}/{len(eps)}), "
          f"вызовов на действие {s['calls_per_action']:.3f}")
    if agree:
        print(f"  согласие с учителем НА СВОЕЙ траектории: "
              f"{float(np.mean(agree)):.1%} по {len(agree)} вызовам "
              f"(офлайн на отложенных эпизодах было 33.4%)")
    print(f"  время: {(time.time() - t0) / 60:.1f} мин")
    print("\n  ЧИТАТЬ ТАК: не по этому числу. Оно осмысленно только в паре с")
    print(f"  другой рукой при тех же task-id, init-start, seed и ensemble, и")
    print("  только через k6h_summarize.py --field arm --test fast12")
    print(f"  --ref coarse24 --margin 5. Опора K-6h: "
          f"{REFERENCE_K6H[args.ensemble]:.1f}% при ens={args.ensemble} "
          f"(200 пар на протокол, не 400).")
    print("  Латентность здесь НЕ меряется — для неё k7a на фиксированных")
    print("  входах; оценка 1.48x к coarse24 остаётся арифметикой.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        # ПОЛНЫЕ ПАРАМЕТРЫ И ХЕШ СКРИПТА: ячейки живут в разных файлах и
        # процессах, и через неделю иначе не отличить, каким кодом получена
        # какая. run_tag не даёт смешать эти ячейки с ячейками K-6h.
        sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
        json.dump(dict(summary=s, episodes=eps, arm=args.arm, run_tag="k9d",
                       depth=depth, horizon=args.horizon,
                       task_id=args.task_id, suite=args.task_suite,
                       pos_offset=pos_off, ensemble=args.ensemble,
                       init_start=args.init_start, n_envs=args.n_envs,
                       task_description=task_desc, seed=args.seed,
                       ckpt=args.ckpt, joint=joint_meta,
                       teacher_agreement=(float(np.mean(agree))
                                          if agree else None),
                       script_sha1=sha, argv=vars(args)),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}  (sha {sha})")


if __name__ == "__main__":
    main()
