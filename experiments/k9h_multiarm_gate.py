"""K-9h: многорукий симуляторный гейт с прямыми парами и контролем шума.

ЗАЧЕМ ЕЩЁ ОДИН ГЕЙТ. K-9d сравнивал ровно две руки и записывал их в поле
`arm`. Этого больше не хватает по двум причинам.

ПЕРВАЯ — НЕЛЬЗЯ ПЕРЕНОСИТЬ НЕ-ХУДШЕСТЬ ПО ЦЕПОЧКЕ. K-6h дал coarse24 не хуже
полной BAR с границей -3.0, K-9d дал fast12 не хуже coarse24 с границей -3.5.
Отсюда НЕ следует, что fast12 не хуже полной BAR: границы не складываются, две
пре-регистрированные пятёрки в сумме дают десять. Нужна прямая пара, и для неё
нужна рука `fullbar` в том же процессе и на тех же начальных состояниях.

ВТОРАЯ — СОБСТВЕННЫЙ ШУМ ЗАМКНУТОГО ЦИКЛА НЕ ИЗМЕРЕН. В K-9d 30 эпизодов
выиграла глубина 24 и 28 выиграла глубина 12. Оракул, выбирающий лучшую руку
задним числом, даёт 97.0% против 90.0% — но это оракул ПОСТФАКТУМ, он знает
исход. Двадцать восемь эпизодов, где меньшая глубина оказалась лучше, сложно
объяснить свойствами состояния; правдоподобнее хаотическое расхождение
траекторий. Прежде чем строить на этих 58 парах адаптивную глубину, надо
измерить расхождение опоры С САМОЙ СОБОЙ.

ТРИ РАЗНЫЕ ПРОВЕРКИ, И ОНИ НЕ ВЗАИМОЗАМЕНИМЫ.
  A. Точный повтор: та же политика, те же флаги, свежий процесс, метка
     `coarse24_b10r`. Ожидается НОЛЬ расхождений. Если они есть, стенд
     недетерминирован и все прежние парные выводы подлежат пересмотру.
  B. Численное возмущение: `coarse24_b5` против `coarse24_b10`. Политика та
     же, состояния те же, но форма батча другая — другие ядра, другое
     округление, другая позиция наблюдения в батче. Это оценка хаотической
     чувствительности, а не повтор.
  C. Межглубинная разница: `fast12` против `coarse24_b10`, уже измерена,
     58 дискордантных пар из 400.
Если B даёт примерно ту же долю дискордантных пар, что C, то 97%-й оракул
адаптивного запаса не доказывает.

ДВА ПОЛЯ ВМЕСТО ОДНОГО. `run_tag` входит в ключ ячейки агрегатора и разводит
эксперименты; если различать им руки, они окажутся в разных экспериментах и не
сопоставятся вовсе. Поэтому `run_tag` ОБЩИЙ у сравниваемых рук, а различает их
`arm_label`, по которому агрегатор и вызывается: --field arm_label.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ. Границы односторонние, по 5% с каждого края
кластерного бутстрапа по задачам:
  * нижняя выше -margin -> не-худшесть ДОКАЗАНА;
  * верхняя ниже -margin -> ухудшение более чем на margin ДОКАЗАНО;
  * иначе -> НЕ ДОКАЗАНО НИЧЕГО, добираются блоки через --init-start.
Для контроля шума (пары A и B) содержательна не граница, а ДОЛЯ
ДИСКОРДАНТНЫХ ПАР: её сравнивают с 14.5% из пары C.

НЕ ЗАДАВАЙТЕ CUDA_VISIBLE_DEVICES. robosuite выводит MUJOCO_EGL_DEVICE_ID из
первого элемента этого списка, маскировка оставляет одно устройство с индексом
0, и EGL падает на каждой ячейке. Стоило 80 падений за 11 минут.

Запуск:
    python3 experiments/k9h_multiarm_gate.py --selftest

    # прямая пара: Frozen-12 + R* против полной BAR
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9h_multiarm_gate.py --ckpt <base> \\
        --policy fast --policy-ckpt data/k9g_frozen12_rstar.pt \\
        --arm-label fast12_rstar --run-tag k9h_direct \\
        --task-id 0 --init-start 0 --ensemble on \\
        --out data/k9h_direct/t0_i0_fast12_rstar.json

    # контроль шума: тот же coarse24 при другой форме батча
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9h_multiarm_gate.py --ckpt <base> \\
        --policy coarse24 --arm-label coarse24_b5 --run-tag k9h_noise \\
        --n-envs 5 --task-id 0 --init-start 0 --ensemble on \\
        --out data/k9h_noise/t0_i0_coarse24_b5.json

Разбор:
    python3 experiments/k6h_summarize.py --glob 'data/k9h_direct/*.json' \\
        --field arm_label --test fast12_rstar --ref fullbar --margin 5 \\
        --expect-pairs 400 --expect-tasks 10 --require-full-hash
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

N_POS, N_LEVEL = 16, 3
POLICIES = ("fullbar", "coarse24", "fast")
# K-6h, опорные числа. ПО 200 ПАР НА ПРОТОКОЛ, а не 400: 400 — сумма двух.
REFERENCE_K6H = {"on": dict(fullbar=88.0, coarse24=89.0),
                 "off": dict(fullbar=90.0, coarse24=89.5)}


def summarize(eps):
    n = len(eps)
    steps = sum(e["env_steps"] for e in eps)
    calls = sum(e["policy_calls"] for e in eps)
    return dict(episodes=n,
                success_rate=sum(e["success"] for e in eps) / max(n, 1),
                env_steps=steps, policy_calls=calls,
                calls_per_action=calls / max(steps, 1))


def levels_of(policy):
    """Сколько уровней RVQ собирается в действие для данной политики."""
    return N_LEVEL if policy == "fullbar" else 1


def rollout_seed(seed, init_start, mode):
    """Сид раскатки. См. пояснение у --rollout-seed-mode.

    `block` воспроизводит K-6h/K-9d; `fixed` нужен, когда руки идут с разным
    числом сред, иначе один и тот же init_state_id получит разные сиды.
    """
    if mode == "fixed":
        return seed
    return seed + 1000 * init_start


def trunk_digest(state, head_prefix=("action_expert.norm.", "fast_head.")):
    """Отпечаток весов ствола: sha1 по именам, формам и байтам.

    ЗАЧЕМ. «Frozen-12» — утверждение о весах, а не о названии файла. Конвертер
    записывает отпечаток, гейт пересчитывает его у загруженного чекпойнта и
    сверяет. Без этого чекпойнт с подменённым стволом принялся бы молча.
    Функция ПОВТОРЕНА в k9g_convert_rstar.py дословно и обязана совпадать;
    вынести её в joint12_vla.py нельзя — sha этого модуля уже входит в
    законченный результат K-9d.
    """
    h = hashlib.sha1()
    for k in sorted(state):
        if any(k.startswith(p) for p in head_prefix):
            continue
        v = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(str(v.dtype).encode())
        h.update(v.view(torch_uint8()).numpy().tobytes())
    return h.hexdigest()[:16]


def torch_uint8():
    import torch
    return torch.uint8


def selftest():
    for H, want in ((4, 0.25), (8, 0.125)):
        eps = [dict(success=True, env_steps=40, policy_calls=40 // H)
               for _ in range(2)]
        assert abs(summarize(eps)["calls_per_action"] - want) < 1e-12
    assert summarize([dict(success=True, env_steps=10, policy_calls=3),
                      dict(success=False, env_steps=10, policy_calls=3)]
                     )["success_rate"] == 0.5

    assert levels_of("fullbar") == 3
    assert levels_of("coarse24") == 1 and levels_of("fast") == 1

    # СИД РАСКАТКИ. Именно здесь прячется конфаундер контроля шума: в режиме
    # block один и тот же init_state_id получает РАЗНЫЕ сиды при разном числе
    # сред, и сравнение b10 против b5 мерило бы размер батча вместе с сидом.
    assert rollout_seed(0, 0, "block") == 0
    assert rollout_seed(0, 10, "block") == 10000
    assert rollout_seed(0, 5, "block") == 5000
    # состояние 5: блок 0 при n_envs=10, блок 5 при n_envs=5
    assert rollout_seed(0, 0, "block") != rollout_seed(0, 5, "block"), \
        "конфаундер существует — ради этого и введён режим fixed"
    assert rollout_seed(0, 0, "fixed") == rollout_seed(0, 5, "fixed") == 0
    for s in (0, 10, 20, 30):
        assert rollout_seed(7, s, "fixed") == 7

    # Раскладка кодов поуровневая: первые 16 из 48 — уровень 0 (bar.py:1500).
    K = np.arange(N_POS * N_LEVEL)[None].reshape(1, N_LEVEL, N_POS)
    assert (K[0, 0] == np.arange(0, 16)).all()
    assert (K[0, 2] == np.arange(32, 48)).all()

    # БЛОКИ НАЧАЛЬНЫХ СОСТОЯНИЙ ПРИ РАЗНОМ РАЗМЕРЕ БАТЧА обязаны покрывать
    # ОДИН И ТОТ ЖЕ набор init_state_id — иначе контроль шума сравнивал бы
    # разные эпизоды, а не одни и те же при другом округлении.
    b10 = [s + i for s in (0, 10, 20, 30) for i in range(10)]
    b5 = [s + i for s in range(0, 40, 5) for i in range(5)]
    assert b10 == list(range(40)) and b5 == list(range(40))
    assert len(set(b10)) == len(set(b5)) == 40

    # run_tag общий, arm_label различающий: если перепутать, пары разъедутся
    # по разным экспериментам и агрегатор не найдёт ни одной.
    cells = {}
    for lab in ("coarse24_b10", "coarse24_b5"):
        cells.setdefault(("k9h_noise", "10", 0, "on", 8, 0), {})[lab] = 1
    assert len(cells) == 1 and len(next(iter(cells.values()))) == 2, \
        "общий run_tag обязан собирать руки в одну ячейку"
    wrong = {}
    for tag, lab in (("a", "x"), ("b", "y")):
        wrong.setdefault((tag, "10", 0, "on", 8, 0), {})[lab] = 1
    assert len(wrong) == 2, "разные run_tag разносят руки по экспериментам"

    print("самопроверка k9h пройдена (версия «run_tag общий, arm_label "
          "различающий»): нормировка вызовов, уровни по политике, покрытие "
          "блоков при batch 10 и 5, ключ ячейки")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", help="базовый чекпойнт BAR (все руки)")
    ap.add_argument("--policy", choices=POLICIES, default=None,
                    help="fullbar: 24 слоя x 3 прохода, 3 уровня; "
                         "coarse24: 24 слоя x 1 проход, уровень 0; "
                         "fast: forward_joint_fast, уровень 0")
    ap.add_argument("--policy-ckpt", default=None,
                    help="чекпойнт формата k9c/k9g; только для --policy fast")
    ap.add_argument("--arm-label", default=None,
                    help="ОБЯЗАТЕЛЕН. Различает руки внутри эксперимента, "
                         "например fullbar, coarse24_b10, coarse24_b5, "
                         "coarse24_b10r, fast12_rstar")
    ap.add_argument("--run-tag", default=None,
                    help="ОБЯЗАТЕЛЕН и ОДИНАКОВ у сравниваемых рук: он входит "
                         "в ключ ячейки агрегатора")
    ap.add_argument("--expect-depth", type=int, default=None,
                    help="глубина, которую обязан объявить --policy-ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--init-start", type=int, default=0)
    ap.add_argument("--ensemble", choices=["on", "off"], default=None,
                    help="ОБЯЗАТЕЛЕН, умолчания нет намеренно")
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    # СИД РАСКАТКИ. Режим `block` повторяет K-5b/K-6h/K-9d дословно:
    # seed + 1000 * init_start. Он корректен, пока все руки идут с одним
    # числом сред, и НЕПРИГОДЕН для контроля численного шума: при n_envs=10
    # состояние 5 лежит в блоке init_start=0 и получает сид 0, а при n_envs=5
    # — в блоке init_start=5 и сид 5000. Сравнение b10 против b5 тогда
    # мерило бы размер батча ВМЕСТЕ с другим сидом. Режим `fixed` берёт один
    # args.seed на все блоки и снимает конфаундер.
    ap.add_argument("--rollout-seed-mode", choices=["block", "fixed"],
                    default="block")
    ap.add_argument("--expect-source", default=None,
                    help="обязательное значение поля source в --policy-ckpt, "
                         "например frozen12_rstar")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")
    for need in ("policy", "arm_label", "run_tag", "ensemble"):
        if getattr(args, need) is None:
            raise SystemExit(f"--{need.replace('_', '-')} обязателен")
    if args.policy == "fast" and not args.policy_ckpt:
        raise SystemExit("--policy fast требует --policy-ckpt")
    if args.policy != "fast" and args.policy_ckpt:
        raise SystemExit(
            f"--policy {args.policy} исполняется на ИСХОДНЫХ весах; "
            f"--policy-ckpt здесь запрещён, иначе опора уедет вместе с рукой")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    import joint12_vla as jv
    from joint12_vla import make_joint12_class
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

    # СРЕДЫ ДО МОДЕЛИ: fork после инициализации CUDA вешает процесс. Порядок
    # тот же, что в K-6h/K-9d, и от него зависит расход глобального ГСЧ, то
    # есть начальные состояния. Менять нельзя.
    seed_everything(args.seed)
    envs, task_desc = get_envs(args.task_suite,
                               {"task_id": args.task_id, "image_size": 224},
                               args.n_envs)

    Joint = make_joint12_class(SmolVLABlockwiseAR)
    model = Joint.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    policy_meta, depth = None, 24
    if args.policy == "fast":
        h = hashlib.sha1()
        with open(args.policy_ckpt, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""):
                h.update(c)
        weights_sha = h.hexdigest()[:12]
        obj = torch.load(args.policy_ckpt, map_location="cpu",
                         weights_only=False)
        depth = int(obj["depth"])
        if args.expect_depth is not None and depth != args.expect_depth:
            raise SystemExit(f"чекпойнт глубины {depth}, ожидалась "
                             f"{args.expect_depth}")
        model.init_joint_fast(depth=depth)
        state = obj["state"]
        stray = [k for k in state
                 if not any(k.startswith(p) or k == p.rstrip(".")
                            for p in model.trainable_prefixes)]
        if stray:
            raise SystemExit(f"{len(stray)} ключей вне белого списка: "
                             f"{stray[:5]}")
        own = dict(model.named_parameters())
        missing = [k for k in own if own[k].requires_grad and k not in state]
        if missing:
            raise SystemExit(f"нет {len(missing)} обучаемых весов: "
                             f"{missing[:5]}")
        with torch.no_grad():
            for k, v in state.items():
                if tuple(own[k].shape) != tuple(v.shape):
                    raise SystemExit(f"форма {k}")
                if not torch.isfinite(v).all():
                    raise SystemExit(f"в {k} есть nan или inf")
                own[k].data = v.to(dev, torch.float32)
        model.eval()
        # SOURCE, ГЛУБИНА И ВЕРСИЯ РЕАЛИЗАЦИИ — обязательные, а не справочные.
        # Чекпойнт глубины 18 под меткой fast12_rstar загрузился бы без
        # единой жалобы и исполнил бы другую сеть.
        src = obj.get("source")
        if args.expect_source is not None and src != args.expect_source:
            raise SystemExit(f"чекпойнт source={src!r}, ожидалось "
                             f"{args.expect_source!r}")
        # ПРИ ЗАДАННОМ --expect-source ОТСУТСТВИЕ ПОЛЯ САМО ЕСТЬ ОШИБКА.
        # Условная проверка «сверить, если поле есть» открыта настежь: старый
        # чекпойнт с правильным source, но без доказательств происхождения,
        # прошёл бы её молча.
        strict = args.expect_source is not None
        cur_vla = hashlib.sha1(open(jv.__file__, "rb").read()).hexdigest()[:12]
        ck_vla = obj.get("joint12_vla_sha1")
        if strict and ck_vla is None:
            raise SystemExit(
                "в чекпойнте нет joint12_vla_sha1, а --expect-source задан: "
                "происхождение не подтверждено. Пересоберите через k9g.")
        if ck_vla is not None and ck_vla != cur_vla:
            raise SystemExit(
                f"чекпойнт собран на joint12_vla.py sha {ck_vla}, а сейчас "
                f"{cur_vla}. forward_joint_fast определяет исполняемую сеть "
                f"не меньше, чем веса.")
        # ОТПЕЧАТОК СТВОЛА. Для frozen12_rstar он обязан совпасть с записанным
        # конвертером: «замороженный» — утверждение о весах, а не о названии.
        dig = trunk_digest(state)
        ck_dig = obj.get("trunk_digest")
        if strict and ck_dig is None:
            raise SystemExit(
                "в чекпойнте нет trunk_digest, а --expect-source задан: "
                "«замороженность» ствола ничем не подтверждена. "
                "Пересоберите через k9g.")
        if ck_dig is not None and ck_dig != dig:
            raise SystemExit(
                f"отпечаток ствола {dig} против {ck_dig} в чекпойнте — веса "
                f"ствола изменились после сборки")
        policy_meta = dict(path=os.path.abspath(args.policy_ckpt),
                           weights_sha1=weights_sha, depth=depth,
                           tensors=len(state), source=src,
                           built_by_sha1=obj.get("sha1"),
                           rstar_sha1=obj.get("rstar_sha1"),
                           trunk_digest=dig, joint12_vla_sha1=cur_vla)
        print(f"  политика fast: {len(state)} тензоров, глубина {depth}, "
              f"веса sha {weights_sha}, source={src}, ствол {dig}")
    else:
        # init_joint_fast НЕ ВЫЗЫВАЕТСЯ: он создаёт fast_head и сдвигает
        # аллокатор, а в K-9b именно сдвиг аллокатора давал расхождение
        # логитов 5.9e-02 при полностью совпадающих весах. Опорные руки
        # обязаны быть побитово той же веткой, что в K-6h.
        print(f"  политика {args.policy}: generate на исходных весах, "
              f"{levels_of(args.policy)} уровн(я)")

    import contextlib
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if args.policy == "fast" else contextlib.nullcontext())

    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь в action_processor")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        # ИНДЕКСЫ НА УСТРОЙСТВЕ КНИГ: arange по умолчанию на CPU, F.embedding
        # падает. Стоило одного упавшего ночного прогона в K-6h.
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    n_lv = levels_of(args.policy)
    print(f"=== suite {args.task_suite}, задача {args.task_id}, офсет {pos_off}")
    print(f"    «{task_desc}»   H={args.horizon}, метка={args.arm_label}, "
          f"эксперимент={args.run_tag}, сред={args.n_envs}, "
          f"глубина={depth}, уровней={n_lv}, ens={args.ensemble}")

    identity_checked = [False]

    def check_assembly(toks48):
        """Своя сборка трёх уровней обязана совпасть с официальным decode.

        Проверяется один раз, на любых валидных индексах: это тождество про
        кодек, а не про качество политики.
        """
        if identity_checked[0]:
            return
        identity_checked[0] = True
        K = toks48.reshape(-1, N_LEVEL, N_POS)
        with torch.no_grad():
            z3 = sum(E[j][torch.as_tensor(K[:, j, :]).long().to(dev)]
                     for j in range(N_LEVEL))
            x3, _ = codec._decode(z3, embodiment_ids=0)
            mine = x3[..., :7].float().cpu().numpy()
        ref = np.asarray(proc.action_processor.decode(toks48.tolist())[0],
                         np.float64)
        d = float(np.abs(mine - ref).max())
        print(f"    тождество сборки при трёх уровнях: max|Δ| = {d:.3e}")
        if d > 1e-3:
            raise SystemExit(f"своя сборка расходится с официальным decode на "
                             f"{d:.3e} — сравнение недействительно")

    def decode(codes):
        """Действие из первых n_lv уровней — то, что исполняет симулятор."""
        K = codes.reshape(-1, n_lv, N_POS) if n_lv > 1 else codes.reshape(-1, 1, N_POS)
        with torch.no_grad():
            zq = E[0][torch.as_tensor(K[:, 0, :]).long().to(dev)]
            for j in range(1, n_lv):
                zq = zq + E[j][torch.as_tensor(K[:, j, :]).long().to(dev)]
            x, _ = codec._decode(zq, embodiment_ids=0)
            return x[..., :7].float().cpu().numpy()

    def policy(batch, first):
        """Коды для сборки действия, форма (B, n_lv * 16)."""
        if args.policy in ("fullbar", "coarse24"):
            with torch.no_grad():
                toks = model.generate(**batch, position_offset=pos_off,
                                      do_sample=False, initial_position_shift=1)
            t = toks.cpu().numpy()
            check_assembly(t)
            K = t.reshape(-1, N_LEVEL, N_POS)
            return K.reshape(len(t), -1) if n_lv == N_LEVEL else K[:, 0, :]

        with torch.no_grad(), autocast:
            v, p = model.build_inputs(position_offset=pos_off, **batch)
            out = model.forward_joint_fast(
                vlm_inputs_embeds=v, attention_mask=batch.get("attention_mask"),
                position_ids=p)
        codes = out["pred_codes"].cpu().numpy()
        if first:
            assert out["layers_run"] == depth, out["layers_run"]
            # Тождество кодека без вызова generate: 48 валидных индексов
            # собираются повтором собственных шестнадцати. Обученная модель не
            # гоняется по пути, который в обучении не исполнялся.
            check_assembly(np.concatenate([codes] * N_LEVEL, axis=1))
        return codes

    roll_seed = rollout_seed(args.seed, args.init_start,
                             args.rollout_seed_mode)
    print(f"    сид раскатки {roll_seed} (режим {args.rollout_seed_mode})")

    def rollout():
        # СИД СБРАСЫВАЕТСЯ ПЕРЕД РАУНДОМ, как в K-5b/K-6h/K-9d: иначе расход
        # глобального ГСЧ различался бы между руками и начальные состояния
        # разошлись бы, а вся статистика здесь парная.
        seed_everything(roll_seed)
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

        def _h(parts):
            return hashlib.sha1(np.ascontiguousarray(
                np.concatenate(parts).astype(np.float32)).tobytes()
            ).hexdigest()[:16]
        # ДВА ХЕША: init_hash повторяет формулу K-6h дословно ради сравнимости
        # со старыми ячейками, init_hash_full добавляет камеру на запястье —
        # политика смотрит в обе, и совпадение по одной слабее, чем нужно.
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
            act = decode(policy(batch, calls == 0))
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
        # rollout_seed ЛЕЖИТ В КАЖДОМ ЭПИЗОДЕ, а не только в шапке файла:
        # агрегатор сверяет его у обеих рук пары и отказывается сравнивать
        # эпизоды, раскатанные с разными сидами.
        return [dict(success=bool(reward[i] >= 1.0), env_steps=steps,
                     policy_calls=calls, init_state_id=args.init_start + i,
                     env_index=i, init_hash=init_hash[i],
                     init_hash_full=init_hash_full[i],
                     rollout_seed=roll_seed)
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
    print(f"\n  метка {args.arm_label}, H={args.horizon}, ens={args.ensemble}: "
          f"успех {s['success_rate']:.1%} "
          f"({sum(e['success'] for e in eps)}/{len(eps)}), "
          f"вызовов на действие {s['calls_per_action']:.3f}")
    print(f"  время: {(time.time() - t0) / 60:.1f} мин")
    ref = REFERENCE_K6H[args.ensemble]
    print("\n  ЧИТАТЬ ТАК: не по этому числу. Оно осмысленно только в паре с")
    print("  другой рукой того же --run-tag при тех же task-id, init-start,")
    print("  seed и ensemble, и только через k6h_summarize.py --field")
    print("  arm_label. Опора K-6h при ens=%s: полная BAR %.1f%%, coarse24 "
          "%.1f%%\n  (по 200 пар на протокол)." % (args.ensemble,
                                                   ref["fullbar"],
                                                   ref["coarse24"]))
    print("  Латентность здесь НЕ меряется — для неё k7a на фиксированных "
          "входах.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
        json.dump(dict(summary=s, episodes=eps,
                       arm_label=args.arm_label, run_tag=args.run_tag,
                       policy=args.policy, levels=n_lv, depth=depth,
                       horizon=args.horizon, task_id=args.task_id,
                       suite=args.task_suite, pos_offset=pos_off,
                       ensemble=args.ensemble, init_start=args.init_start,
                       n_envs=args.n_envs, task_description=task_desc,
                       seed=args.seed, rollout_seed=roll_seed,
                       rollout_seed_mode=args.rollout_seed_mode,
                       ckpt=args.ckpt, joint=policy_meta,
                       script_sha1=sha, argv=vars(args)),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}  (sha {sha})")


if __name__ == "__main__":
    main()
