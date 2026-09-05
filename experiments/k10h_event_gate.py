"""K-10h: замкнутый симуляторный гейт для событийных целей.

ЗАЧЕМ. Офлайновый гейт K-10d признан асимметричным (K-10g): позную его часть
тривиальный повтор предыдущей команды проходит насквозь, во всех бакетах и на
обеих частях, а отвергается он единственным каналом — знаком схвата. Такой
критерий может отвергать, но не может подтверждать, поэтому вопрос «реализуем
ли запас 4.5x по частоте вызовов» им не решается ни в какую сторону. Решает
только замкнутый цикл: успех задачи и фактическое число вызовов VLA.

ЧТО ИЗМЕРЯЕТСЯ. Две величины сразу, и обе обязательны:
  * успех, парно с опорой на тех же начальных состояниях;
  * ВЫЗОВОВ НА ШАГ СРЕДЫ.
Рука, сохранившая успех, но вызывающая VLA столь же часто, не даёт ничего.
Рука, срезавшая вызовы вчетверо и потерявшая успех, — тоже. Ни одну из этих
величин нельзя читать без другой.

ЦЕЛЬ — ОРАКУЛЬНАЯ, И ЭТО ВЕРХНЯЯ ОЦЕНКА. Абсолютные позы событий берутся из
УСПЕШНОЙ раскатки опоры на том же начальном состоянии (режим `record`).
Настоящий монитор, определяющий момент события без запуска VLA, не
спроектирован. Поэтому положительный результат здесь означает лишь «при
идеальном знании моментов событий запас реализуем», а не готовую систему.
Отрицательный же результат окончателен: если не работает при оракульной цели,
то без неё тем более.

РУКИ.
  coarse24  опора: VLA каждые H шагов, уровень 0, 24 слоя. Известные 90%.
  copyprev  отрицательный контроль: VLA в начале фазы, дальше ПОВТОР
            последней команды. Цель не используется вовсе. В офлайне эта
            рука била опору по позиции на 15-25% и по вращению на 35-41%
            (K-10g); если она провалит задачи, это прямая демонстрация
            разрыва между пошаговой метрикой и успехом.
  servo     геометрический П-регулятор к относительной цели, коэффициент
            подобран МНК в режиме `record` на траекториях опоры.
  learned   контроллер из K-10d. Вход честный: `remaining` из будущего
            запрещён, если явно не разрешён флагом, и тогда рука меняет
            смысл на верхнюю границу.

ЧЕГО ЗДЕСЬ НЕТ. Ансамблирование выключено у ВСЕХ рук принудительно.
`ActionEnsembler` усредняет перекрывающиеся чанки, предполагая вызов каждые H
шагов; при событийном расписании вызовы нерегулярны, и это предположение
ложно. Опора с ансамблем была бы другой политикой, а пара — недействительной.

ПОЧЕМУ ОПОРА СЧИТАЕТСЯ ЗДЕСЬ, А НЕ БЕРЁТСЯ ИЗ K-9h. Сравнение парное, и
любое расхождение в сборке сред, порядке сидов или обработке наблюдений
разрушает пару незаметно. Поэтому опора раскатывается ЭТИМ скриптом, а
`--expect-match` сверяет её с ячейкой K-9h эпизод в эпизод. Расхождение —
отказ, а не предупреждение: оно означает, что дублирование среды разошлось.

Запуск:
    python3 experiments/k10h_event_gate.py --selftest

    # 1. запись оракульных целей опорой (она же даёт ячейку опоры)
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k10h_event_gate.py --mode record --ckpt <base> \\
        --task-id 0 --init-start 0 --n-envs 10 \\
        --record data/k10h_goals/t0_i0.json \\
        --out data/k10h_gate/t0_i0_coarse24.json --arm coarse24 \\
        --run-tag k10h --expect-match data/k9h_noise/t0_i0_coarse24_b10.json

    # 2. событийная рука на тех же состояниях
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k10h_event_gate.py --mode gate --ckpt <base> \\
        --task-id 0 --init-start 0 --n-envs 10 --arm copyprev \\
        --record data/k10h_goals/t0_i0.json \\
        --out data/k10h_gate/t0_i0_copyprev.json --run-tag k10h

Разбор:
    python3 experiments/k6h_summarize.py --glob 'data/k10h_gate/*.json' \\
        --field arm_label --test copyprev --ref coarse24 --margin 5 \\
        --expect-pairs 400 --expect-tasks 10 --require-full-hash
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import goal_events as ge                                       # noqa: E402
import goal_dataset as gd                                      # noqa: E402

ARMS = ("coarse24", "copyprev", "servo", "learned")
# Руки, которым нужны оракульные цели. Опора идёт без них.
EVENT_ARMS = ("copyprev", "servo", "learned")
N_LEVEL, N_POS = 3, 16


def summarize(eps):
    """Те же поля, что у K-9h, плюс частота вызовов как отдельная величина."""
    n = len(eps)
    steps = sum(e["env_steps"] for e in eps)
    calls = sum(e["policy_calls"] for e in eps)
    return dict(n_episodes=n,
                success_rate=sum(e["success"] for e in eps) / max(n, 1),
                env_steps=steps, policy_calls=calls,
                calls_per_action=calls / max(steps, 1),
                steps_per_call=steps / max(calls, 1))


def rollout_seed(seed, init_start, mode):
    """Дословно как в K-9h: пара обязана идти с одним сидом."""
    return seed + 1000 * init_start if mode == "block" else seed


class Monitor:
    """Геометрический монитор достижения цели.

    БУДУЩЕЕ НЕДОСТУПНО. Монитор видит только текущую позу и абсолютную позу
    цели. Настоящий `remaining` сюда не передаётся ни в каком виде — иначе
    измерялась бы не реализуемость запаса, а оракул расписания.

    ТАЙМАУТ ОБЯЗАТЕЛЕН. Без него недостижимая цель держала бы фазу до конца
    эпизода, вызовов было бы мало, а успеха ноль — и число «вызовов на шаг»
    выглядело бы прекрасно ровно у самой неработающей руки.
    """

    def __init__(self, eps_pos=0.02, eps_rot=0.15, max_phase=48, min_phase=8):
        self.eps_pos, self.eps_rot = eps_pos, eps_rot
        self.max_phase, self.min_phase = max_phase, min_phase

    def reached(self, pos, aa, goal_pos, goal_aa, phase_steps):
        if phase_steps >= self.max_phase:
            return True, "таймаут"
        if phase_steps < self.min_phase:
            return False, ""
        dp = float(np.linalg.norm(np.asarray(pos) - np.asarray(goal_pos)))
        R = gd.aa_to_R(np.asarray(aa)).T @ gd.aa_to_R(np.asarray(goal_aa))
        dr = float(np.linalg.norm(gd.R_to_aa(R)))
        if dp <= self.eps_pos and dr <= self.eps_rot:
            return True, "достигнута"
        return False, ""


def fit_servo(traj, goals):
    """МНК-коэффициенты П-регулятора по траекториям опоры.

    Регрессия БЕЗ СВОБОДНОГО ЧЛЕНА: команда в нуле ошибки обязана быть нулём,
    иначе регулятор дрейфует при достигнутой цели. Отдельные коэффициенты для
    положения и вращения — величины разной размерности.
    """
    ep, ea, ap, aa_ = [], [], [], []
    for tr, gl in zip(traj, goals):
        st = np.asarray(tr["state"], np.float64)
        ac = np.asarray(tr["action"], np.float64)
        gi = np.asarray(gl["goal_index"], np.int64)
        n = min(len(st), len(ac), len(gi))
        for t in range(n):
            g = np.asarray(gl["goal_pose"][int(gi[t])], np.float64)
            # rel_goal ВОЗВРАЩАЕТ ОДИН ВЕКТОР ИЗ ШЕСТИ ЧИСЕЛ, а не пару:
            # распаковка в две переменные молча взяла бы первые два числа.
            d = gd.rel_goal(st[t, :3], st[t, 3:6], g[:3], g[3:6])
            ep.append(d[:3]); ap.append(ac[t, :3])
            ea.append(d[3:6]); aa_.append(ac[t, 3:6])
    ep, ap = np.concatenate(ep), np.concatenate(ap)
    ea, aa_ = np.concatenate(ea), np.concatenate(aa_)
    kp = float(ep @ ap / max(ep @ ep, 1e-12))
    kr = float(ea @ aa_ / max(ea @ ea, 1e-12))
    return kp, kr


def servo_action(state, goal_pose, kp, kr, grip_hold):
    """Команда П-регулятора в том же пространстве, что выдаёт кодек."""
    g = np.asarray(goal_pose, np.float64)
    d = gd.rel_goal(np.asarray(state)[:3], np.asarray(state)[3:6],
                    g[:3], g[3:6])
    a = np.zeros(7, np.float64)
    a[:3] = np.clip(kp * d[:3], -1.0, 1.0)
    a[3:6] = np.clip(kr * d[3:6], -1.0, 1.0)
    # ЗНАК СХВАТА НЕ ИНТЕРПОЛИРУЕТСЯ. Он дискретен, и П-регулятор о нём
    # ничего не знает; держим последнюю команду VLA до следующего вызова.
    a[6] = grip_hold
    return a


def label_from_rollout(state, action, kind, params):
    """Разметка событий по СОБСТВЕННОЙ траектории опоры, не по демонстрации.

    Цели обязаны быть достижимы той же системой, что их породила. Разметка
    демонстрации дала бы позы, до которых политика может не доводить руку
    вовсе, и таймаут срабатывал бы всегда.
    """
    ev, typ, _ = ge.label(np.asarray(action), np.asarray(state)[:, :3],
                          kind=kind, **params)
    n = len(state)
    tau, ttyp, rem = ge.targets(np.asarray(action), ev, typ)
    poses = [np.asarray(state)[int(i), :6].tolist()
             for i in range(n)]
    # Список УНИКАЛЬНЫХ целей и индекс цели для каждого шага: в замкнутом
    # цикле шаги пойдут по-другому, и обращаться придётся по номеру фазы.
    # ЦЕЛЬ — НЕ ТОЛЬКО ПОЗА. Вход контроллера содержит ещё знак схвата В
    # ЦЕЛИ и тип события; без них онлайновая сборка признаков не повторит
    # обучающую, а сеть примет чужой вектор без единой жалобы.
    uniq, gsign, gtype, gi = [], [], [], np.zeros(n, np.int64)
    last = -1
    for t in range(n):
        if int(tau[t]) != last:
            last = int(tau[t])
            uniq.append(poses[last])
            gsign.append(float(np.sign(np.asarray(action)[last, 6]) or 1.0))
            gtype.append(int(ttyp[t]))
        gi[t] = len(uniq) - 1
    return dict(goal_pose=uniq, goal_sign=gsign, goal_type=gtype,
                goal_index=gi.tolist(),
                events=[int(x) for x in ev], types=[int(x) for x in typ],
                remaining=[int(x) for x in rem])


def selftest():
    ge.selftest()
    gd.selftest()

    # --- монитор -----------------------------------------------------------
    m = Monitor(eps_pos=0.02, eps_rot=0.15, max_phase=48, min_phase=8)
    p, a = np.zeros(3), np.zeros(3)
    # Ближе допуска, но фаза короче минимума — переключаться рано.
    assert m.reached(p, a, p, a, 3)[0] is False
    assert m.reached(p, a, p, a, 10)[0] is True
    # Далеко — не переключаться, пока не выйдет таймаут.
    far = np.array([1.0, 0, 0])
    assert m.reached(p, a, far, a, 20)[0] is False
    ok, why = m.reached(p, a, far, a, 48)
    assert ok and why == "таймаут", (ok, why)
    # ВРАЩЕНИЕ УЧИТЫВАЕТСЯ ОТДЕЛЬНО: совпадение позиции при развороте на пи
    # не есть достижение цели.
    assert m.reached(p, a, p, np.array([np.pi, 0, 0]), 20)[0] is False

    # --- П-регулятор -------------------------------------------------------
    # Синтетика с известным ответом: команда ровно вдвое меньше ошибки.
    rng = np.random.default_rng(0)
    st = rng.normal(0, 0.1, (200, 8))
    goal = np.zeros(6)
    ac = np.zeros((200, 7))
    for t in range(200):
        d = gd.rel_goal(st[t, :3], st[t, 3:6], goal[:3], goal[3:6])
        ac[t, :3], ac[t, 3:6] = 0.5 * d[:3], 0.25 * d[3:6]
    tr = [dict(state=st.tolist(), action=ac.tolist())]
    gl = [dict(goal_pose=[goal.tolist()], goal_index=[0] * 200)]
    kp, kr = fit_servo(tr, gl)
    assert abs(kp - 0.5) < 1e-6 and abs(kr - 0.25) < 1e-6, (kp, kr)
    # Регулятор с этими коэффициентами воспроизводит команду.
    a0 = servo_action(st[0], goal, kp, kr, grip_hold=-1.0)
    assert np.allclose(a0[:6], ac[0, :6], atol=1e-6)
    assert a0[6] == -1.0, "знак схвата держится, а не интерполируется"

    # --- разметка по собственной траектории --------------------------------
    n = 120
    stt = np.zeros((n, 8))
    stt[:, 0] = np.linspace(0, 1, n)
    act = np.zeros((n, 7)); act[:, 6] = 1.0; act[60:, 6] = -1.0
    lab = label_from_rollout(stt, act, "grip", dict())
    assert lab["events"] == [60]
    # Целей ровно две: событие и конец эпизода.
    assert len(lab["goal_pose"]) == 2, lab["goal_pose"]
    # До события все шаги целятся в первую цель, после — во вторую.
    assert lab["goal_index"][0] == 0 and lab["goal_index"][59] == 0
    assert lab["goal_index"][61] == 1
    # Поза первой цели — состояние В МОМЕНТ события, а не в конце.
    assert abs(lab["goal_pose"][0][0] - stt[60, 0]) < 1e-12
    assert lab["remaining"][0] == 60 and lab["remaining"][60] == 0

    # --- сводка ------------------------------------------------------------
    s = summarize([dict(success=True, env_steps=100, policy_calls=5),
                   dict(success=False, env_steps=200, policy_calls=25)])
    assert s["success_rate"] == 0.5
    # ВЫЗОВОВ НА ШАГ считается по СУММАМ, а не средним по эпизодам: короткий
    # успешный эпизод иначе весил бы столько же, сколько длинный провальный.
    assert abs(s["calls_per_action"] - 30 / 300) < 1e-12
    assert abs(s["steps_per_call"] - 10.0) < 1e-12

    print("самопроверка k10h пройдена (версия «оракульная цель, честный "
          "монитор»): монитор учитывает вращение и таймаут, МНК-регулятор "
          "восстанавливает известные коэффициенты, знак схвата держится, "
          "цель ставится в момент события, частота вызовов по суммам")


def load_record(path, task_id, init_start, n_envs, arm):
    with open(path) as f:
        j = json.load(f)
    if j.get("task_id") != task_id or j.get("init_start") != init_start:
        raise SystemExit(
            f"{path}: записан для задачи {j.get('task_id')} / офсета "
            f"{j.get('init_start')}, а просят {task_id} / {init_start}")
    if j.get("n_envs") != n_envs:
        raise SystemExit(f"{path}: сред {j.get('n_envs')}, а просят {n_envs}")
    if arm == "servo" and "servo_kp" not in j:
        raise SystemExit(f"{path}: нет коэффициентов регулятора")
    # ЦЕЛИ ТОЛЬКО ИЗ УСПЕШНЫХ ЭПИЗОДОВ. Разметка провалившейся раскатки
    # описывает, как опора не справилась, и вести по ней бессмысленно.
    bad = [i for i, g in enumerate(j["goals"]) if not g["source_success"]]
    if bad:
        raise SystemExit(
            f"{path}: эпизоды {bad} записаны из НЕУСПЕШНЫХ раскаток опоры; "
            f"событийные руки на них не сравниваются — исключите блок")
    return j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=["record", "gate"], default="gate")
    ap.add_argument("--ckpt")
    ap.add_argument("--arm", choices=ARMS, default=None)
    ap.add_argument("--run-tag", default=None,
                    help="ОБЯЗАТЕЛЕН и ОДИНАКОВ у сравниваемых рук")
    ap.add_argument("--record", default=None,
                    help="файл оракульных целей: пишется в режиме record, "
                         "читается в режиме gate")
    ap.add_argument("--controller", default=None,
                    help="controller.pt от k10d; только для --arm learned")
    ap.add_argument("--allow-oracle-remaining", action="store_true",
                    help="разрешить контроллеру признак remaining. Это "
                         "величина из будущего: рука перестаёт быть честной "
                         "и становится верхней границей")
    ap.add_argument("--expect-match", default=None,
                    help="ячейка K-9h с той же опорой: сверяется эпизод в "
                         "эпизод")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--init-start", type=int, default=0)
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-seed-mode", choices=["block", "fixed"],
                    default="block")
    ap.add_argument("--event", choices=list(ge.EVENT_KINDS), default="union")
    ap.add_argument("--speed-frac", type=float, default=0.3)
    ap.add_argument("--min-dwell", type=int, default=3)
    ap.add_argument("--min-travel", type=float, default=0.02)
    ap.add_argument("--merge-tol", type=int, default=4)
    ap.add_argument("--eps-pos", type=float, default=0.02)
    ap.add_argument("--eps-rot", type=float, default=0.15)
    ap.add_argument("--max-phase", type=int, default=48)
    ap.add_argument("--min-phase", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k10h sha1 {sha}")
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")
    if not args.arm:
        raise SystemExit("нужен --arm")
    if not args.run_tag:
        raise SystemExit("нужен --run-tag: он входит в ключ ячейки")
    if args.mode == "record" and args.arm != "coarse24":
        raise SystemExit("режим record раскатывает только опору coarse24")
    if args.mode == "record" and not args.record:
        raise SystemExit("режим record требует --record")
    if args.arm in EVENT_ARMS and not args.record:
        raise SystemExit(f"рука {args.arm} требует --record с целями")
    if args.arm == "learned" and not args.controller:
        raise SystemExit("рука learned требует --controller")
    if args.min_phase < args.horizon:
        # ФАЗА КОРОЧЕ ЧАНКА НЕ ИМЕЕТ СМЫСЛА: первые H шагов исполняет сам
        # VLA, и переключение раньше означало бы вызов на каждый чанк, то
        # есть расписание опоры под чужой меткой.
        raise SystemExit(f"--min-phase {args.min_phase} меньше горизонта "
                         f"{args.horizon}: рука выродилась бы в опору")

    rec = None
    if args.arm in EVENT_ARMS:
        rec = load_record(args.record, args.task_id, args.init_start,
                          args.n_envs, args.arm)

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    os.chdir(root)
    import torch
    from omegaconf import OmegaConf
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_envs, process_state, prompt_template,
                       seed_everything)
    import actioncodec  # noqa: F401  регистрирует action_codec в AutoModel
    from smolvla.bar import SmolVLABlockwiseAR

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = OmegaConf.load(args.cfg_path)
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    ctrl_meta = None
    if args.arm == "learned":
        obj = torch.load(args.controller, map_location="cpu",
                         weights_only=False)
        # ОТСУТСТВИЕ ПОЛЯ — ОТКАЗ, А НЕ ПУСТОЙ СПИСОК. `obj.get("parts", ())`
        # на старом чекпойнте дал бы вход «только состояние и цель», сеть
        # приняла бы его по ширине и выдавала бы осмысленные с виду чужие
        # команды.
        need = ("parts", "tmap", "n_task", "din", "hidden")
        miss = [k for k in need if k not in obj]
        if miss:
            raise SystemExit(
                f"{args.controller}: нет полей {miss} — чекпойнт собран до "
                f"того, как K-10d стал записывать сборку входа. Переобучите "
                f"контроллер текущей версией K-10d")
        parts = tuple(obj["parts"])
        if "remaining" in parts and not args.allow_oracle_remaining:
            raise SystemExit(
                "контроллер обучен с признаком remaining — это величина из "
                "будущего, в замкнутом цикле она недоступна. Либо возьмите "
                "честный чекпойнт, либо явно поставьте "
                "--allow-oracle-remaining и читайте руку как верхнюю границу")
        ctrl_meta = dict(path=os.path.abspath(args.controller), parts=parts,
                         tmap=obj["tmap"], n_task=int(obj["n_task"]),
                         oracle_remaining="remaining" in parts,
                         weights_sha1=hashlib.sha1(
                             open(args.controller, "rb").read()
                         ).hexdigest()[:12])

    pos_off = args.pos_offset
    if pos_off is None:
        with open(os.path.join(os.path.dirname(root), args.offset_table)) as f:
            pos_off = int(json.load(f)[str(args.task_id)])

    seed_everything(args.seed)
    envs, task_desc = get_envs(args.task_suite,
                               {"task_id": args.task_id, "image_size": 224},
                               args.n_envs)
    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tf = proc.image_processor

    ac_p = proc.action_processor
    codec = ac_p if hasattr(ac_p, "vq") else getattr(ac_p, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    ctrl = None
    task_index = None
    if args.arm == "learned":
        ctrl = gd.make_ctrl()(int(obj["din"]), int(obj["hidden"]))
        ctrl.load_state_dict(obj["state"])
        ctrl = ctrl.to(dev).eval()
        if "task" in ctrl_meta["parts"]:
            # ИНДЕКС ЗАДАЧИ — ПОЗИЦИЯ В `tmap` КОНТРОЛЛЕРА, а не task_id
            # LIBERO. Подстановка task_id включила бы чужой столбец one-hot.
            key = f"{args.task_suite}/{args.task_id}"
            tm = ctrl_meta["tmap"]
            cand = [tm[k] for k in (key, str(args.task_id)) if k in tm]
            if not cand:
                raise SystemExit(
                    f"задачи «{key}» нет в tmap контроллера ({len(tm)} "
                    f"записей): он её не видел, вести по нему нельзя")
            task_index = int(cand[0])

    max_act_q = np.asarray(ac_p.max_act_q, np.float64) \
        if hasattr(ac_p, "max_act_q") else None
    if max_act_q is None:
        raise SystemExit("не нашёл max_act_q — масштабирование неизвестно")

    def norm_state(x):
        """Та же формула, что в K-10d: иначе вход сместился бы целиком."""
        return ((x - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2 - 1
                if x.shape[-1] == len(STATE_Q01) else x)

    def decode(codes):
        K = codes.reshape(-1, 1, N_POS)
        with torch.no_grad():
            zq = E[0][torch.as_tensor(K[:, 0, :]).long().to(dev)]
            x, _ = codec._decode(zq, embodiment_ids=0)
            return x[..., :7].float().cpu().numpy()

    def call_vla(obs, sel):
        """Один вызов VLA для подмножества сред `sel`. Возвращает чанк."""
        state = ((process_state(obs["state"][sel]) - STATE_Q01)
                 / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0)
        i1 = tf(torch.tensor(
            obs["agentview_image"][sel][:, :, ::-1].copy()).permute(0, 3, 1, 2))
        i2 = tf(torch.tensor(
            obs["robot0_eye_in_hand_image"][sel][:, :, ::-1].copy()
        ).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for i in range(len(sel)):
            m = prompt_template(
                state[i], None, task_desc,
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        batch = proc(text=texts, images=[[image[i].numpy()]
                                         for i in range(len(sel))],
                     return_tensors="pt", padding=True, padding_side="left",
                     action_processor_kwargs={"embodiment_ids": 0})
        batch = dict_apply(lambda x: x.to(dev, dtype), batch)
        with torch.no_grad():
            toks = model.generate(**batch, position_offset=pos_off,
                                  do_sample=False, initial_position_shift=1)
        t = toks.cpu().numpy().reshape(-1, N_LEVEL, N_POS)[:, 0, :]
        return decode(t)

    def scale(a):
        out = np.copy(np.asarray(a, np.float64))
        out[..., :-1] = out[..., :-1] * max_act_q[..., :-1]
        out[..., -1] = -out[..., -1]
        return out

    roll_seed = rollout_seed(args.seed, args.init_start,
                             args.rollout_seed_mode)
    mon = Monitor(args.eps_pos, args.eps_rot, args.max_phase, args.min_phase)
    ev_params = dict(speed_frac=args.speed_frac, min_dwell=args.min_dwell,
                     min_travel=args.min_travel, merge_tol=args.merge_tol)
    if args.event == "grip":
        ev_params = dict(merge_tol=args.merge_tol)

    print(f"=== suite {args.task_suite}, задача {args.task_id}, "
          f"офсет {pos_off}")
    print(f"    «{task_desc}»  рука={args.arm}, режим={args.mode}, "
          f"H={args.horizon}, сред={args.n_envs}, сид {roll_seed}")
    print("    ансамбль ВЫКЛЮЧЕН принудительно у всех рук")

    def rollout():
        seed_everything(roll_seed)
        n = args.n_envs
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
        init_hash = [_h([obs["state"][i].ravel(),
                         obs["agentview_image"][i].ravel() / 255.0])
                     for i in range(n)]
        init_hash_full = [_h([obs["state"][i].ravel(),
                              obs["agentview_image"][i].ravel() / 255.0,
                              obs["robot0_eye_in_hand_image"][i].ravel() / 255.0])
                          for i in range(n)]

        # ЦЕЛИ ОБЯЗАНЫ ОТНОСИТЬСЯ К ЭТИМ ЖЕ НАЧАЛЬНЫМ СОСТОЯНИЯМ. Совпадения
        # task_id и init_start мало: другой сид, другая версия среды или
        # другой чекпойнт дали бы те же номера при других сценах, и рука
        # ехала бы к целям из чужой раскатки.
        if rec is not None:
            bad = [i for i in range(n)
                   if rec["goals"][i]["init_hash_full"] != init_hash_full[i]]
            if bad:
                raise SystemExit(
                    f"начальные состояния {bad} не совпадают с файлом целей: "
                    f"цели записаны для другой сцены, сравнение недействительно")

        # СОСТОЯНИЕ ФАЗЫ ПОЭПИЗОДНОЕ: среды идут в одном батче, но события у
        # них свои, и переключаются они вразнобой. Общий счётчик фазы слил бы
        # расписания и превратил бы событийную руку в опору со случайным H.
        phase_i = np.zeros(n, np.int64)      # номер текущей цели
        phase_t = np.zeros(n, np.int64)      # шагов в текущей фазе
        chunk = np.zeros((n, args.horizon, 7))
        chunk_t = np.full(n, args.horizon, np.int64)
        last_act = np.zeros((n, 7))
        prev_state = None
        calls = steps = 0
        traj = [dict(state=[], action=[]) for _ in range(n)]
        switches = np.zeros(n, np.int64)
        timeouts = np.zeros(n, np.int64)

        while not np.all(done) and steps < args.max_steps:
            st_raw = process_state(obs["state"])
            need = ~done & (chunk_t >= args.horizon)
            if args.arm != "coarse24":
                # Событийные руки зовут VLA только в начале фазы; внутри
                # фазы чанк не обновляется.
                need = ~done & (phase_t == 0)
            sel = np.where(need)[0]
            if len(sel):
                chunk[sel] = call_vla(obs, sel)
                chunk_t[sel] = 0
                # ОДИН БАТЧЕВЫЙ ВЫЗОВ — ОДНА ЕДИНИЦА, сколько бы сред в него
                # ни вошло: столько же стоит проход башни. У опоры среды
                # синхронны, у событийных рук расходятся, и рука, у которой
                # фазы разъехались, честно платит вызовом почти на каждом шаге.
                calls += 1
            act = np.zeros((n, 7))
            for i in range(n):
                if done[i]:
                    continue
                if chunk_t[i] < args.horizon:
                    act[i] = chunk[i, chunk_t[i]]
                    chunk_t[i] += 1
                elif args.arm == "copyprev":
                    act[i] = last_act[i]
                elif args.arm == "servo":
                    g = rec["goals"][i]["goal_pose"]
                    gp = g[min(int(phase_i[i]), len(g) - 1)]
                    act[i] = servo_action(st_raw[i], gp,
                                          rec["servo_kp"], rec["servo_kr"],
                                          last_act[i, 6])
                elif args.arm == "learned":
                    g = rec["goals"][i]["goal_pose"]
                    gp = g[min(int(phase_i[i]), len(g) - 1)]
                    gg = rec["goals"][i]
                    j = min(int(phase_i[i]), len(gg["goal_pose"]) - 1)
                    x = gd.online_features(
                        st_raw[i],
                        None if prev_state is None else prev_state[i],
                        None if prev_state is None else last_act[i],
                        gp, gg["goal_sign"][j], gg["goal_type"][j],
                        ctrl_meta["parts"], state_norm=norm_state,
                        task_id=task_index, n_task=ctrl_meta["n_task"],
                        remaining=(args.max_phase - int(phase_t[i]))
                        if "remaining" in ctrl_meta["parts"] else None)
                    with torch.no_grad():
                        p, gl = ctrl(torch.as_tensor(
                            x, dtype=torch.float32, device=dev)[None])
                    act[i, :6] = p[0].cpu().numpy()
                    act[i, 6] = 1.0 if float(gl[0]) > 0 else -1.0
                else:
                    act[i] = chunk[i, -1]
                last_act[i] = act[i]
                traj[i]["state"].append(st_raw[i].tolist())
                traj[i]["action"].append(act[i].tolist())

            obs2, r_, done2, _ = envs.step(scale(act))
            prev_state = st_raw
            obs, done = obs2, done2
            reward = np.clip(reward + r_, 0, 1)
            steps += 1
            phase_t += (~done).astype(np.int64)

            if args.arm in EVENT_ARMS:
                for i in range(n):
                    if done[i]:
                        continue
                    g = rec["goals"][i]["goal_pose"]
                    gp = g[min(int(phase_i[i]), len(g) - 1)]
                    s2 = process_state(obs["state"])[i]
                    ok, why = mon.reached(s2[:3], s2[3:6], gp[:3], gp[3:6],
                                          int(phase_t[i]))
                    if ok:
                        phase_i[i] += 1
                        phase_t[i] = 0
                        switches[i] += 1
                        timeouts[i] += (why == "таймаут")

        eps = [dict(success=bool(reward[i] >= 1.0), env_steps=steps,
                    policy_calls=calls, init_state_id=args.init_start + i,
                    env_index=i, init_hash=init_hash[i],
                    init_hash_full=init_hash_full[i],
                    rollout_seed=roll_seed,
                    phase_switches=int(switches[i]),
                    phase_timeouts=int(timeouts[i]))
               for i in range(n)]
        return eps, traj

    t0 = time.time()
    try:
        eps, traj = rollout()
    finally:
        try:
            envs.close()
        except Exception:
            pass

    if args.expect_match:
        with open(args.expect_match) as f:
            ref = json.load(f)
        by = {(e["init_state_id"], e["init_hash_full"]): e
              for e in ref["episodes"]}
        diff = []
        for e in eps:
            k = (e["init_state_id"], e["init_hash_full"])
            if k not in by:
                diff.append(f"состояние {e['init_state_id']}: нет пары или "
                            f"начальный хеш другой")
            elif by[k]["success"] != e["success"]:
                diff.append(f"состояние {e['init_state_id']}: "
                            f"{by[k]['success']} против {e['success']}")
        if diff:
            raise SystemExit(
                "опора этого скрипта РАСХОДИТСЯ с ячейкой K-9h:\n  "
                + "\n  ".join(diff[:10])
                + "\nдублирование сборки сред разошлось, пары недействительны")
        print(f"  опора совпала с K-9h эпизод в эпизод ({len(eps)} из "
              f"{len(eps)})")

    s = summarize(eps)
    print(f"\n  рука {args.arm}: успех {s['success_rate']:.1%} "
          f"({sum(e['success'] for e in eps)}/{len(eps)}), "
          f"вызовов {s['policy_calls']}, шагов {s['env_steps']}, "
          f"ВЫЗОВОВ НА ШАГ {s['calls_per_action']:.4f} "
          f"({s['steps_per_call']:.2f} шага на вызов)")
    if args.arm in EVENT_ARMS:
        sw = sum(e["phase_switches"] for e in eps)
        to = sum(e["phase_timeouts"] for e in eps)
        print(f"  переключений фаз {sw}, из них по таймауту {to} "
              f"({to / max(sw, 1):.0%})")
        print("  ЧИТАТЬ ВМЕСТЕ: высокая доля таймаутов означает, что монитор "
              "цель не находит,\n  и низкая частота вызовов достигнута "
              "простоем, а не управлением.")
    print(f"  время: {(time.time() - t0) / 60:.1f} мин")
    print("\n  ЧИТАТЬ ТАК: не по одному числу. Успех осмыслен только в паре "
          "с рукой\n  coarse24 того же --run-tag через k6h_summarize.py "
          "--field arm_label,\n  и только ВМЕСТЕ с «вызовов на шаг». Цель "
          "оракульная — это верхняя оценка.")

    out = dict(summary=s, episodes=eps, arm_label=args.arm,
               run_tag=args.run_tag, mode=args.mode, arm=args.arm,
               horizon=args.horizon, task_id=args.task_id,
               suite=args.task_suite, pos_offset=pos_off,
               ensemble="off", init_start=args.init_start,
               n_envs=args.n_envs, task_description=task_desc,
               seed=args.seed, rollout_seed=roll_seed,
               rollout_seed_mode=args.rollout_seed_mode,
               ckpt=args.ckpt, controller=ctrl_meta,
               event=args.event, event_params=ev_params,
               monitor=dict(eps_pos=args.eps_pos, eps_rot=args.eps_rot,
                            max_phase=args.max_phase,
                            min_phase=args.min_phase),
               record=os.path.abspath(args.record) if args.record else None,
               goal_events_sha1=hashlib.sha1(
                   open(ge.__file__, "rb").read()).hexdigest()[:12],
               script_sha1=sha, argv=vars(args))

    if args.mode == "record":
        goals = []
        for i, tr in enumerate(traj):
            lab = label_from_rollout(np.asarray(tr["state"]),
                                     np.asarray(tr["action"]),
                                     args.event, ev_params)
            lab["source_success"] = bool(eps[i]["success"])
            lab["init_state_id"] = int(eps[i]["init_state_id"])
            lab["init_hash_full"] = eps[i]["init_hash_full"]
            goals.append(lab)
        good = [i for i, g in enumerate(goals) if g["source_success"]]
        kp = kr = None
        if good:
            kp, kr = fit_servo([traj[i] for i in good],
                               [goals[i] for i in good])
        rp = os.path.abspath(os.path.join(os.path.dirname(root), args.record)) \
            if not os.path.isabs(args.record) else args.record
        os.makedirs(os.path.dirname(rp) or ".", exist_ok=True)
        json.dump(dict(task_id=args.task_id, init_start=args.init_start,
                       n_envs=args.n_envs, event=args.event,
                       event_params=ev_params, goals=goals,
                       servo_kp=kp, servo_kr=kr,
                       n_success=len(good), seed=args.seed,
                       rollout_seed=roll_seed, ckpt=args.ckpt,
                       goal_events_sha1=out["goal_events_sha1"],
                       script_sha1=sha),
                  open(rp, "w"), ensure_ascii=False, indent=1)
        ev_mean = float(np.mean([len(g["events"]) for g in goals]))
        print(f"\n  цели записаны: {rp}")
        print(f"  успешных раскаток {len(good)}/{len(goals)}, событий на "
              f"эпизод {ev_mean:.2f}, регулятор kp={kp}, kr={kr}")
        if len(good) < len(goals):
            print("  ВНИМАНИЕ: событийные руки на неуспешных эпизодах "
                  "сравнивать нельзя — load_record откажется.")

    if args.out:
        op = os.path.abspath(os.path.join(os.path.dirname(root), args.out)) \
            if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
        json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {op}  (sha {sha})")


if __name__ == "__main__":
    main()
