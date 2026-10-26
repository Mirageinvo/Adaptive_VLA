#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 1: общее для подготовки, адаптера и моделей flat/parents.

Решения агента (2026-10-25), закреплённые здесь:
  - 8 задач до обучения (TASKS) со штатными пределами шагов закреплённой
    ревизии RoboTwin; замена задач по полученному SR запрещена;
  - действие — абсолютные суставные цели 14 каналов в порядке официального
    цикла: [левая рука 6, левый схват, правая рука 6, правый схват];
    состояние на входе — те же 14 каналов (суставы и схваты);
  - схваты — непрерывные [0, 1] (открыт > 0.8, закрыт < 0.2 в коде робота);
  - камеры — cam_head, cam_left_wrist, cam_right_wrist (без cam_third_view);
  - чанк 20, исполняются первые 8 команд;
  - исходные веса — BAR-LIBERO закреплённой ревизии, одинаковые для пары.
Адаптер входа повторяет соглашение BAR-LIBERO: ОДНО изображение — склейка
камер рядом (LIBERO: 224×448 из двух камер; здесь каждая камера 240×320 ->
224×224, порядок head | left_wrist | right_wrist, итог 224×672, CHW uint8
RGB) и состояние текстом в промпте (prompt_template BAR), нормированное в
[−1, 1] по квантилям 1%/99% ОБУЧАЮЩИХ эпизодов; то же нормированное
состояние идёт в state_proj модели. Кадры JPEG декодируются официальной
XPolicyLab.utils.process_data.decode_image_bit (RGB при любой разметке).
"""
import hashlib
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TASKS = dict(lift_pot=400, pick_dual_bottles=400, handover_block=800,
             handover_mic=600, place_dual_shoes=600, place_bread_basket=700,
             stack_blocks_two=800, open_laptop=700)
TASK_CONFIG, EMBODIMENT = "demo_clean", "aloha_agilex"
EMBODIMENT_ASSET = "aloha-agilex"
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CH_KEYS = (("left_arm_joint_states", 6), ("left_ee_joint_states", 1),
           ("right_arm_joint_states", 6), ("right_ee_joint_states", 1))
N_CH = 14
GRIP_IDX = (6, 13)
ARM_IDX = tuple(i for i in range(N_CH) if i not in GRIP_IDX)
T_CHUNK, N_EXEC = 20, 8
TILE = 224
POS_OFFSET = 0                     # RoPE-смещение query: одно для всех задач
GRIP_W = 1.0                       # вес L1 схватов в потере
BAR_REVISION = "97c980cec63071f8ff9e166f8efaae176afd9bbf"
DATA_REL = os.path.join(TASK_CONFIG, "{task}", EMBODIMENT, "data")


def sha(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


# --- каналы ---------------------------------------------------------------
def channels(h, group):
    """[T, 14] из группы action/ или state/ HDF5 в порядке CH_KEYS."""
    parts = []
    for k, d in CH_KEYS:
        a = np.asarray(h[f"{group}/{k}"], np.float64)
        a = a.reshape(len(a), -1)
        if a.shape[1] != d:
            raise SystemExit(f"{group}/{k}: размерность {a.shape[1]}, "
                             f"ожидалась {d}")
        parts.append(a)
    return np.concatenate(parts, 1)


def normalize(x, q01, q99):
    """[q01, q99] -> [−1, 1]; канал с нулевым размахом (постоянный в
    обучающих эпизодах) -> 0."""
    q01, q99 = np.asarray(q01, np.float64), np.asarray(q99, np.float64)
    span = q99 - q01
    ok = span > 1e-8
    y = (np.asarray(x, np.float64) - q01) / np.where(ok, span, 1.0) * 2 - 1
    return np.where(ok, y, 0.0)


def chunk_target(act, t, n=T_CHUNK):
    """Цель [n, 14] с шага t и маска действительных позиций (за концом
    эпизода — последнее действие, маска 0)."""
    T = len(act)
    idx = np.minimum(np.arange(t, t + n), T - 1)
    valid = (np.arange(t, t + n) < T)
    return act[idx].astype(np.float32), valid


# --- кадры ----------------------------------------------------------------
def decoder():
    """Официальный декодер XPolicyLab (RGB при любой разметке JPEG)."""
    try:
        from XPolicyLab.utils.process_data import decode_image_bit
    except Exception as e:  # noqa: BLE001
        raise SystemExit("нужен XPolicyLab (pip install -e "
                         "$ROBOTWIN_ROOT/XPolicyLab --no-deps) для "
                         f"decode_image_bit: {e}")
    return decode_image_bit


def tile(frames):
    """[3 × HxWx3 uint8 RGB] -> CHW uint8 [3, 224, 672] (head|left|right)."""
    import cv2
    if len(frames) != len(CAMS):
        raise SystemExit(f"камер {len(frames)}, ожидалось {len(CAMS)}")
    ims = []
    for f in frames:
        f = np.asarray(f)
        if f.ndim != 3 or f.shape[2] != 3 or f.dtype != np.uint8:
            raise SystemExit(f"кадр формы {f.shape} {f.dtype}")
        ims.append(cv2.resize(f, (TILE, TILE), interpolation=cv2.INTER_AREA))
    return np.ascontiguousarray(np.concatenate(ims, 1).transpose(2, 0, 1))


# --- робот ------------------------------------------------------------------
def robot_limits(robotwin_root):
    """Технические пределы 12 суставов рук из URDF тела (aloha-agilex):
    имена — arm_joints_name конфига тела, пределы — <limit> URDF.
    -> (L[14], U[14], описание); схваты — [0, 1]."""
    import yaml
    base = os.path.join(robotwin_root, "assets", "embodiments",
                        EMBODIMENT_ASSET)
    cfg = yaml.safe_load(open(os.path.join(base, "config.yml"),
                              encoding="utf-8"))
    names = cfg["arm_joints_name"]
    urdf = os.path.join(base, cfg["urdf_path"])
    lim = urdf_limits(open(urdf, encoding="utf-8").read())
    L, U = np.zeros(N_CH), np.ones(N_CH)
    arm_names = list(names[0]) + list(names[1])
    if len(arm_names) != len(ARM_IDX):
        raise SystemExit(f"суставов рук {len(arm_names)}, ожидалось 12")
    for i, n in zip(ARM_IDX, arm_names):
        if n not in lim:
            raise SystemExit(f"в URDF нет предела сустава {n}")
        L[i], U[i] = lim[n]
    return L, U, dict(urdf=os.path.relpath(urdf, robotwin_root),
                      urdf_sha1=sha(urdf), arm_joints=arm_names)


def urdf_limits(text):
    out = {}
    for j in ET.fromstring(text).iter("joint"):
        lm = j.find("limit")
        if lm is not None and "lower" in lm.attrib and "upper" in lm.attrib:
            out[j.attrib["name"]] = (float(lm.attrib["lower"]),
                                     float(lm.attrib["upper"]))
    return out


# --- BAR и промпт -------------------------------------------------------------
LIBERO_ONLY = ("gym", "robosuite", "robosuite.utils",
               "robosuite.utils.transform_utils", "libero", "libero.libero",
               "libero2gym", "libero2gym.vector_env")
SHIMMED = []


def shim_libero_imports():
    """actioncodec/scripts/utils.py импортирует на уровне модуля gym,
    robosuite, libero и libero2gym — их используют только функции среды
    LIBERO; prompt_template, get_cfg и процессор BAR от них не зависят. В
    модельном окружении RoboTwin их нет, поэтому — заглушки: подставляются
    ТОЛЬКО при отсутствии настоящего модуля, любое обращение к ним — ошибка
    (тихо использоваться не могут). Список подставленных — SHIMMED."""
    import importlib.machinery
    import importlib.util
    import types

    def dunder(k):
        return k.startswith("__") and k.endswith("__")

    class _Missing:
        def __init__(self, name):
            self._n = name

        def __getattr__(self, k):
            if dunder(k):                 # служебные — как у обычного объекта
                raise AttributeError(k)
            raise RuntimeError(f"заглушка {self._n}.{k}: модуль LIBERO в этом "
                               f"окружении не установлен")

        def __call__(self, *a, **k):
            raise RuntimeError(f"заглушка {self._n}: модуль LIBERO в этом "
                               f"окружении не установлен")
    # наличие настоящих пакетов — один раз, ДО подстановки заглушек
    missing = set()
    for top in {n.split(".")[0] for n in LIBERO_ONLY}:
        if top in sys.modules and top not in SHIMMED:
            continue
        try:
            if importlib.util.find_spec(top) is None:
                missing.add(top)
        except (ImportError, ValueError):
            missing.add(top)
    for name in LIBERO_ONLY:
        if name in sys.modules or name.split(".")[0] not in missing:
            continue
        m = types.ModuleType(name)

        def mod_getattr(k, n=name):
            if dunder(k):                 # __file__, __spec__ и т. п.
                raise AttributeError(k)
            return _Missing(f"{n}.{k}")
        m.__getattr__ = mod_getattr
        m.__path__ = []
        m.__file__ = None
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = m
        SHIMMED.append(name)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], child, m)
    return list(SHIMMED)


def load_bar(device, dtype_name="bfloat16", root="third_party/actioncodec",
             cfg_path="config/eval/bar.yaml"):
    """Закреплённый BAR-LIBERO (как k21_data.load_bar, но с выбором типа:
    штатный путь приводит к fp16). -> dict(model, proc, cfg, revision,
    pin)."""
    import torch
    import k21_data as d21
    pin = d21.snapshot_pin(BAR_REVISION)
    if pin["revision"] != BAR_REVISION:
        raise SystemExit(f"снимок BAR ревизии {pin['revision']}")
    path = pin["path"]
    r = os.path.abspath(os.path.join(os.path.dirname(HERE), root))
    if r not in sys.path:
        sys.path.insert(0, r)
    shim_libero_imports()
    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import VisionLanguageActionProcessor, get_cfg
    cfg = get_cfg(os.path.join(r, cfg_path))
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = path
    dt = getattr(torch, dtype_name)
    model = SmolVLABlockwiseAR.from_pretrained(**cfg.MODEL.vlm.kwargs).to(
        device, dt).eval()
    if type(model) is not SmolVLABlockwiseAR:
        raise SystemExit(f"модель класса {type(model)}")
    for p in model.parameters():
        p.requires_grad_(False)
    proc = VisionLanguageActionProcessor.from_pretrained(
        path, trust_remote_code=True, mode="discrete")
    return dict(model=model, proc=proc, cfg=cfg, revision=pin["revision"],
                pin=pin, dtype=dtype_name)


def prompt_batch(bar, imgs, states_n, texts, device, dtype):
    """Как k20_data.prompt_batch (то же содержание промпта BAR: одно
    изображение-склейка, состояние текстом), но целые тензоры не
    приводятся к float, а вещественные — к dtype."""
    shim_libero_imports()
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party",
                                    "actioncodec", "scripts"))
    import utils as U
    cfg, proc = bar["cfg"], bar["proc"]
    msgs = []
    for s, t in zip(states_n, texts):
        m = U.prompt_template(
            np.asarray(s, np.float64), None, t,
            mode=cfg.MODEL.vla_processor.kwargs.mode,
            action_vocab_size=cfg.MODEL.action_processor.vocab_size,
            action_token_len=cfg.MODEL.action_processor.token_len)
        m[1]["content"] = m[1]["content"][1:]           # одно изображение
        msgs.append(m)
    txt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    b = proc(text=txt, images=[[im] for im in imgs], return_tensors="pt",
             padding=True, padding_side="left",
             action_processor_kwargs={"embodiment_ids": 0})
    out = {}
    for k, v in b.items():
        if hasattr(v, "to"):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out


def selftest():
    a = np.arange(5 * 14, dtype=np.float64).reshape(5, 14)
    tg, v = chunk_target(a, 3, 4)
    assert tg.shape == (4, 14) and v.tolist() == [True, True, False, False]
    assert np.array_equal(tg[2], a[4]) and np.array_equal(tg[0], a[3])
    q01, q99 = np.zeros(14), np.full(14, 2.0)
    assert np.allclose(normalize(np.ones(14), q01, q99), 0)
    assert np.allclose(normalize(np.ones(14), q01, q01), 0)    # нулевой размах
    xml = ('<robot><joint name="a"><limit lower="-1" upper="2"/></joint>'
           '<joint name="b"/></robot>')
    assert urdf_limits(xml) == {"a": (-1.0, 2.0)}
    assert ARM_IDX == (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
    assert sum(TASKS.values()) / len(TASKS) == 625
    try:
        import cv2  # noqa: F401
        fr = [np.full((240, 320, 3), i * 40, np.uint8) for i in range(3)]
        t = tile(fr)
        assert t.shape == (3, 224, 672) and t[0, 0, 0] == 0 and \
            t[0, 0, 300] == 40 and t[0, 0, 500] == 80
    except ImportError:
        print("  (cv2 нет — проверка склейки пропущена)")
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import rt1_common as rc; "
            "sh = rc.shim_libero_imports(); import robosuite.utils."
            "transform_utils as T; from libero.libero import benchmark; "
            "from libero2gym.vector_env import AsyncVectorEnv; import gym\n"
            "try:\n    benchmark.get_benchmark_dict()\n    raise SystemExit(5)"
            "\nexcept RuntimeError:\n    pass\n"
            "import inspect\n"
            "inspect.getmodule(compile('pass', '/tmp/unloaded.py', 'exec'))\n"
            "assert not hasattr(gym, '__wrapped__')\n"
            "assert getattr(gym, '__file__', None) is None\n"
            "import torch\n"
            "try:\n    import transformers\nexcept ImportError:\n    pass\n"
            "print(len(sh))") % HERE
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-500:]
    print("самопроверка rt1_common: заглушек LIBERO-импортов "
          f"{r.stdout.strip()} (0 — настоящие модули установлены); "
          "inspect.getmodule и импорт transformers/torch после заглушек — "
          "без ошибок")
    print("самопроверка rt1_common пройдена: порядок каналов, цель чанка и "
          "маска, нормировка, пределы URDF, склейка камер, средний предел "
          "шагов 625")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
