"""K-11e: проверка готовности машины к прогону гейта.

ЗАЧЕМ. Прогон идёт 7-10 часов. Каждая причина, по которой он упадёт на
третьем часу, проверяется за полминуты до запуска: не тот файл весов, не
скопированы базис и предел, не работает EGL, задана CUDA_VISIBLE_DEVICES,
не скачан чекпойнт, забит диск. Отдельный смысл на ВТОРОЙ машине: убедиться,
что репозиторий и веса те же самые, иначе два прогона несопоставимы даже
между собой.

Проверки идут ВСЕ, а не до первой ошибки: за один запуск видно весь список
недостающего. Код возврата 1, если хоть одна обязательная не прошла.

Запуск:
    python experiments/k11e_preflight.py                       # основной
    python experiments/k11e_preflight.py --root data/k11e_b --tag k11e_b
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

# SHA рабочей копии, на которой прогон зарегистрирован. На второй машине
# расхождение означает другой код, и складывать такие прогоны нельзя.
EXPECT_SHA = {
    "experiments/k9h_multiarm_gate.py": "c9235b6cc1f8",
    "experiments/k6h_summarize.py": "a52cb228f16b",
    "experiments/k11e_protocol.py": "764e8d5775a5",
    "experiments/run_k11e_gate.sh": "c90ab216fec7",
    "experiments/k11e_precheck.py": "b7199e6ff39c",
}
CKPT = "ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO"
MIN_FREE_GB = 5.0


class Log:
    def __init__(self):
        self.bad, self.warn = [], []

    def ok(self, name, detail=""):
        print(f"  [ да ] {name}" + (f": {detail}" if detail else ""))

    def no(self, name, detail):
        print(f"  [НЕТ ] {name}: {detail}")
        self.bad.append(f"{name}: {detail}")

    def soft(self, name, detail):
        print(f"  [ ?  ] {name}: {detail}")
        self.warn.append(f"{name}: {detail}")


def sha12(p):
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:12]


def check_repo(L, allow_drift):
    print("\n--- репозиторий ---")
    for rel, want in EXPECT_SHA.items():
        if not os.path.exists(rel):
            L.no(rel, "файла нет")
            continue
        got = sha12(rel)
        if got == want:
            L.ok(rel, got)
        elif allow_drift:
            L.soft(rel, f"{got}, ожидалось {want}")
        else:
            L.no(rel, f"{got}, ожидалось {want} — другая версия кода")
    for m in ("hicora_vla.py", "joint12_vla.py"):
        p = os.path.join("experiments", m)
        p = p if os.path.exists(p) else m
        if os.path.exists(p):
            L.ok(f"модуль {m}", sha12(p))
        else:
            L.no(f"модуль {m}", "не найден")


def check_env(L):
    print("\n--- окружение ---")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        L.no("CUDA_VISIBLE_DEVICES",
             f"задана ({os.environ['CUDA_VISIBLE_DEVICES']}) — robosuite "
             f"выведет из неё MUJOCO_EGL_DEVICE_ID и EGL упадёт на каждой "
             f"ячейке. Снять: unset CUDA_VISIBLE_DEVICES")
    else:
        L.ok("CUDA_VISIBLE_DEVICES", "не задана, как и требуется")
    lib = os.path.expanduser("~/LIBERO")
    if os.path.isdir(lib):
        L.ok("каталог LIBERO", lib)
    else:
        L.no("каталог LIBERO", f"нет {lib}")
    free = shutil.disk_usage(".").free / 2 ** 30
    (L.ok if free >= MIN_FREE_GB else L.no)(
        "свободно на диске", f"{free:.1f} ГБ"
        + ("" if free >= MIN_FREE_GB else f" — нужно хотя бы {MIN_FREE_GB}"))


def check_torch(L, dev):
    print("\n--- карты ---")
    try:
        import torch
    except Exception as e:
        L.no("импорт torch", repr(e))
        return
    L.ok("torch", torch.__version__)
    if not torch.cuda.is_available():
        L.no("CUDA", "torch.cuda.is_available() == False. Если драйвер жив, "
                     "это обычно потеря карт контейнером: перезапустить "
                     "докер с хоста")
        return
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    L.ok("карт видно", f"{n}: {', '.join(names)}")
    idx = int(dev.split(":")[1]) if ":" in dev else 0
    if idx >= n:
        L.no("карта прогона", f"{dev} нет, всего карт {n}")
    else:
        free_b, tot_b = torch.cuda.mem_get_info(idx)
        L.ok(f"память на {dev}",
             f"свободно {free_b / 2 ** 30:.1f} из {tot_b / 2 ** 30:.1f} ГБ")


def check_weights(L, joint, h0, h1):
    print("\n--- веса ---")
    for p in (joint, h0, h1):
        if not os.path.exists(p):
            L.no(p, "файла нет")
    if any(not os.path.exists(p) for p in (joint, h0, h1)):
        return None
    L.ok("черновик Joint12", f"{joint}, sha {sha12(joint)}")
    try:
        import torch
    except Exception as e:
        L.no("чтение голов", repr(e))
        return None
    objs = {}
    for tag, p in (("s0", h0), ("s1", h1)):
        try:
            objs[tag] = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            L.no(f"чтение головы {tag}", repr(e))
            return None
        L.ok(f"голова {tag}", f"{os.path.basename(p)}, sha {sha12(p)}, "
                              f"сид {objs[tag].get('seed')}, эпоха "
                              f"{objs[tag].get('selected_epoch')}")
    if sha12(h0) == sha12(h1):
        L.no("репликация", "s0 и s1 — один и тот же файл")
    try:
        sys.path.insert(0, "experiments")
        import k11e_protocol as kp
        kp.check_replication(kp.head_config(objs["s0"]),
                             kp.head_config(objs["s1"]))
        L.ok("репликация s0/s1", "различаются только сидом")
    except SystemExit as e:
        L.no("репликация s0/s1", str(e).replace("\n", " ")[:200])
    except Exception as e:
        L.no("репликация s0/s1", repr(e))
    return objs


def check_cache(L, objs):
    """Базис, предел и meta кэша обязаны лежать рядом и совпасть по sha.

    Сам кэш скрытых состояний (крупный .npz) на раскатке НЕ нужен — только
    эти три файла. На вторую машину достаточно скопировать их.
    """
    print("\n--- кэш головы (базис, предел, meta) ---")
    if not objs:
        L.no("кэш", "головы не прочитаны, проверять нечего")
        return
    for tag, o in objs.items():
        pref = o.get("cache")
        if not pref:
            L.no(f"кэш {tag}", "в чекпойнте нет поля cache")
            continue
        for suf, key, nm in ((".basis.npy", "basis_sha1", "базис"),
                             (".rho.npy", "rho_sha1", "предел"),
                             (".meta.json", None, "meta")):
            p = pref + suf
            if not os.path.exists(p):
                L.no(f"{nm} {tag}", f"нет {p}")
                continue
            if key is None:
                L.ok(f"{nm} {tag}", p)
                continue
            got, want = sha12(p), o.get(key)
            if got == want:
                L.ok(f"{nm} {tag}", f"{got}")
            else:
                L.no(f"{nm} {tag}", f"sha {got}, голова обучена на {want}")


def check_libero(L):
    print("\n--- LIBERO и EGL ---")
    lib = os.path.expanduser("~/LIBERO")
    code = (
        "import os,sys;sys.path.insert(0,%r);os.environ['MUJOCO_GL']='egl';"
        "import mujoco;"
        "m=mujoco.MjModel.from_xml_string("
        "'<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>');"
        "d=mujoco.MjData(m);r=mujoco.Renderer(m,64,64);"
        "mujoco.mj_forward(m,d);r.update_scene(d);r.render();"
        "import libero;print('OK',mujoco.__version__)" % lib)
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=180)
    except Exception as e:
        L.no("EGL", repr(e))
        return
    if out.returncode == 0 and "OK" in out.stdout:
        L.ok("EGL и импорт libero", out.stdout.strip().splitlines()[-1])
    else:
        tail = (out.stderr or out.stdout).strip().splitlines()
        L.no("EGL или импорт libero",
             " | ".join(tail[-3:]) if tail else f"код {out.returncode}")


def check_hf(L, ckpt):
    print("\n--- чекпойнт модели ---")
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception as e:
        L.soft("кэш HuggingFace", f"не определить: {e!r}")
        return
    d = os.path.join(HF_HUB_CACHE, "models--" + ckpt.replace("/", "--"))
    if os.path.isdir(d):
        n = len(glob.glob(os.path.join(d, "snapshots", "*", "*")))
        L.ok("чекпойнт в кэше", f"{d} ({n} файлов в снимке)")
    else:
        L.soft("чекпойнт в кэше",
               f"нет {d}. Скачается при первом запуске — если у машины есть "
               f"сеть. Без сети прогон упадёт на первой ячейке")


def check_root(L, root, tag):
    print(f"\n--- каталог прогона {root} ---")
    cells = os.path.join(root, "cells")
    proto = os.path.join(root, "protocol.json")
    n = len(glob.glob(os.path.join(cells, "*.json")))
    if os.path.exists(proto):
        p = json.load(open(proto))
        if p.get("run_tag") != tag:
            L.no("протокол", f"в нём run_tag {p.get('run_tag')!r}, а запуск "
                             f"под {tag!r}")
        else:
            L.ok("протокол", f"есть, run_tag {tag}, ячеек {n} — прогон "
                             f"продолжится, готовые будут сверены")
    elif n:
        L.no("каталог", f"{n} ячеек без protocol.json. Раннер откажется "
                        f"ставить протокол задним числом — либо перенести "
                        f"ячейки, либо взять другой ROOT")
    else:
        L.ok("каталог", "чист, прогон начнётся с нуля")
    if tag == "k11e" and root != "data/k11e":
        L.soft("тег", f"root {root}, а тег k11e — на второй машине берите "
                      f"TAG={os.path.basename(root)}, иначе два прогона "
                      f"нельзя будет различить в агрегаторе")


def check_selftests(L):
    print("\n--- самопроверки модулей ---")
    for m in ("k9h_multiarm_gate.py", "k6h_summarize.py",
              "k11e_protocol.py", "k11e_precheck.py"):
        p = os.path.join("experiments", m)
        if not os.path.exists(p):
            L.no(f"самопроверка {m}", "файла нет")
            continue
        r = subprocess.run([sys.executable, p, "--selftest"],
                           capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            L.ok(f"самопроверка {m}", "пройдена")
        else:
            tail = (r.stderr or r.stdout).strip().splitlines()
            L.no(f"самопроверка {m}", " | ".join(tail[-2:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/k11e")
    ap.add_argument("--tag", default="k11e")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--joint", default="data/k9d_ep3.pt")
    ap.add_argument("--h0", default="data/k11d/d1_mlp_coef_0.001_wd0_s0.pt")
    ap.add_argument("--h1", default="data/k11d/d1_mlp_coef_0.001_wd0_s1.pt")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--allow-sha-drift", action="store_true",
                    help="считать расхождение версий кода предупреждением. "
                         "Тогда два прогона сопоставимы только внутри себя")
    ap.add_argument("--skip-slow", action="store_true",
                    help="пропустить EGL и самопроверки (быстрый осмотр)")
    a = ap.parse_args()

    print(f"ГОТОВНОСТЬ К K-11e: {os.getcwd()}")
    L = Log()
    check_repo(L, a.allow_sha_drift)
    check_env(L)
    check_torch(L, a.device)
    objs = check_weights(L, a.joint, a.h0, a.h1)
    check_cache(L, objs)
    check_hf(L, a.ckpt)
    check_root(L, a.root, a.tag)
    if a.skip_slow:
        print("\n--- EGL и самопроверки пропущены (--skip-slow) ---")
    else:
        check_libero(L)
        check_selftests(L)

    print("\n" + "=" * 66)
    if L.warn:
        print("ПРЕДУПРЕЖДЕНИЯ (прогон возможен, но посмотрите):")
        for w in L.warn:
            print(f"  - {w}")
    if L.bad:
        print("НЕ ГОТОВО. Обязательные проверки не прошли:")
        for b in L.bad:
            print(f"  - {b}")
        raise SystemExit(1)
    print("ГОТОВО. Сначала одна пробная ячейка (две минуты), потом полный "
          "прогон.")
    print(f"  ROOT={a.root} TAG={a.tag} DEV={a.device}")


if __name__ == "__main__":
    main()
