"""K-11e: протокол прогона и сверка каждой ячейки с ним.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Раннер пропускал готовые ячейки по одному условию —
файл непустой. Тогда возможен такой случай: протокол утверждает голову с sha
NEW, а в `cells/` лежат все ячейки `hicora_s0`, посчитанные головой OLD. Файлы
непустые и пропускаются, внутри метки отпечаток единообразен, агрегатор
sha ячеек с протоколом не сверяет — и прогон выдал бы правдоподобный вердикт
НЕ ДЛЯ ЗАРЕГИСТРИРОВАННОЙ головы. Фраза «продолжить другим чекпойнтом не
выйдет» была верна только для НОВЫХ ячеек.

Здесь протокол пишется атомарно до первой ячейки, а каждая ячейка —
и пропускаемая, и только что посчитанная — сверяется с ним по всем полям,
определяющим, что именно исполнялось.

ПУСТОЙ ПРОТОКОЛ ПРИ НЕПУСТЫХ ЯЧЕЙКАХ — ОТКАЗ, а не запись задним числом:
иначе протокол подстроился бы под то, что уже посчитано.

Запуск:
    python3 experiments/k11e_protocol.py --selftest
    python3 experiments/k11e_protocol.py init --proto ... --ckpt ... ...
    python3 experiments/k11e_protocol.py check --proto ... --cell ... \\
        --arm hicora_s0 --task 0 --block 0
"""

import argparse
import hashlib
import json
import os
import sys

# Метка -> политика. Та же карта, что в агрегаторе; дублируется намеренно,
# чтобы раннер не зависел от импорта модуля со статистикой.
ARM_POLICY = {"fullbar": "fullbar", "coarse24": "coarse24",
              "joint12": "fast", "hicora_s0": "hicora",
              "hicora_s1": "hicora"}
# Поля головы, которые у s0 и s1 обязаны СОВПАДАТЬ. Различаться дозволено
# только сиду, выбранной эпохе и самим весам.
HEAD_SAME = ("arch", "target", "rank", "cache", "lr", "wd", "chan_weights",
             "script_sha1", "basis_sha1", "rho_sha1", "res_norm_sha1")


def sha12(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:12]


def head_config(obj):
    """Конфигурация головы из чекпойнта, без весов."""
    return {k: (None if obj.get(k) is None else
                (list(obj[k]) if isinstance(obj.get(k), (list, tuple))
                 else obj[k]))
            for k in HEAD_SAME + ("seed", "selected_epoch")}


def check_replication(c0, c1):
    """s0 и s1 обязаны различаться ТОЛЬКО сидом (и следствиями обучения).

    Прежде сверялись базис, предел, норма, ранг и мишень, но НЕ параметры
    обучения: две головы, обученные разными скоростью или затуханием, прошли
    бы как «репликация по сиду», и требование «выполнить на обоих» ничего бы
    не удостоверяло.
    """
    bad = []
    for k in HEAD_SAME:
        if k not in c0 or k not in c1:
            bad.append(f"{k}: отсутствует")
        elif str(c0[k]) != str(c1[k]):
            bad.append(f"{k}: {c0[k]!r} против {c1[k]!r}")
    for c, want in ((c0, 0), (c1, 1)):
        if c.get("seed") is None:
            bad.append("seed: отсутствует")
        elif int(c["seed"]) != want:
            bad.append(f"seed: {c['seed']} вместо {want}")
    if bad:
        raise SystemExit(
            "ГОЛОВЫ s0 И s1 РАЗЛИЧАЮТСЯ НЕ ТОЛЬКО СИДОМ:\n    "
            + "\n    ".join(bad)
            + "\n  Тогда это не репликация, и «выполнить на обоих» ничего не "
              "удостоверяет.")
    return True


def build_protocol(ckpt, joint, h0, h1, cfg, load=None):
    """Протокол прогона. `load` — способ прочитать чекпойнт (для тестов)."""
    if load is None:
        import torch

        def load(p):
            return torch.load(p, map_location="cpu", weights_only=False)
    s0, s1 = sha12(h0), sha12(h1)
    if s0 == s1:
        raise SystemExit("головы s0 и s1 — один и тот же файл: это не "
                         "репликация по сиду")
    c0, c1 = head_config(load(h0)), head_config(load(h1))
    check_replication(c0, c1)
    p = dict(ckpt=ckpt, joint_sha1=sha12(joint),
             head_s0_sha1=s0, head_s1_sha1=s1, head_config=c0)
    p.update(cfg)
    return p


def verify_protocol(old, cur):
    """Расхождение с записанным — отказ, а не перезапись."""
    diff = [k for k in cur if str(old.get(k)) != str(cur[k])]
    if diff:
        raise SystemExit(
            f"ПРОТОКОЛ РАСХОДИТСЯ С ЗАПИСАННЫМ по полям {diff}.\n  "
            f"Продолжать прогон другой конфигурацией нельзя: часть ячеек "
            f"была бы посчитана иначе,\n  а пропуск готовых это скрыл бы.")
    return True


def check_cell(cell, proto, arm, task, block, script_sha=None):
    """Ячейка обязана соответствовать протоколу. Отсутствие поля — отказ."""
    bad = []

    def eq(got, want, name):
        if got is None:
            bad.append(f"{name}: нет поля")
        elif str(got) != str(want):
            bad.append(f"{name}: {got!r}, ожидалось {want!r}")

    eq(cell.get("arm_label"), arm, "arm_label")
    eq(cell.get("policy"), ARM_POLICY.get(arm), "policy")
    eq(cell.get("run_tag"), proto.get("run_tag"), "run_tag")
    eq(cell.get("ckpt"), proto.get("ckpt"), "ckpt")
    eq(cell.get("task_id"), task, "task_id")
    eq(cell.get("init_start"), block, "init_start")
    for f, pk in (("n_envs", "n_envs"), ("ensemble", "ensemble"),
                  ("suite", "suite"),
                  ("horizon", "horizon"), ("max_steps", "max_steps"),
                  ("waiting_steps", "waiting_steps"),
                  ("rollout_seed_mode", "rollout_seed_mode"),
                  ("device", "device")):
        if pk in proto:
            eq(cell.get(f), proto[pk], f)
    if script_sha is not None:
        eq(cell.get("script_sha1"), script_sha, "script_sha1")
    j = cell.get("joint")
    if arm in ("joint12", "hicora_s0", "hicora_s1"):
        if not isinstance(j, dict):
            bad.append("joint: нет словаря происхождения")
        else:
            eq(j.get("weights_sha1"), proto.get("joint_sha1"),
               "joint.weights_sha1")
            if arm.startswith("hicora_"):
                key = "head_s0_sha1" if arm.endswith("_s0") else "head_s1_sha1"
                eq(j.get("hicora_sha1"), proto.get(key), "joint.hicora_sha1")
                if not j.get("arm_fingerprint"):
                    bad.append("joint.arm_fingerprint: нет")
    if bad:
        raise SystemExit(
            f"ЯЧЕЙКА НЕ СООТВЕТСТВУЕТ ПРОТОКОЛУ ({arm}, задача {task}, "
            f"блок {block}):\n    " + "\n    ".join(bad)
            + "\n  Пропуск такой ячейки дал бы вердикт не для "
              "зарегистрированной конфигурации.")
    return True


def selftest():
    cfg = dict(run_tag="k11e", pairs=400, tasks=10, n_envs=10, ensemble="on",
               horizon=8, max_steps=600, waiting_steps=10,
               rollout_seed_mode="block", device="cuda:0", suite="10")
    base_h = dict(arch="mlp", target="coef", rank=32, cache="data/c",
                  lr=0.001, wd=0.0, chan_weights=None, script_sha1="sc",
                  basis_sha1="bs", rho_sha1="rh", res_norm_sha1="rn")
    c0 = dict(base_h, seed=0, selected_epoch=4)
    c1 = dict(base_h, seed=1, selected_epoch=4)
    assert check_replication(c0, c1)
    # ПАРАМЕТРЫ ОБУЧЕНИЯ ТОЖЕ СВЕРЯЮТСЯ: прежде две головы с разными lr/wd
    # проходили как репликация по сиду.
    for k, v in (("lr", 1e-4), ("wd", 0.01), ("chan_weights", [1, 1, 0.1]),
                 ("arch", "uncond"), ("cache", "data/other"),
                 ("script_sha1", "иной"), ("target", "star"), ("rank", 16)):
        try:
            check_replication(c0, dict(c1, **{k: v}))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"расхождение по {k} принято")
    # СОВМЕСТНОЕ ОТСУТСТВИЕ ПОЛЯ больше не проходит: раньше проверялось лишь
    # несовпадение, и две головы без полей вовсе считались репликацией.
    for k in HEAD_SAME:
        try:
            check_replication({x: v for x, v in c0.items() if x != k},
                              {x: v for x, v in c1.items() if x != k})
        except SystemExit:
            pass
        else:
            raise AssertionError(f"совместное отсутствие {k} принято")
    # сиды обязаны быть именно 0 и 1, а не просто разными
    for a, b in ((1, 0), (2, 3), (0, 0)):
        try:
            check_replication(dict(c0, seed=a), dict(c1, seed=b))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"сиды {a}/{b} приняты")

    proto = dict(ckpt="A/B", joint_sha1="wj", head_s0_sha1="h0",
                 head_s1_sha1="h1", head_config=c0, **cfg)
    assert verify_protocol(proto, dict(proto))
    for k in ("ckpt", "joint_sha1", "head_s0_sha1", "pairs", "device"):
        try:
            verify_protocol(dict(proto, **{k: "иное"}), proto)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"расхождение протокола по {k} принято")

    cell = dict(arm_label="hicora_s0", policy="hicora", run_tag="k11e",
                ckpt="A/B", task_id=0, init_start=0, n_envs=10, ensemble="on",
                horizon=8, max_steps=600, waiting_steps=10, suite="10",
                rollout_seed_mode="block", device="cuda:0",
                script_sha1="k9h", joint=dict(weights_sha1="wj",
                                              hicora_sha1="h0",
                                              arm_fingerprint="fp"))
    assert check_cell(cell, proto, "hicora_s0", 0, 0, script_sha="k9h")
    # ИМЕННО ВОСПРОИЗВЕДЁННЫЙ СЛУЧАЙ: ячейка от СТАРОЙ головы лежит на диске
    # и прежде молча пропускалась.
    try:
        check_cell(dict(cell, joint=dict(cell["joint"], hicora_sha1="СТАРАЯ")),
                   proto, "hicora_s0", 0, 0)
    except SystemExit as ex:
        assert "hicora_sha1" in str(ex), str(ex)
    else:
        raise AssertionError("ячейка от другой головы пропущена")
    for kw, why in ((dict(policy="fast"), "подмена политики"),
                    (dict(run_tag="иной"), "чужой эксперимент"),
                    (dict(ckpt="X/Y"), "другой базовый чекпойнт"),
                    (dict(task_id=1), "другая задача"),
                    (dict(init_start=10), "другой блок"),
                    (dict(n_envs=5), "другое число сред"),
                    (dict(ensemble="off"), "другой ансамбль"),
                    (dict(device="cuda:1"), "другая карта"),
                    (dict(suite="goal"), "другой набор задач"),
                    (dict(horizon=4), "другой горизонт"),
                    (dict(script_sha1="иной"), "другая версия стенда"),
                    (dict(joint=dict(weights_sha1="иные", hicora_sha1="h0",
                                     arm_fingerprint="fp")), "другой Joint12"),
                    (dict(joint=dict(weights_sha1="wj", hicora_sha1="h0")),
                     "нет отпечатка руки"),
                    (dict(joint=None), "нет происхождения")):
        try:
            check_cell(dict(cell, **kw), proto, "hicora_s0", 0, 0,
                       script_sha="k9h")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"ячейка принята: {why}")
    # отсутствие любого сверяемого поля — отказ, а не пропуск
    for k in ("arm_label", "policy", "run_tag", "ckpt", "task_id",
              "init_start", "n_envs", "ensemble", "device"):
        try:
            check_cell({x: v for x, v in cell.items() if x != k}, proto,
                       "hicora_s0", 0, 0)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"ячейка без {k} принята")
    # опорные руки происхождения Joint12 не несут и не обязаны
    assert check_cell(dict(cell, arm_label="coarse24", policy="coarse24",
                           joint=None), proto, "coarse24", 0, 0)

    print("самопроверка k11e_protocol пройдена: репликация требует совпадения "
          "параметров обучения и сидов ровно 0 и 1, совместное отсутствие "
          "поля не проходит, ячейка сверяется с протоколом по всем полям "
          "исполнения, ячейка от другой головы или другого Joint12 "
          "отвергается, отсутствие любого поля — отказ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    pi = sub.add_parser("init")
    pi.add_argument("--proto", required=True)
    pi.add_argument("--ckpt", required=True)
    pi.add_argument("--joint", required=True)
    pi.add_argument("--h0", required=True)
    pi.add_argument("--h1", required=True)
    pi.add_argument("--cells", required=True)
    pi.add_argument("--cfg", required=True, help="JSON с полями протокола")
    pc = sub.add_parser("check")
    pc.add_argument("--proto", required=True)
    pc.add_argument("--cell", required=True)
    pc.add_argument("--arm", required=True)
    pc.add_argument("--task", type=int, required=True)
    pc.add_argument("--block", type=int, required=True)
    pc.add_argument("--script-sha", default=None)
    args = ap.parse_args()

    if args.selftest or args.cmd is None:
        selftest()
        return
    selftest()

    if args.cmd == "init":
        cfg = json.loads(args.cfg)
        cur = build_protocol(args.ckpt, args.joint, args.h0, args.h1, cfg)
        if os.path.exists(args.proto):
            verify_protocol(json.load(open(args.proto)), cur)
            print(f"  протокол сверен с {args.proto}")
        else:
            # ЗАПИСЬ ЗАДНИМ ЧИСЛОМ ЗАПРЕЩЕНА: иначе протокол подстроился бы
            # под то, что уже посчитано.
            have = [f for f in os.listdir(args.cells)
                    if f.endswith(".json")] if os.path.isdir(args.cells) else []
            if have:
                raise SystemExit(
                    f"нет {args.proto}, но в {args.cells} уже {len(have)} "
                    f"ячеек. Ставить протокол задним числом нельзя: он "
                    f"подстроился бы под посчитанное. Уберите каталог или "
                    f"восстановите протокол.")
            tmp = args.proto + ".tmp"
            json.dump(cur, open(tmp, "w"), ensure_ascii=False, indent=1)
            os.replace(tmp, args.proto)
            print(f"  протокол записан: {args.proto}")
        print(f"    Joint12 {cur['joint_sha1']}, головы {cur['head_s0_sha1']} "
              f"и {cur['head_s1_sha1']}, пар {cur.get('pairs')}, задач "
              f"{cur.get('tasks')}, карта {cur.get('device')}")
        return

    if args.cmd == "check":
        check_cell(json.load(open(args.cell)), json.load(open(args.proto)),
                   args.arm, args.task, args.block, args.script_sha)
        return


if __name__ == "__main__":
    main()
