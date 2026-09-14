#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Перенос результатов K-12i в схему K-12j. Ничего не удаляет и не переписывает.

ЗАЧЕМ. Старый раннер считал те же обновления той же политикой, но помечал
головы на единицу меньше: `head_step1.pt` нёс step_index=0, хотя одно обновление
уже принято. Новая схема требует, чтобы step_index означал ЧИСЛО ПРИНЯТЫХ
обновлений. Пересчитывать по два часа на голову ради одной метки незачем, но и
править на месте нельзя — перенос делает копии в новый каталог и проверяет
цепочку.

ЧТО ПЕРЕНОСИТСЯ
  * head_stepK.pt  -> head_after_000K.pt со step_index=K;
  * train_step(K-1)/ -> train_after_(K-1)/ как есть: их мета уже означает
    «снято политикой после K-1 обновлений», то есть менять там нечего.

ЧТО НЕ ПЕРЕНОСИТСЯ. Оценочные ячейки: в них нет ни run_config, ни повызовных
хэшей шума, добавленных позже, и строгий отчёт их справедливо отвергнет. Они
пересчитываются заново — это десять минут на шаг, а не два часа.
"""
import argparse
import json
import os
import shutil
import sys


def plan(src, dst):
    """Что будет скопировано. Возвращает список (что, откуда, куда)."""
    import re
    out = []
    for p in sorted(os.listdir(src)):
        m = re.fullmatch(r"head_step(\d+)\.pt", p)
        if m:
            k = int(m.group(1))
            out.append(("head", os.path.join(src, p),
                        os.path.join(dst, f"head_after_{k:04d}.pt"), k))
        if re.fullmatch(r"train_step(\d+)", p) and \
                os.path.isdir(os.path.join(src, p)):
            n = int(re.fullmatch(r"train_step(\d+)", p).group(1))
            out.append(("train", os.path.join(src, p),
                        os.path.join(dst, f"train_after_{n}"), n))
    return out


def fill_provenance(dst):
    """Достроить run_config в УЖЕ созданных спутниках.

    Прошлый перенос сделал спутники до того, как появилась эта достройка;
    переносить заново нечего, а без происхождения раннер пересчитывает готовые
    буферы. Функция трогает только спутники и только отсутствующее поле.
    """
    import torch
    done = []
    for d in sorted(os.listdir(dst)):
        if not d.startswith("train_after_"):
            continue
        dd = os.path.join(dst, d)
        if not os.path.isdir(dd):
            continue
        for f in sorted(os.listdir(dd)):
            if not f.endswith(".pt.meta.json"):
                continue
            path = os.path.join(dd, f)
            meta = json.load(open(path))
            if "run_config" in meta:
                continue
            eps = meta.get("episodes") or []
            tasks = sorted({int(e["task_id"]) for e in eps
                            if e.get("task_id") is not None})
            rc = dict(arm="policy", sigma=meta.get("sigma"),
                      step_index=meta.get("step_index"),
                      init_start=meta.get("init_start"),
                      d1_seed=meta.get("d1_seed"),
                      rl_seed=meta.get("rl_seed"))
            if len(tasks) == 1:
                rc["task_id"] = tasks[0]
            meta["run_config"] = {k: v for k, v in rc.items() if v is not None}
            meta["run_config_reconstructed"] = True
            tmp = path + f".tmp.{os.getpid()}"
            json.dump(meta, open(tmp, "w"), ensure_ascii=False, default=str)
            os.replace(tmp, path)
            done.append(path)
    return done


def relabel(dst, replica):
    """Переименовать реплику у уже перенесённых голов, записав прежнее имя.

    Метка реплики участвует в проверке цепочки — она не даёт смешать прогоны.
    Старый раннер писал `smoke_s0`, новый использует `dev_s0_rl0`, и без
    переименования продолжение отказывает по делу. Прежнее имя сохраняется:
    переименование — операция, о которой попросили, а не потеря происхождения.
    """
    import re
    import torch
    done = []
    for f in sorted(os.listdir(dst)):
        if not re.fullmatch(r"head_after_\d{4}\.pt", f):
            continue
        path = os.path.join(dst, f)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("replica") == replica:
            continue
        obj["relabeled_replica_was"] = obj.get("replica")
        obj["replica"] = replica
        tmp = path + f".tmp.{os.getpid()}"
        torch.save(obj, tmp)
        os.replace(tmp, path)
        done.append((f, obj["relabeled_replica_was"], replica))
    return done


def migrate(src, dst, *, apply=False):
    import torch
    os.makedirs(dst, exist_ok=True)
    done, skipped = [], []
    for kind, a, b, k in plan(src, dst):
        if os.path.exists(b):
            skipped.append((b, "уже есть — не трогаю"))
            continue
        if not apply:
            done.append((kind, a, b, k))
            continue
        if kind == "head":
            obj = torch.load(a, map_location="cpu", weights_only=False)
            old = obj.get("step_index")
            obj["step_index"] = int(k)
            obj["migrated_from"] = os.path.abspath(a)
            obj["migrated_step_index_was"] = old
            tmp = b + f".tmp.{os.getpid()}"
            torch.save(obj, tmp)
            os.replace(tmp, b)
        else:
            shutil.copytree(a, b)
            # СПУТНИКИ ДЛЯ СТАРЫХ БУФЕРОВ. Их не было до введения .meta.json, а
            # без них раннер не может сверить происхождение и пересчитал бы
            # готовый буфер заново — полчаса на шаг вместо пары минут чтения.
            for f in sorted(os.listdir(b)):
                if not f.endswith(".pt"):
                    continue
                side = os.path.join(b, f + ".meta.json")
                if os.path.exists(side):
                    continue
                obj = torch.load(os.path.join(b, f), map_location="cpu",
                                 weights_only=False)
                meta = dict(obj.get("meta") or {})
                meta["migrated_sidecar"] = True
                # ДОСТРАИВАЕМ ПРОИСХОЖДЕНИЕ ИЗ ТОГО, ЧТО В МЕТЕ ЕСТЬ. Старые
                # буферы писались до появления run_config, и без него раннер
                # пересчитал бы их заново — полчаса на шаг. Рука здесь известна
                # из назначения каталога: train_* — это обучающая раскатка;
                # задача и состояние берутся из эпизодов, а не из имени файла.
                if "run_config" not in meta:
                    eps = meta.get("episodes") or []
                    tasks = sorted({int(e["task_id"]) for e in eps
                                    if e.get("task_id") is not None})
                    rc = dict(arm="policy", sigma=meta.get("sigma"),
                              step_index=meta.get("step_index"),
                              init_start=meta.get("init_start"),
                              d1_seed=meta.get("d1_seed"),
                              rl_seed=meta.get("rl_seed"))
                    if len(tasks) == 1:
                        rc["task_id"] = tasks[0]
                    meta["run_config"] = {k: v for k, v in rc.items()
                                          if v is not None}
                    meta["run_config_reconstructed"] = True
                tmp = side + f".tmp.{os.getpid()}"
                json.dump(meta, open(tmp, "w"), ensure_ascii=False,
                          default=str)
                os.replace(tmp, side)
        done.append((kind, a, b, k))
    return done, skipped


def verify(dst):
    """Цепочка голов после переноса: номера идут подряд и совпадают с именем."""
    import re
    import torch
    heads = sorted(f for f in os.listdir(dst)
                   if re.fullmatch(r"head_after_\d{4}\.pt", f))
    bad, prev = [], 0
    for f in heads:
        k = int(f[len("head_after_"):-3])
        if k != prev + 1:
            bad.append(f"{f}: разрыв цепочки, ожидался {prev + 1}")
        obj = torch.load(os.path.join(dst, f), map_location="cpu",
                         weights_only=False)
        if int(obj.get("step_index", -1)) != k:
            bad.append(f"{f}: step_index={obj.get('step_index')}, "
                       f"а имя означает {k}")
        if not obj.get("optimizer_state"):
            bad.append(f"{f}: нет состояния Adam — продолжить нельзя")
        prev = k
    if bad:
        raise SystemExit("цепочка после переноса не сходится:\n  - "
                         + "\n  - ".join(bad))
    return len(heads)


def selftest():
    import tempfile
    import torch
    src = tempfile.mkdtemp(prefix="k12i_src_")
    dst = os.path.join(src, "..", os.path.basename(src) + "_dst")
    for k in (1, 2):
        torch.save(dict(state={"a": torch.zeros(2)}, step_index=k - 1,
                        optimizer_state={"state": {0: {"step": 1}}}),
                   os.path.join(src, f"head_step{k}.pt"))
        d = os.path.join(src, f"train_step{k - 1}")
        os.makedirs(d, exist_ok=True)
        # буфер БЕЗ спутника — как у старых прогонов
        # старый буфер: run_config ещё не писался, arm и task_id отсутствуют
        torch.save(dict(meta=dict(step_index=k - 1, sigma=0.1, init_start=0,
                                  d1_seed=0, rl_seed=0,
                                  episodes=[dict(task_id=3, state_id=0)]),
                        data={}), os.path.join(d, "t0.pt"))
    pl = plan(src, dst)
    assert len(pl) == 4, pl
    done, skipped = migrate(src, dst, apply=True)
    assert len(done) == 4 and not skipped, (done, skipped)
    assert verify(dst) == 2
    # спутник появился и несёт мету буфера
    side = os.path.join(dst, "train_after_0", "t0.pt.meta.json")
    assert os.path.exists(side), "спутник не создан"
    sm = json.load(open(side))
    assert sm["step_index"] == 0 and sm["migrated_sidecar"] is True, sm
    # происхождение достроено, и по нему ячейка годна для пропуска
    rc = sm["run_config"]
    assert rc["arm"] == "policy" and rc["task_id"] == 3 and rc["sigma"] == 0.1, rc
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import k12j_cell_ok as ok
    conf, miss = ok.compare(sm, dict(arm="policy", sigma=0.1, step_index=0,
                                     task_id=3, init_start=0, d1_seed=0,
                                     rl_seed=0))
    assert conf == [] and miss == [], (conf, miss)
    o = torch.load(os.path.join(dst, "head_after_0001.pt"), map_location="cpu",
                   weights_only=False)
    assert o["step_index"] == 1 and o["migrated_step_index_was"] == 0, o
    # повторный перенос ничего не трогает
    done2, skipped2 = migrate(src, dst, apply=True)
    assert not done2 and len(skipped2) == 4, (done2, skipped2)
    # переименование реплики: прежнее имя сохраняется
    o = torch.load(os.path.join(dst, "head_after_0001.pt"), map_location="cpu",
                   weights_only=False)
    o["replica"] = "smoke_s0"
    torch.save(o, os.path.join(dst, "head_after_0001.pt"))
    ch = relabel(dst, "dev_s0_rl0")
    assert ch and ch[0][1] == "smoke_s0" and ch[0][2] == "dev_s0_rl0", ch
    o = torch.load(os.path.join(dst, "head_after_0001.pt"), map_location="cpu",
                   weights_only=False)
    assert o["replica"] == "dev_s0_rl0" and o["relabeled_replica_was"] == \
        "smoke_s0", o
    assert relabel(dst, "dev_s0_rl0") == [], "повторное переименование не пусто"

    # разрыв цепочки виден
    os.remove(os.path.join(dst, "head_after_0001.pt"))
    try:
        verify(dst)
    except SystemExit as e:
        assert "разрыв цепочки" in str(e), e
    else:
        raise AssertionError("разрыв цепочки не замечен")
    print("самопроверка k12j_migrate пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--replica",
                    help="новая метка реплики для перенесённых голов, "
                         "например dev_s0_rl0")
    ap.add_argument("--apply", action="store_true",
                    help="без него только показывает, что будет скопировано")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.src or not a.dst:
        ap.error("нужны --src и --dst")
    done, skipped = migrate(a.src, a.dst, apply=a.apply)
    for kind, x, y, k in done:
        print(f"  {'скопировано' if a.apply else 'будет'}: {kind} {x} -> {y}"
              + (f" (step_index={k})" if kind == "head" else ""))
    for y, why in skipped:
        print(f"  пропущено: {y} — {why}")
    if a.apply:
        filled = fill_provenance(a.dst)
        if filled:
            print(f"  достроено происхождение у {len(filled)} спутников")
    if a.apply and a.replica:
        for f, was, now in relabel(a.dst, a.replica):
            print(f"  реплика {f}: {was} -> {now}")
    if a.apply:
        n = verify(a.dst)
        print(f"цепочка проверена: {n} принятых обновлений в {a.dst}")
    else:
        print("это был только план; повторите с --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
