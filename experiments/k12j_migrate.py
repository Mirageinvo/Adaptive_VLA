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
        torch.save(dict(meta=dict(step_index=k - 1, arm="policy", sigma=0.1,
                                  task_id=3, init_start=0, d1_seed=0,
                                  rl_seed=0),
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
    o = torch.load(os.path.join(dst, "head_after_0001.pt"), map_location="cpu",
                   weights_only=False)
    assert o["step_index"] == 1 and o["migrated_step_index_was"] == 0, o
    # повторный перенос ничего не трогает
    done2, skipped2 = migrate(src, dst, apply=True)
    assert not done2 and len(skipped2) == 4, (done2, skipped2)
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
        n = verify(a.dst)
        print(f"цепочка проверена: {n} принятых обновлений в {a.dst}")
    else:
        print("это был только план; повторите с --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
