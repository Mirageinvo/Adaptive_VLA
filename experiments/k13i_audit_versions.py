#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13i: аудит версий кода в законченном прогоне лестницы.

ЗАЧЕМ. §41 утверждает, что лестница посчитана одной версией `k12d_rollout` и
одной версией `k12e_pg_step`. Пока это утверждение существовало только как
вывод команды в терминале, проверить его со стороны было нельзя: каталоги
`data/` в git не попадают. Здесь оно превращается в артефакт, который ложится
рядом с кодом и потому проверяем.

ЧТО СЧИТАЕТСЯ. По каждому виду артефакта (шаги, оценочные ячейки, буферы) —
сколько найдено, сколько без записи версии, какие версии встретились. Плюс
отпечаток отсортированного списка всех проверенных записей: он меняется от
любой перестановки, добавления или подмены файла, поэтому сам список в git
класть не нужно.

ОТСУТСТВИЕ ПОЛЯ — ОТКАЗ, А НЕ ПУСТОЕ МНОЖЕСТВО. Артефакт без `script_sha1`
нельзя засчитать как совпадающий: он просто не сообщает, чем посчитан.
"""
import argparse
import glob
import hashlib
import json
import os
import sys


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def collect(root, head, load_pt):
    """Записи вида (путь, вид, голова, шаг, sha). Чистая функция ради теста."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "step_*.json"))):
        try:
            o = json.load(open(p))
        except (ValueError, OSError):
            o = {}
        k = os.path.basename(p)[5:-5]
        rows.append((os.path.relpath(p, root), "step", head, k,
                     o.get("script_sha1")))
    for p in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        try:
            o = json.load(open(p))
        except (ValueError, OSError):
            o = {}
        rows.append((os.path.relpath(p, root), "cell", head, "",
                     o.get("script_sha1")))
    for p in sorted(glob.glob(os.path.join(root, "*", "*.pt"))):
        m = load_pt(p)
        rows.append((os.path.relpath(p, root), "buffer", head, "",
                     (m or {}).get("script_sha1")))
    return rows


def summarize(rows, expect):
    """Сводка по видам артефактов и общий вердикт."""
    kinds = {}
    for _p, kind, _h, _k, sha in rows:
        d = kinds.setdefault(kind, dict(n=0, missing=0, shas=set()))
        d["n"] += 1
        if sha is None:
            d["missing"] += 1
        else:
            d["shas"].add(str(sha))
    bad = []
    for kind, d in kinds.items():
        if d["missing"]:
            bad.append(f"{kind}: {d['missing']} записей без script_sha1")
        if len(d["shas"]) > 1:
            bad.append(f"{kind}: несколько версий {sorted(d['shas'])}")
    for kind, want in (("step", expect.get("k12e")),
                       ("cell", expect.get("k12d")),
                       ("buffer", expect.get("k12d"))):
        if want and kind in kinds and kinds[kind]["shas"] \
                and sorted(kinds[kind]["shas"]) != [str(want)]:
            bad.append(f"{kind}: версия {sorted(kinds[kind]['shas'])}, "
                       f"в записи запуска {want}")
    out = {k: dict(n=v["n"], missing=v["missing"], shas=sorted(v["shas"]))
           for k, v in kinds.items()}
    return out, bad


def rows_sha(rows):
    blob = "\n".join("\t".join(str(x) for x in r) for r in sorted(rows))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def selftest():
    fake = [("a.json", "step", "s0", "1", "AAA"),
            ("d/b.json", "cell", "s0", "", "BBB"),
            ("d/c.pt", "buffer", "s0", "", "BBB")]
    out, bad = summarize(fake, dict(k12e="AAA", k12d="BBB"))
    assert not bad, bad
    assert out["cell"]["n"] == 1 and out["buffer"]["shas"] == ["BBB"]

    # смешение версий
    mix = fake + [("d/e.json", "cell", "s0", "", "CCC")]
    _o, bad = summarize(mix, dict(k12e="AAA", k12d="BBB"))
    assert any("несколько версий" in x for x in bad), bad

    # отсутствующая версия — отказ, а не пустое множество
    miss = fake + [("d/f.pt", "buffer", "s0", "", None)]
    _o, bad = summarize(miss, dict(k12e="AAA", k12d="BBB"))
    assert any("без script_sha1" in x for x in bad), bad

    # расхождение с записью запуска
    _o, bad = summarize(fake, dict(k12e="AAA", k12d="ZZZ"))
    assert any("в записи запуска" in x for x in bad), bad

    # отпечаток списка меняется от подмены и не зависит от порядка
    a = rows_sha(fake)
    assert a == rows_sha(list(reversed(fake)))
    assert a != rows_sha(fake[:2] + [("d/c.pt", "buffer", "s0", "", "CCC")])
    print("самопроверка k13i_audit_versions пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", action="append", default=[],
                    help="каталог прогона; можно несколько")
    ap.add_argument("--out", default="reports/k13i/version_audit.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.root:
        ap.error("нужен хотя бы один --root")

    import torch

    def load_pt(p):
        try:
            return torch.load(p, map_location="cpu",
                              weights_only=False).get("meta")
        except Exception:                                  # noqa: BLE001
            return None

    per_root, all_rows, all_bad = {}, [], []
    for root in a.root:
        head = os.path.basename(root.rstrip("/"))
        vp = os.path.join(root, "script_versions.jsonl")
        if not os.path.exists(vp):
            raise SystemExit(f"нет {vp}: с чем сверять версии, неизвестно")
        recs = [json.loads(x) for x in open(vp) if x.strip()]
        expect = recs[0]
        if any(r.get("k12d") != expect.get("k12d")
               or r.get("k12e") != expect.get("k12e") for r in recs):
            all_bad.append(f"{head}: в script_versions.jsonl несколько "
                           f"разных версий запуска")
        rows = collect(root, head, load_pt)
        kinds, bad = summarize(rows, expect)
        per_root[head] = dict(
            root=root, expect=expect, kinds=kinds, n_rows=len(rows),
            rows_sha1=rows_sha(rows),
            script_versions_sha1=sha12(vp), failures=bad)
        all_rows += rows
        all_bad += [f"{head}: {x}" for x in bad]
        print(f"\n=== {head}")
        for kind, d in sorted(kinds.items()):
            print(f"  {kind:7s} {d['n']:4d} записей, без версии "
                  f"{d['missing']}, версии {d['shas']}")
        for x in bad:
            print(f"  ОТКАЗ: {x}")

    ok = not all_bad
    out = dict(roots=per_root, all_ok=bool(ok), failures=all_bad,
               n_rows_total=len(all_rows), rows_sha1_total=rows_sha(all_rows),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"\n  all_ok = {ok}; сохранено: {a.out}")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
