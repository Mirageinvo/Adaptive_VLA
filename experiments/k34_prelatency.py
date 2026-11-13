#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-34: основная оценка ДО замера задержки — решение пользователя
(руководителя исследования) от 2026-11-14.

Штатный k34_protocol.pilot_verify_problems не меняется (k34_protocol.py и
k34_run.sh закреплены по SHA в плане пилота). Здесь допускается ровно одно
отличие от passed: пересчитанный статус гейта awaiting_latency. Все прочие
привязки (план, checkpoint, SHA всех блоков, отсутствие/совпадение замера,
пересчёт успехов и статуса) обязаны сходиться; awaiting_latency по
k34_protocol.gate_decision означает выполненные условия 1–2 по качеству;
stop и technical_stop не допускаются. Решение записывается в
reports/k34/eval_before_latency.json с SHA отчёта гейта. Замер задержки
B = 1 на свободной карте остаётся обязательным.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k34_protocol as pc  # noqa: E402

RECORD = os.path.join(pc.OUT, "eval_before_latency.json")
AWAIT = "статус awaiting_latency"


def problems(plan_file):
    """-> (проблемы, статус): допустимы passed или awaiting_latency."""
    p = pc.pilot_verify_problems(plan_file)
    if not p:
        return [], "passed"
    rest = [x for x in p if x != AWAIT]
    if rest or AWAIT not in p:
        return p, None
    return [], "awaiting_latency"


def verify(plan_file):
    p, st = problems(plan_file)
    if p:
        print("  гейт пилота: ОТКАЗ — " + "; ".join(p[:4]))
        return 3
    if st == "passed":
        print("  гейт пилота: passed, привязки сходятся")
        return 0
    g = json.load(open(pc.GATE_FILE, encoding="utf-8"))
    rec = dict(decision="основная оценка до замера задержки",
               authorized_by="пользователь (руководитель исследования), "
                             "2026-11-14",
               gate_status=g["status"], gate_sha1=pc.sha(pc.GATE_FILE),
               plan_sha1=g["plan_sha1"], successes=g["successes"],
               time=time.strftime("%Y-%m-%dT%H:%M:%S"),
               note="задержка B = 1 на свободной карте обязательна; при stop "
                    "основная оценка не используется как свидетельство "
                    "практической пригодности ранних выходов")
    if os.path.exists(RECORD):
        old = json.load(open(RECORD, encoding="utf-8"))
        rec["first"] = old.get("first") or old
    pc.save_json(RECORD, rec)
    print(f"  гейт пилота: awaiting_latency, условия 1–2 выполнены, привязки "
          f"сходятся — основная оценка ДО замера задержки ({RECORD})")
    return 0


def selftest():
    keep = pc.pilot_verify_problems
    try:
        pc.pilot_verify_problems = lambda f: []
        assert problems("x") == ([], "passed")
        pc.pilot_verify_problems = lambda f: [AWAIT]
        assert problems("x") == ([], "awaiting_latency")
        for bad in (["статус stop"], ["статус technical_stop"],
                    [AWAIT, "блоки пилота отсутствуют или изменены после "
                            "гейта"], ["замер задержки другого плана"],
                    ["статус ≠ пересчитанному (stop)", AWAIT]):
            pc.pilot_verify_problems = lambda f, b=bad: b
            p, st = problems("x")
            assert p and st is None, bad
    finally:
        pc.pilot_verify_problems = keep
    print("самопроверка k34_prelatency пройдена: допускаются только passed и "
          "чистый awaiting_latency; stop, technical_stop, изменённые блоки, "
          "чужой замер, расхождение статуса — отказ")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if len(sys.argv) != 3 or sys.argv[1] != "verify":
        raise SystemExit("использование: k34_prelatency.py verify PLAN")
    sys.exit(verify(sys.argv[2]))
