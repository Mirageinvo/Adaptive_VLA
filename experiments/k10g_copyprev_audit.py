"""K-10g: matched-аудит копировщика против опоры coarse24.

ЗАЧЕМ. K-10f показал, что тривиальный повтор предыдущей команды демонстрации
даёт пошаговую ошибку 0.0543 против 0.058-0.068 у coarse24 с 90% успеха. Но
сравнивались РАЗНЫЕ эпизоды и одна усреднённая величина, а гейт K-10d требует
одновременно позицию, вращение И знак схвата в каждом бакете подряд. Копия
почти неизбежно ошибается ровно в момент смены схвата, поэтому «копировщик
проходит гейт» из тех чисел не следует.

ЧТО РЕШАЕТ. Тот же вопрос на ТЕХ ЖЕ строках, тем же гейтом и с тем же
чтением префикса, что и вердикт K-10d. Ответ — размах копировщика:

  0x     отрицательный контроль отвергнут, гейт различает управление и повтор;
  1-2x   часть критерия вырождена: ближние бакеты повтор проходит;
  >=4x   копировщик получает тот же положительный вердикт, что требовался
         архитектуре, и гейт недискриминативен полностью.

ЧЕГО ЭТОТ СКРИПТ НЕ ДЕЛАЕТ. Он ничего не говорит о работоспособности в
замкнутом цикле. Пошаговая ошибка в режиме teacher forcing измеряется на
состояниях демонстрации, где ошибка не накапливается; она не может ни
доказать, ни опровергнуть управление из собственных состояний. Он проверяет
только пригодность САМОГО КРИТЕРИЯ.
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import goal_events as ge                                     # noqa: E402
import k10d_goal_controller as k10d                          # noqa: E402

# Поля, обязанные совпасть у опоры и у контроля. Без них «те же строки» —
# заявление, а не факт.
MUST_MATCH = ("split", "event", "edges", "cache", "ckpt", "goal_events_sha1",
              "event_params", "script_sha1", "n_rows", "n_episodes")


def load(path, want_predictor, part):
    with open(path) as f:
        j = json.load(f)
    pr = j.get("predictor")
    if pr != want_predictor:
        raise SystemExit(
            f"{path}: предсказатель «{pr}», ждали «{want_predictor}»")
    if j.get("split") != part:
        raise SystemExit(f"{path}: часть «{j.get('split')}», ждали «{part}»")
    for f_ in ("eval_keys", "eval_remaining", "buckets", "edges"):
        if f_ not in j:
            raise SystemExit(f"{path}: нет поля «{f_}», файл старой версии")
    return j


def check_matched(base, copy):
    """Строки обязаны совпасть ПОЭЛЕМЕНТНО и в том же порядке."""
    for f_ in MUST_MATCH:
        if base.get(f_) != copy.get(f_):
            raise SystemExit(
                f"расхождение по «{f_}»: опора {base.get(f_)!r}, "
                f"контроль {copy.get(f_)!r} — строки несопоставимы")
    bk = [tuple(x) for x in base["eval_keys"]]
    ck = [tuple(x) for x in copy["eval_keys"]]
    if bk != ck:
        same = len(set(bk) & set(ck))
        raise SystemExit(
            f"eval_keys различаются: общих {same} из {len(bk)}/{len(ck)}")
    if base["eval_remaining"] != copy["eval_remaining"]:
        raise SystemExit("eval_remaining различаются при совпавших ключах — "
                         "разная разметка событий")
    if len(set(bk)) != len(bk):
        raise SystemExit("ключи не уникальны")


def audit(base, copy, margin_pose, margin_grip):
    """Побакетное сравнение и размах копировщика В ВЫЗОВАХ VLA.

    ЕДИНИЦЫ. `prefix_ok` возвращает ШАГИ среды (0/8/16/32/48). Размах, с
    которым сравниваются пороги, измеряется в ВЫЗОВАХ и получается делением
    на H_CALL — ровно как в вердикте K-10d. Без деления один пройденный
    ближний бакет давал бы «8x» вместо «1x», и диапазон «1-2x» был бы
    недостижим, а любой успех читался бы как разгром гейта.

    ПРОПУСК БАКЕТА ОСТАНАВЛИВАЕТ. Пропускать отсутствующий бакет нельзя:
    тогда следующий занимает его место в списке флагов, `prefix_ok` считает
    его ближним, и отсутствие «<= 8» при остальных пройденных даёт ложный
    полный размах. Префикс имеет смысл только на СПЛОШНОЙ шкале удалённости.
    """
    edges = base["edges"]
    names = ge.bucket_names(edges)
    rows, flags = [], []
    for nm in names:
        b, c = base["buckets"].get(nm), copy["buckets"].get(nm)
        if b is None or c is None:
            miss = "опоре" if b is None else "контроле"
            raise SystemExit(
                f"бакет «{nm}» отсутствует в {miss}: шкала удалённости не "
                f"сплошная, размах префиксом не определён")
        if int(b["n"]) != int(c["n"]):
            raise SystemExit(
                f"бакет «{nm}»: строк в опоре {b['n']}, в контроле {c['n']} — "
                f"при совпавших ключах это разная бакетизация")
        if int(b["n"]) == 0:
            raise SystemExit(f"бакет «{nm}» пуст: размах не определён")
        ok = k10d.gate(c, b, margin_pose, margin_grip)
        flags.append(ok)
        rows.append((nm, b, c, ok))
    return rows, flags, k10d.prefix_ok(flags, edges) / k10d.H_CALL


def read_span(span):
    """Чтение в ВЫЗОВАХ VLA. Пороги те же, что у вердикта K-10d."""
    if span <= 0:
        return ("копировщик отвергнут сразу: гейт отличает управление от "
                "повтора, побакетные выводы K-10d остаются осмысленными")
    if span < 4:
        return (f"размах копировщика {span:g}x: ближняя часть критерия "
                f"вырождена — повтор проходит там же, где требовалось "
                f"управление, но положительного вердикта не получает")
    return ("размах копировщика >= 4x: тривиальный повтор получает ТОТ ЖЕ "
            "положительный вердикт, что требовался архитектуре — офлайновый "
            "гейт недискриминативен, решать только замкнутым симулятором")


def selftest():
    ge.selftest()
    ed = [8, 16, 32, 48]
    nm = ge.bucket_names(ed)

    def mk(pred, vals):
        return dict(predictor=pred, split="test", event="union", edges=ed,
                    cache="c", ckpt="k", goal_events_sha1="s",
                    event_params={}, script_sha1="x", n_rows=4, n_episodes=2,
                    eval_keys=[[0, 0], [0, 1], [1, 0], [1, 1]],
                    eval_remaining=[0, 8, 16, 48],
                    buckets={n: dict(n=1, pos=p, rot=r, pose=p, grip=g)
                             for n, (p, r, g) in zip(nm, vals)})

    base = mk("coarse24", [(0.06, 0.06, 0.02)] * 5)

    # ЕДИНИЦЫ. Размах — в ВЫЗОВАХ VLA, не в шагах среды. Пройденный ближний
    # бакет (8 шагов) есть ОДИН вызов, а не восемь: без деления на H_CALL
    # диапазон «1-2x» недостижим, и любой единичный успех читался бы как
    # полный разгром гейта.
    def span_for(flags_vals):
        return audit(base, mk("copy-prev", flags_vals), 0.0, 0.005)[2]

    good, bad = (0.05, 0.05, 0.01), (0.9, 0.9, 0.9)
    assert span_for([good] + [bad] * 4) == 1, "один ближний бакет = 1 вызов"
    assert span_for([good] * 2 + [bad] * 3) == 2
    assert span_for([good] * 3 + [bad] * 2) == 4
    assert span_for([good] * 5) == 6, "полная шкала 48 шагов = 6 вызовов"
    assert "вырождена" in read_span(1) and "вырождена" in read_span(2)
    assert "недискриминативен" in read_span(4)
    assert "отвергнут" in read_span(0)

    # Копия точна по позе везде, но врёт знаком схвата в ближнем бакете —
    # ровно то, чего ждём от повтора в момент переключения.
    grip_bad = mk("copy-prev", [(0.05, 0.05, 0.30)] + [good] * 4)
    _, _, sp = audit(base, grip_bad, 0.0, 0.005)
    assert sp == 0, f"провал знака в ближнем бакете обязан давать 0, дал {sp}"

    # Поза хуже опоры — тоже отказ, даже при идеальном знаке.
    assert span_for([(0.09, 0.05, 0.0)] * 5) == 0

    # Размах ЯВЛЯЕТСЯ ПРЕФИКСОМ: дальний хороший бакет не спасает провал.
    assert span_for([good, good, bad, good, good]) == 2

    # ПРОПУЩЕННЫЙ БАКЕТ ОСТАНАВЛИВАЕТ. Раньше он молча пропускался, и
    # отсутствие «<= 8» при остальных пройденных давало ложный полный размах:
    # следующий бакет занимал место ближнего в списке флагов.
    gap = mk("copy-prev", [good] * 5)
    del gap["buckets"][nm[0]]
    try:
        audit(base, gap, 0.0, 0.005)
        raise AssertionError("отсутствие ближнего бакета дало размах")
    except SystemExit:
        pass
    empty = mk("copy-prev", [good] * 5)
    empty["buckets"][nm[1]]["n"] = 0
    base_empty = mk("coarse24", [(0.06, 0.06, 0.02)] * 5)
    base_empty["buckets"][nm[1]]["n"] = 0
    try:
        audit(base_empty, empty, 0.0, 0.005)
        raise AssertionError("пустой бакет дал размах")
    except SystemExit:
        pass
    # Разное число строк в одноимённом бакете — признак разной бакетизации.
    nmis = mk("copy-prev", [good] * 5)
    nmis["buckets"][nm[2]]["n"] = 7
    try:
        audit(base, nmis, 0.0, 0.005)
        raise AssertionError("разное n прошло проверку")
    except SystemExit:
        pass

    # Несовпадение строк обязано ОСТАНАВЛИВАТЬ, а не предупреждать.
    other = mk("copy-prev", [(0.05, 0.05, 0.01)] * 5)
    other["eval_keys"] = [[0, 0], [0, 1], [1, 0], [2, 9]]
    try:
        check_matched(base, other)
        raise AssertionError("разные ключи прошли проверку")
    except SystemExit:
        pass
    other2 = mk("copy-prev", [(0.05, 0.05, 0.01)] * 5)
    other2["event"] = "grip"
    try:
        check_matched(base, other2)
        raise AssertionError("разное событие прошло проверку")
    except SystemExit:
        pass
    # Опора обязана быть coarse24.
    try:
        load_ok = mk("copy-prev", [(0.0, 0.0, 0.0)] * 5)
        if load_ok["predictor"] != "coarse24":
            raise SystemExit("ok")
        raise AssertionError
    except SystemExit:
        pass
    print("самопроверка k10g пройдена (версия «размах в вызовах»): "
          "1/2/4/6x по префиксу, ближний бакет = 1 вызов, провал знака даёт "
          "ноль, пропуск и пустота бакета останавливают, несовпадение строк "
          "и события останавливает")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--baseline", help="k10e с --predictor coarse24")
    ap.add_argument("--copy", help="k10e с --predictor copy-prev")
    ap.add_argument("--split", default="test")
    # ТЕ ЖЕ ДОПУСКИ, что у вердикта K-10d: иначе аудит отвечал бы на другой
    # вопрос, чем тот, чью пригодность он проверяет.
    ap.add_argument("--margin-pose", type=float, default=0.0)
    ap.add_argument("--margin-grip", type=float, default=0.005)
    ap.add_argument("--out", default="data/k10g_copyprev_audit.json")
    args = ap.parse_args()

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k10g sha {sha}")
    if args.selftest:
        selftest()
        return
    if not (args.baseline and args.copy):
        raise SystemExit("нужны --baseline и --copy")

    base = load(args.baseline, "coarse24", args.split)
    copy = load(args.copy, "copy-prev", args.split)
    check_matched(base, copy)
    print(f"строки совпали поэлементно: {base['n_rows']} наблюдений, "
          f"{base['n_episodes']} эпизодов, событие «{base['event']}», "
          f"часть «{args.split}»")

    rows, flags, span = audit(base, copy, args.margin_pose, args.margin_grip)
    print(f"\n  {'удалённость':>13}{'строк':>8}"
          f"{'поз.оп':>9}{'поз.коп':>9}"
          f"{'вр.оп':>9}{'вр.коп':>9}"
          f"{'зн.оп':>8}{'зн.коп':>8}{'':>4}")
    for nm, b, c, ok in rows:
        print(f"  {nm:>13}{b['n']:>8}"
              f"{b['pos']:>9.4f}{c['pos']:>9.4f}"
              f"{b['rot']:>9.4f}{c['rot']:>9.4f}"
              f"{b['grip']:>7.1%}{c['grip']:>8.1%}"
              f"{'  да' if ok else '  нет':>4}")

    print(f"\nразмах копировщика: {span:g}x "
          f"(вызовов VLA; префикс {span * k10d.H_CALL:g} шагов)")
    print(read_span(span))
    print("\nЭто утверждение о ПРИГОДНОСТИ КРИТЕРИЯ, а не о работоспособности "
          "в замкнутом цикле: пошаговая ошибка считается на состояниях "
          "демонстрации, где ошибка не накапливается.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    json.dump(dict(copy_span=float(span),
                   copy_span_steps=float(span * k10d.H_CALL), split=args.split,
                   event=base["event"], edges=base["edges"],
                   n_rows=base["n_rows"], n_episodes=base["n_episodes"],
                   margin_pose=args.margin_pose, margin_grip=args.margin_grip,
                   per_bucket=[dict(name=nm, n=b["n"],
                                    base=dict(pos=b["pos"], rot=b["rot"],
                                              grip=b["grip"]),
                                    copy=dict(pos=c["pos"], rot=c["rot"],
                                              grip=c["grip"]),
                                    passed=bool(ok))
                               for nm, b, c, ok in rows],
                   baseline_file=os.path.abspath(args.baseline),
                   copy_file=os.path.abspath(args.copy),
                   k10e_script_sha1=base["script_sha1"],
                   goal_events_sha1=base["goal_events_sha1"],
                   script_sha1=sha),
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
