"""K-10c: сколько вызовов VLA переживает одна СОБЫТИЙНАЯ цель.

ПОЧЕМУ ЭТО ГЛАВНОЕ ЧИСЛО. Архитектура «токен как цель» окупается ровно в той
мере, в какой цель сохраняется между вызовами. Если фаза длится 40 шагов, а
политика вызывается каждые 8, одна цель покрывает пять вызовов — потенциально
пятикратное сокращение обращений к VLA, множитель больше всего, что мы до сих
пор получили. Если фазы короткие, экономии нет вовсе, и направление
закрывается до написания архитектуры.

Считается ПО ДЕМОНСТРАЦИЯМ: ни модели, ни симулятора, ни GPU.

ПОЧЕМУ НЕ КОНЕЦ ОКНА. Конец фиксированного окна из t относится к моменту t+20,
из t+8 — к t+28. Это разные физические цели, и «стабильность» такой цели
измерить нельзя. Цель обязана определяться СОБЫТИЕМ, тогда у неё есть
собственный момент времени, не зависящий от того, когда мы посмотрели.

ДВА ОПРЕДЕЛЕНИЯ СОБЫТИЯ, оба проверяются:
  * СХВАТ — смена знака команды схвата. Их немного (обычно 2-4 за эпизод), они
    надёжны и именно знак схвата объяснял сохранность успеха в K-6h;
  * ОСТАНОВКА — участок низкой скорости достаточной длительности после
    значимого перемещения. Их больше, они дробят эпизод мельче и дают нижнюю
    оценку персистентности.

ЧТО ПЕЧАТАЕТСЯ И КАК ЧИТАТЬ. Главная величина — отношение числа вызовов к
числу целей: во сколько раз можно было бы сократить обращения к VLA при
идеальном мониторе. Доля `NO_EDIT` печатается рядом СПРАВОЧНО и систематически
завышает выигрыш. Отдельно печатается доля вызовов после последнего события:
там цель одна по построению, и большой хвост означает, что весь выигрыш держится
на конце эпизода.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ, записано ДО запуска (горизонт H=8):
  * отношение «вызовов к целям» >= 2.5 -> обращений к VLA можно было бы
    сделать в 2.5 раза меньше, направление живо;
  * <= 1.5 -> цель меняется почти каждый вызов, экономии нет,
    направление закрывается;
  * между -> не доказано ничего.
Вердикт выносится по ЭТОМУ отношению, а не по доле NO_EDIT: последняя
считается на сетке вызовов и теряет события, попавшие между двумя вызовами,
поэтому систематически завышает выигрыш.

ЧЕГО ЭТО НЕ ДОКАЗЫВАЕТ. Верхнюю оценку, а не достижимую: она предполагает
идеальный дешёвый монитор, который распознаёт достижение цели, не запуская
VLA. Без такого монитора `NO_EDIT` не даёт пропустить вызов вовсе — вопрос
отдельный и здесь не решается.

Запуск:
    python3 experiments/k10c_goal_persistence.py --selftest
    python3 experiments/k10c_goal_persistence.py --n-ep 200 \\
        --out data/k10c_goal_persistence.json
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np


def grip_events(actions):
    """Индексы смены знака команды схвата (канал 6)."""
    s = np.sign(actions[:, 6])
    s[s == 0] = 1.0
    return np.where(np.diff(s) != 0)[0] + 1


def stop_events(state_xyz, speed_frac=0.3, min_dwell=3, min_travel=0.02):
    """Остановки: участки НИЗКОЙ скорости достаточной длительности.

    ПОЧЕМУ НЕ «ЛОКАЛЬНЫЙ МИНИМУМ». Первая версия искала точки, где скорость
    не больше соседних с допуском 1e-12. При РАВНОМЕРНОМ движении все скорости
    равны, значит каждая точка проходит проверку: на прямой из 300 шагов
    находилось 73 несуществующие остановки. Самопроверка это пропустила,
    потому что была написана как `assert len(...) >= 0` — истинное всегда.

    Здесь остановка определяется абсолютно, а не относительно соседей:
    скорость ниже доли `speed_frac` от медианной по эпизоду, подряд не менее
    `min_dwell` шагов. При равномерном движении скорость РАВНА медианной и
    порога не пересекает ни разу — ровно ноль событий.

    Дополнительно требуется перемещение не меньше `min_travel` с прошлого
    принятого события: иначе дрожание на месте дробится на много «целей».
    """
    xyz = np.asarray(state_xyz, np.float64)
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if len(v) < min_dwell:
        return np.array([], np.int64)
    med = float(np.median(v))
    if med <= 0:
        return np.array([], np.int64)          # робот не двигался вовсе
    slow = v < speed_frac * med
    ev, last, i = [], 0, 0
    n = len(slow)
    while i < n:
        if not slow[i]:
            i += 1
            continue
        j = i
        while j < n and slow[j]:
            j += 1
        if j - i >= min_dwell:
            c = (i + j) // 2                   # середина остановки
            if np.linalg.norm(xyz[c] - xyz[last]) >= min_travel:
                ev.append(c); last = c
        i = j
    return np.asarray(ev, np.int64)


def persistence(events, n_steps, horizon):
    """Доля вызовов с той же целью и длина фазы в вызовах.

    Цель в момент t — БЛИЖАЙШЕЕ СЛЕДУЮЩЕЕ событие. Вызовы идут в моменты
    0, H, 2H, ... Цель «та же», если следующее событие для двух соседних
    вызовов одно и то же.
    """
    calls = np.arange(0, n_steps, horizon)
    if len(calls) < 2:
        return None
    ev = np.asarray(events, np.int64)
    # np.searchsorted даёт индекс первого события строго правее вызова;
    # при отсутствии — len(ev), то есть «цель — конец эпизода», и это тоже
    # законная цель, одинаковая для всех оставшихся вызовов.
    goal_id = np.searchsorted(ev, calls, side="right")
    same = goal_id[1:] == goal_id[:-1]
    # Длина фазы в вызовах: сколько подряд идущих вызовов делят одну цель.
    lens, cur = [], 1
    for s in same:
        if s:
            cur += 1
        else:
            lens.append(cur); cur = 1
    lens.append(cur)
    # ТОЧНЫЙ ВЕРХНИЙ ПРЕДЕЛ СОКРАЩЕНИЯ. Доля NO_EDIT меряет разреженность
    # ВЫБРАННЫХ СОБЫТИЙ на сетке вызовов и может завышать выигрыш: события,
    # попавшие между двумя вызовами, в ней теряются. Прямой учёт числа целей
    # от этого свободен: целей ровно на одну больше, чем событий.
    n_goal = int(len(ev)) + 1
    s_max = len(calls) / n_goal
    # ХВОСТ ПОСЛЕ ПОСЛЕДНЕГО СОБЫТИЯ автоматически объявляется одной целью и
    # тянет результат вверх. Печатается отдельно, чтобы это было видно.
    tail = int((calls > (ev[-1] if len(ev) else -1)).sum())
    return dict(n_calls=int(len(calls)), n_events=int(len(ev)),
                n_goals=n_goal, s_max=float(s_max),
                tail_calls=int(tail),
                tail_frac=float(tail / max(len(calls), 1)),
                no_edit=float(same.mean()),
                phase_calls_mean=float(np.mean(lens)),
                phase_calls_median=float(np.median(lens)))


def read_rule(s_max, n_events=None, tail_frac=None,
              good=2.5, bad=1.5, min_events=0.5, max_tail=0.5):
    """Вердикт по ПРЯМОМУ УЧЁТУ целей, а не по доле NO_EDIT.

    NO_EDIT считается на сетке вызовов и теряет события, попавшие между двумя
    вызовами, поэтому завышает выигрыш. Отношение «вызовов к целям» от этого
    свободно: целей ровно на одну больше, чем событий.

    ВЫРОЖДЕННЫЙ СЛУЧАЙ ОТСЕКАЕТСЯ ПЕРВЫМ. Если детектор не нашёл событий, то
    цель ровно одна — «конец эпизода», — и s_max тождественно равен числу
    вызовов. Прежняя версия правила выдавала на этом «до 19.4x меньше
    обращений», то есть объявляла успехом отказ детектора. То же и при
    большом хвосте: если почти все вызовы приходятся на участок после
    последнего события, выигрыш держится не на сегментации, а на её
    отсутствии.
    """
    if s_max is None:
        return "недействительно"
    if n_events is not None and n_events < min_events:
        return (f"ДЕТЕКТОР НИЧЕГО НЕ НАШЁЛ ({n_events:.1f} событий на эпизод) "
                f"— число не читается")
    if tail_frac is not None and tail_frac > max_tail:
        return (f"ХВОСТ {tail_frac:.0%} ВЫЗОВОВ ПОСЛЕ ПОСЛЕДНЕГО СОБЫТИЯ — "
                f"выигрыш держится на отсутствии сегментации")
    if s_max >= good:
        return f"ЦЕЛЬ ПЕРЕЖИВАЕТ ВЫЗОВЫ: до {s_max:.1f}x меньше обращений"
    if s_max <= bad:
        return "ЦЕЛЬ МЕНЯЕТСЯ ПОЧТИ КАЖДЫЙ ВЫЗОВ: экономии нет"
    return "НЕ ДОКАЗАНО НИЧЕГО"


def selftest():
    # Смена знака схвата ловится ровно на переходе, а не на каждом шаге.
    a = np.zeros((20, 7)); a[:, 6] = 1.0; a[7:, 6] = -1.0
    assert list(grip_events(a)) == [7], grip_events(a)
    a2 = np.zeros((20, 7)); a2[:, 6] = 1.0
    assert len(grip_events(a2)) == 0

    # ПЕРСИСТЕНТНОСТЬ. Одно событие в середине: все вызовы кроме одного
    # делят цель, значит доля NO_EDIT высокая.
    r = persistence([50], 100, 8)
    assert r["n_calls"] == 13
    assert r["no_edit"] > 0.9, r

    # События на каждом вызове: цель меняется всегда, NO_EDIT равен нулю.
    r2 = persistence(list(range(0, 100, 8)), 100, 8)
    assert r2["no_edit"] < 0.1, r2

    # Событий нет вовсе: цель одна на весь эпизод.
    r3 = persistence([], 100, 8)
    assert r3["no_edit"] == 1.0 and r3["phase_calls_mean"] == r3["n_calls"]

    # ГОРИЗОНТ ВЛИЯЕТ В ПРАВИЛЬНУЮ СТОРОНУ: чем реже вызовы, тем чаще цель
    # успевает смениться между ними.
    ev = list(range(0, 200, 20))
    assert persistence(ev, 200, 4)["no_edit"] > persistence(ev, 200, 16)["no_edit"]

    # РАВНОМЕРНОЕ ДВИЖЕНИЕ — РОВНО НОЛЬ СОБЫТИЙ. Прежняя реализация искала
    # локальный минимум с допуском и находила здесь 73 остановки, а прежняя
    # самопроверка была написана как `assert len(...) >= 0` и не могла упасть.
    line = np.stack([np.linspace(0, 1, 300)] * 3, 1)
    assert len(stop_events(line)) == 0, len(stop_events(line))

    # ДРОЖАНИЕ НА МЕСТЕ — тоже ноль: перемещения нет, порог travel не пройден.
    rng = np.random.default_rng(0)
    assert len(stop_events(np.cumsum(rng.normal(0, 1e-4, (300, 3)), 0))) == 0

    # НЕПОДВИЖНОСТЬ — ноль, а не деление на ноль.
    assert len(stop_events(np.zeros((300, 3)))) == 0

    # СИНТЕТИЧЕСКОЕ stop-go: три паузы в известных местах, три события.
    seg, pause = 40, 10
    parts = []
    pos = np.zeros(3)
    for k in range(4):
        step = np.array([0.01, 0.0, 0.0])
        parts.append(pos + np.arange(1, seg + 1)[:, None] * step)
        pos = parts[-1][-1]
        if k < 3:
            parts.append(np.repeat(pos[None], pause, axis=0))
    xyz = np.concatenate(parts)
    ev = stop_events(xyz, min_travel=0.05)
    assert len(ev) == 3, f"ждали 3 остановки, нашли {len(ev)}: {ev}"
    for e, want in zip(ev, [seg, 2 * seg + pause, 3 * seg + 2 * pause]):
        assert abs(int(e) - want) <= pause, (e, want)

    # ПОРОГ ПЕРЕМЕЩЕНИЯ работает: при огромном min_travel событий нет.
    assert len(stop_events(xyz, min_travel=10.0)) == 0

    # УЧЁТ СОКРАЩЕНИЯ. Целей на одну больше, чем событий; хвост считается.
    p = persistence([50], 100, 8)
    assert p["n_goals"] == 2 and abs(p["s_max"] - 13 / 2) < 1e-12, p
    assert p["tail_calls"] == 6, p            # вызовы 56,64,...,96
    assert persistence([], 100, 8)["tail_frac"] == 1.0

    assert "ПЕРЕЖИВАЕТ" in read_rule(4.0, n_events=2.2, tail_frac=0.17)
    assert "экономии нет" in read_rule(1.1, n_events=2.2, tail_frac=0.17)
    assert "НЕ ДОКАЗАНО" in read_rule(2.0, n_events=2.2, tail_frac=0.17)
    assert read_rule(None) == "недействительно"
    # ВЫРОЖДЕННЫЕ СЛУЧАИ ОТСЕКАЮТСЯ РАНЬШЕ ПОРОГОВ. Прогон 200 эпизодов дал
    # у детектора остановок 0.0 событий и хвост 98%, а правило выдало
    # «до 19.4x меньше обращений» — успехом был объявлен отказ детектора.
    assert "НИЧЕГО НЕ НАШЁЛ" in read_rule(19.4, n_events=0.0, tail_frac=0.98)
    assert "ХВОСТ" in read_rule(19.4, n_events=3.0, tail_frac=0.98)
    # Настоящий результат при тех же порогах читается как успех.
    assert "ПЕРЕЖИВАЕТ" in read_rule(6.74, n_events=2.2, tail_frac=0.174)
    print("самопроверка k10c пройдена (версия «фаза в вызовах»): события "
          "схвата, персистентность на четырёх случаях, зависимость от "
          "горизонта, порог перемещения")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-ep", type=int, default=200)
    ap.add_argument("--horizons", default="4,8,12,16")
    ap.add_argument("--gate-horizon", type=int, default=8)
    ap.add_argument("--min-travel", type=float, default=0.02)
    ap.add_argument("--speed-frac", type=float, default=0.3)
    ap.add_argument("--min-dwell", type=int, default=3)
    ap.add_argument("--good", type=float, default=2.5)
    ap.add_argument("--bad", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k10c sha1 {sha}")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rid, rev = "physical-intelligence/libero", "v2.0"
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(1693)[: args.n_ep * 2]
    hz = [int(x) for x in args.horizons.split(",")]
    acc = {d: {h: [] for h in hz} for d in ("grip", "stop")}
    ep_len, n_used = [], 0

    for e in order:
        if n_used >= args.n_ep:
            break
        try:
            f = hf_hub_download(
                rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                repo_type="dataset", revision=rev)
            tab = pq.read_table(f)
            acts = np.asarray(tab.column("actions").to_pylist(), np.float32)
            st = np.asarray(tab.column("state").to_pylist(), np.float32)
        except Exception as ex:                      # noqa: BLE001
            print(f"  эпизод {e}: пропуск ({type(ex).__name__}: {ex})")
            continue
        if len(acts) < 40:
            continue
        # ТОЛЬКО ИМЕНОВАННЫЕ АРГУМЕНТЫ. Позиционный вызов
        # `stop_events(xyz, args.min_travel)` клал 0.02 в speed_frac, то есть
        # порог был 2% медианной скорости вместо 30% — в пятнадцать раз
        # строже. Самопроверка не поймала: в синтетике паузы идеальные, там
        # любой порог срабатывает.
        ev = dict(grip=grip_events(acts),
                  stop=stop_events(st[:, :3], speed_frac=args.speed_frac,
                                   min_dwell=args.min_dwell,
                                   min_travel=args.min_travel))
        for d in acc:
            for h in hz:
                r = persistence(ev[d], len(acts), h)
                if r is not None:
                    acc[d][h].append(r)
        ep_len.append(len(acts))
        n_used += 1
        if n_used % 50 == 0:
            print(f"  эпизодов {n_used}/{args.n_ep}", flush=True)

    if not ep_len:
        raise SystemExit("ни одного эпизода не загрузилось")
    print(f"\nэпизодов {n_used}, средняя длина {np.mean(ep_len):.0f} шагов\n")

    res = {}
    for d in ("grip", "stop"):
        name = "схват" if d == "grip" else "остановка"
        print(f"  событие «{name}»:")
        print(f"    {'H':>4}{'micro':>9}{'macro':>9}{'NO_EDIT':>10}"
              f"{'событий':>10}{'хвост':>9}")
        res[d] = {}
        for h in hz:
            if not acc[d][h]:
                continue
            m = {k: float(np.mean([r[k] for r in acc[d][h]]))
                 for k in ("no_edit", "phase_calls_mean", "phase_calls_median",
                           "n_events", "n_calls", "s_max", "n_goals",
                           "tail_calls", "tail_frac")}
            # MICRO — ОБЩЕЕ СОКРАЩЕНИЕ, macro — среднее отношений по
            # эпизодам. Для «во сколько раз меньше обращений суммарно» верно
            # micro; macro завышает, потому что короткие эпизоды с одной
            # целью дают большие отношения и тянут среднее вверх.
            m["s_macro"] = m.pop("s_max")
            m["s_micro"] = (sum(r["n_calls"] for r in acc[d][h])
                            / max(sum(r["n_goals"] for r in acc[d][h]), 1))
            m["s_max"] = m["s_micro"]
            res[d][str(h)] = m
            print(f"    {h:>4}{m['s_micro']:>9.2f}{m['s_macro']:>9.2f}"
                  f"{m['no_edit']:>9.1%}{m['n_events']:>10.1f}"
                  f"{m['tail_frac']:>8.1%}")
        print()

    g = res["grip"].get(str(args.gate_horizon))
    s = res["stop"].get(str(args.gate_horizon))
    print(f"  ЧИТАТЬ по горизонту {args.gate_horizon} — наше исполняемое окно.")
    print(f"  «вызовов/целей» — во сколько раз можно было бы сократить")
    print(f"  обращения к VLA при ИДЕАЛЬНОМ дешёвом мониторе. NO_EDIT дан")
    print(f"  для справки и завышает выигрыш. «Хвост» — доля вызовов после")
    print(f"  последнего события: там цель одна по построению, и большой")
    print(f"  хвост означает, что выигрыш держится на конце эпизода.\n")
    for d, r in (("схват", g), ("остановка", s)):
        if r is None:
            continue
        print(f"  событие «{d}»: micro {r['s_micro']:.2f} "
              f"(macro {r['s_macro']:.2f}), "
              f"NO_EDIT {r['no_edit']:.1%}, хвост {r['tail_frac']:.1%} — "
              f"{read_rule(r['s_max'], r['n_events'], r['tail_frac'], args.good, args.bad)}")
    print("\n  ЭТО ВЕРХНЯЯ ОЦЕНКА. Она предполагает монитор, который узнаёт")
    print("  достижение цели НЕ ЗАПУСКАЯ VLA. Без такого монитора пропустить")
    print("  вызов нельзя вовсе, и этот вопрос здесь не решается.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(dict(by_event=res, n_episodes=n_used,
                       mean_episode_len=float(np.mean(ep_len)),
                       gate_horizon=args.gate_horizon, script_sha1=sha,
                       argv=vars(args)),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}  (sha {sha})")


if __name__ == "__main__":
    main()
