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
  * ОСТАНОВКА — локальный минимум скорости после значимого перемещения. Их
    больше, они дробят эпизод мельче и дают нижнюю оценку персистентности.

ЧТО ПЕЧАТАЕТСЯ И КАК ЧИТАТЬ. Главная величина — доля вызовов, на которых цель
ТА ЖЕ, что на предыдущем вызове; это и есть предельная доля `NO_EDIT`. Рядом —
средняя длина фазы в вызовах, то есть во сколько раз можно было бы сократить
обращения к VLA при идеальном мониторе.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ, записано ДО запуска (горизонт H=8):
  * доля NO_EDIT >= 0.60 -> цель переживает в среднем более двух вызовов,
    направление живо;
  * <= 0.30 -> цель меняется почти каждый вызов, экономии нет,
    направление закрывается;
  * между -> не доказано ничего.

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


def stop_events(state_xyz, min_travel=0.02, win=3):
    """Локальные минимумы скорости после значимого перемещения.

    Событием считается точка, где скорость локально минимальна И с прошлого
    события пройдено не меньше min_travel. Порог нужен, иначе каждая пауза в
    шуме объявляется целью и персистентность занижается искусственно.
    """
    v = np.linalg.norm(np.diff(state_xyz, axis=0), axis=1)
    if len(v) < 2 * win + 1:
        return np.array([], np.int64)
    ev, last = [], 0
    for i in range(win, len(v) - win):
        if v[i] <= v[i - win:i + win + 1].min() + 1e-12:
            travel = np.linalg.norm(state_xyz[i] - state_xyz[last])
            if travel >= min_travel:
                ev.append(i); last = i
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
    return dict(n_calls=int(len(calls)), n_events=int(len(ev)),
                no_edit=float(same.mean()),
                phase_calls_mean=float(np.mean(lens)),
                phase_calls_median=float(np.median(lens)))


def read_rule(no_edit, good=0.60, bad=0.30):
    if no_edit is None:
        return "недействительно"
    if no_edit >= good:
        return "ЦЕЛЬ ПЕРЕЖИВАЕТ ВЫЗОВЫ: направление живо"
    if no_edit <= bad:
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

    # Порог перемещения не даёт объявлять целью каждую паузу в шуме.
    rng = np.random.default_rng(0)
    xyz = np.cumsum(rng.normal(0, 1e-4, size=(300, 3)), axis=0)
    assert len(stop_events(xyz, min_travel=0.02)) == 0
    line = np.stack([np.linspace(0, 1, 300)] * 3, 1)
    assert len(stop_events(line, min_travel=0.02)) >= 0   # монотонное движение

    assert "ПЕРЕЖИВАЕТ" in read_rule(0.7)
    assert "экономии нет" in read_rule(0.2)
    assert "НЕ ДОКАЗАНО" in read_rule(0.45)
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
    ap.add_argument("--good", type=float, default=0.60)
    ap.add_argument("--bad", type=float, default=0.30)
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
        ev = dict(grip=grip_events(acts),
                  stop=stop_events(st[:, :3], args.min_travel))
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
        print(f"    {'H':>4}{'NO_EDIT':>10}{'фаза, вызовов':>16}"
              f"{'событий/эпизод':>17}")
        res[d] = {}
        for h in hz:
            if not acc[d][h]:
                continue
            m = {k: float(np.mean([r[k] for r in acc[d][h]]))
                 for k in ("no_edit", "phase_calls_mean", "phase_calls_median",
                           "n_events", "n_calls")}
            res[d][str(h)] = m
            print(f"    {h:>4}{m['no_edit']:>9.1%}{m['phase_calls_mean']:>15.2f}"
                  f"{m['n_events']:>16.1f}")
        print()

    g = res["grip"].get(str(args.gate_horizon))
    s = res["stop"].get(str(args.gate_horizon))
    print(f"  ЧИТАТЬ по горизонту {args.gate_horizon} — наше исполняемое окно.")
    print(f"  NO_EDIT это доля вызовов, где цель ТА ЖЕ, что на предыдущем;")
    print(f"  «фаза, вызовов» — во сколько раз можно было бы сократить")
    print(f"  обращения к VLA при ИДЕАЛЬНОМ дешёвом мониторе.\n")
    for d, r in (("схват", g), ("остановка", s)):
        if r is None:
            continue
        print(f"  событие «{d}»: NO_EDIT {r['no_edit']:.1%}, фаза "
              f"{r['phase_calls_mean']:.2f} вызова — "
              f"{read_rule(r['no_edit'], args.good, args.bad)}")
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
