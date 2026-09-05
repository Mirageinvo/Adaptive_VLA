"""K-10f: две диагностики, снимающие неоднозначность отрицательного результата.

K-10d показал промах контроллера в 2.2-4.1 раза мимо matched coarse24. Я
объявил это нехваткой ИНФОРМАЦИИ и закрыл направление. Разбор показал, что
такой вывод не следует: эксперимент не различает три причины — неверную
границу фазы, недоопределённый вход и недостаток ёмкости. Здесь измеряются
первые две; третья снимается в K-10d повторным прогоном с полным входом.

--- ПРОБА 1: ВРЕМЕННОЕ ВЫРАВНИВАНИЕ ---

ВОПРОС. Что лучше объясняет смещение `state[t+1] - state[t]` — команда
`action[t]` или `action[t-1]`? От ответа зависит, какому шагу принадлежит
событие, а значит и семантика границы фазы.

ПОЧЕМУ ЭТО НЕ ПРИДИРКА. `grip_events` возвращает ПЕРВЫЙ индекс нового знака.
При `side="left"` строка этого момента получает остаток 0 (цель равна текущей
позе, смещение нулевое) и одновременно целевое действие УЖЕ новой фазы. Такая
строка внутренне противоречива, и все они лежат в ближнем бакете — том самом,
по которому выносится вердикт.

ЧТО ДЕЛАЕТСЯ. Развёртка по задержке от -2 до +2: для каждой считается R^2
регрессии смещения позиции на команду с этой задержкой. Побеждает та, при
которой действие объясняет последующее движение.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ, записано ДО запуска:
  * лучший лаг 0 -> `action[t]` порождает `state[t+1]-state[t]`, событие
    принадлежит шагу своего первого нового знака, и согласованная граница —
    `side="right"`; строки с нулевым остатком при `side="left"` подлежат
    исключению;
  * лучший лаг +1 -> команда исполняется с задержкой, событие следует
    сдвинуть на шаг назад, и `side="left"` согласован;
  * R^2 у обоих ниже 0.3 -> связь команды с движением слабая, и вся
    постановка «контроллер выдаёт приращение позы» под вопросом.

--- ПРОБА 2: ЁМКОСТЬ ПРОТИВ ИНФОРМАЦИИ ---

ВОПРОС. Может ли та же сеть переобучить МАЛУЮ выборку почти до нуля? Если да,
ёмкости достаточно, и провал на полной выборке объясняется тем, что вход не
определяет выход. Если нет, виновата сеть или оптимизация, и вывод про
информацию неправомерен.

ПОЧЕМУ ЭТО НУЖНО. Я обосновывал «это не ёмкость» отсутствием разрыва между
обучением и валидацией. Отсутствие разрыва одинаково совместимо с
информационным пределом И с недообучением, так что довод был неверен. Кроме
того, я сравнивал train MSE по шести каналам с валидационным RMS позиции
ближнего бакета — разные величины.

ЧТО ДЕЛАЕТСЯ. Обучение до сходимости на 256 и 1024 строках, потом развёртка по
ёмкости на полной выборке. Печатается ОДНА И ТА ЖЕ метрика везде.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ:
  * малая выборка переобучается ниже 0.02 RMS -> ёмкости хватает, причина в
    входе;
  * не переобучается -> виновата сеть или оптимизация, и отрицательный
    результат K-10d о информации ничего не говорит;
  * развёртка по ёмкости на полной выборке не снижает ошибку -> ещё один
    довод за информационный предел.

Запуск:
    python3 experiments/k10f_diagnostics.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k10f_diagnostics.py --ckpt <base> \\
        --cache data/k9_teacher_150k.npz --n-ep 200 --out data/k10f.json
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np


def lag_r2(disp, act, lag):
    """R^2 регрессии смещения на команду со сдвигом `lag`.

    СОГЛАШЕНИЕ О ЗНАКЕ: `lag` — на сколько шагов действие ОТСТАЁТ, то есть
    сравнивается `disp[t]` с `act[t - lag]`. Лаг +1 означает «смещение
    объясняется командой ПРЕДЫДУЩЕГО шага». Первая версия сдвигала индекс в
    другую сторону, и задержка выходила отрицательной, вопреки собственной
    докстроке.

    Коэффициент подбирается наименьшими квадратами: масштаб между нормированной
    командой и метрами нам не известен и не важен, важна объяснённая доля
    дисперсии.
    """
    n = len(disp)
    lo, hi = max(0, lag), min(n, n + lag)
    if hi - lo < 10:
        return None
    d, a = disp[lo:hi], act[lo - lag:hi - lag]
    den = float((a ** 2).sum())
    if den <= 0:
        return None
    k = float((d * a).sum()) / den
    ss_res = float(((d - k * a) ** 2).sum())
    ss_tot = float(((d - d.mean(0)) ** 2).sum())
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else None


def read_alignment(best_lag, r2):
    if r2 is None or r2 < 0.3:
        return ("СВЯЗЬ СЛАБАЯ: команда плохо объясняет движение, постановка "
                "под вопросом")
    if best_lag == 0:
        return ("ЛАГ 0: action[t] порождает state[t+1]-state[t]; согласованная "
                "граница — side=\"right\", строки с нулевым остатком при "
                "side=\"left\" противоречивы")
    if best_lag == 1:
        return ("ЛАГ +1: команда исполняется с задержкой, событие сдвигается "
                "на шаг назад, side=\"left\" согласован")
    return f"ЛУЧШИЙ ЛАГ {best_lag:+d}: неожиданно, разбираться отдельно"


def read_capacity(small_rms, thr=0.02):
    if small_rms is None:
        return "недействительно"
    if small_rms <= thr:
        return ("ЁМКОСТИ ХВАТАЕТ: малая выборка переобучается, значит причина "
                "провала во ВХОДЕ, а не в сети")
    return ("ЁМКОСТИ НЕ ХВАТАЕТ или оптимизация не сходится: вывод K-10d об "
            "информации неправомерен")


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import goal_dataset as gd
    gd.selftest()

    # ЛАГ НАХОДИТСЯ ТАМ, ГДЕ ОН ЗАЛОЖЕН. Синтетика: смещение порождается
    # командой предыдущего шага, значит выигрывает лаг +1.
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, size=(400, 3))
    disp = np.zeros_like(a)
    disp[1:] = 0.05 * a[:-1] + rng.normal(0, 1e-4, size=(399, 3))
    r2 = {l: lag_r2(disp, a, l) for l in (-2, -1, 0, 1, 2)}
    best = max((l for l in r2 if r2[l] is not None), key=lambda l: r2[l])
    assert best == 1, (best, r2)
    assert r2[1] > 0.99 and r2[0] < 0.1, r2

    # И при нулевой задержке — лаг 0.
    disp0 = 0.05 * a + rng.normal(0, 1e-4, size=(400, 3))
    r20 = {l: lag_r2(disp0, a, l) for l in (-1, 0, 1)}
    assert max(r20, key=lambda l: r20[l]) == 0, r20

    # Чистый шум даёт низкий R^2 на всех лагах — не выбираем лучший из мусора.
    noise = rng.normal(0, 1.0, size=(400, 3))
    rn = {l: lag_r2(noise, a, l) for l in (-1, 0, 1)}
    assert max(v for v in rn.values() if v is not None) < 0.1, rn

    assert "side=\"right\"" in read_alignment(0, 0.9)
    assert "side=\"left\"" in read_alignment(1, 0.9)
    assert "СЛАБАЯ" in read_alignment(0, 0.1)
    assert "ХВАТАЕТ" in read_capacity(0.01)
    assert "НЕ ХВАТАЕТ" in read_capacity(0.2)
    print("самопроверка k10f пройдена: лаг находится там, где заложен, шум "
          "не даёт ложного лага, правила чтения на четырёх исходах")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-ep", type=int, default=200)
    ap.add_argument("--event", default="union")
    ap.add_argument("--small-sizes", default="256,1024")
    ap.add_argument("--small-epochs", type=int, default=2000)
    ap.add_argument("--caps", default="256x2,512x2,1024x4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--rich", action="store_true",
                    help="полный вход: приращение, прошлое действие, остаток, "
                         "идентификатор задачи")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k10f.json")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k10f sha1 {sha}")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download

    import actioncodec  # noqa: F401
    import goal_dataset as gd
    import goal_events as ge
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       process_state)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    z = np.load(args.cache, allow_pickle=True)
    ep_split, ep_task = {}, {}
    for e, s, tk in zip(z["episode"], z["split"], z["task"]):
        ep_split[int(e)] = str(s)
        ep_task[int(e)] = str(tk)
    train_ep = sorted({e for e, s in ep_split.items() if s == "train"})
    rng = np.random.default_rng(args.seed)
    if len(train_ep) > args.n_ep:
        train_ep = sorted(rng.choice(train_ep, args.n_ep, replace=False).tolist())
    tasks = sorted({ep_task[e] for e in train_ep})
    tmap = {t: i for i, t in enumerate(tasks)}
    print(f"эпизодов {len(train_ep)}, различных задач {len(tasks)}")

    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    def to_codec_space(a):
        a = np.asarray(a, np.float64).copy()
        a[..., :-1] = a[..., :-1] / max_act_q[..., :-1]
        a[..., -1] = -a[..., -1]
        return np.clip(a, -1.0, 1.0)

    rid, rev = "physical-intelligence/libero", "v2.0"
    lags = [-2, -1, 0, 1, 2]
    lag_num = {l: 0.0 for l in lags}
    lag_den = {l: 0.0 for l in lags}
    S, G, Y = [], [], []
    zero_rem, near_rows = 0, 0

    for i, e in enumerate(train_ep):
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/"
                            f"episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        tab = pq.read_table(f)
        a = to_codec_space(np.asarray(tab.column("actions").to_pylist(),
                                      np.float32))
        st = np.asarray(tab.column("state").to_pylist(), np.float32)
        if st.shape[1] == len(STATE_Q01) + 1:
            st = process_state(st)
        if len(a) < 20:
            continue
        # --- проба 1 ---------------------------------------------------------
        disp = st[1:, :3] - st[:-1, :3]
        for l in lags:
            r = lag_r2(disp, a[:len(disp), :3], l)
            if r is not None:
                lag_num[l] += r * len(disp)
                lag_den[l] += len(disp)
        # --- проба 2: тот же вход, что у K-10d -------------------------------
        ev, typ, _ = ge.label(a, st[:, :3], kind=args.event)
        tau, ttyp, rem = ge.targets(a, ev, typ)
        t = np.arange(len(a) - 1)
        zero_rem += int((rem[t] == 0).sum())
        near_rows += int((rem[t] <= 8).sum())
        sn = lambda x: (x - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2 - 1 \
            if x.shape[1] == len(STATE_Q01) else x
        s_, g_, y_ = gd.build(a, st, tau, ttyp, rem, t, state_norm=sn,
                              task_id=tmap[ep_task[e]], rich=args.rich,
                              n_task=len(tasks))
        S.append(s_); G.append(g_); Y.append(y_)
        if (i + 1) % 50 == 0:
            print(f"  эпизодов {i + 1}/{len(train_ep)}", flush=True)

    S, G, Y = np.concatenate(S), np.concatenate(G), np.concatenate(Y)
    r2 = {l: (lag_num[l] / lag_den[l] if lag_den[l] else None) for l in lags}
    best = max((l for l in lags if r2[l] is not None), key=lambda l: r2[l])

    print(f"\n=== ПРОБА 1: временное выравнивание ===")
    for l in lags:
        mark = "  <-- лучший" if l == best else ""
        print(f"  лаг {l:+d}: R^2 = "
              + (f"{r2[l]:.4f}" if r2[l] is not None else "н/д") + mark)
    print(f"\n  {read_alignment(best, r2[best])}")
    print(f"\n  строк с нулевым остатком: {zero_rem} "
          f"({zero_rem / max(len(S), 1):.2%} всех, "
          f"{zero_rem / max(near_rows, 1):.2%} ближнего бакета)")

    # --- проба 2 --------------------------------------------------------------
    din, dout = S.shape[1] + G.shape[1], 6
    X = np.concatenate([S, G], 1)
    print(f"\n=== ПРОБА 2: ёмкость против информации ===")
    print(f"  троек {len(X)}, вход {din} признаков"
          + (" (полный)" if args.rich else " (как в K-10d)"))

    def mk(hid, depth):
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, hid), nn.GELU()]
            d = hid
        return nn.Sequential(*layers, nn.Linear(d, dout)).to(dev)

    def rms(net, xs, ys):
        net.eval()
        with torch.no_grad():
            p = net(torch.from_numpy(xs).to(dev)).cpu().numpy()
        return float(np.sqrt(((p - ys[:, :6]) ** 2).mean()))

    res = dict(alignment=dict(r2={str(k): v for k, v in r2.items()},
                              best_lag=int(best),
                              verdict=read_alignment(best, r2[best]),
                              zero_remaining=zero_rem,
                              zero_frac=zero_rem / max(len(S), 1)),
               overfit={}, capacity={})
    small_rms = None
    for n in [int(x) for x in args.small_sizes.split(",")]:
        idx = np.random.default_rng(0).choice(len(X), min(n, len(X)),
                                              replace=False)
        xs, ys = X[idx], Y[idx]
        torch.manual_seed(0)
        net = mk(256, 2)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        xt = torch.from_numpy(xs).to(dev)
        yt = torch.from_numpy(ys[:, :6]).to(dev)
        for _ in range(args.small_epochs):
            net.train()
            loss = ((net(xt) - yt) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        e = rms(net, xs, ys)
        res["overfit"][str(n)] = e
        print(f"  переобучение на {n} строках: RMS позы {e:.4f}")
        if small_rms is None:
            small_rms = e
    print(f"\n  {read_capacity(small_rms)}")

    # развёртка по ёмкости на полной выборке, та же метрика
    cut = int(0.85 * len(X))
    per = np.random.default_rng(1).permutation(len(X))
    tr, va = per[:cut], per[cut:]
    print(f"\n  развёртка по ёмкости (обучение {len(tr)}, отложено {len(va)}):")
    for spec in args.caps.split(","):
        hid, depth = (int(v) for v in spec.split("x"))
        torch.manual_seed(0)
        net = mk(hid, depth)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        rg = np.random.default_rng(0)
        for _ in range(args.epochs):
            net.train()
            o = rg.permutation(tr)
            for i0 in range(0, len(o), args.batch):
                s_ = o[i0:i0 + args.batch]
                loss = ((net(torch.from_numpy(X[s_]).to(dev))
                         - torch.from_numpy(Y[s_][:, :6]).to(dev)) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        etr, eva = rms(net, X[tr], Y[tr]), rms(net, X[va], Y[va])
        res["capacity"][spec] = dict(train=etr, val=eva)
        print(f"    {spec:>9}: обучение {etr:.4f}, отложено {eva:.4f}")

    print(f"\n  ЧИТАТЬ ТАК: если малая выборка переобучается, а развёртка по "
          f"ёмкости\n  ошибку не снижает — предел информационный. Если малая "
          f"НЕ переобучается —\n  вывод K-10d об информации неправомерен "
          f"вовсе.")
    res["script_sha1"] = sha
    res["rich"] = bool(args.rich)
    res["argv"] = vars(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
