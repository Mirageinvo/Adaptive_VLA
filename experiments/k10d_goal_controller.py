"""K-10d: офлайновая достаточность ORACLE-цели для локального контроллера.

ЧТО ЭТО НЕ ИЗМЕРЯЕТ, И ЭТО ГЛАВНОЕ. Скрипт получает ИСТИННУЮ будущую цель и
работает на состояниях ДЕМОНСТРАЦИИ, то есть в режиме teacher forcing. Даже
идеальная офлайновая ошибка не проверяет, сможет ли контроллер выполнить своё
действие, попасть в слегка ошибочное состояние, снова скорректироваться и
пройти так 30-50 шагов, не уйдя с траектории. Замкнутый цикл здесь НЕ
измеряется, и называть результат «сокращением вызовов» нельзя.

Это отборочный гейт: если контроллер не справляется даже с истинной целью и
без накопления ошибки, дальше идти незачем. Порядок после него — симулятор с
oracle-целью, и только затем с предсказанной целью и монитором.

ЗАЧЕМ. K-10c дал теоретический запас по частоте вызовов: 4.5x по объединению
событий, 6.2x если значимы только события схвата. Запас реализуем ровно
настолько, насколько дешёвый контроллер способен покрыть фазу.

РАЗМЕТКА СОБЫТИЙ ОБЩАЯ — из `goal_events`, тем же вызовом, что в K-10e. Прежде
она была продублирована и разъехалась: опора строилась по объединению и
`side="left"`, а контроллер по схвату и `side="right"`, поэтому остатки,
бакеты и пороги относились к разным определениям цели.

ЧТО СТРОИТСЯ:
    вход:  состояние s_t и цель ОТНОСИТЕЛЬНО него
    выход: ОДНО следующее действие a_t
Цель: смещение позиции, поворот R_goal * R_cur^T, знак схвата в момент
события и ТИП события (схват / остановка / конец эпизода). Без типа конец
эпизода неотличим от схвата; без знака контроллер не знает, чем фаза
заканчивается — а именно это решает исход.

ОДНО ДЕЙСТВИЕ, А НЕ ЧАНК: чанк a[t:t+8] при трёх шагах до события содержал бы
пять действий следующей фазы, цель которой контроллеру не сообщена, и ближний
бакет портился бы искусственно.

ОЦЕНКА НА ТЕХ ЖЕ СТРОКАХ, ЧТО ОПОРА. Список (эпизод, шаг) берётся из опоры
K-10e, и посчитанный здесь остаток до события СВЕРЯЕТСЯ с записанным там —
расхождение хотя бы в одной строке останавливает разбор. Собственное
случайное разбиение запрещено: обучающие эпизоды берутся из манифеста K-9a.

ГЕЙТ ЖЁСТКИЙ И ПО ТРЁМ ВЕЛИЧИНАМ: бакет годен, если позиция, вращение И знак
схвата не хуже matched coarse24. Прежняя версия проверяла только смешанную
позу, и контроллер мог полностью провалить схват, пройдя гейт, — при том что
знак схвата у нас самый обоснованный механизм успеха (K-6h).

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ, записано ДО запуска:
  * годен на 32 шагах и дальше -> офлайновый размах >= 4x, идём в симулятор;
  * годен только на 8 и ближе -> запаса нет, направление закрывается;
  * между -> размах равен d/H, и это результат.
Достаточность считается НЕПРЕРЫВНЫМ ПРЕФИКСОМ: ошибка растёт с удалением,
годный дальний бакет через провалившийся ближний не засчитывается.

Запуск:
    python3 experiments/k10d_goal_controller.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k10d_goal_controller.py \\
        --ckpt <base> --cache data/k9_teacher_150k.npz \\
        --baseline-val data/k10e_union_val.json \\
        --baseline-test data/k10e_union_test.json --out data/k10d_union
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

H_CALL = 8


def aa_to_R(v):
    """Axis-angle -> матрица поворота (Родриг), пакетно."""
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.divide(v, np.where(th > 1e-12, th, 1.0))
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def R_to_aa(R):
    """Матрица поворота -> axis-angle, с устойчивой ветвью около pi."""
    R = np.asarray(R, np.float64)
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(tr)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.where(n > 1e-12, n, 1.0) * th[..., None]
    # ДВА ВЫРОЖДЕНИЯ, А НЕ ОДНО: кососимметричная часть равна нулю и при
    # theta=0, и при theta=pi. Одна ветка «малый угол» возвращала бы для
    # поворота на pi нулевой вектор.
    flat, Rf = out.reshape(-1, 3), R.reshape(-1, 3, 3)
    trf, nf, vf = tr.reshape(-1), n.reshape(-1), v.reshape(-1, 3)
    for i in np.where(nf < 1e-8)[0]:
        if trf[i] > 0:
            flat[i] = vf[i] / 2.0
            continue
        A = (Rf[i] + np.eye(3)) / 2.0        # при theta=pi это a a^T
        k = int(np.argmax(np.diag(A)))
        ak = math.sqrt(max(A[k, k], 0.0))
        if ak < 1e-12:
            flat[i] = 0.0
            continue
        u = A[:, k] / ak
        flat[i] = u / max(np.linalg.norm(u), 1e-12) * np.pi
    return flat.reshape(out.shape)


def rel_goal(pos_t, aa_t, pos_g, aa_g):
    """Цель относительно текущей позы: смещение и поворот R_g * R_t^T."""
    dR = aa_to_R(aa_g) @ np.swapaxes(aa_to_R(aa_t), -1, -2)
    return np.concatenate([pos_g - pos_t, R_to_aa(dR)], axis=-1)


def prefix_ok(flags, edges):
    """До какого удаления контроллер годен — НЕПРЕРЫВНЫМ префиксом."""
    d = 0
    for i, ok in enumerate(flags):
        if not ok:
            break
        d = edges[min(i, len(edges) - 1)]
    return d


def gate(row, base, margin_pose, margin_grip):
    """Годен ли бакет: позиция И вращение И знак схвата не хуже опоры."""
    return (row["pos"] <= base["pos"] + margin_pose
            and row["rot"] <= base["rot"] + margin_pose
            and row["grip"] <= base["grip"] + margin_grip)


def verdict(span):
    if span >= 4:
        return "офлайновый гейт пройден, идём в симулятор с oracle-целью"
    if span <= 1:
        return "запаса нет, направление закрывается"
    return f"офлайновый размах {span:.1f}x — это и есть результат"


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import goal_events as ge
    ge.selftest()

    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.5, size=(50, 3))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True) * rng.uniform(
        0.05, 2.5, size=(50, 1))
    assert np.abs(R_to_aa(aa_to_R(v)) - v).max() < 1e-8
    R = aa_to_R(v)
    assert np.abs(R @ np.swapaxes(R, -1, -2) - np.eye(3)).max() < 1e-9
    # ПОВОРОТ РОВНО НА PI: прежняя версия возвращала здесь почти ноль.
    for ax in ([0, 0, 1.0], [1.0, 0, 0], [0.6, 0.8, 0.0]):
        w = np.array([ax]) * np.pi
        got = R_to_aa(aa_to_R(w))
        assert abs(np.linalg.norm(got) - np.pi) < 1e-6
        assert np.abs(aa_to_R(got) - aa_to_R(w)).max() < 1e-9

    p = rng.normal(size=(10, 3)); a = rng.normal(0, 0.3, size=(10, 3))
    assert np.abs(rel_goal(p, a, p, a)).max() < 1e-8
    sh = rng.normal(size=(1, 3))
    assert np.abs(rel_goal(p, a, p + 1.0, a)
                  - rel_goal(p + sh, a, p + sh + 1.0, a)).max() < 1e-9
    a1, a2 = np.array([[0.0, 0.0, 3.0]]), np.array([[0.0, 3.0, 0.0]])
    honest = rel_goal(np.zeros((1, 3)), a1, np.zeros((1, 3)), a2)[:, 3:]
    assert np.abs(honest - (a2 - a1)).max() > 0.5, "вращения не вычитаются"

    ed = [8, 16, 32, 48]
    assert prefix_ok([True, True, False, True, True], ed) == 16
    assert prefix_ok([False] + [True] * 4, ed) == 0
    assert prefix_ok([True] * 5, ed) == 48

    # ГЕЙТ ПО ТРЁМ ВЕЛИЧИНАМ. Провал знака схвата обязан заваливать бакет,
    # даже когда поза идеальна: прежняя версия смотрела только на позу.
    b = dict(pos=0.05, rot=0.05, grip=0.02)
    assert gate(dict(pos=0.04, rot=0.04, grip=0.02), b, 0.0, 0.005)
    assert not gate(dict(pos=0.001, rot=0.001, grip=0.30), b, 0.0, 0.005)
    assert not gate(dict(pos=0.30, rot=0.001, grip=0.01), b, 0.0, 0.005)
    assert not gate(dict(pos=0.001, rot=0.30, grip=0.01), b, 0.0, 0.005)

    # ИНТЕГРАЦИОННЫЙ ПРОХОД: синтетическая опора -> гейт -> вердикт.
    names = ge.bucket_names(ed)
    base = {n: dict(pos=0.05, rot=0.05, grip=0.02) for n in names}
    good = [gate(dict(pos=0.01, rot=0.01, grip=0.01), base[n], 0.0, 0.005)
            for n in names]
    assert verdict(prefix_ok(good, ed) / H_CALL).startswith("офлайновый гейт")
    bad = [gate(dict(pos=0.9, rot=0.9, grip=0.9), base[n], 0.0, 0.005)
           for n in names]
    assert "запаса нет" in verdict(prefix_ok(bad, ed) / H_CALL)
    assert abs(prefix_ok([True, True, False, False, False], ed) / H_CALL
               - 2.0) < 1e-12
    # Контроллер, идеальный по позе и провальный по схвату, обязан НЕ пройти.
    only_pose = [gate(dict(pos=0.001, rot=0.001, grip=0.9), base[n], 0.0,
                      0.005) for n in names]
    assert "запаса нет" in verdict(prefix_ok(only_pose, ed) / H_CALL)

    print("самопроверка k10d пройдена (версия «общая разметка, жёсткий "
          "гейт»): pi, инвариантность цели, вращения не вычитаются, префикс, "
          "гейт по трём величинам, сквозной проход опора->вердикт")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--baseline-val")
    ap.add_argument("--baseline-test")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--margin-pose", type=float, default=0.0)
    ap.add_argument("--margin-grip", type=float, default=0.005)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lrs", default="1e-3,3e-3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--grip-loss-weight", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-train-ep", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k10d")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    for need in ("ckpt", "baseline_val", "baseline_test"):
        if not getattr(args, need):
            raise SystemExit(
                f"нужен --{need.replace('_', '-')}. Опоры обязательны: без "
                f"matched coarse24 на тех же строках сравнивать не с чем, а "
                f"перенесённое 0.0478 измерено на другой выборке и против "
                f"учителя.")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k10d sha1 {sha}")
    os.makedirs(args.out, exist_ok=True)

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download

    import actioncodec  # noqa: F401
    import goal_events as ge
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       process_state)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- опоры ----------------------------------------------------------------
    bases = {}
    for part, path in (("val", args.baseline_val),
                       ("test", args.baseline_test)):
        bj = json.load(open(path))
        for f in ("buckets", "edges", "event", "eval_keys", "eval_remaining",
                  "split"):
            if f not in bj:
                raise SystemExit(f"{path}: нет поля «{f}» — опора собрана "
                                 f"старой версией k10e, пересоберите")
        if bj["split"] != part:
            raise SystemExit(f"{path}: часть «{bj['split']}», ждали «{part}»")
        bases[part] = bj
    if bases["val"]["event"] != bases["test"]["event"]:
        raise SystemExit(f"опоры по разным событиям: {bases['val']['event']} "
                         f"и {bases['test']['event']}")
    if bases["val"]["edges"] != bases["test"]["edges"]:
        raise SystemExit("опоры по разным границам бакетов")
    kind = bases["val"]["event"]
    edges = [int(x) for x in bases["val"]["edges"]]
    names = ge.bucket_names(edges)
    ev_par = bases["val"].get("argv", {})
    print(f"опоры: событие «{kind}», границы {edges}; val "
          f"{bases['val']['n_rows']} строк, test {bases['test']['n_rows']}")

    # --- разбиение из манифеста K-9a -----------------------------------------
    z = np.load(args.cache, allow_pickle=True)
    ep_split = {}
    for e, s in zip(z["episode"], z["split"]):
        ep_split[int(e)] = str(s)
    train_ep = sorted({e for e, s in ep_split.items() if s == "train"})
    rng = np.random.default_rng(args.seed)
    if len(train_ep) > args.max_train_ep:
        train_ep = sorted(rng.choice(train_ep, args.max_train_ep,
                                     replace=False).tolist())
    print(f"обучающих эпизодов из манифеста: {len(train_ep)}")

    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    def to_codec_space(a):
        """Та же нормировка и ОБРЕЗКА, что в K-9a и K-10e."""
        a = np.asarray(a, np.float64).copy()
        a[..., :-1] = a[..., :-1] / max_act_q[..., :-1]
        a[..., -1] = -a[..., -1]
        return np.clip(a, -1.0, 1.0)

    rid, rev = "physical-intelligence/libero", "v2.0"
    cache_ep = {}

    def load_ep(e):
        if e in cache_ep:
            return cache_ep[e]
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/"
                            f"episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        tab = pq.read_table(f)
        a = to_codec_space(np.asarray(tab.column("actions").to_pylist(),
                                      np.float32))
        st = np.asarray(tab.column("state").to_pylist(), np.float32)
        if st.shape[1] == len(STATE_Q01) + 1:
            st = process_state(st)
        ev, typ, _ = ge.label(
            a, st[:, :3], kind=kind,
            speed_frac=ev_par.get("speed_frac", 0.3),
            min_dwell=ev_par.get("min_dwell", 3),
            min_travel=ev_par.get("min_travel", 0.02),
            merge_tol=ev_par.get("merge_tol", 4))
        tau, ttyp, rem = ge.targets(a, ev, typ)
        cache_ep[e] = (a, st, tau, ttyp, rem)
        return cache_ep[e]

    def build(e, steps):
        a, st, tau, ttyp, rem = load_ep(e)
        t = np.asarray(steps, np.int64)
        g = rel_goal(st[t, :3], st[t, 3:6], st[tau[t], :3], st[tau[t], 3:6])
        # ЦЕЛЬ НЕСЁТ ЗНАК СХВАТА И ТИП СОБЫТИЯ одним горячим вектором.
        oh = np.zeros((len(t), 3))
        oh[np.arange(len(t)), np.clip(ttyp[t], 0, 2)] = 1.0
        g = np.concatenate([g, np.sign(a[tau[t], 6:7]), oh], 1)
        sn = ((st[t] - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2 - 1
              if st.shape[1] == len(STATE_Q01) else st[t])
        return (sn.astype(np.float32), g.astype(np.float32),
                a[t].astype(np.float32), rem[t])

    # --- обучающая выборка: все шаги обучающих эпизодов -----------------------
    S, G, Y = [], [], []
    for i, e in enumerate(train_ep):
        a, *_ = load_ep(e)
        if len(a) < 4:
            continue
        s_, g_, y_, _ = build(e, np.arange(len(a) - 1))
        S.append(s_); G.append(g_); Y.append(y_)
        if (i + 1) % 100 == 0:
            print(f"  обучающих эпизодов {i + 1}/{len(train_ep)}", flush=True)
    S, G, Y = np.concatenate(S), np.concatenate(G), np.concatenate(Y)
    print(f"обучающих троек {len(S)}")

    # --- оценочные выборки: РОВНО строки опоры --------------------------------
    ev_sets = {}
    for part in ("val", "test"):
        bj = bases[part]
        keys = [(int(a_), int(b_)) for a_, b_ in bj["eval_keys"]]
        want = np.asarray(bj["eval_remaining"], np.int64)
        by_ep = {}
        for e, s_ in keys:
            by_ep.setdefault(e, []).append(s_)
        piece, order = {}, []
        for e in sorted(by_ep):
            steps = sorted(by_ep[e])
            piece[e] = build(e, steps)
            order += [(e, s_) for s_ in steps]
        idx_of = {k: i for i, k in enumerate(order)}
        perm = np.array([idx_of[k] for k in keys])
        cat = lambda j: np.concatenate([piece[e][j] for e in sorted(by_ep)])
        Se, Ge, Ye, Re = cat(0)[perm], cat(1)[perm], cat(2)[perm], cat(3)[perm]
        Ep = np.concatenate([np.full(len(piece[e][0]), e)
                             for e in sorted(by_ep)])[perm]
        # СВЕРКА ОСТАТКА С ОПОРОЙ. Если разметка здесь и там разошлась, бакеты
        # и пороги относятся к разным целям — сравнение недействительно.
        bad = int((Re != want).sum())
        if bad:
            raise SystemExit(
                f"часть «{part}»: у {bad} из {len(Re)} строк остаток до "
                f"события не совпал с опорой. Разметка разошлась.")
        print(f"  «{part}»: {len(Se)} строк, остатки совпали со всеми")
        ev_sets[part] = dict(S=Se, G=Ge, Y=Ye, R=Re, E=Ep)

    # --- контроллер: отдельная голова схвата ---------------------------------
    din = S.shape[1] + G.shape[1]

    class Ctrl(nn.Module):
        """Общий ствол, отдельные головы позы и знака схвата.

        Раньше был общий MSE по семи каналам: схват со значениями около +-1
        доминировал над малыми приращениями позы, а знак при этом в вердикт
        не входил вовсе.
        """

        def __init__(self, din, hid):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(din, hid), nn.GELU(),
                                      nn.Linear(hid, hid), nn.GELU())
            self.pose = nn.Linear(hid, 6)
            self.grip = nn.Linear(hid, 1)

        def forward(self, x):
            h = self.body(x)
            return self.pose(h), self.grip(h).squeeze(-1)

    def metrics(net, d):
        net.eval()
        P, Gp = [], []
        with torch.no_grad():
            for i0 in range(0, len(d["S"]), args.batch):
                x = torch.from_numpy(np.concatenate(
                    [d["S"][i0:i0 + args.batch],
                     d["G"][i0:i0 + args.batch]], 1)).to(dev)
                p, g = net(x)
                P.append(p.cpu().numpy()); Gp.append(g.cpu().numpy())
        P, Gp = np.concatenate(P), np.concatenate(Gp)
        dd = P - d["Y"][:, :6]
        return dict(pos=np.linalg.norm(dd[:, :3], axis=1) / math.sqrt(3),
                    rot=np.linalg.norm(dd[:, 3:6], axis=1) / math.sqrt(3),
                    grip=(np.sign(Gp) != np.sign(d["Y"][:, 6])).astype(float))

    def curve(m, d):
        bi = ge.bucketize(d["R"], edges)
        out = {}
        for b, nm in enumerate(names):
            k = bi == b
            if not k.any():
                continue
            out[nm] = dict(n=int(k.sum()),
                           pos=float(np.sqrt((m["pos"][k] ** 2).mean())),
                           rot=float(np.sqrt((m["rot"][k] ** 2).mean())),
                           grip=float(m["grip"][k].mean()))
        return out

    best = None
    for lr in [float(x) for x in args.lrs.split(",")]:
        for sd in [int(x) for x in args.seeds.split(",")]:
            torch.manual_seed(sd)
            net = Ctrl(din, args.hidden).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=lr)
            rg = np.random.default_rng(sd)
            local = None
            for ep in range(0, args.epochs + 1):
                if ep:                       # ЭПОХА 0 — ПОЛНОПРАВНЫЙ КАНДИДАТ
                    net.train()
                    o = rg.permutation(len(S))
                    for i0 in range(0, len(o), args.batch):
                        s_ = o[i0:i0 + args.batch]
                        x = torch.from_numpy(np.concatenate(
                            [S[s_], G[s_]], 1)).to(dev)
                        p, g = net(x)
                        y = torch.from_numpy(Y[s_]).to(dev)
                        loss = ((p - y[:, :6]) ** 2).mean() + \
                            args.grip_loss_weight * \
                            nn.functional.binary_cross_entropy_with_logits(
                                g, (y[:, 6] > 0).float())
                        opt.zero_grad(set_to_none=True)
                        loss.backward(); opt.step()
                mv = metrics(net, ev_sets["val"])
                cv = curve(mv, ev_sets["val"])
                # ОТБОР: сначала ЖЁСТКИЙ гейт по схвату на валидации, среди
                # прошедших — лучший по позе. Вес схвата в потере на отбор не
                # влияет: он не должен подменять условие.
                passed = sum(
                    1 for nm in names if nm in cv
                    and cv[nm]["grip"] <= bases["val"]["buckets"][nm]["grip"]
                    + args.margin_grip)
                pose_v = float(np.sqrt((mv["pos"] ** 2 + mv["rot"] ** 2).mean()))
                key = (-passed, pose_v)
                if local is None or key < local[0]:
                    local = (key, ep)
                if best is None or key < best[0]:
                    best = (key, dict(lr=lr, seed=sd, epoch=ep),
                            {k: v.detach().clone()
                             for k, v in net.state_dict().items()})
            print(f"    lr={lr:g} сид={sd}: лучшая эпоха {local[1]}, "
                  f"бакетов по схвату {-local[0][0]}/{len(names)}, "
                  f"поза {local[0][1]:.4f}", flush=True)

    _, cfg, state = best
    net = Ctrl(din, args.hidden).to(dev)
    net.load_state_dict(state)
    print(f"\n  выбрано по валидации: {cfg}")

    # --- вердикт на test ------------------------------------------------------
    ct = curve(metrics(net, ev_sets["test"]), ev_sets["test"])
    bt = bases["test"]["buckets"]
    print(f"\n  {'удалённость':>13}{'строк':>7}{'позиция к/о':>21}"
          f"{'вращение к/о':>21}{'знак схвата к/о':>22}")
    flags, rows = [], {}
    for nm in names:
        if nm not in ct or nm not in bt:
            flags.append(False)
            continue
        c, b = ct[nm], bt[nm]
        ok = gate(c, b, args.margin_pose, args.margin_grip)
        flags.append(ok)
        rows[nm] = dict(ctrl=c, base=b, ok=ok)
        print(f"  {nm:>13}{c['n']:>7}"
              f"{c['pos']:>11.4f} /{b['pos']:>8.4f}"
              f"{c['rot']:>11.4f} /{b['rot']:>8.4f}"
              f"{c['grip']:>12.1%} /{b['grip']:>8.1%}"
              + ("  ok" if ok else "  --"))

    ok_upto = prefix_ok(flags, edges)
    span = ok_upto / H_CALL
    v = verdict(span)
    print(f"\n  ГЕЙТ: позиция И вращение И знак схвата не хуже matched "
          f"coarse24\n  (запас по позе {args.margin_pose}, по знаку "
          f"{args.margin_grip:.1%}).")
    print(f"  Годен вплоть до {ok_upto} шагов до события.")
    print(f"  ОФЛАЙНОВЫЙ РАЗМАХ при истинной цели: {span:.1f}x")
    print(f"  Это НЕ сокращение вызовов: цель oracle, ошибка не "
          f"накапливается,\n  замкнутого цикла нет. Его измерит только "
          f"симулятор.")
    print(f"  ВЕРДИКТ: {v}")

    p = os.path.join(args.out, "table.json")
    json.dump(dict(buckets=rows, ok_upto=ok_upto, offline_span=span,
                   verdict=v, cfg=cfg, event=kind, edges=edges,
                   n_train=int(len(S)), n_train_ep=len(train_ep),
                   baseline_val=args.baseline_val,
                   baseline_test=args.baseline_test,
                   script_sha1=sha, argv=vars(args)),
              open(p, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {p}")


if __name__ == "__main__":
    main()
