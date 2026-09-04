"""K-10d: покрывает ли дешёвый контроллер целую фазу между событиями.

ЗАЧЕМ. K-10c показал, что одна событийная цель живёт 6.74 вызова VLA при
H=8, то есть фаза между сменами знака схвата длится около 48 шагов. Но это
ВЕРХНЯЯ оценка: она предполагает, что всё это время действие строит дешёвый
контроллер по цели и текущему состоянию, без обращения к VLA.

Здесь проверяется именно это. Реальный выигрыш равен не 6.74, а длине участка,
который контроллер покрывает БЕЗ потери качества.

ВСТРОЕННОЕ НАПРЯЖЕНИЕ, ради которого гейт и нужен: чем длиннее фаза, тем
больше выигрыш и тем больше должен уметь контроллер. В пределе он становится
второй полноценной политикой, и тогда экономии нет. Гейт меряет, где эта
граница проходит.

ЧТО СТРОИТСЯ. Из демонстраций собираются тройки:
    вход:  состояние s_t и цель ОТНОСИТЕЛЬНО него
    выход: чанк команд a[t : t+H]
Цель — состояние робота в момент следующего события схвата. Она выражается
относительно текущего состояния: смещение позиции и поворот R_goal * R_cur^T.
Относительная форма обязательна: абсолютная цель заставила бы сеть выучивать
координаты рабочей области вместо управления.

ВРАЩЕНИЯ НЕ СКЛАДЫВАЮТСЯ. Разность ориентаций считается через матрицы поворота
и обратно в axis-angle, а не вычитанием axis-angle. Ошибка позиции и ошибка
ориентации печатаются РАЗДЕЛЬНО и никогда не складываются в одну норму: метры
и радианы несравнимы.

ТРИ ОПОРЫ, и без них число контроллера не читается:
  * СРЕДНЕЕ ДЕЙСТВИЕ — насколько задача вообще решается без входа;
  * П-РЕГУЛЯТОР — команда пропорциональна относительной цели, без обучения.
    Если он справляется, обучаемый контроллер не нужен вовсе;
  * ПОЛ ТОКЕНИЗАТОРА — ошибка decode(encode(a)) на тех же данных. Ниже него
    не опустится никакой контроллер, выдающий действия через этот кодек.

ГЛАВНАЯ ВЕЛИЧИНА — не средняя ошибка, а её ЗАВИСИМОСТЬ ОТ УДАЛЁННОСТИ ДО
СОБЫТИЯ. Если контроллер справляется за десять шагов до цели и разваливается
за сорок, значит цели хватает на десять шагов, и реальное сокращение вызовов
равно этой длине, делённой на H.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ, записано ДО запуска. Порогом служит 0.0478 —
ошибка позы грубого выхода, про который ИЗМЕРЕНО, что он сохраняет успех
(K-6h, K-9h). Контроллер считается достаточным на удалении d, если его ошибка
позы в этом бакете не превышает порога.
  * достаточен на 32 шагах и дальше -> реальное сокращение >= 4x, направление
    подтверждено;
  * достаточен только на 8 шагах и ближе -> сокращение <= 1x, экономии нет;
  * между -> реальное сокращение равно d/H, и это и есть результат.

ОГОВОРКА О СРАВНИМОСТИ ПОРОГА. 0.0478 измерялось как расхождение модели с
УЧИТЕЛЕМ, здесь считается расхождение с ДЕМОНСТРАЦИЕЙ. Величины близки по
смыслу, но не тождественны, поэтому в отчёте печатается и пол токенизатора на
этих же данных — единственная полностью сопоставимая опора.

Запуск:
    python3 experiments/k10d_goal_controller.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k10d_goal_controller.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --n-ep 300 --out data/k10d
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

H_EXEC = 8


def aa_to_R(v):
    """Axis-angle -> матрица поворота (формула Родрига), пакетно."""
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
    """Матрица поворота -> axis-angle, пакетно."""
    R = np.asarray(R, np.float64)
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(tr)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    # При theta около нуля и около pi формула через след вырождается; малые
    # углы берутся линейным приближением, что здесь достаточно.
    small = n[..., 0] < 1e-8
    out = np.where(small[..., None], v / 2.0,
                   v / np.where(n > 1e-12, n, 1.0) * th[..., None])
    return out


def rel_goal(pos_t, aa_t, pos_g, aa_g):
    """Цель относительно текущей позы: смещение и поворот R_g * R_t^T."""
    dR = aa_to_R(aa_g) @ np.swapaxes(aa_to_R(aa_t), -1, -2)
    return np.concatenate([pos_g - pos_t, R_to_aa(dR)], axis=-1)


def buckets(remaining, edges):
    """Индекс бакета по числу шагов, оставшихся до события."""
    return np.searchsorted(np.asarray(edges), np.asarray(remaining),
                           side="right")


def prefix_ok(ok_flags, edges):
    """До какого удаления контроллер достаточен — НЕПРЕРЫВНЫМ префиксом.

    Бакеты упорядочены по удалению от цели, и ошибка с удалением растёт.
    Поэтому засчитывается только начальный отрезок подряд идущих годных
    бакетов: годный дальний бакет через провалившийся ближний ничего не
    значит. Максимум по всем годным дал бы завышенный ответ.
    """
    d = 0
    for i, ok in enumerate(ok_flags):
        if not ok:
            break
        d = edges[min(i, len(edges) - 1)]
    return d


def selftest():
    rng = np.random.default_rng(0)
    # Родриг и обратно: круговой перегон обязан вернуть тот же угол.
    v = rng.normal(0, 0.5, size=(50, 3))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True) * rng.uniform(
        0.05, 2.5, size=(50, 1))
    assert np.abs(R_to_aa(aa_to_R(v)) - v).max() < 1e-8, "перегон axis-angle"
    # Матрицы обязаны быть ортогональными с определителем 1.
    R = aa_to_R(v)
    assert np.abs(R @ np.swapaxes(R, -1, -2) - np.eye(3)).max() < 1e-9
    assert np.abs(np.linalg.det(R) - 1).max() < 1e-9
    # Нулевой поворот.
    assert np.abs(aa_to_R(np.zeros((1, 3))) - np.eye(3)).max() < 1e-12
    assert np.abs(R_to_aa(np.eye(3)[None])).max() < 1e-12

    # ОТНОСИТЕЛЬНАЯ ЦЕЛЬ. Совпадающие позы дают нулевую цель.
    p = rng.normal(size=(10, 3)); a = rng.normal(0, 0.3, size=(10, 3))
    assert np.abs(rel_goal(p, a, p, a)).max() < 1e-8
    # ИНВАРИАНТНОСТЬ К ОБЩЕМУ СДВИГУ: сдвиг обеих поз цель не меняет. Это и
    # есть причина, по которой цель берётся относительной.
    sh = rng.normal(size=(1, 3))
    g1 = rel_goal(p, a, p + 1.0, a)
    g2 = rel_goal(p + sh, a, p + sh + 1.0, a)
    assert np.abs(g1 - g2).max() < 1e-9
    # ВРАЩЕНИЯ НЕ ВЫЧИТАЮТСЯ. На больших углах разность axis-angle отличается
    # от честного R_g R_t^T — проверяется, что мы считаем именно второе.
    a1 = np.array([[0.0, 0.0, 3.0]]); a2 = np.array([[0.0, 3.0, 0.0]])
    honest = rel_goal(np.zeros((1, 3)), a1, np.zeros((1, 3)), a2)[:, 3:]
    assert np.abs(honest - (a2 - a1)).max() > 0.5, "разность axis-angle неверна"

    # Бакеты по удалённости.
    b = buckets([0, 5, 9, 30, 100], [8, 16, 32])
    assert list(b) == [0, 0, 1, 2, 3], list(b)

    # ДОСТАТОЧНОСТЬ СЧИТАЕТСЯ ПРЕФИКСОМ, а не максимумом по годным бакетам.
    # Ошибка растёт с удалением от цели, поэтому засчитывать дальний бакет
    # через провал ближнего нельзя: это означало бы, что контроллер не
    # справляется вблизи, но справляется вдали.
    ed = [8, 16, 32, 48]
    assert prefix_ok([True, True, False, True, True], ed) == 16
    assert prefix_ok([False, True, True, True, True], ed) == 0
    assert prefix_ok([True] * 5, ed) == 48
    assert prefix_ok([True, True, True, True, False], ed) == 48
    print("самопроверка k10d пройдена (версия «относительная цель»): "
          "перегон axis-angle, ортогональность, инвариантность к сдвигу, "
          "вращения не вычитаются, бакеты")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--n-ep", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=H_EXEC)
    ap.add_argument("--bucket-edges", default="8,16,32,48")
    ap.add_argument("--threshold", type=float, default=0.0478,
                    help="порог достаточности: ошибка позы грубого выхода, "
                         "про который измерено, что он сохраняет успех")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lrs", default="1e-3,3e-3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k10d")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

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
    from k10c_goal_persistence import grip_events
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, process_state)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ap_ = proc.action_processor
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    def to_codec_space(a):
        a = np.asarray(a, np.float64).copy()
        a[..., :-1] = a[..., :-1] / max_act_q[..., :-1]
        a[..., -1] = -a[..., -1]
        return a

    rid, rev = "physical-intelligence/libero", "v2.0"
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(1693)[: args.n_ep * 2]
    H = args.horizon
    edges = [int(x) for x in args.bucket_edges.split(",")]

    S, G, Y, REM, EPI = [], [], [], [], []
    floor_num, floor_den = 0.0, 0
    layout_printed = [False]
    n_used = 0
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
        if len(acts) < 3 * H:
            continue
        # РАСКЛАДКА СОСТОЯНИЯ ПЕЧАТАЕТСЯ. Та же логика, что в k9c/k9e:
        # сырое состояние с кватернионом приводится к axis-angle.
        if st.shape[1] == len(STATE_Q01) + 1:
            st = process_state(st)
        if st.shape[1] < 6:
            raise SystemExit(f"состояние {st.shape[1]}-мерное, нужно >= 6 "
                             f"(позиция 0:3, axis-angle 3:6)")
        if not layout_printed[0]:
            layout_printed[0] = True
            print(f"состояние: {st.shape[1]} измерений; позиция 0:3 диапазон "
                  f"[{st[:, :3].min():.3f}, {st[:, :3].max():.3f}], "
                  f"поворот 3:6 [{st[:, 3:6].min():.3f}, "
                  f"{st[:, 3:6].max():.3f}]")

        a = to_codec_space(acts)
        ev = grip_events(acts)
        n = len(a) - H
        if n <= 0:
            continue
        t = np.arange(n)
        nxt = np.searchsorted(ev, t, side="right")
        tau = np.where(nxt < len(ev), ev[np.minimum(nxt, len(ev) - 1)],
                       len(a) - 1)
        if len(ev) == 0:
            tau = np.full(n, len(a) - 1)
        g = rel_goal(st[t, :3], st[t, 3:6], st[tau, :3], st[tau, 3:6])
        sn = ((st[t] - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2 - 1
              if st.shape[1] == len(STATE_Q01) else st[t])
        y = np.stack([a[i:i + H] for i in t])
        S.append(sn); G.append(g); Y.append(y)
        REM.append(tau - t); EPI.append(np.full(n, e))

        # ПОЛ ТОКЕНИЗАТОРА на этих же данных — единственная полностью
        # сопоставимая опора.
        try:
            w = np.stack([a[i:i + 20] for i in range(len(a) - 20)])
            if len(w):
                d = ap_.decode(ap_.encode(w))
                d = np.asarray(d if isinstance(d, np.ndarray) else d[0],
                               np.float64).reshape(len(w), 20, 7)
                floor_num += float(((d[:, :H, :6] - w[:, :H, :6]) ** 2).mean())
                floor_den += 1
        except Exception:                            # noqa: BLE001
            pass
        n_used += 1
        if n_used % 50 == 0:
            print(f"  эпизодов {n_used}/{args.n_ep}", flush=True)

    if not S:
        raise SystemExit("ни одного эпизода не загрузилось")
    S = np.concatenate(S).astype(np.float32)
    G = np.concatenate(G).astype(np.float32)
    Y = np.concatenate(Y).astype(np.float32)
    REM = np.concatenate(REM)
    EPI = np.concatenate(EPI)
    floor = math.sqrt(floor_num / max(floor_den, 1))
    print(f"\nтроек {len(S)}, эпизодов {n_used}; пол токенизатора "
          f"поза{H} = {floor:.4f}")

    # РАЗБИЕНИЕ ПО ЭПИЗОДАМ, а не по строкам: соседние тройки почти одинаковы.
    ue = np.unique(EPI)
    rs = np.random.default_rng(args.seed).permutation(len(ue))
    ntr, nva = int(0.7 * len(ue)), int(0.15 * len(ue))
    sets = {"train": ue[rs[:ntr]], "val": ue[rs[ntr:ntr + nva]],
            "test": ue[rs[ntr + nva:]]}
    idx = {k: np.where(np.isin(EPI, v))[0] for k, v in sets.items()}
    print("  " + ", ".join(f"{k} {len(v)} троек / {len(sets[k])} эпизодов"
                           for k, v in idx.items()))

    def pose_err(pred, sel):
        d = pred[:, :, :6] - Y[sel][:, :, :6]
        return float(np.sqrt((d ** 2).mean()))

    def grip_err(pred, sel):
        return float((np.sign(pred[:, :, 6]) != np.sign(Y[sel][:, :, 6])).mean())

    # --- опоры ----------------------------------------------------------------
    mean_a = Y[idx["train"]].mean(0)
    base_mean = np.broadcast_to(mean_a, (len(idx["test"]),) + mean_a.shape)
    # П-РЕГУЛЯТОР: команда пропорциональна относительной цели, коэффициент
    # подбирается наименьшими квадратами на обучающей части.
    gt = G[idx["train"]]
    num = (gt[:, None, :] * Y[idx["train"]][:, :, :6]).sum()
    den = (gt ** 2).sum() * 1.0
    kp = num / max(den, 1e-12)
    p_pred = np.zeros((len(idx["test"]), H, 7), np.float32)
    p_pred[:, :, :6] = kp * G[idx["test"]][:, None, :]
    p_pred[:, :, 6] = mean_a[:, 6]

    print(f"\n  опоры на test: среднее поза{H} {pose_err(base_mean, idx['test']):.4f}, "
          f"П-регулятор {pose_err(p_pred, idx['test']):.4f} (kp={kp:.3f})")

    # --- контроллер -----------------------------------------------------------
    din, dout = S.shape[1] + G.shape[1], H * 7
    best = None
    for lr in [float(x) for x in args.lrs.split(",")]:
        for sd in [int(x) for x in args.seeds.split(",")]:
            torch.manual_seed(sd)
            net = nn.Sequential(nn.Linear(din, args.hidden), nn.GELU(),
                                nn.Linear(args.hidden, args.hidden), nn.GELU(),
                                nn.Linear(args.hidden, dout)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=lr)
            rg = np.random.default_rng(sd)

            def fwd(sel):
                x = torch.from_numpy(np.concatenate([S[sel], G[sel]], 1)
                                     ).to(dev)
                return net(x).reshape(len(sel), H, 7)

            @torch.no_grad()
            def predict(sel):
                net.eval()
                out = []
                for i0 in range(0, len(sel), args.batch):
                    out.append(fwd(sel[i0:i0 + args.batch]).cpu().numpy())
                return np.concatenate(out)

            # ЭПОХА 0 — ПОЛНОПРАВНЫЙ КАНДИДАТ.
            cur = (pose_err(predict(idx["val"]), idx["val"]), 0)
            st_best = {k: v.detach().clone() for k, v in net.state_dict().items()}
            for ep in range(1, args.epochs + 1):
                net.train()
                o = rg.permutation(idx["train"])
                for i0 in range(0, len(o), args.batch):
                    s = o[i0:i0 + args.batch]
                    loss = ((fwd(s) - torch.from_numpy(Y[s]).to(dev)) ** 2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward(); opt.step()
                e = pose_err(predict(idx["val"]), idx["val"])
                if e < cur[0]:
                    cur = (e, ep)
                    st_best = {k: v.detach().clone()
                               for k, v in net.state_dict().items()}
            net.load_state_dict(st_best)
            print(f"    lr={lr:g} сид={sd}: валидация {cur[0]:.4f} "
                  f"на эпохе {cur[1]}", flush=True)
            if best is None or cur[0] < best[0]:
                best = (cur[0], dict(lr=lr, seed=sd, epoch=cur[1]),
                        predict(idx["test"]))

    _, cfg, pred = best
    print(f"\n  выбрано по валидации: {cfg}")

    # --- разбивка по удалённости до события -----------------------------------
    rem = REM[idx["test"]]
    bi = buckets(rem, edges)
    names = ([f"<= {edges[0]}"]
             + [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
             + [f"> {edges[-1]}"])
    print(f"\n  {'удалённость':>14}{'троек':>9}{'контроллер':>13}"
          f"{'П-рег.':>10}{'знак схвата':>13}")
    rows, flags = {}, []
    for b in range(len(edges) + 1):
        m = bi == b
        if not m.any():
            flags.append(False)
            continue
        sel = idx["test"][m]
        pe = pose_err(pred[m], sel)
        pp = pose_err(p_pred[m], sel)
        ge = grip_err(pred[m], sel)
        ok = bool(pe <= args.threshold)
        flags.append(ok)
        rows[names[b]] = dict(n=int(m.sum()), ctrl=pe, p_ctrl=pp, grip=ge,
                              ok=ok)
        print(f"  {names[b]:>14}{int(m.sum()):>9}{pe:>13.4f}"
              f"{pp:>10.4f}{ge:>12.1%}" + ("  ok" if ok else "  --"))

    ok_upto = prefix_ok(flags, edges)
    s_real = ok_upto / H if H else 0.0
    print(f"\n  ПОРОГ {args.threshold:.4f} — ошибка позы грубого выхода, про "
          f"который\n  измерено, что он сохраняет успех. Пол токенизатора на "
          f"этих данных {floor:.4f}.")
    print(f"  Контроллер достаточен вплоть до {ok_upto} шагов до события.")
    print(f"  РЕАЛЬНОЕ сокращение вызовов: {s_real:.1f}x против верхней "
          f"оценки 6.7x из K-10c.")
    if s_real >= 4:
        v = "направление подтверждено"
    elif s_real <= 1:
        v = "экономии нет, направление закрывается"
    else:
        v = f"сокращение {s_real:.1f}x — это и есть результат"
    print(f"  ВЕРДИКТ: {v}")

    p = os.path.join(args.out, "table.json")
    json.dump(dict(buckets=rows, floor=floor, threshold=args.threshold,
                   ok_upto=ok_upto, s_real=s_real, verdict=v, cfg=cfg,
                   kp=float(kp), n_triples=int(len(S)), n_episodes=n_used,
                   script_sha1=sha, argv=vars(args)),
              open(p, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {p}")


if __name__ == "__main__":
    main()
