"""K-6a: сколько несогласованности чанков даёт САМ ТОКЕНИЗАТОР.

ЗАЧЕМ. Зонд K-5c намерил, что политика расходится сама с собой уже через один
шаг: D(1) = 0.169, то есть 27% от величины действия и 25° по направлению. Мы
собирались это лечить обучением. Но прежде надо исключить объяснение, при
котором обучение бессильно.

Чанк в момент t кодирует действия a_t..a_{t+19}, чанк в t+1 — a_{t+1}..a_{t+20}.
Это РАЗНЫЕ ОКНА, и RVQ квантует каждое окно целиком. Квантование позиции j в
первом окне не обязано совпасть с квантованием позиции 0 во втором ДАЖЕ ЕСЛИ
исходные действия побитово одинаковы. Тогда D(1) — ошибка квантования при
сдвиге окна, свойство токенизатора, и никакое дообучение политики её не уберёт.

ЧТО МЕРЯЕТСЯ. Берутся ЭТАЛОННЫЕ действия демонстрации. Никакой политики нет
вовсе. Для каждого t кодируется окно a[t:t+20], декодируется обратно, и
считается ровно та же величина, что в K-5c:

    D_кв(j) = d( decode(encode(a[t:t+20]))[j],
                 decode(encode(a[t+j:t+j+20]))[0] )

Если D_кв(j) сравнима с измеренной D(j) политики — виноват токенизатор.
Если D_кв(j) близка к нулю — виновата политика, и обучение осмысленно.

ЧТО ЭТО НЕ МЕРЯЕТ. Ошибку реконструкции саму по себе: она считается отдельно
как ‖decode(encode(a)) − a‖ и служит проверкой, что нормировка действий верна.
Большая ошибка реконструкции означает, что мы кормим кодек не тем масштабом,
а не что кодек плох.

Запуск:
    python3 experiments/k6a_quant_floor.py --selftest
    python3 experiments/k6a_quant_floor.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO --n-ep 40
"""

import argparse
import io
import json
import os
import sys

import numpy as np

CHUNK, POSE, GRIP = 20, 6, 6


def shifted_consistency(dec, chunk=CHUNK):
    """Кривые несогласованности по массиву декодированных окон.

    dec: (n_win, chunk, 7), где dec[t] — декодированное окно, начинающееся в t.
    Возвращает те же величины, что печатает K-5c, чтобы числа были сравнимы
    напрямую: расхождение плана, базлайн «держать» и косинус направления.
    """
    n = dec.shape[0]
    out = {k: np.full(chunk, np.nan) for k in ("plan", "hold", "cos")}
    out["plan"][0] = out["hold"][0] = 0.0
    out["cos"][0] = 1.0
    for j in range(1, chunk):
        t = np.arange(0, n - j)
        if t.size == 0:
            continue
        stale = dec[t, j, :POSE]                  # позиция j окна из t
        fresh = dec[t + j, 0, :POSE]              # позиция 0 окна из t+j
        held = dec[t, 0, :POSE]                   # «держать первое действие»
        out["plan"][j] = np.linalg.norm(stale - fresh, axis=1).mean()
        out["hold"][j] = np.linalg.norm(held - fresh, axis=1).mean()
        den = np.linalg.norm(stale, axis=1) * np.linalg.norm(fresh, axis=1)
        ok = den > 1e-12
        out["cos"][j] = ((stale * fresh).sum(axis=1)[ok] / den[ok]).mean() \
            if ok.any() else np.nan
    return out


def roundtrip(actions, encode, decode, chunk=CHUNK):
    """Окна со сдвигом 1: кодируем, декодируем, собираем (n_win, chunk, 7)."""
    n = len(actions) - chunk
    if n <= 0:
        return np.zeros((0, chunk, 7))
    wins = np.stack([actions[t:t + chunk] for t in range(n)])
    toks = encode(wins)
    dec = np.asarray(decode(toks))
    return dec.reshape(n, chunk, -1)


def selftest():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, size=(200, 7))

    # 1. КОДЕК БЕЗ ПОТЕРЬ ОБЯЗАН ДАТЬ РОВНО НОЛЬ. Если окна декодируются
    #    точно, то позиция j окна из t и позиция 0 окна из t+j — это одно и
    #    то же действие a[t+j]. Любое отклонение здесь означает ошибку в
    #    индексации, а не свойство кодека.
    ident = roundtrip(a, lambda w: w, lambda w: w)
    c = shifted_consistency(ident)
    assert np.nanmax(np.abs(c["plan"])) < 1e-12, \
        f"кодек без потерь обязан дать ноль, получено {np.nanmax(c['plan']):.3e}"
    assert abs(c["cos"][5] - 1.0) < 1e-12, "косинус при точном кодеке = 1"
    # а «держать» при этом обязано быть БОЛЬШИМ: это разные действия
    assert c["hold"][5] > 0.1, \
        f"базлайн «держать» не может быть нулевым на случайных данных: {c['hold'][5]}"

    # 2. ГРУБОЕ КВАНТОВАНИЕ ОБЯЗАНО ДАТЬ НЕНУЛЕВУЮ, НО ОГРАНИЧЕННУЮ ВЕЛИЧИНУ.
    #    Округление до шага q независимо от окна: тогда позиция j и позиция 0
    #    дают ОДИН И ТОТ ЖЕ код, и несогласованность обязана остаться нулём.
    #    Это важный случай: поэлементное квантование окно НЕ ломает.
    q = 0.05
    per_elem = roundtrip(a, lambda w: np.round(w / q), lambda w: np.asarray(w) * q)
    cq = shifted_consistency(per_elem)
    assert np.nanmax(np.abs(cq["plan"])) < 1e-12, \
        ("поэлементное квантование не зависит от окна и обязано давать ноль; "
         f"получено {np.nanmax(cq['plan']):.3e}")

    # 3. КВАНТОВАНИЕ, ЗАВИСЯЩЕЕ ОТ ОКНА, обязано дать ненулевую величину —
    #    именно этот случай мы и подозреваем у RVQ. Вычитаем среднее по окну
    #    перед округлением: тогда одно и то же действие в разных окнах
    #    округляется по-разному.
    def enc_win(w):
        m = w.mean(axis=1, keepdims=True)
        return (np.round((w - m) / q), m)

    def dec_win(tk):
        r, m = tk
        return r * q + m

    win_dep = roundtrip(a, enc_win, dec_win)
    cw = shifted_consistency(win_dep)
    assert cw["plan"][1] > 1e-3, \
        (f"зависящее от окна квантование обязано ломать согласованность, "
         f"получено {cw['plan'][1]:.3e} — проверка бессильна")

    print("самопроверка пройдена:")
    print("  точный кодек и ПОЭЛЕМЕНТНОЕ квантование дают ровно ноль")
    print(f"  зависящее от ОКНА квантование ломает согласованность: "
          f"D(1) = {cw['plan'][1]:.4f}")
    print("  значит ненулевой результат на настоящем кодеке будет означать "
          "именно оконную зависимость")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--n-ep", type=int, default=40, help="эпизодов демонстраций")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    import actioncodec  # noqa: F401
    from utils import (ACTION_Q01, ACTION_Q99,  # noqa: E402
                       VisionLanguageActionProcessor)

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ap_ = proc.action_processor
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    # ЗАГРУЗКА ЧЕРЕЗ hf_hub_download И РАЗБОР PARQUET, а не LeRobotDataset:
    # обёртка lerobot 0.4.4 на ревизии v2.0 падает с BackwardCompatibilityError
    # (см. k4b0_build_router_dataset.py).
    rid, rev = "physical-intelligence/libero", "v2.0"
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(1693)[: args.n_ep]

    def to_codec_space(a):
        """Обратное к тому, что делает eval_libero после декода."""
        a = np.asarray(a, np.float64).copy()
        a[..., :-1] = a[..., :-1] / max_act_q[..., :-1]
        a[..., -1] = -a[..., -1]
        return a

    curves, recon, n_used = [], [], 0
    for e in order:
        try:
            # номер chunk-каталога зависит от эпизода, см.
            # k4b0_build_router_dataset.py:143 — при chunk-000 всё, что
            # начиная с 1000-го эпизода, молча не находилось бы
            f = hf_hub_download(
                rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                repo_type="dataset", revision=rev)
            # столбец называется "actions", не "action" — см.
            # k4b0_build_router_dataset.py:158. Первая версия молча пропускала
            # ВСЕ эпизоды по KeyError, поэтому имена столбцов печатаются.
            tab = pq.read_table(f)
            acts = np.asarray(tab.column("actions").to_pylist(), np.float32)
        except Exception as ex:                      # noqa: BLE001
            cols = locals().get("tab")
            print(f"  эпизод {e}: пропуск ({type(ex).__name__}: {ex})"
                  + (f"; столбцы: {cols.column_names}" if cols is not None
                     else ""))
            continue
        if len(acts) <= CHUNK + 2:
            continue
        a = to_codec_space(acts)

        def enc(w):
            return ap_.encode(w)

        def dec(tk):
            return ap_.decode(tk)

        dec_w = roundtrip(a, enc, dec)
        if dec_w.shape[0] == 0:
            continue
        curves.append(shifted_consistency(dec_w))
        # ОШИБКА РЕКОНСТРУКЦИИ — проверка нормировки, а не свойство кодека.
        recon.append(float(np.linalg.norm(
            dec_w[:, 0, :POSE] - a[:dec_w.shape[0], :POSE], axis=1).mean()))
        n_used += 1
        if n_used >= args.n_ep:
            break

    if not curves:
        raise SystemExit(
            "ни одного эпизода не загрузилось — смотрите причины пропуска выше; "
            "молчаливый пропуск ВСЕХ эпизодов уже случался из-за неверного "
            "имени столбца")

    plan = np.nanmean([c["plan"] for c in curves], axis=0)
    hold = np.nanmean([c["hold"] for c in curves], axis=0)
    cos = np.nanmean([c["cos"] for c in curves], axis=0)
    rec = float(np.mean(recon))

    # измеренное политикой (K-5c, LIBERO-10, задача 0) — для прямого сравнения
    POL = {1: 0.1693, 4: 0.2172, 8: 0.2496, 19: 0.3185}

    print(f"\nэпизодов: {n_used}, ошибка реконструкции позы: {rec:.4f}")
    if rec > 0.15:
        print("  ВНИМАНИЕ: реконструкция плохая — вероятно, действия поданы в\n"
              "  неверном масштабе, и всё ниже недействительно.")
    print("\n" + "=" * 70)
    print(f"  {'j':>3}{'D_кв':>10}{'D_держать':>11}{'cos':>8}"
          f"{'D политики':>12}{'доля кв.':>10}")
    for j in (1, 2, 4, 8, 12, 19):
        s = f"  {j:>3}{plan[j]:>10.4f}{hold[j]:>11.4f}{cos[j]:>8.3f}"
        if j in POL:
            s += f"{POL[j]:>12.4f}{plan[j] / POL[j]:>10.1%}"
        print(s)

    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    print("  Доля кв. > ~60% — несогласованность создаёт ТОКЕНИЗАТОР, а не")
    print("  политика. Тогда дообучение политики её не уберёт, и чинить надо")
    print("  кодирование окна.")
    print("  Доля кв. < ~25% — виновата политика, обучение на согласованность")
    print("  осмысленно.")
    print("  Между — обе причины вкладываются, и нужно разделять дальше.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dict(plan=plan.tolist(), hold=hold.tolist(),
                       cos=cos.tolist(), recon=rec, episodes=n_used,
                       ckpt=args.ckpt, policy_reference=POL),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
