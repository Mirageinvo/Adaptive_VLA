"""K-10a: гасятся ли ошибки токенизатора при сдвиге окна под интегралом.

ЧТО ИМЕННО МЕРЯЕТСЯ, И ЧТО НЕТ. Сравниваются два окна на ОБЩЕМ абсолютном
отрезке: окно из t на позициях shift..19 и окно из t+shift на позициях 0..L-1
покрывают одни и те же моменты времени. Считается расхождение пошаговое и
расхождение их ИНТЕГРАЛА по этому отрезку.

Это свойство ТОКЕНИЗАТОРА: насколько согласованно два кодирования одного и
того же отрезка говорят, куда придёт рука. Политики здесь нет вовсе.

ЭТО НЕ ПРОВЕРКА ПЕРСИСТЕНТНОСТИ СОБЫТИЙНОЙ ЦЕЛИ. Конец окна из t относится к
моменту t+20, а из t+8 — к t+28; это разные физические цели, и сравнивать их
бессмысленно. Поэтому здесь и берётся общий отрезок. Вопрос «переживает ли
цель несколько вызовов» — другой, и на него отвечает K-10c: там цель
определяется событием (смена состояния схвата, точка минимальной скорости), а
не концом фиксированного окна.

ЗАЧЕМ ТОГДА ЭТО ИЗМЕРЕНИЕ. Оно отвечает на предварительный вопрос: если
токенизатор рассогласован пошагово (K-6a: D(1) = 0.169), то настолько же
рассогласован ли он в том, КУДА приводит. Если ошибки гасятся под интегралом,
приход устойчив, и кодировать его осмысленно. Если накапливаются — приход
устойчив хуже пути, и событийная цель наследует ту же беду.

ЧТО УЖЕ ИЗМЕРЕНО И НЕ ПЕРЕСЧИТЫВАЕТСЯ. K-6a меряет ПОШАГОВОЕ расхождение
двух окон на общем времени — свойство токенизатора без всякой политики. Его
функции `roundtrip` и `shifted_consistency` берутся отсюда как есть.

ЧТО ДОБАВЛЯЕТСЯ. Действия LIBERO — приращения позы. Значит «куда придём» это
их ИНТЕГРАЛ по общему отрезку. Два окна могут расходиться пошагово и при этом
приводить в одну точку, если ошибки взаимно гасятся. Считаются обе величины на
одном и том же отрезке:

    шаговое:  mean_i ||Â_t[j+i] - Â_{t+j}[i]||        (как в K-6a)
    целевое:  || sum_i (Â_t[j+i] - Â_{t+j}[i]) ||     (расхождение прихода)

ОБЯЗАТЕЛЬНЫЙ КОНТРОЛЬ — НЕЗАВИСИМЫЙ ШУМ. Само по себе `||сумма|| < сумма
норм` верно всегда и ничего не доказывает. Если бы пошаговые расхождения были
независимы, интеграл рос бы как sqrt(L). Поэтому печатается отношение

    ||sum_i d_i||  /  ( sqrt(L) * mean_i ||d_i|| )

Меньше единицы — ошибки ГАСЯТСЯ, цель устойчивее пути, посылка подтверждена.
Около единицы — ошибки независимы, гашения нет. Больше единицы — ошибки
когерентны и НАКАПЛИВАЮТСЯ, цель устойчива ХУЖЕ пути, и направление
закрывается.

СХВАТ СЧИТАЕТСЯ ОТДЕЛЬНО. Он не приращение, а команда, интегрировать его
бессмысленно; для него меряется согласие знака на общем отрезке.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ, записано ДО запуска:
  * коэффициент гашения <= 0.6 на сдвиге 8 (наш горизонт исполнения) ->
    посылка подтверждена, цель кодировать имеет смысл;
  * >= 1.0 -> ошибки накапливаются, направление закрывается;
  * между -> не доказано ничего, нужен другой признак цели.

Запуск:
    python3 experiments/k10a_goal_stability.py --selftest
    python3 experiments/k10a_goal_stability.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO --n-ep 40 \\
        --out data/k10a_goal_stability.json
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

CHUNK, POSE, GRIP = 20, 6, 6


def overlap_stats(dec, true_a, shift, pose=POSE):
    """Расхождение двух окон на общем отрезке: пошаговое и целевое.

    dec: (n_win, CHUNK, 7) — декодированные окна со сдвигом 1.
    true_a: (n, 7) — эталонные действия, нужны только для масштаба.
    Возвращает величины, усреднённые по всем допустимым t.
    """
    n_win = dec.shape[0]
    L = CHUNK - shift
    if L <= 0 or n_win - shift <= 0:
        return None
    t = np.arange(0, n_win - shift)
    # Â_t на позициях shift..CHUNK-1 и Â_{t+shift} на позициях 0..L-1
    # покрывают ОДИН И ТОТ ЖЕ отрезок абсолютного времени t+shift..t+CHUNK-1.
    a1 = dec[t][:, shift:CHUNK, :pose]
    a2 = dec[t + shift][:, 0:L, :pose]
    d = a1 - a2                                   # (T, L, pose)

    step = float(np.linalg.norm(d, axis=-1).mean())
    goal = float(np.linalg.norm(d.sum(axis=1), axis=-1).mean())
    # Масштабы: средняя величина шага и величина истинного прихода.
    tt = t[:, None] + shift + np.arange(L)[None, :]
    tt = np.clip(tt, 0, len(true_a) - 1)
    a_true = true_a[tt][..., :pose]
    step_scale = float(np.linalg.norm(a_true, axis=-1).mean())
    goal_scale = float(np.linalg.norm(a_true.sum(axis=1), axis=-1).mean())
    # КОНТРОЛЬ НЕЗАВИСИМОСТИ: при независимых d_i интеграл рос бы как sqrt(L).
    cancel = goal / max(np.sqrt(L) * step, 1e-12)

    g1 = np.sign(dec[t][:, shift:CHUNK, GRIP])
    g2 = np.sign(dec[t + shift][:, 0:L, GRIP])
    return dict(shift=int(shift), L=int(L), n=int(t.size),
                step=step, goal=goal,
                step_rel=step / max(step_scale, 1e-12),
                goal_rel=goal / max(goal_scale, 1e-12),
                cancel=float(cancel),
                grip_sign_disagree=float((g1 != g2).mean()))


def selftest():
    rng = np.random.default_rng(0)
    n, sh = 200, 8
    a = rng.normal(0, 0.3, size=(n + CHUNK, 7))
    wins = np.stack([a[t:t + CHUNK] for t in range(n)])

    # 1. Идеальный кодек: расхождения нет ни пошагового, ни целевого.
    r = overlap_stats(wins.copy(), a, sh)
    assert r["step"] < 1e-12 and r["goal"] < 1e-12, r

    # ЗНАК ЗАВИСИТ ОТ ОКНА СЛУЧАЙНО, а не с периодом 2: при периоде 2 окна t и
    # t+8 получали бы одинаковый знак, расхождение занулялось бы, и тест
    # проверял бы пустоту. Первая версия этой самопроверки так и делала.
    sgn = rng.choice([-1.0, 1.0], size=n)[:, None, None]

    # 2. ГАСЯЩИЕСЯ ошибки: смещение чередуется ПО ПОЗИЦИИ внутри окна.
    # Пошагово расхождение есть, а сумма по чётному отрезку гасится точно.
    alt = ((-1.0) ** np.arange(CHUNK))[None, :, None]
    d2 = wins + 0.05 * alt * sgn
    r2 = overlap_stats(d2, a, sh)
    assert r2["step"] > 1e-3, r2
    assert r2["cancel"] < 0.4, f"гашение не обнаружено: {r2['cancel']:.3f}"

    # 3. КОГЕРЕНТНЫЕ ошибки: смещение постоянно внутри окна. Сумма растёт как
    # L, коэффициент гашения обязан выйти около sqrt(L).
    d3 = wins + 0.05 * sgn
    r3 = overlap_stats(d3, a, sh)
    assert r3["cancel"] > 2.0, f"накопление не обнаружено: {r3['cancel']:.3f}"

    # 4. НЕЗАВИСИМЫЕ ошибки: коэффициент обязан быть около единицы.
    d4 = wins + rng.normal(0, 0.05, size=wins.shape)
    r4 = overlap_stats(d4, a, sh)
    assert 0.6 < r4["cancel"] < 1.6, f"независимый шум дал {r4['cancel']:.3f}"

    # 5. Отрезок и число окон считаются верно.
    assert r["L"] == CHUNK - sh and r["n"] == n - sh

    # 6. Схват берётся из своего канала и не участвует в интеграле. Знак
    # переворачивается ЧЕРЕЗ БЛОК ДЛИНЫ sh, тогда окна t и t+sh всегда лежат в
    # соседних блоках и расходятся гарантированно. Простой переворот у ВСЕХ
    # окон ничего не дал бы: сравниваются два окна одного и того же массива.
    d6 = wins.copy()
    d6[:, :, GRIP] *= ((np.arange(n) // sh) % 2 * 2 - 1)[:, None]
    r6 = overlap_stats(d6, a, sh)
    assert r6["grip_sign_disagree"] > 0.9, r6["grip_sign_disagree"]
    assert overlap_stats(wins.copy(), a, sh)["grip_sign_disagree"] < 1e-12
    print("самопроверка k10a пройдена (версия «коэффициент гашения»): "
          f"идеальный кодек 0, гасящиеся {r2['cancel']:.2f}, независимые "
          f"{r4['cancel']:.2f}, когерентные {r3['cancel']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--n-ep", type=int, default=40)
    ap.add_argument("--shifts", default="1,2,4,8,12,16")
    ap.add_argument("--gate-shift", type=int, default=8,
                    help="сдвиг, по которому читается вердикт: наш горизонт")
    ap.add_argument("--cancel-good", type=float, default=0.6)
    ap.add_argument("--cancel-bad", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k10a sha1 {sha}")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    # roundtrip ПЕРЕИСПОЛЬЗУЕТСЯ ИЗ K-6a, а не переписывается: там уже учтено,
    # что официальный decode возвращает контейнер, а не массив.
    from k6a_quant_floor import roundtrip

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ap_ = proc.action_processor
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    def to_codec_space(a):
        """Обратное тому, что делает eval_libero после декода."""
        a = np.asarray(a, np.float64).copy()
        a[..., :-1] = a[..., :-1] / max_act_q[..., :-1]
        a[..., -1] = -a[..., -1]
        return a

    rid, rev = "physical-intelligence/libero", "v2.0"
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(1693)[: args.n_ep * 2]
    shifts = [int(x) for x in args.shifts.split(",")]

    acc = {s: [] for s in shifts}
    n_used = 0
    for e in order:
        if n_used >= args.n_ep:
            break
        try:
            f = hf_hub_download(
                rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                repo_type="dataset", revision=rev)
            acts = np.asarray(pq.read_table(f).column("actions").to_pylist(),
                              np.float32)
        except Exception as ex:                      # noqa: BLE001
            print(f"  эпизод {e}: пропуск ({type(ex).__name__}: {ex})")
            continue
        if len(acts) <= CHUNK + max(shifts) + 2:
            continue
        a = to_codec_space(acts)
        dec = roundtrip(a, ap_.encode, ap_.decode)
        if dec.shape[0] <= max(shifts):
            continue
        for s in shifts:
            r = overlap_stats(dec, a, s)
            if r is not None:
                acc[s].append(r)
        n_used += 1
        if n_used % 10 == 0:
            print(f"  эпизодов {n_used}/{args.n_ep}", flush=True)

    if not any(acc.values()):
        raise SystemExit("ни одного эпизода не загрузилось")

    print(f"\nэпизодов использовано: {n_used}\n")
    print(f"  {'сдвиг':>6}{'шаг отн.':>11}{'цель отн.':>11}"
          f"{'гашение':>10}{'знак схвата':>13}")
    res = {}
    for s in shifts:
        if not acc[s]:
            continue
        m = {k: float(np.mean([r[k] for r in acc[s]]))
             for k in ("step", "goal", "step_rel", "goal_rel", "cancel",
                       "grip_sign_disagree")}
        m["n_episodes"] = len(acc[s])
        res[str(s)] = m
        print(f"  {s:>6}{m['step_rel']:>10.1%}{m['goal_rel']:>11.1%}"
              f"{m['cancel']:>10.2f}{m['grip_sign_disagree']:>12.1%}")

    g = res.get(str(args.gate_shift))
    print(f"\n  ЧИТАТЬ по коэффициенту гашения на сдвиге {args.gate_shift} "
          f"(наш горизонт исполнения).")
    print(f"  Он равен ||сумма расхождений|| / (sqrt(L) * среднее "
          f"расхождение).")
    print(f"  Меньше {args.cancel_good} — ошибки гасятся, цель устойчивее "
          f"пути, посылка верна.")
    print(f"  Больше {args.cancel_bad} — ошибки накапливаются, цель устойчива "
          f"ХУЖЕ, направление закрывается.")
    if g is None:
        verdict = "сдвиг для вердикта не посчитан"
    elif g["cancel"] <= args.cancel_good:
        verdict = ("ГАШЕНИЕ: цель устойчивее пути, кодировать цель "
                   "осмысленно")
    elif g["cancel"] >= args.cancel_bad:
        verdict = ("НАКОПЛЕНИЕ: цель устойчива хуже пути, направление "
                   "закрывается")
    else:
        verdict = "НЕ ДОКАЗАНО НИЧЕГО, нужен другой признак цели"
    print(f"\n  ВЕРДИКТ: {verdict}"
          + (f" (гашение {g['cancel']:.2f})" if g else ""))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(dict(by_shift=res, n_episodes=n_used, verdict=verdict,
                       gate_shift=args.gate_shift, script_sha1=sha,
                       ckpt=args.ckpt, argv=vars(args)),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}  (sha {sha})")


if __name__ == "__main__":
    main()
