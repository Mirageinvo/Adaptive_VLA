"""K-6f: сколько вообще стоят уровни 1 и 2, и есть ли что закрывать.

ЗАЧЕМ. K-6e показал странное: кросс-энтропия уточнителя на валидации 10.6-12.7
при ln(2048)=7.62, то есть ХУЖЕ равномерного угадывания, а лучшая ошибка
действия достигается на первой эпохе. Прежде чем чинить обучение, надо
проверить более простую возможность: уровни 1-2 могут почти не влиять на
декодированное действие, и тогда закрывать нечего в принципе.

ЧТО СЧИТАЕТСЯ. Всё на уже сохранённых признаках, без модели и без обучения:

  A. Совпадение кодов BAR с кодами токенизатора ПО УРОВНЯМ. Если BAR угадывает
     уровень 0 хорошо, а 1-2 плохо, значит и она их не предсказывает, и разрыв
     не в них.
  B. Ошибка действия при разной комплектации:
        только уровень 0 (BAR)              что даёт грубый код сам по себе
        уровень 0 BAR + ИСТИННЫЕ 1,2        ПОТОЛОК любого уточнителя
        уровень 0 BAR + СЛУЧАЙНЫЕ 1,2       пол: сколько теряет мусор
        все три BAR                         фактическая работа BAR
        все три истинные                    эксперт
  C. Обратный срез: ИСТИННЫЙ уровень 0 + уровни 1,2 от BAR. Показывает, в каком
     уровне на самом деле сидит ошибка BAR.

ПОЧЕМУ ДЕКОДИРОВАНИЕ ПРЕФИКСОМ ЗАКОННО. Кодек обучался с quantizer_dropout=0.25
(rvq.py:301), то есть декодирование по неполному набору уровней предусмотрено
конструкцией, а не является насилием над моделью.

ПРАВИЛО ЧТЕНИЯ, записано до запуска:
  «уровень 0 BAR + истинные 1,2» близко к «все три BAR» -> уровни 1-2 стоят
      мало, разрыв сидит в уровне 0, и уточнитель для 1-2 бесполезен ПО
      ПОСТРОЕНИЮ, сколько его ни обучай;
  «уровень 0 BAR + истинные 1,2» заметно ЛУЧШЕ «все три BAR» -> есть что
      закрывать, и провал K-6e действительно в обучении;
  «истинный 0 + BAR 1,2» близко к эксперту -> ошибка BAR почти вся в уровне 0.

Запуск:
    python3 experiments/k6f_level_worth.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6f_level_worth.py --feats data/k6d_features.npz \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO
"""

import argparse
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL = 16, 3


def selftest():
    # Подмена уровня обязана менять ТОЛЬКО его. Ошибка легко пишет не в ту ось.
    K = np.arange(2 * N_LEVEL * N_POS).reshape(2, N_LEVEL, N_POS)
    K2 = K.copy()
    K2[:, 1, :] = -1
    assert (K2[:, 0, :] == K[:, 0, :]).all(), "уровень 0 не должен меняться"
    assert (K2[:, 2, :] == K[:, 2, :]).all(), "уровень 2 не должен меняться"
    assert (K2[:, 1, :] == -1).all()
    # Раскладка при сборке в плоский список: уровни идут подряд, не вперемежку
    flat = K[0].reshape(-1)
    assert (flat[:N_POS] == K[0, 0]).all(), "первые 16 — уровень 0"
    assert (flat[2 * N_POS:] == K[0, 2]).all(), "последние 16 — уровень 2"
    print("самопроверка пройдена: подмена уровня локальна, раскладка поуровневая")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--feats", default="data/k6d_features.npz")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    sys.path.insert(0, os.path.abspath(args.root))
    import actioncodec  # noqa: F401
    from utils import VisionLanguageActionProcessor  # noqa: E402

    z = np.load(args.feats, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    K_true, K_bar, act = z["K_true"], z["K_bar"], z["act"]
    n_codes = int(meta["n_codes"])
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    def decode(codes):
        d = proc.action_processor.decode(codes.reshape(len(codes), -1).tolist())
        return np.asarray(d if isinstance(d, np.ndarray) else d[0], np.float64)

    rng_pose = float(act[..., :6].max() - act[..., :6].min())

    def err(codes):
        d = decode(codes)
        return (float(np.sqrt(((d[..., :6] - act[..., :6]) ** 2).mean())) / rng_pose,
                float((np.sign(d[..., 6]) != np.sign(act[..., 6])).mean()))

    # --- A. совпадение по уровням --------------------------------------------
    print("\n  совпадение кодов BAR с кодами токенизатора:")
    for lv in range(N_LEVEL):
        m = float((K_bar[:, lv, :] == K_true[:, lv, :]).mean())
        print(f"    уровень {lv}: {m:.1%}")
    print(f"    все вместе: {float((K_bar == K_true).mean()):.1%}")

    # --- B. комплектации ------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    variants = {}
    variants["все три истинные (эксперт)"] = K_true
    variants["все три BAR"] = K_bar
    mix = K_bar.copy(); mix[:, 1:, :] = K_true[:, 1:, :]
    variants["0 от BAR + ИСТИННЫЕ 1,2"] = mix
    rnd = K_bar.copy(); rnd[:, 1:, :] = rng.integers(0, n_codes, size=rnd[:, 1:, :].shape)
    variants["0 от BAR + СЛУЧАЙНЫЕ 1,2"] = rnd
    inv = K_true.copy(); inv[:, 1:, :] = K_bar[:, 1:, :]
    variants["ИСТИННЫЙ 0 + 1,2 от BAR"] = inv

    print("\n" + "=" * 66)
    print(f"  {'комплектация':<32}{'поза RMS':>11}{'схват':>10}")
    res = {}
    for name, K in variants.items():
        p, g = err(K)
        res[name] = dict(pose_rms=p, gripper=g)
        print(f"  {name:<32}{p:>11.4f}{g:>10.1%}")

    e_bar = res["все три BAR"]["pose_rms"]
    e_ceil = res["0 от BAR + ИСТИННЫЕ 1,2"]["pose_rms"]
    e_rnd = res["0 от BAR + СЛУЧАЙНЫЕ 1,2"]["pose_rms"]
    print(f"\n  ПОТОЛОК уточнителя уровней 1-2: {e_ceil:.4f}")
    print(f"  фактическая BAR:                {e_bar:.4f}")
    print(f"  мусор вместо 1-2:               {e_rnd:.4f}")
    gain = e_rnd - e_ceil
    print(f"\n  вся ценность уровней 1-2 = {gain:.4f} "
          f"({gain / max(e_rnd, 1e-9):.0%} от ошибки с мусором)")

    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    if e_ceil >= e_bar - 1e-4:
        print("  ПОТОЛОК НЕ ЛУЧШЕ BAR: даже ИДЕАЛЬНОЕ предсказание уровней 1-2")
        print("  не догоняет BAR. Значит разрыв сидит в УРОВНЕ 0, и уточнитель")
        print("  для 1-2 бесполезен по построению — сколько его ни обучай.")
    else:
        print("  Потолок лучше BAR: закрывать есть что, и провал K-6e —")
        print("  действительно в постановке обучения.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        res["meta"] = dict(feats=args.feats, n=int(len(K_true)),
                           level_match=[float((K_bar[:, lv, :] == K_true[:, lv, :]).mean())
                                        for lv in range(N_LEVEL)])
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
