"""K-10e: сопоставимая опора coarse24 для гейта K-10d.

ЗАЧЕМ. K-10d сравнивал ошибку контроллера с числом 0.0478. Оно измерено на
другой выборке и как расхождение модели с УЧИТЕЛЕМ, а K-10d считает
расхождение с ДЕМОНСТРАЦИЕЙ. Пол токенизатора этого не чинит: контроллер через
кодек не проходит вовсе. Поэтому K-10d теперь отказывается считать без опоры, а
опору строит этот скрипт.

ЧТО СТРОИТСЯ. Для каждой строки кэша K-9a берётся предсказание coarse24 —
грубые коды учителя, декодированные из уровня 0, — и сравнивается с
ДЕМОНСТРАЦИОННЫМ действием того же наблюдения. Ошибка раскладывается по тем же
бакетам удалённости до следующего события, что и в K-10d.

ОДНО ДЕЙСТВИЕ, А НЕ ЧАНК. K-10d предсказывает одно следующее действие, поэтому
и опора считается на позиции 0 декодированного чанка. Усреднение по восьми
позициям дало бы другую величину, и сравнение было бы нечестным.

ПОЗИЦИЯ, ВРАЩЕНИЕ И СХВАТ — ТРИ ОТДЕЛЬНЫЕ КРИВЫЕ. Метры и радианы несравнимы,
а знак схвата вообще не метрический. Смешанная величина по шести каналам тоже
печатается, но только потому, что по ней ставится порог в K-10d; вердикт по
схвату выносится по своей кривой.

КЛЮЧИ СОПОСТАВЛЯЮТСЯ СТРОГО. Объединение идёт по паре (эпизод, шаг), ключи
проверяются на уникальность и стопроцентное покрытие, разбиение берётся из
манифеста K-9a — никакого нового случайного деления. Скрипт выдаёт не только
кривую, но и ТОЧНЫЙ СПИСОК строк для оценки, чтобы K-10d считался ровно на
тех же наблюдениях, а не на «похожих».

СОБЫТИЯ БЕРУТСЯ ИЗ ОБЩЕГО МОДУЛЯ `goal_events`, тем же вызовом, что в K-10d.
Раньше разметка была продублирована в трёх скриптах и разъехалась: здесь
объединение и `side="left"`, там схват и `side="right"`. По умолчанию
объединение — консервативный выбор: если контроллеру надо перепланировать при
любом из событий, целей столько, сколько в нём.

Запуск:
    python3 experiments/k10e_coarse_baseline.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k10e_coarse_baseline.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --cache data/k9_teacher_150k.npz --out data/k10e_baseline.json
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL = 16, 3


def curve(err_pos, err_rot, err_pose, grip_bad, bi, names):
    """Кривые по бакетам. Взвешивание по числу строк, а не по бакетам."""
    out = {}
    for b, nm in enumerate(names):
        m = bi == b
        if not m.any():
            continue
        out[nm] = dict(
            n=int(m.sum()),
            pos=float(np.sqrt((err_pos[m] ** 2).mean())),
            rot=float(np.sqrt((err_rot[m] ** 2).mean())),
            pose=float(np.sqrt((err_pose[m] ** 2).mean())),
            grip=float(grip_bad[m].mean()))
    return out


def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import goal_events as ge
    ge.selftest()
    ed = [8, 16, 32, 48]
    bucket_names = ge.bucket_names

    # Кривые считаются по строкам, а не по бакетам: бакет из одной строки не
    # весит столько же, сколько бакет из тысячи.
    bi = np.array([0, 0, 0, 1])
    ep = np.array([0.0, 0.0, 0.0, 10.0])
    c = curve(ep, ep, ep, np.zeros(4), bi, bucket_names(ed))
    assert c["<= 8"]["n"] == 3 and c["<= 8"]["pose"] == 0.0
    assert abs(c["8-16"]["pose"] - 10.0) < 1e-12

    # RMS складывается по квадратам.
    x = np.array([3.0, 4.0])
    c2 = curve(x, x, x, np.zeros(2), np.zeros(2, int), bucket_names(ed))
    assert abs(c2["<= 8"]["pose"] - np.sqrt(12.5)) < 1e-12
    print("самопроверка k10e пройдена (версия «граница у завершающейся "
          "фазы»): имена бакетов, остаток в момент события ноль, взвешивание "
          "по строкам, RMS по квадратам")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--event", choices=["grip", "stop", "union"],
                    default="union")
    ap.add_argument("--bucket-edges", default="8,16,32,48")
    ap.add_argument("--split", default="test")
    ap.add_argument("--speed-frac", type=float, default=0.3)
    ap.add_argument("--min-dwell", type=int, default=3)
    ap.add_argument("--min-travel", type=float, default=0.02)
    ap.add_argument("--merge-tol", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="data/k10e_baseline.json")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k10e sha1 {sha}")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download

    import actioncodec  # noqa: F401
    import goal_events as ge
    from utils import (ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor,
                       STATE_Q01, process_state)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    z = np.load(args.cache, allow_pickle=True)
    q = z["teacher_codes_q0"].astype(np.int64)
    epi, stp, split = z["episode"], z["step"], z["split"]
    N = len(q)

    # КЛЮЧИ ОБЯЗАНЫ БЫТЬ УНИКАЛЬНЫ. Дубль (эпизод, шаг) означал бы, что одна
    # и та же строка попала в кэш дважды и веса бакетов поехали.
    keys = np.stack([epi, stp], 1)
    uniq = np.unique(keys, axis=0)
    if len(uniq) != N:
        raise SystemExit(f"ключи (эпизод, шаг) не уникальны: {N} строк, "
                         f"{len(uniq)} различных")
    sel = np.where(split == args.split)[0]
    if len(sel) == 0:
        raise SystemExit(f"в кэше нет части «{args.split}»")
    print(f"кэш: {N} строк, часть «{args.split}» — {len(sel)}, "
          f"эпизодов {len(np.unique(epi[sel]))}")

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([qq.out_project(qq.decode_code(ii))[0]
                         for qq in codec.vq.quantizers]).float().to(dev)

    # ПРЕДСКАЗАНИЕ coarse24: уровень 0, позиция 0 чанка.
    pred = np.zeros((len(sel), 7), np.float64)
    for i0 in range(0, len(sel), args.batch):
        b = sel[i0:i0 + args.batch]
        with torch.no_grad():
            k = torch.as_tensor(q[b]).long().to(dev)
            x, _ = codec._decode(E[0][k], embodiment_ids=0)
        pred[i0:i0 + len(b)] = x[:, 0, :7].float().cpu().numpy()

    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    def to_codec_space(a):
        """Та же нормировка и ОБРЕЗКА, что в K-9a и K-10d."""
        a = np.asarray(a, np.float64).copy()
        a[..., :-1] = a[..., :-1] / max_act_q[..., :-1]
        a[..., -1] = -a[..., -1]
        return np.clip(a, -1.0, 1.0)

    rid, rev = "physical-intelligence/libero", "v2.0"
    edges = [int(x) for x in args.bucket_edges.split(",")]
    names = ge.bucket_names(edges)

    demo = np.full((len(sel), 7), np.nan)
    rem = np.full(len(sel), -1, np.int64)
    pos_in_sel = {int(e): np.where(epi[sel] == e)[0] for e in np.unique(epi[sel])}
    n_ep = 0
    for e, rows in pos_in_sel.items():
        try:
            f = hf_hub_download(
                rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                repo_type="dataset", revision=rev)
            tab = pq.read_table(f)
            acts = np.asarray(tab.column("actions").to_pylist(), np.float32)
            stt = np.asarray(tab.column("state").to_pylist(), np.float32)
        except Exception as ex:                      # noqa: BLE001
            raise SystemExit(f"эпизод {e} не загрузился: {ex}. Опора обязана "
                             f"покрывать 100% строк.")
        if stt.shape[1] == len(STATE_Q01) + 1:
            stt = process_state(stt)
        a = to_codec_space(acts)
        # РАЗМЕТКА ИЗ ОБЩЕГО МОДУЛЯ, тем же вызовом, что в K-10d.
        ev, typ, _ = ge.label(a, stt[:, :3], kind=args.event,
                              speed_frac=args.speed_frac,
                              min_dwell=args.min_dwell,
                              min_travel=args.min_travel,
                              merge_tol=args.merge_tol)
        _, _, r = ge.targets(a, ev, typ)
        st_rows = stp[sel][rows]
        if st_rows.max() >= len(a):
            raise SystemExit(f"эпизод {e}: шаг {st_rows.max()} вне длины "
                             f"{len(a)}")
        demo[rows] = a[st_rows]
        rem[rows] = r[st_rows]
        n_ep += 1
        if n_ep % 100 == 0:
            print(f"  эпизодов {n_ep}/{len(pos_in_sel)}", flush=True)

    if np.isnan(demo).any() or (rem < 0).any():
        raise SystemExit("покрытие неполное — часть строк не сопоставлена")
    print(f"покрытие 100%: {len(sel)} строк, {n_ep} эпизодов")

    d = pred - demo
    err_pos = np.linalg.norm(d[:, :3], axis=1) / np.sqrt(3)
    err_rot = np.linalg.norm(d[:, 3:6], axis=1) / np.sqrt(3)
    err_pose = np.linalg.norm(d[:, :6], axis=1) / np.sqrt(6)
    grip_bad = (np.sign(pred[:, 6]) != np.sign(demo[:, 6])).astype(float)
    bi = np.searchsorted(np.asarray(edges), rem, side="right")
    cur = curve(err_pos, err_rot, err_pose, grip_bad, bi, names)

    print(f"\n  опора coarse24, событие «{args.event}», часть «{args.split}»")
    print(f"  {'удалённость':>13}{'строк':>9}{'позиция':>10}{'вращение':>11}"
          f"{'смесь':>9}{'знак':>8}")
    for nm in names:
        if nm not in cur:
            continue
        c = cur[nm]
        print(f"  {nm:>13}{c['n']:>9}{c['pos']:>10.4f}{c['rot']:>11.4f}"
              f"{c['pose']:>9.4f}{c['grip']:>7.1%}")

    print(f"\n  ЧИТАТЬ ТАК: это НЕ результат, а опора. Контроллер в K-10d "
          f"сравнивается\n  с этой кривой побакетно, и только с ней: "
          f"перенесённое число 0.0478\n  измерено на другой выборке и против "
          f"учителя.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    json.dump(dict(buckets=cur, edges=edges, event=args.event,
                   split=args.split, n_rows=int(len(sel)), n_episodes=n_ep,
                   # ТОЧНЫЕ КЛЮЧИ ДЛЯ ОЦЕНКИ: K-10d обязан считаться на тех же
                   # наблюдениях, а не на похожих.
                   eval_keys=[[int(a_), int(b_)] for a_, b_ in
                              zip(epi[sel], stp[sel])],
                   eval_remaining=[int(x) for x in rem],
                   cache=os.path.abspath(args.cache), ckpt=args.ckpt,
                   script_sha1=sha,
                   goal_events_sha1=hashlib.sha1(
                       open(ge.__file__, "rb").read()).hexdigest()[:12],
                   event_params=dict(speed_frac=args.speed_frac,
                                     min_dwell=args.min_dwell,
                                     min_travel=args.min_travel,
                                     merge_tol=args.merge_tol),
                   argv=vars(args)),
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {args.out}  (sha {sha})")


if __name__ == "__main__":
    main()
