"""K-9f: факториальная таблица «ствол x голова» и отдельный контроль head-only.

ЗДЕСЬ ДВА РАЗНЫХ ВОПРОСА, И ИХ НЕЛЬЗЯ СМЕШИВАТЬ В ОДНОЙ ТАБЛИЦЕ.

ВОПРОС 1 — ИЗМЕНИЛОСЬ ЛИ ПРЕДСТАВЛЕНИЕ. Отвечает настоящий факториал: два
ствола на две КОНКРЕТНЫЕ головы, все четыре клетки без единого шага обучения.

                    голова R0 (исходная)   голова R1 (из Joint-12)
    ствол T0 (исх.)      T0+R0                  T0+R1
    ствол T1 (Joint)     T1+R0                  T1+R1

Ранняя версия этого файла ставила во вторую клетку ТРЕТЬЮ голову — заново
обученную R*. Это был не факториал, а сравнение трёх разных голов, и вывод по
нему был бы неверен.

ВОПРОС 2 — БЫЛО ЛИ ИЗМЕНЕНИЕ СТВОЛА НЕОБХОДИМО. Отвечает отдельный контроль
T0+R*, где R* обучается с нуля на замороженном исходном стволе. Это НЕ «доля
вклада головы»: ствол и голова адаптируются совместно, и причинную долю такое
сравнение не даёт. Корректная формулировка — доля прироста Joint-12,
воспроизводимая отдельно обученным считывателем.

ПРЕ-РЕГИСТРИРОВАННОЕ ЧТЕНИЕ, записано ДО запуска.
  Контроль T0+R*, доля прироста:
    * >= 85%  -> обновление ствола не требуется: тот же результат
                 воспроизводится одним считывателем;
    * <= 35%  -> одного считывателя недостаточно: изменение ствола необходимо
                 в проверенной постановке;
    * между   -> результат обеспечивается совместной адаптацией.
  Внедиагональные клетки:
    * T0+R1 близко к T1+R1 -> голова Joint-12 читает и ИСХОДНОЕ представление,
      то есть менявшийся ствол ей не нужен;
    * обе внедиагональные низкие -> сильная со-адаптация: представление
      изменилось так, что исходная голова его не читает, а новая требует
      именно его. Это свидетельство ИЗМЕНЕНИЯ, но не улучшения.
  Клетка T1+R0 доказательна АСИММЕТРИЧНО: высокое значение — сильное
  свидетельство переноса, низкое не значит ничего.

ДВЕ КЛЕТКИ ИЗ ЧЕТЫРЁХ ИМЕЮТ ИЗВЕСТНЫЙ ОТВЕТ, и это главная защита от ошибки в
конвейере: T0+R0 обязана воспроизвести эпоху 0, T1+R1 — эпоху 3. Сверяются не
только по согласию, но и по поза8 и знаку схвата: совпадение одной маргинальной
точности не гарантирует, что строки сопоставлены верно. Порог отказа — один
пункт, потому что весь исследуемый эффект составляет около десяти, и допуск в
три пункта прятал бы треть эффекта.

ЭПОХА 0 — ПОЛНОПРАВНЫЙ КАНДИДАТ у R*. Если всякий шаг обучения только портит
голову, контроль обязан вернуть исходную, а не худшую из восьми. Эта же ошибка
уже ловилась в K-7c и K-8b.

ОТБОР НА ВАЛИДАЦИИ, ИТОГ НА TEST. Шаг обучения, эпоха и сид выбираются по
валидации; доля воспроизводимого прироста считается на нетронутом test, иначе
она получила бы фору от того же отбора.

ЧЕСТНОСТЬ КОНТРОЛЯ. R* опровергает нашу же гипотезу, и поддавки ему в нашу
пользу обесценили бы вывод: перебор шага обучения, два сида, лучшая эпоха.

PYTHONPATH ОБЯЗАТЕЛЕН, ХОТЯ СИМУЛЯТОР ЗДЕСЬ НЕ ЗАПУСКАЕТСЯ: `utils` тянет
`libero.libero.benchmark` на импорте.

Запуск:
    python3 experiments/k9f_readout_table.py --selftest

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9f_readout_table.py --ckpt <base> \\
        --cache data/k9_teacher_150k.npz \\
        --orig data/k9e_orig --trained data/k9e_ep3 \\
        --expect-cell-t0r0 0.231 --expect-cell-t1r1 0.330 --out data/k9f
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8


def gain_fraction(control, joint, base):
    """Доля прироста Joint-12, воспроизводимая отдельным считывателем."""
    span = joint - base
    if span <= 0:
        return None
    return (control - base) / span


def read_control(frac):
    if frac is None:
        return "прирост не воспроизведён, таблица недействительна"
    if frac >= 0.85:
        return ("обновление ствола НЕ ТРЕБУЕТСЯ: результат воспроизводится "
                "одним считывателем")
    if frac <= 0.35:
        return ("одного считывателя НЕДОСТАТОЧНО: изменение ствола необходимо "
                "в проверенной постановке")
    return "результат обеспечивается совместной адаптацией"


def read_offdiag(t0r1, t1r0, base, joint):
    """Чтение внедиагональных клеток. Пороги записаны до запуска."""
    span = joint - base
    if span <= 0:
        return "недействительно"
    near = joint - 0.02                    # «близко к совместному»
    low = base + 0.2 * span                # «низко»
    if t0r1 >= near:
        return ("голова Joint-12 читает и ИСХОДНОЕ представление: менявшийся "
                "ствол ей не нужен")
    if t0r1 <= low and t1r0 <= low:
        return ("сильная со-адаптация: представление ИЗМЕНИЛОСЬ (но это не "
                "значит, что улучшилось)")
    if t1r0 >= near:
        return "перенос: исходная голова читает новое представление"
    return "промежуточный случай, однозначного чтения нет"


def selftest():
    assert abs(gain_fraction(0.330, 0.330, 0.231) - 1.0) < 1e-12
    assert gain_fraction(0.25, 0.20, 0.231) is None
    f = gain_fraction(0.26, 0.330, 0.231)
    assert 0.25 < f < 0.35, f          # доля от ПРИРОСТА, не от результата
    assert "НЕ ТРЕБУЕТСЯ" in read_control(0.9)
    assert "НЕДОСТАТОЧНО" in read_control(0.2)
    assert "совместной" in read_control(0.6)
    assert "недействительна" in read_control(None)

    b, j = 0.231, 0.330
    assert "ИСХОДНОЕ" in read_offdiag(0.325, 0.10, b, j)
    assert "со-адаптация" in read_offdiag(0.24, 0.24, b, j)
    assert "перенос" in read_offdiag(0.28, 0.329, b, j)
    assert "промежуточный" in read_offdiag(0.29, 0.28, b, j)

    # Взвешенное усреднение по батчам разного размера равно усреднению сразу.
    rng = np.random.default_rng(0)
    x = rng.random(1000)
    parts = [x[:300], x[300:700], x[700:]]
    w = sum(float(p.mean()) * len(p) for p in parts) / len(x)
    assert abs(w - float(x.mean())) < 1e-12
    # RMS складывается по КВАДРАТАМ.
    a, c = rng.random(400), rng.random(600)
    rms = math.sqrt((float((a ** 2).mean()) * 400
                     + float((c ** 2).mean()) * 600) / 1000)
    assert abs(rms - float(np.sqrt((np.concatenate([a, c]) ** 2).mean()))) < 1e-12

    # Эпоха 0 — кандидат: если все эпохи хуже, побеждает она.
    cands = [("эпоха 0", 0.231), ("эпоха 1", 0.20), ("эпоха 2", 0.19)]
    assert max(cands, key=lambda t: t[1])[0] == "эпоха 0"

    # Адресация внутри train: itr[pos] эквивалентна прямой и монотонна.
    sp = np.array(["train"] * 7 + ["val"] * 3)[rng.permutation(10)]
    itr = np.where(sp == "train")[0]
    TL = np.arange(10) * 10
    pos = np.sort(rng.permutation(len(itr))[:3])
    assert (TL[itr][pos] == TL[itr[pos]]).all()
    assert (np.diff(itr[pos]) > 0).all()
    print("самопроверка k9f пройдена (версия «факториал на двух головах»): "
          "доля прироста, чтение контроля и внедиагоналей, эпоха 0 в "
          "кандидатах, взвешенные средние и RMS, адресация train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--orig", help="префикс от k9e без --joint-ckpt")
    ap.add_argument("--trained", help="префикс от k9e с --joint-ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3")
    ap.add_argument("--seeds", default="0,1",
                    help="контроль обязан быть не слабее, чем может быть")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--hard-weight", type=float, default=0.25)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--preload-logits", choices=["auto", "on", "off"],
                    default="auto")
    ap.add_argument("--preload-limit-gib", type=float, default=16.0)
    ap.add_argument("--fit-trained", action="store_true",
                    help="дополнительно обучить считыватель на ОБУЧЕННОМ "
                         "стволе: показывает, ограничена ли клетка T1+R0 "
                         "со-адаптацией")
    ap.add_argument("--expect-cell-t0r0", type=float, default=None)
    ap.add_argument("--expect-cell-t1r1", type=float, default=None)
    ap.add_argument("--expect-pose-t0r0", type=float, default=None)
    ap.add_argument("--expect-pose-t1r1", type=float, default=None)
    ap.add_argument("--tol-warn", type=float, default=0.0025)
    ap.add_argument("--tol-fail", type=float, default=0.01)
    ap.add_argument("--tol-pose-fail", type=float, default=0.004)
    ap.add_argument("--out", default="data/k9f")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    for need in ("ckpt", "orig", "trained"):
        if not getattr(args, need):
            raise SystemExit(f"нужен --{need} (или --selftest)")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k9f sha1 {sha}")
    os.makedirs(args.out, exist_ok=True)

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import actioncodec  # noqa: F401
    from joint12_vla import kd_loss
    from utils import VisionLanguageActionProcessor

    dev = torch.device(args.device)

    # --- кэш учителя ----------------------------------------------------------
    z = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    q_teach = z["teacher_codes_q0"].astype(np.int64)
    K_true0 = z["K_true_q0"].astype(np.int64)
    split = z["split"]
    N, V = len(q_teach), int(meta["vocab"])
    T_LOG = np.load(args.cache + ".logits.npy", mmap_mode="r")
    assert T_LOG.shape == (N, N_POS, V), T_LOG.shape
    itr = np.where(split == "train")[0]
    iva = np.where(split == "val")[0]
    ite = np.where(split == "test")[0]
    print(f"кэш: {N} наблюдений; обучение {len(itr)}, валидация {len(iva)}, "
          f"test {len(ite)}")
    if len(ite) == 0:
        print("  ВНИМАНИЕ: части test нет, итог будет посчитан на валидации "
              "и получит фору от отбора")

    tl_gib = len(itr) * N_POS * V * T_LOG.dtype.itemsize / 2 ** 30
    n_runs = args.epochs * len(args.lrs.split(",")) * len(args.seeds.split(","))
    pre = (args.preload_logits == "on" or
           (args.preload_logits == "auto" and tl_gib <= args.preload_limit_gib))
    TL_tr = None
    if pre:
        print(f"логиты учителя обучающей части в память: {tl_gib:.1f} ГиБ "
              f"(иначе до {n_runs} проходов по диску)", flush=True)
        TL_tr = np.asarray(T_LOG[itr])
    else:
        print(f"логиты остаются на диске: {tl_gib:.1f} ГиБ больше порога")

    # --- два кэша h12 ---------------------------------------------------------
    def load_side(prefix, want_trunk):
        md = json.load(open(prefix + ".json"))
        if md["trunk"] != want_trunk:
            raise SystemExit(f"{prefix}: ствол «{md['trunk']}», ожидался "
                             f"«{want_trunk}»")
        if md["n"] != N:
            raise SystemExit(f"{prefix}: {md['n']} строк против {N}")
        if os.path.abspath(md["cache"]) != os.path.abspath(args.cache):
            raise SystemExit(f"{prefix} снят с другого кэша: {md['cache']}")
        h = np.load(md["h12_file"], mmap_mode="r")
        # ФОРМА И ТИП ПРОВЕРЯЮТСЯ ПО ФАКТУ, а не по метаданным: файл мог быть
        # оборван, перезаписан другим прогоном или снят другой версией.
        if h.shape != (N, N_POS, md["dim"]) or h.dtype != np.float16:
            raise SystemExit(f"{md['h12_file']}: форма {h.shape} тип "
                             f"{h.dtype}, ожидалось "
                             f"{(N, N_POS, md['dim'])} float16")
        rd = torch.load(md["readout_file"], map_location="cpu",
                        weights_only=False)
        mm = md.get("cache_vs_live_token_mismatch")
        cov = md.get("checked_fraction")
        if mm is None or cov is None:
            raise SystemExit(f"{prefix}: шум хранения не измерен, кэш снят "
                             f"старой версией k9e")
        if cov < 0.99:
            raise SystemExit(
                f"{prefix}: сверено лишь {cov:.1%} токенов. Порог 0.5% на "
                f"такой доле ничего не гарантирует; пересоберите с "
                f"--check-batches -1.")
        print(f"  {want_trunk}: {md['h12_file']}, шум хранения {mm:.4%} "
              f"на {cov:.1%} токенов, голова из {md['readout_file']}")
        return md, h, rd

    print("кэши h12:")
    md_o, H_o, rd_o = load_side(args.orig, "original")
    md_t, H_t, rd_t = load_side(args.trained, "trained")
    for k in ("joint12_vla_sha1", "script_sha1", "depth", "dim", "ckpt"):
        if md_o[k] != md_t[k]:
            raise SystemExit(
                f"кэши расходятся по «{k}»: {md_o[k]} против {md_t[k]}. "
                f"Клетки посчитаны разным кодом и несравнимы.")
    D = md_o["dim"]
    print(f"  оба кэша: глубина {md_o['depth']}, dim {D}, k9e sha "
          f"{md_o['script_sha1']}, joint12_vla sha {md_o['joint12_vla_sha1']}")

    # --- кодек и опорные действия --------------------------------------------
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь в action_processor")
    codec = codec.to(dev).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    def dec0(codes):
        """Действие из грубого уровня — ровно то, что исполняет симулятор."""
        out = []
        for i0 in range(0, len(codes), 256):
            k = torch.as_tensor(codes[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                x, _ = codec._decode(E[0][k], embodiment_ids=0)
            out.append(x[..., :7].float().cpu())
        return torch.cat(out)

    A_teach = dec0(q_teach)
    A_star = None
    if "K_true" in z.files:
        Kt3 = z["K_true"].astype(np.int64)
        outs = []
        for i0 in range(0, N, 256):
            k = torch.as_tensor(Kt3[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                zz = sum(E[j][k[:, j, :]] for j in range(N_LEVEL))
                x, _ = codec._decode(zz, embodiment_ids=0)
            outs.append(x[..., :7].float().cpu())
        A_star = torch.cat(outs)

    # --- считыватель ----------------------------------------------------------
    class Readout(nn.Module):
        """Норма плюс линейная голова, всё в fp32.

        Живой путь шёл под autocast fp16, здесь точнее. Разница обязана быть
        мала — и это не предположение: клетки T0+R0 и T1+R1 сверяются с
        известными числами при пороге в один пункт.
        """

        def __init__(self, norm, head):
            super().__init__()
            self.norm, self.head = norm, head

        def forward(self, h):
            return self.head(self.norm(h))

    def fresh(rd):
        import copy
        return Readout(copy.deepcopy(rd["norm"]),
                       copy.deepcopy(rd["head"])).to(dev).float()

    # --- оценка ---------------------------------------------------------------
    @torch.no_grad()
    def evaluate(Hc, ro, idxs, tag, quiet=False):
        ro.eval()
        acc_t = acc_k = 0.0
        se_i = sg_i = se_e = fl4 = fl8 = 0.0
        wsum = 0
        for i0 in range(0, len(idxs), args.batch):
            sel = idxs[i0:i0 + args.batch]
            h = torch.from_numpy(np.asarray(Hc[sel])).to(dev).float()
            pc = ro(h).argmax(-1).cpu().numpy()
            w = len(sel)
            acc_t += float((pc == q_teach[sel]).mean()) * w
            acc_k += float((pc == K_true0[sel]).mean()) * w
            a = dec0(pc)
            d_i = a - A_teach[sel]
            se_i += float((d_i[:, :H_EXEC, :6] ** 2).mean()) * w
            sg_i += float((d_i[:, :H_EXEC, 6] ** 2).mean()) * w
            fl4 += float((torch.sign(a[:, :4, 6])
                          != torch.sign(A_teach[sel][:, :4, 6])).float().mean()) * w
            fl8 += float((torch.sign(a[:, :H_EXEC, 6])
                          != torch.sign(A_teach[sel][:, :H_EXEC, 6])).float().mean()) * w
            if A_star is not None:
                se_e += float(((a - A_star[sel])[:, :H_EXEC, :6] ** 2).mean()) * w
            wsum += w
        r = dict(acc_teacher=acc_t / wsum, acc_ktrue=acc_k / wsum,
                 imit_pose8=math.sqrt(se_i / wsum),
                 imit_grip8=math.sqrt(sg_i / wsum),
                 grip_flip4=fl4 / wsum, grip_flip8=fl8 / wsum,
                 expert_pose8=(math.sqrt(se_e / wsum)
                               if A_star is not None else None), n=wsum)
        if not quiet:
            print(f"  [{tag}] согласие {r['acc_teacher']:.1%} "
                  f"(с токенизатором {r['acc_ktrue']:.1%}); поза8 "
                  f"{r['imit_pose8']:.4f}, знак8 {r['grip_flip8']:.2%}"
                  + (f"; до эксперта {r['expert_pose8']:.4f}"
                     if r['expert_pose8'] is not None else ""))
        return r

    # =========================================================================
    # ВОПРОС 1: факториал на двух КОНКРЕТНЫХ головах, без обучения
    # =========================================================================
    print(f"\n{'=' * 70}\n  ВОПРОС 1: изменилось ли представление "
          f"(четыре клетки без обучения)")
    cells = {}
    for key, Hc, rd, tag in (
            ("T0R0", H_o, rd_o, "T0+R0 ствол исходный, голова исходная"),
            ("T0R1", H_o, rd_t, "T0+R1 ствол исходный, голова Joint-12"),
            ("T1R0", H_t, rd_o, "T1+R0 ствол Joint-12, голова исходная"),
            ("T1R1", H_t, rd_t, "T1+R1 ствол Joint-12, голова Joint-12")):
        cells[key] = evaluate(Hc, fresh(rd), iva, tag)

    # --- сверка клеток с известным ответом ------------------------------------
    print("\n  клетки с известным ответом:")
    bad = []
    for nm, r, ea, ep in (("T0+R0", cells["T0R0"], args.expect_cell_t0r0,
                           args.expect_pose_t0r0),
                          ("T1+R1", cells["T1R1"], args.expect_cell_t1r1,
                           args.expect_pose_t1r1)):
        if ea is not None:
            d = abs(r["acc_teacher"] - ea)
            print(f"    {nm} согласие: {r['acc_teacher']:.1%} против "
                  f"{ea:.1%}, расхождение {d * 100:.2f} пп")
            if d > args.tol_fail:
                bad.append(f"{nm} согласие {d * 100:.2f} пп")
            elif d > args.tol_warn:
                print(f"      ВНИМАНИЕ: больше {args.tol_warn * 100:.2f} пп; "
                      f"вероятно, fp32 против autocast fp16")
        else:
            print(f"    {nm} согласие: ожидание не задано, сверки нет")
        if ep is not None:
            d = abs(r["imit_pose8"] - ep)
            print(f"    {nm} поза8: {r['imit_pose8']:.4f} против {ep:.4f}, "
                  f"расхождение {d:.4f}")
            if d > args.tol_pose_fail:
                bad.append(f"{nm} поза8 {d:.4f}")
    if bad:
        raise SystemExit(
            "клетки с известным ответом не воспроизведены (" + "; ".join(bad)
            + ").\nЭто отказ конвейера, а не результат: остальные клетки "
            "читать нельзя.")

    base, joint = cells["T0R0"]["acc_teacher"], cells["T1R1"]["acc_teacher"]
    print(f"\n  {'':<6}{'R0 исходная':>14}{'R1 Joint-12':>14}")
    print(f"  {'T0':<6}{cells['T0R0']['acc_teacher']:>13.1%}"
          f"{cells['T0R1']['acc_teacher']:>14.1%}")
    print(f"  {'T1':<6}{cells['T1R0']['acc_teacher']:>13.1%}"
          f"{cells['T1R1']['acc_teacher']:>14.1%}")
    od = read_offdiag(cells["T0R1"]["acc_teacher"],
                      cells["T1R0"]["acc_teacher"], base, joint)
    print(f"  ЧТЕНИЕ: {od}")
    print(f"  напоминание: T1+R0 доказательна только если ВЫСОКА")

    # =========================================================================
    # ВОПРОС 2: контроль head-only, единственный требующий обучения
    # =========================================================================
    def fit(Hc, rd, lr, seed, tag):
        """Считыватель с нуля на замороженном стволе.

        ЭПОХА 0 — ПОЛНОПРАВНЫЙ КАНДИДАТ. Если обучение только портит, контроль
        обязан вернуть исходную голову, а не худшую из восьми: иначе контроль
        окажется ниже бездействия и мы объявим прирост принадлежащим стволу по
        артефакту отбора.
        """
        torch.manual_seed(seed)
        ro = fresh(rd)
        opt = torch.optim.AdamW(ro.parameters(), lr=lr,
                                weight_decay=args.weight_decay)
        rng = np.random.default_rng(seed)
        # ПОДВЫБОРКА ОБУЧЕНИЯ РАЗМЕРОМ С ВАЛИДАЦИЮ. Без неё «контроль встал»
        # неотличимо от трёх разных причин: переобучение, нехватка ёмкости
        # считывателя, плохая цель. У Joint-12 разрыв дошёл до +40.8 пункта, и
        # у контроля его надо мерить тем же способом, а не предполагать.
        tr_sub = rng.choice(itr, size=min(len(iva), len(itr)), replace=False)
        ev0 = evaluate(Hc, ro, iva, f"{tag} эпоха 0", quiet=True)
        best = dict(ev0, epoch=0)
        best_state = {k: v.detach().clone() for k, v in ro.state_dict().items()}
        hist = [dict(epoch=0, loss=None,
                     **{k: v for k, v in ev0.items() if k != "n"})]
        for ep in range(1, args.epochs + 1):
            ro.train()
            # ПЕРЕМЕШИВАЮТСЯ ПОЗИЦИИ ВНУТРИ itr: одна выборка адресует и
            # предзагруженный массив, и memmap, порядок остаётся монотонным.
            order = rng.permutation(len(itr))
            tot, nb = 0.0, 0
            for i0 in range(0, len(order), args.batch):
                pos = np.sort(order[i0:i0 + args.batch])
                sel = itr[pos]
                h = torch.from_numpy(np.asarray(Hc[sel])).to(dev).float()
                tl = torch.from_numpy(
                    TL_tr[pos] if TL_tr is not None
                    else np.asarray(T_LOG[sel])).to(dev).float()
                y = torch.from_numpy(q_teach[sel]).to(dev)
                lg = ro(h)
                # ТА ЖЕ ПОТЕРЯ, ЧТО У JOINT-12: KD при T=2 плюс 0.25 жёсткой.
                loss = kd_loss(lg, tl, args.temperature) + args.hard_weight * \
                    F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss); nb += 1
            ev = evaluate(Hc, ro, iva, f"{tag} lr={lr:g} сид {seed} эпоха {ep}")
            evt = evaluate(Hc, ro, tr_sub, "", quiet=True)
            gap = evt["acc_teacher"] - ev["acc_teacher"]
            print(f"      разрыв обучение−валидация: согласие {gap * 100:+.1f} "
                  f"пп, поза8 {evt['imit_pose8'] - ev['imit_pose8']:+.4f}")
            hist.append(dict(epoch=ep, loss=tot / max(nb, 1), gap=gap,
                             train_acc=evt["acc_teacher"],
                             train_pose8=evt["imit_pose8"],
                             **{k: v for k, v in ev.items() if k != "n"}))
            if ev["acc_teacher"] > best["acc_teacher"]:
                best = dict(ev, epoch=ep)
                best_state = {k: v.detach().clone()
                              for k, v in ro.state_dict().items()}
        ro.load_state_dict(best_state)
        return ro, best, hist

    def sweep(Hc, rd, tag):
        best_ro = best_ev = None
        best_cfg, hists = None, {}
        for lr in [float(x) for x in args.lrs.split(",")]:
            for sd in [int(x) for x in args.seeds.split(",")]:
                ro, ev, hist = fit(Hc, rd, lr, sd, tag)
                hists[f"lr{lr:g}_seed{sd}"] = hist
                print(f"  {tag} lr={lr:g} сид {sd}: лучшее "
                      f"{ev['acc_teacher']:.1%} на эпохе {ev['epoch']}")
                if best_ev is None or ev["acc_teacher"] > best_ev["acc_teacher"]:
                    best_ro, best_ev = ro, ev
                    best_cfg = dict(lr=lr, seed=sd, epoch=ev["epoch"])
        return best_ro, best_ev, best_cfg, hists

    print(f"\n{'=' * 70}\n  ВОПРОС 2: было ли изменение ствола необходимо")
    print("  контроль T0+R*: считыватель с нуля на ЗАМОРОЖЕННОМ исходном "
          "стволе")
    ro_star, ev_star, cfg_star, hist_star = sweep(H_o, rd_o, "T0+R*")
    torch.save(ro_star.state_dict(), os.path.join(args.out, "head_only.pt"))
    print(f"  выбрано по валидации: {cfg_star}")

    extra = None
    if args.fit_trained:
        print(f"\n  дополнительно T1+R*: считыватель с нуля на ОБУЧЕННОМ "
              f"стволе (ограничена ли T1+R0 со-адаптацией)")
        _, ev_x, cfg_x, hist_x = sweep(H_t, rd_o, "T1+R*")
        extra = dict(best=ev_x, cfg=cfg_x)
        hist_star.update({f"trained_{k}": v for k, v in hist_x.items()})

    # =========================================================================
    # ИТОГ НА TEST
    # =========================================================================
    fin = ite if len(ite) else iva
    fin_name = "test" if len(ite) else "валидация (test отсутствует)"
    print(f"\n{'=' * 70}\n  ИТОГ на {fin_name}: {len(fin)} наблюдений")
    t_base = evaluate(H_o, fresh(rd_o), fin, "T0+R0")
    t_joint = evaluate(H_t, fresh(rd_t), fin, "T1+R1")
    t_star = evaluate(H_o, ro_star, fin, "T0+R*")
    t_t0r1 = evaluate(H_o, fresh(rd_t), fin, "T0+R1")
    t_t1r0 = evaluate(H_t, fresh(rd_o), fin, "T1+R0")

    b, j, s = (t_base["acc_teacher"], t_joint["acc_teacher"],
               t_star["acc_teacher"])
    frac = gain_fraction(s, j, b)
    print(f"\n  {'ствол':<8}{'голова':<12}{'согласие':>10}{'поза8':>9}"
          f"{'знак8':>8}")
    for ls, lh, r in (("исходный", "исходная", t_base),
                      ("исходный", "Joint-12", t_t0r1),
                      ("Joint-12", "исходная", t_t1r0),
                      ("Joint-12", "Joint-12", t_joint),
                      ("исходный", "R* обуч.", t_star)):
        print(f"  {ls:<8}{lh:<12}{r['acc_teacher']:>9.1%}"
              f"{r['imit_pose8']:>9.4f}{r['grip_flip8']:>8.2%}")
    print(f"\n  прирост Joint-12: {(j - b) * 100:+.1f} пп")
    print(f"  воспроизведено отдельным считывателем: {(s - b) * 100:+.1f} пп"
          + (f" = {frac:.0%} прироста" if frac is not None else ""))
    print(f"  ВОПРОС 1 (изменилось ли представление): "
          f"{read_offdiag(t_t0r1['acc_teacher'], t_t1r0['acc_teacher'], b, j)}")
    print(f"  ВОПРОС 2 (нужно ли было менять ствол): {read_control(frac)}")
    print("\n  Это НЕ причинная «доля вклада головы»: ствол и голова "
          "адаптируются\n  совместно. Корректно — доля прироста Joint-12, "
          "воспроизводимая\n  отдельно обученным считывателем.")

    md = dict(script_sha1=sha,
              val_cells={k: v for k, v in cells.items()},
              test_cells=dict(T0R0=t_base, T0R1=t_t0r1, T1R0=t_t1r0,
                              T1R1=t_joint, T0Rstar=t_star),
              control_val=ev_star, control_cfg=cfg_star, extra=extra,
              history=hist_star, final_split=fin_name,
              base=b, joint=j, control=s, gain_fraction=frac,
              verdict_representation=read_offdiag(
                  t_t0r1["acc_teacher"], t_t1r0["acc_teacher"], b, j),
              verdict_control=read_control(frac),
              orig=md_o, trained=md_t, argv=vars(args))
    p = os.path.join(args.out, "table.json")
    json.dump(md, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {p}")


if __name__ == "__main__":
    main()
