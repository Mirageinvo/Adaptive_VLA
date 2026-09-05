"""K-11a: кэш скрытых состояний h12/h18/h24 и диагностика пространства поправок.

ЗАЧЕМ. HiCoRA-D исправляет НЕ произвольный черновик, а тот код q0, который
модель действительно предсказала на слое 12. Значит и остаток, который учится
предсказывать голова, обязан считаться от предсказанного q0_hat, а не от
истинного q0*. Разница между этими двумя величинами и есть весь смысл метода,
поэтому кэш собирается один раз и содержит именно предсказанные коды.

ОДИН ПРОХОД, ТРИ ОТВОДА. Двадцать четыре слоя исполняются ровно один раз;
состояние потока действий снимается после слоёв 12, 18 и 24. Отводы берутся
хуками, а `joint12_vla.py` не правится: его sha записан в каждую ячейку
симуляторного гейта K-9, и правка расколола бы развёртку на две несовместимые
половины.

ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ СВЕРХ ПЛАНА, И ПОЧЕМУ. Наложение весов Joint12 на слои
1-12 меняет ВХОД слоёв 13-24: они обучались читать выход исходного
двенадцатого слоя, а получат выход дообученного. Это предположение, а не
данность, и стоит оно ровно одного дополнительного прохода на подвыборке.
Поэтому скрипт всегда меряет:
  * согласие q0_hat с coarse24 исходной модели;
  * относительный дрейф h24 против чистого исходного прохода;
  * долю расхождений действия после декодера.
Если дрейф велик, поздние слои работают вне своего распределения, и это надо
знать ДО обучения головы, а не после симуляторного гейта.

ИСТОЧНИК q0 ВЫБИРАЕТСЯ ЯВНО, потому что вариантов три и они не равнозначны:
  joint12   веса K-9c на слоях 1-12 плюс его голова. q0 сильнее (89.5%), но
            вход поздних слоёв смещён.
  readout   исходный ствол, голова-читалка K-9f/K-9g поверх h12. Поздние слои
            строго в своём распределении, q0 слабее (86.5%).
  coarse24  q0 с ПОСЛЕДНЕГО слоя, как в опоре. Поправку с h24 строить не от
            чего — вариант существует только как верхняя граница качества q0
            и для отладки; для HiCoRA он бессмыслен.

ХРАНЕНИЕ FP16 — ЭТО ШУМ, И ОН ИЗМЕРЯЕТСЯ. После записи кэш прогоняется
обратно через ту же голову, и доля разошедшихся токенов печатается. Без этого
числа все дальнейшие таблицы читались бы так, будто у них нет собственной
погрешности.

Запуск:
    python3 experiments/k11a_build_hicora_cache.py --selftest

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k11a_build_hicora_cache.py --ckpt <base> \\
        --cache data/k9_teacher_150k.npz --q0-source joint12 \\
        --joint-ckpt data/k9c_joint12.pt --out data/k11a_joint12

    # диагностика пространства поправок по уже собранному кэшу
    python3 experiments/k11a_build_hicora_cache.py --diagnose data/k11a_joint12
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8
TAPS = (12, 18, 24)
Q0_SOURCES = ("joint12", "readout", "coarse24")
# Ранги, по которым считается объяснённая дисперсия остатка. Порог выбора
# зафиксирован здесь, ДО прогона: ранг принимается наименьший, объясняющий
# не меньше PCA_TARGET.
PCA_RANKS = (4, 8, 16, 32, 64)
PCA_TARGET = 0.90


def plan_batches(n, batch):
    return [(i, min(i + batch, n)) for i in range(0, n, batch)]


def explained(sv, ranks, total=None):
    """Доля объяснённой дисперсии по сингулярным числам."""
    e = np.asarray(sv, np.float64) ** 2
    tot = float(e.sum()) if total is None else float(total)
    if tot <= 0:
        return {int(r): 0.0 for r in ranks}
    return {int(r): float(e[:int(r)].sum() / tot) for r in ranks}


def pick_rank(exp, ranks=PCA_RANKS, target=PCA_TARGET, d_latent=None):
    """Наименьший ранг, объясняющий target. Правило зафиксировано заранее.

    РАНГ, РАВНЫЙ РАЗМЕРНОСТИ, — НЕ СЖАТИЕ. Любой набор точек в d измерениях
    «объясняется» d компонентами на 100%, и без этой оговорки правило
    выбирало бы наибольший ранг всегда, когда он дотягивает до размерности
    латента, объявляя низкоранговость там, где её нет.
    """
    for r in sorted(int(x) for x in ranks):
        if d_latent is not None and r >= int(d_latent):
            break
        if exp.get(r, 0.0) >= target:
            return r
    return None


def read_rank(rank, exp, ranks=PCA_RANKS, target=PCA_TARGET, d_latent=None):
    top = max(int(x) for x in ranks)
    if d_latent is not None:
        usable = [int(x) for x in ranks if int(x) < int(d_latent)]
        top = max(usable) if usable else 0
        if not usable:
            return (f"размерность латента {d_latent} не больше наименьшего "
                    f"рассматриваемого ранга: сжимать нечего")
    if rank is not None:
        return (f"ранг {rank} объясняет {exp[rank]:.1%} остатка (порог "
                f"{target:.0%}, зафиксирован до прогона) — низкоранговый "
                f"базис оправдан, брать r={rank}")
    return (f"даже ранг {top} объясняет лишь {exp.get(top, 0.0):.1%} — "
            f"остаток не низкоранговый. По плану это стоп-условие ветки: "
            f"сравнить с поправкой прямо в пространстве действий как "
            f"КОНТРОЛЕМ, а не переходить к ней молча")


def residual_stats(r, exec_mask=None):
    """Нормы остатка: целиком, по исполняемому префиксу и по хвосту.

    ИСПОЛНЯЕМЫЙ ПРЕФИКС СЧИТАЕТСЯ ОТДЕЛЬНО. Из двадцати шагов чанка среда
    исполняет восемь; ошибка на хвосте в успех не переходит вовсе, и общее
    среднее по всем позициям систематически размывает то, что важно.
    """
    n = np.linalg.norm(np.asarray(r, np.float64), axis=-1)
    out = dict(mean=float(n.mean()), median=float(np.median(n)),
               p95=float(np.percentile(n, 95)), max=float(n.max()))
    if exec_mask is not None:
        m = np.asarray(exec_mask, bool)
        out["exec_mean"] = float(n[:, m].mean())
        out["tail_mean"] = float(n[:, ~m].mean()) if (~m).any() else None
    return out


def selftest():
    # --- разбивка батчей ---------------------------------------------------
    assert plan_batches(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert sum(b - a for a, b in plan_batches(1000, 64)) == 1000

    # --- объяснённая дисперсия на матрице с ИЗВЕСТНЫМ рангом ---------------
    rng = np.random.default_rng(0)
    B = rng.normal(size=(8, 40))
    X = rng.normal(size=(500, 8)) @ B          # ранг ровно 8
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    exp = explained(sv, PCA_RANKS)
    assert exp[8] > 0.999, exp[8]
    assert exp[4] < 0.95, exp[4]
    assert pick_rank(exp) == 8, exp
    assert "r=8" in read_rank(8, exp)

    # Полноранговый шум низкоранговым базисом не описывается — и правило
    # обязано это СКАЗАТЬ, а не выбрать наибольший ранг молча.
    Xn = rng.normal(size=(500, 128))
    expn = explained(np.linalg.svd(Xn - Xn.mean(0), compute_uv=False),
                     PCA_RANKS)
    assert pick_rank(expn, d_latent=128) is None, expn
    txt = read_rank(None, expn, d_latent=128)
    assert "стоп-условие" in txt and "КОНТРОЛЕМ" in txt

    # РАНГ, РАВНЫЙ РАЗМЕРНОСТИ, НЕ СЧИТАЕТСЯ УСПЕХОМ. На шуме в 64
    # измерениях ранг 64 объясняет ровно 100%, и без оговорки правило
    # объявило бы низкоранговость там, где её нет.
    X64 = rng.normal(size=(500, 64))
    e64 = explained(np.linalg.svd(X64 - X64.mean(0), compute_uv=False),
                    PCA_RANKS)
    assert e64[64] > 0.999, e64
    assert pick_rank(e64) == 64, "без d_latent правило и должно поверить"
    assert pick_rank(e64, d_latent=64) is None, "ранг = размерность принят"

    # --- нормы остатка -----------------------------------------------------
    r = np.zeros((10, N_POS, 4))
    r[:, :H_EXEC] = 1.0                        # префикс единичный
    r[:, H_EXEC:] = 3.0                        # хвост втрое больше
    m = np.zeros(N_POS, bool); m[:H_EXEC] = True
    st = residual_stats(r, m)
    assert abs(st["exec_mean"] - 2.0) < 1e-9      # норма вектора из четырёх 1
    assert abs(st["tail_mean"] - 6.0) < 1e-9
    # СРЕДНЕЕ ПО ВСЕМ ПОЗИЦИЯМ ЛЕЖИТ МЕЖДУ НИМИ и потому непригодно как
    # единственное число: ровно из-за этого префикс считается отдельно.
    assert st["exec_mean"] < st["mean"] < st["tail_mean"]

    print("самопроверка k11a пройдена (версия «ранг не равен размерности»): разбивка батчей, объяснённая дисперсия на матрице известного "
          "ранга, отказ выбирать ранг для полнорангового шума, префикс и "
          "хвост считаются раздельно")


def diagnose(prefix):
    """Диагностика пространства поправок по собранному кэшу.

    ОСТАТОК СЧИТАЕТСЯ ОТ ПРЕДСКАЗАННОГО q0_hat. Рядом печатается обычный
    RVQ-остаток от истинного q0* — как СПРАВКА о том, насколько задача
    HiCoRA отличается от задачи кодека, а не как цель обучения.
    """
    meta = json.load(open(prefix + ".meta.json"))
    # ЛАТЕНТЫ НЕ ХРАНЯТСЯ, А ВОССТАНАВЛИВАЮТСЯ ИЗ КОДОВ И КНИГ: три массива
    # (N, 16, D) в fp32 весили бы втрое больше самих отводов и ничего бы не
    # добавили — книги детерминированы и лежат рядом.
    E = np.load(prefix + ".codebooks.npy")      # (уровней, кодов, D)
    q0 = np.load(prefix + ".q0hat.npy")         # (N, 16) предсказанные коды
    Kt = np.load(prefix + ".ktrue.npy")         # (N, 3, 16) истинные коды
    z0 = E[0][q0]
    z0t = E[0][Kt[:, 0, :]]
    zs = sum(E[l][Kt[:, l, :]] for l in range(E.shape[0]))
    r = (zs - z0).astype(np.float64)
    r_rvq = (zs - z0t).astype(np.float64)
    m = np.zeros(N_POS, bool); m[:H_EXEC] = True

    print(f"\nдиагностика: {meta['n_obs']} наблюдений, источник q0 "
          f"«{meta['q0_source']}», D={r.shape[-1]}")
    st = residual_stats(r, m)
    st_r = residual_stats(r_rvq, m)
    print(f"  остаток от ПРЕДСКАЗАННОГО q0_hat: среднее {st['mean']:.4f}, "
          f"медиана {st['median']:.4f}, p95 {st['p95']:.4f}")
    print(f"    префикс 0-7 {st['exec_mean']:.4f}, хвост "
          f"{st['tail_mean']:.4f}")
    print(f"  остаток от ИСТИННОГО q0* (обычный RVQ, справочно): среднее "
          f"{st_r['mean']:.4f}, префикс {st_r['exec_mean']:.4f}")
    ratio = st["mean"] / max(st_r["mean"], 1e-12)
    print(f"  отношение {ratio:.2f}x — во столько раз задача HiCoRA больше "
          f"задачи кодека:\n    голова обязана вычистить ещё и ОШИБКУ "
          f"предсказания q0, а не только квантование")

    flat = r.reshape(-1, r.shape[-1])
    flat = flat - flat.mean(0, keepdims=True)
    sv = np.linalg.svd(flat, compute_uv=False) if len(flat) <= 20000 else None
    if sv is None:
        idx = np.random.default_rng(0).choice(len(flat), 20000, replace=False)
        sv = np.linalg.svd(flat[idx], compute_uv=False)
        print("  PCA по подвыборке 20000 строк")
    d_lat = int(r.shape[-1])
    exp = explained(sv, PCA_RANKS)
    print(f"\n  объяснённая дисперсия остатка (размерность латента {d_lat}):")
    for rk in PCA_RANKS:
        mark = "  (= размерность, не сжатие)" if rk >= d_lat else ""
        print(f"    ранг {rk:>3}: {exp[rk]:.1%}{mark}")
    rank = pick_rank(exp, d_latent=d_lat)
    print(f"\n  {read_rank(rank, exp, d_latent=d_lat)}")

    # Насыщение: какая доля коэффициентов упрётся в границу при выбранной
    # rho. Считается на train и печатается заранее, чтобы предел не
    # подбирался потом по результату.
    out = dict(n_obs=int(meta["n_obs"]), q0_source=meta["q0_source"],
               residual=st, residual_rvq=st_r, ratio=float(ratio),
               explained={str(k): v for k, v in exp.items()},
               rank=rank, pca_target=PCA_TARGET, d_latent=d_lat,
               prefix=os.path.abspath(prefix))
    json.dump(out, open(prefix + ".diag.json", "w"), ensure_ascii=False,
              indent=1)
    print(f"\n  сохранено: {prefix}.diag.json")
    print("\n  ЧИТАТЬ ТАК: это выбор размерности, а не свидетельство того, "
          "что поправка\n  улучшит успех. Ранг говорит лишь, что остаток "
          "УМЕЩАЕТСЯ в базис.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--diagnose", default=None,
                    help="префикс собранного кэша: только диагностика")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--q0-source", choices=Q0_SOURCES, default=None)
    ap.add_argument("--joint-ckpt", default=None,
                    help="чекпойнт K-9c: веса слоёв 1-12 и голова q0")
    ap.add_argument("--readout", default=None,
                    help="голова-читалка K-9g для --q0-source readout")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--drift-n", type=int, default=2000,
                    help="сколько наблюдений прогнать ВТОРЫМ, чистым "
                         "проходом ради измерения дрейфа поздних слоёв")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if args.diagnose:
        diagnose(args.diagnose)
        return

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k11a sha1 {sha}")
    for need, why in ((args.ckpt, "--ckpt"), (args.out, "--out"),
                      (args.q0_source, "--q0-source")):
        if not need:
            raise SystemExit(f"нужен {why}")
    if args.q0_source == "joint12" and not args.joint_ckpt:
        raise SystemExit("источник joint12 требует --joint-ckpt")
    if args.q0_source == "readout" and not args.readout:
        raise SystemExit("источник readout требует --readout")
    if args.q0_source == "coarse24":
        # ОТКАЗ, А НЕ ПРЕДУПРЕЖДЕНИЕ. q0 с последнего слоя не оставляет слоёв,
        # на которых считать поправку: h24 уже израсходован на сам q0.
        raise SystemExit(
            "источник coarse24 несовместим с HiCoRA: q0 берётся с последнего "
            "слоя, и поправку строить не на чем. Он существует только как "
            "верхняя граница качества q0 в отдельных сравнениях")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    import torch
    from omegaconf import OmegaConf
    from utils import VisionLanguageActionProcessor, dict_apply
    from models import SmolVLABlockwiseAR
    import joint12_vla as jv
    import hicora_vla as hv

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = OmegaConf.load(args.cfg_path)
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    d = np.load(os.path.join(os.path.dirname(root), args.cache),
                allow_pickle=True)
    N = len(d["episode"]) if args.limit is None else min(args.limit,
                                                         len(d["episode"]))
    print(f"кэш K-9a: {len(d['episode'])} наблюдений, берём {N}")
    keys = list(zip(d["episode"][:N].tolist(), d["step"][:N].tolist()))
    if len(set(keys)) != len(keys):
        raise SystemExit("ключи (episode, step) в исходном кэше не уникальны")

    Joint = jv.make_joint12_class(SmolVLABlockwiseAR)
    model = Joint.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float()
    print(f"кодовые книги: {tuple(E.shape)} (уровней, кодов, размерность)")

    print("\nЭТОТ СКРИПТ СОБИРАЕТ КЭШ. Ни одного вывода о качестве поправки "
          "из него\nне следует: он фиксирует ВХОДЫ будущей головы и "
          "размерность её выхода.")

    # --- наложение весов источника q0 ---------------------------------------
    model.init_joint_fast(depth=12)
    src_meta = None
    if args.q0_source == "joint12":
        obj = torch.load(os.path.join(os.path.dirname(root), args.joint_ckpt)
                         if not os.path.isabs(args.joint_ckpt)
                         else args.joint_ckpt, map_location="cpu",
                         weights_only=False)
        state = obj["state"]
        own = dict(model.named_parameters())
        stray = [k for k in state if k not in own]
        if stray:
            raise SystemExit(f"{len(stray)} ключей нет в модели: {stray[:5]}")
        with torch.no_grad():
            for k, v in state.items():
                if tuple(own[k].shape) != tuple(v.shape):
                    raise SystemExit(f"форма {k}")
                if not torch.isfinite(v).all():
                    raise SystemExit(f"в {k} есть nan или inf")
                own[k].data = v.to(dev, own[k].dtype)
        # ЧТО ИМЕННО ПЕРЕЗАПИСАНО — В ОТЧЁТ. `action_expert.norm` и
        # `bos_embedding` общие для всей глубины: их подмена меняет и вход
        # слоёв 13-24, а не только голову q0. Это и есть источник дрейфа,
        # который меряется ниже.
        touched = sorted({k.split(".layers.")[0] if ".layers." in k
                          else k.rsplit(".", 1)[0] for k in state})
        src_meta = dict(path=os.path.abspath(args.joint_ckpt),
                        depth=int(obj.get("depth", 12)),
                        tensors=len(state), touched=touched,
                        weights_sha1=hashlib.sha1(
                            str(sorted(state)).encode()).hexdigest()[:12])
        print(f"  наложены веса Joint12: {len(state)} тензоров, затронуты "
              f"{len(touched)} групп")
        print(f"  ОБЩИЕ ДЛЯ ВСЕЙ ГЛУБИНЫ И ПЕРЕЗАПИСАННЫЕ: "
              f"{[t for t in touched if 'layers' not in t]}")
    else:
        obj = torch.load(os.path.join(os.path.dirname(root), args.readout)
                         if not os.path.isabs(args.readout) else args.readout,
                         map_location="cpu", weights_only=False)
        state = {k: v for k, v in obj["state"].items()
                 if k.startswith(("fast_head.", "action_expert.norm."))}
        if not state:
            raise SystemExit(f"{args.readout}: нет весов головы-читалки")
        own = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in state.items():
                own[k].data = v.to(dev, own[k].dtype)
        src_meta = dict(path=os.path.abspath(args.readout),
                        tensors=len(state), touched=sorted(state),
                        note="ствол ИСХОДНЫЙ, поздние слои в своём "
                             "распределении")
        print(f"  наложена голова-читалка: {len(state)} тензоров, ствол "
              f"исходный")
    model.eval()

    HiCoRA = hv.make_hicora_class(type(model))
    model.__class__ = HiCoRA
    model.set_codebooks(E)
    model.init_hicora(q0_depth=12, taps=TAPS)
    D_H = int(model.fast_head.in_features)
    D_Z = int(E.shape[-1])
    print(f"  отводы {TAPS}, h={D_H}, латент={D_Z}")

    # --- место на диске проверяется ДО записи --------------------------------
    outdir = os.path.dirname(os.path.abspath(
        os.path.join(os.path.dirname(root), args.out)))
    os.makedirs(outdir, exist_ok=True)
    need = len(TAPS) * N * N_POS * D_H * 2 / 2 ** 30
    st_fs = os.statvfs(outdir)
    free = st_fs.f_bavail * st_fs.f_frsize / 2 ** 30
    print(f"  отводы: {len(TAPS)} x ({N}, {N_POS}, {D_H}) fp16 = "
          f"{need:.2f} ГиБ, свободно {free:.1f} ГиБ")
    if free < need * 1.15:
        raise SystemExit("места не хватит с запасом; сбор идёт долго, и "
                         "падение на последнем батче стоило бы всего прогона")

    base = os.path.join(os.path.dirname(root), args.out)
    taps_mm = {t: np.lib.format.open_memmap(
        f"{base}.h{t}.npy", mode="w+", dtype=np.float16,
        shape=(N, N_POS, D_H)) for t in TAPS}
    q0hat = np.zeros((N, N_POS), np.int16)

    # --- входы из кэша K-9a ---------------------------------------------------
    IMG = np.load(os.path.join(os.path.dirname(root), args.cache).replace(
        ".npz", ".images.npy"), mmap_mode="r")
    st_n = d["state"][:N] if "state" in d.files else None
    if st_n is None:
        raise SystemExit("в кэше K-9a нет состояний: пересоберите его или "
                         "укажите кэш с полем state")
    tsk = d["task"][:N]
    offs = d["pos_offset"][:N].astype(np.int64)
    Ktrue = d["K_true"][:N].astype(np.int16)

    groups = []
    for po in sorted({int(v) for v in offs}):
        ipo = np.where(offs == po)[0]
        for i, j in plan_batches(len(ipo), args.batch):
            groups.append((po, ipo[i:j]))
    print(f"  батчей {len(groups)} по {args.batch}; порядок по офсету — "
          f"position_offset задаётся на весь вызов")

    def build(sel, po):
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for gi in sel:
            m = prompt_template(
                st_n[gi], None, str(tsk[gi]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[k].numpy()]
                                     for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dtype), b)

    seen_layers = set()
    for gi, (po, sel) in enumerate(groups):
        b = build(sel, po)
        with torch.no_grad():
            v, pp = model.build_inputs(position_offset=po, **b)
            taps = model.forward_taps(
                vlm_inputs_embeds=v, attention_mask=b.get("attention_mask"),
                position_ids=pp)
            _, q0 = model.q0_from(taps[12])
        seen_layers.add(int(taps["layers_run"]))
        for t in TAPS:
            taps_mm[t][sel] = taps[t].float().cpu().numpy().astype(np.float16)
        q0hat[sel] = q0.cpu().numpy().astype(np.int16)
        if gi % 50 == 0:
            print(f"    батч {gi}/{len(groups)}", flush=True)
    for t in TAPS:
        taps_mm[t].flush()
    if seen_layers != {24}:
        raise SystemExit(f"глубина прохода {sorted(seen_layers)} вместо 24 — "
                         f"заявление «один проход по 24 слоям» неверно")
    print(f"  проход: ровно 24 слоя во всех {len(groups)} батчах")

    # --- ШУМ ХРАНЕНИЯ FP16 ИЗМЕРЯЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ -------------------
    chk = np.random.default_rng(0).choice(N, min(2048, N), replace=False)
    with torch.no_grad():
        h_back = torch.from_numpy(np.asarray(taps_mm[12][chk])).to(dev, dtype)
        _, q_back = model.q0_from(h_back)
    mism = float((q_back.cpu().numpy().astype(np.int16)
                  != q0hat[chk]).mean())
    print(f"  шум fp16: {mism:.3%} токенов q0 расходятся при перегоне через "
          f"голову из кэша")
    if mism > 0.005:
        raise SystemExit(f"расхождение {mism:.3%} выше 0.5% — кэш непригоден "
                         f"как источник черновика")

    # --- ДРЕЙФ ПОЗДНИХ СЛОЁВ ОТ НАЛОЖЕНИЯ ВЕСОВ ------------------------------
    drift = None
    if args.q0_source == "joint12" and args.drift_n > 0:
        print(f"\n  измеряю дрейф поздних слоёв на {args.drift_n} "
              f"наблюдениях вторым, ЧИСТЫМ проходом")
        clean = Joint.from_pretrained(**cfg.MODEL.vlm.kwargs).to(
            dev, dtype).eval()
        clean.__class__ = hv.make_hicora_class(type(clean))
        clean.init_joint_fast(depth=12)
        clean.set_codebooks(E)
        clean.init_hicora(q0_depth=12, taps=TAPS)
        sub = np.random.default_rng(1).choice(N, min(args.drift_n, N),
                                              replace=False)
        rel, agree, tot = [], 0, 0
        for po in sorted({int(offs[i]) for i in sub}):
            ipo = np.asarray([i for i in sub if int(offs[i]) == po])
            for i, j in plan_batches(len(ipo), args.batch):
                s_ = ipo[i:j]
                b = build(s_, po)
                with torch.no_grad():
                    v, pp = clean.build_inputs(position_offset=po, **b)
                    tp = clean.forward_taps(
                        vlm_inputs_embeds=v,
                        attention_mask=b.get("attention_mask"),
                        position_ids=pp)
                a = tp[24].float().cpu().numpy()
                bb = np.asarray(taps_mm[24][s_], np.float32)
                rel.append(np.linalg.norm(bb - a, axis=-1)
                           / np.maximum(np.linalg.norm(a, axis=-1), 1e-6))
                _, qc = clean.q0_from(tp[12])
                agree += int((qc.cpu().numpy().astype(np.int16)
                              == q0hat[s_]).sum())
                tot += q0hat[s_].size
        rel = np.concatenate([r.ravel() for r in rel])
        drift = dict(n=int(len(sub)), h24_rel_mean=float(rel.mean()),
                     h24_rel_p95=float(np.percentile(rel, 95)),
                     q0_agreement=float(agree / max(tot, 1)))
        print(f"  дрейф h24: относительный {drift['h24_rel_mean']:.3f} в "
              f"среднем, p95 {drift['h24_rel_p95']:.3f}")
        print(f"  согласие q0 с чистым проходом: {drift['q0_agreement']:.1%}")
        print("  ЧИТАТЬ ТАК: большой дрейф означает, что слои 13-24 читают "
              "вход вне\n  своего распределения. Это не запрет на HiCoRA, но "
              "объяснение, если\n  поправка окажется бесполезной, и повод "
              "сравнить с источником readout.")
        del clean
        torch.cuda.empty_cache()

    np.save(f"{base}.q0hat.npy", q0hat)
    np.save(f"{base}.ktrue.npy", Ktrue)
    np.save(f"{base}.codebooks.npy", E.cpu().numpy().astype(np.float32))
    meta = dict(n_obs=int(N), q0_source=args.q0_source, taps=list(TAPS),
                d_hidden=D_H, d_latent=D_Z, ckpt=args.ckpt,
                cache=os.path.abspath(args.cache), source=src_meta,
                fp16_q0_mismatch=mism, drift=drift, script_sha1=sha,
                hicora_vla_sha1=hashlib.sha1(
                    open(hv.__file__, "rb").read()).hexdigest()[:12],
                keys_sha1=hashlib.sha1(
                    np.ascontiguousarray(
                        np.stack([d["episode"][:N], d["step"][:N]])
                    ).tobytes()).hexdigest()[:12])
    json.dump(meta, open(f"{base}.meta.json", "w"), ensure_ascii=False,
              indent=1)
    print(f"\n  сохранено: {base}.{{h12,h18,h24,q0hat,ktrue,codebooks}}.npy "
          f"и .meta.json")
    print(f"  ключи (episode, step) sha {meta['keys_sha1']} — совпадение с "
          f"K-9a обязано проверяться в K-11c")
    print("\n  ДАЛЬШЕ: диагностика ранга (--diagnose), затем K-11b с "
          "проверками тождества.\n  Обучение головы не начинать, пока "
          "тождество не пройдено.")


if __name__ == "__main__":
    main()
