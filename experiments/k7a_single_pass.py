"""K-7a: настоящая однопроходная генерация грубого уровня, честная латентность.

ЗАЧЕМ. K-6h показал, что отказ от тонких уровней не стоит успеха: на LIBERO-10
при H=8 грубый уровень дал 89.0% против 88.0% у полной BAR с усреднением и
89.5% против 90.0% без него, односторонние границы -3.0 и -3.5 пункта. Но там
вызывался обычный generate со всеми тремя блоками, а уровни отбрасывались перед
декодированием: так мерился УСПЕХ, и намеренно не мерилось ВРЕМЯ.

Поэтому число «1.53x» пока ничем не обеспечено, кроме вычитания по разложению
k5a/k6c. Здесь оно измеряется.

ЧТО ДЕЛАЕТСЯ.
  1. Наследник SmolVLABlockwiseAR с generate_coarse: один блок вместо трёх.
  2. Проверка эквивалентности: первые 16 токенов обычного generate обязаны
     СОВПАСТЬ с выходом generate_coarse, и декодированные действия тоже.
  3. Явная статистика различия «уровень 1» и «уровни 1-3» — то, чего не хватало
     в K-6h. Проверка тождества там подтверждала лишь, что при трёх уровнях наша
     сборка равна официальному decode, и НЕ подтверждала, что при одном уровне
     действия отличаются. Здесь это измеряется прямо.
  4. Латентность на одинаковых входах с прогревом и синхронизацией.

ПОЧЕМУ НАСЛЕДНИК НЕ ПЕРЕПИСЫВАЕТ ЦИКЛ. generate устроен как
`for block_idx in range(self.num_blocks)`, причём блок 0 не зависит от
последующих (внимание причинно по блокам), а block_size хранится отдельным
полем и не пересчитывается. Поэтому достаточно ограничить число блоков на время
вызова — и исполняется буквально тот же код, включая построение позиций и
масок. Переписывание цикла добавило бы расхождение, которое потом пришлось бы
искать. Файлы third_party не изменяются.

ЧЕСТНОСТЬ СРАВНЕНИЯ С КЭШЕМ. Здесь меряются официальная BAR (три прохода, без
кэша) и generate_coarse (один проход). «BAR с кэшем префикса» в репозитории не
реализована — 1.31x из k5a получен зондом, а не рабочей реализацией. Поэтому
измеренным считается только отношение к официальной BAR, а пересчёт к кэшу
помечается в выводе как ВЗЯТЫЙ ИЗ k5a, а не измеренный здесь.

Запуск:
    python3 experiments/k7a_single_pass.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k7a_single_pass.py --ckpt <ckpt> \\
        --n-obs 1000 --out data/k7a_single_pass.json
"""

import argparse
import io
import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20
CACHE_FACTOR_K5A = 148.9 / 195.0      # НЕ измеряется здесь, см. докстроку


def _block(a, b, tag, out):
    """Один набор показателей для среза. Поза и схват отдельно: у них разные
    диапазоны, и общий RMS прятал бы схват."""
    d = a - b
    out[f"pose_rms{tag}"] = float(np.sqrt((d[..., :6] ** 2).mean()))
    out[f"pose_max{tag}"] = float(np.abs(d[..., :6]).max())
    out[f"grip_rms{tag}"] = float(np.sqrt((d[..., 6] ** 2).mean()))
    out[f"grip_sign_flip{tag}"] = float(
        (np.sign(a[..., 6]) != np.sign(b[..., 6])).mean())
    out[f"frac_steps_diff{tag}"] = float((np.abs(d).max(axis=-1) > 1e-6).mean())
    out[f"frac_rows_identical{tag}"] = float(
        (np.abs(d).reshape(len(d), -1).max(axis=1) <= 1e-6).mean())


def diff_stats(a, b, scale=None, horizons=(4, 8)):
    """Различие двух наборов действий, по всему чанку и по ИСПОЛНЯЕМЫМ позициям.

    Различать надо именно исполняемое. Чанк содержит 20 шагов, но при H=8
    среда видит только первые восемь, а остальные заменяются новым планом.
    Совпадение хвоста ничего не говорит о том, различались ли режимы в
    симуляторе; совпадение первых H — говорит.
    """
    out = {}
    _block(a, b, "", out)
    for H in horizons:
        _block(a[:, :H], b[:, :H], f"_first{H}", out)
    if scale is not None:
        ds = (a - b) * scale
        out["pose_rms_physical"] = float(np.sqrt((ds[..., :6] ** 2).mean()))
        out["pose_max_physical"] = float(np.abs(ds[..., :6]).max())
    return out


def summarize_times(ms):
    a = np.asarray(ms, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                sd=float(a.std(ddof=1)), p95=float(np.percentile(a, 95)),
                n=len(a))


def selftest():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, T_CHUNK, 7))

    # 1. Идентичные наборы: всё нулевое, доля одинаковых строк единица.
    s = diff_stats(A, A.copy())
    assert s["pose_rms"] == 0.0 and s["frac_rows_identical"] == 1.0, s

    # 2. Различие ОБЯЗАНО обнаруживаться, даже если оно в одном шаге из двадцати.
    B = A.copy(); B[3, 7, 2] += 0.5
    s = diff_stats(A, B)
    assert s["frac_rows_identical"] == 49 / 50, s["frac_rows_identical"]
    assert s["pose_max"] == 0.5 and s["frac_steps_diff"] == 1 / (50 * T_CHUNK)

    # 3. ИСПОЛНЯЕМЫЕ позиции считаются отдельно, и это не косметика: правка на
    #    шаге 7 при H=4 не исполняется вовсе. Если бы режимы различались ТОЛЬКО
    #    в хвосте чанка, симулятор их бы не различил.
    assert s["pose_rms_first4"] == 0.0 and s["frac_rows_identical_first4"] == 1.0
    assert s["pose_rms_first8"] > 0.0 and s["frac_rows_identical_first8"] < 1.0

    # 4. Схват не тонет в позе, и считается по срезам тоже.
    C = A.copy(); C[..., 6] = -A[..., 6]
    s = diff_stats(A, C)
    assert s["pose_rms"] == 0.0 and s["grip_rms"] > 0
    assert s["grip_sign_flip"] > 0.9 and s["grip_sign_flip_first4"] > 0.9
    assert s["grip_rms_first8"] > 0 and s["frac_steps_diff_first8"] > 0.9

    # 5. Хвост чанка не должен маскировать совпадение в исполняемой части:
    #    различие только на шагах 10-19 даёт нулевые показатели first8.
    D = A.copy(); D[:, 10:, :] += 1.0
    s = diff_stats(A, D)
    assert s["pose_rms"] > 0 and s["frac_rows_identical"] == 0.0
    assert s["frac_rows_identical_first8"] == 1.0, (
        "совпадение исполняемой части обязано быть видно отдельно")

    # 5. Сводка времён: медиана устойчива к выбросу, среднее нет.
    t = summarize_times([10.0] * 99 + [1000.0])
    assert t["median"] == 10.0 and t["mean"] > 19.0

    # Сообщение НАЗЫВАЕТ проверенное, а не просто говорит «пройдена». Иначе
    # старая и новая версии файла рапортуют одинаково, и устаревшую копию на
    # кластере по выводу не отличить — уже случилось однажды.
    print("самопроверка k7a пройдена: исполняемая часть чанка (first4/first8) "
          "считается отдельным набором показателей, схват не тонет в позе, "
          "различие в хвосте не маскирует совпадение в исполняемой части, "
          "медиана устойчива к выбросу")


@contextmanager
def only_blocks(model, n):
    """Ограничить число блоков на время вызова.

    block_size — отдельное поле, вычисленное в __init__ как
    token_budget // num_blocks, и от этой подмены не меняется. Маска и позиции
    строятся от фактических длин, а не от num_blocks (bar.py:840+). Поэтому
    подмена ограничивает ровно число проходов и ничего больше.
    """
    saved = model.num_blocks
    try:
        model.num_blocks = n
        yield
    finally:
        model.num_blocks = saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-obs", type=int, default=1000)
    ap.add_argument("--n-ep", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lat-batches", default="1,10",
                    help="батчи для замера латентности; 1 — как в статьях, "
                         "10 — как в нашем симуляторном стенде")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k7a_single_pass.json")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       process_state, prompt_template, seed_everything)

    class SmolVLASinglePassAR(SmolVLABlockwiseAR):
        """Один проход башни вместо трёх. Файлы third_party не меняются."""

        @torch.no_grad()
        def generate_coarse(self, **kw):
            with only_blocks(self, 1):
                out = super().generate(**kw)
            # generated выделяется на token_budget=48, а заполняется только
            # первый блок; остальное — неинициализированная память.
            return out[:, :self.block_size].clone()

    seed_everything(args.seed)
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    model = SmolVLASinglePassAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    assert model.num_blocks == N_LEVEL and model.block_size == N_POS, (
        f"ожидались {N_LEVEL} блока по {N_POS}, получено "
        f"{model.num_blocks}x{model.block_size}")

    off_by_task = None
    if args.pos_offset is None:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; задайте --pos-offset")
        tb = json.load(open(args.offset_table))
        off_by_task = {k: int(v["pos_offset"]) for k, v in tb["tasks"].items()}

    # --- данные: ДАТАСЕТНЫЙ путь предобработки ------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(1693)
    per_ep = max(1, args.n_obs // max(args.n_ep, 1))

    def png(cell):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))

    im1, im2, st, tsk = [], [], [], []
    for e in order:
        if len(tsk) >= args.n_obs:
            break
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        if t.num_rows < T_CHUNK + 1:
            continue
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        n_st = t.num_rows - T_CHUNK + 1
        for s0 in rng.choice(n_st, size=min(per_ep, n_st), replace=False):
            im1.append(png(c1[int(s0)])); im2.append(png(c2[int(s0)]))
            st.append(S_[int(s0)]); tsk.append(tasks_map[ti[int(s0)]])
    N = len(tsk)
    hw = im1[0].shape[0]
    tf = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])   # ДОЛЯ, не число
    print(f"собрано {N} наблюдений, кадр {hw}x{hw}")

    ST = np.asarray(st, np.float64)
    if ST.shape[1] == len(STATE_Q01) + 1:
        ST = process_state(ST)
    st_n = (ST - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    offs = np.array([args.pos_offset if args.pos_offset is not None
                     else off_by_task[tsk[i]] for i in range(N)])

    def build_batch(sel):
        i1 = tf(torch.tensor(np.stack([im1[j] for j in sel])).permute(0, 3, 1, 2))
        i2 = tf(torch.tensor(np.stack([im2[j] for j in sel])).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for gi in sel:
            m = prompt_template(
                st_n[gi], None, tsk[gi],
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[k].numpy()] for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dtype), b)

    def decode(codes):
        d = proc.action_processor.decode(np.asarray(codes).reshape(len(codes), -1).tolist())
        return np.asarray(d if isinstance(d, np.ndarray) else d[0], np.float64)

    # --- 1. эквивалентность ---------------------------------------------------
    print("\n  проверка эквивалентности одного прохода и первых 16 токенов...")
    tok_full, tok_coarse = [], []
    for po in sorted({int(v) for v in offs}):
        idx_po = np.where(offs == po)[0]
        for i0 in range(0, len(idx_po), args.batch):
            sel = idx_po[i0:i0 + args.batch]
            batch = build_batch(sel)
            with torch.no_grad():
                tf_ = model.generate(**batch, position_offset=po,
                                     do_sample=False, initial_position_shift=1)
                tc_ = model.generate_coarse(**batch, position_offset=po,
                                            do_sample=False,
                                            initial_position_shift=1)
            tok_full.append(tf_.cpu().numpy())
            tok_coarse.append(tc_.cpu().numpy())
    K_full = np.concatenate(tok_full).reshape(N, N_LEVEL, N_POS)
    K_coarse = np.concatenate(tok_coarse).reshape(N, N_POS)

    same = (K_full[:, 0, :] == K_coarse)
    print(f"    совпадение грубых токенов: {same.mean():.6%} "
          f"({int(same.sum())} из {same.size})")
    if not same.all():
        raise SystemExit(
            "грубые токены одного прохода НЕ совпали с первыми 16 токенами\n"
            "полного generate. Либо блок 0 всё-таки зависит от последующих,\n"
            "либо подмена num_blocks меняет что-то ещё. Дальше идти нельзя.")

    # --- 2. насколько уровень 1 ОТЛИЧАЕТСЯ от трёх уровней --------------------
    # Пробел K-6h: там проверялось лишь, что при трёх уровнях сборка равна
    # официальному decode. Что при одном уровне действия ДРУГИЕ — не проверялось.
    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    codec = codec.to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    def decode_levels(K, n_lv):
        outs = []
        for i0 in range(0, len(K), 256):
            k = torch.as_tensor(K[i0:i0 + 256]).long().to(dev)
            with torch.no_grad():
                z = E[0][k[:, 0, :]]
                for j in range(1, n_lv):
                    z = z + E[j][k[:, j, :]]
                x, _ = codec._decode(z, embodiment_ids=0)
            outs.append(x[..., :7].float().cpu().numpy())
        return np.concatenate(outs)

    a1 = decode_levels(K_full, 1)
    a3 = decode_levels(K_full, 3)
    ref3 = decode(K_full)
    d_ident = float(np.abs(a3 - ref3).max())
    print(f"    тождество сборки при трёх уровнях: max|Δ| = {d_ident:.3e}")
    if d_ident > 1e-3:
        raise SystemExit("своя сборка расходится с официальным decode")

    scale = np.ones(7); scale[:6] = max_act_q[:6]
    st13 = diff_stats(a1, a3, scale=scale)
    print("\n    различие «уровень 1» и «уровни 1-3»:")
    for k_ in sorted(st13):
        print(f"      {k_:<30}{st13[k_]:.6f}")

    # ОТКАЗ ФОРМУЛИРУЕТСЯ ПО ИСПОЛНЯЕМЫМ ПОЗИЦИЯМ. Доля совпавших чанков сама
    # по себе ничего не решает: если тонкие уровни меняют лишь часть состояний,
    # K-6h остаётся в силе — он и мерил успех, а не долю различий. Артефактом
    # результат был бы только в одном случае: если режимы почти не различаются
    # ИМЕННО в том, что доходит до среды.
    fi8 = st13["frac_rows_identical_first8"]
    if fi8 > 0.99 and st13["pose_rms_first8"] < 1e-6:
        raise SystemExit(
            f"{fi8:.1%} исполняемых префиксов (первые 8 шагов) совпадают при\n"
            f"одном и трёх уровнях, ошибка позы там {st13['pose_rms_first8']:.2e}.\n"
            f"Тогда K-6h сравнивал режимы, которые среда почти не различает,\n"
            f"и его вывод об успехе ничего не говорит о тонких уровнях.")
    print(f"\n    исполняемая часть (первые 8 шагов) различается в "
          f"{1 - fi8:.1%} наблюдений")

    # --- 3. латентность -------------------------------------------------------
    print("\n  латентность (прогрев "
          f"{args.warmup}, замеров {args.iters}, синхронизация вокруг)")
    lat = {}
    for bs in [int(v) for v in args.lat_batches.split(",")]:
        sel = np.where(offs == offs[0])[0][:bs]
        if len(sel) < bs:
            print(f"    батч {bs}: пропуск, мало наблюдений с одним офсетом")
            continue
        batch = build_batch(sel)
        po = int(offs[sel[0]])
        fns = {
            "bar_3_blocks": lambda: model.generate(
                **batch, position_offset=po, do_sample=False,
                initial_position_shift=1),
            "coarse_1_block": lambda: model.generate_coarse(
                **batch, position_offset=po, do_sample=False,
                initial_position_shift=1),
        }
        with torch.no_grad():
            for _ in range(args.warmup):
                for fn in fns.values():
                    fn()
            torch.cuda.synchronize()

        # ЧЕРЕДОВАНИЕ ПОРЯДКА. Замеры подряд в фиксированном порядке смешивают
        # разницу режимов с дрейфом частоты и температуры GPU: второй всегда
        # мерится на прогретой карте. Чередуем и считаем ускорение ПО РАУНДАМ,
        # тогда дрейф попадает в разброс раундов, а не в саму оценку.
        per_round, samples = [], {k: [] for k in fns}
        n_rounds, per_iter = 4, max(1, args.iters // 4)
        torch.cuda.reset_peak_memory_stats(dev)
        for rd in range(n_rounds):
            order_ = list(fns) if rd % 2 == 0 else list(fns)[::-1]
            med = {}
            with torch.no_grad():
                for name in order_:
                    ts = []
                    for _ in range(per_iter):
                        t0 = time.perf_counter()
                        fns[name]()
                        torch.cuda.synchronize()
                        ts.append((time.perf_counter() - t0) * 1e3)
                    samples[name] += ts
                    med[name] = float(np.median(ts))
            per_round.append(med["bar_3_blocks"] / med["coarse_1_block"])
        peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30

        for name in fns:
            s = summarize_times(samples[name])
            lat[f"b{bs}_{name}"] = s
            print(f"    батч {bs:>2} {name:<15} "
                  f"med {s['median']:7.2f} мс  ср {s['mean']:7.2f}  "
                  f"sd {s['sd']:5.2f}  p95 {s['p95']:7.2f}")
        r = float(np.median(per_round))
        lat[f"b{bs}_speedup_vs_official"] = r
        lat[f"b{bs}_speedup_per_round"] = per_round
        lat[f"b{bs}_peak_mem_gib"] = peak
        print(f"    батч {bs:>2}: ИЗМЕРЕНО {r:.2f}x против официальной BAR "
              f"(по раундам: {', '.join(f'{x:.2f}' for x in per_round)})")
        print(f"              пик памяти {peak:.2f} ГиБ; пересчёт к BAR с кэшем "
              f"~{r * CACHE_FACTOR_K5A:.2f}x — НЕ измерено здесь, из k5a")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(dict(n_obs=N, ckpt=args.ckpt, coarse_tokens_match=bool(same.all()),
                   identity_max_abs=d_ident, diff_l1_vs_l3=st13, latency=lat,
                   warmup=args.warmup, iters=args.iters,
                   note="cached-BAR не реализована; множитель кэша взят из k5a"),
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
