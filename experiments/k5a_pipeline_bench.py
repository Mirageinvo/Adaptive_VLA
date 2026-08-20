"""K-5a2: разложение полного вызова политики по времени.

ВОПРОС. Все наши заявки про ускорение упираются в одну неизмеренную величину —
долю той части, которую мы оптимизируем, в полном вызове политики. §7б мерил
сквозной вызов один раз (273.3 мс), но модель тогда фактически грузилась во
float32: `from_pretrained` игнорирует запрошенный dtype, и в логе builder стоит
«dtype модели по факту: torch.float32 (запрошен bfloat16)». Значит все прежние
абсолютные числа относятся к float32, а не к режиму развёртывания.

ЧТО МЕРЯЕТСЯ. Полный путь «наблюдение -> чанк действий», разложенный на части:

  процессор      — подготовка входа, CPU: кроп, шаблон промпта, токенизация;
  префикс VLM    — _build_vlm_inputs_embeds, включая кодировщик зрения;
  блоки 1..3     — три блочных прохода BAR;
  декод кодека   — ActionCodec из кодов в непрерывные действия;
  весь вызов     — измеряется ЦЕЛИКОМ, а не суммой медиан: складывать медианы
                   разных распределений некорректно (§7б наступал на это).

ЗАЧЕМ ИМЕННО ТАК. Из разложения считается f — доля блочной части — и потолок
любой оптимизации внутри неё: при ускорении блоков в S раз сквозное ускорение
не превысит 1/(1 - f*(1 - 1/S)). Это число решает, какую формулировку вклада мы
вообще имеем право заявлять, и его надо знать ДО вложений в метод.

ДАННЫЕ НЕ НУЖНЫ. Время определяется формами, а не содержанием: картинки
синтезируются случайным uint8 нужной формы и проходят через НАСТОЯЩИЙ
процессор, поэтому длина префикса получается подлинной, а не предполагаемой.
Проверить это отдельно важно: §7б исходил из 168 токенов.

РАЗБРОС. На этой машине опорные времена гуляют на 20-25% между прогонами
(поймано в K-5a). Поэтому режимы меряются сериями с ПЕРЕМЕШАННЫМ порядком, а
доли считаются посерийно: общий дрейф сокращается в отношении.

Запуск:
    python3 experiments/k5a_pipeline_bench.py --selftest
    python3 experiments/k5a_pipeline_bench.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
        --k5a data/k5a.json --out data/k5a_pipeline.json
"""

import argparse
import json
import os
import random
import statistics
import sys
import time


def ceiling(f, S):
    """Потолок сквозного ускорения при ускорении доли f в S раз (Амдал)."""
    return 1.0 / (1.0 - f * (1.0 - 1.0 / S))


def selftest():
    """Формула потолка против ПРЯМОГО отношения времён, с известным ответом.

    Пусть часть C не ускоряется, часть D ускоряется в S раз. Тогда сквозное
    ускорение равно (C+D)/(C+D/S), и формула Амдала обязана дать то же. Это
    единственное содержательное утверждение скрипта, всё остальное — замер.
    """
    for C, D, S in ((100.0, 141.0, 1.383), (250.0, 20.0, 4.0),
                    (5.0, 200.0, 2.0), (0.0, 50.0, 3.0)):
        f = D / (C + D)
        direct = (C + D) / (C + D / S)
        assert abs(ceiling(f, S) - direct) < 1e-9, \
            f"потолок разошёлся с прямым отношением: {C=} {D=} {S=}"
    # предельные случаи: ускорять нечего либо ускорение отсутствует
    assert abs(ceiling(0.0, 10.0) - 1.0) < 1e-12, "при f=0 потолок обязан быть 1"
    assert abs(ceiling(0.7, 1.0) - 1.0) < 1e-12, "при S=1 потолок обязан быть 1"
    assert abs(ceiling(1.0, 2.0) - 2.0) < 1e-12, "при f=1 потолок равен S"
    print("самопроверка пройдена: формула потолка совпадает с прямым "
          "отношением времён, предельные случаи верны")


def bench(fn, warmup, iters, dev, torch, cpu_timer=False):
    """cpu_timer=True для чисто CPU-шагов.

    События CUDA меряют GPU-таймлайн, поэтому работа процессора — кроп,
    шаблон промпта, токенизация — на них выглядела бы почти нулевой. Для неё
    нужен настенный таймер, и это НЕ придирка: в развёрнутой политике
    подготовка входа выполняется на каждом вызове и входит в его стоимость.
    """
    for _ in range(warmup):
        fn()
    ts = []
    if dev.type == "cuda" and not cpu_timer:
        torch.cuda.synchronize()
        for _ in range(iters):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(sorted(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--tokenizer", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--pos-offset", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--k5a", default=None,
                    help="JSON из k5a_prefix_cost_probe: подставит измеренное "
                         "ускорение блочной части в оценку сквозного")
    ap.add_argument("--center-crop", type=int, default=1)
    ap.add_argument("--tiled", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    import copy
    import importlib.util

    import numpy as np
    import torch

    sys.path.insert(0, os.path.abspath(args.root))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import actioncodec  # noqa: F401,E402
    from k1_residual_cost import latent_from_codes, projected_codebooks  # noqa: E402
    from k3_bar_suffix_repair import build_batch  # noqa: E402
    from smolvla.bar import SmolVLABlockwiseAR  # noqa: E402

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    if dev.type == "cuda":
        cc = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]}), "
              f"torch {torch.__version__}, CUDA {torch.version.cuda}")
        if args.dtype == "bfloat16" and cc[0] < 8:
            print("  ВНИМАНИЕ: bfloat16 без аппаратной поддержки на этом GPU")

    sp = importlib.util.spec_from_file_location(
        "ac_vla_tok", os.path.join(os.path.abspath(args.root), "utils",
                                   "vla_tokenizer.py"))
    mm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mm)
    proc = mm.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tok = proc.action_processor.to(dev).eval()
    tok32 = copy.deepcopy(tok).float().eval()
    L, P = tok.num_quantizers, tok.n_tokens_per_quantizer
    cfg = tok.config.embodiment_config[
        list(tok.config.embodiment_config.keys())[args.embodiment]]
    T_act, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]
    E = projected_codebooks(tok32, dev)

    model = SmolVLABlockwiseAR.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=dtype,
        token_budget=P * L, num_blocks=L,
        action_vocab_size=tok.vocab_size).to(dev).eval()
    got = next(model.parameters()).dtype
    if got != dtype:
        print(f"  from_pretrained вернул {got} вместо {dtype} — привожу "
              f"принудительно (то же было в builder)")
        model = model.to(dtype)
    print(f"  dtype модели по факту: {next(model.parameters()).dtype}")
    n, nb = model.block_size, model.num_blocks
    print(f"блоков {nb} по {n} токенов, позиций {P}, уровней {L}, "
          f"действий в чанке {T_act}x{D_act}, batch {args.batch}")

    # ---- синтетическое наблюдение через НАСТОЯЩИЙ процессор ---------------
    B = args.batch
    rng = np.random.default_rng(0)
    im1 = torch.as_tensor(rng.integers(0, 256, (B, 3, 256, 256), dtype=np.uint8))
    im2 = torch.as_tensor(rng.integers(0, 256, (B, 3, 256, 256), dtype=np.uint8))
    state = rng.uniform(-1, 1, (B, 8)).astype(np.float32)
    tasks = ["pick up the black bowl and place it on the plate"] * B

    class NS:
        center_crop = bool(args.center_crop)
        tiled = bool(args.tiled)
        source = "lerobot"
        flip = ""

    def make_batch():
        return build_batch(im1, im2, tasks, state, proc, NS(), dev,
                           pad_to=None, pad_side="left")

    batch = make_batch()
    with torch.no_grad():
        _, vlen, VLM, _ = model._build_vlm_inputs_embeds(
            input_ids=batch["input_ids"], inputs_embeds=None,
            pixel_values=batch.get("pixel_values"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            image_hidden_states=None)
    print(f"  ДЛИНА ПРЕФИКСА ПО ФАКТУ: {vlen} токенов "
          f"(в §7б предполагалось 168)")

    amask = batch.get("attention_mask")

    def pos_ids(alen):
        apos = model._build_action_pos_ids_strided(
            batch_size=B, base_pos=vlen, action_seq_len=alen, device=dev,
            position_offset=args.pos_offset)
        return model._build_joint_position_ids(
            batch_size=B, vlm_seq_len=vlen, action_pos_ids=apos, device=dev)

    def blocks(vlm):
        """Три блочных прохода: ровно как в builder."""
        hist = None
        for _ in range(nb):
            lg = model._predict_next_block_logits(
                vlm_inputs_embeds=vlm, attention_mask=amask,
                history_tokens=hist,
                position_ids=pos_ids(n + (0 if hist is None
                                          else hist.shape[1])))
            nxt = lg.argmax(-1)
            hist = nxt if hist is None else torch.cat([hist, nxt], 1)
        return hist

    def decode(hist):
        z = hist.reshape(-1, L, P).transpose(1, 2)
        return tok32._decode(latent_from_codes(E, z), args.embodiment,
                             None)[0][..., :D_act]

    with torch.no_grad():
        hist0 = blocks(VLM)

    def one_block(k):
        h = None if k == 0 else hist0[:, :k * n]
        pid = pos_ids(n + (0 if h is None else h.shape[1]))

        def f():
            with torch.no_grad():
                model._predict_next_block_logits(
                    vlm_inputs_embeds=VLM, attention_mask=amask,
                    history_tokens=h, position_ids=pid)
        return f

    def full_call():
        """Весь путь целиком: наблюдение -> непрерывные действия."""
        with torch.no_grad():
            b = make_batch()
            _, vl, V, _ = model._build_vlm_inputs_embeds(
                input_ids=b["input_ids"], inputs_embeds=None,
                pixel_values=b.get("pixel_values"),
                pixel_attention_mask=b.get("pixel_attention_mask"),
                image_hidden_states=None)
            assert vl == vlen, f"длина префикса поплыла: {vl} против {vlen}"
            return decode(blocks(V))

    def prefix_only():
        with torch.no_grad():
            model._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"], inputs_embeds=None,
                pixel_values=batch.get("pixel_values"),
                pixel_attention_mask=batch.get("pixel_attention_mask"),
                image_hidden_states=None)

    tasks_t = {"процессор (CPU)": lambda: make_batch(),
               "префикс VLM": prefix_only,
               "блок 1": one_block(0), "блок 2": one_block(1),
               "блок 3": one_block(2),
               "декод кодека": lambda: decode(hist0),
               "ВЕСЬ ВЫЗОВ": full_call}

    print("\n" + "=" * 74)
    print(f"ЗАМЕРЫ: {args.repeats} серий, порядок перемешан")
    print("=" * 74)
    r = random.Random(args.order_seed)
    series = []
    for s_ in range(args.repeats):
        names = list(tasks_t)
        r.shuffle(names)
        row = {nm: bench(tasks_t[nm], args.warmup,
                         max(10, args.iters // args.repeats), dev, torch,
                         cpu_timer=nm.endswith("(CPU)"))
               for nm in names}
        row["блоки 1-3"] = row["блок 1"] + row["блок 2"] + row["блок 3"]
        row["f_блоки"] = row["блоки 1-3"] / row["ВЕСЬ ВЫЗОВ"]
        series.append(row)
        print(f"  серия {s_}: весь вызов {row['ВЕСЬ ВЫЗОВ']:7.2f} мс, "
              f"блоки {row['блоки 1-3']:7.2f} мс, доля {row['f_блоки']:.3f}")

    med = lambda k: statistics.median(x[k] for x in series)
    tot = med("ВЕСЬ ВЫЗОВ")
    print(f"\n  {'часть':<20}{'медиана, мс':>14}{'доля вызова':>14}"
          f"{'разброс серий':>16}")
    for nm in ("процессор (CPU)", "префикс VLM", "блок 1", "блок 2", "блок 3",
               "блоки 1-3", "декод кодека", "ВЕСЬ ВЫЗОВ"):
        v = [x[nm] for x in series]
        print(f"  {nm:<20}{med(nm):>14.2f}{med(nm) / tot:>13.1%}"
              f"{max(v) - min(v):>16.2f}")
    ssum = med("процессор (CPU)") + med("префикс VLM") + med("блоки 1-3") \
        + med("декод кодека")
    print(f"\n  сумма частей {ssum:.2f} против измеренного целого {tot:.2f} мс "
          f"(расхождение {abs(ssum - tot) / tot:.1%})")
    print("  Складывать медианы разных распределений некорректно, поэтому "
          "целое\n  измеряется отдельно; расхождение показывает, насколько "
          "разложение полно.")

    f_blocks = med("f_блоки")
    print("\n" + "=" * 74)
    print("ПОТОЛОК ЛЮБОЙ ОПТИМИЗАЦИИ БЛОЧНОЙ ЧАСТИ")
    print("=" * 74)
    print(f"  доля блочной части f = {f_blocks:.3f}")
    print(f"  {'ускорение блоков S':>20}{'потолок сквозного':>20}")
    for S in (1.383, 1.5, 2.0, 3.0, 1e9):
        lbl = "бесконечное" if S > 1e8 else f"{S:.3f}x"
        print(f"  {lbl:>20}{ceiling(f_blocks, S):>19.3f}x")
    res = dict(vlen=int(vlen), f_blocks=f_blocks, total_ms=tot,
               parts={k: med(k) for k in tasks_t}, series=series)
    if args.k5a and os.path.exists(args.k5a):
        k5 = json.load(open(args.k5a))
        s_meas = k5["total_ref"] / k5["total_cached"]
        e2e = ceiling(f_blocks, s_meas)
        print(f"\n  ИЗМЕРЕННОЕ ускорение блоков из {args.k5a}: {s_meas:.3f}x")
        print(f"  сквозное от кэша префикса: {e2e:.3f}x")
        print("  Прежняя оценка 1.84x относилась к БЛОЧНОЙ части и была "
              "завышена;\n  сквозная величина меньше ещё и на долю зрения, "
              "процессора и кодека.")
        res |= dict(k5a_block_speedup=s_meas, e2e_from_cache=e2e)
    print("\n  ЧИТАТЬ ТАК: если потолок при БЕСКОНЕЧНОМ ускорении блоков ниже\n"
          "  1.15x, то никакая работа внутри блочной части не даёт заявки на\n"
          "  сквозное ускорение, и оптимизировать надо число ВЫЗОВОВ, а не\n"
          "  стоимость одного.")

    if args.out:
        import hashlib
        import subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             text=True).strip()
        except Exception:
            commit = "unknown"
        res |= dict(commit=commit, ckpt=args.ckpt, dtype=args.dtype,
                    gpu=(torch.cuda.get_device_name(0)
                         if dev.type == "cuda" else None),
                    torch=torch.__version__, batch=B, repeats=args.repeats,
                    pos_offset=args.pos_offset,
                    self_sha256=hashlib.sha256(
                        open(__file__, "rb").read()).hexdigest()[:16])
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
