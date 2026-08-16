"""K-4a, фаза A: даёт ли сокращение активных позиций ВЫИГРЫШ ПО ВРЕМЕНИ.

ЗАЧЕМ ЭТО РАНЬШЕ ФАЗЫ E. План ставит воротам E требование 1.25x на голове и
1.15x end-to-end. Но при batch 1 проход по декодеру упирается не в
арифметику, а в чтение весов из памяти: 2.2 млрд параметров грузятся целиком
независимо от того, шестнадцать позиций в запросе или четыре. Тогда
разреженность по токенам не даёт wall-clock вообще, и экономить можно только
числом проходов (NFE). Это kill-критерий 5 всей линии, и он проверяется за
двадцать минут, а не после реализации разреженного графа.

ЧТО МЕРЯЕМ. У BAR метод _run_action_sequence принимает bos_len — число
позиций действия в запросе. Меняем его при неизменном префиксе VLM и получаем
ровно ту величину, которая нужна: стоимость плотного прохода как функция числа
активных позиций. Их код при этом не трогаем, только вызываем.

ВАЖНО ПРО РЕАЛИЗАЦИЮ. В выложенном BAR кэша KV нет: каждый вызов блока заново
прогоняет весь префикс VLM. Поэтому доля времени, приходящаяся на позиции
действия, тем более мала. Отчёт даёт обе величины отдельно.

Модель t(q) = a + b*q:
  a — постоянная часть (префикс, загрузка весов, накладные расходы);
  b — предельная цена одной активной позиции.
Максимально возможное ускорение от перехода 16 -> K:
  (t(16) - t(K)) / t(16) = b*(16-K) / (a + 16b).
Если b*12 составляет единицы процентов от t(16), ворота E недостижимы в
принципе, и вклад надо формулировать как сокращение вычислений, а не как
ускорение реального времени.

Запуск:
    python3 experiments/k4a3_latency_probe.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3_bar_suffix_repair import (  # noqa: E402
    MAX_ACTION_Q,
    STATE_Q01,
    STATE_Q99,
    build_batch,
    load_lerobot,
)

QS = (1, 2, 4, 8, 16, 32, 48)


def timeit(fn, warmup: int, iters: int):
    """Медиана, p90, p99 в миллисекундах. Синхронизация вокруг каждого прогона:
    иначе замеряется постановка в очередь, а не исполнение."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return float(np.median(a)), float(np.percentile(a, 90)), float(np.percentile(a, 99))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pos-offset", type=int, default=3)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--tiled", action="store_true", default=True)
    ap.add_argument("--source", default="lerobot")
    ap.add_argument("--flip", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.root))
    import importlib.util

    import actioncodec  # noqa: F401

    sp = importlib.util.spec_from_file_location(
        "ac_vla_tok", os.path.join(os.path.abspath(args.root), "utils",
                                   "vla_tokenizer.py"))
    mm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mm)
    proc = mm.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tok = proc.action_processor.to(args.device).eval()
    L, P = tok.num_quantizers, tok.n_tokens_per_quantizer
    cfg = tok.config.embodiment_config[
        list(tok.config.embodiment_config.keys())[args.embodiment]]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]

    bmax = max(int(x) for x in args.batches.split(","))
    IM1, IM2, ST_RAW, A, PREV, tasks, EPI = load_lerobot(bmax, T, n_ep=bmax)
    st = ((ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0).astype(np.float32)

    from smolvla.bar import SmolVLABlockwiseAR

    model = SmolVLABlockwiseAR.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
        token_budget=P * L, num_blocks=L,
        action_vocab_size=tok.vocab_size).to(args.device).eval()

    n_par = sum(p.numel() for p in model.parameters())
    print(f"параметров модели: {n_par / 1e9:.2f} млрд, "
          f"уровней {L}, позиций {P}, блок {model.block_size}\n")
    print(f"устройство: {torch.cuda.get_device_name(0)}, "
          f"dtype bfloat16, прогрев {args.warmup}, замеров {args.iters}\n")

    for Bs in [int(x) for x in args.batches.split(",")]:
        batch = build_batch(IM1[:Bs], IM2[:Bs], tasks[:Bs], st[:Bs], proc,
                            args, args.device)

        def prefix():
            return model._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"], inputs_embeds=None,
                pixel_values=batch.get("pixel_values"),
                pixel_attention_mask=batch.get("pixel_attention_mask"),
                image_hidden_states=None)

        with torch.no_grad():
            _, vlen, VLM, _ = prefix()

        def run_q(q):
            apos = model._build_action_pos_ids_strided(
                batch_size=Bs, base_pos=vlen, action_seq_len=q,
                device=VLM.device, position_offset=args.pos_offset)
            pids = model._build_joint_position_ids(
                batch_size=Bs, vlm_seq_len=vlen, action_pos_ids=apos,
                device=VLM.device)
            return model._run_action_sequence(
                vlm_inputs_embeds=VLM, attention_mask=batch.get("attention_mask"),
                bos_len=q,
                action_input_ids=torch.empty((Bs, 0), device=args.device,
                                             dtype=torch.long),
                position_ids=pids)

        print("=" * 74)
        print(f"BATCH {Bs}   длина префикса VLM {vlen} токенов")
        print("=" * 74)
        with torch.no_grad():
            m, p90, p99 = timeit(lambda: prefix(), args.warmup // 2, args.iters // 2)
            print(f"{'префикс VLM (зрение+эмбеддинги)':>40}"
                  f"{m:>10.2f}{p90:>9.2f}{p99:>9.2f}  мс")
            print(f"\n{'активных позиций q':>40}{'медиана':>10}{'p90':>9}{'p99':>9}")
            med = {}
            for q in QS:
                m, p90, p99 = timeit(lambda q=q: run_q(q), args.warmup, args.iters)
                med[q] = m
                print(f"{q:>40}{m:>10.2f}{p90:>9.2f}{p99:>9.2f}")

            # полная генерация: три блока по block_size
            def full():
                hist = None
                for _ in range(model.num_blocks):
                    apos = model._build_action_pos_ids_strided(
                        batch_size=Bs, base_pos=vlen,
                        action_seq_len=model.block_size
                        + (0 if hist is None else hist.shape[1]),
                        device=VLM.device, position_offset=args.pos_offset)
                    pids = model._build_joint_position_ids(
                        batch_size=Bs, vlm_seq_len=vlen, action_pos_ids=apos,
                        device=VLM.device)
                    c = model._predict_next_block_logits(
                        vlm_inputs_embeds=VLM,
                        attention_mask=batch.get("attention_mask"),
                        history_tokens=hist, position_ids=pids).argmax(-1)
                    hist = c if hist is None else torch.cat([hist, c], 1)
                return hist

            m, p90, p99 = timeit(full, args.warmup // 5, args.iters // 5)
            print(f"\n{'полная генерация, 3 блока':>40}"
                  f"{m:>10.2f}{p90:>9.2f}{p99:>9.2f}  мс")

        qs = np.array(QS, float)
        ys = np.array([med[q] for q in QS])
        b, a = np.polyfit(qs, ys, 1)
        print(f"\n  линейная модель t(q) = {a:.2f} + {b:.4f}*q мс")
        print(f"  предельная цена одной активной позиции: {b:.4f} мс")
        print(f"  t(16) = {a + 16 * b:.2f} мс, из них на позиции "
              f"{16 * b:.2f} мс ({16 * b / (a + 16 * b):.1%})")
        for K in (1, 2, 4, 8):
            sp_ = (a + 16 * b) / max(a + K * b, 1e-9)
            print(f"  максимум ускорения одного прохода при 16 -> {K:>2}: "
                  f"{sp_:.3f}x  (экономия {(16 - K) * b:.2f} мс)")
        print("""
ЧИТАТЬ ТАК, правило зафиксировано до запуска.
  доля времени на позиции действия < 10% -> разреженность по токенам НЕ даёт
      ускорения при batch 1; ворота E недостижимы, вклад формулировать как
      сокращение вычислений, а экономию искать в числе проходов;
  10-30% -> ускорение возможно только вместе с кэшем префикса;
  > 30% -> линия про wall-clock жизнеспособна.
ОГОВОРКА. В выложенном BAR кэша KV нет, поэтому постоянная часть a включает
полный пересчёт префикса. С идеальным кэшем a уменьшится, и доля позиций
вырастет. Верхняя оценка выигрыша от кэша даётся строкой префикса выше:
вычесть её из a нельзя напрямую, но порядок она задаёт.""")


if __name__ == "__main__":
    main()
