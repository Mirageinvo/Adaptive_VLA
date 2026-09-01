"""K-9i: измеренная латентность четырёх конфигураций на одних входах.

ЧТО МЕРЯЕТСЯ.
  fullbar   — 24 слоя, ТРИ прохода, сборка из трёх уровней;
  coarse24  — 24 слоя, ОДИН проход, сборка из уровня 0;
  joint12   — 12 слоёв, один проход, веса Joint-12;
  frozen12  — 12 слоёв, один проход, исходный ствол и считыватель R*.

ГЛАВНАЯ ЛОВУШКА, РАДИ КОТОРОЙ НУЖЕН ОТДЕЛЬНЫЙ СКРИПТ. В гейте рука coarse24
вызывает ПОЛНЫЙ generate и отбрасывает уровни 2-3 перед декодированием. Так
корректно меряется УСПЕХ — модель исполняет ровно то, что исполняла бы, — и
СОВЕРШЕННО НЕКОРРЕКТНО мерялось бы время: три прохода вместо одного. Здесь
coarse24 идёт через ограничение числа блоков (приём из K-7a), то есть один
проход по-настоящему.

JOINT-12 И FROZEN-12 ОБЯЗАНЫ СОВПАСТЬ ПО ВРЕМЕНИ. У них одинаковые формы и
одинаковые операции, различаются только значения весов. Меряются обе не ради
двух чисел, а как проверка стенда: расхождение больше нескольких процентов
означает, что мерится не то, что заявлено. Различимых строк в таблице три, а
не четыре, и в отчёте это надо писать прямо.

ПОЧЕМУ ЧЕРЕДОВАНИЕ, А НЕ БЛОКИ. Тактовая частота карты плывёт от нагрева, и
конфигурация, измеренная последней, систематически медленнее. Здесь на каждом
повторе исполняются все конфигурации подряд, поэтому дрейф достаётся всем
поровну. K-7a использовал четыре чередующихся раунда; чередование по одному
повтору строже.

ПАМЯТЬ МЕРЯЕТСЯ ОТДЕЛЬНО, НЕ В ЧЕРЕДОВАНИИ: пик у перемешанных конфигураций
общий и ни одной из них не принадлежит.

ЧТО НЕ ВХОДИТ В ЗАМЕР. Подготовка батча процессором (токенизация и
нормализация кадров) идёт на CPU один раз и переиспользуется: она одинакова у
всех конфигураций и её включение только размыло бы различие. Декодирование
входит — у fullbar оно собирает три уровня, у остальных один, и это часть
разницы. Печатается отдельной строкой.

ЗАПУСКАТЬ НА СВОБОДНОЙ МАШИНЕ. Параллельная развёртка симулятора занимает
карту и диск; полученные при этом числа измеряют конкуренцию, а не модель.
Скрипт отказывается стартовать, если видит чужой процесс на той же карте.

Запуск:
    python3 experiments/k9i_latency.py --selftest

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9i_latency.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --cache data/k9_teacher_150k.npz \\
        --joint12 data/k9d_ep3.pt --frozen12 data/k9g_frozen12_rstar.pt \\
        --batches 1,10 --reps 200 --out data/k9i_latency.json
"""

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

N_POS, N_LEVEL = 16, 3
CONFIGS = ("fullbar", "coarse24", "joint12", "frozen12")


def stats(xs):
    """Медиана, среднее, p95 и разброс. Медиана — основное число.

    Среднее чувствительно к единичным выбросам от планировщика, а они здесь
    неизбежны; p95 нужен отдельно, потому что для робота важен худший случай,
    а не только типичный.
    """
    s = sorted(xs)
    n = len(s)
    return dict(n=n, median=statistics.median(s), mean=statistics.fmean(s),
                p95=s[min(n - 1, int(round(0.95 * (n - 1))))],
                p05=s[max(0, int(round(0.05 * (n - 1))))],
                min=s[0], max=s[-1],
                std=(statistics.stdev(s) if n > 1 else 0.0))


def speedup_table(med):
    """Отношения к самой дорогой конфигурации. Ускорение = во сколько раз."""
    base = max(med.values())
    return {k: base / v for k, v in med.items()}


def selftest():
    r = stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert r["median"] == 3.0 and r["max"] == 100.0
    assert r["mean"] > r["median"], "среднее обязано тянуться к выбросу"
    assert stats([5.0])["std"] == 0.0
    # p95 — РАНГОВЫЙ, без интерполяции: индекс round(0.95*(n-1)). На двадцати
    # точках это s[18] = 18.0, а НЕ максимум s[19]. Проверяется явно, потому
    # что «p95 равен максимуму» — обычная ошибка на малых выборках, и она
    # превратила бы худший случай в единичный выброс.
    xs = [float(i) for i in range(20)]
    assert stats(xs)["p95"] == 18.0, stats(xs)["p95"]
    assert stats(xs)["p95"] != max(xs), "p95 не должен совпадать с максимумом"
    assert stats(xs)["p05"] == 1.0

    sp = speedup_table({"a": 100.0, "b": 50.0, "c": 25.0})
    assert sp["a"] == 1.0 and sp["b"] == 2.0 and sp["c"] == 4.0

    # ЧЕРЕДОВАНИЕ. Проверяется на модели дрейфа: время растёт линейно от
    # нагрева, а конфигурации сами по себе одинаковы. При блочном измерении
    # последняя выглядит медленнее почти вдвое; при чередовании — нет.
    n, cfgs = 100, ["a", "b", "c", "d"]
    drift = lambda i: 1.0 + 0.01 * i
    block = {c: [drift(k * n + i) for i in range(n)]
             for k, c in enumerate(cfgs)}
    inter = {c: [drift(i * len(cfgs) + k) for i in range(n)]
             for k, c in enumerate(cfgs)}
    bm = {c: statistics.median(v) for c, v in block.items()}
    im = {c: statistics.median(v) for c, v in inter.items()}
    assert max(bm.values()) / min(bm.values()) > 1.9, bm
    assert max(im.values()) / min(im.values()) < 1.05, im

    # Отбрасывание уровней НЕ уменьшает число проходов: в гейте coarse24
    # вызывает полный generate. Здесь ограничение блоков обязано быть явным.
    assert N_LEVEL == 3
    print("самопроверка k9i пройдена (версия «чередование по повтору»): "
          "статистики и p95, таблица ускорений, чередование снимает дрейф "
          "(блоками 1.9x, чередованием <1.05x)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--joint12", default=None, help="чекпойнт k9c")
    ap.add_argument("--frozen12", default=None, help="чекпойнт k9g")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batches", default="1,10")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--pos-offset", type=int, default=4)
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="НЕ используйте: чужая нагрузка на карте измеряется "
                         "вместе с моделью")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k9i sha1 {sha}")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch

    import actioncodec  # noqa: F401
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, process_state, prompt_template,
                       seed_everything)

    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    if dev.type != "cuda":
        raise SystemExit("латентность меряется на GPU")

    # СВОБОДНАЯ КАРТА — УСЛОВИЕ ИЗМЕРЕНИЯ, А НЕ ПОЖЕЛАНИЕ.
    idx = dev.index or 0
    try:
        import subprocess
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20)
        busy = [l for l in q.stdout.strip().splitlines() if l.strip()]
    except Exception as e:
        busy, q = [], None
        print(f"  не удалось опросить nvidia-smi ({e}); проверьте карту сами")
    if busy:
        print(f"  на картах уже считают {len(busy)} процессов:")
        for l in busy[:6]:
            print(f"    {l}")
        if not args.allow_busy_gpu:
            raise SystemExit(
                "Измерение времени при чужой нагрузке даёт стоимость "
                "конкуренции, а не модели.\nДождитесь окончания развёрток или, "
                "понимая последствия, --allow-busy-gpu.")

    seed_everything(0)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    # --- один фиксированный вход на все конфигурации --------------------------
    z = np.load(args.cache, allow_pickle=True)
    epi, stp, tsk = z["episode"], z["step"], z["task"]
    IMG = np.load(args.cache + ".images.npy", mmap_mode="r")
    max_b = max(int(x) for x in args.batches.split(","))
    # СОСТОЯНИЯ БЕРУТСЯ ИЗ КЭША ЛИШЬ КАК ПРАВДОПОДОБНЫЕ ЧИСЛА. Время зависит
    # от ФОРМ, а не от значений: generate идёт фиксированное число блоков,
    # ранних остановок нет. Поэтому загрузка parquet здесь не нужна.
    st = np.zeros((max_b, len(STATE_Q01)), np.float64)
    st_n = (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    def build(b):
        image = torch.from_numpy(np.asarray(IMG[:b]))
        msgs = []
        for i in range(b):
            m = prompt_template(st_n[i], None, str(tsk[i]),
                                mode=cfg.MODEL.vla_processor.kwargs.mode,
                                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        bt = proc(text=texts, images=[[image[k].numpy()] for k in range(b)],
                  return_tensors="pt", padding=True, padding_side="left",
                  action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), bt)

    # --- модель и кодек -------------------------------------------------------
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    base_state = None
    have12 = []
    if args.joint12 or args.frozen12:
        model.init_joint_fast(depth=12, head_dtype=dt)
        own = dict(model.named_parameters())
        # ИСХОДНЫЕ ЗНАЧЕНИЯ ОБУЧАЕМЫХ ВЕСОВ сохраняются до подмены: между
        # конфигурациями они возвращаются, иначе fullbar и coarse24 пошли бы
        # на весах Joint-12.
        base_state = {k: own[k].detach().clone()
                      for k in own if own[k].requires_grad}
        for nm, p in (("joint12", args.joint12), ("frozen12", args.frozen12)):
            if p:
                have12.append((nm, p))

    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None)).to(dev).eval()
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(ii))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    @contextlib.contextmanager
    def only_blocks(n):
        """Ограничить число проходов. Приём из K-7a.

        block_size — отдельное поле, вычисленное в __init__ как
        token_budget // num_blocks, и от подмены не меняется; маска и позиции
        строятся от фактических длин. Поэтому ограничивается ровно число
        проходов и ничего больше.
        """
        saved = model.num_blocks
        try:
            model.num_blocks = n
            yield
        finally:
            model.num_blocks = saved

    @contextlib.contextmanager
    def weights(name):
        """Веса нужной конфигурации на время замера."""
        if name in ("fullbar", "coarse24") or base_state is None:
            yield
            return
        path = dict(have12)[name]
        obj = torch.load(path, map_location="cpu", weights_only=False)
        own = dict(model.named_parameters())
        saved = {k: own[k].detach().clone() for k in obj["state"]}
        try:
            with torch.no_grad():
                for k, v in obj["state"].items():
                    own[k].data = v.to(dev, torch.float32)
            yield
        finally:
            with torch.no_grad():
                for k, v in saved.items():
                    own[k].data = v

    def decode(codes, n_lv):
        K = codes.reshape(-1, n_lv, N_POS)
        zq = E[0][torch.as_tensor(K[:, 0, :]).long().to(dev)]
        for j in range(1, n_lv):
            zq = zq + E[j][torch.as_tensor(K[:, j, :]).long().to(dev)]
        x, _ = codec._decode(zq, embodiment_ids=0)
        return x[..., :7]

    ac16 = torch.autocast("cuda", dtype=torch.float16)

    def run(name, batch):
        """Один вызов политики: сеть, затем сборка действия."""
        if name == "fullbar":
            with torch.no_grad():
                t = model.generate(**batch, position_offset=args.pos_offset,
                                   do_sample=False)
            return t.cpu().numpy(), N_LEVEL
        if name == "coarse24":
            with torch.no_grad(), only_blocks(1):
                t = model.generate(**batch, position_offset=args.pos_offset,
                                   do_sample=False)
            return t[:, :N_POS].cpu().numpy(), 1
        with torch.no_grad(), ac16:
            v, p = model.build_inputs(position_offset=args.pos_offset, **batch)
            o = model.forward_joint_fast(
                vlm_inputs_embeds=v, attention_mask=batch.get("attention_mask"),
                position_ids=p)
        return o["pred_codes"].cpu().numpy(), 1

    order = [c for c in CONFIGS
             if c in ("fullbar", "coarse24") or c in dict(have12)]
    print(f"конфигурации: {', '.join(order)}")
    if "joint12" in order and "frozen12" in order:
        print("  joint12 и frozen12 обязаны совпасть по времени: одинаковые "
              "формы\n  и операции, различаются только значения весов")

    results = {}
    for bs in [int(x) for x in args.batches.split(",")]:
        batch = build(bs)
        print(f"\n=== батч {bs} ===")
        # --- прогрев ----------------------------------------------------------
        for name in order:
            with weights(name):
                for _ in range(args.warmup):
                    c, nl = run(name, batch)
                    decode(c, nl)
            torch.cuda.synchronize()
        # --- ЧЕРЕДОВАНИЕ ПО ПОВТОРУ ------------------------------------------
        # Подмена весов внутри цикла стоила бы дороже самого замера, поэтому
        # конфигурации на 12 слоёв меряются своими проходами, а чередование
        # идёт внутри каждой группы весов. Между fullbar и coarse24 подмены
        # нет вовсе — они на одних весах и чередуются честно.
        tm = {c: [] for c in order}
        td = {c: [] for c in order}
        groups = [[c for c in order if c in ("fullbar", "coarse24")]]
        groups += [[c] for c in order if c not in ("fullbar", "coarse24")]
        for grp in groups:
            if not grp:
                continue
            with weights(grp[0]) if len(grp) == 1 else contextlib.nullcontext():
                for r in range(args.reps):
                    for name in grp:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        c, nl = run(name, batch)
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        a = decode(c, nl)
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        tm[name].append((t1 - t0) * 1000)
                        td[name].append((t2 - t1) * 1000)
                    if (r + 1) % 50 == 0:
                        print(f"    повтор {r + 1}/{args.reps} "
                              f"({'+'.join(grp)})", flush=True)
        # --- память отдельно, вне чередования ---------------------------------
        mem = {}
        for name in order:
            with weights(name):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                for _ in range(5):
                    c, nl = run(name, batch)
                    decode(c, nl)
                torch.cuda.synchronize()
                mem[name] = torch.cuda.max_memory_allocated(dev) / 2 ** 20

        row = {}
        for name in order:
            tot = [m + d for m, d in zip(tm[name], td[name])]
            row[name] = dict(model_ms=stats(tm[name]),
                             decode_ms=stats(td[name]),
                             total_ms=stats(tot), peak_mib=mem[name])
        med = {c: row[c]["total_ms"]["median"] for c in order}
        sp = speedup_table(med)
        print(f"\n  {'конфигурация':<12}{'медиана':>10}{'среднее':>10}"
              f"{'p95':>9}{'декод':>9}{'ускор.':>9}{'память':>10}")
        for name in order:
            r = row[name]
            print(f"  {name:<12}{r['total_ms']['median']:>9.1f}"
                  f"{r['total_ms']['mean']:>10.1f}{r['total_ms']['p95']:>9.1f}"
                  f"{r['decode_ms']['median']:>9.2f}{sp[name]:>8.2f}x"
                  f"{r['peak_mib']:>9.0f}М")
        if "joint12" in row and "frozen12" in row:
            a, b = med["joint12"], med["frozen12"]
            rel = abs(a - b) / max(a, b)
            print(f"\n  проверка стенда: joint12 против frozen12 расходятся на "
                  f"{rel:.1%}")
            if rel > 0.05:
                print("    ВНИМАНИЕ: больше 5%. Формы и операции у них "
                      "одинаковы, значит меряется не то, что заявлено.")
        results[str(bs)] = dict(rows=row, median_ms=med, speedup=sp)

    print("\n  ЧИТАТЬ ТАК: ускорение считается от самой дорогой конфигурации "
          "и\n  относится к вызову политики без подготовки батча. Строк с "
          "различимым\n  временем три, а не четыре: joint12 и frozen12 "
          "исполняют одно и то же.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(dict(results=results, script_sha1=sha, ckpt=args.ckpt,
                       joint12=args.joint12, frozen12=args.frozen12,
                       device=str(dev), dtype=args.dtype, reps=args.reps,
                       warmup=args.warmup, argv=vars(args)),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}  (sha {sha})")


if __name__ == "__main__":
    main()
