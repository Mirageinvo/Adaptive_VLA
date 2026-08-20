"""K-4c0-lite: даёт ли разреженный action-only refiner реальную экономию времени.

ВОПРОС. Всё, что измерено до сих пор, отвечает на «сколько ошибки можно
исправить». Ни один замер не отвечал на «станет ли быстрее». А вся идея метода
— экономия вычислений: если пересчёт четырёх позиций вместо шестнадцати не
даёт выигрыша по времени, то самый точный router бесполезен.

ПОВОД ДЛЯ БЕСПОКОЙСТВА. K-4a3 показал, что внутри существующего BAR переход
16 -> 4 даёт 1.03x, то есть практически ничего: шестнадцать позиций при batch 1
— слишком мало работы, GPU упирается в запуск ядер. Целевая архитектура другая
(лёгкий refiner поверх закэшированного контекста VLM, где позиции действия
составляют почти всю его стоимость), но её пока не существует, и переносить на
неё вывод из BAR нельзя.

ЧТО СРАВНИВАЕТСЯ.
  dense          — 16 query-позиций, как сейчас;
  static sparse  — K заранее известных позиций, без router;
  dynamic sparse — router, top-k, gather запросов, scatter результата;
  masked dense   — считаются все 16, сохраняются K. ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ:
                   так делать нельзя, и замер обязан показать нулевую экономию.

Во всех разреженных режимах ключи и значения ВСЕХ 16 позиций остаются
доступны как контекст, экономия только на стороне запросов. В время
dynamic-режима ВХОДЯТ router, top-k, gather и scatter — иначе получилось бы
нечестное «время без накладных расходов».

ПОЧЕМУ СЛУЧАЙНЫЕ ВЕСА ДОПУСТИМЫ. Время работы определяется формами тензоров и
набором операций, а не значениями весов. Обученность влияет на КАЧЕСТВО, а его
здесь не измеряют. Архитектура и формы обязаны соответствовать будущей модели —
это единственное требование.

ВОРОТА.
  основной критерий — измеренное сквозное ускорение >= 1.15x;
  аналитическая проверка Амдала — f * r >= 1 - 1/1.15 = 0.130, где f это доля
  времени action-refiner в полном выводе, а r — реально сэкономленная доля его
  времени. Она показывает, обречён ли режим заранее: если refiner занимает 10%
  времени, никакая экономия внутри него до 1.15x не дотянет.

РАЗВИЛКА ПРИ ПРОВАЛЕ, записана заранее:
  1. ускорение >= 1.15x — заявляем реальную экономию времени;
  2. ускорения нет, но разреженная модель лучше плотной при РАВНОЙ ИЗМЕРЕННОЙ
     latency или равных FLOPs — работа переформулируется в «эффективное
     распределение бюджета» и продолжается;
  3. нет ни ускорения, ни преимущества при честно выровненной стоимости —
     ветку разреженных позиций закрываем, переходим к adaptive NFE.
Выравнивать по числу пересчитанных позиций НЕЛЬЗЯ: §7б показал, что оно плохо
соответствует реальному времени.

Запуск:
    python3 experiments/k4c0_sparse_refiner_bench.py --selftest
    python3 experiments/k4c0_sparse_refiner_bench.py --out data/k4c0.json
"""

import argparse
import json
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

P_POS, N_LEVELS, VOCAB = 16, 3, 2048


class Block(nn.Module):
    """Один слой refiner: запросы смотрят на все позиции действия и на
    закэшированный контекст VLM.

    Запросы строятся ТОЛЬКО для выбранных позиций — в этом и состоит
    разреженность. Ключи и значения берутся от всех 16 позиций, поэтому
    глобальный контекст плана не теряется.
    """

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.xq = nn.Linear(d, d, bias=False)
        self.o1 = nn.Linear(d, d, bias=False)
        self.o2 = nn.Linear(d, d, bias=False)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d), nn.LayerNorm(d),
                                     nn.LayerNorm(d))
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    def forward(self, qx, full, ctx_kv):
        B = qx.shape[0]
        h = self.n1(qx)
        k, v = self.kv(self.n1(full)).chunk(2, -1)
        a = F.scaled_dot_product_attention(self._split(self.q(h)),
                                           self._split(k), self._split(v))
        qx = qx + self.o1(a.transpose(1, 2).reshape(B, -1, self.h * self.dh))
        ck, cv = ctx_kv
        a = F.scaled_dot_product_attention(self._split(self.xq(self.n2(qx))),
                                           ck, cv)
        qx = qx + self.o2(a.transpose(1, 2).reshape(B, -1, self.h * self.dh))
        return qx + self.mlp(self.n3(qx))


class Refiner(nn.Module):
    """Action-only refiner поверх кэша VLM.

    Голова предсказывает все N_LEVELS кодов выбранной временной позиции: router
    выбирает позицию целиком, а не отдельный уровень RVQ. Стоимость головы
    (d -> 3*2048) заметна и масштабируется числом запросов, поэтому её нельзя
    выносить за скобки замера.
    """

    def __init__(self, d=512, layers=4, heads=8, ctx=512):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.zeros(P_POS, d))
        self.tstep = nn.Embedding(64, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.head = nn.Linear(d, N_LEVELS * VOCAB)
        self.ctx_proj = nn.Linear(d, 2 * d, bias=False)
        self.heads, self.ctx_len = heads, ctx

    def state(self, tokens, step):
        """(B, 16, 3) кодов -> (B, 16, d). Считается один раз на шаг."""
        x = self.emb(tokens).sum(2) + self.pos
        return x + self.tstep(step)[:, None, :]

    def cache_ctx(self, ctx):
        k, v = self.ctx_proj(ctx).chunk(2, -1)
        B, L, _ = k.shape
        sp = lambda t: t.view(B, L, self.heads, -1).transpose(1, 2)
        return sp(k), sp(v)

    def forward(self, full, idx, ctx_kv):
        """full: (B, 16, d) — состояние всех позиций; idx: (B, K) — выбранные."""
        qx = torch.gather(full, 1, idx[..., None].expand(-1, -1, self.d))
        for b in self.blocks:
            qx = b(qx, full, ctx_kv)
        return self.head(qx)


class Router(nn.Module):
    """Маленький set-aware router: self-attention по 16 позициям, top-K."""

    def __init__(self, d=512, dr=128, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(d, dr)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dr, heads, 4 * dr, batch_first=True,
                                       norm_first=True, dropout=0.0), layers)
        self.out = nn.Linear(dr, 1)

    def forward(self, full, K):
        s = self.out(self.enc(self.inp(full))).squeeze(-1)
        return s.topk(K, -1).indices


class Progress:
    """Полоска прогресса по ЗАМЕРАМ, а не по итерациям.

    Единица работы — один вызов bench: он занимает секунды, и более мелкая
    гранулярность только замусорила бы вывод. В терминале строка обновляется
    на месте, при перенаправлении в файл (tee) печатаются отдельные строки —
    иначе лог превращается в кашу из возвратов каретки.
    """

    def __init__(self, total):
        self.total, self.done, self.t0 = total, 0, time.perf_counter()
        self.tty = sys.stderr.isatty()

    def step(self, label):
        self.done += 1
        el = time.perf_counter() - self.t0
        eta = el / self.done * (self.total - self.done)
        frac = self.done / self.total
        bar = "#" * int(30 * frac) + "." * (30 - int(30 * frac))
        msg = (f"[{bar}] {self.done}/{self.total} {frac:5.1%}  "
               f"прошло {el / 60:.1f} мин, осталось ~{eta / 60:.1f} мин  "
               f"{label}")
        if self.tty:
            print("\r" + msg + " " * 8, end="", file=sys.stderr, flush=True)
        elif self.done % max(1, self.total // 40) == 0 or self.done == self.total:
            print(msg, file=sys.stderr, flush=True)

    def close(self):
        if self.tty:
            print(file=sys.stderr, flush=True)


def bench(fn, warmup, iters, device):
    """Медиана и хвосты. На CUDA — события с синхронизацией, иначе perf_counter.

    Синхронизация обязательна с обеих сторон интервала: запуск ядра
    асинхронный, и без неё замерялось бы время постановки в очередь.
    """
    for _ in range(warmup):
        fn()
    ts = []
    if device.type == "cuda":
        torch.cuda.synchronize()
        for _ in range(iters):
            a, b = (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    q = lambda p: ts[min(len(ts) - 1, int(p * len(ts)))]
    return dict(median=statistics.median(ts), p10=q(0.10), p90=q(0.90),
                p95=q(0.95), mean=statistics.fmean(ts))


def make_step(model, router, mode, K, B, device, dtype):
    """Замыкание одного шага refinement в заданном режиме.

    ВАЖНО: в dynamic входят router, top-k, gather и scatter. В static —
    только gather и scatter с заранее известными индексами. В masked dense
    считаются все 16 позиций, и лишь потом K из них сохраняются.
    """
    tok = torch.randint(0, VOCAB, (B, P_POS, N_LEVELS), device=device)
    ctx = torch.randn(B, model.ctx_len, model.d, device=device, dtype=dtype)
    step = torch.zeros(B, dtype=torch.long, device=device)
    fixed = torch.arange(K, device=device).expand(B, K).contiguous()
    all_idx = torch.arange(P_POS, device=device).expand(B, P_POS).contiguous()

    # CUDA-ГРАФЫ И ПЕРЕДАЧА ТЕНЗОРОВ МЕЖДУ СКОМПИЛИРОВАННЫМИ МОДУЛЯМИ.
    # При mode="reduce-overhead" выход скомпилированного модуля живёт в
    # буфере графа и перезаписывается следующим запуском. Индексы, выданные
    # router, идут дальше в refiner — то есть в ДРУГОЙ граф, и без разметки
    # шага это падает с «accessing tensor output of CUDAGraphs that has been
    # overwritten». Лечится ровно так, как советует сама ошибка: отметить
    # начало шага и скопировать тензор, пересекающий границу графов.
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)

    # КЭШ КОНТЕКСТА СЧИТАЕТСЯ ВНЕ ИЗМЕРЯЕМОГО ИНТЕРВАЛА. В развёрнутой системе
    # ключи и значения VLM-контекста вычисляются ОДИН РАЗ на наблюдение и
    # переиспользуются всеми T шагами refinement. Раньше вызов стоял внутри
    # run(), то есть его стоимость заряжалась каждому шагу — а она растёт
    # линейно с длиной контекста и от числа запросов не зависит. Это раздувало
    # фиксированную часть шага и занижало долю, которую даёт разреженность,
    # причём тем сильнее, чем длиннее контекст.
    with torch.no_grad():
        ctx_kv = model.cache_ctx(ctx)

    def run():
        if mark is not None:
            mark()
        with torch.no_grad():
            full = model.state(tok, step)
            if mode == "dense":
                out = model(full, all_idx, ctx_kv)
                idx = all_idx
            elif mode == "static":
                out, idx = model(full, fixed, ctx_kv), fixed
            elif mode == "dynamic":
                # clone обязателен: индексы пересекают границу двух графов.
                # Он же входит в измеряемое время — это честная часть цены
                # динамического выбора, а не накладные расходы стенда.
                idx = router(full, K).clone()
                out = model(full, idx, ctx_kv)
            elif mode == "masked":
                out = model(full, all_idx, ctx_kv)  # считаем всё
                idx = fixed                          # сохраняем K — контроль
                out = torch.gather(
                    out, 1, idx[..., None].expand(-1, -1, out.shape[-1]))
            else:
                raise ValueError(mode)
            codes = out.view(B, idx.shape[1], N_LEVELS, VOCAB).argmax(-1)
            new = tok.clone()
            new.scatter_(1, idx[..., None].expand(-1, -1, N_LEVELS), codes)
            return new
    return run


def selftest():
    """Проверка КОРРЕКТНОСТИ разреженного пути, а не времени.

    Время проверить синтетикой нельзя, а вот перепутать gather со scatter —
    очень легко, и такая ошибка не видна в замере: он просто померит не то.
    Проверяем на CPU с фиксированными весами:
      1. разреженный проход меняет ТОЛЬКО выбранные позиции;
      2. masked dense даёт на выбранных позициях РОВНО то же, что dense;
      3. static с теми же индексами даёт то же, что dynamic;
      4. тождество Амдала f*r = 1 - 1/R выполняется численно.
    """
    torch.manual_seed(0)
    dev = torch.device("cpu")
    m = Refiner(d=64, layers=2, heads=4, ctx=8).to(dev).eval()
    r = Router(d=64, dr=32, heads=4, layers=1).to(dev).eval()
    B, K = 2, 4
    tok = torch.randint(0, VOCAB, (B, P_POS, N_LEVELS))
    ctx = torch.randn(B, 8, 64)
    step = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        kv = m.cache_ctx(ctx)
        full = m.state(tok, step)
        idx = r(full, K)
        out_s = m(full, idx, kv)
        out_d = m(full, torch.arange(P_POS).expand(B, P_POS).contiguous(), kv)

    codes = out_s.view(B, K, N_LEVELS, VOCAB).argmax(-1)
    new = tok.clone()
    new.scatter_(1, idx[..., None].expand(-1, -1, N_LEVELS), codes)
    changed = (new != tok).any(-1)
    sel = torch.zeros_like(changed)
    sel.scatter_(1, idx, True)
    assert bool((changed & ~sel).sum() == 0), \
        "разреженный проход изменил позицию вне выбранных"

    g = torch.gather(out_d, 1, idx[..., None].expand(-1, -1, out_d.shape[-1]))
    assert torch.allclose(g, out_s, atol=1e-5), \
        "masked dense и sparse разошлись на выбранных позициях — " \
        "значит gather/scatter или маскирование сделаны неверно"

    with torch.no_grad():
        out_f = m(full, idx, kv)
    assert torch.allclose(out_f, out_s, atol=1e-6), "путь недетерминирован"

    # ТОЖДЕСТВО ПРОВЕРЯЕТСЯ ПРОТИВ НЕЗАВИСИМОГО СЧЁТА, а не подстановкой в
    # собственную формулу. Пусть C — время контекста, D и S — время плотного и
    # разреженного расписания. Тогда f = D/(C+D), r = 1 - S/D, и сквозное
    # ускорение (C+D)/(C+S) обязано равняться 1/(1 - f*r).
    for C, D, S in ((273.3, 40.0, 15.0), (10.0, 100.0, 25.0),
                    (500.0, 5.0, 1.0)):
        f_ = D / (C + D)
        r_ = 1 - S / D
        assert abs(1 / (1 - f_ * r_) - (C + D) / (C + S)) < 1e-9, \
            "формула Амдала не сходится с прямым отношением времён"
    # и обратный ход: требуемая доля refiner при заданной экономии
    assert abs((1 - 1 / 1.15) / 0.656 - 0.1988) < 1e-3, \
        "требуемая доля refiner посчитана не так, как в FINDINGS §7б"
    print("самопроверка пройдена: разреженный путь трогает только выбранные "
          "позиции, совпадает с dense на них, детерминирован; Амдал сходится")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--dmodel", type=int, nargs="+", default=[256, 512, 768])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--K", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16],
                    help="число шагов refinement T; первый шаг всегда плотный")
    ap.add_argument("--ctx-len", type=int, default=512,
                    help="длина закэшированного контекста VLM в токенах")
    ap.add_argument("--context-ms", type=float, default=None,
                    help="измеренное время VLM/контекста на вызов; если задано,"
                         " считается сквозное ускорение, иначе — требуемая "
                         "доля refiner")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    dev = torch.device(args.device if torch.cuda.is_available()
                       or args.device == "cpu" else "cpu")
    dtype = getattr(torch, args.dtype)
    if dev.type == "cuda":
        cc = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]}), "
              f"torch {torch.__version__}, CUDA {torch.version.cuda}")
        # BF16 АППАРАТНО ПОЯВИЛСЯ В AMPERE (sm_80). На Volta и Turing он
        # выполняется эмуляцией, и соотношения времён не переносятся ни на
        # какое реальное развёртывание. Молча мерить в этом режиме нельзя.
        if args.dtype == "bfloat16" and cc[0] < 8:
            print("  ВНИМАНИЕ: bfloat16 без аппаратной поддержки на sm_"
                  f"{cc[0]}{cc[1]} — числа НЕ ПЕРЕНОСИМЫ. Повторите с "
                  "--dtype float16 (нативен здесь) и --dtype float32.")
    else:
        print("CUDA недоступна — замер на CPU, к целевой машине НЕ ОТНОСИТСЯ")
    print(f"dtype {args.dtype}, контекст {args.ctx_len} токенов, "
          f"прогрев {args.warmup}, итераций {args.iters}, "
          f"compile={args.compile}")

    # ОБЩИЙ ОБЪЁМ РАБОТЫ считается заранее: dense меряется один раз на
    # конфигурацию, остальные три режима — на каждое K.
    total = (len(args.dmodel) * len(args.layers) * len(args.batch)
             * (len(args.K) * 3 + 1))
    print(f"всего замеров: {total}, по {args.warmup} прогревочных и "
          f"{args.iters} измеряемых итераций в каждом")
    prog = Progress(total)

    res = {}
    for d in args.dmodel:
        for L in args.layers:
            m = Refiner(d=d, layers=L, heads=8, ctx=args.ctx_len)
            m = m.to(dev).to(dtype).eval()
            rt = Router(d=d).to(dev).to(dtype).eval()
            n_par = sum(p.numel() for p in m.parameters())
            n_rt = sum(p.numel() for p in rt.parameters())
            if args.compile:
                m = torch.compile(m, mode="reduce-overhead")
                rt = torch.compile(rt, mode="reduce-overhead")
            for B in args.batch:
                key = f"d{d}_L{L}_b{B}"
                print(f"\n=== d_model {d}, слоёв {L}, batch {B} "
                      f"(refiner {n_par / 1e6:.1f}M, router {n_rt / 1e6:.2f}M)")
                t = {}
                for K in args.K:
                    for mode in ("dense", "static", "dynamic", "masked"):
                        if mode == "dense" and K != args.K[0]:
                            continue          # dense от K не зависит
                        fn = make_step(m, rt, mode, K, B, dev, dtype)
                        t[f"{mode}_K{K}"] = bench(fn, args.warmup, args.iters,
                                                  dev)
                        prog.step(f"d{d} L{L} b{B} {mode} K={K}")
                dense = t[f"dense_K{args.K[0]}"]["median"]
                print(f"  {'режим':<16}{'медиана мс':>12}{'p90':>9}{'p95':>9}"
                      f"{'ускорение шага':>16}")
                print(f"  {'dense (16)':<16}{dense:>12.3f}"
                      f"{t[f'dense_K{args.K[0]}']['p90']:>9.3f}"
                      f"{t[f'dense_K{args.K[0]}']['p95']:>9.3f}"
                      f"{1.0:>16.3f}")
                for K in args.K:
                    for mode in ("static", "dynamic", "masked"):
                        v = t[f"{mode}_K{K}"]
                        print(f"  {mode + f' (K={K})':<16}{v['median']:>12.3f}"
                              f"{v['p90']:>9.3f}{v['p95']:>9.3f}"
                              f"{dense / v['median']:>16.3f}")
                # РАЗЛОЖЕНИЕ ЭКОНОМИИ. Три числа отвечают на три разных
                # вопроса, и смешивать их нельзя:
                #   dense - masked  — экономия на ВЫХОДЕ (argmax и scatter для
                #                     4 позиций вместо 16). Она реальна, но к
                #                     разреженности запросов отношения не
                #                     имеет: логиты всё равно посчитаны все.
                #   masked - static — экономия на ЗАПРОСАХ, то есть ровно то,
                #                     ради чего затевается метод.
                #   dynamic - static — цена router, top-k, gather и scatter.
                res[key] = dict(times=t, params=int(n_par),
                                router_params=int(n_rt), dense=dense)
                for K in args.K:
                    if K >= P_POS:
                        continue
                    ms, st = t[f"masked_K{K}"]["median"], t[f"static_K{K}"]["median"]
                    dy = t[f"dynamic_K{K}"]["median"]
                    # ПОРОГ РАЗРЕШЕНИЯ. Разброс самого плотного режима задаёт
                    # масштаб, ниже которого разность неотличима от шума.
                    # Без этого «экономия» в 0.03 мс на шаге 1.1 мс читается
                    # как результат, хотя знак у неё случайный — что и
                    # обнаружилось: static выходил медленнее masked, хотя
                    # делает строго меньше работы.
                    d0 = t[f"dense_K{args.K[0]}"]
                    noise = d0["p90"] - d0["p10"]
                    tag = lambda v: ("" if abs(v) > noise
                                     else "  В ПРЕДЕЛАХ ШУМА")
                    print(f"\n  разложение при K={K} "
                          f"(порог разрешения {noise:.3f} мс):")
                    print(f"    на выходе  (dense-masked)   {dense - ms:>8.3f} мс"
                          f"  {(dense - ms) / dense:>7.1%}{tag(dense - ms)}")
                    print(f"    НА ЗАПРОСАХ (masked-static) {ms - st:>8.3f} мс"
                          f"  {(ms - st) / dense:>7.1%}{tag(ms - st)}"
                          + ("" if abs(ms - st) <= noise else "  <- предмет метода"))
                    print(f"    цена router (dynamic-static){dy - st:>8.3f} мс"
                          f"  {(dy - st) / dense:>7.1%}{tag(dy - st)}")
                    if ms < st - noise:
                        print("    ВНИМАНИЕ: static медленнее masked за "
                              "пределами шума — считать меньше запросов не "
                              "может быть дороже, замер под вопросом")
                    if dy - st > ms - st:
                        print("    router дороже всей экономии на запросах: "
                              "динамический выбор себя не окупает")
                    # ПОТОЛОК ВЕТКИ. Даже при БЕСПЛАТНОМ router и refiner,
                    # занимающем ВСЁ время вывода, сквозное ускорение не
                    # превысит 1/(1 - r_max), где r_max = (T-1)/T*(1 - S/D).
                    # Если этот потолок ниже 1.15x, ворота недостижимы в
                    # принципе, а не из-за качества реализации.
                    for T_ in args.steps:
                        rmax = (T_ - 1) / T_ * max(0.0, 1 - st / dense)
                        print(f"    потолок ветки при T={T_}, бесплатном "
                              f"router и f=100%: {1 / (1 - rmax):.3f}x"
                              + ("" if 1 / (1 - rmax) >= 1.15
                                 else "  <- НЕДОСТИЖИМО"))
                    res[key][f"decomp_K{K}"] = dict(
                        out_side=dense - ms, query_side=ms - st,
                        router_cost=dy - st)

                # ---- РАСПИСАНИЕ: первый шаг плотный, остальные разреженные
                print(f"\n  цикл refinement, первый шаг плотный:")
                print(f"  {'T':>4}{'K':>4}{'плотный мс':>12}{'разрежен. мс':>14}"
                      f"{'экономия r':>12}{'нужна доля f':>14}")
                for T in args.steps:
                    for K in args.K:
                        if K >= P_POS:
                            continue
                        sp = t[f"dynamic_K{K}"]["median"]
                        tot_d = T * dense
                        tot_s = dense + (T - 1) * sp
                        r_ = 1.0 - tot_s / tot_d
                        need_f = ((1 - 1 / 1.15) / r_) if r_ > 0 else float("inf")
                        print(f"  {T:>4}{K:>4}{tot_d:>12.3f}{tot_s:>14.3f}"
                              f"{r_:>12.1%}{min(need_f, 9.99):>13.1%}")
                        res[key][f"sched_T{T}_K{K}"] = dict(
                            dense_ms=tot_d, sparse_ms=tot_s, r=r_,
                            required_share=need_f)
                        if args.context_ms is not None:
                            e2e = ((args.context_ms + tot_d)
                                   / (args.context_ms + tot_s))
                            print(f"      сквозное при контексте "
                                  f"{args.context_ms:.1f} мс: {e2e:.3f}x"
                                  + ("  ВОРОТА ПРОЙДЕНЫ" if e2e >= 1.15
                                     else "  ниже 1.15x"))
                            res[key][f"sched_T{T}_K{K}"]["e2e"] = e2e

    prog.close()
    print("\n" + "=" * 74)
    print("ЧИТАТЬ ТАК, правило записано до запуска")
    print("=" * 74)
    print("  masked считает все 16 запросов, но argmax и scatter делает\n"
          "  только для K. Поэтому он законно быстрее dense — на величину\n"
          "  экономии НА ВЫХОДЕ, которая к разреженности запросов отношения\n"
          "  не имеет. Предмет метода — только (masked - static).\n"
          "  Если masked оказался быстрее static, замер неверен: считать\n"
          "  меньше запросов не может быть дороже.\n"
          "  dynamic минус static — это цена router, top-k, gather и scatter.\n"
          "  Если она съедает экономию, разреженность нужна СТАТИЧЕСКАЯ, а\n"
          "  обучаемый выбор себя не окупает.\n"
          "  «нужна доля f» — какую долю полного вывода обязан занимать\n"
          "  refiner, чтобы вышло 1.15x. Больше 100% означает, что режим\n"
          "  обречён независимо от качества router.")

    if args.out:
        import hashlib
        import subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             text=True).strip()
        except Exception:
            commit = "unknown"
        res["meta"] = dict(
            commit=commit, device=str(dev), dtype=args.dtype,
            gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
            torch=torch.__version__, cuda=torch.version.cuda,
            ctx_len=args.ctx_len, warmup=args.warmup, iters=args.iters,
            compiled=bool(args.compile), context_ms=args.context_ms,
            self_sha256=hashlib.sha256(
                open(__file__, "rb").read()).hexdigest()[:16])
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
