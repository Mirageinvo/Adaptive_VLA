"""K-6c: сколько на самом деле стоит уточнитель над шестнадцатью позициями.

ЗАЧЕМ. Планируя однопроходную схему, легко назначить уточнителю бюджет «на
глаз» — и промахнуться на порядок. Проход экспертной башни стоит 25.8 мс,
потому что обрабатывает 187 токенов (171 префикса плюс 16 действий). Уточнитель
читает ТОЛЬКО шестнадцать. Прикидка по цене «слой-токена» даёт для четырёх
слоёв над шестнадцатью позициями около 0.4 мс — то есть в бюджет 9.1 мс,
нужный для ускорения 1.4x, укладывается уточнитель ГЛУБЖЕ полной экспертной
башни.

Прикидка — не измерение. При таких величинах доминируют накладные расходы на
запуск ядер, и реальная цена может оказаться в разы выше расчётной, зато почти
не зависящей от глубины. Что именно — решает этот замер, и он определяет,
проектировать ли маленькую голову или глубокую.

ЧТО МЕРЯЕТСЯ. Время прямого прохода уточнителя при batch 1, с прогревом и
синхронизацией вокруг каждого запуска, для нескольких глубин и ширин. Отдельно
печатается доля от прохода экспертной башни и запас до бюджета.

ЧЕСТНАЯ ОГОВОРКА. Меряется голая архитектура, без интеграции: в живой системе
добавятся перекладывания тензоров и, возможно, синхронизация с основным
потоком. Поэтому число — НИЖНЯЯ оценка, и в ворота его закладывать надо с
запасом.

Запуск:
    python3 experiments/k6c_refiner_cost.py --selftest
    python3 experiments/k6c_refiner_cost.py --device cuda --out data/k6c.json
"""

import argparse
import json
import os

# измерено в K-5a на V100, fp16, batch 1
EXPERT_PASS_MS = 25.8
FULL_CALL_MS = 194.9
CACHED_BAR_MS = 148.9
ONE_PASS_MS = 97.3


def budget_for(speedup):
    """Сколько мс остаётся уточнителю, чтобы выйти на заданное ускорение
    против BAR с честным кэшем башни."""
    return CACHED_BAR_MS / speedup - ONE_PASS_MS


def selftest():
    for sp, want in ((1.5, 2.0), (1.4, 9.1), (1.3, 17.2)):
        got = budget_for(sp)
        assert abs(got - want) < 0.15, f"бюджет для {sp}x: {got:.1f}, ждали {want}"
    assert budget_for(2.0) < 0, \
        "при 2x бюджет обязан быть отрицательным: один проход столько не даёт"
    print("самопроверка пройдена: бюджеты 1.5x/1.4x/1.3x = "
          f"{budget_for(1.5):.1f}/{budget_for(1.4):.1f}/{budget_for(1.3):.1f} мс")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-pos", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-codes", type=int, default=2048)
    ap.add_argument("--d-action", type=int, default=768,
                    help="размерность h (экспертная башня), из k6d")
    ap.add_argument("--d-vlm", type=int, default=2048,
                    help="размерность префикса VLM; ПРОВЕРИТЬ по выводу k6d, "
                         "умолчание — догадка")
    ap.add_argument("--ctx-len", type=int, default=171,
                    help="длина закэшированного префикса VLM для cross-attention")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return

    import time

    import torch
    import torch.nn as nn

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if dev.type == "cuda":
        cc = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")

    class Refiner(nn.Module):
        """Уточнитель над позициями чанка с РАЗРЕЖЁННЫМ доступом к префиксу.

        Три отличия от прежней прикидки, каждое стоит времени и потому меряется,
        а не оценивается:

        1. ВХОДНАЯ И КОНТЕКСТНАЯ ПРОЕКЦИИ. h имеет размерность экспертной башни,
           префикс — размерность VLM, и они РАЗНЫЕ. Прежний замер подавал обоим
           одно d, то есть проекции не считал вовсе.
        2. ПЕРЕКРЁСТНОЕ ВНИМАНИЕ НЕ В КАЖДОМ СЛОЕ. Префикс надо ВПРЫСНУТЬ, а не
           перечитывать двадцать четыре раза. Раньше таблица разрежённых схем
           получалась умножением «слои x 0.29 + внимания x 0.50» — то есть была
           арифметикой, поданной рядом с измерениями.
        3. ТРИ ВЫХОДНЫЕ ГОЛОВЫ на 2048 кодов каждая — тоже не бесплатны.
        """

        def __init__(self, layers, d, d_in, d_ctx, xa_at, heads=8, ff=4):
            super().__init__()
            self.inp = nn.Linear(d_in, d)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(d, heads, d * ff, batch_first=True,
                                           norm_first=True, dropout=0.0)
                for _ in range(layers)])
            self.xa_at = set(xa_at)
            if self.xa_at:
                self.ctx_proj = nn.Linear(d_ctx, d)
                self.xa = nn.ModuleDict({
                    str(i): nn.MultiheadAttention(d, heads, batch_first=True)
                    for i in self.xa_at})
                self.xa_norm = nn.ModuleDict({
                    str(i): nn.LayerNorm(d) for i in self.xa_at})
            self.out = nn.ModuleList([nn.Linear(d, args.n_codes)
                                      for _ in range(3)])

        def forward(self, x, mem=None):
            x = self.inp(x)
            m = self.ctx_proj(mem) if self.xa_at else None
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.xa_at:
                    a, _ = self.xa[str(i)](self.xa_norm[str(i)](x), m, m,
                                           need_weights=False)
                    x = x + a
            return [o(x) for o in self.out]

    def timeit(m, *xs):
        with torch.no_grad():
            for _ in range(args.warmup):
                m(*xs)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                m(*xs)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.iters * 1e3

    print(f"\n  вход: batch 1, {args.n_pos} позиций, d={args.d_model}, "
          f"{args.dtype}; прогрев {args.warmup}, замеров {args.iters}")
    print(f"  для сравнения: проход экспертной башни {EXPERT_PASS_MS} мс "
          f"(24 слоя над 187 токенами)")
    print(f"  бюджет: {budget_for(1.4):.1f} мс на 1.4x, "
          f"{budget_for(1.3):.1f} мс на 1.3x\n")
    print(f"  {'слоёв':>6}{'d':>6}{'вход':>12}{'мс':>9}{'доля прохода':>15}"
          f"{'ускорение итого':>17}{'вердикт':>10}")
    rows = []
    x = torch.randn(1, args.n_pos, args.d_action, device=dev, dtype=dtype)
    mem = torch.randn(1, args.ctx_len, args.d_vlm, device=dev, dtype=dtype)
    # СХЕМЫ: (слоёв, где перекрёстное внимание). Пустой кортеж — только h.
    schemes = []
    for L in (2, 4, 6, 12, 24):
        schemes.append((L, ()))
    for L, k in ((6, 6), (12, 2), (12, 3), (24, 2), (24, 3)):
        step = max(1, L // k)
        schemes.append((L, tuple(range(0, L, step))[:k]))
    for d in (args.d_model, args.d_model * 2):
        for L, xa in schemes:
            m = Refiner(L, d, args.d_action, args.d_vlm, xa).to(dev, dtype).eval()
            ms = timeit(m, x, mem) if xa else timeit(m, x)
            total = ONE_PASS_MS + ms
            sp = CACHED_BAR_MS / total
            ok = "да" if ms <= budget_for(1.4) else (
                "1.3x" if ms <= budget_for(1.3) else "нет")
            tag = f"кэш x{len(xa)}" if xa else "только h"
            rows.append(dict(layers=L, d_model=d, xa=list(xa), ms=ms,
                             total_ms=total, speedup=sp))
            print(f"  {L:>6}{d:>6}{tag:>12}{ms:>9.3f}"
                  f"{ms / EXPERT_PASS_MS:>14.1%}{sp:>16.2f}x{ok:>10}")
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    print("\n  ЧИТАТЬ ТАК, правило записано до запуска.")
    print("  Если время почти не растёт с глубиной — доминируют накладные")
    print("  расходы на запуск ядер, и брать надо САМЫЙ ГЛУБОКИЙ вариант,")
    print("  влезающий в бюджет: ёмкость достаётся бесплатно.")
    print("  Если растёт линейно — глубина стоит денег, и выбирать её надо")
    print("  по кривой качества из развёртки, а не по максимуму.")
    print("  Замер — НИЖНЯЯ оценка: интеграция добавит перекладывания.")
    print("  Разница «сам» и «сам+кэш» — цена ДОСТУПА К ПРЕФИКСУ. Сам префикс")
    print("  уже посчитан в единственном тяжёлом проходе и кэшируется точно")
    print("  (k5a: расхождение 0.000e+00), поэтому платим только за внимание.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(dict(rows=rows, expert_pass_ms=EXPERT_PASS_MS,
                       cached_bar_ms=CACHED_BAR_MS, one_pass_ms=ONE_PASS_MS,
                       budget_1_4=budget_for(1.4), budget_1_3=budget_for(1.3),
                       n_pos=args.n_pos, dtype=args.dtype),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
