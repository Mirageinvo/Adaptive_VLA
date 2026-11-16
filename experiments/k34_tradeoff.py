#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-34: кривая SR — задержка по выходам 8/16/24 (дерево T и flat F).

Читает reports/k34/analysis.json (SR по сидам, 150 эпизодов на руку) и
reports/k34/latency_eval.json (p50 total по глубинам, модели seed 0 —
задержка свойство архитектуры). Пишет reports/k34/tradeoff.png (две
панели: B = 1 и B = 5, общая ось SR) и reports/k34/tradeoff.json
(таблица тех же чисел). Средняя по двум сидам — линия с точками,
каждый сид — полые маркеры той же формы. Ничего не пересчитывает.
"""
import json
import os
import sys

import numpy as np

ANALYSIS = "reports/k34/analysis.json"
LATENCY = "reports/k34/latency_eval.json"
OUT_PNG = "reports/k34/tradeoff.png"
OUT_JSON = "reports/k34/tradeoff.json"
DEPTHS = (8, 16, 24)
ARCH = (("T", "parents_attn24", "дерево (T)", "#2a78d6", "o"),
        ("F", "flat_attn24", "flat (F)", "#eb6834", "s"))
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def table(an, lat, batches=(1, 5)):
    rows = []
    for k, mode, _nm, _c, _m in ARCH:
        for d in DEPTHS:
            sr = [an["sr"][f"s{s}:{k}{d}"] for s in (0, 1)]
            r = dict(arch=k, depth=d, sr_s0=sr[0], sr_s1=sr[1],
                     sr_mean=float(np.mean(sr)))
            for b in batches:
                q = lat[mode][f"batch{b}"][f"d{d}"]["total"]
                r[f"p50_b{b}_ms"] = float(q["median"])
                r[f"p95_b{b}_ms"] = float(q["p95"])
            rows.append(r)
    return rows


def plot(rows, path, batches=(1, 5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(batches), figsize=(5.2 * len(batches),
                                                      4.2), sharey=True)
    fig.patch.set_facecolor(SURF)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, batches):
        ax.set_facecolor(SURF)
        for k, _mode, nm, col, mk in ARCH:
            rr = [r for r in rows if r["arch"] == k]
            x = [r[f"p50_b{b}_ms"] for r in rr]
            y = [100 * r["sr_mean"] for r in rr]
            ax.plot(x, y, color=col, lw=2, zorder=3, label=nm + ", среднее")
            ax.scatter(x, y, s=64, marker=mk, color=col, zorder=4,
                       edgecolors=SURF, linewidths=2)
            for s, a in ((0, 0.9), (1, 0.9)):
                ax.scatter(x, [100 * r[f"sr_s{s}"] for r in rr], s=36,
                           marker=mk, facecolors="none", edgecolors=col,
                           linewidths=1.2, alpha=a, zorder=2)
            for r, xi, yi in zip(rr, x, y):
                ax.annotate(f"{r['depth']}", (xi, yi), xytext=(6, 6),
                            textcoords="offset points", fontsize=9,
                            color=INK2)
        ax.set_title(f"B = {b}", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("задержка p50, мс (полный путь вызова)", fontsize=9,
                      color=INK2)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("SR, % (150 эпизодов на руку и сид)", fontsize=9,
                       color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                 markeredgecolor=MUTED, label="отдельный сид")]
    lab += ["отдельный сид (полый маркер)"]
    axes[-1].legend(h, lab, fontsize=8, frameon=False, loc="lower right",
                    labelcolor=INK2)
    fig.suptitle("K-34: SR и задержка выходов 8/16/24 (числа у точек — "
                 "глубина выхода)", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.text(0.01, 0.005, "Задержка — модели сида 0 (одна архитектура на "
             "оба сида). Разности SR между выходами в пределах ±5–8 п.п. "
             "интервалов; различия n.s.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    if "--selftest" in sys.argv:
        return selftest()
    for f in (ANALYSIS, LATENCY):
        if not os.path.exists(f):
            print(f"  ОТКАЗ: нет {f}")
            return 3
    an = json.load(open(ANALYSIS, encoding="utf-8"))
    lat = json.load(open(LATENCY, encoding="utf-8"))
    rows = table(an, lat)
    plot(rows, OUT_PNG)
    json.dump(dict(rows=rows, analysis=ANALYSIS, latency=LATENCY,
                   latency_device=lat.get("device"),
                   latency_plan_sha1=lat.get("plan_sha1")),
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    for r in rows:
        print(f"  {r['arch']}{r['depth']}: SR {100 * r['sr_s0']:.1f} / "
              f"{100 * r['sr_s1']:.1f} (ср. {100 * r['sr_mean']:.1f}); p50 "
              f"B1 {r['p50_b1_ms']:.1f} мс, B5 {r['p50_b5_ms']:.1f} мс")
    print(f"  {OUT_PNG}, {OUT_JSON}")
    return 0


def selftest():
    import tempfile
    sr = {f"s{s}:{k}{d}": 0.85 + 0.01 * d / 8 + 0.01 * s
          for s in (0, 1) for k in "TF" for d in DEPTHS}
    lat = {m: {f"batch{b}": {f"d{d}": dict(total=dict(median=d * 8.0 * b,
                                                       p95=d * 9.0 * b))
                             for d in DEPTHS} for b in (1, 5)}
           for _k, m, _n, _c, _mk in ARCH}
    rows = table(dict(sr=sr), lat)
    assert len(rows) == 6 and rows[0]["p50_b5_ms"] == 8 * 8.0 * 5
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.png")
        plot(rows, p)
        assert os.path.getsize(p) > 10000
    print("самопроверка k34_tradeoff пройдена: таблица 6 строк, график "
          "строится")
    return 0


if __name__ == "__main__":
    sys.exit(main())
