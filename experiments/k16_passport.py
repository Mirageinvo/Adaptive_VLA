#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16 (шаг 1): паспорт данных и проверка событийного маркера.

ТОЛЬКО ЧТЕНИЕ, только CPU, новые проходы VLM не нужны.

ДАННЫЕ (кэш h18 K-15f + исходный кэш демонстраций K-9a/K-11a):
  * наблюдения, эпизоды, задачи; строки частей train / val_sel кэша h18 и
    их эпизоды; пересечение эпизодов частей (val_confirm не открывается);
  * есть ли для строки h18, a0, экспертный чанк и ключ (эпизод, шаг);
    совпадает ли экспертный чанк кэша h18 с действием исходного кэша;
  * состояния (размерность, диапазоны, пример), источник и ревизия
    датасета, fps из его метаданных (если они есть локально);
  * шаг сетки внутри эпизода (непрерывны ли шаги) — соседние строки НЕ
    считаются соседними вызовами без проверки.
РАЗМЕТКА (пример): устойчивые переключения команды схвата в демонстрации
  (новый знак держится ≥ 2 отсчётов); число на эпизод, первое, эпизоды без
  переключений, несколько примеров.
МАРКЕР В СРЕДЕ: по журналам q0ref K-15h/K-15g — первый вызов, в восьми
  командах a0 которого знак схвата отличается от предыдущей ИСПОЛНЕННОЙ
  команды (прогрев среды не учитывается; в вызове 1 сравнение — с первой
  командой самого чанка; ноль не меняет знак); доля эпизодов с событием и
  распределение вызова.

Выход: печать и reports/k16/passport.json.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

H = 8
STABLE = 2


def sign_seq(x, prev=0):
    """Знаки с переносом: ноль сохраняет предыдущий знак (начальный —
    prev)."""
    out = []
    for v in np.asarray(x, np.float64):
        s = 1 if v > 0 else -1 if v < 0 else prev
        out.append(s)
        prev = s
    return np.asarray(out)


def stable_switches(grip, k=STABLE):
    """Индексы, где знак команды меняется и новый знак держится ≥ k
    отсчётов подряд."""
    s = sign_seq(grip)
    idx = []
    cur = s[0] if len(s) else 0
    for t in range(1, len(s)):
        if s[t] != cur and s[t] != 0 and len(s) - t >= k and \
                (s[t:t + k] == s[t]).all():
            idx.append(t)
            cur = s[t]
    return idx


def first_event_call(a0_grip, executed_grip, done_step, n_calls):
    """a0_grip [C, 8] (команды схвата в единицах исполнения), executed_grip
    [T] — исполненные команды. Номер первого вызова (с 1) с событием или
    None."""
    T = len(executed_grip)
    end = done_step + 1 if done_step >= 0 else T
    for c in range(n_calls):
        t0 = c * H
        if t0 >= end:
            return None
        if c == 0:
            ref = sign_seq(a0_grip[0])[0]
        else:
            ref = sign_seq(executed_grip[:t0])[-1]
        chunk = sign_seq(a0_grip[c], prev=ref)
        if ref != 0 and (chunk != ref).any():
            return c + 1
    return None


def passport_data(a):
    out, p = {}, []
    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    N = int(meta["n_obs"])
    epi = np.asarray(d["episode"])[:N].astype(np.int64)
    stp = np.asarray(d["step"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]
    act = np.asarray(d["action"])[:N]
    out["source"] = dict(meta=f"{a.cache}.meta.json", cache=src,
                         n_obs=N, episodes=int(len(np.unique(epi))),
                         tasks=int(len(np.unique(tsk))),
                         action_shape=list(act.shape),
                         keys=sorted(d.files))
    man = meta.get("manifest") or {}
    out["dataset"] = dict(repo=man.get("dataset_repo"),
                          revision=man.get("dataset_revision"))
    # шаг сетки внутри эпизода
    order = np.lexsort((stp, epi))
    e_s, s_s = epi[order], stp[order]
    same = e_s[1:] == e_s[:-1]
    diffs = (s_s[1:] - s_s[:-1])[same]
    out["step_grid"] = dict(
        diff_counts={str(k): int(v) for k, v in collections.Counter(
            diffs.tolist()).most_common(6)},
        steps_per_episode=dict(
            min=int(min(collections.Counter(epi.tolist()).values())),
            median=float(np.median(list(collections.Counter(
                epi.tolist()).values()))),
            max=int(max(collections.Counter(epi.tolist()).values()))),
        first_step_min=int(min(s_s[np.r_[True, ~same]])))
    # fps из локальных метаданных датасета (без сети)
    fps = None
    try:
        from huggingface_hub import try_to_load_from_cache
        f = try_to_load_from_cache(out["dataset"]["repo"], "meta/info.json",
                                   revision=out["dataset"]["revision"],
                                   repo_type="dataset")
        if isinstance(f, str) and os.path.exists(f):
            fps = json.load(open(f)).get("fps")
    except Exception as e:                      # noqa: BLE001
        out["dataset"]["fps_error"] = type(e).__name__
    out["dataset"]["fps"] = fps
    # состояния
    st_p = src + ".state.npy"
    if os.path.exists(st_p):
        st = np.load(st_p, mmap_mode="r")
        sm = json.load(open(src + ".state.json"))
        X = np.asarray(st[:min(len(st), 200000)], np.float64)
        out["state"] = dict(path=st_p, shape=list(st.shape),
                            meta={k: sm.get(k) for k in sm if k not in (
                                "keys_sha1",)},
                            min=X.min(0).round(4).tolist(),
                            max=X.max(0).round(4).tolist(),
                            example_episode=int(e_s[0]),
                            example_first_rows=np.asarray(
                                st[order[:3]], np.float64).round(4).tolist())
    else:
        p.append(f"нет состояний {st_p}")
    # кэш h18 K-15f
    import k15f_build_cache as kb
    c = kb.load_cache(a.h18_cache, verify_h18=False)
    eps = {}
    for part in kb.PARTS:
        rows = c[part]["rows"]
        e_c = c[part]["epi"]
        if not np.array_equal(e_c, epi[rows]):
            p.append(f"{part}: эпизоды кэша h18 не равны исходному кэшу")
        n_h = min(H, act.shape[1])
        same_act = bool(np.allclose(c[part]["act"][:, :n_h],
                                    act[rows, :n_h, :7], atol=1e-6))
        if not same_act:
            p.append(f"{part}: экспертный чанк кэша h18 не равен исходному")
        eps[part] = set(e_c.tolist())
        out[f"h18_{part}"] = dict(
            rows=int(len(rows)), episodes=int(len(eps[part])),
            tasks=int(len(np.unique(tsk[rows]))),
            h18=list(c[part]["h18"].shape), a0=list(c[part]["a0"].shape),
            act=list(c[part]["act"].shape), act_equals_source=same_act,
            has_key_episode_step=True)
    inter = eps["train"] & eps["val_sel"]
    out["h18_episode_overlap_train_val_sel"] = len(inter)
    if inter:
        p.append(f"эпизоды train и val_sel пересекаются: {len(inter)}")
    # разметка событий схвата на демонстрациях (действие текущего шага)
    sw_n, first, none_ep, examples = [], [], 0, []
    for e in np.unique(e_s)[:2000]:
        m = order[epi[order] == e]
        g = act[m, 0, 6]
        sw = stable_switches(g)
        sw_n.append(len(sw))
        if sw:
            first.append(int(stp[m][sw[0]]))
        else:
            none_ep += 1
        if len(examples) < 3:
            examples.append(dict(episode=int(e), task=str(tsk[m[0]]),
                                 steps=int(len(m)),
                                 switch_steps=[int(stp[m][i]) for i in sw],
                                 signs=[int(sign_seq(g)[i]) for i in sw]))
    out["grip_switch_labels"] = dict(
        rule=f"смена знака команды схвата, новый знак держится ≥ {STABLE} "
             f"отсчётов; ноль сохраняет знак",
        per_episode=dict(collections.Counter(sw_n).most_common(8)),
        episodes_without=none_ep,
        first_switch_step=dict(
            median=float(np.median(first)) if first else None,
            p10=float(np.percentile(first, 10)) if first else None,
            p90=float(np.percentile(first, 90)) if first else None),
        examples=examples,
        note="переключение команды — метка этапа, не успешный захват")
    return out, p


def passport_marker(globs):
    """Событийный маркер по журналам q0ref (единицы исполнения: схват
    харнесс инвертирует, исполняемая команда = −a0)."""
    calls, none_, total = [], 0, 0
    for g in globs:
        for pth in sorted(glob.glob(g)):
            d = json.load(open(pth))
            z = np.load(os.path.join(os.path.dirname(pth), d["actions_npz"]))
            if "k15d_a0" not in z.files:
                continue
            a0 = np.asarray(z["k15d_a0"])            # [C, B, 8, 7]
            A = np.asarray(z["actions"])
            for b in range(a0.shape[1]):
                total += 1
                ev = first_event_call(-a0[:, b, :, 6], A[:, b, 6],
                                      int(z["done_step"][b]), a0.shape[0])
                if ev is None:
                    none_ += 1
                else:
                    calls.append(ev)
    cnt = collections.Counter(calls)
    return dict(episodes=total, with_event=len(calls), without=none_,
                call_counts={str(k): v for k, v in sorted(cnt.items())[:15]},
                median_call=float(np.median(calls)) if calls else None,
                share_call1=(cnt.get(1, 0) / len(calls)) if calls else None)


def selftest():
    assert sign_seq([1, 0, -1, 0]).tolist() == [1, 1, -1, -1]
    assert stable_switches([-1, -1, 1, 1, 1, -1, 1, 1]) == [2]  # всплеск
    assert stable_switches([-1, -1, 1, 1, 1, -1, -1, 1]) == [2, 5]
    assert stable_switches([-1, 1, -1, -1]) == []    # одиночный скачок — нет
    assert stable_switches([1, 1, 1]) == []
    # маркер: вызов 1 без внутренней смены; вызов 3 содержит смену
    a0g = np.full((5, 8), -1.0)
    a0g[2, 5:] = 1.0
    ex = np.full(40, -1.0)
    assert first_event_call(a0g, ex, -1, 5) == 3
    # смена внутри первого чанка — событие в вызове 1
    b = a0g.copy()
    b[0, 4:] = 1.0
    assert first_event_call(b, ex, -1, 5) == 1
    # среда завершилась до вызова 3 — события нет
    assert first_event_call(a0g, ex, 10, 5) is None
    # ноль не меняет знак
    c = np.full((3, 8), -1.0)
    c[1, :] = 0.0
    assert first_event_call(c, np.full(24, -1.0), -1, 3) is None
    print("самопроверка k16_passport пройдена: устойчивое переключение, "
          "маркер по a0 и исполненной команде, завершение, ноль")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-16: паспорт данных")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--out", default="reports/k16/passport.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    data, prob = passport_data(a)
    marker = dict(
        k15h=passport_marker(["reports/k15h/run/p*/q0ref_t*_i*.json"]),
        k15g=passport_marker(["reports/k15g/local/run/p*/q0ref_t*_i*.json"]))
    res = dict(data=data, marker=marker, problems=prob)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 3 if prob else 0


if __name__ == "__main__":
    sys.exit(main())
