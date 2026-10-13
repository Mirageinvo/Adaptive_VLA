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
import hashlib
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


def stable_switches(grip, k=STABLE, steps=None):
    """Индексы, где знак команды меняется и новый знак держится ≥ k
    ПОСЛЕДОВАТЕЛЬНЫХ отсчётов. Первый ненулевой знак задаёт начальный
    режим (начальный ноль переключения не создаёт). Если даны шаги
    (steps), устойчивость требует подряд идущих шагов; иначе — строк."""
    s = sign_seq(grip)
    nz = np.nonzero(s)[0]
    if not len(nz):
        return []
    idx, cur = [], s[nz[0]]
    st = None if steps is None else np.asarray(steps)
    for t in range(nz[0] + 1, len(s)):
        if s[t] == cur or len(s) - t < k or not (s[t:t + k] == s[t]).all():
            continue
        if st is not None and not (np.diff(st[t - 1:t + k]) == 1).all():
            continue                     # пробел в шагах: момент не известен
        idx.append(t)
        cur = s[t]
    return idx


def first_event_call(a0_grip, executed_grip, n_calls, n_exec):
    """a0_grip [C, 8] (команды схвата в единицах исполнения), executed_grip
    [T] — исполненные команды, n_exec — число исполненных шагов (до
    завершения среды). Решение причинное: смотрится весь чанк a0
    достигнутого вызова. Возвращает (вызов с 1 или None, исполнилась ли
    смена знака в этом вызове)."""
    for c in range(n_calls):
        t0 = c * H
        if t0 >= n_exec:
            return None, False
        if c == 0:
            ref = sign_seq(a0_grip[0])
            ref = ref[np.nonzero(ref)[0][0]] if ref.any() else 0
        else:
            ref = sign_seq(executed_grip[:t0])[-1]
        chunk = sign_seq(a0_grip[c], prev=ref)
        if ref != 0 and (chunk != ref).any():
            ex = sign_seq(executed_grip[t0:min(t0 + H, n_exec)], prev=ref)
            return c + 1, bool((ex != ref).any())
    return None, False


def passport_data(a):
    out, p = {}, []
    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    N = int(meta["n_obs"])
    epi = np.asarray(d["episode"])[:N].astype(np.int64)
    stp_raw = np.asarray(d["step"])[:N]           # для keys_sha1 — как есть
    stp = stp_raw.astype(np.int64)
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
    contiguous = bool(len(diffs) and (diffs == 1).all())
    out["step_grid"] = dict(
        contiguous=contiguous,
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
    out["timing"] = dict(
        demo_fps=fps,
        actioncodec_loader="delta_timestamps actions = i/10 (scripts/utils.py)",
        env_control_hz=("не задано в коде репозитория; robosuite/LIBERO по "
                        "умолчанию 20 Гц — проверить в обёртке среды"),
        note=("при 20 Гц восемь исполняемых команд = 0.4 с, восемь отсчётов "
              "демонстрации при 10 fps = 0.8 с; историю собирать по времени "
              "вызовов, контракт q0 не меняется"))
    out["state_channels"] = dict(
        process_state="utils.process_state: [pos 3, axis-angle 3, пальцы 2]"
                      " из сырого [пальцы 2, pos 3, quat 4]",
        dataset="соответствие порядку датасета подтвердить по диапазонам "
                "(min/max ниже)")
    # состояния
    st_p = src + ".state.npy"
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([epi, stp_raw])).tobytes()).hexdigest()[:12]
    if os.path.exists(st_p):
        st = np.load(st_p, mmap_mode="r")
        sm = json.load(open(src + ".state.json"))
        # ПРОИСХОЖДЕНИЕ состояний: те же наблюдения, датасет и ревизия
        for k_, want in (("keys_sha1", keys_sha), ("n_obs", N),
                         ("dataset_repo", out["dataset"]["repo"]),
                         ("dataset_revision", out["dataset"]["revision"])):
            if str(sm.get(k_)) != str(want):
                p.append(f"состояния: {k_} = {sm.get(k_)!r}, ожидалось "
                         f"{want!r}")
        if len(st) < N:
            p.append(f"состояний {len(st)} меньше наблюдений {N}")
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
        sw = stable_switches(g, steps=stp[m])
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
             f"ПОСЛЕДОВАТЕЛЬНЫХ шагов; первый ненулевой знак — начальный "
             f"режим; ноль сохраняет знак",
        diagnostic_only=not contiguous,
        note_end=("последняя строка кэша не обязательно конец демонстрации"
                  if not contiguous else None),
        per_episode=dict(collections.Counter(sw_n).most_common(8)),
        episodes_without=none_ep,
        first_switch_step=dict(
            median=float(np.median(first)) if first else None,
            p10=float(np.percentile(first, 10)) if first else None,
            p90=float(np.percentile(first, 90)) if first else None),
        examples=examples,
        note="переключение команды — метка этапа, не успешный захват")
    return out, p


def passport_marker(globs, expect):
    """Событийный маркер по журналам q0ref (единицы исполнения: схват
    харнесс инвертирует, исполняемая команда = −a0). Каждое (задача,
    состояние) — один раз; пропущенные файлы перечисляются."""
    calls, executed, none_, seen, skipped = [], 0, 0, set(), []
    for g in globs:
        for pth in sorted(glob.glob(g)):
            d = json.load(open(pth))
            z = np.load(os.path.join(os.path.dirname(pth), d["actions_npz"]))
            if "k15d_a0" not in z.files:
                skipped.append(pth)
                continue
            a0 = np.asarray(z["k15d_a0"])            # [C, B, 8, 7]
            A = np.asarray(z["actions"])
            for b, e in enumerate(d["episodes"]):
                key = (int(d["task_id"]), int(e["init_state_id"]))
                if key in seen:
                    continue
                seen.add(key)
                ds = int(z["done_step"][b])
                n_exec = ds + 1 if ds >= 0 else len(A)
                ev, ex = first_event_call(-a0[:, b, :, 6], A[:, b, 6],
                                          a0.shape[0], n_exec)
                if ev is None:
                    none_ += 1
                else:
                    calls.append(ev)
                    executed += ex
    cnt = collections.Counter(calls)
    return dict(states=len(seen), expected_states=expect,
                complete=len(seen) == expect, skipped_files=skipped,
                predicted_event_in_reached_call=len(calls),
                executed_switch_in_that_call=executed, without=none_,
                call_counts={str(k): v for k, v in sorted(cnt.items())[:15]},
                median_call=float(np.median(calls)) if calls else None,
                share_call1=(cnt.get(1, 0) / len(calls)) if calls else None,
                note="момент решения причинный: done_step в выбор не входит")


def selftest():
    assert sign_seq([1, 0, -1, 0]).tolist() == [1, 1, -1, -1]
    assert stable_switches([-1, -1, 1, 1, 1, -1, 1, 1]) == [2]  # всплеск
    assert stable_switches([-1, -1, 1, 1, 1, -1, -1, 1]) == [2, 5]
    assert stable_switches([-1, 1, -1, -1]) == []    # одиночный скачок — нет
    assert stable_switches([1, 1, 1]) == []
    assert stable_switches([0, 1, 1]) == []            # начальный ноль
    assert stable_switches([0, 0, -1, -1, 1, 1]) == [4]
    # прореженные шаги: устойчивость не подтверждена
    assert stable_switches([-1, 1, 1], steps=[0, 20, 40]) == []
    assert stable_switches([-1, 1, 1], steps=[0, 1, 2]) == [1]
    # маркер: вызов 3 содержит смену; исполнилась ли она
    a0g = np.full((5, 8), -1.0)
    a0g[2, 5:] = 1.0
    ex = np.full(40, -1.0)
    assert first_event_call(a0g, ex, 5, 40) == (3, False)
    ex2 = ex.copy()
    ex2[21:24] = 1.0
    assert first_event_call(a0g, ex2, 5, 40) == (3, True)
    # хвост чанка за завершением: событие предсказано, смена не исполнена
    assert first_event_call(a0g, ex2, 5, 20) == (3, False)
    # смена внутри первого чанка — вызов 1
    b = a0g.copy()
    b[0, 4:] = 1.0
    assert first_event_call(b, ex, 5, 40)[0] == 1
    # среда завершилась до вызова 3 — события нет
    assert first_event_call(a0g, ex, 5, 11) == (None, False)
    # ноль не меняет знак; начальный ноль в первом чанке
    c = np.full((3, 8), -1.0)
    c[1, :] = 0.0
    assert first_event_call(c, np.full(24, -1.0), 3, 24) == (None, False)
    d0 = np.full((2, 8), 1.0)
    d0[0, :2] = 0.0
    assert first_event_call(d0, np.full(16, 1.0), 2, 16) == (None, False)
    print("самопроверка k16_passport пройдена: устойчивое переключение по "
          "последовательным шагам, начальный ноль, маркер: предсказано и "
          "исполнено, хвост за завершением, ноль")
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
        k15h=passport_marker(["reports/k15h/run/p*/q0ref_t*_i*.json"], 150),
        k15g=passport_marker(["reports/k15g/local/run/p*/q0ref_t*_i*.json"],
                             40))
    for k, v in marker.items():
        if not v["complete"] or v["skipped_files"]:
            prob.append(f"маркер {k}: состояний {v['states']} из "
                        f"{v['expected_states']}, пропущено файлов "
                        f"{len(v['skipped_files'])}")
    res = dict(data=data, marker=marker, problems=prob)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 3 if prob else 0


if __name__ == "__main__":
    sys.exit(main())
