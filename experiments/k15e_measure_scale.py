#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15e M0: анализ раскаток семейства a0 + α·δ на всех задачах.

РУКИ: q0 (канонический черновик), a000, a025, a050, a075, a100 (масштабы
поправки h18) и зеркальный контроль am050, am100 (та же поправка против
направления, то же число масштабов).

ГЕЙТЫ — ВСЕ FAIL-CLOSED (любое нарушение — код 3, решения нет):
  * сид раскатки ровно 101 (`--allow-partial` его не снимает);
  * каждая метка исполняет своё (q0 — depthrvq/fast; aXXX — k15e с этим
    α, фаза h1p1, 18 слоёв);
  * все семь рук K-15e имеют ОДИН контракт (`ALPHA_CONTRACT`): тот же
    чекпойнт h18, выбранная точка, модули, статистика, замороженное,
    гейты, кодек — иначе масштабировалась бы не одна и та же δ;
  * ТОЖДЕСТВА: a000 побитово равен q0 и a100 побитово равен прежней руке
    h18 K-15d на КАЖДОМ эпизоде. Сравнение парное: та же (задача,
    состояние, сид), те же init_hash/init_hash_full/rollout_seed,
    противоречивые дубликаты эталона — отказ, непокрытый эпизод — отказ.
    Прежняя рука h18 обязана иметь тот же контракт модели
    (`H18_CONTRACT`);
  * отпечаток npz действий пересчитывается.

ЧТО СЧИТАЕТСЯ (равный вес задач, кластер = задача × состояние):
  * успех каждой руки, rescue/harm каждого α против q0;
  * оракул best-of-set: dir5 = {0, .25, .5, .75, 1}; dir3 = {0, .5, 1} и
    ctrl3 = {0, −.5, −1} — равные по размеру наборы по направлению δ и
    зеркально;
  * разность dir3 − ctrl3 с 90 % интервалом. Это зеркальный контроль: он
    снимает выигрыш от одного лишь числа кандидатов, но не обязан быть
    точной мерой «чистого разнообразия» — нелинейная динамика и
    параметризация схвата могут делать направления несимметричными;
  * гистограмма лучшего α (наименьший |α| среди успешных).

РЕШЕНИЕ О ДОРОГОМ ДАТАСЕТЕ (зарегистрировано до роллаутов):
  перспективно  выигрыш оракула dir5 >= 5 п.п. И dir3 − ctrl3 >= 2 п.п.;
  закрыть       выигрыш dir5 <= 2 п.п. ИЛИ dir3 − ctrl3 <= 0;
  иначе         расширить probe до 10 состояний на задачу.
Это решение о сборе данных, а не критерий эффективности метода.

КОДЫ: 0 — набор исправен; 3 — технический отказ.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR5 = ("a000", "a025", "a050", "a075", "a100")
DIR3 = ("a000", "a050", "a100")
CTRL3 = ("a000", "am050", "am100")
ALL_ALPHA = DIR5 + ("am050", "am100")
EXPECTED_SEEDS = [101]
DECISION = dict(promising_gain=0.05, promising_dir_minus_ctrl=0.02,
                close_gain=0.02, close_dir_minus_ctrl=0.0,
                registered="07.10.2026, до роллаутов M0")
# Контракт, общий для ВСЕХ рук K-15e: одна и та же δ.
ALPHA_CONTRACT = ("checkpoint_sha1", "selected_level_sha1",
                  "refine_module_sha1", "policy_module_sha1",
                  "k15e_policy_sha1", "candidates_module_sha1",
                  "inference_report_sha1", "stats_sha1",
                  "frozen_content_sha", "joint_sha1", "k15d_gate_run_id",
                  "code_version", "codec", "k15a_gate")
# Контракт МОДЕЛИ, общий с прежней рукой h18 K-15d. Отпечаток отчёта
# проверки вывода сюда не входит: повторный запуск проверки переписывает
# файл (время, длительность), не меняя исполняемого вычисления; модель и
# код задаются полями ниже.
H18_CONTRACT = ("checkpoint_sha1", "selected_level_sha1",
                "refine_module_sha1", "policy_module_sha1", "stats_sha1",
                "frozen_content_sha", "joint_sha1", "k15d_gate_run_id",
                "code_version", "codec", "k15a_gate")
START_FIELDS = ("init_hash", "init_hash_full", "rollout_seed")
ACTION_SHA_RE = re.compile(r"^[0-9a-f]{16}$")


def present(v):
    """Поле контракта задано: непустая строка, непустой словарь, не None."""
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (dict, list)):
        return len(v) > 0
    return True


def check_episodes(paths, seeds=EXPECTED_SEEDS):
    """Каждый эпизод: хеш действий формата k9h, сид раскатки = сиду
    заголовка = зарегистрированному; npz действий связан с JSON — тот же
    порядок init_state_id и те же action_sha1."""
    bad = []
    for p in paths:
        d = json.load(open(p))
        name = os.path.basename(p)
        if int(d.get("seed", -1)) not in seeds:
            bad.append(f"{name}: сид заголовка {d.get('seed')}")
        for e in d.get("episodes", []):
            if not ACTION_SHA_RE.match(str(e.get("action_sha1") or "")):
                bad.append(f"{name}: эпизод {e.get('init_state_id')} без "
                           f"хеша действий формата k9h")
            if e.get("rollout_seed") is None or \
                    int(e["rollout_seed"]) != int(d.get("seed", -1)):
                bad.append(f"{name}: эпизод {e.get('init_state_id')}: сид "
                           f"раскатки {e.get('rollout_seed')} не равен "
                           f"сиду заголовка {d.get('seed')}")
        npz = d.get("actions_npz")
        if not npz:
            continue        # отсутствие npz ловит check_actions_npz
        f = os.path.join(os.path.dirname(p), npz)
        if not os.path.exists(f):
            continue
        try:
            with np.load(f, allow_pickle=True) as z:
                sha = [str(x) for x in z["action_sha1"]]
                ids = [int(x) for x in z["init_state_id"]]
        except Exception as ex:          # noqa: BLE001 — любой сбой = отказ
            bad.append(f"{name}: npz не читается ({ex})")
            continue
        if sha != [str(e.get("action_sha1")) for e in d["episodes"]] or \
                ids != [int(e["init_state_id"]) for e in d["episodes"]]:
            bad.append(f"{name}: npz действий не совпадает с эпизодами JSON")
    return bad


def check_arm(d, arm):
    """Метка исполняет своё: q0 — depthrvq/fast, aXXX — k15e с этим α."""
    import k15e_candidates as kc
    j = d.get("joint") or {}
    if arm == "q0":
        ok = (d.get("policy") == "depthrvq" and d.get("levels") == 1
              and (d.get("argv") or {}).get("depth_rvq_mode") == "fast")
        return [] if ok else [f"q0: не depthrvq/fast ({d.get('policy')})"]
    want = kc.label_alpha(arm)
    bad = []
    if d.get("policy") != "k15e":
        bad.append(f"{arm}: policy {d.get('policy')!r}")
    if j.get("alpha") is None or abs(float(j["alpha"]) - want) > 1e-12:
        bad.append(f"{arm}: α {j.get('alpha')!r}, ожидалось {want}")
    if j.get("layers_per_call") != 18 or j.get("phase") != "h1p1":
        bad.append(f"{arm}: не h18 ({j.get('phase')}, "
                   f"{j.get('layers_per_call')} слоёв)")
    return bad


def contract_of(d, fields):
    j = d.get("joint") or {}
    return {k: j.get(k) for k in fields}


def check_contracts(contracts, fields):
    """contracts: {метка: [контракт по каждому артефакту]} -> проблемы."""
    p, seen = [], set()
    for arm, lst in contracts.items():
        for c in lst:
            miss = [k for k in fields if not present(c.get(k))]
            if miss:
                p.append(f"{arm}: в контракте нет {miss}")
            seen.add(json.dumps(c, sort_keys=True, ensure_ascii=False))
    if len(seen) > 1:
        p.append(f"руки K-15e исполняют разные h18: {len(seen)} разных "
                 f"контрактов")
    return p


def check_h18_ref(d):
    j = d.get("joint") or {}
    ok = (d.get("policy") == "k15d" and j.get("phase") == "h1p1"
          and j.get("layers_per_call") == 18 and j.get("level") == "1")
    return [] if ok else [f"эталон h18: не рука h18 K-15d ({d.get('policy')},"
                          f" {j.get('phase')}, {j.get('layers_per_call')})"]


def episodes_of(files):
    """[(ключ (задача, состояние, сид), эпизод, путь)]."""
    out = []
    for p in files:
        d = json.load(open(p))
        for e in d["episodes"]:
            out.append(((int(d["task_id"]), int(e["init_state_id"]),
                         int(d["seed"])), e, p))
    return out


def paired_identity(cur_files, ref_files):
    """Побитовое равенство действий на КАЖДОМ эпизоде текущей руки.

    Пара — та же (задача, состояние, сид) с теми же стартами. Эталон с
    противоречивыми дубликатами, непокрытый эпизод или иной старт — отказ.
    """
    ref, problems = {}, []
    for key, e, p in episodes_of(ref_files):
        sig = tuple(str(e.get(k)) for k in ("action_sha1",) + START_FIELDS)
        if key in ref and ref[key] != sig:
            problems.append(f"эталон: противоречивые дубликаты {key}")
        ref[key] = sig
    cur = episodes_of(cur_files)
    keys = [k for k, _e, _p in cur]
    if len(set(keys)) != len(keys):
        problems.append("текущая рука: повторяющиеся эпизоды")
    compared = identical = 0
    for key, e, _p in cur:
        if key not in ref:
            problems.append(f"эпизод {key} не покрыт эталоном")
            continue
        r = ref[key]
        if tuple(str(e.get(k)) for k in START_FIELDS) != r[1:]:
            problems.append(f"эпизод {key}: другой старт или сид раскатки")
            continue
        compared += 1
        identical += int(str(e.get("action_sha1")) == r[0])
    if not cur:
        problems.append("нет эпизодов для сравнения")
    if compared and identical != compared:
        problems.append(f"действия различаются в {compared - identical} из "
                        f"{compared} эпизодов")
    return dict(episodes=len(cur), compared=compared, identical=identical,
                problems=sorted(set(problems))[:10],
                passed=not problems and compared == len(cur) > 0)


def oracle(sr, arms):
    return np.max(np.stack([sr[a] for a in arms]), axis=0)


def decide(gain5, dir_minus_ctrl, rule=DECISION):
    if gain5 <= rule["close_gain"] or dir_minus_ctrl <= \
            rule["close_dir_minus_ctrl"]:
        return "закрыть одномерное масштабирование"
    if gain5 >= rule["promising_gain"] and dir_minus_ctrl >= \
            rule["promising_dir_minus_ctrl"]:
        return "перспективно: строить replay gate и пилот датасета"
    return "неясно: расширить probe до 10 состояний на задачу"


def best_alpha(sr, clusters_n):
    import k15e_candidates as kc
    order = sorted(DIR5, key=lambda a: abs(kc.label_alpha(a)))
    hist, none = {a: 0 for a in DIR5}, 0
    for i in range(clusters_n):
        hit = [a for a in order if sr[a][i] > 0.5]
        if hit:
            hist[hit[0]] += 1
        else:
            none += 1
    return hist, none


def analyze(arts, h18_arts, tasks, states, allow_partial=False,
            n_boot=None):
    """Весь анализ M0. Возвращает сводку; код в ней — 0 или 3."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import k14q_behavior as kb
    from k15d_behavior import check_actions_npz
    obs, meta = kb.load(arts)
    arms = ["q0"] + [x for x in ALL_ALPHA if x in meta]
    if set(meta) != set(arms):
        raise SystemExit(f"посторонние руки: {sorted(set(meta) - set(arms))}")
    technical = []
    missing = [x for x in ALL_ALPHA if x not in meta]
    if missing and not allow_partial:
        technical.append(f"нет рук {missing}")
    contracts = {}
    for nm in arms:
        for p in meta[nm]["files"]:
            d = json.load(open(p))
            technical += [f"{os.path.basename(p)}: {x}"
                          for x in check_arm(d, nm)]
            if str(d.get("suite")) != "10":
                technical.append(f"{os.path.basename(p)}: не LIBERO-10")
            if nm != "q0":
                contracts.setdefault(nm, []).append(
                    contract_of(d, ALPHA_CONTRACT))
        technical += check_actions_npz(meta[nm]["files"])
        technical += check_episodes(meta[nm]["files"])
        if len(meta[nm]["fingerprints"]) != 1:
            technical.append(f"у руки {nm} несколько моделей")
    technical += check_contracts(contracts, ALPHA_CONTRACT)
    shas = set()
    for nm in arms:
        shas |= set(meta[nm]["script_shas"])
    if len(shas) > 1:
        technical.append(f"разные версии харнесса: {sorted(shas)}")
    clusters, seeds = kb.align(obs, arms)
    # СИД ФИКСИРОВАН РЕГИСТРАЦИЕЙ; --allow-partial его не снимает.
    if seeds != EXPECTED_SEEDS:
        technical.append(f"сиды {seeds}, зарегистрирован {EXPECTED_SEEDS}")
    want = sorted((int(t), int(s)) for t in tasks for s in states)
    if clusters != want and not allow_partial:
        technical.append(f"кластеров {len(clusters)} из {len(want)}")

    # --- ТОЖДЕСТВА — ГЕЙТЫ ----------------------------------------------
    ident = {}
    if "a000" in meta:
        ident["a000_vs_q0"] = paired_identity(meta["a000"]["files"],
                                              meta["q0"]["files"])
    else:
        technical.append("нет руки a000 — тождество с q0 не проверено")
    if "a100" in meta:
        ref_ok = bool(h18_arts)
        # ЭТАЛОН ПРОВЕРЯЕТСЯ ТАК ЖЕ, КАК ТЕКУЩИЕ РУКИ: npz, хеши, сиды.
        technical += [f"эталон h18: {x}" for x in check_actions_npz(h18_arts)]
        technical += [f"эталон h18: {x}" for x in check_episodes(h18_arts)]
        h18_contracts = []
        for p in h18_arts:
            d = json.load(open(p))
            technical += [f"{os.path.basename(p)}: {x}"
                          for x in check_h18_ref(d)]
            h18_contracts.append(contract_of(d, H18_CONTRACT))
        if ref_ok:
            a100c = [{k: c[k] for k in H18_CONTRACT}
                     for c in contracts.get("a100", [])]
            technical += [f"h18 K-15d против a100: {x}" for x in
                          check_contracts({"h18": h18_contracts,
                                           "a100": a100c}, H18_CONTRACT)]
        ident["a100_vs_h18_k15d"] = paired_identity(meta["a100"]["files"],
                                                    h18_arts)
    else:
        technical.append("нет руки a100 — тождество с h18 не проверено")
    for k, v in ident.items():
        if not v["passed"]:
            technical.append(f"тождество {k} нарушено: {v['identical']}/"
                             f"{v['episodes']} эпизодов; {v['problems']}")

    sr = {nm: kb.cluster_means(obs, nm, clusters, seeds) for nm in arms}
    base = sr["q0"]
    vals = {f"sr_{nm}": sr[nm] for nm in arms}
    for nm in arms[1:]:
        res, hrm = kb.pair_rates(obs, "q0", nm, clusters, seeds)
        vals[f"rescue_{nm}"], vals[f"harm_{nm}"] = res, hrm
    sets = {"dir5": [x for x in DIR5 if x in sr],
            "dir3": [x for x in DIR3 if x in sr],
            "ctrl3": [x for x in CTRL3 if x in sr]}
    for k, v in sets.items():
        if v:
            vals[f"oracle_{k}"] = oracle(sr, v)
            vals[f"gain_{k}"] = vals[f"oracle_{k}"] - base
    if sets["dir3"] == list(DIR3) and sets["ctrl3"] == list(CTRL3):
        vals["dir3_minus_ctrl3"] = vals["oracle_dir3"] - vals["oracle_ctrl3"]
    ci, _ = kb.boot(vals, clusters, n=int(n_boot or kb.N_BOOT),
                    qs=(kb.Q_LO, kb.Q_HI))
    point = {k: kb.point(v, clusters) for k, v in vals.items()}
    per_task = {}
    for t, idx in kb.strata(clusters).items():
        per_task[int(t)] = {k: float(np.asarray(v)[idx].mean())
                            for k, v in vals.items()
                            if k.startswith(("sr_", "oracle_"))}
    hist, none = (best_alpha(sr, len(clusters))
                  if sets["dir5"] == list(DIR5) else ({}, None))
    gain5, dmc = point.get("gain_dir5"), point.get("dir3_minus_ctrl3")
    if technical:
        decision = "НЕ ПРИНИМАЕТСЯ: технический отказ"
    elif gain5 is None or dmc is None:
        decision = "не определено: неполный набор рук"
    else:
        decision = decide(gain5, dmc)
    return dict(kind="k15e_m0_scale", technical=technical,
                code=3 if technical else 0, decision=decision,
                decision_rule=DECISION, point=point, ci90=ci,
                per_task=per_task, best_alpha_hist=hist,
                no_success_clusters=none, identity=ident, arms=arms,
                clusters=len(clusters), seeds=seeds,
                script_sha1=sorted(shas), partial=bool(allow_partial),
                fingerprints={nm: sorted(meta[nm]["fingerprints"])
                              for nm in arms})


def report(out):
    p = out["point"]
    print(f"  кластеров {out['clusters']}, сиды {out['seeds']}, руки "
          f"{out['arms']}")
    for k, v in out["identity"].items():
        print(f"  тождество {k}: {v['identical']}/{v['episodes']} эпизодов"
              + ("" if v["passed"] else f" — НАРУШЕНО {v['problems']}"))
    print("  успех: " + ", ".join(f"{nm} {p['sr_' + nm]:.3f}"
                                  for nm in out["arms"]))
    for nm in out["arms"][1:]:
        print(f"    {nm}: rescue {p['rescue_' + nm]:.3f}, harm "
              f"{p['harm_' + nm]:.3f}")
    for k in ("dir5", "dir3", "ctrl3"):
        if f"gain_{k}" in p:
            c = out["ci90"][f"gain_{k}"]
            print(f"  оракул {k} {p['oracle_' + k]:.3f}, выигрыш к q0 "
                  f"{p['gain_' + k]:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]")
    if "dir3_minus_ctrl3" in p:
        c = out["ci90"]["dir3_minus_ctrl3"]
        print(f"  НАПРАВЛЕНИЕ (зеркальный контроль): dir3 − ctrl3 = "
              f"{p['dir3_minus_ctrl3']:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]")
    print(f"  лучший α: {out['best_alpha_hist']}, без успеха "
          f"{out['no_success_clusters']}")
    print("  по задачам: " + "; ".join(
        f"{t}: q0 {v['sr_q0']:.2f}, dir5 "
        f"{v.get('oracle_dir5', float('nan')):.2f}"
        for t, v in out["per_task"].items()))
    if out["technical"]:
        print(f"  ТЕХНИЧЕСКИЙ ОТКАЗ: {out['technical'][:8]}")
    print(f"  РЕШЕНИЕ M0: {out['decision']} (код {out['code']})")


def main():
    ap = argparse.ArgumentParser(description="K-15e M0: анализ масштабов")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arts", nargs="*", default=[])
    ap.add_argument("--h18-arts", nargs="*", default=[])
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--states", default="0,1,2,3,4")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.arts or not a.out:
        raise SystemExit("нужны --arts и --out")
    out = analyze(a.arts, a.h18_arts,
                  [int(x) for x in a.tasks.split(",")],
                  [int(x) for x in a.states.split(",")], a.allow_partial)
    report(out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False)
    os.replace(tmp, a.out)
    print(f"  сводка: {a.out}")
    return int(out["code"])


# --- самопроверка: синтетические артефакты формата k9h --------------------
def _fake_set(td, tasks=(0, 8), mutate=None):
    """Полный набор восьми рук и эталон h18 с npz. mutate(арм, t, d) -> d."""
    import hashlib
    import k15e_candidates as kc
    rng = np.random.default_rng(0)
    cont = dict(checkpoint_sha1="C", selected_level_sha1="L",
                refine_module_sha1="R", policy_module_sha1="P",
                k15e_policy_sha1="E", candidates_module_sha1="K",
                inference_report_sha1="I", stats_sha1="S",
                frozen_content_sha="F", joint_sha1="J",
                k15d_gate_run_id="G", code_version={"x": 1},
                codec={"c": 1}, k15a_gate={"g": 1})
    arts, h18 = [], []
    for arm in ("q0",) + ALL_ALPHA + ("h18",):
        for t in tasks:
            eps = []
            for s in range(5):
                tag = ("q" if arm in ("q0", "a000") else
                       "h" if arm in ("a100", "h18") else arm)
                ah = hashlib.sha1(f"{tag}{t}{s}".encode()).hexdigest()[:16]
                eps.append(dict(success=bool(rng.random() < 0.7),
                                init_state_id=s, env_index=s,
                                init_hash=f"i{t}{s}", init_hash_full=f"I{t}{s}",
                                rollout_seed=101, action_sha1=ah))
            d = dict(episodes=eps, arm_label=arm, task_id=t, init_start=0,
                     seed=101, n_envs=5, suite="10", horizon=8,
                     max_steps=600, waiting_steps=10, ensemble="off",
                     rollout_seed_mode="fixed", ckpt="X",
                     script_sha1="K9H", levels=1)
            if arm == "q0":
                d.update(policy="depthrvq", argv=dict(depth_rvq_mode="fast"),
                         joint=dict(model_fingerprint="fq0"))
            elif arm == "h18":
                d.update(policy="k15d", script_sha1="OLD", joint=dict(
                    cont, phase="h1p1", layers_per_call=18, level="1",
                    model_fingerprint="fh18", inference_report_sha1="OLDREP"))
            else:
                d.update(policy="k15e", joint=dict(
                    cont, alpha=kc.label_alpha(arm), phase="h1p1",
                    layers_per_call=18, model_fingerprint=f"f{arm}"))
            if mutate:
                d = mutate(arm, t, d)
            base = os.path.join(td, f"{arm}_t{t}_i0")
            np.savez(base + ".actions.npz", actions=np.zeros(2),
                     action_sha1=np.asarray(
                         [str(e.get("action_sha1")) for e in d["episodes"]],
                         dtype="U64"),
                     init_state_id=np.asarray(
                         [e["init_state_id"] for e in d["episodes"]]))
            d["actions_npz"] = os.path.basename(base) + ".actions.npz"
            d["actions_npz_sha1"] = hashlib.sha1(open(
                base + ".actions.npz", "rb").read()).hexdigest()[:12]
            json.dump(d, open(base + ".json", "w"))
            (h18 if arm == "h18" else arts).append(base + ".json")
    return arts, h18


def selftest():
    import tempfile
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    assert decide(0.06, 0.03).startswith("перспективно")
    assert decide(0.06, 0.0).startswith("закрыть")
    assert decide(0.02, 0.05).startswith("закрыть")
    assert decide(0.04, 0.01).startswith("неясно")

    def run(mutate=None, h18_drop=False, h18_extra=None):
        with tempfile.TemporaryDirectory() as td:
            arts, h18 = _fake_set(td, mutate=mutate)
            if h18_drop:
                h18 = []
            if h18_extra:
                h18 = h18 + h18_extra(td)
            return analyze(arts, h18, [0, 8], range(5), n_boot=200)

    ok = run()
    assert ok["code"] == 0, ok["technical"]
    assert all(v["passed"] for v in ok["identity"].values())

    def sha(arm_, val):
        def m(arm, t, d):
            if arm == arm_:
                d["episodes"][0]["action_sha1"] = val
            return d
        return m

    def joint(arm_, k, v):
        def m(arm, t, d):
            if arm == arm_:
                d["joint"] = dict(d["joint"], **{k: v})
            return d
        return m

    def start(arm_):
        def m(arm, t, d):
            if arm == arm_:
                d["episodes"][1]["init_hash"] = "ИНОЙ"
            return d
        return m

    def dup(td):
        """Противоречивый дубликат эталона h18."""
        src = os.path.join(td, "h18_t0_i0.json")
        d = json.load(open(src))
        d["episodes"][0]["action_sha1"] = "e" * 16
        p = os.path.join(td, "h18dup_t0_i0.json")
        json.dump(d, open(p, "w"))
        return [p]

    # мутация -> (аргументы, фрагмент ПРИЧИНЫ отказа): тест требует, чтобы
    # поймала именно нужная проверка, а не любая другая
    cases = {
        "a000 != q0": (dict(mutate=sha("a000", "f" * 16)), "тождество a000"),
        "a100 != h18": (dict(mutate=sha("a100", "f" * 16)), "тождество a100"),
        "другие старты у эталона h18": (dict(mutate=start("h18")),
                                        "другой старт"),
        "нет эталона h18": (dict(h18_drop=True), "не покрыт эталоном"),
        "противоречивый дубликат h18": (dict(h18_extra=dup),
                                        "противоречивые дубликаты"),
        "у одной α другой чекпойнт": (dict(
            mutate=joint("a050", "checkpoint_sha1", "ДРУГОЙ")),
            "разные h18"),
        "у одной α нет поля контракта": (dict(
            mutate=joint("am100", "stats_sha1", None)), "в контракте нет"),
        "эталон h18 с другим чекпойнтом": (dict(
            mutate=joint("h18", "checkpoint_sha1", "ДРУГОЙ")),
            "h18 K-15d против a100"),
        "эталон — не h18": (dict(mutate=joint("h18", "layers_per_call", 24)),
                            "не рука h18"),
    }

    def all_seed(arm, t, d):
        d["seed"] = 999
        for e in d["episodes"]:
            e["rollout_seed"] = 999
        return d
    cases["сид 999"] = (dict(mutate=all_seed), "сиды")

    def no_sha(arms_):
        def m(arm, t, d):
            if arm in arms_:
                for e in d["episodes"]:
                    e.pop("action_sha1", None)
            return d
        return m

    def ep_seed(arm, t, d):
        for e in d["episodes"]:
            e["rollout_seed"] = 999          # заголовок остаётся 101
        return d

    def empty_contract(arm, t, d):
        if arm not in ("q0", "h18"):
            d["joint"] = dict(d["joint"], k15e_policy_sha1="",
                              candidates_module_sha1=" ",
                              inference_report_sha1="")
        return d

    def empty_dict(arm, t, d):
        if arm not in ("q0", "h18"):
            d["joint"] = dict(d["joint"], codec={})
        return d
    cases["нет action_sha1 у q0/a000/a100/h18"] = (
        dict(mutate=no_sha(("q0", "a000", "a100", "h18"))),
        "без хеша действий")
    cases["сид раскатки 999 при заголовке 101"] = (dict(mutate=ep_seed),
                                                    "сид раскатки")
    cases["пустые строки контракта"] = (dict(mutate=empty_contract),
                                        "в контракте нет")
    cases["пустой словарь кодека"] = (dict(mutate=empty_dict),
                                      "в контракте нет")
    for why, (kw, reason) in cases.items():
        r = run(**kw)
        assert r["code"] == 3, f"мутация «{why}» не поймана"
        assert r["decision"].startswith("НЕ ПРИНИМАЕТСЯ"), why
        assert any(reason in x for x in r["technical"]), \
            f"мутация «{why}» поймана не той проверкой: {r['technical']}"
    # подменённый npz
    with tempfile.TemporaryDirectory() as td:
        arts, h18 = _fake_set(td)
        np.savez(os.path.join(td, "a025_t0_i0.actions.npz"),
                 actions=np.ones(3))
        r = analyze(arts, h18, [0, 8], range(5), n_boot=200)
        assert r["code"] == 3, "подменённый npz не пойман"
        assert any("отпечаток" in x for x in r["technical"])
    # повреждённый npz ЭТАЛОНА h18
    with tempfile.TemporaryDirectory() as td:
        arts, h18 = _fake_set(td)
        np.savez(os.path.join(td, "h18_t0_i0.actions.npz"),
                 actions=np.ones(3))
        r = analyze(arts, h18, [0, 8], range(5), n_boot=200)
        assert r["code"] == 3 and any("эталон h18" in x and "отпечаток" in x
                                      for x in r["technical"]), \
            "повреждённый npz эталона не пойман"
    # npz с верным отпечатком, но другими хешами действий, чем в JSON
    with tempfile.TemporaryDirectory() as td:
        arts, h18 = _fake_set(td)
        import hashlib
        f = os.path.join(td, "a050_t8_i0.actions.npz")
        np.savez(f, actions=np.zeros(2),
                 action_sha1=np.asarray(["0" * 16] * 5),
                 init_state_id=np.arange(5))
        j = os.path.join(td, "a050_t8_i0.json")
        d = json.load(open(j))
        d["actions_npz_sha1"] = hashlib.sha1(open(f, "rb").read()
                                             ).hexdigest()[:12]
        json.dump(d, open(j, "w"))
        r = analyze(arts, h18, [0, 8], range(5), n_boot=200)
        assert r["code"] == 3 and any("не совпадает с эпизодами" in x
                                      for x in r["technical"]), \
            "npz, не связанный с JSON, не пойман"
    # --allow-partial не снимает сид и тождества
    with tempfile.TemporaryDirectory() as td:
        arts, h18 = _fake_set(td, mutate=all_seed)
        r = analyze(arts, h18, [0, 8], range(5), allow_partial=True,
                    n_boot=200)
        assert r["code"] == 3
    print(f"самопроверка k15e_measure_scale пройдена: {len(cases) + 4} "
          f"мутаций пойманы")
    return 0


if __name__ == "__main__":
    sys.exit(main())
