#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36: манифест входов (§3.3) и проверка источников K-32 / K-25 / банка K-33.

Логические входы (таблица размещения: имя -> путь машины; SHA1 — формат
K-32, SHA256 — сверка переноса):
  k32_base, k32_norm_man, k32_norm_s<S>, k32_ledger;
  k32_<op>_final_s<S>, k32_<op>_report_s<S> (op = T, P), k32_T_state_s<S>;
  k25_ckpt_s<S>, k25_report_s<S> — из закреплённого sources[str(S)] K-32;
  bank_manifest, bank_check, bank_s<space>_t<t> — банк K-33 v2.
Проверки (до записи prepare и до любого резерва):
  - контракт K-32 = k32_protocol.contract() (код K-32 и нормировка не
    изменены); отчёт обучения: тот же контракт, finished, SHA final.pt;
  - actor final.pt: tensor-hash = отчёту = записанному в final.pt;
  - T: state.pt — контракт K-32 (op T, seed), finished, actor state.pt =
    actor final.pt побитово (tensor-hash), есть critic и оба оптимизатора;
  - K-25: SHA ckpt/отчёта = закреплённым в K-32;
  - банк: bank_id начинается с ddf3f626f0b5, файлы = манифесту, bank-check
    пройден для этого bank_id.
Импорт продолжения (§8.1) — import_continuation(): копия state.pt в
неизменяемый снимок data/k36/runs/Tplus_s<S>/import_state.pt (SHA), три
раздельных отпечатка: source_weights_hash (K-25), rl_init_hash (исходный
init_hash K-32) и continuation_start_actor_hash.
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SEEDS = (0, 1)
OPS = ("T", "P")
BANK_ID_PREFIX = "ddf3f626f0b5"
RUNS = "data/k36/runs"


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def logical_paths():
    """Имя -> путь (относительно корня репозитория либо абсолютный из
    закреплённых документов K-32)."""
    import k33_bank as kb
    P = dict(k32_base="reports/k32/base_prepare.json",
             k32_norm_man="reports/k32/normalizer_manifest.json",
             k32_ledger="data/k32/ledger.json",
             bank_manifest=kb.MANIFEST,
             bank_check="reports/k33/bank_check.json")
    for s in SEEDS:
        P[f"k32_norm_s{s}"] = f"data/k32/normalizer_s{s}.npz"
        P[f"k32_T_state_s{s}"] = f"data/k32/runs/T_s{s}/state.pt"
        for op in OPS:
            P[f"k32_{op}_final_s{s}"] = f"data/k32/runs/{op}_s{s}/final.pt"
            P[f"k32_{op}_report_s{s}"] = f"reports/k32/train_{op}_s{s}.json"
    if os.path.exists(P["k32_base"]):
        b = json.load(open(P["k32_base"], encoding="utf-8"))
        for s in SEEDS:
            e = b["sources"][str(s)]
            P[f"k25_ckpt_s{s}"] = e["ckpt"]
            P[f"k25_report_s{s}"] = e["report"]
    for space in (0, 1):
        for t in range(10):
            P[f"bank_s{space}_t{t}"] = kb.task_path(space, t)
    return P


def placement(P=None):
    P = P or logical_paths()
    out, miss = {}, []
    for k, p in sorted(P.items()):
        if not os.path.exists(p):
            miss.append(k)
            continue
        out[k] = dict(path=os.path.abspath(p), sha1=sha1(p),
                      sha256=sha256(p), bytes=os.path.getsize(p))
    if miss:
        raise SystemExit(f"нет входов K-36: {miss}")
    return out


def placement_problems(pl):
    p = []
    for k, v in pl.items():
        if not os.path.exists(v["path"]) or sha256(v["path"]) != v["sha256"]:
            p.append(f"вход {k}: файл изменён или отсутствует")
    return p


def k32_audit(torch, P=None):
    """Проверки K-32 и K-25 -> (сведения, проблемы). Ничего не пишет."""
    import k32_ppo as kppo
    import k32_protocol as pc32
    P = P or logical_paths()
    info, p = dict(seeds={}), []
    try:
        b = pc32.load_base()
        c = pc32.contract(b)
    except SystemExit as e:
        return info, [f"контракт K-32: {e.code}"]
    info["k32_contract"] = c
    led = json.load(open(P["k32_ledger"], encoding="utf-8"))
    for s in SEEDS:
        si = {}
        e = b["sources"][str(s)]
        if sha1(P[f"k25_ckpt_s{s}"]) != e["ckpt_sha1"] or \
                sha1(P[f"k25_report_s{s}"]) != e["report_sha1"]:
            p.append(f"s{s}: K-25 ckpt/отчёт ≠ закреплённым в K-32")
        si["source"] = dict(ckpt_sha1=e["ckpt_sha1"],
                            weights_hash=e["weights_hash"],
                            update=e["update"])
        for op in OPS:
            rj = json.load(open(P[f"k32_{op}_report_s{s}"],
                                encoding="utf-8"))
            fpth = P[f"k32_{op}_final_s{s}"]
            if rj.get("contract") != c or rj.get("finished") is not True or \
                    sha1(fpth) != rj.get("final_sha1") or \
                    rj.get("op") != op or int(rj.get("seed", -1)) != s:
                p.append(f"{op} s{s}: отчёт обучения K-32 ≠ final.pt / "
                         f"контракту")
                continue
            ck = torch.load(fpth, map_location="cpu", weights_only=False)
            ah = kppo.tensor_hash(ck["actor"])
            if not (ah == ck.get("actor_hash") == rj.get("actor_hash")) or \
                    ck.get("op") != op or int(ck.get("seed", -1)) != s or \
                    ck.get("contract") != c:
                p.append(f"{op} s{s}: actor final.pt ≠ отчёту")
            si[op] = dict(actor_hash=ah, final_sha1=rj["final_sha1"],
                          updates=rj.get("updates"),
                          accepted=rj.get("accepted"),
                          train_steps=rj.get("train_steps"))
            if op == "T":
                st = torch.load(P[f"k32_T_state_s{s}"], map_location="cpu",
                                weights_only=False)
                miss = [k for k in ("actor", "critic", "opt_a", "opt_c", "st",
                                    "init_hash", "actor_hash", "contract")
                        if k not in st]
                if miss:
                    p.append(f"T s{s}: state.pt неполон ({miss}) — "
                             f"blocked_missing_trainer_state")
                    continue
                sh = kppo.tensor_hash(st["actor"])
                if st["contract"] != dict(contract=c, op="T", seed=s):
                    p.append(f"T s{s}: контракт state.pt ≠ K-32")
                if not (sh == st["actor_hash"] == ah):
                    p.append(f"T s{s}: actor state.pt ≠ final.pt")
                if st["st"].get("finished") is not True:
                    p.append(f"T s{s}: state.pt K-32 не завершён")
                spent = int(led["spent"].get(f"train_T_s{s}", -1))
                if spent != int(rj.get("train_steps", -2)):
                    p.append(f"T s{s}: расход K-32 в журнале {spent} ≠ "
                             f"отчёту {rj.get('train_steps')}")
                si["T_state"] = dict(
                    actor_hash=sh, rl_init_hash=st["init_hash"],
                    wave=int(st["st"]["wave"]),
                    update=int(st["st"]["update"]),
                    accepted=int(st["st"]["accepted"]),
                    val_tags=sorted(st["st"]["val"]),
                    used_waves=len(st["st"]["used_waves"]),
                    spent_imported=spent,
                    critic_hash=kppo.tensor_hash(st["critic"]))
        info["seeds"][str(s)] = si
    return info, p


def bank_audit(P=None):
    import k33_bank as kb
    P = P or logical_paths()
    m = json.load(open(P["bank_manifest"], encoding="utf-8"))
    p = []
    if not str(m.get("bank_id", "")).startswith(BANK_ID_PREFIX):
        p.append(f"bank_id {m.get('bank_id')} ≠ {BANK_ID_PREFIX}…")
    bad = [f for f, s in m["files"].items()
           if not os.path.exists(f) or sha1(f) != s]
    if bad:
        p.append(f"файлы банка ≠ манифесту: {bad[:3]}")
    c = json.load(open(P["bank_check"], encoding="utf-8"))
    if c.get("bank_id") != m.get("bank_id") or c.get("ok") is not True:
        p.append("bank-check не пройден для этого банка")
    want = {kb.task_path(sp, t) for sp in (0, 1) for t in range(10)}
    if not want <= set(m["files"]):
        p.append("манифест не покрывает все файлы задач")
    return dict(bank_id=m.get("bank_id"), recipe=m.get("recipe")), p


# --- импорт продолжения ---------------------------------------------------------------
def run_dir(seed):
    return os.path.join(RUNS, f"Tplus_s{seed}")


def import_path(seed):
    return os.path.join(run_dir(seed), "import_state.pt")


def import_continuation(torch, seed, k32_state_path, want):
    """Неизменяемый снимок state.pt K-32 (копия + SHA). want — сведения
    аудита T_state. -> dict(путь, sha256, отпечатки). Повтор: снимок
    обязан совпасть."""
    import k32_ppo as kppo
    dst = import_path(seed)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_sha = sha256(k32_state_path)
    if os.path.exists(dst):
        if sha256(dst) != src_sha:
            raise SystemExit(f"{dst}: снимок импорта ≠ state.pt K-32")
    else:
        shutil.copyfile(k32_state_path, dst + ".tmp")
        if sha256(dst + ".tmp") != src_sha:
            raise SystemExit("копия state.pt повреждена")
        os.replace(dst + ".tmp", dst)
    st = torch.load(dst, map_location="cpu", weights_only=False)
    ah = kppo.tensor_hash(st["actor"])
    if ah != want["actor_hash"] or st["init_hash"] != want["rl_init_hash"]:
        raise SystemExit("снимок импорта: actor / init_hash ≠ аудиту")
    return dict(path=os.path.abspath(dst), sha256=src_sha,
                rl_init_hash=st["init_hash"],
                continuation_start_actor_hash=ah)


def selftest():
    import tempfile
    import torch
    import k32_ppo as kppo
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "a.bin")
        open(f, "wb").write(b"abc")
        pl = {"x": dict(path=f, sha1=sha1(f), sha256=sha256(f), bytes=3)}
        assert placement_problems(pl) == []
        open(f, "wb").write(b"abd")
        assert placement_problems(pl)
        os.remove(f)
        assert placement_problems(pl)
        # импорт: копия неизменяема, повтор совпадает, порча — отказ
        g = globals()
        keep = g["RUNS"]
        try:
            g["RUNS"] = os.path.join(td, "runs")
            lin = torch.nn.Linear(3, 2)
            st = dict(actor=lin.state_dict(), init_hash="I0",
                      critic={}, opt_a={}, opt_c={}, st={})
            sp = os.path.join(td, "state.pt")
            torch.save(st, sp)
            want = dict(actor_hash=kppo.tensor_hash(lin.state_dict()),
                        rl_init_hash="I0")
            r1 = import_continuation(torch, 0, sp, want)
            r2 = import_continuation(torch, 0, sp, want)
            assert r1 == r2 and r1["rl_init_hash"] == "I0" and \
                r1["continuation_start_actor_hash"] == want["actor_hash"]
            try:
                import_continuation(torch, 0, sp, dict(want,
                                                       rl_init_hash="X"))
                raise AssertionError("чужой init_hash принят")
            except SystemExit:
                pass
            torch.save(dict(st, init_hash="I1"), sp)     # state.pt изменён
            try:
                import_continuation(torch, 0, sp, want)
                raise AssertionError("изменённый источник принят")
            except SystemExit:
                pass
        finally:
            g["RUNS"] = keep
    print("самопроверка k36_inputs пройдена: размещение — подмена/пропажа "
          "файла отвергаются; снимок импорта неизменяем, повтор совпадает, "
          "чужой init_hash и изменённый state.pt — отказ")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; запуск — через k36_protocol.py")
