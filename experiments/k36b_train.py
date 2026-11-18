#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-36, этап B (§8): одно продолжение обучения T на каждый source seed.

Разрешено только при свежем решении A = passed_A_development
(k36_analyze.gate_a_problems). Источник продолжения — неизменяемый снимок
state.pt K-32 (k36_inputs.import_continuation): actor, critic, оба
оптимизатора; глобальные счётчики волн и обновлений продолжаются (keyed RNG
маршрута/шума — (space 0, seed, эпизод = волна·5 + j, вызов); порядок
minibatch — (32, seed, 3, update)), поэтому новый каталог не начинает
поток с эпизода 0. KL-reference — прежний равномерный/нулевой (k32_dist),
не финальный actor K-32. Старые буферы, validation и history K-32 не
импортируются в новое состояние (хранятся в разделе imported).
Тренер — k32_ppo.Trainer без изменений математики и CFG; отличия только в
пространстве имён: scope train_tplus_s<S> / val_tplus_s<S> журнала K-36,
eid волн k36/tplus/s<S>/wave<w>, каталог data/k36/runs/Tplus_s<S>.
Добавочный кап 200 000 физических шагов на сид; отметка mid — 100 000
ДОБАВЛЕННЫХ шагов (spent нового scope). Финальная модель — последний
принятый actor. Три отпечатка хранятся раздельно: source_weights_hash
(K-25), rl_init_hash (исходный init K-32), continuation_start_actor_hash.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k32_ppo as kppo  # noqa: E402
import k36_inputs as kin  # noqa: E402
import k36_protocol as pc  # noqa: E402

CFG = dict(kppo.CFG)            # без изменений: lr, KL, clip, cap 200 000
CODE_B = ("k36b_train.py", "k32_ppo.py", "k32_collect.py", "k32_policy.py",
          "k32_dist.py", "k32_model.py", "k32_operator.py")
REPORT = "reports/k36/train_tplus_s{seed}.json"


def report_path(seed):
    return REPORT.format(seed=seed)


def final_path(seed):
    return os.path.join(kin.run_dir(seed), "final.pt")


def code_b():
    return {f: pc.sha(os.path.join(HERE, f)) for f in CODE_B}


class ContTrainer(kppo.Trainer):
    """k32_ppo.Trainer в пространстве имён K-36 + импорт продолжения."""

    def __init__(self, torch, *, seed, H, actor, critic, make_arm, sim, io,
                 ledger, odir, contract, imported, cfg=CFG, log=print):
        super().__init__(torch, seed=seed, op="T", H=H, actor=actor,
                         critic=critic, make_arm=make_arm, sim=sim, io=io,
                         ledger=ledger, odir=odir, contract=contract, cfg=cfg,
                         log=log)
        self.scope = f"train_tplus_s{seed}"
        self.vscope = f"val_tplus_s{seed}"
        self.imported = imported
        if self.init_hash != imported["continuation_start_actor_hash"]:
            raise SystemExit("actor тренера ≠ началу продолжения")

    def wave_eid(self, w):
        return f"k36/tplus/s{self.seed}/wave{w}"

    def save(self):
        d = dict(actor=self.actor.state_dict(),
                 critic=self.critic.state_dict(),
                 opt_a=self.opt_a.state_dict(), opt_c=self.opt_c.state_dict(),
                 st=self.st, contract=self.contract,
                 actor_hash=self.actor_hash(), init_hash=self.init_hash,
                 imported=self.imported)
        self.torch.save(d, self.state_path + ".tmp")
        os.replace(self.state_path + ".tmp", self.state_path)

    def load(self):
        if not os.path.exists(self.state_path):
            return False
        d = self.torch.load(self.state_path, map_location="cpu",
                            weights_only=False)
        if d.get("imported") != self.imported:
            raise SystemExit(f"{self.state_path}: другой импорт продолжения")
        return super().load()

    def start_from_import(self, snap):
        """Первый запуск: оптимизаторы и счётчики из снимка K-32."""
        self.opt_a.load_state_dict(snap["opt_a"])
        self.opt_c.load_state_dict(snap["opt_c"])
        s = snap["st"]
        self.st = dict(wave=int(s["wave"]), update=int(s["update"]),
                       accepted=0, updates=[], val={}, used_waves=[],
                       discarded=[], finished=False, pending_val=None)
        self.save()

    def run(self, snap=None):
        if not os.path.exists(self.state_path):
            if snap is None:
                raise SystemExit("нет состояния продолжения и снимка импорта")
            self.start_from_import(snap)
        return super().run()


def _train_seed(torch, prep, L, sim, s, device, log=print, loader=None):
    """Область жизни источника s; наружу — только CPU-результаты."""
    import k32_model as km32
    import k32_policy as kpol
    import k32_protocol as pc32
    import k36_harness as kh
    v = kh.load_source(torch, s, device, loader)
    info = prep["k32"]["seeds"][str(s)]["T_state"]
    pth = prep["placement"][f"k32_T_state_s{s}"]["path"]
    imp = kin.import_continuation(torch, s, pth, info)
    snap = torch.load(imp["path"], map_location="cpu", weights_only=False)
    actor, critic = km32.build(torch, v["mod"].H, 6, s, pc32.load_norm(s))
    actor.load_state_dict(snap["actor"])
    critic.load_state_dict(snap["critic"])
    imported = dict(
        import_sha256=imp["sha256"], rl_init_hash=imp["rl_init_hash"],
        continuation_start_actor_hash=imp["continuation_start_actor_hash"],
        source_weights_hash=prep["k32"]["seeds"][str(s)]["source"][
            "weights_hash"],
        wave=int(snap["st"]["wave"]), update=int(snap["st"]["update"]),
        accepted_k32=int(snap["st"]["accepted"]),
        spent_imported=int(info["spent_imported"]),
        k32_val=json.loads(json.dumps(snap["st"]["val"], default=float)))

    def make_arm(act, stochastic):
        return kpol.Arm(torch, v["mod"], v["dec"], v["M"], kind="T", seed=s,
                        actor=act, stochastic=stochastic)
    contract = dict(k36_prepare=prep["sha1"],
                    k32_contract=prep["k32"]["k32_contract"], op="T",
                    seed=s, code=code_b(), cfg=CFG)
    t = ContTrainer(torch, seed=s, H=v["mod"].H, actor=actor, critic=critic,
                    make_arm=make_arm, sim=sim, io=v["io"], ledger=L,
                    odir=kin.run_dir(s), contract=contract, imported=imported,
                    log=log)
    st = t.run(snap)
    return finish_report(torch, t, st, s)


def finish_report(torch, t, st, s):
    fpth = final_path(s)
    ah = t.actor_hash()
    torch.save(dict(actor=t.actor.state_dict(), actor_hash=ah, op="T",
                    seed=s, contract=t.contract), fpth + ".tmp")
    os.replace(fpth + ".tmp", fpth)
    imp = t.imported
    rep = dict(
        contract=t.contract, seed=s, finished=bool(st["finished"]),
        final_sha1=pc.sha(fpth), actor_hash=ah,
        source_weights_hash=imp["source_weights_hash"],
        rl_init_hash=imp["rl_init_hash"],
        continuation_start_actor_hash=imp["continuation_start_actor_hash"],
        actor_changed=ah != imp["continuation_start_actor_hash"],
        spent_imported=imp["spent_imported"], spent_added=t.spent(),
        spent_total=imp["spent_imported"] + t.spent(),
        updates_imported=imp["update"],
        updates_added=int(st["update"]) - imp["update"],
        accepted_added=int(st["accepted"]),
        waves=dict(first=imp["wave"], next=int(st["wave"])),
        discarded=st["discarded"],
        continuation_val={f"continuation_{k}": v for k, v in
                          st["val"].items()},
        k32_val=imp["k32_val"],
        history=[{k: v for k, v in u.items() if k != "history"}
                 for u in st["updates"]])
    pc.save_json(report_path(s), rep)
    return rep


def train_problems(seed, prep):
    """Отчёт продолжения сида seed полон и принадлежит ИМЕННО этому сиду:
    seed/оператор/контракт/prepare, три исходных отпечатка = prepare этого
    сида, final.pt (SHA, seed, op, контракт, фактический tensor-hash actor)
    и состояние завершённого тренера (actor, импорт, finished)."""
    import torch
    p = report_path(seed)
    if not os.path.exists(p):
        return [f"нет {p}"]
    r = json.load(open(p, encoding="utf-8"))
    sd = prep["k32"]["seeds"][str(seed)]
    pr = []
    c = r.get("contract") or {}
    want_c = dict(k36_prepare=prep["sha1"],
                  k32_contract=prep["k32"]["k32_contract"], op="T",
                  seed=int(seed), code=code_b(),
                  cfg=json.loads(json.dumps(CFG)))
    if r.get("seed") != int(seed) or json.loads(json.dumps(c)) != want_c:
        pr.append("seed/оператор/контракт/код/prepare ≠ этому сиду")
    if r.get("finished") is not True:
        pr.append("продолжение не завершено")
    for k, v in (("rl_init_hash", sd["T_state"]["rl_init_hash"]),
                 ("continuation_start_actor_hash", sd["T"]["actor_hash"]),
                 ("source_weights_hash", sd["source"]["weights_hash"])):
        if r.get(k) != v:
            pr.append(f"{k} ≠ prepare сида {seed}")
    if not isinstance(r.get("spent_added"), int) or \
            not 0 < r["spent_added"] <= pc.CAPS[f"train_tplus_s{seed}"]:
        pr.append("добавленный расход вне (0, кап]")
    fp_ = final_path(seed)
    if not os.path.exists(fp_) or pc.sha(fp_) != r.get("final_sha1"):
        return pr + ["final.pt ≠ отчёту"]
    ck = torch.load(fp_, map_location="cpu", weights_only=False)
    ah = kppo.tensor_hash(ck["actor"])
    if ck.get("seed") != int(seed) or ck.get("op") != "T" or \
            json.loads(json.dumps(ck.get("contract"))) != want_c or \
            not (ah == ck.get("actor_hash") == r.get("actor_hash")):
        pr.append("final.pt: seed/op/контракт/фактический actor ≠ отчёту")
    stp = os.path.join(kin.run_dir(seed), "state.pt")
    if not os.path.exists(stp):
        return pr + ["нет состояния тренера продолжения"]
    st = torch.load(stp, map_location="cpu", weights_only=False)
    imp = st.get("imported") or {}
    if kppo.tensor_hash(st["actor"]) != ah or st.get("actor_hash") != ah or \
            (st.get("st") or {}).get("finished") is not True or \
            imp.get("rl_init_hash") != sd["T_state"]["rl_init_hash"] or \
            imp.get("continuation_start_actor_hash") != \
            sd["T"]["actor_hash"] or \
            json.loads(json.dumps(st.get("contract"))) != want_c:
        pr.append("состояние тренера ≠ final.pt / импорту этого сида")
    return pr


def continue_b(a):
    import k18_collector as k18c
    sim = k18c.Sim(1, "persistent")
    import torch
    import k36_harness as kh
    try:
        import k36b_gate as kg
        p = kg.gate_a_problems()
        if p:
            raise SystemExit("этап B не разрешён: " + "; ".join(p[:3]))
        prep = pc.load_prepare()
        pr = kg.pending_problems(pc.ledger(), kg.TRAIN_PENDING)
        if pr:
            raise SystemExit("; ".join(pr))
        pr = kh.resources_problems(torch, a.device)
        if pr:
            raise SystemExit("; ".join(pr))
        L = pc.ledger()
        for s in pc.parse_seeds():
            if not train_problems(s, prep):
                print(f"  Tplus_s{s}: продолжение завершено и закреплено")
                continue
            rep = _train_seed(torch, prep, L, sim, s, a.device,
                              log=lambda x: print(x, flush=True))
            kh.free(torch)
            print(f"  Tplus_s{s}: добавлено шагов {rep['spent_added']}, "
                  f"обновлений {rep['updates_added']} (принято "
                  f"{rep['accepted_added']}); actor изменился "
                  f"{rep['actor_changed']}", flush=True)
        return 0
    finally:
        sim.close()


# --- самопроверка ---------------------------------------------------------------------
def selftest():
    import copy
    import tempfile
    import torch
    import k21_model as km
    import k25_model as k25m
    import k29_model as k29m
    import k31_collect as kcol
    import k32_collect as kcol32
    import k32_model as km32
    import k32_policy as kpol
    k29m.no_tf32(torch)
    M = k29m.build(torch)
    Lb = np.array([-1.1, -2.0, -1.5, -0.5, -3.0, -1.0])
    Ub = np.array([4.0, 1.0, 2.5, 0.7, 3.0, 2.0])
    mod = k25m.build(torch)["K25"](km.fake_bar(torch), Lb, Ub, seed=0).eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p_ in list(mod.cond.parameters()) + list(
                mod.readout.parameters()):
            p_.add_(torch.randn(p_.shape, generator=g) * 0.05)
    for p_ in mod.parameters():
        p_.requires_grad_(False)
    dec = M["Decoder"](mod).freeze()
    H = mod.H
    norm = km32.fit_normalizer(np.random.default_rng(0).normal(
        size=(64, 6 * H + 72)), H)

    class FIO(kcol.FakeIO):
        def prep(self, obs, desc, task):
            x = float(obs["state"][0, 0])
            bt = km.fake_batch(torch, B=1, seed=int(abs(x) * 1e3) % 13)
            s8 = np.zeros((1, 8))
            s8[0, :3] = x
            return bt, 1, s8
    io = FIO()

    def make_arm(act, stochastic):
        return kpol.Arm(torch, mod, dec, M, kind="T", seed=0, actor=act,
                        stochastic=stochastic)
    caps1 = {"train_T_s0": 2 * kcol32.CYCLE_MAX + 50,
             "val_T_s0": kppo.VAL_CAP}
    cfg1 = dict(CFG, cap=caps1["train_T_s0"], mid_mark=10 ** 9)
    caps2 = dict(pc.CAPS, train_tplus_s0=2 * kcol32.CYCLE_MAX + 50)
    cfg2 = dict(CFG, cap=caps2["train_tplus_s0"], mid_mark=10 ** 9)
    with tempfile.TemporaryDirectory() as td:
        # этап K-32 (игрушечный) -> state.pt
        a0, c0 = km32.build(torch, H, 6, 0, norm)
        L1 = kcol.Ledger(os.path.join(td, "led32.json"), caps1)
        t1 = kppo.Trainer(torch, seed=0, op="T", H=H, actor=a0, critic=c0,
                          make_arm=make_arm, sim=kppo.ToySim(), io=io,
                          ledger=L1, odir=os.path.join(td, "k32"),
                          contract=dict(c=1), cfg=cfg1, log=lambda *x: None)
        st1 = t1.run()
        snap = torch.load(os.path.join(td, "k32", "state.pt"),
                          weights_only=False)
        assert st1["finished"] and st1["update"] >= 1

        def cont(odir, crash=None, mid=10 ** 9, imported=None):
            a_, c_ = km32.build(torch, H, 6, 0, norm)
            a_.load_state_dict(snap["actor"])
            c_.load_state_dict(snap["critic"])
            imp = imported or dict(
                rl_init_hash=snap["init_hash"],
                continuation_start_actor_hash=kppo.tensor_hash(
                    a_.state_dict()),
                wave=int(snap["st"]["wave"]),
                update=int(snap["st"]["update"]),
                spent_imported=L1.spent("train_T_s0"),
                source_weights_hash="W", k32_val={})
            L2 = kcol.Ledger(os.path.join(odir, "led36.json"), caps2)
            t = ContTrainer(torch, seed=0, H=H, actor=a_, critic=c_,
                            make_arm=make_arm, sim=kppo.ToySim(), io=io,
                            ledger=L2, odir=os.path.join(odir, "run"),
                            contract=dict(c=36), imported=imp,
                            cfg=dict(cfg2, mid_mark=mid),
                            log=lambda *x: None)
            t.crash_at = crash
            return t
        # непрерывное продолжение (эталон этапа)
        d0 = os.path.join(td, "ref")
        os.makedirs(d0)
        tr = cont(d0)
        st = tr.run(snap)
        h_ref, n_ref = tr.actor_hash(), st["update"]
        assert st["finished"] and n_ref > int(snap["st"]["update"])
        # счётчики продолжаются: первая новая волна = импортированной
        assert tr.st["used_waves"] and min(tr.st["used_waves"]) == \
            int(snap["st"]["wave"])
        # KL-reference — исходная равномерная/нулевая, не финальный actor:
        # KL_init перед первым обновлением продолжения > 0
        assert st["updates"][0]["kl_init_before"] > 0
        # прежняя validation не переносится; mid — по ДОБАВЛЕННЫМ шагам
        assert set(st["val"]) == {"start", "end"}
        # три отпечатка раздельно, init_hash K-32 не подменён
        assert tr.imported["rl_init_hash"] == snap["init_hash"] != \
            tr.init_hash
        assert tr.ledger.spent("train_tplus_s0") <= caps2["train_tplus_s0"]
        # расход K-32 не переносится в журнал K-36 как новый
        assert "train_T_s0" not in tr.ledger.d["spent"]
        # аварии: запись волны / commit / сохранение обновления -> тот же
        # actor, число обновлений и расход без аварийных списаний
        for crash in ("after_write", "after_commit", "after_update_save"):
            dc = os.path.join(td, crash)
            os.makedirs(dc)
            t = cont(dc, crash)
            try:
                t.run(snap)
                raise AssertionError(f"авария {crash} не сработала")
            except KeyboardInterrupt:
                pass
            t2 = cont(dc)
            s2 = t2.run(snap)
            assert t2.actor_hash() == h_ref and s2["update"] == n_ref, crash
            assert not t2.ledger.d["crashes"], crash
            assert t2.spent() == tr.spent(), crash
        # mid-валидация по добавленным шагам
        dm = os.path.join(td, "mid")
        os.makedirs(dm)
        sm = cont(dm, mid=1).run(snap)
        assert "mid" in sm["val"]
        # чужой импорт — отказ возобновления
        bad = dict(cont(d0).imported, rl_init_hash="другой")
        try:
            cont(d0, imported=bad).run(snap)
            raise AssertionError("чужой импорт принят")
        except SystemExit:
            pass
        # actor тренера ≠ началу продолжения — отказ конструктора
        try:
            cont(os.path.join(td, "x"), imported=dict(
                cont(d0).imported, continuation_start_actor_hash="x"))
            raise AssertionError("несоответствие начала принято")
        except SystemExit:
            pass
        # отчёт: final.pt, добавленный/импортированный/суммарный расход
        g = globals()
        keep = dict(REPORT=g["REPORT"])
        kr = kin.RUNS
        try:
            g["REPORT"] = os.path.join(td, "rep_s{seed}.json")
            kin.RUNS = os.path.join(td, "runs")
            os.makedirs(kin.run_dir(0), exist_ok=True)
            rep = finish_report(torch, tr, st, 0)
            assert rep["spent_total"] == rep["spent_imported"] + \
                rep["spent_added"] and rep["actor_changed"] and \
                rep["updates_added"] == n_ref - int(snap["st"]["update"])
            assert set(rep["continuation_val"]) == {"continuation_start",
                                                   "continuation_end"}
            # принадлежность сиду: отчёт/final.pt/состояние s0 на местах
            # s1 (SHA каждого файла сходится со своим отчётом) — отказ
            import shutil
            a_, c_ = km32.build(torch, H, 6, 0, norm)
            a_.load_state_dict(snap["actor"])
            c_.load_state_dict(snap["critic"])
            start = kppo.tensor_hash(a_.state_dict())
            prepf = dict(sha1="P", k32=dict(k32_contract="K", seeds={
                str(sd): dict(T_state=dict(rl_init_hash=snap["init_hash"]),
                              T=dict(actor_hash=start),
                              source=dict(weights_hash="W"))
                for sd in (0, 1)}))
            want_c = dict(k36_prepare="P", k32_contract="K", op="T", seed=0,
                          code=code_b(), cfg=CFG)
            imp = dict(cont(d0).imported, continuation_start_actor_hash=start,
                       rl_init_hash=snap["init_hash"])
            L3 = kcol.Ledger(os.path.join(td, "led_own.json"), caps2)
            t0_ = ContTrainer(torch, seed=0, H=H, actor=a_, critic=c_,
                              make_arm=make_arm, sim=kppo.ToySim(), io=io,
                              ledger=L3, odir=kin.run_dir(0),
                              contract=want_c, imported=imp, cfg=cfg2,
                              log=lambda *x: None)
            st0 = t0_.run(snap)
            finish_report(torch, t0_, st0, 0)
            assert train_problems(0, prepf) == [], train_problems(0, prepf)
            os.makedirs(kin.run_dir(1), exist_ok=True)
            for f in ("final.pt", "state.pt"):
                shutil.copyfile(os.path.join(kin.run_dir(0), f),
                                os.path.join(kin.run_dir(1), f))
            shutil.copyfile(report_path(0), report_path(1))
            assert train_problems(1, prepf), "actor s0 принят как s1"
            # фактический actor final.pt ≠ отчёту — отказ
            ck = torch.load(final_path(0), weights_only=False)
            ck["actor"] = {k: v + 1e-3 for k, v in ck["actor"].items()}
            torch.save(ck, final_path(0))
            rj = json.load(open(report_path(0)))
            rj["final_sha1"] = pc.sha(final_path(0))
            pc.save_json(report_path(0), rj)
            assert train_problems(0, prepf)
        finally:
            g.update(keep)
            kin.RUNS = kr
        assert copy
    print("самопроверка k36b_train пройдена: продолжение из state.pt K-32 — "
          "волны и обновления продолжают глобальные счётчики, KL-reference "
          "исходный, прежняя validation не переносится, mid — по "
          "добавленным шагам; аварии записи/commit/обновления дают тот же "
          "actor и расход без списаний; чужой импорт и несоответствие "
          "начала — отказ; отчёт разделяет imported/added/total и три "
          "отпечатка; отчёт/final.pt/состояние другого сида и подменённый "
          "actor — отказ")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="K-36 B: продолжение T")
    ap.add_argument("cmd", choices=("continue-b",))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    try:
        return continue_b(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
