#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-18: обучение адаптивного базиса поправок по успеху (TD3), оценки SR.

ЭТАПЫ (один процесс; возобновление на границе завершённой волны):
  precheck  q0 0 -> 8 -> 0 (старты 40–44) против холодных эталонов K-17
            (действия, успех, шаг завершения, хеш старта) в постоянных
            воркерах; при несовпадении — та же проверка в режиме "fresh"
            (процессы воркеров заново на каждую волну); оба не прошли —
            технический отказ. Затем нулевая политика = q0 побитово на
            обучающих стартах задач 0 и 8.
  q0set     общий набор q0 (по 5 обучающих стартов каждой задачи, нулевая
            политика): переходы копируются в три replay; стоимость входит в
            номинальный бюджет каждого варианта (собран один раз).
  smoke     по одной обучающей волне каждого варианта с обновлениями.
  train1    волны по кругу (вариант с наименьшим числом команд) до 100 тыс.
            команд живых сред у каждого.
  eval1     промежуточная оценка (старты 25–29) каждого варианта без шума и
            обновлений + q0 (сверка с холодными эталонами K-15h); отчёт SR.
  train2    до 300 тыс. команд.
  eval2     промежуточная (25–29) и итоговая (30–44) оценки каждого
            варианта и q0 (сверка с эталонами K-15h и K-17).
После каждой обучающей волны: 4 обновления критиков на новый переход,
актёр — каждые 2 обновления критиков после 64 собственных эпизодов
варианта; новая версия поведения публикуется между волнами.

Бюджет — команды живых сред (ожидание, префикс q0, окно, продолжение q0);
команды завершённых сред, которые векторная обёртка продолжает
исполнять, и затраты на оценки учитываются отдельно.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CONFIG = "data/k18/config.json"
STATE = "data/k18/run/state.pt"
WAVES_LOG = "logs/k18/waves.jsonl"
EVAL_DIR = "reports/k18/eval"
STAGES = ("precheck", "q0set", "smoke", "train1", "eval1", "train2",
          "eval2", "done")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def config_problems(cfg):
    p = []
    if cfg.get("kind") != "k18_config":
        return ["не конфигурация K-18"]
    for k in ("pca", "enc"):
        f = cfg[k]["path"]
        if not os.path.exists(f) or sha(f) != cfg[k]["sha1"]:
            p.append(f"{k} изменён или отсутствует")
    inp = cfg["inputs"]
    for k in ("windows", "k16_models", "k17_models", "harness",
              "harness_utils"):
        if not os.path.exists(inp[k]) or sha(inp[k]) != inp[k + "_sha1"]:
            p.append(f"вход {k} изменён")
    for t, v in cfg["libero_bank"].items():
        if not os.path.exists(v["path"]) or sha(v["path"]) != v["sha1"]:
            p.append(f"файл стартов задачи {t} изменён")
    for k, v in cfg["cold_refs"].items():
        if not os.path.exists(v["path"]) or sha(v["path"]) != v["sha1"]:
            p.append(f"холодный эталон {k} изменён")
    for f, s_ in cfg["code"].items():
        fp = os.path.join(HERE, f)
        if not os.path.exists(fp) or sha(fp) != s_:
            p.append(f"код {f} изменён после подготовки")
    tr = set(cfg["starts"]["train"])
    for wv in cfg["q0_set"] + cfg["schedule"]:
        if not set(wv["starts"]) <= tr:
            p.append("в расписании обучения есть отложенный старт")
            break
    if cfg.get("bounds") is None:
        p.append("пределы контроллера не измерены (prepare --check-env)")
    return p


def ref_problems(eps, ref_path):
    rd = json.load(open(ref_path))
    by = {int(e["init_state_id"]): e for e in rd["episodes"]}
    p = []
    for e in eps:
        r = by.get(int(e["init_state_id"]))
        if r is None:
            p.append(f"старт {e['init_state_id']}: нет в эталоне")
            continue
        for f in ("action_sha1", "success", "done_step", "init_hash"):
            if str(e[f]) != str(r.get(f)):
                p.append(f"старт {e['init_state_id']}: {f} {e[f]} ≠ "
                         f"эталон {r.get(f)}")
    return p


class Runner:
    def __init__(self, a, cfg, torch):
        import k18_collector as kc
        self.a, self.cfg, self.torch, self.kc = a, cfg, torch, kc
        self.env = cfg["env"]
        self.scale = np.asarray(cfg["scale"])
        self.rho = float(cfg["rho"])
        self.state = None
        self.pending = []

    # --- состояние --------------------------------------------------------------
    def new_state(self):
        return dict(stage="precheck", worker_mode=None, variants={
            v: dict(ptr=0, commands=0, after_done=0, episodes=0, waves=0,
                    transitions=0, calls=0, seconds=0.0, updates=[])
            for v in self.cfg["variants"]}, q0set=None, evals={},
            precheck={}, episode_counter=0, eval_commands=0,
            config_sha1=sha(self.a.config))

    def save(self):
        st = dict(self.state)
        st["agents"] = {v: ag.state() for v, ag in self.agents.items()}
        st["replays"] = {v: r.state() for v, r in self.replays.items()}
        os.makedirs(os.path.dirname(self.a.state), exist_ok=True)
        self.torch.save(st, self.a.state + ".tmp")
        os.replace(self.a.state + ".tmp", self.a.state)
        # зеркало журнала волн — только после фиксации состояния; анализ
        # берёт записи из состояния (wave_log), не из зеркала
        for rec in self.pending:
            os.makedirs(os.path.dirname(WAVES_LOG), exist_ok=True)
            with open(WAVES_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=float)
                         + "\n")
        self.pending = []

    def load_or_new(self):
        import k18_td3 as td
        U = np.load(self.cfg["pca"]["path"])["U"]
        b = self.cfg["bounds"]
        td3 = dict(self.cfg["td3"], **{k: self.cfg["noise"][k] for k in
                                       ("target_sigma", "target_cap")})
        self.agents = {v: td.Agent(self.torch, v, U, self.rho, self.scale,
                                   b["lo"], b["hi"], td3, self.rl_dev,
                                   seed=self.cfg["seeds"]["train"],
                                   batch_seed=self.cfg["seeds"]["batch"])
                       for v in self.cfg["variants"]}
        self.replays = {v: td.Replay(td3["capacity"])
                        for v in self.cfg["variants"]}
        if os.path.exists(self.a.state):
            st = self.torch.load(self.a.state, map_location="cpu",
                                 weights_only=False)
            if st["config_sha1"] != sha(self.a.config):
                raise SystemExit("состояние от другой конфигурации")
            for v in self.cfg["variants"]:
                self.agents[v].load(st["agents"][v])
                self.replays[v].load(st["replays"][v])
            self.state = {k: v for k, v in st.items()
                          if k not in ("agents", "replays")}
            print(f"  возобновление: этап {self.state['stage']}", flush=True)
        else:
            self.state = self.new_state()

    # --- волны --------------------------------------------------------------------
    def wave(self, task, starts, decide, visits=None):
        return self.kc.run_wave(self.sim, self.vla, self.enc, task, starts,
                                decide, scale=self.scale, rho=self.rho,
                                env=self.env, visits=visits)

    def check_build(self):
        """Сборка q0 закрепляется при первом запуске и сверяется при каждом
        возобновлении до новой волны."""
        meta = dict(self.vla.meta)
        got = self.state.get("vla_meta")
        if got is None:
            self.state["vla_meta"] = meta
            self.state["vla_build"] = self.vla.build_id
            self.save()
            print(f"  сборка q0 закреплена: {self.vla.build_id}", flush=True)
        elif got != meta or self.state["vla_build"] != self.vla.build_id:
            diff = sorted(k for k in set(got) | set(meta)
                          if got.get(k) != meta.get(k))
            raise SystemExit(f"сборка q0 изменилась после начала обучения: "
                             f"{diff}")
        else:
            print(f"  сборка q0 сверена: {self.vla.build_id}", flush=True)

    def zero_decide(self, ob, keys):
        m = len(ob)
        return dict(delta=np.zeros((m, 48)), coef=np.zeros((m, 4)),
                    noise_len=np.zeros(m), f=np.ones(m))

    def agent_decide(self, v, task, noise, diag=None):
        import k18_models as m18
        import k18_td3 as td
        ag, nz = self.agents[v], self.cfg["noise"]
        U = self.agents[v].actor.U.cpu().numpy()

        def f(ob, keys):
            noise_np, lens = None, np.zeros(len(ob))
            if noise:
                rows = [td.key_noise(self.cfg["seeds"]["explore"], task, s,
                                     vis, pos, ag.k, nz["explore_sigma"],
                                     nz["explore_cap"])
                        for s, vis, pos in keys]
                noise_np = np.stack([r[0] for r in rows])
                lens = np.array([r[1] for r in rows])
            o = ag.act(ob, noise_np)
            if diag is not None:
                for j in range(len(ob)):
                    ang = (m18.principal_angles(o["B"][j], U).tolist()
                           if o["B"] is not None and v == "basis4_rl"
                           else None)
                    diag.append(dict(pos=keys[j][2], coef=o["c"][j].tolist(),
                                     norm=float(np.linalg.norm(
                                         o["delta"][j])),
                                     f=float(o["f"][j]), angles=ang))
            return dict(delta=o["delta"], coef=o["c"], noise_len=lens,
                        f=o["f"])
        return f

    def log_wave(self, rec):
        """Запись волны с устойчивым идентификатором; фиксируется вместе с
        состоянием (save)."""
        self.state.setdefault("wave_log", []).append(rec)
        self.pending.append(rec)

    def add_transitions(self, v, trs, version):
        import k18_models as m18
        if not trs:
            return
        ag = self.agents[v]
        k = ag.k
        self.replays[v].add(
            obs=np.stack([t["obs"] for t in trs]),
            delta=np.stack([t["delta"] for t in trs]),
            rew=np.array([t["rew"] for t in trs]),
            next=np.stack([t["next"] for t in trs]),
            done=np.array([t["done"] for t in trs]),
            episode=np.array([t["episode"] for t in trs]),
            pos=np.array([t["pos"] for t in trs]),
            version=np.full(len(trs), version),
            task=np.array([t["task"] for t in trs]),
            coef=np.stack([np.resize(np.asarray(t["coef"], np.float32),
                                     k) for t in trs]),
            k=np.full(len(trs), k),
            exec_mask=np.stack([t["exec_mask"] for t in trs]))
        for t in trs:
            if not (np.isfinite(t["obs"]).all() and np.isfinite(
                    t["delta"]).all()):
                raise SystemExit("нечисловой переход")
            if np.linalg.norm(t["delta"]) > self.rho * (1 + 1e-6):
                raise SystemExit("поправка в replay выше ρ")
            assert t["delta"].shape == (m18.D_EXEC,)

    def tag_episodes(self, trs, eps):
        ids = {}
        for e in eps:
            self.state["episode_counter"] += 1
            ids[e["init_state_id"]] = self.state["episode_counter"]
        for t in trs:
            t["episode"] = ids[t["start"]]

    # --- этапы ----------------------------------------------------------------------
    def precheck(self):
        refs = self.cfg["cold_refs"]
        for mode in ("persistent", "fresh"):
            if self.sim.mode != mode:
                self.sim.close()
                self.sim = self.kc.Sim(self.cfg["n_envs"], mode)
            bad = []
            for t in (0, 8, 0):
                eps, _d, _l, c = self.wave(t, list(range(40, 45)), None)
                self.state["eval_commands"] += c["commands_total"]
                pr = ref_problems(eps, refs[f"{t}:40"]["path"])
                print(f"  проверка холодного старта ({mode}): задача {t}, "
                      f"старты 40–44 — "
                      + ("совпадает с эталоном" if not pr else
                         f"РАСХОЖДЕНИЕ: {pr[:2]}"), flush=True)
                bad += pr
            self.state["precheck"][mode] = bad[:10]
            if not bad:
                self.state["worker_mode"] = mode
                break
        if self.state["worker_mode"] is None:
            raise SystemExit("ни постоянные, ни свежие воркеры не "
                             "воспроизвели холодный q0")
        for wv in self.cfg["q0_set"]:
            if wv["task"] not in (0, 8):
                continue
            e1, _d, _l, c1 = self.wave(wv["task"], wv["starts"], None)
            e2, d2, _l2, c2 = self.wave(wv["task"], wv["starts"],
                                        self.zero_decide)
            self.state["eval_commands"] += c1["commands_total"] + \
                c2["commands_total"]
            same = [a["action_sha1"] == b["action_sha1"] and
                    a["success"] == b["success"] for a, b in zip(e1, e2)]
            print(f"  нулевая политика = q0 (задача {wv['task']}, старты "
                  f"{wv['starts']}): {sum(same)}/{len(same)}, решений окна "
                  f"{sum(len(x) for x in d2)}", flush=True)
            if not all(same):
                raise SystemExit("нулевая политика не воспроизвела q0")
        self.state["stage"] = "q0set"
        self.save()

    def q0set(self):
        acc = self.state.get("q0set_acc") or dict(done=[], trs=[], live=0,
                                                  after=0, eps=[])
        for i, wv in enumerate(self.cfg["q0_set"]):
            if i in acc["done"]:
                continue
            eps, dec, _l, c = self.wave(wv["task"], wv["starts"],
                                        self.zero_decide, wv["visit"])
            trs = self.kc.transitions(eps, dec, wv["task"])
            self.tag_episodes(trs, eps)
            acc["trs"] += trs
            acc["live"] += c["commands_live"]
            acc["after"] += c["commands_after_done"]
            acc["eps"] += eps
            acc["done"].append(i)
            self.state["q0set_acc"] = acc
            print(f"  q0-набор: задача {wv['task']}, успех "
                  f"{sum(e['success'] for e in eps)}/5, переходов "
                  f"{len(trs)}, {c['seconds']:.0f} с", flush=True)
            self.save()
        for v in self.cfg["variants"]:
            self.add_transitions(v, acc["trs"], 0)
            s = self.state["variants"][v]
            s["commands"] += acc["live"]
            s["after_done"] += acc["after"]
            s["transitions"] += len(acc["trs"])
        self.state["q0set"] = dict(
            commands_live=acc["live"], commands_after_done=acc["after"],
            transitions=len(acc["trs"]), episodes=len(acc["eps"]),
            success=sum(e["success"] for e in acc["eps"]),
            note="собран физически один раз; входит в номинальный бюджет "
                 "каждого варианта")
        self.state.pop("q0set_acc", None)
        self.state["stage"] = "smoke"
        self.save()

    def train_wave(self, v):
        s = self.state["variants"][v]
        wv = self.cfg["schedule"][s["ptr"]]
        if not set(wv["starts"]) <= set(self.cfg["starts"]["train"]):
            raise SystemExit("обучающая волна с отложенным стартом")
        ag = self.agents[v]
        eps, dec, _l, c = self.wave(wv["task"], wv["starts"],
                                    self.agent_decide(v, wv["task"], True),
                                    wv["visit"])
        trs = self.kc.transitions(eps, dec, wv["task"])
        self.tag_episodes(trs, eps)
        self.add_transitions(v, trs, ag.version)
        s["episodes"] += len(eps)
        actor_on = s["episodes"] >= self.cfg["td3"]["init_noise_episodes"]
        t0 = time.time()
        rec = ag.update(self.replays[v], self.cfg["td3"]["utd"] * len(trs),
                        actor_on)
        ag.version += 1
        s["ptr"] += 1
        s["waves"] += 1
        s["commands"] += c["commands_live"]
        s["after_done"] += c["commands_after_done"]
        s["transitions"] += len(trs)
        s["calls"] += c["calls"]
        s["seconds"] += c["seconds"]
        upd = dict(n=len(rec), td=float(np.mean([r["td"] for r in rec]))
                   if rec else None,
                   q=float(np.mean([r["q"] for r in rec])) if rec else None,
                   actor=sum("actor_q" in r for r in rec))
        s["updates"].append(upd)
        nrm = [float(np.linalg.norm(t["delta"])) for t in trs]
        self.log_wave(dict(
            wave_id=f"{v}:{s['waves']}", vla_build=self.vla.build_id,
            variant=v, wave=s["waves"], ptr=s["ptr"] - 1, task=wv["task"],
            starts=wv["starts"], version=ag.version - 1,
            success=[e["success"] for e in eps],
            windows=sum(len(x) > 0 for x in dec), transitions=len(trs),
            commands_live=c["commands_live"],
            commands_after_done=c["commands_after_done"], calls=c["calls"],
            seconds=c["seconds"], update_seconds=time.time() - t0,
            delta_norm=dict(median=float(np.median(nrm)) if nrm else None,
                            max=float(np.max(nrm)) if nrm else None),
            f_min=float(min([t["f"] for t in trs], default=1.0)),
            noise_len_mean=float(np.mean([t["noise_len"] for t in trs]))
            if trs else None, updates=upd, actor_on=actor_on,
            budget=s["commands"]))
        return eps, c

    def forecast(self, target):
        vs = self.state["variants"].values()
        live = sum(s["commands"] for s in vs)
        sec = sum(s["seconds"] for s in vs)
        waves = sum(s["waves"] for s in vs)
        if not sec:
            return
        rate = sum(s["commands"] - (self.state["q0set"] or {}).get(
            "commands_live", 0) for s in vs) / sec
        left = sum(max(0, target - s["commands"]) for s in vs)
        wave_s = sec / max(waves, 1)
        n_eval = 3 * 10 + 10 + (3 * 40 + 30 if target >= 300_000 else 0)
        print(f"  скорость: {rate:.1f} команд живых сред/с, волна "
              f"{wave_s / 60:.1f} мин; до {target // 1000} тыс.: "
              f"{left / max(rate, 1e-9) / 3600:.1f} ч обучения + ~"
              f"{n_eval * wave_s / 3600:.1f} ч оценок (всего команд "
              f"обучения {live})", flush=True)

    def train_to(self, target, stage_next):
        while True:
            vs = [v for v in self.cfg["variants"]
                  if self.state["variants"][v]["commands"] < target]
            if not vs:
                break
            v = min(vs, key=lambda x: (self.state["variants"][x]["commands"],
                                       self.cfg["variants"].index(x)))
            eps, c = self.train_wave(v)
            s = self.state["variants"][v]
            print(f"  {v}: волна {s['waves']}, задача "
                  f"{self.cfg['schedule'][s['ptr'] - 1]['task']}, успех "
                  f"{sum(e['success'] for e in eps)}/5, команд "
                  f"{s['commands']}, переходов в replay "
                  f"{self.replays[v].n}, {c['seconds']:.0f} с", flush=True)
            self.save()
            if sum(x["waves"] for x in self.state["variants"].values()) \
                    % 6 == 3:
                self.forecast(self.cfg["budget"]["final"])
        for v, s in self.state["variants"].items():
            print(f"  {v}: {s['commands']} команд живых сред (сверх отметки "
                  f"{s['commands'] - target}), эпизодов {s['episodes']}, "
                  f"переходов {s['transitions']}, вызовов VLM {s['calls']}, "
                  f"после завершения сред {s['after_done']}", flush=True)
        self.state["stage"] = stage_next
        self.save()

    def smoke(self):
        for v in self.cfg["variants"]:
            if self.state["variants"][v]["waves"] >= 1:
                continue
            eps, c = self.train_wave(v)
            u = self.state["variants"][v]["updates"][-1]
            if u["n"] and not np.isfinite(u["td"]):
                raise SystemExit(f"smoke {v}: нечисловые обновления")
            print(f"  smoke {v}: успех {sum(e['success'] for e in eps)}/5, "
                  f"обновлений критиков {u['n']}, {c['seconds']:.0f} с",
                  flush=True)
            self.save()
        self.forecast(self.cfg["budget"]["final"])
        self.state["stage"] = "train1"
        self.save()

    def eval_bank(self, label, mark, bank):
        """label: вариант или q0; bank: mid или final."""
        key = f"{label}@{mark}@{bank}"
        ev = self.state["evals"].setdefault(key, dict(done=[], eps=[],
                                                      diag=[]))
        starts = self.cfg["starts"][bank]
        blocks = [starts[i:i + 5] for i in range(0, len(starts), 5)]
        for t in self.cfg["tasks"]:
            for blk in blocks:
                tag = f"{t}:{blk[0]}"
                if tag in ev["done"]:
                    continue
                diag = []
                if label != "q0":
                    pin = self.state["ckpts"][f"{label}@{mark}"]
                    if self.actor_sha(label) != pin["actor_sha1"]:
                        raise SystemExit(f"оцениваемые веса {label} ≠ "
                                         f"закреплённому чекпойнту {mark}")
                dcd = None if label == "q0" else self.agent_decide(
                    label, t, False, diag)
                eps, dec, _l, c = self.wave(t, blk, dcd)
                self.state["eval_commands"] += c["commands_total"]
                if label == "q0":
                    pr = ref_problems(eps, self.cfg["cold_refs"][tag]["path"])
                    if pr:
                        raise SystemExit(f"q0 оценки {tag} ≠ холодный "
                                         f"эталон: {pr[:3]}")
                for i, e in enumerate(eps):
                    e["decisions"] = len(dec[i])
                    e["decisions_executed"] = sum(
                        self.kc.reached(e["done_step"], d["call"])
                        for d in dec[i])
                ev["eps"] += eps
                ev["diag"] += diag
                ev["done"].append(tag)
                self.save()
        succ = sum(e["success"] for e in ev["eps"])
        print(f"  оценка {label} ({mark}, старты {starts[0]}–{starts[-1]}): "
              f"SR {100 * succ / len(ev['eps']):.1f}% ({succ}/"
              f"{len(ev['eps'])})", flush=True)
        os.makedirs(EVAL_DIR, exist_ok=True)
        with open(os.path.join(EVAL_DIR, f"{label}_{mark}_{bank}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(dict(label=label, mark=mark, bank=bank,
                           config_sha1=sha(self.a.config),
                           vla_build=self.vla.build_id,
                           ckpt=None if label == "q0" else
                           self.state["ckpts"][f"{label}@{mark}"],
                           episodes=ev["eps"], diag=ev["diag"],
                           version=None if label == "q0" else
                           self.agents[label].version), fh,
                      ensure_ascii=False, default=float)
        return ev

    def actor_sha(self, v):
        import k18_prepare as kp
        return kp.state_sha(self.agents[v].actor.state_dict())

    def checkpoint(self, mark):
        """Чекпойнт оцениваемых весов: временный файл + os.replace;
        существующий загружается и сверяется (вариант, отметка,
        конфигурация, сборка q0, содержимое весов = текущим)."""
        import k18_prepare as kp
        cdir = getattr(self.a, "ckpt_dir", None) or "data/k18/ckpt"
        os.makedirs(cdir, exist_ok=True)
        cs = sha(self.a.config)
        for v, ag in self.agents.items():
            p = os.path.join(cdir, f"{v}_{mark}.pt")
            cur = self.actor_sha(v)
            if os.path.exists(p):
                try:
                    ck = self.torch.load(p, map_location="cpu",
                                         weights_only=False)
                except Exception as e:                       # noqa: BLE001
                    raise SystemExit(f"чекпойнт {p} не читается: {e}")
                prob = [k for k, want in (
                    ("variant", v), ("mark", mark), ("config_sha1", cs),
                    ("vla_build", self.vla.build_id), ("actor_sha1", cur))
                    if ck.get(k) != want]
                if kp.state_sha(ck.get("actor", {})) != cur:
                    prob.append("содержимое весов")
                if prob:
                    raise SystemExit(f"чекпойнт {p} не совпадает: {prob}")
            else:
                self.torch.save(dict(actor=ag.actor.state_dict(),
                                     version=ag.version, variant=v,
                                     mark=mark, config_sha1=cs,
                                     vla_build=self.vla.build_id,
                                     actor_sha1=cur,
                                     commands=self.state["variants"][v][
                                         "commands"]), p + ".tmp")
                os.replace(p + ".tmp", p)
            self.state.setdefault("ckpts", {})[f"{v}@{mark}"] = dict(
                path=os.path.abspath(p), sha1=sha(p), actor_sha1=cur)
        self.save()

    def report(self, mark, banks):
        rows = {}
        for lab in ["q0"] + list(self.cfg["variants"]):
            for bank in banks:
                key = f"{lab}@{'q0' if lab == 'q0' else mark}@{bank}"
                ev = self.state["evals"].get(key)
                if not ev:
                    continue
                pt = {}
                for e in ev["eps"]:
                    x = pt.setdefault(str(e["task"]), [0, 0])
                    x[0] += e["success"]
                    x[1] += 1
                sr = np.mean([v[0] / v[1] for v in pt.values()])
                rows[f"{lab}:{bank}"] = dict(sr_equal_weight=float(sr),
                                             per_task=pt)
                print(f"    {lab:10} {bank:5} SR {100 * sr:5.1f}%  " +
                      " ".join(f"t{t} {v[0]}/{v[1]}" for t, v in pt.items()),
                      flush=True)
        out = os.path.join("reports/k18", f"sr_{mark}.json")
        os.makedirs("reports/k18", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(dict(mark=mark, rows=rows, variants=self.state[
                "variants"], q0set=self.state["q0set"],
                eval_commands=self.state["eval_commands"],
                worker_mode=self.state["worker_mode"],
                ckpts=self.state.get("ckpts")), fh, ensure_ascii=False,
                indent=1, default=float)
        print(f"  отчёт SR: {out}", flush=True)

    def eval1(self):
        self.checkpoint("100k")
        for v in self.cfg["variants"]:
            self.eval_bank(v, "100k", "mid")
        self.eval_bank("q0", "q0", "mid")
        print("  === SR после 100 тыс. команд (старты 25–29) ===", flush=True)
        self.report("100k", ["mid"])
        self.state["stage"] = "train2"
        self.save()

    def eval2(self):
        self.checkpoint("300k")
        for v in self.cfg["variants"]:
            self.eval_bank(v, "300k", "mid")
        for v in self.cfg["variants"]:
            self.eval_bank(v, "300k", "final")
        self.eval_bank("q0", "q0", "final")
        print("  === SR после 300 тыс. команд ===", flush=True)
        self.report("300k", ["mid", "final"])
        self.state["stage"] = "done"
        self.save()

    def run(self):
        import k18_collector as kc
        # СРЕДЫ ДО CUDA: воркеры создаются раньше модели и малых сетей
        self.sim = kc.Sim(self.cfg["n_envs"], "persistent")
        self.rl_dev = self.torch.device(self.a.rl_device)
        self.load_or_new()
        if self.state.get("worker_mode") and \
                self.state["worker_mode"] != "persistent":
            self.sim.close()
            self.sim = kc.Sim(self.cfg["n_envs"], self.state["worker_mode"])
        self.vla = kc.VLA(self.a.vla_device)
        self.enc = kc.Encoder(self.cfg["enc"]["path"], self.rl_dev,
                              self.cfg["enc"]["state_sha1"])
        self.check_build()
        if self.state["stage"] == "precheck":
            self.precheck()
        if self.state["stage"] == "q0set":
            self.q0set()
        if self.state["stage"] == "smoke":
            self.smoke()
        if self.state["stage"] == "train1":
            self.train_to(self.cfg["budget"]["first"], "eval1")
        if self.state["stage"] == "eval1":
            self.eval1()
        if self.state["stage"] == "train2":
            self.train_to(self.cfg["budget"]["final"], "eval2")
        if self.state["stage"] == "eval2":
            self.eval2()
        self.sim.close()
        return 0


def main():
    ap = argparse.ArgumentParser(description="K-18: обучение и оценка")
    ap.add_argument("mode", choices=("run", "selftest", "check-config"))
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--vla-device", default="cuda:1")
    ap.add_argument("--rl-device", default="cuda:0")
    ap.add_argument("--ckpt-dir", default="data/k18/ckpt")
    ap.add_argument("--stop-after", default=None, choices=STAGES)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    cfg = json.load(open(a.config))
    prob = config_problems(cfg)
    if prob:
        print("  ОТКАЗ: " + "; ".join(prob[:6]))
        return 3
    if a.mode == "check-config":
        print("  конфигурация K-18 сверена")
        return 0
    import torch
    try:
        return Runner(a, cfg, torch).run()
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


# --- самопроверка: полный цикл на игрушечных среде и модели ----------------------
def selftest():
    import tempfile
    import torch
    import k18_collector as kc
    import k18_prepare as kp
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((48, 4)))
    q0_set, sched = kp.make_schedule(n=60)
    with tempfile.TemporaryDirectory() as td:
        np.savez(os.path.join(td, "pca.npz"), U=U)
        cfg = dict(kind="k18_config", variants=list(kp.VARIANTS),
                   tasks=[0, 1], n_envs=5,
                   starts=dict(train=list(kp.TRAIN_STARTS),
                               mid=list(kp.MID_STARTS),
                               final=list(kp.FINAL_STARTS)),
                   rho=0.7068, scale=kc.FakeVLA().max_act_q.tolist(),
                   bounds=dict(lo=[-1] * 6, hi=[1] * 6), seeds=kp.SEEDS,
                   budget=dict(first=12000, final=20000),
                   td3=dict(kp.TD3, critic_start=16, init_noise_episodes=10),
                   noise=kp.NOISE,
                   env=dict(kp.ENV, max_steps=600),
                   q0_set=[w for w in q0_set if w["task"] in (0, 1)],
                   schedule=[w for w in sched if w["task"] in (0, 1)],
                   pca=dict(path=os.path.join(td, "pca.npz")),
                   cold_refs={})
        cfgp = os.path.join(td, "config.json")
        json.dump(cfg, open(cfgp, "w"))
        a = argparse.Namespace(config=cfgp, state=os.path.join(td, "s.pt"),
                               rl_device="cpu", vla_device="cpu",
                               ckpt_dir=os.path.join(td, "ckpt"))
        global WAVES_LOG, EVAL_DIR
        WAVES_LOG = os.path.join(td, "waves.jsonl")
        EVAL_DIR = os.path.join(td, "eval")
        r = Runner(a, cfg, torch)
        r.sim, r.vla, r.rl_dev = kc.FakeSim(5), kc.FakeVLA(), "cpu"
        r.enc = lambda h: np.zeros((len(h), 512), np.float32)
        r.load_or_new()
        r.check_build()
        r.state["stage"] = "q0set"
        r.q0set()
        n0 = {v: rr.n for v, rr in r.replays.items()}
        assert len(set(n0.values())) == 1 and list(n0.values())[0] > 0
        r.smoke()
        r.train_to(cfg["budget"]["first"], "eval1")
        for v, s in r.state["variants"].items():
            assert s["commands"] >= cfg["budget"]["first"], (v, s)
        # возобновление: загрузка воспроизводит следующую волну побитово
        r.save()
        st_before = {v: dict(r.state["variants"][v]) for v in cfg["variants"]}
        eps1, _c = r.train_wave("basis4_rl")
        upd1 = r.state["variants"]["basis4_rl"]["updates"][-1]
        r2 = Runner(a, cfg, torch)
        r2.sim, r2.vla, r2.rl_dev = kc.FakeSim(5), kc.FakeVLA(), "cpu"
        r2.enc = r.enc
        r2.load_or_new()
        r2.check_build()
        # журнал волн: записи только зафиксированных волн, id уникальны
        ids = [w["wave_id"] for w in r2.state["wave_log"]]
        assert len(ids) == len(set(ids))
        assert r2.state["variants"]["basis4_rl"]["ptr"] == \
            st_before["basis4_rl"]["ptr"]
        eps2, _c2 = r2.train_wave("basis4_rl")
        assert [e["action_sha1"] for e in eps1] == \
            [e["action_sha1"] for e in eps2]
        assert r2.state["variants"]["basis4_rl"]["updates"][-1] == upd1
        # оценка: без шума, без обновлений и без записи в replay
        n_before = r2.replays["pca4_rl"].n
        ver = r2.agents["pca4_rl"].version
        r2.cfg = dict(cfg, starts=dict(cfg["starts"], mid=[25, 26, 27, 28,
                                                           29]))
        r2.checkpoint("t")
        # прерванная запись: неполный файл не принимается
        bad = os.path.join(td, "ckpt", "full48_rl_t.pt")
        open(bad, "wb").write(open(bad, "rb").read()[:100])
        try:
            r2.checkpoint("t")
            raise AssertionError("принят неполный чекпойнт")
        except SystemExit:
            pass
        os.remove(bad)
        r2.checkpoint("t")
        # другая сборка q0 при возобновлении — отказ
        r3 = Runner(a, cfg, torch)
        r3.vla = kc.FakeVLA("other")
        r3.rl_dev = "cpu"
        r3.load_or_new()
        try:
            r3.check_build()
            raise AssertionError("принята другая сборка q0")
        except SystemExit:
            pass
        r2.eval_bank("pca4_rl", "t", "mid")
        # веса изменились после закрепления — оценка отказывает
        with torch.no_grad():
            r2.agents["basis4_rl"].actor.coef.bias.add_(1.0)
        try:
            r2.eval_bank("basis4_rl", "t", "mid")
            raise AssertionError("оценены незакреплённые веса")
        except SystemExit:
            pass
        assert len(os.listdir(os.path.join(td, "ckpt"))) == 3
        assert r2.replays["pca4_rl"].n == n_before
        assert r2.agents["pca4_rl"].version == ver
        ev = r2.state["evals"]["pca4_rl@t@mid"]
        assert len(ev["eps"]) == 10 and all(d["f"] <= 1 for d in ev["diag"])
        lines = open(WAVES_LOG).read().strip().split("\n")
        assert len(lines) >= 3
    # сверка с эталоном
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "r.json")
        e = dict(init_state_id=40, action_sha1="a", success=True,
                 done_step=5, init_hash="h")
        json.dump(dict(episodes=[e]), open(p, "w"))
        assert ref_problems([dict(e)], p) == []
        assert ref_problems([dict(e, done_step=6)], p)
    print("самопроверка k18_train пройдена: q0-набор в три replay, smoke, "
          "бюджет по командам, возобновление воспроизводит волну и "
          "обновления, журнал только зафиксированных волн, атомарный "
          "сверяемый чекпойнт (неполный — отказ), закреплённая сборка q0, "
          "оценка только закреплённых весов, без шума/обновлений/replay, "
          "сверка с эталоном")
    return 0


if __name__ == "__main__":
    sys.exit(main())
