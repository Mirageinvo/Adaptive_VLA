#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 0: технический замер стоимости официального цикла оценки
и паспорт готовности (не SR модели и не время успешного выполнения задачи).

Цикл — официальный `scripts/eval_policy_xpolicylab.py` закреплённой ревизии
RoboTwin (его main() вызывается как есть, тем же порядком, что
`scripts/eval_policy.sh`, включая Sapien_TEST). Время меряется внешними
обёртками: методы среды (setup_demo, play_once — экспертная проверка сида,
get_obs, take_action, close_env) и вызовы клиента политики (reset,
prepare_case, update_obs, get_action, trial_end — время туда-обратно).
Политика — техническое удержание текущей конфигурации, поднятое сервером
XPolicyLab (client_server.ws.PolicyServer) в отдельном процессе; XPolicyLab
не меняется (у demo_policy на закреплённой ревизии нет model.py). Удержание
— в выбранном пространстве действий: joint — текущие суставы и схваты из
state наблюдения; ee — текущие позы концевиков и схваты. Нулей, выданных за
удержание, нет: при отсутствии нужного поля — отказ.

Режимы:
  serve  — сервер удерживающей политики (запускается из run сам);
  run    — задача: --episodes N после --warmup W прогревочных (не входят в
           сводку); --debug: два эпизода без прогрева и проверка формата
           наблюдения/действия, reset и очистки между эпизодами;
  passport — reports/rt0/readiness.json из env_check/render/inventory/
           benchmark, версий пакетов и рабочих копий.
Отчёт: reports/rt0/benchmark.json (дополняется по ключу задача|конфиг|
тип действия|режим). Запуск — в окружении RoboTwin, из корня Adaptive_VLA.
"""
import argparse
import contextlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "reports/rt0/benchmark.json"
READY = "reports/rt0/readiness.json"


# --- удерживающая политика (на стороне сервера) ---------------------------------
JOINT_KEYS = ("left_arm_joint_state", "left_ee_joint_state",
              "right_arm_joint_state", "right_ee_joint_state")
EE_KEYS = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose",
           "right_ee_joint_state")


class HoldModel:
    """Возвращает чанк из `chunk` одинаковых команд «оставаться в текущей
    конфигурации» в пространстве action_type."""

    def __init__(self, chunk=8, action_type="joint", log=None):
        if action_type not in ("joint", "ee"):
            raise ValueError(action_type)
        self.chunk, self.action_type, self.log = int(chunk), action_type, log
        self.state, self.ep = None, 0

    def _log(self, ev, t0):
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps(dict(ev=ev, ep=self.ep, dt=time.perf_counter()
                                        - t0)) + "\n")

    def prepare_case(self, case_meta=None):
        t0 = time.perf_counter()
        self.case = case_meta
        self._log("prepare_case", t0)

    def reset(self):
        t0 = time.perf_counter()
        self.state = None
        self._log("reset", t0)

    def update_obs(self, obs):
        t0 = time.perf_counter()
        st = obs.get("state") or {}
        keys = JOINT_KEYS if self.action_type == "joint" else EE_KEYS
        miss = [k for k in keys if k not in st]
        if miss:
            raise KeyError(f"в state наблюдения нет {miss} для удержания "
                           f"({self.action_type})")
        self.state = {k: np.asarray(st[k], np.float32).reshape(-1).tolist()
                      for k in keys}
        self._log("update_obs", t0)

    def get_action(self):
        t0 = time.perf_counter()
        if self.state is None:
            raise RuntimeError("get_action до update_obs")
        a = dict(self.state, action_type=self.action_type)
        out = [dict(a) for _ in range(self.chunk)]
        self._log("get_action", t0)
        return out

    def on_trial_end(self, result=None):
        t0 = time.perf_counter()
        self._log("trial_end", t0)
        self.ep += 1


def serve(a):
    root = os.path.abspath(a.robotwin_root)
    xpl = os.path.join(root, "XPolicyLab")
    for p in (xpl, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import asyncio
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    model = HoldModel(a.chunk, a.action_type, a.log)
    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1",
                                                    port=int(a.port)))
    asyncio.run(server.serve_forever())


# --- замер на стороне среды -------------------------------------------------
class Recorder:
    """События по эпизодам: всё между концом прошлого эпизода и началом
    раскатки (prepare_case) — «подготовка» следующего; от prepare_case до
    trial_end — «раскатка»."""

    def __init__(self):
        self.episodes, self.cur = [], None
        self.pre = self._bucket()
        self.t_pre0 = time.perf_counter()
        self.first_obs = self.first_chunk = None
        self.counts = {}

    @staticmethod
    def _bucket():
        return dict(sums={}, counts={})

    def add(self, name, dt):
        self.counts[name] = self.counts.get(name, 0) + 1
        b = self.cur["roll"] if self.cur is not None else self.pre
        b["sums"][name] = b["sums"].get(name, 0.0) + dt
        b["counts"][name] = b["counts"].get(name, 0) + 1

    def start(self, seed):
        now = time.perf_counter()
        self.cur = dict(seed=int(seed), pre=self.pre,
                        pre_wall_s=now - self.t_pre0, t0=now,
                        roll=self._bucket())

    def end(self, success, task_env):
        now = time.perf_counter()
        c = self.cur or dict(seed=None, pre=self.pre, pre_wall_s=None,
                             t0=now, roll=self._bucket())
        cnt = int(getattr(task_env, "take_action_cnt", -1)) \
            if task_env is not None else -1
        lim = int(getattr(task_env, "step_lim", -1)) \
            if task_env is not None else -1
        reason = ("success" if success else "step_lim" if lim > 0 and
                  cnt >= lim else "прервано (ошибка или ранний выход)")
        self.episodes.append(dict(
            seed=c["seed"], success=bool(success), end_reason=reason,
            take_action_cnt=cnt, step_lim=lim,
            policy_calls=c["roll"]["counts"].get("client.get_action", 0),
            rollout_wall_s=now - c["t0"], rollout=c["roll"],
            pre_rollout_wall_s=c["pre_wall_s"], pre_rollout=c["pre"]))
        self.cur, self.pre, self.t_pre0 = None, self._bucket(), now


def timed(rec, name, fn):
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            rec.add(name, time.perf_counter() - t0)
    return w


def schema(x, depth=0):
    """Краткая схема наблюдения/действия: ключи, формы, типы."""
    if isinstance(x, dict):
        return {k: schema(v, depth + 1) for k, v in x.items()} if depth < 4 \
            else "dict"
    if isinstance(x, (list, tuple)):
        if x and all(isinstance(v, (int, float)) for v in x):
            return f"list[{len(x)}]"
        return [schema(x[0], depth + 1), f"len {len(x)}"] if x else "[]"
    if isinstance(x, np.ndarray):
        return f"{x.dtype}{list(x.shape)}"
    if isinstance(x, (bytes, bytearray)):
        return f"bytes[{len(x)}]"
    return type(x).__name__


def summarize(episodes, warmup):
    meas = episodes[warmup:]

    def stat(vals):
        v = [x for x in vals if x is not None]
        return dict(mean=float(np.mean(v)), min=float(np.min(v)),
                    max=float(np.max(v)), n=len(v)) if v else None
    out = dict(episodes_measured=len(meas), warmup=warmup)
    out["rollout_wall_s"] = stat([e["rollout_wall_s"] for e in meas])
    out["pre_rollout_wall_s"] = stat([e["pre_rollout_wall_s"] for e in meas])
    out["steps"] = stat([e["take_action_cnt"] for e in meas])
    out["policy_calls"] = stat([e["policy_calls"] for e in meas])
    keys = sorted({k for e in meas for k in e["rollout"]["sums"]})
    out["rollout_parts_s"] = {k: stat([e["rollout"]["sums"].get(k, 0.0)
                                       for e in meas]) for k in keys}
    pk = sorted({k for e in meas for k in e["pre_rollout"]["sums"]})
    out["pre_rollout_parts_s"] = {k: stat([e["pre_rollout"]["sums"].get(
        k, 0.0) for e in meas]) for k in pk}
    tot = [e["rollout_wall_s"] for e in meas]
    st = [max(e["take_action_cnt"], 1) for e in meas]
    out["rollout_s_per_step"] = float(np.sum(tot) / np.sum(st)) if meas \
        else None
    out["end_reasons"] = {r: sum(e["end_reason"] == r for e in meas)
                          for r in {e["end_reason"] for e in meas}}
    return out


def server_times(log):
    """Время модели на сервере по эпизодам и событиям."""
    out = {}
    if not os.path.exists(log):
        return out
    for line in open(log):
        r = json.loads(line)
        d = out.setdefault(int(r["ep"]), {})
        d[r["ev"]] = d.get(r["ev"], 0.0) + float(r["dt"])
    return out


def free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"сервер политики завершился, код {proc.returncode}")
        with contextlib.closing(socket.socket()) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return time.time() - t0
        time.sleep(0.5)
    raise SystemExit("сервер политики не поднялся")


def run(a):
    root = os.path.abspath(a.robotwin_root)
    repo = os.getcwd()
    os.makedirs("reports/rt0", exist_ok=True)
    episodes = 2 if a.debug else a.warmup + a.episodes
    warmup = 0 if a.debug else a.warmup
    mode = "debug" if a.debug else "bench"
    key = f"{a.task}|{a.task_config}|{a.action_type}|{mode}"
    port = a.port or free_port()
    slog = os.path.abspath(f"reports/rt0/server_{a.task}_{mode}.jsonl")
    if os.path.exists(slog):
        os.remove(slog)
    t_all = time.perf_counter()
    srv = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "serve", "--port",
         str(port), "--chunk", str(a.chunk), "--action-type", a.action_type,
         "--log", slog, "--robotwin-root", root],
        stdout=open(f"reports/rt0/server_{a.task}_{mode}.log", "w"),
        stderr=subprocess.STDOUT)
    rec = Recorder()
    res = dict(task=a.task, task_config=a.task_config,
               action_type=a.action_type, chunk=a.chunk, seed=a.seed,
               expert_check=a.expert_check, episodes_total=episodes,
               mode=mode, note="технический замер стоимости цикла с "
               "удерживающей политикой: не SR модели и не время успешного "
               "выполнения задачи")
    err = None
    try:
        res["server_start_s"] = wait_port(port, srv)
        for p in (root, os.path.join(root, "scripts"),
                  os.path.join(root, "description", "utils")):
            if p not in sys.path:
                sys.path.insert(0, p)
        os.chdir(root)
        spec = importlib.util.spec_from_file_location(
            "rt_eval_xpl", os.path.join(root, "scripts",
                                        "eval_policy_xpolicylab.py"))
        evx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evx)
        t0 = time.perf_counter()
        from test_render import Sapien_TEST
        Sapien_TEST()
        res["sapien_test_s"] = time.perf_counter() - t0
        envh = {}
        orig_cd, orig_bpc = evx.class_decorator, evx.build_policy_client
        orig_prep, orig_end = evx.prepare_policy_case, evx.notify_trial_end

        def class_decorator(task_name):
            t0 = time.perf_counter()
            env = orig_cd(task_name)
            res["env_create_s"] = time.perf_counter() - t0
            for m in ("setup_demo", "play_once", "get_obs", "take_action",
                      "close_env"):
                setattr(env, m, timed(rec, f"env.{m}", getattr(env, m)))
            envh["env"] = env
            return env

        def build_policy_client(usr_args):
            cl = orig_bpc(usr_args)
            call0 = cl.call

            def call(func_name=None, obs=None, **kw):
                if func_name == "update_obs" and rec.first_obs is None:
                    rec.first_obs = schema(obs)
                t0 = time.perf_counter()
                try:
                    out = call0(func_name=func_name, obs=obs, **kw)
                finally:
                    rec.add(f"client.{func_name}", time.perf_counter() - t0)
                if func_name == "get_action" and rec.first_chunk is None:
                    rec.first_chunk = schema(out)
                return out
            cl.call = call
            return cl

        def prepare_policy_case(model_client, task_name, seed, instruction,
                                action_type):
            rec.start(seed)
            return orig_prep(model_client, task_name, seed, instruction,
                             action_type)

        def notify_trial_end(model_client, task_name, seed, success):
            out = orig_end(model_client, task_name, seed, success)
            rec.end(success, envh.get("env"))
            return out
        evx.class_decorator = class_decorator
        evx.build_policy_client = build_policy_client
        evx.prepare_policy_case = prepare_policy_case
        evx.notify_trial_end = notify_trial_end
        usr = dict(bench_name="RoboTwin", task_name=a.task,
                   env_cfg_type=a.env_cfg_type, policy_name="rt0_hold",
                   host="localhost", port=str(port), protocol="ws",
                   root_dir=root, device_id=str(a.device_id), seed=a.seed,
                   task_config=a.task_config, instruction_type=None,
                   xpolicylab_root=os.path.join(root, "XPolicyLab"),
                   eval_batch=False, test_num=episodes,
                   expert_check=a.expert_check, ckpt_name="hold",
                   ckpt_setting="hold", action_type=a.action_type)
        evx.main(usr)
    except BaseException as e:  # noqa: BLE001 — записать и завершить
        err = f"{type(e).__name__}: {e}"
    finally:
        os.chdir(repo)
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
    res["total_wall_s"] = time.perf_counter() - t_all
    res["error"] = err
    sv = server_times(slog)
    for i, e in enumerate(rec.episodes):
        e["server_model_s"] = sv.get(i)
        e["warmup"] = i < warmup
    res["episodes"] = rec.episodes
    res["calls_total"] = rec.counts
    res["summary"] = summarize(rec.episodes, warmup) if rec.episodes else None
    prob = []
    if err:
        prob.append(f"исключение: {err}")
    if len(rec.episodes) != episodes:
        prob.append(f"эпизодов {len(rec.episodes)} из {episodes}")
    if a.debug:
        n = len(rec.episodes)
        dbg = dict(first_obs_schema=rec.first_obs,
                   first_action_chunk_schema=rec.first_chunk,
                   resets=rec.counts.get("client.reset", 0),
                   prepares=rec.counts.get("client.prepare_case", 0),
                   trial_ends=rec.counts.get("client.trial_end", 0),
                   env_close=rec.counts.get("env.close_env", 0))
        for k in ("resets", "prepares", "trial_ends"):
            if dbg[k] != n:
                prob.append(f"debug: {k} {dbg[k]} при эпизодах {n}")
        if dbg["env_close"] < n:
            prob.append(f"debug: close_env {dbg['env_close']} < {n}")
        if not rec.first_obs or not rec.first_chunk:
            prob.append("debug: нет наблюдения или действия")
        res["debug"] = dbg
    res["problems"] = prob
    res["ok"] = not prob
    allr = json.load(open(OUT)) if os.path.exists(OUT) else {}
    allr[key] = res
    json.dump(allr, open(OUT, "w"), ensure_ascii=False, indent=1,
              default=str)
    s = res["summary"] or {}
    print(f"  {key}: эпизодов {len(rec.episodes)}/{episodes}, причины "
          f"{s.get('end_reasons')}, раскатка {s.get('rollout_wall_s')}, "
          f"подготовка {s.get('pre_rollout_wall_s')}, с/шаг "
          f"{s.get('rollout_s_per_step')}")
    print("  " + ("ОТКАЗ: " + "; ".join(prob) if prob else "замер пройден")
          + f"; отчёт {OUT}")
    return 3 if prob else 0


# --- паспорт ---------------------------------------------------------------
def git(root, *args):
    try:
        return subprocess.check_output(["git", "-C", root, *args],
                                       text=True, stderr=subprocess.STDOUT
                                       ).strip()
    except Exception as e:  # noqa: BLE001
        return f"ошибка: {e}"


def passport(a):
    from importlib import metadata
    root = os.path.abspath(a.robotwin_root)
    xpl = os.path.join(root, "XPolicyLab")
    rep = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rep["robotwin"] = dict(path=root, head=git(root, "rev-parse", "HEAD"),
                           xpolicylab_pin=git(root, "rev-parse",
                                              "HEAD:XPolicyLab"),
                           status=git(root, "status", "--porcelain"))
    rep["xpolicylab"] = dict(head=git(xpl, "rev-parse", "HEAD"),
                             status=git(xpl, "status", "--porcelain"))
    rep["xpolicylab"]["matches_pin"] = \
        rep["xpolicylab"]["head"] == rep["robotwin"]["xpolicylab_pin"]
    rep["xpolicylab"]["demo_policy_has_model_py"] = os.path.exists(
        os.path.join(xpl, "policy", "demo_policy", "model.py"))
    pk = {}
    for p in ("sapien", "torch", "torchvision", "numpy", "mplib",
              "nvidia-curobo", "warp-lang", "setuptools", "open3d",
              "opencv-python", "h5py", "XPolicyLab", "websockets",
              "huggingface-hub"):
        try:
            pk[p] = metadata.version(p)
        except Exception:  # noqa: BLE001
            pk[p] = None
    rep["packages"] = pk
    rep["python"] = sys.version.split()[0]
    rep["env"] = {k: os.environ.get(k) for k in (
        "VK_ICD_FILENAMES", "NVIDIA_DRIVER_CAPABILITIES", "CONDA_PREFIX",
        "HF_REVISION", "HF_ENDPOINT")}
    try:
        rep["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,"
             "driver_version", "--format=csv"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        rep["gpu"] = f"ошибка: {e}"
    for nm, f in (("env_check", "reports/rt0/env_check.txt"),
                  ("render", "reports/rt0/render_check.json"),
                  ("inventory", "reports/rt0/inventory.json"),
                  ("benchmark", OUT)):
        if not os.path.exists(f):
            rep[nm] = None
        elif f.endswith(".txt"):
            rep[nm] = open(f).read()[-3000:]
        else:
            rep[nm] = json.load(open(f))
    try:
        st = os.statvfs(a.data_root)
        rep["disk_free_gb"] = st.f_bavail * st.f_frsize / 2 ** 30
    except OSError:
        rep["disk_free_gb"] = None
    inv = rep.get("inventory") or {}
    per_ep = [t["gb_per_episode"] for t in inv.get("tasks", [])
              if t.get("gb_per_episode")]
    if per_ep:
        g = float(np.mean(per_ep))
        rep["disk_forecast_gb"] = {
            f"{n}_tasks_x_{e}_episodes": g * n * e
            for n in (8, 10) for e in (50,)}
        rep["disk_forecast_basis"] = (f"среднее измеренное {g:.3f} ГБ/эпизод "
                                      f"по {len(per_ep)} задачам; без ZIP-"
                                      f"копий (HF_KEEP_ARCHIVES)")
    lim = []
    if not rep["xpolicylab"]["matches_pin"]:
        lim.append("XPolicyLab не на закреплённой ревизии RoboTwin")
    if not rep["xpolicylab"]["demo_policy_has_model_py"]:
        lim.append("demo_policy закреплённой ревизии без model.py: "
                   "официальная нулевая политика не запускается как есть; "
                   "для замера — собственная удерживающая модель")
    lim += a.note or []
    rep["limitations"] = lim
    json.dump(rep, open(READY, "w"), ensure_ascii=False, indent=1,
              default=str)
    print(json.dumps({k: rep[k] for k in ("robotwin", "xpolicylab",
                                          "packages", "limitations")},
                     ensure_ascii=False, indent=1, default=str))
    print(f"  паспорт: {READY}")
    return 0


# --- самопроверка -----------------------------------------------------------
def selftest():
    obs = dict(state=dict(left_arm_joint_state=np.arange(6),
                          left_ee_joint_state=np.array([0.5]),
                          right_arm_joint_state=np.arange(6) + 10,
                          right_ee_joint_state=np.array([0.1]),
                          left_ee_pose=np.arange(7), right_ee_pose=np.ones(7)))
    m = HoldModel(chunk=3, action_type="joint")
    m.reset()
    try:
        m.get_action()
        raise AssertionError("действие без наблюдения")
    except RuntimeError:
        pass
    m.update_obs(obs)
    ch = m.get_action()
    assert len(ch) == 3 and ch[0]["action_type"] == "joint"
    assert ch[2]["right_arm_joint_state"] == list(range(10, 16))
    assert ch[0]["left_ee_joint_state"] == [0.5]
    me = HoldModel(chunk=2, action_type="ee")
    me.update_obs(obs)
    assert me.get_action()[0]["left_ee_pose"] == list(range(7))
    try:
        HoldModel(action_type="joint").update_obs(dict(state=dict(
            left_arm_joint_state=[0])))
        raise AssertionError("неполное состояние принято")
    except KeyError:
        pass
    rec = Recorder()
    rec.add("env.setup_demo", 1.0)
    rec.add("env.play_once", 2.0)
    rec.start(100000)
    for _ in range(4):
        rec.add("env.get_obs", 0.1)
        rec.add("env.take_action", 0.2)
    rec.add("client.get_action", 0.05)

    class Env:
        take_action_cnt, step_lim = 400, 400
    rec.end(False, Env())
    rec.add("env.close_env", 0.3)
    rec.start(100001)
    rec.add("env.take_action", 0.2)
    Env.take_action_cnt = 10
    rec.end(True, Env())
    e0, e1 = rec.episodes
    assert e0["end_reason"] == "step_lim" and e1["end_reason"] == "success"
    assert abs(e0["pre_rollout"]["sums"]["env.play_once"] - 2.0) < 1e-12
    assert abs(e0["rollout"]["sums"]["env.take_action"] - 0.8) < 1e-9
    assert e1["pre_rollout"]["sums"] == {"env.close_env": 0.3}
    assert e0["policy_calls"] == 1
    s = summarize(rec.episodes, 1)
    assert s["episodes_measured"] == 1 and s["end_reasons"] == {"success": 1}
    assert schema(dict(a=np.zeros((2, 3), np.uint8), b=[1, 2]))["a"] == \
        "uint8[2, 3]"
    print("самопроверка rt0_benchmark пройдена: удержание joint/ee из "
          "текущего состояния (без нулей), отказ при неполном состоянии, "
          "раскладка времени по эпизодам (подготовка/раскатка), причины "
          "завершения, прогрев исключён")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="RoboTwin этап 0: замер цикла")
    ap.add_argument("mode", choices=("serve", "run", "passport"))
    ap.add_argument("--robotwin-root", default=os.environ.get(
        "ROBOTWIN_ROOT", ""))
    ap.add_argument("--data-root", default=os.environ.get("RT_DATA_ROOT",
                                                          "."))
    ap.add_argument("--task", default="adjust_bottle")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--env-cfg-type", default="aloha_agilex")
    ap.add_argument("--action-type", choices=("joint", "ee"),
                    default="joint")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-id", default="0")
    ap.add_argument("--expert-check", default="true",
                    choices=("true", "false"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--note", action="append",
                    help="ограничение для паспорта (можно несколько)")
    a = ap.parse_args()
    a.expert_check = a.expert_check == "true"
    if not a.robotwin_root:
        raise SystemExit("нужен --robotwin-root или ROBOTWIN_ROOT")
    if a.mode == "serve":
        return serve(a)
    if a.mode == "passport":
        return passport(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
