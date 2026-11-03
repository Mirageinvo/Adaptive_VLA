#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 3: явный банк оценочных случаев и клиент оценки по нему.

Запуск — в окружении RoboTwin из $ROBOTWIN_ROOT (как официальный клиент).
Используются функции официального scripts/eval_policy_xpolicylab.py
(конфигурация задачи, класс среды, инструкция, преобразования наблюдений и
действий, ws-клиент, prepare_case/trial_end) — меняется только управление
случаями: список (задача, seed) задаётся банком, замена следующим seed
запрещена.

bank — кандидаты: seed из журнала источника (по умолчанию пилот flat24 RT2)
  в порядке возрастания; для каждого: экспертная проверка ОДИН раз
  (setup_demo + play_once, plan_success и check_success), инструкция —
  build_instruction (тот же test_num, что в источнике), затем два свежих
  setup_demo без эксперта -> отпечатки начального состояния должны совпасть
  (состояние руки, позы всех объектов среды с get_pose, кадры камер) —
  каждый на НОВОМ объекте среды (class_decorator), не на том, где шёл
  эксперт: иначе в отпечаток попадают объекты, оставшиеся после play_once;
  случай, не прошедший эксперта или воспроизводимость, пропускается с
  причиной; берутся первые N принятых. Закрепляются: задача, seed,
  инструкция, отпечатки, SHA конфигураций задачи/эмбодимента/камер, коммит
  RoboTwin, параметры.
eval — для каждого случая задачи: новый объект среды, setup_demo(seed) без
  эксперта -> отпечаток
  физического состояния (рука и позы объектов) = банку, иначе технический
  отказ (кадры — сверяются и пишутся, несовпадение кадров — тоже отказ);
  подтверждённый случай (seed, ep_num, инструкция, отпечаток, отпечаток
  банка) дописывается в <log_dir>/<задача>/rt3_cases.jsonl до раскатки;
  --limit k — только первые k случаев (smoke);
  инструкция — из банка; официальный цикл исполнения эпизода (update_obs /
  get_action / take_action, успех, лимит шагов); prepare_case/trial_end.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

KIND = "rt3_bank"
FP_PROTOCOL = "fresh_env_v2"    # отпечаток — на НОВОМ объекте среды (без
                                # хвостов экспертного прогона/прошлых случаев)
CASES = "rt3_cases.jsonl"       # подтверждённые случаи (сверяет rt3_check)
SOURCE_TEST_NUM = 10            # test_num пилота RT2 — от него строится инструкция


def sha_file(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def sha_arrays(items):
    h = hashlib.sha1()
    for k, v in items:
        a = np.ascontiguousarray(np.asarray(v))
        h.update(str(k).encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def official(root):
    for p in (root, os.path.join(root, "scripts"),
              os.path.join(root, "description", "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import eval_policy_xpolicylab as ep
    return ep


def flat_items(prefix, obj):
    """Вложенные словари массивов -> [(путь, массив)] в отсортированном
    порядке."""
    out = []
    if isinstance(obj, dict):
        for k in sorted(obj):
            out += flat_items(f"{prefix}/{k}", obj[k])
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += flat_items(f"{prefix}/{i}", v)
    elif obj is not None and not isinstance(obj, str):
        try:
            out.append((prefix, np.asarray(obj)))
        except Exception:                                   # noqa: BLE001
            pass
    return out


def poses_of(task_env):
    """Позы объектов среды: атрибуты с get_pose() (p, q)."""
    out = []
    for name in sorted(vars(task_env)):
        val = getattr(task_env, name, None)
        if val is None or not hasattr(val, "get_pose") or \
                isinstance(val, type):
            continue
        try:
            pz = val.get_pose()
            arr = np.concatenate([np.asarray(pz.p, np.float64).ravel(),
                                  np.asarray(pz.q, np.float64).ravel()])
            out.append((name, arr))
        except Exception:                                   # noqa: BLE001
            continue
    return out


def fingerprint(ep, task_env):
    obs = task_env.get_obs()
    st = ep.convert_state(obs, task_env=task_env)
    vis = ep.convert_vision(obs)
    poses = poses_of(task_env)
    return dict(state=sha_arrays(flat_items("state", st)),
                poses=sha_arrays(poses), pose_names=[n for n, _ in poses],
                vision=sha_arrays(flat_items("vision", vis)))


def task_setup(ep, task, policy_name="adaptive_vla_rt2",
               task_config="demo_clean"):
    usr = dict(task_name=task, policy_name=policy_name,
               task_config=task_config, ckpt_setting="rt3")
    args, emb = ep.load_task_args(usr)
    itype = ep.resolve_eval_instruction(usr, args)
    args["eval_instruction"] = itype
    args["eval_mode"] = True
    args["eval_video_log"] = False          # видео не пишем: на физику не влияет
    args["render_freq"] = 0
    return args, itype, emb


def config_shas(ep, args, root):
    out = {"task_config": sha_file(ep.TASK_CONFIG_ROOT /
                                   f"{args['task_config']}.yml")}
    for side in ("left", "right"):
        f = os.path.join(args[f"{side}_robot_file"], "config.yml")
        out[f"{side}_embodiment"] = sha_file(f)
    try:
        out["robotwin_commit"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                       # noqa: BLE001
        out["robotwin_commit"] = None
    out["client_sha1"] = sha_file(os.path.join(
        root, "scripts", "eval_policy_xpolicylab.py"))
    return out


def setup_fp(ep, task_env, args, ep_num, seed):
    task_env.setup_demo(now_ep_num=ep_num, seed=seed, is_test=True, **args)
    fp = fingerprint(ep, task_env)
    return fp


def build_bank(a):
    ep = official(a.root)
    from envs.utils.create_actor import UnStableError
    bank = dict(kind=KIND, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                source=os.path.abspath(a.source), n=a.n, tasks={},
                source_test_num=SOURCE_TEST_NUM, fp_protocol=FP_PROTOCOL)
    for task in a.tasks:
        cands = sorted(json.loads(x)["seed"] for x in open(
            os.path.join(a.source, task, "episodes.jsonl"), encoding="utf-8")
            if x.strip())
        args, itype, _emb = task_setup(ep, task)
        task_env = ep.class_decorator(task)
        cases, skipped = [], []
        for i, seed in enumerate(cands):
            if len(cases) >= a.n:
                break
            try:
                task_env.setup_demo(now_ep_num=i, seed=seed, is_test=True,
                                    **args)
                info = task_env.play_once()
                ok = bool(task_env.plan_success and task_env.check_success())
                task_env.close_env()
            except UnStableError:
                ep.safe_close_env(task_env)
                skipped.append(dict(seed=seed, why="unstable (эксперт)"))
                continue
            except Exception as e:                          # noqa: BLE001
                ep.safe_close_env(task_env)
                skipped.append(dict(seed=seed, why=f"эксперт: {e}"))
                continue
            if not ok:
                skipped.append(dict(seed=seed, why="эксперт не справился"))
                continue
            instr = ep.build_instruction(args, info, itype, SOURCE_TEST_NUM,
                                         seed)
            fps = []
            for rep in range(2):
                fe = ep.class_decorator(task)       # свежий объект среды
                try:
                    fps.append(setup_fp(ep, fe, args, i, seed))
                    fe.close_env()
                except Exception as e:                      # noqa: BLE001
                    ep.safe_close_env(fe)
                    fps.append(dict(error=f"{type(e).__name__}: {e}"))
            if "error" in fps[0] or fps[0] != fps[1]:
                skipped.append(dict(seed=seed, why="начальное состояние не "
                                    "воспроизводится", fps=fps))
                continue
            if not fps[0]["pose_names"]:
                raise SystemExit(f"{task}: у среды не найдено объектов с "
                                 f"get_pose — физическое состояние не "
                                 f"закрепить")
            cases.append(dict(task=task, seed=int(seed), ep_num=i,
                              instruction=instr, instruction_type=itype,
                              fingerprint=fps[0]))
            print(f"  {task} seed {seed}: принят; объектов "
                  f"{len(fps[0]['pose_names'])}; «{instr[:60]}»", flush=True)
        if len(cases) < a.n:
            raise SystemExit(f"{task}: принято {len(cases)} < {a.n} случаев "
                             f"(пропущено: {skipped})")
        bank["tasks"][task] = dict(cases=cases, skipped=skipped,
                                   config=config_shas(ep, args, a.root))
        print(f"  {task}: случаи {[c['seed'] for c in cases]}; пропущено "
              f"{len(skipped)}", flush=True)
    bank["fingerprint"] = hashlib.sha1(json.dumps(
        bank["tasks"], sort_keys=True).encode()).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if os.path.exists(a.out):
        old = json.load(open(a.out, encoding="utf-8"))
        if old.get("fingerprint") != bank["fingerprint"]:
            raise SystemExit(f"{a.out} уже закреплён и отличается от нового "
                             f"построения — банк не перезаписывается")
        print(f"  {a.out}: закреплён, повторное построение совпало")
        return 0
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(bank, fh, ensure_ascii=False, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"  банк {a.out}: {sum(len(t['cases']) for t in bank['tasks'].values())}"
          f" случаев, отпечаток {bank['fingerprint'][:12]}", flush=True)
    return 0


def run_case(ep, task_env, args, case, model_client, frequency,
             action_type):
    """Официальный цикл исполнения одного эпизода (без замены случая)."""
    succ, steps = False, 0
    ep.prepare_policy_case(model_client, case["task"], case["seed"],
                           case["instruction"], action_type)
    ep.reset_policy(model_client)
    try:
        while not ep.is_episode_end(task_env):
            observation = task_env.get_obs()
            xpl = ep.robotwin_obs_to_xpolicylab(
                observation, instruction=task_env.get_instruction(),
                env_idx=0, frequency=frequency, task_env=task_env)
            model_client.call(func_name="update_obs", obs=xpl)
            chunk = ep.normalize_action_chunk(
                model_client.call(func_name="get_action"))
            if len(chunk) == 0:
                raise RuntimeError("Policy returned an empty action chunk.")
            for idx, action in enumerate(chunk):
                flat, atype = ep.xpolicylab_action_to_robotwin(
                    action, action_type=action_type,
                    current_observation=observation)
                task_env.take_action(flat, action_type=atype)
                steps += 1
                if task_env.eval_success:
                    succ = True
                    break
                if ep.is_episode_end(task_env) or idx + 1 == len(chunk):
                    break
                observation = task_env.get_obs()
                xpl = ep.robotwin_obs_to_xpolicylab(
                    observation, instruction=task_env.get_instruction(),
                    env_idx=0, frequency=frequency, task_env=task_env)
                model_client.call(func_name="update_obs", obs=xpl)
            if succ:
                break
    except Exception:
        print("\n\033[91mPolicy rollout error:\033[0m")
        print(traceback.format_exc())
        ep.notify_trial_end(model_client, case["task"], case["seed"], False)
        raise SystemExit("Aborting task: ошибка раскатки случая "
                         f"{case['task']}/{case['seed']}")
    ep.notify_trial_end(model_client, case["task"], case["seed"], succ)
    return succ, steps


def eval_bank(a):
    ep = official(a.root)
    bank = json.load(open(a.bank, encoding="utf-8"))
    if bank.get("kind") != KIND or hashlib.sha1(json.dumps(
            bank["tasks"], sort_keys=True).encode()).hexdigest() != \
            bank["fingerprint"]:
        raise SystemExit("банк повреждён")
    if bank.get("fp_protocol") != FP_PROTOCOL:
        raise SystemExit(f"банк снят по протоколу {bank.get('fp_protocol')}, "
                         f"нужен {FP_PROTOCOL} (пересобрать банк)")
    tb = bank["tasks"][a.task]
    args, itype, _emb = task_setup(ep, a.task)
    cfg = config_shas(ep, args, a.root)
    if cfg != tb["config"]:
        raise SystemExit(f"конфигурация ≠ банку: {cfg} vs {tb['config']}")
    usr = dict(host=a.host, port=a.port, protocol="ws",
               xpolicylab_root=os.path.join(a.root, "XPolicyLab"),
               evaluation_id=f"rt3-{a.task}", trial_id=f"{a.task}-bank",
               action_case_id=f"{a.task}-case")
    cases = tb["cases"][:a.limit] if a.limit else tb["cases"]
    cdir = os.path.join(a.log_dir, a.task)
    os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, CASES)
    if os.path.exists(cpath):
        raise SystemExit(f"{cpath} уже есть: незавершённый журнал задачи "
                         f"должен быть убран в карантин")
    model_client = ep.build_policy_client(usr)
    suc = 0
    try:
        for k, case in enumerate(cases):
            task_env = ep.class_decorator(a.task)   # свежий объект на случай
            task_env.setup_demo(now_ep_num=case["ep_num"], seed=case["seed"],
                                is_test=True, **args)
            fp = fingerprint(ep, task_env)
            bad = [c for c in ("state", "poses", "pose_names", "vision")
                   if fp[c] != case["fingerprint"][c]]
            if bad:
                ep.safe_close_env(task_env)
                raise SystemExit(f"Aborting task: случай {a.task}/"
                                 f"{case['seed']} не воспроизводится ({bad})")
            with open(cpath, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(
                    task=a.task, seed=case["seed"], ep_num=case["ep_num"],
                    instruction=case["instruction"],
                    bank_fingerprint=bank["fingerprint"], fingerprint=fp),
                    ensure_ascii=False) + "\n")
            task_env.set_instruction(instruction=case["instruction"])
            succ, steps = run_case(ep, task_env, args, case, model_client,
                                   a.frequency, "joint")
            suc += int(succ)
            print("\033[92mSuccess!\033[0m" if succ else
                  "\033[91mFail!\033[0m")
            task_env.close_env(clear_cache=((k + 1) %
                                            args["clear_cache_freq"] == 0))
            print(f"{a.task} | case {k + 1}/{len(cases)} seed "
                  f"{case['seed']} | шагов {steps}\nSuccess rate: "
                  f"{suc}/{k + 1}", flush=True)
    finally:
        ep.close_policy_client(model_client)
    n = len(cases)
    print(f"Final success rate: {suc}/{n} = {round(suc / n * 100, 1)}%")
    return 0


def selftest():
    class P:
        def __init__(self, p, q):
            self.p, self.q = p, q

    class Obj:
        def __init__(self, p):
            self._p = p

        def get_pose(self):
            return P(self._p, [1, 0, 0, 0])

    class Env:
        pass
    e = Env()
    e.pot = Obj([0.1, 0.2, 0.3])
    e.table = Obj([0, 0, 0.7])
    e.robot = object()
    names = [n for n, _ in poses_of(e)]
    assert names == ["pot", "table"]
    h1 = sha_arrays(poses_of(e))
    e.pot._p = [0.1, 0.2, 0.30001]
    assert sha_arrays(poses_of(e)) != h1
    it = flat_items("v", dict(cam_head=dict(color=np.zeros((2, 2, 3))),
                              x=[1, 2], s="txt"))
    assert [k for k, _ in it] == ["v/cam_head/color", "v/x/0", "v/x/1"]
    print("самопроверка rt3_cases пройдена: позы объектов (get_pose), "
          "отпечатки чувствительны к сдвигу, обход наблюдения детерминирован")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="RoboTwin: банк случаев")
    ap.add_argument("cmd", choices=("bank", "eval"))
    ap.add_argument("--root", default=os.environ.get("ROBOTWIN_ROOT"))
    ap.add_argument("--source", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bank", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--frequency", type=int, default=15)
    ap.add_argument("--log-dir", default=None,
                    help="eval: каталог журнала оценки (rt3_cases.jsonl)")
    ap.add_argument("--limit", type=int, default=0,
                    help="eval: только первые случаи банка (smoke)")
    a = ap.parse_args()
    if a.cmd == "eval" and not a.log_dir:
        raise SystemExit("eval: нужен --log-dir")
    if not a.root:
        raise SystemExit("нужен ROBOTWIN_ROOT")
    official(a.root)
    from test_render import Sapien_TEST
    Sapien_TEST()                            # как официальный клиент
    try:
        return build_bank(a) if a.cmd == "bank" else eval_bank(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}", flush=True)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
