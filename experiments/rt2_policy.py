#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 2: модель для официальной оценки (интерфейс XPolicyLab
ModelTemplate: update_obs / get_action / reset / prepare_case /
on_trial_end).

Вход — последнее наблюдение (сервер XPolicyLab уже декодировал кадры в RGB,
см. decode_obs_images; здесь каналы не переставляются): три камеры
cam_head | cam_left_wrist | cam_right_wrist -> та же склейка 224×672, что в
обучении (rt1_common.tile); состояние — 14 каналов в порядке обучения
(левая рука 6, левый схват, правая рука 6, правый схват), нормировка q01/q99
обучающих эпизодов подготовки; инструкция — obs["instruction"]; промпт и
обработка — rt1_common.prompt_batch; bfloat16 autocast, как в обучении.
Выход — первые N_EXEC = 8 из 20 команд чанка, каждая — словарь абсолютных
целей: left_arm_joint_state[6], left_ee_joint_state[1],
right_arm_joint_state[6], right_ee_joint_state[1] (рука — в пределах URDF
после ограничителя readout, схват — sigmoid в [0, 1]). Официальный клиент
исполняет все возвращённые команды и только потом снова просит действие,
поэтому внутренний буфер не нужен; reset очищает наблюдение и счётчики.
Журнал эпизода (JSON-строка в <log_dir>/<задача>/episodes.jsonl): задача, seed, инструкция,
успех (из trial_end клиента), вызовы, задержки, проверки выхода.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt1_common as rc  # noqa: E402

STATE_KEYS = (("left_arm_joint_state", 6), ("left_ee_joint_state", 1),
              ("right_arm_joint_state", 6), ("right_ee_joint_state", 1))
VERSION = "rt2_policy1"


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def state_vector(state):
    """state XPolicyLab -> [14] в порядке каналов обучения (CH_KEYS)."""
    parts = []
    for k, d in STATE_KEYS:
        if k not in state:
            raise SystemExit(f"в наблюдении нет state.{k}")
        v = np.asarray(state[k], np.float64).reshape(-1)
        if v.shape != (d,):
            raise SystemExit(f"state.{k} формы {v.shape}, ожидалось ({d},)")
        parts.append(v)
    x = np.concatenate(parts)
    if not np.isfinite(x).all():
        raise SystemExit("состояние не конечно")
    return x


def frames_of(vision):
    out = []
    for c in rc.CAMS:
        cam = vision.get(c)
        if cam is None or "color" not in cam:
            raise SystemExit(f"в наблюдении нет vision.{c}.color")
        f = np.asarray(cam["color"])
        if f.dtype != np.uint8 or f.ndim != 3 or f.shape[2] != 3:
            raise SystemExit(f"кадр {c} формы {f.shape} {f.dtype}")
        out.append(f)
    return out


def pack(chunk, n_exec):
    """[20, 14] -> список n_exec словарей команд XPolicyLab."""
    a = np.asarray(chunk, np.float32)
    if a.shape != (rc.T_CHUNK, rc.N_CH):
        raise SystemExit(f"чанк формы {a.shape}")
    out = []
    for t in range(n_exec):
        row, o, act = a[t], 0, {}
        for k, d in STATE_KEYS:
            act[k] = row[o:o + d].copy()
            o += d
        out.append(act)
    return out


MODEL_KEYS = ("ckpt_sha1", "weights_hash", "prep_id", "policy_sha1",
              "factory", "seed", "version")


def load_spec(torch, ck_path, rep_path, prep_id):
    """Чекпойнт RT2 (формат rt2_train.Trainer.check: weights, update, val_L,
    contract, init_hash, weights_hash) -> спецификация модели со сверкой:
    SHA файла = отчёту обучения, отпечаток весов, prep_id подготовки, SHA
    кода контракта."""
    import k21_train as ktr
    rep = json.load(open(rep_path, encoding="utf-8"))
    if sha(ck_path) != rep["best_sha1"]:
        raise SystemExit(f"{ck_path} ≠ отчёту обучения {rep_path}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if "contract" not in ck:
        raise SystemExit(f"{ck_path}: нет контракта обучения (ключи "
                         f"{sorted(ck)})")
    cfg = ck["contract"]
    if ck["weights_hash"] != rep.get("best_weights_hash") or \
            ktr.state_hash(ck["weights"]) != ck["weights_hash"]:
        raise SystemExit("отпечаток весов чекпойнта ≠ отчёту / содержимому")
    if cfg["prep_id"] != prep_id:
        raise SystemExit("чекпойнт обучен на другой подготовке")
    bad = [f for f, s in cfg["code"].items()
           if sha(os.path.join(HERE, f)) != s]
    if bad:
        raise SystemExit(f"код модели изменён после обучения: {bad}")
    meta = dict(ckpt=os.path.abspath(ck_path), ckpt_sha1=sha(ck_path),
                weights_hash=ck["weights_hash"], factory=cfg["factory"],
                seed=int(cfg["seed"]), prep_id=cfg["prep_id"],
                best_update=ck.get("update"),
                policy_sha1=sha(os.path.abspath(__file__)), version=VERSION)
    return dict(ck=ck, contract=cfg, meta=meta)


def build_model(torch, spec, D, device):
    """Модель RT2 на BAR с весами спецификации -> predict(img, s_n, text)."""
    import k21_train as ktr
    import rt2_train as r2
    cfg, ck = spec["contract"], spec["ck"]
    dev = torch.device(device)
    mod_m, arch = r2.factory(cfg["factory"])
    bar = rc.load_bar(dev, "bfloat16")
    m = bar["model"]
    m.action_expert.float()
    mod = mod_m.build(torch)[arch](m, D.L, D.U, checkpoint=False,
                                   seed=int(cfg["seed"])).to(dev)
    own = dict(mod.named_parameters())
    own.update(dict(mod.named_buffers()))
    with torch.no_grad():
        for k, v in ck["weights"].items():
            if k not in own:
                raise SystemExit(f"в модели нет {k}")
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if ktr.state_hash(ktr.trainable_state(mod)) != ck["weights_hash"]:
        raise SystemExit("веса после загрузки ≠ отпечатку чекпойнта")
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)

    def predict(img, s_n, text):
        batch = rc.prompt_batch(bar, [img], [s_n], [text], dev,
                                torch.bfloat16)
        st = torch.as_tensor(np.asarray(s_n, np.float32)[None], device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(batch, rc.POS_OFFSET, st)
        return out["action"][0].float().cpu().numpy()
    return predict


class Model:
    """Модель для XPolicyLab PolicyServer. model_cfg: ckpt, train_report,
    device, log_dir, n_exec (8). Для самопроверки — _predict (функция
    (img, s_n, text) -> [20, 14]) и _prep (q01, q99, L, U) вместо загрузки."""

    def __init__(self, model_cfg):
        self.cfg = dict(model_cfg)
        self.n_exec = int(self.cfg.get("n_exec", rc.N_EXEC))
        if not 1 <= self.n_exec <= rc.T_CHUNK:
            raise SystemExit(f"n_exec {self.n_exec}")
        self.log_dir = self.cfg.get("log_dir")
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        if "_predict" in self.cfg:
            self._predict = self.cfg["_predict"]
            self.q01, self.q99, self.L, self.U = self.cfg["_prep"]
            self.meta = dict(ckpt=None)
        else:
            self._load()
        self.obs = None
        self.case = None
        self.ep = None

    # --- загрузка -------------------------------------------------------------
    def _load(self):
        import torch
        import rt1_data as rd
        D = self.cfg.get("_data") or rd.Data(check=True, check_files=False)
        spec = load_spec(torch, self.cfg["ckpt"], self.cfg["train_report"],
                         D.prep["prep_id"])
        build = self.cfg.get("_build") or build_model
        self._predict = build(torch, spec, D, self.cfg.get("device",
                                                           "cuda:0"))
        self.q01, self.q99, self.L, self.U = D.q01, D.q99, D.L, D.U
        self.meta = spec["meta"]
        print(f"[rt2_policy] загружено: {self.meta}", flush=True)

    # --- интерфейс XPolicyLab -------------------------------------------------
    def update_obs(self, obs):
        self.obs = obs

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("пакетная оценка не используется")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("пакетная оценка не используется")

    def get_action(self):
        try:
            return self._get_action()
        except BaseException as e:      # отказ модели — в журнал эпизода
            if self.ep is not None:
                self.ep.setdefault("errors", []).append(
                    f"{type(e).__name__}: {getattr(e, 'code', None) or e}")
            raise

    def _get_action(self):
        if self.obs is None:
            raise SystemExit("get_action до update_obs")
        t0 = time.perf_counter()
        img = rc.tile(frames_of(self.obs.get("vision") or {}))
        s_n = rc.normalize(state_vector(self.obs.get("state") or {}),
                           self.q01, self.q99)
        text = str(self.obs.get("instruction") or "")
        if not text:
            raise SystemExit("пустая инструкция в наблюдении")
        chunk = np.asarray(self._predict(img, s_n, text), np.float64)
        if chunk.shape != (rc.T_CHUNK, rc.N_CH) or \
                not np.isfinite(chunk).all():
            raise SystemExit(f"выход модели {chunk.shape} не конечен или "
                             f"неверной формы")
        L, U = np.asarray(self.L), np.asarray(self.U)
        if ((chunk < L - 1e-4) | (chunk > U + 1e-4)).any():
            raise SystemExit("выход вне пределов (рука — URDF, схват — [0, 1])")
        ms = (time.perf_counter() - t0) * 1e3
        if self.ep is not None:
            self.ep["calls"] += 1
            self.ep["ms"].append(ms)
            if self.ep["calls"] == 1:
                self.ep["first_chunk_sha1"] = hashlib.sha1(np.ascontiguousarray(
                    chunk[:self.n_exec].astype(np.float32)).tobytes()
                ).hexdigest()[:16]
                self.ep["instruction_obs"] = text
        return pack(chunk, self.n_exec)

    def reset(self):
        self.obs = None

    def prepare_case(self, case_meta=None):
        m = dict(case_meta or {})
        self.ep = dict(task=m.get("task_name"), seed=m.get("seed"),
                       instruction=m.get("instruction"),
                       action_type=m.get("action_type"), calls=0, ms=[],
                       started=time.time())
        self.obs = None

    def on_trial_end(self, result=None):
        r = dict(result or {})
        if self.ep is None:
            self.ep = dict(task=r.get("task_name"), seed=r.get("seed"),
                           calls=0, ms=[], started=time.time())
        ep = dict(self.ep)
        ms = ep.pop("ms")
        ep.update(success=bool(r.get("success")),
                  seed_end=r.get("seed"), n_exec=self.n_exec,
                  ms_median=float(np.median(ms)) if ms else None,
                  ms_p95=float(np.percentile(ms, 95)) if ms else None,
                  seconds=time.time() - ep.pop("started"), model=self.meta)
        if ep["seed_end"] != ep["seed"] or (r.get("task_name") is not None and
                                            r.get("task_name") != ep["task"]):
            ep["mismatch"] = "задача/seed trial_end ≠ prepare_case"
        if self.log_dir:
            d = os.path.join(self.log_dir, str(ep["task"]))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "episodes.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(ep, ensure_ascii=False, default=str) +
                         "\n")
        self.ep = None
        return ep


def selftest():
    import tempfile
    import types
    import importlib.util
    if importlib.util.find_spec("cv2") is None:   # локально без OpenCV
        def resize(f, size, interpolation=None):
            w, h = size
            yi = (np.arange(h) * f.shape[0] // h)
            xi = (np.arange(w) * f.shape[1] // w)
            return f[yi][:, xi]
        sys.modules["cv2"] = types.SimpleNamespace(resize=resize,
                                                   INTER_AREA=3)
    q01, q99 = np.zeros(14), np.full(14, 2.0)
    L = np.full(14, -3.0)
    U = np.full(14, 3.0)
    L[[6, 13]], U[[6, 13]] = 0.0, 1.0
    seen = {}

    def predict(img, s_n, text):
        seen.update(img=img, s_n=np.asarray(s_n), text=text)
        c = np.zeros((20, 14))
        c[:, :6] = np.arange(20)[:, None] * 0.01            # левая рука
        c[:, 6] = 0.25                                       # левый схват
        c[:, 7:13] = -np.arange(20)[:, None] * 0.01          # правая рука
        c[:, 13] = 0.75
        return c
    with tempfile.TemporaryDirectory() as td:
        m = Model(dict(_predict=predict, _prep=(q01, q99, L, U), log_dir=td))
        cam = np.zeros((48, 64, 3), np.uint8)
        obs = dict(vision={c: dict(color=cam + i) for i, c in
                           enumerate(rc.CAMS)},
                   state=dict(left_arm_joint_state=np.arange(6) * 0.1,
                              left_ee_joint_state=[0.5],
                              right_arm_joint_state=-np.arange(6) * 0.1,
                              right_ee_joint_state=[1.0]),
                   instruction="lift the pot")
        m.prepare_case(dict(task_name="lift_pot", seed=100003,
                            instruction="lift the pot", action_type="joint"))
        m.reset()
        m.update_obs(obs)
        acts = m.get_action()
        assert len(acts) == 8 and set(acts[0]) == {k for k, _d in STATE_KEYS}
        assert np.allclose(acts[3]["left_arm_joint_state"], 0.03) and \
            float(acts[3]["left_ee_joint_state"][0]) == 0.25
        assert np.allclose(acts[7]["right_arm_joint_state"], -0.07) and \
            float(acts[7]["right_ee_joint_state"][0]) == 0.75
        # склейка камер в порядке head | left | right, состояние нормировано
        assert seen["img"].shape == (3, 224, 672)
        assert seen["img"][:, 0, 0].tolist() == [0, 0, 0] and \
            seen["img"][:, 0, 224].tolist() == [1, 1, 1] and \
            seen["img"][:, 0, 448].tolist() == [2, 2, 2]
        sv = state_vector(obs["state"])
        assert sv[6] == 0.5 and sv[13] == 1.0 and sv[7] == 0.0
        assert np.allclose(seen["s_n"], rc.normalize(sv, q01, q99))
        assert seen["text"] == "lift the pot"
        m.update_obs(obs)
        m.get_action()
        ep = m.on_trial_end(dict(task_name="lift_pot", seed=100003,
                                 success=True))
        assert ep["success"] and ep["calls"] == 2 and "mismatch" not in ep
        line = json.loads(open(os.path.join(td, "lift_pot",
                                            "episodes.jsonl")).read())
        assert line["seed"] == 100003 and line["task"] == "lift_pot"
        # reset очищает наблюдение; ошибки входа — понятный отказ
        m.reset()
        for bad in (lambda: m.get_action(),
                    lambda: (m.update_obs(dict(obs, state=dict(
                        obs["state"], left_arm_joint_state=np.zeros(5)))),
                             m.get_action()),
                    lambda: (m.update_obs(dict(obs, vision={})),
                             m.get_action()),
                    lambda: (m.update_obs(dict(obs, instruction="")),
                             m.get_action())):
            try:
                bad()
                raise AssertionError("ошибка входа принята")
            except SystemExit:
                pass
        # выход вне пределов и NaN — отказ
        for bad_out in (np.full((20, 14), 5.0), np.full((20, 14), np.nan)):
            mb = Model(dict(_predict=lambda *a, _o=bad_out: _o,
                            _prep=(q01, q99, L, U)))
            mb.update_obs(obs)
            try:
                mb.get_action()
                raise AssertionError("плохой выход принят")
            except SystemExit:
                pass
    # путь _load на настоящем файле в формате rt2_train.Trainer.check
    import types as _t
    import torch
    import k21_train as ktr
    with tempfile.TemporaryDirectory() as td:
        w = {"x.weight": torch.arange(6.0).reshape(2, 3)}
        contract = dict(mode="train", factory="rt1_model:flat24", seed=0,
                        prep_id="P1", code={"rt1_common.py": sha(os.path.join(
                            HERE, "rt1_common.py"))})
        ckp, rp_ = os.path.join(td, "best.pt"), os.path.join(td, "r.json")

        def write(ck):
            torch.save(ck, ckp)
            json.dump(dict(best_sha1=sha(ckp),
                           best_weights_hash=ck.get("weights_hash")),
                      open(rp_, "w"))
        good = dict(weights=w, update=500, val_L=0.1, contract=contract,
                    init_hash="i", weights_hash=ktr.state_hash(w))
        write(good)
        Dd = _t.SimpleNamespace(prep=dict(prep_id="P1"), q01=q01, q99=q99,
                                L=L, U=U)
        built = {}

        def fake_build(torch_, spec, D_, device):
            built.update(spec=spec, device=device)
            return predict
        mm = Model(dict(ckpt=ckp, train_report=rp_, device="cuda:1",
                        _data=Dd, _build=fake_build))
        assert mm.meta["prep_id"] == "P1" and mm.meta["factory"] == \
            "rt1_model:flat24" and built["device"] == "cuda:1"
        assert set(MODEL_KEYS) <= set(mm.meta)
        mm.update_obs(obs)
        assert len(mm.get_action()) == 8
        for mut, word in (
                (lambda c: c.update(contract=dict(contract, prep_id="P2")),
                 "подготовке"),
                (lambda c: c.update(weights_hash="0" * 40), "отпечаток"),
                (lambda c: c.update(contract=dict(contract, code={
                    "rt1_common.py": "0" * 40})), "код"),
                (lambda c: c.update(config=c.pop("contract")), "контракт")):
            ck = dict(good)
            mut(ck)
            write(ck)
            try:
                Model(dict(ckpt=ckp, train_report=rp_, _data=Dd,
                           _build=fake_build))
                raise AssertionError(word)
            except SystemExit as e:
                assert word in str(e.code), (word, e.code)
        write(good)
        open(ckp, "ab").write(b"x")             # файл ≠ отчёту
        try:
            Model(dict(ckpt=ckp, train_report=rp_, _data=Dd,
                       _build=fake_build))
            raise AssertionError("изменённый файл принят")
        except SystemExit:
            pass
    # отказ модели попадает в журнал эпизода
    mb = Model(dict(_predict=lambda *a: np.full((20, 14), np.nan),
                    _prep=(q01, q99, L, U)))
    mb.prepare_case(dict(task_name="lift_pot", seed=100001))
    mb.update_obs(obs)
    try:
        mb.get_action()
    except SystemExit:
        pass
    ep = mb.on_trial_end(dict(task_name="lift_pot", seed=100001,
                              success=False))
    assert ep.get("errors") and "SystemExit" in ep["errors"][0]
    print("самопроверка rt2_policy пройдена: 8 команд из 20, ключи и порядок "
          "каналов (левая рука, левый схват, правая рука, правый схват), "
          "склейка head|left|right, нормировка состояния, инструкция из "
          "наблюдения, журнал эпизода (prepare_case/trial_end), reset, отказы "
          "на неверный вход, NaN и выход за пределы; загрузка чекпойнта "
          "формата Trainer.check (contract) и отказы при чужом prep_id, "
          "отпечатке, коде, ключе config, изменённом файле; ошибка модели — "
          "в журнале эпизода")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль модели; сервер — rt2_eval.py serve")
