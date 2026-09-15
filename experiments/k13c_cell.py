#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13c: одна ячейка детерминированного сравнения. Одна рука, одна задача.

РУКИ

    fast12          forward_joint_fast, нулевой уровень. Исходная политика
                    раннего выхода БЕЗ поправки.
    coarse24        generate с only_blocks(1), нулевой уровень. Полный
                    generate выполняет три BAR-блока, из которых нужен только
                    первый: успех тот же, стоимость втрое меньше. Коды
                    сверяются с первым блоком полного generate на первом
                    батче.
    hicora_d1_det   forward_hicora: 16 наборов коэффициентов, по одному на
                    позицию чанка.
    hicora_t_d1_det forward_hicora_t: один набор из 64 на весь чанк.

ТОЖДЕСТВО ПРОВЕРЯЕТСЯ ИСПОЛНЕНИЕМ. На первом настоящем батче голова HiCoRA-T
обнуляется, и сверяются q0, dZ, Z и ДЕКОДИРОВАННЫЕ ДЕЙСТВИЯ против fast12.
Проверка `dZ == 0` на сохранённых латентах этого не даёт: она пройдёт и при
неверном пути получения черновика, и при чужом декодировании — при нулевой
поправке выход от них не зависит.

ЧЕГО ТОЖДЕСТВО НЕ ПРОВЕРЯЕТ: правильность базиса и rho. При нулевых
коэффициентах любой базис исчезает, поэтому их происхождение подтверждается
только отпечатками, и они пишутся в ячейку.

СЧЁТЧИКИ ПРОХОДОВ. Один проход VLA и одно декодирование на вызов политики —
это заявленное свойство архитектуры, и оно измеряется, а не декларируется.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

H_EXEC = 8
ARMS = ("fast12", "coarse24", "hicora_d1_det", "hicora_t_d1_det")
PREPROCESS = "CenterCrop(196)->Resize(224)"


class Counter:
    """Счётчик вызовов: заявление о «одном проходе» обязано измеряться."""

    def __init__(self):
        self.n = {}

    def wrap(self, obj, name):
        fn = getattr(obj, name)
        self.n.setdefault(name, 0)

        def counted(*a, **k):
            self.n[name] += 1
            return fn(*a, **k)
        setattr(obj, name, counted)
        return fn

    def reset(self):
        for k in self.n:
            self.n[k] = 0


@contextlib.contextmanager
def only_blocks(model, n):
    """Ограничить число BAR-блоков. Приём из K-9i.

    `block_size` вычисляется в __init__ и от подмены не меняется, маска и
    позиции строятся от фактических длин — ограничивается ровно число
    проходов и ничего больше.
    """
    saved = model.num_blocks
    try:
        model.num_blocks = n
        yield
    finally:
        model.num_blocks = saved


def check_identity(t_head, fast_codes, z_fast, a_fast, out_t, decode, tol=0.0):
    """Тождество HiCoRA-T с fast12 при ОБНУЛЁННОЙ голове. Исполнением.

    Требуется ТОЧНОЕ совпадение: при dZ = 0 путь HiCoRA-T отличается от fast12
    только сложением нуля, поэтому допуск нулевой. Ненулевой допуск скрыл бы
    расхождение в пути получения черновика.
    """
    import torch
    res = dict(tol=tol)
    res["q0_equal"] = bool(torch.equal(out_t["q0"], fast_codes))
    res["dz_max"] = float(out_t["dz"].abs().max())
    res["z_max_diff"] = float((out_t["z"] - z_fast).abs().max())
    a_t = decode(out_t["z"])
    res["action_max_diff"] = float(np.abs(a_t - a_fast).max())
    res["action_equal"] = bool(np.array_equal(a_t, a_fast))
    res["ok"] = bool(res["q0_equal"] and res["dz_max"] == 0.0
                     and res["z_max_diff"] == 0.0 and res["action_equal"])
    return res


def selftest():
    import torch

    # --- счётчик считает вызовы -------------------------------------------
    class Fake:
        def go(self, x):
            return x * 2
    f, c = Fake(), Counter()
    c.wrap(f, "go")
    assert f.go(3) == 6 and f.go(4) == 8
    assert c.n["go"] == 2, c.n
    c.reset()
    assert c.n["go"] == 0

    # --- ограничение числа блоков возвращается на место -------------------
    class M:
        num_blocks = 3
    m = M()
    with only_blocks(m, 1):
        assert m.num_blocks == 1
    assert m.num_blocks == 3
    try:
        with only_blocks(m, 1):
            raise ValueError("сбой внутри")
    except ValueError:
        pass
    assert m.num_blocks == 3, "число блоков не восстановлено после ошибки"

    # --- тождество: допуск нулевой, любое расхождение видно ---------------
    q = torch.randint(0, 5, (2, 4))
    z = torch.randn(2, 4, 6)
    a = np.random.default_rng(0).normal(size=(2, 8, 7)).astype(np.float32)
    out_ok = dict(q0=q.clone(), dz=torch.zeros_like(z), z=z.clone())
    r = check_identity(None, q, z, a, out_ok, lambda _z: a)
    assert r["ok"] and r["dz_max"] == 0.0 and r["action_equal"], r
    # чужой черновик
    out_bad = dict(out_ok, q0=q + 1)
    assert not check_identity(None, q, z, a, out_bad, lambda _z: a)["ok"]
    # ненулевая поправка
    out_bad2 = dict(q0=q.clone(), dz=torch.full_like(z, 1e-7), z=z + 1e-7)
    r2 = check_identity(None, q, z, a, out_bad2, lambda _z: a)
    assert not r2["ok"] and r2["dz_max"] > 0, r2
    # расхождение ТОЛЬКО в действиях: латенты совпали, декодирование чужое
    r3 = check_identity(None, q, z, a, out_ok, lambda _z: a + 1e-6)
    assert not r3["ok"] and r3["action_max_diff"] > 0, r3
    print("самопроверка k13c_cell пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--head", default=None, help="s0 или s1 для рук с головой")
    ap.add_argument("--ckpt")
    ap.add_argument("--policy-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--hicora-ckpt", default=None)
    ap.add_argument("--hicora-t-ckpt", default=None)
    ap.add_argument("--basis", default="data/k13a_traj_basis")
    ap.add_argument("--task-id", type=int, default=3)
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--init-start", type=int, default=30)
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=H_EXEC)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-seed-mode", default="block")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--expect-depth", type=int, default=12)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--verify-coarse", action="store_true", default=True,
                    help="сверить коды only_blocks(1) с первым блоком полного "
                         "generate на первом батче")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    for need in ("arm", "ckpt", "out"):
        if not getattr(a, need):
            ap.error(f"нужен --{need.replace('_', '-')}")
    if a.arm == "hicora_d1_det" and not a.hicora_ckpt:
        ap.error("--arm hicora_d1_det требует --hicora-ckpt")
    if a.arm == "hicora_t_d1_det" and not a.hicora_t_ckpt:
        ap.error("--arm hicora_t_d1_det требует --hicora-t-ckpt")
    run(a)
    return 0


def run(a):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import k9h_multiarm_gate as k9h
    import k12b_protocol as kb

    if a.pos_offset is not None:
        pos_off, off_sha = int(a.pos_offset), None
    else:
        tb = json.load(open(a.offset_table))
        pos_off = int(tb["offsets_by_suite"][a.task_suite][a.task_id])
        off_sha = k9h.file_sha12(a.offset_table)

    from torchvision.transforms.v2 import CenterCrop, Compose, Resize
    import actioncodec  # noqa: F401
    import joint12_vla as jv
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       get_envs, process_state, prompt_template,
                       seed_everything)
    import hicora_vla as hv
    import hicora_t_vla as ht

    cfg = get_cfg(os.path.join(root, a.cfg_path))
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt

    state_ids = [a.init_start + i for i in range(a.n_envs)]
    roll_seed = k9h.rollout_seed(a.seed, a.init_start, a.rollout_seed_mode)
    seed_everything(roll_seed)
    envs, task_desc = get_envs(a.task_suite,
                               {"task_id": a.task_id, "image_size": 224},
                               a.n_envs)
    print(f"  среды до модели: задача {a.task_id}, состояния {state_ids}, "
          f"сид {roll_seed}", flush=True)

    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))

    import copy
    Cls = ht.make_hicora_t_class(make_joint12_class(SmolVLABlockwiseAR))
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    res_norm_orig = copy.deepcopy(model.action_expert.norm)
    model.init_joint_fast(depth=a.expect_depth, head_dtype=dt)
    j_obj = torch.load(a.policy_ckpt, map_location="cpu", weights_only=False)
    joint_sha = k9h.file_sha12(a.policy_ckpt)
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in j_obj["state"].items():
            if k not in own:
                raise SystemExit(f"ключ вне модели: {k}")
            own[k].data = v.to(dev, torch.float32)

    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(ii))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    model.set_codebooks(E)
    model.set_res_norm(res_norm_orig.to(dev))
    model.taps, model.q0_depth = (12, 18, 24), a.expect_depth
    model.n_layers_total = len(model.action_expert.layers)
    rn = hashlib.sha1()
    for k_ in sorted(model.res_norm.state_dict()):
        rn.update(k_.encode())
        rn.update(np.ascontiguousarray(model.res_norm.state_dict()[k_]
                                       .detach().float().cpu().numpy()
                                       ).tobytes())
    rn_sha = rn.hexdigest()[:12]

    head_sha = basis_sha = rho_sha = None
    t_head = None
    if a.arm == "hicora_d1_det":
        h_obj = torch.load(a.hicora_ckpt, map_location="cpu",
                           weights_only=False)
        k9h.check_hicora_ckpt(h_obj, f"hicora_s{h_obj['seed']}",
                              h_obj.get("target", "coef"))
        head_sha = k9h.file_sha12(a.hicora_ckpt)
        if h_obj["res_norm_sha1"] != rn_sha:
            raise SystemExit(f"res_norm sha {rn_sha}, голова обучена на "
                             f"{h_obj['res_norm_sha1']}")
        pref = h_obj["cache"]
        B = np.load(pref + ".basis.npy").astype(np.float32)
        rho = np.load(pref + ".rho.npy").astype(np.float32)
        basis_sha, rho_sha = h_obj["basis_sha1"], h_obj["rho_sha1"]
        d_h = int(model.fast_head.in_features)
        kw = dict(rank=int(h_obj["rank"]), hidden=int(h_obj.get("hidden", 512)),
                  proj=int(h_obj.get("proj", 64)))
        head = hv.make_residual_head()(d_h, int(E.shape[-1]), **kw).to(dev)
        head.set_basis(torch.as_tensor(B))
        head.set_rho(torch.as_tensor(rho))
        st = {k[len("hicora_head."):]: v for k, v in h_obj["state"].items()}
        with torch.no_grad():
            for k, v in st.items():
                head.state_dict()[k].copy_(v.to(dev, torch.float32))
        model.hicora_head = head.eval()
    elif a.arm == "hicora_t_d1_det":
        t_obj = torch.load(a.hicora_t_ckpt, map_location="cpu",
                           weights_only=False)
        head_sha = k9h.file_sha12(a.hicora_t_ckpt)
        if t_obj["res_norm_sha1"] != rn_sha:
            raise SystemExit(f"res_norm sha {rn_sha}, голова обучена на "
                             f"{t_obj['res_norm_sha1']}")
        bmeta = json.load(open(f"{a.basis}.meta.json"))
        B = np.load(f"{a.basis}.basis.npy")
        rho = np.load(f"{a.basis}.rho.npy")
        for nm, arr, want in (("базис", B, t_obj["basis_sha1"]),
                              ("rho", rho, t_obj["rho_sha1"])):
            got = hashlib.sha1(
                np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
            if got != want:
                raise SystemExit(f"{nm} на диске {got}, голова обучена на "
                                 f"{want}")
        basis_sha, rho_sha = t_obj["basis_sha1"], t_obj["rho_sha1"]
        model.init_hicora_t(q0_depth=a.expect_depth, rank=int(t_obj["rank"]),
                            proj=int(t_obj["proj"]),
                            hidden=int(t_obj["hidden"]),
                            n_pos=int(t_obj["n_pos"]))
        t_head = model.hicora_t_head
        t_head.set_basis(torch.as_tensor(B).to(dev))
        t_head.set_rho(torch.as_tensor(rho).to(dev))
        with torch.no_grad():
            for k, v in t_obj["state"].items():
                if k in ("basis", "rho", "basis_set", "rho_set"):
                    continue
                t_head.state_dict()[k].copy_(v.to(dev, torch.float32))
        t_head.eval()
        print(f"  HiCoRA-T: ранг {t_obj['rank']}, ||rho|| "
              f"{t_obj['rho_norm']:.4f}, эпоха {t_obj['selected_epoch']}, "
              f"снято на подтверждении "
              f"{100 * (t_obj.get('val_confirm_gain') or 0):.1f}%", flush=True)

    cnt = Counter()
    cnt.wrap(model, "forward_taps")
    cnt.wrap(model, "generate")
    cnt.wrap(codec, "_decode")
    ac16 = torch.autocast("cuda", dtype=torch.float16)

    def decode_latent(z):
        x, _ = codec._decode(z.float(), embodiment_ids=0)
        return x[..., :7].detach().float().cpu().numpy()

    ident, coarse_check, per_call = None, None, []
    t0 = time.time()
    try:
        n = a.n_envs
        obs = envs.reset(options=[{"init_state_id": j} for j in state_ids])
        reward, done = np.zeros(n), np.zeros(n, bool)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n)
        for _ in range(a.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)

        def _h(parts):
            return hashlib.sha1(np.ascontiguousarray(
                np.concatenate(parts).astype(np.float32)).tobytes()
            ).hexdigest()[:16]
        init_hash = [_h([obs["state"][i].ravel(),
                         obs["agentview_image"][i].ravel() / 255.0,
                         obs["robot0_eye_in_hand_image"][i].ravel() / 255.0])
                     for i in range(n)]

        calls = steps = 0
        while not np.all(done) and steps < a.max_steps:
            st = ((process_state(obs["state"]) - STATE_Q01)
                  / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0)
            i1 = tf(torch.tensor(
                obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2))
            i2 = tf(torch.tensor(
                obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()
            ).permute(0, 3, 1, 2))
            image = torch.cat([i1, i2], dim=-1)
            msgs = []
            for i in range(n):
                m = prompt_template(
                    st[i], None, task_desc,
                    mode=cfg.MODEL.vla_processor.kwargs.mode,
                    action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                    action_token_len=cfg.MODEL.action_processor.token_len)
                m[1]["content"] = m[1]["content"][1:]
                msgs.append(m)
            texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
            batch = proc(text=texts,
                         images=[[image[i].numpy()] for i in range(n)],
                         return_tensors="pt", padding=True, padding_side="left",
                         action_processor_kwargs={"embodiment_ids": 0})
            batch = dict_apply(lambda x: x.to(dev, dt), batch)
            cnt.reset()

            with torch.no_grad(), ac16:
                if a.arm == "coarse24":
                    with only_blocks(model, 1):
                        toks = model.generate(**batch, position_offset=pos_off,
                                              do_sample=False,
                                              initial_position_shift=1)
                    codes = toks[:, :16]
                    if coarse_check is None and a.verify_coarse:
                        # ОДИН БЛОК ПРОТИВ ПЕРВОГО БЛОКА ПОЛНОГО generate
                        full = model.generate(**batch, position_offset=pos_off,
                                              do_sample=False,
                                              initial_position_shift=1)
                        same = bool(torch.equal(codes, full[:, :16]))
                        coarse_check = dict(equal=same,
                                            n_full=int(full.shape[1]),
                                            n_one=int(toks.shape[1]))
                        if not same:
                            raise SystemExit(
                                "only_blocks(1) дал не те коды, что первый "
                                "блок полного generate: экономия изменила бы "
                                "политику")
                        cnt.n["generate"] -= 1      # сверка не в счёт
                    z = E[0][codes.long()]
                    a_exec = decode_latent(z)
                else:
                    v_, p_ = model.build_inputs(position_offset=pos_off,
                                                **batch)
                    if a.arm == "fast12":
                        out = model.forward_joint_fast(
                            vlm_inputs_embeds=v_,
                            attention_mask=batch.get("attention_mask"),
                            position_ids=p_)
                        z = E[0][out["pred_codes"].long()]
                        a_exec = decode_latent(z)
                    elif a.arm == "hicora_d1_det":
                        out = model.forward_hicora(
                            vlm_inputs_embeds=v_,
                            attention_mask=batch.get("attention_mask"),
                            position_ids=p_)
                        a_exec = decode_latent(out["z"])
                    else:
                        out = model.forward_hicora_t(
                            vlm_inputs_embeds=v_,
                            attention_mask=batch.get("attention_mask"),
                            position_ids=p_)
                        a_exec = decode_latent(out["z"])
                        if ident is None:
                            # ТОЖДЕСТВО НА НАСТОЯЩЕМ БАТЧЕ: голова обнуляется,
                            # и выход обязан совпасть с fast12 точно
                            jf = model.forward_joint_fast(
                                vlm_inputs_embeds=v_,
                                attention_mask=batch.get("attention_mask"),
                                position_ids=p_)
                            z_f = E[0][jf["pred_codes"].long()]
                            a_f = decode_latent(z_f)
                            sd = {k: v.detach().clone()
                                  for k, v in t_head.state_dict().items()}
                            with torch.no_grad():
                                t_head.net[-1].weight.zero_()
                                t_head.net[-1].bias.zero_()
                                # ОБНУЛЕНИЕ ПРОВЕРЯЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ:
                                # если голова, которую исполняет модель, —
                                # другой объект, zero_() не даст эффекта, и
                                # тождество провалится без объяснения
                                w_max = float(
                                    t_head.net[-1].weight.abs().max())
                                b_max = float(t_head.net[-1].bias.abs().max())
                                same_obj = (model.hicora_t_head is t_head)
                                out0 = model.forward_hicora_t(
                                    vlm_inputs_embeds=v_,
                                    attention_mask=batch.get("attention_mask"),
                                    position_ids=p_)
                                c_max = float(out0["coeffs"].abs().max())
                                # РЕШАЮЩЕЕ РАЗЛИЧЕНИЕ: если коэффициенты не
                                # изменились, прямой проход идёт мимо
                                # обнулённого модуля
                                c_before = float(out["coeffs"].abs().max())
                                mod_id = (id(t_head.net[-1])
                                          == id(model.hicora_t_head.net[-1]))
                                direct = float(torch.tanh(
                                    t_head.mean_coeffs(
                                        model.res_norm(
                                            model.forward_taps(
                                                vlm_inputs_embeds=v_,
                                                attention_mask=batch.get(
                                                    "attention_mask"),
                                                position_ids=p_)[24]).float(),
                                        out0["z0"])).abs().max())
                            print(f"    обнуление: |W| {w_max:.2e}, |b| "
                                  f"{b_max:.2e}, та же голова {same_obj}, тот "
                                  f"же модуль {mod_id}\n    |c| до "
                                  f"{c_before:.3e} -> после {c_max:.3e}; "
                                  f"прямой вызов головы даёт {direct:.3e}",
                                  flush=True)
                            ident = check_identity(
                                t_head, jf["pred_codes"], z_f, a_f, out0,
                                decode_latent)
                            t_head.load_state_dict(sd)
                            print(f"  тождество с fast12: q0 "
                                  f"{ident['q0_equal']}, |dZ| "
                                  f"{ident['dz_max']:.2e}, |dZ_latent| "
                                  f"{ident['z_max_diff']:.2e}, |d действий| "
                                  f"{ident['action_max_diff']:.2e} -> "
                                  f"{'ОК' if ident['ok'] else 'НЕ СОШЛОСЬ'}",
                                  flush=True)
                            if not ident["ok"]:
                                raise SystemExit(
                                    "тождество не выполнено: при нулевой "
                                    "голове HiCoRA-T обязана совпадать с "
                                    "fast12 точно")
                            # сверка не должна попасть в счётчики
                            cnt.n["forward_taps"] -= 2
                            cnt.n["_decode"] -= 2
            if not per_call:
                per_call = dict(cnt.n)
            calls += 1
            action = np.copy(a_exec)
            action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
            action[..., -1] = -action[..., -1]
            for t in range(a.horizon):
                if np.all(done) or steps >= a.max_steps:
                    break
                obs, r_, done, _ = envs.step(action[:, t])
                reward = np.clip(reward + r_, 0, 1)
                steps += 1
        eps = [dict(task_id=int(a.task_id), state_id=int(state_ids[i]),
                    suite=str(a.task_suite), env_index=i,
                    init_hash_full=init_hash[i],
                    success=bool(reward[i] >= 1.0), env_steps=steps,
                    policy_calls=calls, rollout_seed=roll_seed)
               for i in range(n)]
    finally:
        envs.close()

    # ОДИН ПРОХОД И ОДНО ДЕКОДИРОВАНИЕ — ИЗМЕРЕНО, А НЕ ЗАЯВЛЕНО
    want_pass = 1 if a.arm != "coarse24" else 0
    want_gen = 1 if a.arm == "coarse24" else 0
    if per_call.get("forward_taps", 0) != want_pass or \
            per_call.get("generate", 0) != want_gen or \
            per_call.get("_decode", 0) != 1:
        raise SystemExit(f"на вызов политики пришлось {per_call}, ожидалось "
                         f"forward_taps={want_pass}, generate={want_gen}, "
                         f"_decode=1")

    cell = dict(
        arm=a.arm, head=a.head, stage="k13c_det", suite=a.task_suite,
        task_id=int(a.task_id), task_ids=[int(a.task_id)],
        state_ids=state_ids, init_start=int(a.init_start),
        n_envs=int(a.n_envs), episodes=eps, horizon=int(a.horizon),
        max_steps=int(a.max_steps), waiting_steps=int(a.waiting_steps),
        ensemble="off", seed=int(a.seed), rollout_seed=roll_seed,
        rollout_seed_mode=a.rollout_seed_mode, ckpt=a.ckpt,
        joint_sha1=joint_sha, head_sha1=head_sha, basis_sha1=basis_sha,
        rho_sha1=rho_sha, res_norm_sha1=rn_sha, pos_offset=pos_off,
        offset_table_sha1=off_sha, preprocess=PREPROCESS, image_size=224,
        device=str(dev), dtype=a.dtype, identity=ident,
        coarse_one_block=coarse_check, calls_per_policy=per_call,
        task_description=task_desc,
        code_version=kb.code_version([
            os.path.abspath(__file__), os.path.join(here, "hicora_t_vla.py"),
            os.path.join(here, "hicora_vla.py")]),
        script_sha1=k9h.file_sha12(os.path.abspath(__file__)),
        minutes=(time.time() - t0) / 60.0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    json.dump(cell, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    succ = sum(1 for e in eps if e["success"])
    print(f"\n  {a.arm}{'/' + a.head if a.head else ''}, задача {a.task_id}, "
          f"состояния {state_ids[0]}..{state_ids[-1]}: успех {succ}/{n}, "
          f"вызовов {calls}, на вызов {per_call}")
    print(f"  сохранено: {a.out} ({cell['minutes']:.1f} мин)")


if __name__ == "__main__":
    sys.exit(main())
