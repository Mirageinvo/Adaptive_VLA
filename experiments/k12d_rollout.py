#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-12d: раскатка HiCoRA-G с записью входа головы и правдоподобий.

ОТЛИЧИЯ ОТ K-11g И ПОЧЕМУ ИМЕННО ТАКИЕ

  1. ГОЛОВА СЧИТАЕТСЯ В fp32, ВНЕ autocast. В K-11g она шла под autocast
     fp16, и там это было безразлично: измерялся успех. Здесь по сохранённым
     сэмплам ПЕРЕСЧИТЫВАЕТСЯ log pi, и при fp16 разрешение mu (~1e-3
     относительных) крупнее шага при lr 3e-6: отношение правдоподобий осталось
     бы единицей из-за округления, и шаг был бы неотличим от его отсутствия.
     Ствол остаётся под autocast fp16 — это та же политика, что в K-11e/K-11g.
     Цена перехода ИЗМЕРЯЕТСЯ и пишется в ячейку (fp32_vs_autocast_head):
     из-за неё базовые доли успеха придётся перемерить в этом же режиме, а не
     брать из K-11g. Сравнивается та же голова внутри autocast против неё же
     вне его — приводить голову к fp16 нельзя, set_basis отвергает такой базис
     как неортонормированный, и это не придирка проверки, а реальная потеря
     точности на произведении из 512 слагаемых.

  2. ПИШЕТСЯ ВХОД ГОЛОВЫ, А НЕ ЕЁ ВЫХОД. Сохраняются h24 (fp32) и ИНДЕКСЫ
     черновика q0 (int16) вместо самого z0: z0 = codebook0[q0] восстанавливается
     точно, а книга одна на прогон. Это на два порядка меньше места и нет
     потери точности — при fp16 для z0 пересчёт log pi разошёлся бы с
     записанным.

  3. ШУМ ЗАВИСИТ ОТ НОМЕРА ШАГА. Соль потока eps включает сид RL и индекс
     шага: иначе каждая итерация переиспользовала бы тот же шум, и градиент
     считался бы по одним и тем же реализациям.

  4. ЗАПИСЫВАЮТСЯ ТОЛЬКО АКТИВНЫЕ СРЕДЫ. У завершившейся среды действие
     игнорируется, и её вызов в градиенте — это вес, приклеенный к решению,
     которое ни на что не влияло.
"""
import argparse
import ast
import hashlib
import json
import os
import sys
import time

import numpy as np

H_EXEC = 8
SAT_THR = 0.99


def eps_salt(rl_seed, step_index):
    """Соль потока шума. Разные шаги — разные реализации (см. п.3 шапки)."""
    return int(hashlib.sha1(f"k12d|{int(rl_seed)}|{int(step_index)}".encode()
                            ).hexdigest()[:8], 16)


class Store:
    """Накопитель записей. Раздельно по полям, чтобы склейка была одной
    операцией и типы не поехали: int16 для индексов, fp32 для всего остального.
    """

    FIELDS = ("h", "q0", "u", "mu", "logp", "task", "state", "call")

    def __init__(self):
        self.rows = {k: [] for k in self.FIELDS}
        self.n = 0

    def add(self, *, h, q0, u, mu, logp, task, state, call):
        import torch
        k = h.shape[0]
        for nm, v, want in (("q0", q0, k), ("u", u, k), ("mu", mu, k),
                            ("logp", logp, k)):
            if v.shape[0] != want:
                raise ValueError(f"{nm}: {v.shape[0]} записей против {want}")
        if int(q0.max()) > 32767 or int(q0.min()) < 0:
            raise ValueError(f"индекс черновика {int(q0.max())} не влезает в "
                             f"int16")
        self.rows["h"].append(h.detach().to("cpu", torch.float32))
        self.rows["q0"].append(q0.detach().to("cpu", torch.int16))
        self.rows["u"].append(u.detach().to("cpu", torch.float32))
        self.rows["mu"].append(mu.detach().to("cpu", torch.float32))
        self.rows["logp"].append(logp.detach().to("cpu", torch.float32))
        self.rows["task"].append(torch.full((k,), int(task), dtype=torch.int32))
        self.rows["state"].append(torch.as_tensor(
            [int(s) for s in state], dtype=torch.int32))
        self.rows["call"].append(torch.full((k,), int(call), dtype=torch.int32))
        self.n += k

    def stack(self):
        import torch
        if not self.n:
            raise ValueError("буфер пуст: ни одного активного вызова политики")
        return {k: torch.cat(self.rows[k]) for k in self.FIELDS}


def build_meta(**kw):
    """Мета-запись ячейки. Отдельной функцией, потому что её проверяет
    собственная самопроверка ЧЕРЕЗ k12e.check_rollouts: контракт воркера и
    агрегатора один раз уже разошёлся (K-11i), и ловить это в конце долгой
    раскатки недопустимо."""
    need = ("protocol_sha1", "replica", "stage", "sigma", "episodes",
            "d_hidden", "rank", "head_sha1", "codebooks_sha1", "joint_sha1")
    miss = [k for k in need if kw.get(k) in (None, "", [])]
    if miss:
        raise ValueError(f"в мете нет обязательных полей: {miss}")
    m = dict(kw)
    m["head_precision"] = "fp32"
    return m


def save_cell(path, meta, data):
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(dict(meta=meta, data=data), tmp)
    os.replace(tmp, path)          # атомарно: оборванная запись не оставит
    return path                    # полуячейку, которую шаг примет за целую


def write_cb0(path, E0):
    """Книга черновика — один файл на прогон, с проверкой при повторе.

    Если файл уже есть и отличается, это ДРУГАЯ книга: восстановленный z0 не
    соответствовал бы записанным правдоподобиям.
    """
    import torch
    sha = hashlib.sha1(np.ascontiguousarray(
        E0.float().cpu().numpy()).tobytes()).hexdigest()[:12]
    if os.path.exists(path):
        old = torch.load(path, map_location="cpu", weights_only=False)
        t = old["codebook0"] if isinstance(old, dict) else old
        osha = hashlib.sha1(np.ascontiguousarray(
            t.float().cpu().numpy()).tobytes()).hexdigest()[:12]
        if osha != sha:
            raise SystemExit(f"{path} содержит книгу {osha}, а сейчас {sha}: "
                             f"черновик восстановился бы из другой книги")
        return sha
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(dict(codebook0=E0.float().cpu(), sha1=sha), tmp)
    os.replace(tmp, path)
    return sha


# ------------------------------ самопроверка ------------------------------

def check_orchestration(path=None):
    """Порядок в main проверяется по дереву разбора, а не на глаз.

    В K-11h рефакторинг оставил весь хвост main недостижимым после return, и
    прогон завершился с нулевым кодом, ничего не измерив. Здесь это ловится
    без запуска.
    """
    src = open(path or os.path.abspath(__file__)).read()
    tree = ast.parse(src)
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if len(fn) != 1:
        raise AssertionError("в файле не одна функция run")
    # ни одного return на верхнем уровне run: именно так в K-11h весь хвост
    # стал недостижим, и прогон завершился нулём, ничего не измерив
    tops = [i for i, st in enumerate(fn[0].body) if isinstance(st, ast.Return)]
    if tops:
        raise AssertionError(f"return на верхнем уровне run (оператор "
                             f"{tops[0]}): хвост стал бы недостижим")
    mains = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "main"]
    if len(mains) != 1:
        raise AssertionError("в файле не одна функция main")
    if "run" not in [c.func.id for c in ast.walk(mains[0])
                     if isinstance(c, ast.Call)
                     and isinstance(c.func, ast.Name)]:
        raise AssertionError("main не вызывает run")
    names, order = [], []
    for st in ast.walk(fn[0]):
        if isinstance(st, ast.Call) and isinstance(st.func, ast.Name):
            names.append((st.lineno, st.func.id))
        if isinstance(st, ast.Call) and isinstance(st.func, ast.Attribute):
            names.append((st.lineno, st.func.attr))
    seq = [n for _l, n in sorted(names)]
    for a, b, why in (("get_envs", "from_pretrained",
                       "среды обязаны создаваться ДО модели: в K-11g "
                       "обратный порядок менял расход ГСЧ и политику"),
                      ("get_envs", "stack", "склейка после раскатки"),
                      ("stack", "save_cell", "сохранение после склейки"),
                      ("build_meta", "save_cell", "мета до сохранения")):
        if a not in seq or b not in seq:
            raise AssertionError(f"в main нет вызова {a} или {b}")
        if seq.index(a) > seq.index(b):
            raise AssertionError(f"{a} вызывается после {b}: {why}")
    for must in ("check_rollouts", "write_cb0", "parity"):
        if must not in src:
            raise AssertionError(f"в файле нет {must}")
    return True


def check_names(root="third_party/actioncodec"):
    """Все внешние имена существуют, и подписи те, что предполагает код.

    Запускается ЗА СЕКУНДЫ и до длинной раскатки. Однажды в этой серии
    импортируемые пути были попросту выдуманы, и выяснилось это в конце
    прогона; здесь такая ошибка стоит одной секунды.
    """
    root = os.path.abspath(root)
    for p in (root, os.path.join(root, "src"),
              os.path.dirname(os.path.abspath(__file__)),
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    bad = []

    def need(mod, names, label):
        for nm in names:
            if not hasattr(mod, nm):
                bad.append(f"{label}.{nm} не существует")

    import k9h_multiarm_gate as k9h
    need(k9h, ("file_sha12", "rollout_seed", "check_hicora_ckpt",
               "check_hicora_meta"), "k9h_multiarm_gate")
    import k11g_cell as k11g
    need(k11g, ("eps_seed",), "k11g_cell")
    import k11a_build_hicora_cache as k11a
    need(k11a, ("check_fingerprints", "decoder_probe", "state_sha1"),
         "k11a_build_hicora_cache")
    import hicora_vla as hv
    need(hv, ("make_hicora_class", "make_residual_head"), "hicora_vla")
    import hicora_g as hg
    need(hg, ("make_gaussian_residual_head",), "hicora_g")
    import utils
    need(utils, ("ACTION_Q01", "ACTION_Q99", "STATE_Q01", "STATE_Q99",
                 "VisionLanguageActionProcessor", "dict_apply", "get_cfg",
                 "get_envs", "process_state", "prompt_template",
                 "seed_everything"), "utils")
    import joint12_vla as jv
    need(jv, ("make_joint12_class",), "joint12_vla")
    import k12b_protocol as kb
    need(kb, ("load_protocol", "check_run", "ProtocolError"), "k12b_protocol")
    import k12e_pg_step as k12e
    need(k12e, ("check_rollouts", "concat_buffer", "parity_check",
                "loo_advantage", "one_step"), "k12e_pg_step")

    # подписи, на которые опирается код
    try:
        v = k11g.eps_seed(1, 2, 3, 4)
        if not isinstance(v, int):
            bad.append(f"k11g.eps_seed вернул {type(v).__name__}, а не int")
        if k11g.eps_seed(1, 2, 3, 4) == k11g.eps_seed(1, 2, 3, 5):
            bad.append("k11g.eps_seed не зависит от соли: шаги обучения "
                       "переиспользовали бы тот же шум")
    except TypeError as e:
        bad.append(f"k11g.eps_seed(task, init, call, salt) не вызывается: {e}")
    try:
        k9h.rollout_seed(0, 0, "block")
    except TypeError as e:
        bad.append(f"k9h.rollout_seed(seed, init, mode) не вызывается: {e}")

    # методы модели, которые использует раскатка
    try:
        from smolvla.bar import SmolVLABlockwiseAR
        Cls = hv.make_hicora_class(jv.make_joint12_class(SmolVLABlockwiseAR))
        need(Cls, ("build_inputs", "forward_taps", "q0_from", "init_joint_fast",
                   "set_codebooks", "set_res_norm", "forward_hicora"),
             "класс модели")
    except Exception as e:                          # noqa: BLE001
        bad.append(f"класс модели не собирается: {type(e).__name__}: {e}")

    if bad:
        print("ПРОВЕРКА ИМЁН НЕ ПРОЙДЕНА:")
        for b in bad:
            print(f"  - {b}")
        raise SystemExit(1)
    print("проверка имён пройдена: все внешние имена и подписи на месте")
    return True


def selftest():
    import torch
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import k12b_protocol as kb
    import k12e_pg_step as k12e

    check_orchestration()

    # --- соль шума: разные шаги и сиды дают разные потоки ------------------
    s = {eps_salt(a, b) for a in (0, 1) for b in (0, 1, 2)}
    assert len(s) == 6, "соли совпали: шаги переиспользовали бы тот же шум"
    assert eps_salt(1, 2) == eps_salt(1, 2)

    # --- накопитель: типы, формы, отказы ----------------------------------
    st = Store()
    n_pos, rank, d_h = 4, 3, 6
    st.add(h=torch.randn(2, n_pos, d_h), q0=torch.randint(0, 50, (2, n_pos)),
           u=torch.randn(2, n_pos, rank), mu=torch.randn(2, n_pos, rank),
           logp=torch.randn(2), task=3, state=[7, 8], call=0)
    st.add(h=torch.randn(1, n_pos, d_h), q0=torch.randint(0, 50, (1, n_pos)),
           u=torch.randn(1, n_pos, rank), mu=torch.randn(1, n_pos, rank),
           logp=torch.randn(1), task=3, state=[7], call=1)
    d = st.stack()
    assert d["h"].shape == (3, n_pos, d_h) and d["q0"].dtype == torch.int16
    assert d["task"].tolist() == [3, 3, 3] and d["state"].tolist() == [7, 8, 7]
    assert d["call"].tolist() == [0, 0, 1]
    try:
        st.add(h=torch.randn(2, n_pos, d_h),
               q0=torch.randint(0, 50, (1, n_pos)),
               u=torch.randn(2, n_pos, rank), mu=torch.randn(2, n_pos, rank),
               logp=torch.randn(2), task=3, state=[7, 8], call=2)
    except ValueError:
        pass
    else:
        raise AssertionError("рассогласованные формы приняты")
    try:
        st.add(h=torch.randn(1, n_pos, d_h),
               q0=torch.full((1, n_pos), 40000),
               u=torch.randn(1, n_pos, rank), mu=torch.randn(1, n_pos, rank),
               logp=torch.randn(1), task=3, state=[7], call=3)
    except ValueError:
        pass
    else:
        raise AssertionError("индекс вне int16 принят")
    try:
        Store().stack()
    except ValueError:
        pass
    else:
        raise AssertionError("пустой буфер сохранён")

    # --- КОНТРАКТ С ШАГОМ: мета и данные воркера проходят проверки шага ----
    proto = kb._proto_ok(splits={"train": "0-9", "dev": "10-14",
                                 "final": "15-94"}, n_episodes_final=80)
    sigma = 0.1
    head = k12e._stub_head(d_h, 5, rank, sigma)
    cb0 = torch.randn(50, 5)
    st2 = Store()
    eps_rows = []
    for task in (0, 1):
        for state in (0, 1):
            for call in (0, 1):
                h = torch.randn(1, n_pos, d_h)
                q0 = torch.randint(0, 50, (1, n_pos))
                with torch.no_grad():
                    mu = head.mean_coeffs(h, cb0[q0])
                    u = mu + sigma * torch.randn(mu.shape)
                    lp = head.log_prob_u(u, mu, head.std())
                st2.add(h=h, q0=q0, u=u, mu=mu, logp=lp, task=task,
                        state=[state], call=call)
            eps_rows.append(dict(task_id=task, state_id=state,
                                 success=bool((task + state) % 2 == 0),
                                 init_hash_full=f"h{task}{state}",
                                 env_steps=16, policy_calls=2, n_records=2,
                                 rollout_seed=123))
    meta = build_meta(protocol_sha1=proto["sha1"], replica="d10_rl0",
                      stage="train", sigma=sigma, episodes=eps_rows,
                      d_hidden=d_h, rank=rank, head_sha1="a" * 12,
                      codebooks_sha1="b" * 12, joint_sha1="c" * 12)
    meta["path"] = "cell.pt"
    cell = dict(meta=meta, data=st2.stack())
    info = k12e.check_rollouts([cell], proto, replica="d10_rl0",
                              stage="train", sigma=sigma)
    assert info["n_episodes"] == 4, info
    buf = k12e.concat_buffer([cell], cb0, torch.device("cpu"))
    par = k12e.parity_check(head, buf, head.std())
    assert par["ok"], par        # правдоподобие воркера пересчитывается шагом
    # неполная мета отвергается ЗДЕСЬ, а не в конце раскатки
    try:
        build_meta(protocol_sha1="x", replica="r", stage="train", sigma=0.1,
                   episodes=eps_rows, d_hidden=d_h, rank=rank)
    except ValueError as e:
        assert "head_sha1" in str(e), e
    else:
        raise AssertionError("мета без sha принята")

    # --- книга черновика: повтор с другой книгой отвергается ---------------
    import tempfile
    tmpd = tempfile.mkdtemp(prefix="k12d_")
    p = os.path.join(tmpd, "cb0.pt")
    sha = write_cb0(p, cb0)
    assert write_cb0(p, cb0) == sha, "повторная запись той же книги отказала"
    try:
        write_cb0(p, cb0 + 1.0)
    except SystemExit:
        pass
    else:
        raise AssertionError("другая книга принята под тем же путём")

    # --- сохранение ячейки атомарно и читается шагом -----------------------
    cp = os.path.join(tmpd, "cell.pt")
    save_cell(cp, meta, cell["data"])
    back = torch.load(cp, map_location="cpu", weights_only=False)
    assert set(back["data"]) == set(Store.FIELDS)
    assert not os.path.exists(cp + ".tmp")
    print("самопроверка k12d_rollout пройдена")


# -------------------------------- прогон ----------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-names", action="store_true")
    ap.add_argument("--protocol", default="data/k12b/protocol.json")
    ap.add_argument("--replica", required=False)
    ap.add_argument("--stage", default="train", choices=["train", "dev"])
    ap.add_argument("--step-index", type=int, default=0)
    ap.add_argument("--sigma", type=float, required=False)
    ap.add_argument("--ckpt", required=False)
    ap.add_argument("--policy-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--head-ckpt", required=False,
                    help="чекпойнт D1 (s0 или s1) — начало обучения")
    ap.add_argument("--resume-head", default=None,
                    help="голова после предыдущего шага K-12e")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--init-start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=H_EXEC)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rl-seed", type=int, default=0)
    ap.add_argument("--rollout-seed-mode", default="block",
                    choices=["block", "fixed"])
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--expect-depth", type=int, default=12)
    ap.add_argument("--expect-hicora-target", default="coef")
    ap.add_argument("--cb0-out", default="data/k12d/cb0.pt")
    ap.add_argument("--out", required=False)
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.check_names:
        check_names(args.root)
    else:
        check_orchestration()
        for need in ("replica", "sigma", "ckpt", "head_ckpt", "out"):
            if getattr(args, need) in (None, ""):
                ap.error(f"нужен --{need.replace('_', '-')}")
        run(args)


def run(args):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(args.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import k12b_protocol as kb
    import k12e_pg_step as k12e

    proto = kb.load_protocol(args.protocol)
    sigma = round(float(args.sigma), 6)
    if sigma not in [round(float(s), 6) for s in proto["sigma_grid"]]:
        raise SystemExit(f"sigma={sigma} вне зарегистрированной сетки "
                         f"{proto['sigma_grid']}")
    state_ids = [args.init_start + i for i in range(args.n_envs)]
    kb.check_run(proto, dict(stage=args.stage, protocol_sha1=proto["sha1"],
                             state_ids=state_ids, task_ids=[args.task_id],
                             sigma=sigma, replica=args.replica))
    if proto["step"].get("head_precision") != "fp32":
        raise SystemExit("протокол требует не fp32 для головы, а эта раскатка "
                         "считает её в fp32")

    if args.pos_offset is not None:
        pos_off, off_sha = int(args.pos_offset), None
    else:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; задайте --pos-offset")
        tb = json.load(open(args.offset_table))
        pos_off = int(tb["offsets_by_suite"][args.task_suite][args.task_id])
        off_sha = hashlib.sha1(
            open(args.offset_table, "rb").read()).hexdigest()[:12]

    import torch
    import k9h_multiarm_gate as k9h
    import k11g_cell as k11g        # eps_seed берётся из проверенного кода

    h_obj = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
    k9h.check_hicora_ckpt(h_obj, "hicora", args.expect_hicora_target)
    head_sha = k9h.file_sha12(args.head_ckpt)
    j_obj = torch.load(args.policy_ckpt, map_location="cpu",
                       weights_only=False)
    joint_sha = k9h.file_sha12(args.policy_ckpt)

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
    import hicora_g as hg
    import k11a_build_hicora_cache as k11a

    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    # --- СРЕДЫ ДО МОДЕЛИ: порядок и расход ГСЧ те же, что в K-9h/K-11g ----
    roll_seed = k9h.rollout_seed(args.seed, args.init_start,
                                 args.rollout_seed_mode)
    seed_everything(roll_seed)
    envs, task_desc = get_envs(args.task_suite,
                               {"task_id": args.task_id, "image_size": 224},
                               args.n_envs)
    print(f"  среды созданы до модели: задача {args.task_id}, сред "
          f"{args.n_envs}, состояния {state_ids}, сид {roll_seed}", flush=True)

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tf = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    act_range = np.asarray(ACTION_Q99, float) - np.asarray(ACTION_Q01, float)
    if not np.all(act_range[:7] > 0):
        raise SystemExit("диапазон действия не положителен")

    import copy
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    res_norm_orig = copy.deepcopy(model.action_expert.norm)
    model.init_joint_fast(depth=args.expect_depth, head_dtype=dtype)
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in j_obj["state"].items():
            if k not in own:
                raise SystemExit(f"ключ вне модели: {k}")
            own[k].data = v.to(dev, torch.float32)
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(ii))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    model.__class__ = hv.make_hicora_class(type(model))
    model.set_codebooks(E)
    model.set_res_norm(res_norm_orig.to(dev))
    model.taps, model.q0_depth = (12, 18, 24), args.expect_depth
    model.n_layers_total = len(model.action_expert.layers)
    if max(model.taps) != model.n_layers_total:
        raise SystemExit(f"последний отвод {max(model.taps)} против "
                         f"{model.n_layers_total} слоёв")
    rn = hashlib.sha1()
    for k_ in sorted(model.res_norm.state_dict()):
        v_ = model.res_norm.state_dict()[k_]
        rn.update(k_.encode())
        rn.update(np.ascontiguousarray(
            v_.detach().float().cpu().numpy()).tobytes())
    rn_sha = rn.hexdigest()[:12]
    if rn_sha != h_obj["res_norm_sha1"]:
        raise SystemExit(f"res_norm sha {rn_sha}, голова обучена на "
                         f"{h_obj['res_norm_sha1']}")
    pref = h_obj["cache"]
    bp, rp, mp = pref + ".basis.npy", pref + ".rho.npy", pref + ".meta.json"
    for f_ in (bp, rp, mp):
        if not os.path.exists(f_):
            raise SystemExit(f"нет {f_}")
    for f_, want_, lbl in ((bp, h_obj["basis_sha1"], "базис"),
                           (rp, h_obj["rho_sha1"], "предел")):
        got_ = k9h.file_sha12(f_)
        if got_ != want_:
            raise SystemExit(f"{lbl} sha {got_}, голова обучена на {want_}")
    meta_c = json.load(open(mp))
    k9h.check_hicora_meta(meta_c, args.ckpt, joint_sha,
                          k9h.file_sha12(hv.__file__),
                          k9h.file_sha12(jv.__file__))
    cb_sha = hashlib.sha1(np.ascontiguousarray(
        E.cpu().numpy().astype(np.float32)).tobytes()).hexdigest()[:12]
    k11a.check_fingerprints(meta_c, dict(
        codebooks_sha1=cb_sha,
        decoder_probe=k11a.decoder_probe(codec, E, dev),
        codec_state_sha1=k11a.state_sha1(codec)))
    cb0_sha = write_cb0(args.cb0_out, E[0])

    B = np.load(bp).astype(np.float32)
    rho = np.load(rp).astype(np.float32)
    rho_norm = float(np.linalg.norm(rho))
    d_h = int(model.fast_head.in_features)
    kw = dict(rank=int(h_obj["rank"]), hidden=int(h_obj.get("hidden", 512)),
              proj=int(h_obj.get("proj", 64)))
    st_h = {k[len("hicora_head."):]: v for k, v in h_obj["state"].items()}
    stray = [k for k in h_obj["state"] if not k.startswith("hicora_head.")]
    if stray:
        raise SystemExit(f"ключи вне hicora_head.: {stray[:5]}")
    heads = {}
    for which, cls in (("det", hv.make_residual_head()),
                       ("gau", hg.make_gaussian_residual_head())):
        h_ = cls(d_h, int(E.shape[-1]), **kw).to(dev, torch.float32)
        h_.set_basis(torch.as_tensor(B))
        h_.set_rho(torch.as_tensor(rho))
        want = {k for k in h_.state_dict() if k.startswith(("proj.", "net."))}
        if set(st_h) != want:
            raise SystemExit(f"набор весов головы не совпал ({which})")
        with torch.no_grad():
            for k, v in st_h.items():
                h_.state_dict()[k].copy_(v.to(dev, torch.float32))
        h_.eval()
        heads[which] = h_
    det_h, gau_h = heads["det"], heads["gau"]
    # ОТДЕЛЬНОЙ ГОЛОВЫ В fp16 НЕТ, И ЭТО НЕ УПРОЩЕНИЕ. set_basis проверяет
    # ортонормированность с допуском 1e-4, а в fp16 произведение B^T B на 512
    # слагаемых этот допуск не выдерживает: голова, приведённая к fp16, просто
    # не построилась бы. В K-11g fp16 и не было — там голова с буферами fp32
    # считалась ПОД autocast. Цена точности мерится тем же способом: та же
    # голова внутри autocast против неё же вне его.

    if args.resume_head:
        sd = torch.load(args.resume_head, map_location="cpu",
                        weights_only=False)
        if sd.get("protocol_sha1") != proto["sha1"]:
            raise SystemExit("продолжение от головы под другим протоколом")
        if sd.get("replica") != args.replica:
            raise SystemExit(f"голова от реплики {sd.get('replica')}, а "
                             f"раскатка для {args.replica}")
        gau_h.load_state_dict({k: v.to(dev, torch.float32)
                               for k, v in sd["state"].items()})
        print(f"  продолжение от {args.resume_head}, шаг "
              f"{sd.get('step_index')}", flush=True)
    with torch.no_grad():
        gau_h.log_std.fill_(float(np.log(sigma)))
    std = gau_h.std().detach()
    if abs(float(std.max()) - sigma) > 1e-6 or \
            float(std.min()) != float(std.max()):
        raise SystemExit(f"std политики {float(std.max())} не равна sigma "
                         f"{sigma}: действие сэмплировалось бы под одним "
                         f"распределением, а правдоподобие под другим")

    ac16 = torch.autocast("cuda", dtype=torch.float16)
    parity = {"ok": False}
    store = Store()
    salt = eps_salt(args.rl_seed, args.step_index)

    def decode_latent(z):
        x, _ = codec._decode(z.float(), embodiment_ids=0)
        return x[..., :7].detach().float().cpu().numpy()

    t0 = time.time()
    try:
        n = args.n_envs
        obs = envs.reset(options=[{"init_state_id": j} for j in state_ids])
        reward = np.zeros(n)
        done = np.zeros(n, bool)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n)
        for _ in range(args.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)

        def _h(parts):
            return hashlib.sha1(np.ascontiguousarray(
                np.concatenate(parts).astype(np.float32)).tobytes()
            ).hexdigest()[:16]
        init_hash_full = [
            _h([obs["state"][i].ravel(),
                obs["agentview_image"][i].ravel() / 255.0,
                obs["robot0_eye_in_hand_image"][i].ravel() / 255.0])
            for i in range(n)]

        calls = steps = 0
        sat = float("nan")        # если цикл не выполнится ни разу, мета всё
        n_rec = np.zeros(n, int)  # равно собирается, а не падает на NameError
        while not np.all(done) and steps < args.max_steps:
            state = ((process_state(obs["state"]) - STATE_Q01)
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
                    state[i], None, task_desc,
                    mode=cfg.MODEL.vla_processor.kwargs.mode,
                    action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                    action_token_len=cfg.MODEL.action_processor.token_len)
                m[1]["content"] = m[1]["content"][1:]
                msgs.append(m)
            texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
            batch = proc(text=texts,
                         images=[[image[i].numpy()] for i in range(n)],
                         return_tensors="pt", padding=True,
                         padding_side="left",
                         action_processor_kwargs={"embodiment_ids": 0})
            batch = dict_apply(lambda x: x.to(dev, dtype), batch)
            active = ~done.copy()
            # СТВОЛ — ПОД autocast fp16, как в K-11e/K-11g: политика та же
            with torch.no_grad(), ac16:
                v_, p_ = model.build_inputs(position_offset=pos_off, **batch)
                taps = model.forward_taps(
                    vlm_inputs_embeds=v_,
                    attention_mask=batch.get("attention_mask"),
                    position_ids=p_)
                n_lay = int(taps["layers_run"])
                _, q0 = model.q0_from(taps[model.q0_depth])
                z0 = model.codebooks[0][q0]
                h24 = model.res_norm(taps[max(model.taps)]).float()
                if not parity["ok"]:
                    dz_ac, _c_ac = det_h(h24, z0)     # путь K-11g: fp32-голова
                    model.hicora_head = det_h         # внутри autocast fp16
                    o_fw = model.forward_hicora(
                        vlm_inputs_embeds=v_,
                        attention_mask=batch.get("attention_mask"),
                        position_ids=p_)
            # ГОЛОВА — В fp32, ВНЕ autocast: см. п.1 шапки
            with torch.no_grad():
                h32, z32 = h24.float(), z0.float()
                o_mean = gau_h(h32, z32, deterministic=True)
                gen = torch.Generator(device=h32.device)
                gen.manual_seed(k11g.eps_seed(args.task_id, args.init_start,
                                              calls, salt) % (2 ** 63))
                eps = torch.empty_like(o_mean["mu"]).normal_(generator=gen)
                o_exec = gau_h(h32, z32, u=o_mean["mu"] + sigma * eps)
                if not parity["ok"]:
                    dz_d, _c = det_h(h32, z32)
                    d_head = float((o_mean["dz"] - dz_d).abs().max())
                    # ЭТА сверка воспроизводит проверку K-11g один в один:
                    # обе величины считаны под autocast, поэтому порог 1e-4
                    # здесь тот же, что там, и она ГЕЙТИРУЕТ прогон
                    d_full = float((dz_ac.float()
                                    - o_fw["dz"].float()).abs().max())
                    # а ЭТА — не гейт, а измеренная цена перехода головы в
                    # fp32: из-за неё базовые доли успеха придётся перемерить
                    d_prec = float((dz_d - dz_ac.float()).abs().max())
                    parity = dict(gauss_mean_vs_d1_fp32=d_head,
                                  d1_vs_forward_hicora_autocast=d_full,
                                  fp32_vs_autocast_head=d_prec,
                                  rho_norm=rho_norm, layers_run=n_lay,
                                  ok=bool(d_head <= 1e-4 and d_full <= 1e-4
                                          and n_lay == 24))
                    print(f"  паритет: гауссова(mean) против D1 в fp32 "
                          f"{d_head:.2e}; D1 против forward_hicora под "
                          f"autocast {d_full:.2e}; ЦЕНА перехода в fp32 "
                          f"{d_prec:.2e} при ||rho||={rho_norm:.4f}; слоёв "
                          f"{n_lay}", flush=True)
                    if not parity["ok"]:
                        raise SystemExit(
                            "ПАРИТЕТ НЕ СОШЁЛСЯ: в режиме среднего гауссова "
                            "голова обязана давать в fp32\n  ровно то же, что "
                            "D1, иначе обучение стартует не из D1")
                if not torch.isfinite(o_exec["dz"]).all():
                    raise SystemExit("в поправке nan или inf")
                dzn = float(torch.linalg.norm(o_exec["dz"], dim=-1).max())
                if dzn > rho_norm + 1e-4:
                    raise SystemExit(f"||dz|| = {dzn:.4f} превысила предел "
                                     f"{rho_norm:.4f}")
                a_exec = decode_latent(z32 + o_exec["dz"])
                sat = float((o_exec["coeffs"].abs() > SAT_THR).float().mean())
            # ТОЛЬКО АКТИВНЫЕ СРЕДЫ (п.4 шапки)
            sel = np.flatnonzero(active)
            if sel.size:
                idx = torch.as_tensor(sel, device=h32.device)
                store.add(h=h32.index_select(0, idx),
                          q0=q0.index_select(0, idx),
                          u=o_exec["u"].index_select(0, idx),
                          mu=o_mean["mu"].index_select(0, idx),
                          logp=o_exec["log_prob_u"].index_select(0, idx),
                          task=args.task_id,
                          state=[state_ids[i] for i in sel.tolist()],
                          call=calls)
                n_rec[sel] += 1
            calls += 1
            action = np.copy(a_exec)
            action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
            action[..., -1] = -action[..., -1]
            for t in range(args.horizon):
                if np.all(done) or steps >= args.max_steps:
                    break
                obs, r_, done, _ = envs.step(action[:, t])
                reward = np.clip(reward + r_, 0, 1)
                steps += 1
        eps_rows = [dict(task_id=int(args.task_id),
                         state_id=int(state_ids[i]),
                         env_index=i, init_hash_full=init_hash_full[i],
                         success=bool(reward[i] >= 1.0), env_steps=steps,
                         policy_calls=calls, n_records=int(n_rec[i]),
                         rollout_seed=roll_seed) for i in range(n)]
    finally:
        envs.close()

    data = store.stack()
    meta = build_meta(
        protocol_sha1=proto["sha1"], replica=args.replica, stage=args.stage,
        step_index=int(args.step_index), sigma=sigma, episodes=eps_rows,
        d_hidden=d_h, rank=int(h_obj["rank"]), head_sha1=head_sha,
        codebooks_sha1=cb_sha, joint_sha1=joint_sha, cb0_sha1=cb0_sha,
        cb0_path=args.cb0_out, res_norm_sha1=rn_sha,
        basis_sha1=h_obj["basis_sha1"], rho_sha1=h_obj["rho_sha1"],
        rho_norm=rho_norm, hicora_seed=h_obj.get("seed"),
        selected_epoch=h_obj.get("selected_epoch"),
        resume_head=args.resume_head, eps_salt=int(salt),
        rl_seed=int(args.rl_seed), seed=int(args.seed),
        rollout_seed=roll_seed, rollout_seed_mode=args.rollout_seed_mode,
        suite=args.task_suite, task_description=task_desc,
        init_start=int(args.init_start), n_envs=int(args.n_envs),
        horizon=int(args.horizon), max_steps=int(args.max_steps),
        waiting_steps=int(args.waiting_steps), pos_offset=pos_off,
        offset_table_sha1=off_sha, image_size=224, ckpt=args.ckpt,
        parity=parity, sat_frac_last=sat, device=str(dev),
        trunk_dtype=args.dtype, n_records=int(store.n),
        script_sha1=k9h.file_sha12(os.path.abspath(__file__)),
        step_script_sha1=k9h.file_sha12(k12e.__file__),
        protocol_script_sha1=k9h.file_sha12(kb.__file__),
        hicora_g_sha1=k9h.file_sha12(hg.__file__),
        hicora_vla_sha1=k9h.file_sha12(hv.__file__),
        joint12_vla_sha1=k9h.file_sha12(jv.__file__),
        k11g_cell_sha1=k9h.file_sha12(k11g.__file__),
        k9h_sha1=k9h.file_sha12(k9h.__file__),
        minutes=(time.time() - t0) / 60.0)
    meta["path"] = args.out
    # ЯЧЕЙКА ПРОВЕРЯЕТСЯ ТЕМ ЖЕ КОДОМ, ЧТО БУДЕТ ЕЁ ЧИТАТЬ: рассогласование
    # воркера и агрегатора один раз уже обесценило целый перебор
    k12e.check_rollouts([dict(meta=meta, data=data)], proto,
                        replica=args.replica, stage=args.stage, sigma=sigma)
    save_cell(args.out, meta, data)
    succ = sum(1 for e in eps_rows if e["success"])
    print(f"\n  задача {args.task_id}, состояния {state_ids}: успех "
          f"{succ}/{n}, вызовов {calls}, записей {store.n}, "
          f"насыщение {100 * sat:.2f}%")
    print(f"  сохранено: {args.out} ({meta['minutes']:.1f} мин)")


if __name__ == "__main__":
    main()
