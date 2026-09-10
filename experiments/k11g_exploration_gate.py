"""K-11g: окно исследования для HiCoRA-G. Диагностический go/no-go до PPO.

ЗАЧЕМ. У гауссовой головы 16 x rank случайных координат на чанк. Есть два
вырожденных режима, и оба дороги, если обнаружить их уже внутри RL: при малой
sigma шум не меняет исполняемое действие вовсе и PPO стоит на месте, при
большой — каждый сэмпл в насыщении tanh, поправка становится шумом и стратегия
рушится с первого же обновления. Этот стенд ищет окно между ними ЗА ОДИН
короткий прогон, до всякого обучения.

ЧТО ЭТО НЕ ЕСТЬ. Не статистическое доказательство. Пилот фиксирован заранее,
но 50 эпизодов на ячейку не дают интервалов, по которым можно утверждать
не-худшесть. Правило здесь — инженерное: «есть ли вообще режим, в котором
поправка меняет поведение, не разрушая стратегию».

ПИЛОТ ЗАФИКСИРОВАН ДО ЗАПУСКА:
  головы s0 и s1 — обе обязательны;
  sigma в {0, 0.03, 0.10, 0.30, 0.50}, где 0 означает ДЕТЕРМИНИРОВАННОЕ
    среднее, а не log(0);
  ensemble=off, H=8, все 10 задач, 5 общих начальных состояний на задачу;
  итого 2 x 5 x 50 = 500 эпизодов.

ОДИН И ТОТ ЖЕ ПОТОК eps ДЛЯ РАЗНЫХ sigma. Шум выводится из (задача, начальное
состояние, номер вызова), а не из общего генератора. Иначе разные sigma
отличались бы и масштабом, и самой случайностью, и сравнивать их было бы
нельзя.

СРАВНЕНИЕ НА ОДНОМ СОСТОЯНИИ. На каждом стохастическом вызове декодируются ДВА
действия из одних и тех же h24 и z0: исполняемое сэмплированное и
контрфактическое по среднему. Траектории потом расходятся, и без этого
«насколько шум меняет действие» мерилось бы на разных состояниях.

ENSEMBLE=OFF НЕ СЛУЧАЙНО. При включённом ансамбле несколько случайных чанков
смешиваются, и один сэмпл перестаёт однозначно отвечать за следующие H
действий — приписывание заслуги становится непрозрачным ровно в том месте, где
оно нужно.

Запуск:
    python3 experiments/k11g_exploration_gate.py --selftest
    python experiments/k11g_exploration_gate.py --ckpt <hf> --policy-ckpt ...
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_POS, N_LEVEL = 16, 3
SIGMAS = (0.0, 0.03, 0.10, 0.30, 0.50)
HEADS = ("s0", "s1")
# ПОРОГИ ОКНА. Зафиксированы до запуска; правило инженерное, не статистическое.
MIN_CHANGED = 0.10     # доля вызовов с существенным изменением действия
CHANGE_THR = 0.01      # «существенное» — более 1% диапазона действия
MAX_SAT = 0.10         # доля насыщенных координат |tanh(u)| > 0.99
MAX_DROP = 10.0        # падение успеха относительно своей детерминированной руки, пп
SAT_THR = 0.99


def eps_seed(task, init, call, salt=0):
    """Сид шума из (задача, начальное состояние, номер вызова).

    НЕ ИЗ ОБЩЕГО ГЕНЕРАТОРА. Тогда при разных sigma поток eps совпадает, и
    разница между ячейками — только масштаб шума. С общим генератором
    отличались бы и масштаб, и сама случайность.
    """
    h = hashlib.sha1(f"k11g|{int(task)}|{int(init)}|{int(call)}|{int(salt)}"
                     .encode()).digest()
    return int.from_bytes(h[:8], "little")


def action_change(sampled, mean, act_range, thr=CHANGE_THR):
    """Насколько сэмпл изменил действие относительно среднего.

    Нормируется на ДИАПАЗОН действия по каждому каналу, а не на норму самого
    действия: около нуля относительная величина взрывается и «изменение на
    1000%» означало бы сдвиг на микрон.

    Схват считается отдельно: он бинарный по смыслу, и усреднять его с
    непрерывными каналами нельзя.
    """
    s = np.asarray(sampled, float)
    m = np.asarray(mean, float)
    if s.shape != m.shape:
        raise ValueError(f"формы {s.shape} и {m.shape} не совпадают")
    r = np.asarray(act_range, float)
    if r.ndim != 1 or r.shape[0] != s.shape[-1] or not np.all(r > 0):
        raise ValueError("диапазон действия должен быть положительным "
                         "вектором по каналам")
    d = np.abs(s - m) / r
    cont = d[..., :-1]
    return dict(rms=float(np.sqrt((cont ** 2).mean())),
                max=float(cont.max()),
                changed=bool(cont.max() > thr),
                grip_flip=bool(np.any(np.sign(s[..., -1])
                                      != np.sign(m[..., -1]))))


def read_window(cells, min_changed=MIN_CHANGED, max_sat=MAX_SAT,
                max_drop=MAX_DROP):
    """Есть ли sigma, годная для PPO. Правило: ОДНА И ТА ЖЕ на обеих головах.

    Четыре условия, и все обязательны:
      инварианты не нарушены (предел амплитуды, конечность);
      существенно меняется не менее min_changed вызовов;
      насыщение ниже max_sat;
      успех падает не более чем на max_drop пп относительно СВОЕЙ
        детерминированной руки, а не чужой.

    Если проходит несколько sigma, берётся МИНИМАЛЬНАЯ: большая амплитуда
    исследования без нужды — это риск разрушить стратегию, а не запас.
    """
    out, passing = {}, []
    for sg in sorted({s for _, s in cells}):
        if sg == 0.0:
            continue
        per, ok = {}, True
        for hd in HEADS:
            c = cells.get((hd, sg))
            base = cells.get((hd, 0.0))
            if c is None or base is None:
                per[hd] = dict(reason="нет ячейки")
                ok = False
                continue
            drop = (base["success"] - c["success"]) * 100.0
            r = dict(changed=c["changed_frac"], sat=c["sat_frac"],
                     success=c["success"], det_success=base["success"],
                     drop_pp=drop, invariants_ok=bool(c["invariants_ok"]),
                     ok_changed=bool(c["changed_frac"] >= min_changed),
                     ok_sat=bool(c["sat_frac"] < max_sat),
                     ok_drop=bool(drop <= max_drop))
            r["ok"] = bool(r["invariants_ok"] and r["ok_changed"]
                           and r["ok_sat"] and r["ok_drop"])
            per[hd] = r
            ok = ok and r["ok"]
        out[sg] = dict(per_head=per, ok=bool(ok))
        if ok:
            passing.append(sg)
    return dict(per_sigma=out, passing=sorted(passing),
                chosen=(min(passing) if passing else None),
                thresholds=dict(min_changed=min_changed, max_sat=max_sat,
                                max_drop=max_drop, change_thr=CHANGE_THR,
                                sat_thr=SAT_THR))


def selftest():
    # --- поток шума ---------------------------------------------------------
    # ОДИН И ТОТ ЖЕ для разных sigma, РАЗНЫЙ для разных вызовов и состояний.
    assert eps_seed(3, 10, 7) == eps_seed(3, 10, 7)
    assert eps_seed(3, 10, 7) != eps_seed(3, 10, 8)
    assert eps_seed(3, 10, 7) != eps_seed(3, 11, 7)
    assert eps_seed(3, 10, 7) != eps_seed(4, 10, 7)
    assert eps_seed(3, 10, 7) != eps_seed(3, 10, 7, salt=1)
    seeds = {eps_seed(t, i, c) for t in range(10) for i in range(5)
             for c in range(40)}
    assert len(seeds) == 10 * 5 * 40, "сиды столкнулись"

    # --- изменение действия -------------------------------------------------
    rng_ = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0])
    a = np.zeros((8, 7))
    b = np.zeros((8, 7))
    r0 = action_change(a, b, rng_)
    assert r0["rms"] == 0.0 and not r0["changed"] and not r0["grip_flip"]
    b2 = b.copy()
    b2[0, 0] = 0.02                      # 2% диапазона
    assert action_change(a, b2, rng_)["changed"]
    b3 = b.copy()
    b3[0, 0] = 0.005                     # 0.5% — не существенно
    assert not action_change(a, b3, rng_)["changed"]
    # НОРМИРОВКА НА ДИАПАЗОН, А НЕ НА САМО ДЕЙСТВИЕ: сдвиг 0.02 в канале с
    # диапазоном 2.0 это 1%, а в канале с диапазоном 1.0 — 2%.
    b4 = b.copy(); b4[0, 3] = 0.02
    assert not action_change(a, b4, rng_)["changed"]
    # СХВАТ ОТДЕЛЬНО И НЕ ПОПАДАЕТ В rms
    g1 = np.zeros((4, 7)); g1[:, -1] = 1.0
    g2 = np.zeros((4, 7)); g2[:, -1] = -1.0
    rg = action_change(g1, g2, rng_)
    assert rg["grip_flip"] and rg["rms"] == 0.0, rg
    for bad, why in (((np.zeros((2, 7)), np.zeros((3, 7)), rng_), "формы"),
                     ((a, b, np.ones(6)), "длина диапазона"),
                     ((a, b, np.zeros(7)), "нулевой диапазон")):
        try:
            action_change(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"принято: {why}")

    # --- правило окна -------------------------------------------------------
    def cell(succ, changed, sat, inv=True):
        return dict(success=succ, changed_frac=changed, sat_frac=sat,
                    invariants_ok=inv)

    base = {}
    for hd in HEADS:
        base[(hd, 0.0)] = cell(0.90, 0.0, 0.0)
    good = dict(base)
    for hd in HEADS:
        good[(hd, 0.03)] = cell(0.88, 0.05, 0.00)   # шум ничего не меняет
        good[(hd, 0.10)] = cell(0.87, 0.35, 0.02)   # рабочее окно
        good[(hd, 0.30)] = cell(0.80, 0.90, 0.04)   # тоже проходит
        good[(hd, 0.50)] = cell(0.40, 0.99, 0.50)   # разрушает
    w = read_window(good)
    assert w["passing"] == [0.10, 0.30], w["passing"]
    # МИНИМАЛЬНАЯ ИЗ ПРОШЕДШИХ: лишняя амплитуда — риск, а не запас.
    assert w["chosen"] == 0.10
    assert not w["per_sigma"][0.03]["ok"], "шум без эффекта прошёл"
    assert not w["per_sigma"][0.50]["ok"], "разрушающая sigma прошла"
    assert 0.0 not in w["per_sigma"], "контрольная рука попала в кандидаты"
    # ОБЕ ГОЛОВЫ: хватает одной негодной, чтобы sigma не прошла
    one_bad = dict(good)
    one_bad[("s1", 0.10)] = cell(0.87, 0.02, 0.02)
    assert read_window(one_bad)["chosen"] == 0.30
    # ПАДЕНИЕ СЧИТАЕТСЯ ОТ СВОЕЙ ДЕТЕРМИНИРОВАННОЙ РУКИ, А НЕ ОТ ЧУЖОЙ
    shifted = dict(good)
    shifted[("s1", 0.0)] = cell(0.70, 0.0, 0.0)
    shifted[("s1", 0.10)] = cell(0.68, 0.35, 0.02)
    assert read_window(shifted)["per_sigma"][0.10]["ok"], \
        "сравнение пошло с чужой опорой"
    # НАРУШЕННЫЕ ИНВАРИАНТЫ — ОТКАЗ, даже если всё прочее идеально
    inv_bad = dict(good)
    inv_bad[("s0", 0.10)] = cell(0.90, 0.40, 0.01, inv=False)
    assert not read_window(inv_bad)["per_sigma"][0.10]["ok"]
    # ОТСУТСТВИЕ ЯЧЕЙКИ — ОТКАЗ, а не пропуск проверки
    miss = {k: v for k, v in good.items() if k != ("s1", 0.10)}
    assert not read_window(miss)["per_sigma"][0.10]["ok"]
    miss2 = {k: v for k, v in good.items() if k != ("s0", 0.0)}
    assert not read_window(miss2)["per_sigma"][0.10]["ok"], \
        "отсутствие контрольной руки прошло"
    # НИ ОДНОЙ ГОДНОЙ — chosen=None, а не «возьмём хоть что-то»
    none_ok = dict(base)
    for hd in HEADS:
        none_ok[(hd, 0.10)] = cell(0.20, 0.99, 0.80)
    assert read_window(none_ok)["chosen"] is None

    print("самопроверка k11g пройдена: поток шума одинаков для разных sigma "
          "и различен\n  по задаче, состоянию и вызову; изменение действия "
          "нормировано на диапазон,\n  схват считается отдельно; окно требует "
          "обеих голов, сравнивает со СВОЕЙ\n  детерминированной рукой, "
          "отвергает и бездействующий, и разрушающий шум,\n  а при отсутствии "
          "ячейки или нарушенном инварианте — отказывает")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--policy-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--hicora-s0",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s0.pt")
    ap.add_argument("--hicora-s1",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s1.pt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--task-suite", default="10")
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--init-start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--waiting-steps", type=int, default=10)
    ap.add_argument("--sigmas", default=",".join(str(s) for s in SIGMAS))
    ap.add_argument("--expect-depth", type=int, default=12)
    ap.add_argument("--expect-hicora-target", default="coef")
    ap.add_argument("--out", default="data/k11g/exploration.json")
    ap.add_argument("--run-tag", default="k11g")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt")

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"нет каталога ActionCodec: {root}")
    sys.path.insert(0, root)

    import torch
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
    import k9h_multiarm_gate as k9h
    import k11a_build_hicora_cache as k11a
    import k11e_protocol as kp

    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    sigmas = tuple(float(x) for x in args.sigmas.split(","))
    if 0.0 not in sigmas:
        raise SystemExit("sigma=0 обязательна: это детерминированный контроль, "
                         "от которого считается падение успеха")
    tasks = tuple(int(x) for x in args.tasks.split(","))

    seed_everything(0)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tf = Compose([Resize(512), CenterCrop(512)])
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    # ДИАПАЗОН ДЕЙСТВИЯ ПО КАНАЛАМ — из тех же квантилей, что масштабируют
    # действие в раскатке. Нормировать изменение на саму величину нельзя:
    # около нуля она взрывается.
    act_range = np.asarray(ACTION_Q99 - ACTION_Q01, float)[:7]
    if not np.all(act_range > 0):
        raise SystemExit("диапазон действия не положителен по всем каналам")

    import copy
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    res_norm_orig = copy.deepcopy(model.action_expert.norm)
    model.init_joint_fast(depth=args.expect_depth, head_dtype=dtype)
    own = dict(model.named_parameters())
    j_obj = torch.load(args.policy_ckpt, map_location="cpu",
                       weights_only=False)
    joint_sha = k9h.file_sha12(args.policy_ckpt)
    with torch.no_grad():
        for k, v in j_obj["state"].items():
            if k not in own:
                raise SystemExit(f"ключ вне модели: {k}")
            own[k].data = v.to(dev, torch.float32)
    depth = len(model.action_expert.layers)
    print(f"  черновик Joint12: {len(j_obj['state'])} тензоров, sha "
          f"{joint_sha}", flush=True)

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

    # --- головы: детерминированная (опора) и гауссова (испытуемая) ----------
    det_heads, gau_heads, head_sha, head_cfg = {}, {}, {}, {}
    for tag, path in (("s0", args.hicora_s0), ("s1", args.hicora_s1)):
        o = torch.load(path, map_location="cpu", weights_only=False)
        k9h.check_hicora_ckpt(o, f"hicora_{tag}", args.expect_hicora_target)
        pref = o["cache"]
        bp, rp, mp = pref + ".basis.npy", pref + ".rho.npy", pref + ".meta.json"
        for f_ in (bp, rp, mp):
            if not os.path.exists(f_):
                raise SystemExit(f"нет {f_}")
        for f_, want_, lbl in ((bp, o["basis_sha1"], "базис"),
                               (rp, o["rho_sha1"], "предел")):
            got_ = k9h.file_sha12(f_)
            if got_ != want_:
                raise SystemExit(f"{lbl} sha {got_}, голова обучена на {want_}")
        if rn_sha != o["res_norm_sha1"]:
            raise SystemExit(f"res_norm sha {rn_sha}, голова {tag} обучена на "
                             f"{o['res_norm_sha1']}")
        meta = json.load(open(mp))
        k9h.check_hicora_meta(meta, args.ckpt, joint_sha,
                              k9h.file_sha12(hv.__file__),
                              k9h.file_sha12(jv.__file__))
        k11a.check_fingerprints(meta, dict(
            codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
                E.cpu().numpy().astype(np.float32)).tobytes()).hexdigest()[:12],
            decoder_probe=k11a.decoder_probe(codec, E, dev),
            codec_state_sha1=k11a.state_sha1(codec)))
        B = np.load(bp).astype(np.float32)
        rho = np.load(rp).astype(np.float32)
        d_h = int(model.fast_head.in_features)
        kw = dict(rank=int(o["rank"]), hidden=int(o.get("hidden", 512)),
                  proj=int(o.get("proj", 64)))
        st_h = {k[len("hicora_head."):]: v for k, v in o["state"].items()}
        stray = [k for k in o["state"] if not k.startswith("hicora_head.")]
        if stray:
            raise SystemExit(f"в чекпойнте {tag} ключи вне hicora_head.: "
                             f"{stray[:5]}")
        for which, cls in (("det", hv.make_residual_head()),
                           ("gau", hg.make_gaussian_residual_head())):
            h_ = cls(d_h, int(E.shape[-1]), **kw).to(dev)
            h_.set_basis(torch.as_tensor(B))
            h_.set_rho(torch.as_tensor(rho))
            want = {k for k in h_.state_dict()
                    if k.startswith(("proj.", "net."))}
            if set(st_h) != want:
                raise SystemExit(f"набор весов головы {tag}/{which} не совпал")
            with torch.no_grad():
                for k, v in st_h.items():
                    h_.state_dict()[k].copy_(v.to(dev, torch.float32))
            h_.eval()
            (det_heads if which == "det" else gau_heads)[tag] = h_
        head_sha[tag] = k9h.file_sha12(path)
        head_cfg[tag] = kp.head_config(o)
        print(f"  голова {tag}: sha {head_sha[tag]}, ранг {o['rank']}, сид "
              f"{o['seed']}", flush=True)
    if head_sha["s0"] == head_sha["s1"]:
        raise SystemExit("обе головы — один файл")
    kp.check_replication(head_cfg["s0"], head_cfg["s1"])
    rho_norm = float(np.linalg.norm(
        gau_heads["s0"].rho.detach().cpu().numpy()))
    print(f"  репликация s0/s1: различаются только сидом; предел "
          f"||rho|| = {rho_norm:.4f}", flush=True)

    ac16 = torch.autocast("cuda", dtype=torch.float16)

    def taps_of(batch):
        """h24 и z0 одним проходом. Повторяет forward_hicora по частям.

        ПОВТОРЯЕТ, А НЕ ВЫЗЫВАЕТ: `forward_hicora` сам применяет голову и
        ждёт от неё пару, а гауссова возвращает словарь. Паритет этой
        раскладки с `forward_hicora` проверяется ниже на НАСТОЯЩЕЙ
        детерминированной голове и под autocast, а не на игрушечной.
        """
        v, p = model.build_inputs(position_offset=4, **batch)
        taps = model.forward_taps(
            vlm_inputs_embeds=v,
            attention_mask=batch.get("attention_mask"), position_ids=p)
        _, q0 = model.q0_from(taps[model.q0_depth])
        z0 = model.codebooks[0][q0]
        h24 = model.res_norm(taps[max(model.taps)]).float()
        return h24, z0, int(taps["layers_run"]), v, p

    def decode_latent(z):
        x, _ = codec._decode(z.float(), embodiment_ids=0)
        return x[..., :7].detach().float().cpu().numpy()

    envs, task_desc_all = None, {}
    out = dict(run_tag=args.run_tag, ckpt=args.ckpt, joint_sha1=joint_sha,
               head_sha1=head_sha, res_norm_sha1=rn_sha,
               script_sha1=k9h.file_sha12(os.path.abspath(__file__)),
               hicora_g_sha1=k9h.file_sha12(hg.__file__),
               hicora_vla_sha1=k9h.file_sha12(hv.__file__),
               device=str(dev), dtype=args.dtype, ensemble="off",
               horizon=args.horizon, max_steps=args.max_steps,
               waiting_steps=args.waiting_steps, n_envs=args.n_envs,
               init_start=args.init_start, tasks=list(tasks),
               sigmas=list(sigmas), rho_norm=rho_norm,
               thresholds=dict(min_changed=MIN_CHANGED, sat_thr=SAT_THR,
                               max_sat=MAX_SAT, max_drop=MAX_DROP,
                               change_thr=CHANGE_THR),
               parity=None, cells={}, episodes=[])
    parity_done = [False]

    def rollout(task_id, hd, sigma):
        nonlocal envs
        envs, task_desc = get_envs(args.task_suite, task_id, args.n_envs,
                                   args.init_start, envs)
        task_desc_all[task_id] = task_desc
        n = args.n_envs
        obs = envs.reset()
        reward = np.zeros(n)
        done = np.zeros(n, bool)
        dummy = np.array([[0, 0, 0, 0, 0, 0, -1]] * n)
        for _ in range(args.waiting_steps):
            obs, r_, done, _ = envs.step(dummy)
            reward = np.clip(reward + r_, 0, 1)
        det_h, gau_h = det_heads[hd], gau_heads[hd]
        stat = dict(calls=0, changed=0, rms=[], mx=[], grip=0,
                    sat=[], dz_frac=[], logp=[], layers=set())
        calls = steps = 0
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
            with torch.no_grad(), ac16:
                h24, z0, n_lay, vv, pp = taps_of(batch)
                stat["layers"].add(n_lay)
                o_mean = gau_h(h24, z0, deterministic=True)
                if sigma == 0.0:
                    o_exec = o_mean
                else:
                    # ШУМ ИЗ ФИКСИРОВАННОГО СИДА, одинаковый для всех sigma.
                    gen = torch.Generator(device=h24.device)
                    gen.manual_seed(eps_seed(task_id, args.init_start, calls)
                                    % (2 ** 63))
                    eps = torch.empty_like(o_mean["mu"]).normal_(
                        generator=gen)
                    o_exec = gau_h(h24, z0,
                                   u=o_mean["mu"] + float(sigma) * eps)
                # ПАРИТЕТ НА НАСТОЯЩЕЙ ГОЛОВЕ И ПОД autocast, один раз.
                if not parity_done[0]:
                    dz_d, c_d = det_h(h24, z0)
                    # ТЕ ЖЕ ВХОДЫ, что у taps_of: повторный build_inputs
                    # был бы и лишней работой, и риском сравнить разное.
                    model.hicora_head = det_h
                    o_fw = model.forward_hicora(
                        vlm_inputs_embeds=vv,
                        attention_mask=batch.get("attention_mask"),
                        position_ids=pp)
                    d_head = float((o_mean["dz"] - dz_d).abs().max())
                    d_full = float((dz_d - o_fw["dz"]).abs().max())
                    out["parity"] = dict(
                        gauss_mean_vs_d1=d_head, d1_vs_forward_hicora=d_full,
                        layers_run=n_lay,
                        ok=bool(d_head <= 1e-4 and d_full <= 1e-4
                                and n_lay == 24))
                    print(f"  паритет на настоящем D1 под autocast: "
                          f"гауссова(mean) против D1 {d_head:.2e}, D1 против "
                          f"forward_hicora {d_full:.2e}, слоёв {n_lay}",
                          flush=True)
                    if not out["parity"]["ok"]:
                        raise SystemExit(
                            "ПАРИТЕТ НЕ СОШЁЛСЯ: гауссова голова в режиме "
                            "среднего обязана\n  давать ровно то же, что D1, "
                            "иначе сравнение с K-11e не имеет смысла")
                    parity_done[0] = True
                a_exec = decode_latent(z0 + o_exec["dz"])
                a_mean = decode_latent(z0 + o_mean["dz"])
                dzn = torch.linalg.norm(o_exec["dz"], dim=-1).max()
                if not torch.isfinite(o_exec["dz"]).all():
                    raise SystemExit("в поправке появились nan или inf")
                if float(dzn) > rho_norm + 1e-4:
                    raise SystemExit(
                        f"||dz|| = {float(dzn):.4f} превысила предел "
                        f"{rho_norm:.4f}: ограничение перестало действовать")
                stat["dz_frac"].append(float(dzn) / rho_norm)
                stat["sat"].append(float(
                    (o_exec["coeffs"].abs() > SAT_THR).float().mean()))
                stat["logp"].append(float(o_exec["log_prob_u"].mean()))
            ch = action_change(a_exec, a_mean, act_range)
            stat["calls"] += 1
            stat["changed"] += int(ch["changed"])
            stat["rms"].append(ch["rms"])
            stat["mx"].append(ch["max"])
            stat["grip"] += int(ch["grip_flip"])
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
        eps_rows = [dict(task_id=task_id, head=hd, sigma=float(sigma),
                         env_index=i, init_state_id=args.init_start + i,
                         success=bool(reward[i] >= 1.0), env_steps=steps,
                         policy_calls=calls) for i in range(n)]
        return eps_rows, stat

    t0 = time.time()
    try:
        for hd in HEADS:
            for sg in sigmas:
                rows, agg = [], dict(calls=0, changed=0, grip=0, rms=[],
                                     mx=[], sat=[], dz_frac=[], logp=[],
                                     layers=set())
                for tid in tasks:
                    r, s = rollout(tid, hd, sg)
                    rows += r
                    agg["calls"] += s["calls"]
                    agg["changed"] += s["changed"]
                    agg["grip"] += s["grip"]
                    for k in ("rms", "mx", "sat", "dz_frac", "logp"):
                        agg[k] += s[k]
                    agg["layers"] |= s["layers"]
                succ = float(np.mean([x["success"] for x in rows]))
                cell = dict(
                    head=hd, sigma=float(sg), episodes=len(rows),
                    success=succ, calls=agg["calls"],
                    changed_frac=(agg["changed"] / agg["calls"]
                                  if agg["calls"] else 0.0),
                    grip_flip_frac=(agg["grip"] / agg["calls"]
                                    if agg["calls"] else 0.0),
                    rms_median=float(np.median(agg["rms"])) if agg["rms"] else 0.0,
                    max_p95=(float(np.percentile(agg["mx"], 95))
                             if agg["mx"] else 0.0),
                    sat_frac=float(np.mean(agg["sat"])) if agg["sat"] else 0.0,
                    dz_frac_max=(float(np.max(agg["dz_frac"]))
                                 if agg["dz_frac"] else 0.0),
                    logp_median=(float(np.median(agg["logp"]))
                                 if agg["logp"] else None),
                    layers_run=sorted(agg["layers"]),
                    invariants_ok=bool(agg["layers"] == {24}))
                out["cells"][f"{hd}|{sg}"] = cell
                out["episodes"] += rows
                print(f"  {hd}, sigma={sg}: успех {100 * succ:.1f}% "
                      f"({len(rows)} эп), изменено {100 * cell['changed_frac']:.1f}% "
                      f"вызовов, насыщение {100 * cell['sat_frac']:.1f}%, "
                      f"||dz||/||rho|| макс {cell['dz_frac_max']:.3f}",
                      flush=True)
    finally:
        if envs is not None:
            envs.close()

    cells = {(c["head"], c["sigma"]): c for c in out["cells"].values()}
    win = read_window(cells)
    out["window"] = win
    out["minutes"] = (time.time() - t0) / 60.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  сохранено: {args.out} ({out['minutes']:.1f} мин)")

    print(f"\n  ОКНО ИССЛЕДОВАНИЯ (правило записано до запуска)")
    print(f"    {'sigma':>7}{'s0 изм':>9}{'s0 нас':>8}{'s0 усп':>8}"
          f"{'s1 изм':>9}{'s1 нас':>8}{'s1 усп':>8}{'вердикт':>10}")
    for sg in sorted(win["per_sigma"]):
        p = win["per_sigma"][sg]["per_head"]
        def g(hd, k, d="—"):
            v = p.get(hd, {}).get(k)
            return d if v is None else f"{100 * v:.0f}%"
        print(f"    {sg:>7.2f}{g('s0','changed'):>9}{g('s0','sat'):>8}"
              f"{g('s0','success'):>8}{g('s1','changed'):>9}"
              f"{g('s1','sat'):>8}{g('s1','success'):>8}"
              f"{('годна' if win['per_sigma'][sg]['ok'] else 'нет'):>10}")
    if win["chosen"] is None:
        print("\n  ОКНА НЕТ. Ни одна sigma не меняет поведение существенно, "
              "оставаясь\n  безопасной на обеих головах. PPO в таком виде "
              "запускать нельзя:\n  либо шум ничего не меняет, либо он "
              "разрушает стратегию.")
        raise SystemExit(1)
    print(f"\n  ОКНО ЕСТЬ. Годные sigma: {win['passing']}; для PPO берём "
          f"МИНИМАЛЬНУЮ: {win['chosen']}")
    print("  ЭТО ДИАГНОСТИКА, НЕ ДОКАЗАТЕЛЬСТВО: 50 эпизодов на ячейку не "
          "дают интервалов,\n  по которым можно утверждать не-худшесть. "
          "Окончательная оценка RL — на НОВЫХ\n  начальных состояниях.")


if __name__ == "__main__":
    main()
