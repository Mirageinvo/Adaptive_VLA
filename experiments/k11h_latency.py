"""K-11h: стоимость ИМЕННО ТОЙ политики, которая проверена в K-11e.

ЧТО МЕРЯЕТСЯ (K-11h-A, зарегистрированный критерий):
  fullbar    — 24 слоя, ТРИ прохода, сборка из трёх уровней;
  coarse24   — 24 слоя, ОДИН проход, сборка из уровня 0;
  joint12    — 12 слоёв, один проход, веса Joint-12;
  hicora_s0  — 24 слоя, один проход, q0 со слоя 12 + поправка от h24;
  hicora_s1  — то же, вторая голова.

ЧТО ЭТО НЕ ЕСТЬ. Здесь НЕ меряется потоковая политика «исполнить черновик со
слоя 12, пока считаются слои 13-24». Такой политики в коде нет: forward_hicora
исполняет все 24 слоя и лишь затем берёт q0 из сохранённого h12. Ранний выход
пришлось бы реализовать, и у него было бы ДВА декодирования, ГРУБОЕ первое
действие и СВОЙ, неизвестный успех. Соединять успех 92.0% из K-11e с
задержкой в 12 слоёв нельзя: это метрики двух разных политик.

ПОРОГ. Зарегистрирован до запуска, батч 1 — первичный:
    max_s  T(hicora_s) / T(coarse24)  <= 1.10
    min_s  T(fullbar)  / T(hicora_s)  >= 1.80
Батч 10 — обязательный сопутствующий результат, но НЕ критерий.

ПОЧЕМУ 1.10, А НЕ 0.60. Прежнее предложение «0.60 от coarse24» противоречило
нашим же измерениям K-9i: там 12 слоёв против 24 дают 65.1 против 84.7 мс,
то есть 0.77 на батче 1 и 0.88 на батче 10. Постоянная часть — башня зрения,
подготовка входа и декодер — не масштабируется с числом слоёв. Отношение 0.60
достижимо только если исключить общую часть, а это уже не «задержка до
действия», а время одних лишь слоёв потока действий.

ЧЕРЕДУЮТСЯ ВСЕ ПЯТЬ РУК, А НЕ ГРУППЫ ВЕСОВ. K-9i чередовал fullbar с coarse24
честно, а 12-слойные конфигурации мерил своей группой: там сравнение шло
внутри группы. Здесь первичное сравнение hicora против coarse24 пересекает
группы весов, и дрейф частоты карты достался бы ему целиком. Поэтому веса
переставляются на КАЖДОМ повторе, вне измеряемого участка, а состояния
предзагружены на устройство: перестановка стоит миллисекунды и в замер не
входит.

ПАМЯТЬ — ОТДЕЛЬНО, вне чередования: пик у перемешанных конфигураций общий и
ни одной из них не принадлежит.

Запуск:
    python experiments/k11h_latency.py --selftest
    python experiments/k11h_latency.py --ckpt <hf> --joint12 data/k9d_ep3.pt \\
        --hicora-s0 ... --hicora-s1 ... --out data/k11h/latency.json
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_POS, N_LEVEL = 16, 3
CONFIGS = ("fullbar", "coarse24", "joint12", "hicora_s0", "hicora_s1")
HICORA = ("hicora_s0", "hicora_s1")
BASE_W = ("fullbar", "coarse24")          # идут на исходных весах
# ЗАРЕГИСТРИРОВАННЫЕ ПОРОГИ. Менять только через явный флаг, и тогда результат
# перестаёт быть зарегистрированным.
MAX_VS_COARSE = 1.10
MIN_VS_FULLBAR = 1.80
PRIMARY_BATCH = 1
# Сколько проходов и декодирований делает каждая рука. Печатается рядом с
# временем: без этого «дешевле» нечем объяснить.
PASSES = {"fullbar": N_LEVEL, "coarse24": 1, "joint12": 1,
          "hicora_s0": 1, "hicora_s1": 1}
LAYERS = {"fullbar": 24, "coarse24": 24, "joint12": 12,
          "hicora_s0": 24, "hicora_s1": 24}
DECODES = {c: 1 for c in CONFIGS}


def stats(xs):
    a = np.asarray(xs, float)
    return dict(median=float(np.median(a)), mean=float(a.mean()),
                p95=float(np.percentile(a, 95)), std=float(a.std()),
                n=int(a.size))


def read_gate(med, max_vs_coarse=MAX_VS_COARSE,
              min_vs_fullbar=MIN_VS_FULLBAR):
    """Вердикт K-11h-A по медианам одного батча.

    ХУДШАЯ ГОЛОВА, А НЕ СРЕДНЯЯ. Правило K-11e требовало выполнения на обоих
    сидах; здесь так же: берётся max отношения к coarse24 и min ускорения
    против fullbar. Отсутствие любой руки — отказ, а не пропуск проверки.
    """
    need = ("coarse24", "fullbar") + HICORA
    miss = [c for c in need if c not in med or med[c] is None]
    if miss:
        raise SystemExit(f"нет измерений для {miss}: вердикт K-11h-A "
                         f"невозможен")
    r_coarse = {s: med[s] / med["coarse24"] for s in HICORA}
    r_full = {s: med["fullbar"] / med[s] for s in HICORA}
    worst_c = max(r_coarse.values())
    worst_f = min(r_full.values())
    ok_c = worst_c <= max_vs_coarse
    ok_f = worst_f >= min_vs_fullbar
    return dict(ratio_vs_coarse=r_coarse, speedup_vs_fullbar=r_full,
                worst_vs_coarse=worst_c, worst_vs_fullbar=worst_f,
                ok_vs_coarse=bool(ok_c), ok_vs_fullbar=bool(ok_f),
                passed=bool(ok_c and ok_f),
                thresholds=dict(max_vs_coarse=max_vs_coarse,
                                min_vs_fullbar=min_vs_fullbar))


def split_estimate(med, decode_med):
    """K-11h-B: оценка разрыва между черновиком и поправкой. НЕ ГЕЙТ.

    ЭТО ОЦЕНКА, А НЕ ЗАМЕР ПОЛИТИКИ. Потоковой руки не существует, поэтому
    прямо измерить «когда готов черновик» внутри hicora нельзя: все 24 слоя
    уже исполнены к моменту, когда берётся q0. Складывается из измеренных
    величин в предположении, что общий префикс из 12 слоёв считается один раз:

        dT = (T_hicora - T_joint12) + T_decode

    Второе слагаемое — потому что потоковая политика декодирует ДВАЖДЫ:
    сначала черновик, потом исправленный хвост. Величина показывает, успевает
    ли поправка внутрь одного шага управления (ActionCodec идёт около 20 Гц,
    то есть шаг 50 мс), и ничего не говорит об успехе такой политики.
    """
    if "joint12" not in med:
        return None
    out = {}
    for s in HICORA:
        if s not in med:
            continue
        out[s] = dict(t_draft_est=med["joint12"],
                      t_refined=med[s],
                      delta_ms=med[s] - med["joint12"] + decode_med,
                      extra_decode_ms=decode_med)
    return out


def selftest():
    # --- вердикт по худшей голове ------------------------------------------
    base = dict(coarse24=84.7, fullbar=185.5, joint12=65.1,
                hicora_s0=88.0, hicora_s1=89.0)
    g = read_gate(base)
    assert g["passed"], g
    assert abs(g["worst_vs_coarse"] - 89.0 / 84.7) < 1e-9
    assert abs(g["worst_vs_fullbar"] - 185.5 / 89.0) < 1e-9
    # одна голова за порогом — не проходит ВЕСЬ критерий
    g2 = read_gate(dict(base, hicora_s1=95.0))
    assert not g2["passed"] and not g2["ok_vs_coarse"]
    assert g2["ok_vs_fullbar"], "ускорение против fullbar тут ещё держится"
    # слишком медленно против fullbar
    g3 = read_gate(dict(base, fullbar=150.0))
    assert not g3["passed"] and not g3["ok_vs_fullbar"]
    # ровно на пороге — проходит (нестрогое неравенство)
    g4 = read_gate(dict(base, hicora_s0=84.7 * 1.10, hicora_s1=84.7 * 1.10))
    assert g4["passed"], g4
    # отсутствие руки — отказ, а не «проверка неприменима»
    for gone in ("coarse24", "fullbar", "hicora_s0", "hicora_s1"):
        try:
            read_gate({k: v for k, v in base.items() if k != gone})
        except SystemExit:
            pass
        else:
            raise AssertionError(f"отсутствие {gone} принято")
    try:
        read_gate(dict(base, hicora_s1=None))
    except SystemExit:
        pass
    else:
        raise AssertionError("None вместо измерения принят")

    # --- оценка разрыва ------------------------------------------------------
    sp = split_estimate(base, 2.0)
    assert abs(sp["hicora_s0"]["delta_ms"] - (88.0 - 65.1 + 2.0)) < 1e-9
    assert sp["hicora_s0"]["t_draft_est"] == 65.1
    assert split_estimate({"hicora_s0": 1.0}, 2.0) is None

    # --- ОПОРА K-9i: порог 0.60 ей противоречил ------------------------------
    # 65.1/84.7 = 0.77 на батче 1 и 261/296.8 = 0.88 на батче 10. Двенадцать
    # слоёв не дают 0.60 даже без головы поправки.
    assert abs(65.1 / 84.7 - 0.7686) < 1e-3
    assert abs(261.0 / 296.8 - 0.8794) < 1e-3

    print("самопроверка k11h пройдена: вердикт берёт ХУДШУЮ голову по обоим "
          "порогам,\n  отсутствие любой руки — отказ, порог нестрогий; "
          "оценка разрыва учитывает\n  второе декодирование и без joint12 не "
          "считается")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--joint12", default="data/k9d_ep3.pt")
    ap.add_argument("--hicora-s0",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s0.pt")
    ap.add_argument("--hicora-s1",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s1.pt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batches", default="1,10")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--pos-offset", type=int, default=4)
    ap.add_argument("--expect-hicora-target", default="coef")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--allow-nonstandard-threshold", action="store_true",
                    help="разрешить пороги, отличные от зарегистрированных "
                         "1.10 и 1.80. Результат тогда НЕ зарегистрированный")
    ap.add_argument("--max-vs-coarse", type=float, default=MAX_VS_COARSE)
    ap.add_argument("--min-vs-fullbar", type=float, default=MIN_VS_FULLBAR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt")
    std = (abs(args.max_vs_coarse - MAX_VS_COARSE) < 1e-12
           and abs(args.min_vs_fullbar - MIN_VS_FULLBAR) < 1e-12)
    if not std and not args.allow_nonstandard_threshold:
        raise SystemExit(
            f"пороги {args.max_vs_coarse}/{args.min_vs_fullbar} отличаются от "
            f"зарегистрированных {MAX_VS_COARSE}/{MIN_VS_FULLBAR}.\n"
            f"  Подобрать порог под измеренное — то же, что выбрать гипотезу "
            f"после данных.\n  Осознанно: --allow-nonstandard-threshold")

    # КОРЕНЬ ActionCodec ДОБАВЛЯЕТСЯ ДО ИМПОРТОВ. Раньше он добавлялся ниже,
    # и `import actioncodec` падал с ModuleNotFoundError: пакет лежит в
    # third_party и в sys.path сам по себе не попадает. Так же устроен K-9i.
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"нет каталога ActionCodec: {root}")
    sys.path.insert(0, root)

    import torch
    from actioncodec.utils import get_cfg, seed_everything, dict_apply
    from actioncodec.models.smolvla_bar import SmolVLABlockwiseAR
    from actioncodec.processors.vla_processor import (
        VisionLanguageActionProcessor, prompt_template)
    from actioncodec.constants import STATE_Q01, STATE_Q99
    from joint12_vla import make_joint12_class
    import hicora_vla as hv
    import k9h_multiarm_gate as k9h
    import k11a_build_hicora_cache as k11a

    dev = torch.device(args.device)
    dt = getattr(torch, args.dtype)
    if dev.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(dev)
        if free_b / 2 ** 30 < 20 and not args.allow_busy_gpu:
            raise SystemExit(
                f"на {dev} свободно {free_b / 2 ** 30:.1f} ГБ: чужая нагрузка "
                f"исказит замер. --allow-busy-gpu, если так и задумано")

    seed_everything(0)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    # --- один фиксированный вход на все руки ---------------------------------
    z = np.load(args.cache, allow_pickle=True)
    tsk = z["task"]
    IMG = np.load(args.cache + ".images.npy", mmap_mode="r")
    batches = [int(x) for x in args.batches.split(",")]
    max_b = max(batches)
    st = np.zeros((max_b, len(STATE_Q01)), np.float64)
    st_n = (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    def build(b):
        image = torch.from_numpy(np.asarray(IMG[:b]))
        msgs = []
        for i in range(b):
            m = prompt_template(
                st_n[i], None, str(tsk[i]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        bt = proc(text=texts, images=[[image[k].numpy()] for k in range(b)],
                  return_tensors="pt", padding=True, padding_side="left",
                  action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), bt)

    # --- модель --------------------------------------------------------------
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    import copy
    # ИСХОДНАЯ ФИНАЛЬНАЯ НОРМА СНИМАЕТСЯ ДО init_joint_fast: после него это уже
    # норма Joint12, обученная на h12, и голова читала бы h24 не тем.
    res_norm_orig = copy.deepcopy(model.action_expert.norm)
    model.init_joint_fast(depth=12, head_dtype=dt)
    own = dict(model.named_parameters())
    base_state = {k: own[k].detach().clone()
                  for k in own if own[k].requires_grad}

    j_obj = torch.load(args.joint12, map_location="cpu", weights_only=False)
    joint_sha = k9h.file_sha12(args.joint12)
    j_state = {k: v.to(dev, torch.float32) for k, v in j_obj["state"].items()}
    stray = [k for k in j_state if k not in own]
    if stray:
        raise SystemExit(f"в чекпойнте Joint12 ключи вне модели: {stray[:5]}")
    # ПЕРЕСТАНОВКА ОБЯЗАНА БЫТЬ ПОЛНОЙ В ОБЕ СТОРОНЫ. Ключ, который есть у
    # Joint12, но не сохранён в base_state, при возврате к fullbar остался бы
    # от Joint12 — и «исходные веса» были бы смесью.
    lost = [k for k in j_state if k not in base_state]
    if lost:
        raise SystemExit(
            f"{len(lost)} весов Joint12 не сохранены как исходные: {lost[:5]}. "
            f"Тогда fullbar и coarse24 пошли бы частично на весах черновика")
    print(f"  черновик Joint12: {len(j_state)} тензоров, sha {joint_sha}")

    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь в action_processor")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(ii))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    # --- головы поправки ------------------------------------------------------
    model.__class__ = hv.make_hicora_class(type(model))
    model.set_codebooks(E)
    model.set_res_norm(res_norm_orig.to(dev))
    model.taps, model.q0_depth = (12, 18, 24), 12
    model.n_layers_total = len(model.action_expert.layers)
    if max(model.taps) != model.n_layers_total:
        raise SystemExit(f"последний отвод {max(model.taps)} против "
                         f"{model.n_layers_total} слоёв")
    rn_sha = hashlib.sha1()
    for k_ in sorted(model.res_norm.state_dict()):
        v_ = model.res_norm.state_dict()[k_]
        rn_sha.update(k_.encode())
        rn_sha.update(np.ascontiguousarray(
            v_.detach().float().cpu().numpy()).tobytes())
    rn_sha = rn_sha.hexdigest()[:12]

    heads, head_sha = {}, {}
    for nm, path in (("hicora_s0", args.hicora_s0),
                     ("hicora_s1", args.hicora_s1)):
        o = torch.load(path, map_location="cpu", weights_only=False)
        k9h.check_hicora_ckpt(o, nm, args.expect_hicora_target)
        pref = o["cache"]
        bp, rp, mp = pref + ".basis.npy", pref + ".rho.npy", pref + ".meta.json"
        for f_ in (bp, rp, mp):
            if not os.path.exists(f_):
                raise SystemExit(f"нет {f_}: привязать голову к кэшу нечем")
        for f_, want_, lbl in ((bp, o["basis_sha1"], "базис"),
                               (rp, o["rho_sha1"], "предел")):
            got_ = k9h.file_sha12(f_)
            if got_ != want_:
                raise SystemExit(f"{lbl} sha {got_}, а голова обучена на "
                                 f"{want_}")
        if rn_sha != o["res_norm_sha1"]:
            raise SystemExit(f"res_norm sha {rn_sha}, а голова {nm} обучена "
                             f"на {o['res_norm_sha1']}")
        meta = json.load(open(mp))
        k9h.check_hicora_meta(meta, args.ckpt, joint_sha,
                              k9h.file_sha12(hv.__file__),
                              k9h.file_sha12(
                                  sys.modules["joint12_vla"].__file__))
        k11a.check_fingerprints(meta, dict(
            codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
                E.cpu().numpy().astype(np.float32)).tobytes()).hexdigest()[:12],
            decoder_probe=k11a.decoder_probe(codec, E, dev),
            codec_state_sha1=k11a.state_sha1(codec)))
        B = np.load(bp).astype(np.float32)
        rho = np.load(rp).astype(np.float32)
        d_h = int(model.fast_head.in_features)
        h = hv.make_residual_head()(
            d_h, int(E.shape[-1]), rank=int(o["rank"]),
            hidden=int(o.get("hidden", 512)),
            proj=int(o.get("proj", 64))).to(dev)
        h.set_basis(torch.as_tensor(B))
        h.set_rho(torch.as_tensor(rho))
        bad = [k for k in o["state"] if not k.startswith("hicora_head.")]
        if bad:
            raise SystemExit(f"в чекпойнте {nm} ключи вне hicora_head.: "
                             f"{bad[:5]}")
        st_h = {k[len("hicora_head."):]: v for k, v in o["state"].items()}
        want_h = {k for k in h.state_dict() if k.startswith(("proj.", "net."))}
        if set(st_h) != want_h:
            raise SystemExit(f"набор весов головы {nm} не совпал")
        with torch.no_grad():
            for k, v in st_h.items():
                h.state_dict()[k].copy_(v.to(dev, torch.float32))
        h.eval()
        heads[nm] = h
        head_sha[nm] = k9h.file_sha12(path)
        print(f"  голова {nm}: sha {head_sha[nm]}, ранг {o['rank']}, "
              f"сид {o['seed']}, эпоха {o.get('selected_epoch')}")
    if head_sha["hicora_s0"] == head_sha["hicora_s1"]:
        raise SystemExit("обе головы — один файл: это не две руки")

    @contextlib.contextmanager
    def only_blocks(n):
        saved = model.num_blocks
        try:
            model.num_blocks = n
            yield
        finally:
            model.num_blocks = saved

    def apply_weights(name):
        """Веса нужной руки. Состояния предзагружены, копирование дешёвое и
        ВНЕ измеряемого участка."""
        src = base_state if name in BASE_W else j_state
        with torch.no_grad():
            for k, v in src.items():
                own[k].data = v
        if name in HICORA:
            model.hicora_head = heads[name]

    def decode_codes(codes, n_lv):
        K = codes.reshape(-1, n_lv, N_POS)
        zq = E[0][torch.as_tensor(K[:, 0, :]).long().to(dev)]
        for j in range(1, n_lv):
            zq = zq + E[j][torch.as_tensor(K[:, j, :]).long().to(dev)]
        x, _ = codec._decode(zq, embodiment_ids=0)
        return x[..., :7]

    def decode_latent(zl):
        x, _ = codec._decode(zl.float(), embodiment_ids=0)
        return x[..., :7]

    ac16 = torch.autocast("cuda", dtype=torch.float16)

    def run(name, batch):
        """Один вызов политики. Возвращает то, что пойдёт в декодер."""
        if name == "fullbar":
            with torch.no_grad():
                t = model.generate(**batch, position_offset=args.pos_offset,
                                   do_sample=False)
            return ("codes", t.cpu().numpy(), N_LEVEL)
        if name == "coarse24":
            with torch.no_grad(), only_blocks(1):
                t = model.generate(**batch, position_offset=args.pos_offset,
                                   do_sample=False)
            return ("codes", t[:, :N_POS].cpu().numpy(), 1)
        with torch.no_grad(), ac16:
            v, p = model.build_inputs(position_offset=args.pos_offset, **batch)
            if name == "joint12":
                o = model.forward_joint_fast(
                    vlm_inputs_embeds=v,
                    attention_mask=batch.get("attention_mask"), position_ids=p)
                return ("codes", o["pred_codes"].cpu().numpy(), 1)
            o = model.forward_hicora(
                vlm_inputs_embeds=v,
                attention_mask=batch.get("attention_mask"), position_ids=p)
            if int(o["layers_run"]) != 24:
                raise SystemExit(f"{name}: исполнено {o['layers_run']} слоёв "
                                 f"вместо 24 — мерится не та политика")
            return ("latent", o["z"], 1)

    def do_decode(kind, payload, n_lv):
        return (decode_codes(payload, n_lv) if kind == "codes"
                else decode_latent(payload))

    print(f"\nруки: {', '.join(CONFIGS)}")
    print("  ВНИМАНИЕ: потоковая политика (черновик со слоя 12 в исполнение, "
          "поправка\n  следом) здесь НЕ мерится — её нет в коде, и успех у "
          "неё был бы свой.")

    out = dict(ckpt=args.ckpt, joint_sha1=joint_sha, head_sha1=head_sha,
               res_norm_sha1=rn_sha, reps=args.reps, warmup=args.warmup,
               script_sha1=k9h.file_sha12(os.path.abspath(__file__)),
               device=str(dev), dtype=args.dtype,
               passes=PASSES, layers=LAYERS, decodes=DECODES,
               primary_batch=PRIMARY_BATCH, batches={})

    for bs in batches:
        batch = build(bs)
        print(f"\n=== батч {bs} ===")
        for name in CONFIGS:
            apply_weights(name)
            for _ in range(args.warmup):
                k, pl, nl = run(name, batch)
                do_decode(k, pl, nl)
            torch.cuda.synchronize()
        tm = {c: [] for c in CONFIGS}
        td = {c: [] for c in CONFIGS}
        for r in range(args.reps):
            for name in CONFIGS:
                apply_weights(name)          # ВНЕ замера
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                k, pl, nl = run(name, batch)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                do_decode(k, pl, nl)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                tm[name].append((t1 - t0) * 1000)
                td[name].append((t2 - t1) * 1000)
            if (r + 1) % 50 == 0:
                print(f"    повтор {r + 1}/{args.reps}", flush=True)

        mem = {}
        for name in CONFIGS:
            apply_weights(name)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            for _ in range(5):
                k, pl, nl = run(name, batch)
                do_decode(k, pl, nl)
            torch.cuda.synchronize()
            mem[name] = torch.cuda.max_memory_allocated(dev) / 2 ** 20

        row = {}
        for name in CONFIGS:
            tot = [m + d for m, d in zip(tm[name], td[name])]
            row[name] = dict(model_ms=stats(tm[name]),
                             decode_ms=stats(td[name]),
                             total_ms=stats(tot), peak_mib=mem[name])
        med = {c: row[c]["total_ms"]["median"] for c in CONFIGS}
        print(f"\n  {'рука':<12}{'слоёв':>7}{'прох':>6}{'декод':>7}"
              f"{'медиана':>10}{'сред':>9}{'p95':>9}{'память':>10}")
        for name in CONFIGS:
            r = row[name]
            print(f"  {name:<12}{LAYERS[name]:>7}{PASSES[name]:>6}"
                  f"{DECODES[name]:>7}{r['total_ms']['median']:>10.1f}"
                  f"{r['total_ms']['mean']:>9.1f}{r['total_ms']['p95']:>9.1f}"
                  f"{r['peak_mib']:>9.0f}М")
        gate = read_gate(med, args.max_vs_coarse, args.min_vs_fullbar)
        sp = split_estimate(med, np.median(
            [row[s]["decode_ms"]["median"] for s in HICORA]))
        print(f"\n  против coarse24: "
              + ", ".join(f"{s} {v:.3f}x"
                          for s, v in sorted(gate['ratio_vs_coarse'].items()))
              + f" (худшая {gate['worst_vs_coarse']:.3f}, порог "
                f"<= {args.max_vs_coarse})")
        print(f"  против fullbar:  "
              + ", ".join(f"{s} {v:.2f}x"
                          for s, v in sorted(gate['speedup_vs_fullbar'].items()))
              + f" (худшая {gate['worst_vs_fullbar']:.2f}, порог "
                f">= {args.min_vs_fullbar})")
        if sp:
            for s in sorted(sp):
                d = sp[s]
                print(f"  K-11h-B, {s}: черновик ~{d['t_draft_est']:.1f} мс, "
                      f"поправка готова через ~{d['delta_ms']:.1f} мс "
                      f"(+{d['extra_decode_ms']:.2f} на второе декодирование)"
                      + ("  — внутрь шага 50 мс" if d['delta_ms'] < 50
                         else "  — НЕ внутрь шага 50 мс"))
            print("  ЭТО ОЦЕНКА, НЕ ЗАМЕР: потоковой руки не существует, "
                  "величина сложена\n  из измеренных и об успехе такой "
                  "политики не говорит.")
        if bs == PRIMARY_BATCH:
            # БЕЗ ВЛОЖЕННОГО МНОГОСТРОЧНОГО ВЫРАЖЕНИЯ В f-СТРОКЕ: такое
            # разрешено лишь с Python 3.12, а на кластере интерпретатор
            # старше и это была бы синтаксическая ошибка при запуске.
            verd = "ПРОЙДЕН" if gate["passed"] else "НЕ ПРОЙДЕН"
            print(f"\n  K-11h-A ({verd}) — первичный критерий, "
                  f"батч {PRIMARY_BATCH}")
        else:
            print(f"\n  батч {bs} — сопутствующий результат, в критерий НЕ "
                  f"входит")
        out["batches"][str(bs)] = dict(rows=row, median_ms=med, gate=gate,
                                       split_estimate=sp,
                                       is_primary=bool(bs == PRIMARY_BATCH))

    prim = out["batches"].get(str(PRIMARY_BATCH))
    if prim is None:
        raise SystemExit(f"батч {PRIMARY_BATCH} не измерялся: первичного "
                         f"критерия нет")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")
    if not prim["gate"]["passed"]:
        print("\n  K-11h-A НЕ ПРОЙДЕН. Порог зарегистрирован до запуска; "
              "подбирать его\n  под измеренное нельзя.")
        raise SystemExit(1)
    print("\n  K-11h-A ПРОЙДЕН.")
    print("  ЧТО ЭТО ЗНАЧИТ: преимущество по стоимости при СОПОСТАВИМОМ "
          "НАБЛЮДАЕМОМ успехе.\n  Вывод «не хуже и дешевле» отсюда НЕ следует: "
          "K-11e превосходства не показал,\n  а не-худшесть с допуском мы не "
          "регистрировали — интервалы допускают\n  ухудшение до 1.25-2.5 пп.")


if __name__ == "__main__":
    main()
