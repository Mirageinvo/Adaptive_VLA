#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14f: проба читающих голов по кэшу h18 (§46).

ВОПРОС. §45.1 показал, что правило вывода узким местом не является: мягкое
декодирование удваивает захват, но и удвоенного вдвое меньше порога. Следующее
по порядку — хватает ли ЛИНЕЙНОГО чтения h18. Проба обучает двухслойную голову
на том же входе и той же потере и различает три исхода: растут обе точности
(дело было в чтении), растёт только обучающая (в обобщении), не растёт и
обучающая (информации в этом h18 недостаточно).

ПОЧЕМУ ЭТО ДЁШЕВО. Магистраль заморожена, и её прямой проход от варианта
головы не зависит. K-14e посчитал его один раз; здесь остаётся норма, голова
и декодер. Семь часов за вариант превращаются в минуты.

ЧЕГО ПРОБА НЕ МОЖЕТ. Обратная связь вшита в кэш (проекция прибавляется к
состоянию до слоёв 13-18), поэтому речь о чтении ЭТОГО h18, а не об
информации вообще.

ПЕРЕД ЛЮБЫМ ОБУЧЕНИЕМ — СВЕРКА. Нынешняя ЛИНЕЙНАЯ голова, применённая к кэшу,
обязана воспроизвести живые числа канонического прогона. Не воспроизвела —
кэш или воспроизведение нормы неверны, и обучать на них нечего.

    python experiments/k14f_mlp_probe.py --verify-only --device cuda:1
    python experiments/k14f_mlp_probe.py --device cuda:1 \
        --hidden 512,1024,2048 --seeds 0,1,2
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def arr_sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def make_rms_norm(torch):
    """RMSNorm, воспроизведённая явно; правильность доказывается сверкой.

    ЗАЧЕМ СВОЯ РЕАЛИЗАЦИЯ. Класс нормы живёт в модели, а загружать 2.2 млрд
    параметров ради одного вектора весов — это те самые семь часов, от
    которых проба и уходит. Совпадение доказывается не рассуждением, а
    обязательной сверкой линейной головы на кэше с живыми числами: если
    норма воспроизведена неточно, RMS разойдётся.
    """
    import torch.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = float(eps)

        def forward(self, x):
            f = x.float()
            f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + self.eps)
            return self.weight * f

    return RMSNorm


def make_head(torch, d_model, vocab, hidden, eps):
    """Голова пробы: норма, затем один или два слоя.

    hidden = 0 воспроизводит НЫНЕШНЮЮ голову — норма и один Linear. Это не
    удобство, а опора сравнения: обе головы обязаны различаться только тем,
    что заявлено.
    """
    import torch.nn as nn
    RMSNorm = make_rms_norm(torch)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNorm(d_model, eps)
            if hidden:
                self.net = nn.Sequential(
                    nn.Linear(d_model, hidden), nn.GELU(),
                    nn.Linear(hidden, vocab))
            else:
                self.net = nn.Linear(d_model, vocab)

        def forward(self, h):
            return self.net(self.norm(h))

    return Head()


def select_variant(results, key="val_sel"):
    """Выбор варианта и эпохи ПО val_sel, детерминированно.

    При равенстве берётся меньшее число параметров, затем меньший сид: без
    этого правила выбор зависел бы от порядка перебора, а он от случая.
    """
    if not results:
        raise SystemExit("нечего выбирать: нет результатов")
    def rank(r):
        return (round(float(r[key]), 12), int(r["hidden"]), int(r["seed"]))
    best = min(results, key=rank)
    return best


def check_manifest(man, *, q1_man, need_parts):
    """Кэш h18 обязан быть построен на тех же артефактах, что цели.

    ОТСУТСТВИЕ ПОЛЯ — ОТКАЗ. Кэш состояний и кэш целей строились разными
    запусками; если они разошлись по черновику, плану или кодовым книгам, то
    голова обучалась бы читать одно, а целилась бы в другое.
    """
    need = ("kind", "parts", "n_rows", "n_pos", "d_model", "dtype",
            "rows_sha1", "h18_sha1", "meta_sha1", "norm_class", "norm_eps",
            "head_ckpt", "head_state_sha1", "feedback_baked_in",
            "codebooks_sha1", "keys_sha1", "q0_npz_sha1", "plan_sha1",
            "gate_r_sha1", "q0_manifest_sha1")
    miss = [k for k in need if man.get(k) is None]
    if miss:
        raise SystemExit(f"в манифесте кэша h18 нет полей {miss}")
    if man["kind"] != "k14_h18_cache":
        raise SystemExit(f"манифест описывает {man['kind']}")
    if man.get("feedback_baked_in") is not True:
        raise SystemExit("кэш снят без вшитой обратной связи: §46 описывает "
                         "другую постановку")
    absent = [p for p in need_parts if p not in (man.get("parts") or ())]
    if absent:
        raise SystemExit(f"в кэше нет частей {absent}")
    bad = [k for k in ("codebooks_sha1", "keys_sha1", "plan_sha1",
                       "q0_npz_sha1", "gate_r_sha1")
           if str(man[k]) != str(q1_man.get(k))]
    if bad:
        raise SystemExit(f"кэш h18 и кэш целей расходятся по {bad}")
    return True


def selftest():
    import torch
    torch.manual_seed(0)
    d, V = 16, 32
    # --- НОРМА ВОСПРОИЗВЕДЕНА ТОЧНО ---------------------------------------
    RMSNorm = make_rms_norm(torch)
    eps = 1e-5
    n = RMSNorm(d, eps)
    with torch.no_grad():
        n.weight.copy_(torch.randn(d))
    x = torch.randn(4, 3, d)
    ref = n.weight * (x.float()
                      * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                    + eps))
    assert torch.allclose(n(x), ref, atol=0, rtol=0)
    # нулевой вход не даёт nan благодаря eps
    assert torch.isfinite(n(torch.zeros(2, 2, d))).all()

    # --- hidden = 0 ТОЖДЕСТВЕННО ЛИНЕЙНОЙ ГОЛОВЕ ---------------------------
    h0 = make_head(torch, d, V, 0, eps)
    assert isinstance(h0.net, torch.nn.Linear)
    assert h0(x).shape == (4, 3, V)
    h1 = make_head(torch, d, V, 8, eps)
    assert len(list(h1.net)) == 3 and h1(x).shape == (4, 3, V)
    n0 = sum(p.numel() for p in h0.parameters())
    n1 = sum(p.numel() for p in h1.parameters())
    assert n1 != n0

    # --- ВЫБОР ВАРИАНТА ----------------------------------------------------
    res = [dict(hidden=1024, seed=1, val_sel=0.130),
           dict(hidden=512, seed=0, val_sel=0.129),
           dict(hidden=2048, seed=2, val_sel=0.131)]
    assert select_variant(res)["hidden"] == 512
    # при равенстве — меньше параметров, затем меньший сид
    tie = [dict(hidden=2048, seed=0, val_sel=0.129),
           dict(hidden=512, seed=3, val_sel=0.129),
           dict(hidden=512, seed=1, val_sel=0.129)]
    b = select_variant(tie)
    assert (b["hidden"], b["seed"]) == (512, 1), b
    try:
        select_variant([])
    except SystemExit:
        pass
    else:
        raise AssertionError("выбор из пустого списка")

    # --- СВЕРКА МАНИФЕСТОВ -------------------------------------------------
    q1m = dict(codebooks_sha1="CB", keys_sha1="KS", plan_sha1="PL",
               q0_npz_sha1="QN", gate_r_sha1="GR")
    man = dict(kind="k14_h18_cache", parts=["train", "val_sel"], n_rows=10,
               n_pos=16, d_model=8, dtype="float16", rows_sha1="R",
               h18_sha1="H", meta_sha1="M", norm_class="RMSNorm",
               norm_eps=1e-5, head_ckpt="c.pt", head_state_sha1="S",
               feedback_baked_in=True, q0_manifest_sha1="QM", **q1m)
    assert check_manifest(man, q1_man=q1m, need_parts=("train", "val_sel"))
    for patch, why in (({"kind": "x"}, "описывает"),
                       ({"feedback_baked_in": False}, "вшитой"),
                       ({"codebooks_sha1": "Z"}, "расходятся"),
                       ({"plan_sha1": "Z"}, "расходятся")):
        try:
            check_manifest(dict(man, **patch), q1_man=q1m,
                           need_parts=("train",))
        except SystemExit as e:
            assert why in str(e), (why, e)
        else:
            raise AssertionError(f"манифест принят при {patch}")
    try:
        check_manifest(man, q1_man=q1m,
                       need_parts=("train", "val_confirm"))
    except SystemExit as e:
        assert "нет частей" in str(e), e
    else:
        raise AssertionError("принят кэш без нужной части")
    for k in sorted(man):
        try:
            check_manifest({x: v for x, v in man.items() if x != k},
                           q1_man=q1m, need_parts=("train",))
        except SystemExit as e:
            assert "нет полей" in str(e) or "описывает" in str(e) \
                or "вшитой" in str(e), (k, e)
        else:
            raise AssertionError(f"манифест без {k} принят")
    print("самопроверка k14f_mlp_probe пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--h18", default="data/k14e/h18")
    ap.add_argument("--q1-cache", default="data/k14b/q1_canonical")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--verify-only", action="store_true",
                    help="только сверка линейной головы на кэше, без обучения")
    ap.add_argument("--expect-rms", type=float, default=0.142997,
                    help="RMS-8 линейной головы на val_sel из живого прогона")
    ap.add_argument("--expect-top1", type=float, default=0.1394)
    ap.add_argument("--verify-tol", type=float, default=2e-5)
    ap.add_argument("--hidden", default="512,1024,2048")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lambda-action", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="reports/k14f/mlp_probe.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if os.path.exists(a.out) and not a.verify_only:
        raise SystemExit(f"{a.out} уже существует")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import torch.nn.functional as F
    import k12b_protocol as kb
    import k14_common as kc
    from depth_rvq_vla import straight_through
    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    H_EXEC = 8
    git_head, dirty, _ = kc.check_code_clean(True)
    dev = torch.device(a.device)

    man = json.load(open(a.h18 + ".manifest.json"))
    q1_man = json.load(open(a.q1_cache + ".manifest.json"))
    need_parts = ("train", "val_sel") if a.verify_only else \
        ("train", "val_sel", "val_confirm")
    check_manifest(man, q1_man=q1_man, need_parts=need_parts)
    if sha12(a.h18 + ".h18.npy") != man["h18_sha1"]:
        raise SystemExit("кэш h18 не совпал с отпечатком манифеста")
    if sha12(a.h18 + ".meta.npz") != man["meta_sha1"]:
        raise SystemExit("мета кэша не совпала с отпечатком манифеста")
    if str(man["norm_class"]).lower() not in ("rmsnorm", "llamarmsnorm",
                                              "smolvlmrmsnorm"):
        raise SystemExit(
            f"норма класса {man['norm_class']} не воспроизводится локальной "
            f"RMSNorm; сверка ниже это и покажет, но лучше остановиться сразу")

    d_model, n_pos = int(man["d_model"]), int(man["n_pos"])
    Hm = np.load(a.h18 + ".h18.npy", mmap_mode="r")
    mt = np.load(a.h18 + ".meta.npz", allow_pickle=True)
    rows, part = np.asarray(mt["rows"], np.int64), mt["part"].astype(str)
    if arr_sha(rows) != man["rows_sha1"]:
        raise SystemExit("номера строк кэша не совпали с отпечатком")
    if Hm.shape != (len(rows), n_pos, d_model):
        raise SystemExit(f"кэш формы {Hm.shape}")
    q0_c = np.asarray(mt["q0"], np.int64)
    ACT = np.asarray(mt["action"], np.float32)

    with np.load(a.q1_cache + ".npz", allow_pickle=True) as z:
        t_rows = np.asarray(z["rows"], np.int64)
        t_q1 = np.asarray(z["q1"], np.int64)
    pos_of = {int(r): i for i, r in enumerate(t_rows)}
    miss = [int(r) for r in rows if int(r) not in pos_of]
    if miss:
        raise SystemExit(f"{len(miss)} строк кэша нет в кэше целей")
    TG = t_q1[[pos_of[int(r)] for r in rows]]

    E = np.load(f"{a.cache}.codebooks.npy")
    if arr_sha(np.asarray(E, np.float32)) != man["codebooks_sha1"]:
        raise SystemExit("книги не совпали с теми, на которых снят кэш")
    books = torch.from_numpy(np.asarray(E, np.float32)).to(dev)
    V = int(books.shape[1])

    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    codec = proc.action_processor.to(dev).eval()
    for p_ in codec.parameters():
        p_.requires_grad_(False)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    wq = torch.as_tensor(max_act_q[:7], device=dev,
                         dtype=torch.float32).clone()
    wq[-1] = 1.0

    idx_of = {p: np.where(part == p)[0] for p in np.unique(part)}
    print(f"  кэш h18: {len(rows)} строк, d_model {d_model}, "
          f"части " + ", ".join(f"{k} {len(v)}" for k, v in idx_of.items()))
    print(f"  обратная связь вшита из {man['head_ckpt']} "
          f"(состояние {man['head_state_sha1']})")

    def decode(z, batch=512):
        out = []
        for i in range(0, len(z), batch):
            x, _ = codec._decode(z[i:i + batch].float(), embodiment_ids=0)
            out.append(x[..., :7].float())
        return torch.cat(out)

    def evaluate(head, ii, bs=512, soft=False):
        """RMS-8, CE и top-1 на строках `ii`. argmax, без ST."""
        head.eval()
        se = n = 0.0
        ce_s = corr = ntok = 0.0
        with torch.no_grad():
            for i in range(0, len(ii), bs):
                j = ii[i:i + bs]
                h = torch.from_numpy(np.asarray(Hm[j])).to(dev).float()
                tg = torch.from_numpy(TG[j]).to(dev)
                lg = head(h)
                ce_s += float(F.cross_entropy(
                    lg.reshape(-1, V), tg.reshape(-1), reduction="sum"))
                corr += float((lg.argmax(-1) == tg).sum())
                ntok += int(tg.numel())
                e0 = books[0][torch.from_numpy(q0_c[j]).to(dev)]
                emb = books[1][lg.argmax(-1)]
                ah = decode(e0 + emb)
                at = torch.from_numpy(ACT[j]).to(dev)[..., :7]
                dd = (ah[:, :H_EXEC] - at[:, :H_EXEC]) * wq
                se += float((dd ** 2).sum()); n += int(dd.numel())
        return dict(rms=float(np.sqrt(se / max(n, 1))),
                    ce=ce_s / max(ntok, 1), top1=corr / max(ntok, 1),
                    n_rows=int(len(ii)))

    # --- ОБЯЗАТЕЛЬНАЯ СВЕРКА: ЛИНЕЙНАЯ ГОЛОВА НА КЭШЕ -----------------------
    ck = torch.load(man["head_ckpt"], map_location="cpu", weights_only=False)
    st = ck["state"]
    lin = make_head(torch, d_model, V, 0, float(man["norm_eps"])).to(dev)
    with torch.no_grad():
        lin.norm.weight.copy_(st["depth_rvq_norms.0.weight"].float())
        lin.net.weight.copy_(st["depth_rvq_heads.0.weight"].float())
        if "depth_rvq_heads.0.bias" in st:
            lin.net.bias.copy_(st["depth_rvq_heads.0.bias"].float())
        elif lin.net.bias is not None:
            lin.net.bias.zero_()
    vs = idx_of["val_sel"]
    got = evaluate(lin, vs)
    print(f"\n  СВЕРКА линейной головы на кэше: RMS-8 {got['rms']:.6f} "
          f"(ожидалось {a.expect_rms:.6f}), top-1 {100 * got['top1']:.2f}% "
          f"(ожидалось {100 * a.expect_top1:.2f}%), CE {got['ce']:.5f}")
    bad = []
    if abs(got["rms"] - a.expect_rms) > a.verify_tol:
        bad.append(f"RMS {got['rms']:.6f} против {a.expect_rms:.6f}")
    if abs(got["top1"] - a.expect_top1) > 5e-4:
        bad.append(f"top-1 {got['top1']:.4f} против {a.expect_top1:.4f}")
    if bad:
        raise SystemExit(
            "линейная голова на кэше НЕ воспроизвела живые числа: "
            + "; ".join(bad)
            + ". Либо кэш снят не с того состояния, либо норма "
              "воспроизведена неточно. Обучать на таком кэше нечего")
    print("  сверка пройдена: кэш и воспроизведение нормы верны")
    if a.verify_only:
        return 0

    # --- ОБУЧЕНИЕ ПРОБЫ -----------------------------------------------------
    hiddens = [int(x) for x in a.hidden.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    tr, vc = idx_of["train"], idx_of["val_confirm"]
    results, hist_all = [], {}
    t0 = time.time()
    for hid in hiddens:
        for sd in seeds:
            torch.manual_seed(0)          # инициализация от сида НЕ зависит
            head = make_head(torch, d_model, V, hid,
                             float(man["norm_eps"])).to(dev)
            with torch.no_grad():
                head.norm.weight.copy_(st["depth_rvq_norms.0.weight"].float())
            npar = sum(p.numel() for p in head.parameters())
            opt = torch.optim.AdamW(head.parameters(), lr=a.lr,
                                    weight_decay=a.wd)
            rng = np.random.default_rng(sd)
            best = None
            hist = []
            for ep in range(1, a.epochs + 1):
                head.train()
                order = rng.permutation(tr)
                run = nb = 0.0
                for i in range(0, len(order), a.batch):
                    j = np.sort(order[i:i + a.batch])
                    h = torch.from_numpy(np.asarray(Hm[j])).to(dev).float()
                    tg = torch.from_numpy(TG[j]).to(dev)
                    lg = head(h)
                    ce = F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1))
                    e0 = books[0][torch.from_numpy(q0_c[j]).to(dev)]
                    emb, _, _ = straight_through(lg, books[1], tau=1.0)
                    ah = decode(e0 + emb)
                    at = torch.from_numpy(ACT[j]).to(dev)[..., :7]
                    dd = (ah[:, :H_EXEC] - at[:, :H_EXEC])
                    al = (dd ** 2).mean()
                    loss = ce + a.lambda_action * al
                    if not torch.isfinite(loss):
                        raise SystemExit(f"потеря не число на эпохе {ep}")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    run += float(loss.detach()); nb += 1
                ev = evaluate(head, vs)
                hist.append(dict(epoch=ep, loss=run / max(nb, 1), **ev))
                if best is None or ev["rms"] < best["val_sel"]:
                    best = dict(hidden=hid, seed=sd, epoch=ep,
                                val_sel=ev["rms"], top1=ev["top1"],
                                ce=ev["ce"], n_params=int(npar),
                                state={k: v.detach().cpu().clone()
                                       for k, v in head.state_dict().items()})
                print(f"    h={hid} сид {sd} эпоха {ep}: потеря "
                      f"{run / max(nb, 1):.5f}, val_sel RMS {ev['rms']:.6f}, "
                      f"top-1 {100 * ev['top1']:.2f}% "
                      f"({(time.time() - t0) / 60:.1f} мин)", flush=True)
            tr_ev = evaluate(head, tr[:len(vs)])
            best["train_slice"] = tr_ev
            hist_all[f"h{hid}_s{sd}"] = hist
            results.append(best)

    pick = select_variant(results)
    print(f"\n  ВЫБРАНО по val_sel: h={pick['hidden']}, сид {pick['seed']}, "
          f"эпоха {pick['epoch']}, val_sel RMS {pick['val_sel']:.6f}, "
          f"top-1 {100 * pick['top1']:.2f}%")

    head = make_head(torch, d_model, V, pick["hidden"],
                     float(man["norm_eps"])).to(dev)
    head.load_state_dict(pick["state"])
    conf_mlp = evaluate(head, vc)
    conf_lin = evaluate(lin, vc)
    print("\n  ПОДТВЕРЖДЕНИЕ (post-hoc, повторно использованная половина):")
    print(f"    MLP      RMS-8 {conf_mlp['rms']:.6f}  top-1 "
          f"{100 * conf_mlp['top1']:.2f}%")
    print(f"    линейная RMS-8 {conf_lin['rms']:.6f}  top-1 "
          f"{100 * conf_lin['top1']:.2f}%")
    print(f"    разность {conf_lin['rms'] - conf_mlp['rms']:+.6f} "
          f"(положительная — MLP лучше)")

    out = dict(kind="k14f_mlp_probe", note="post-hoc проверка на повторно "
               "использованной val_confirm, не независимое подтверждение",
               results=[{k: v for k, v in r.items() if k != "state"}
                        for r in results],
               selected={k: v for k, v in pick.items() if k != "state"},
               history=hist_all, verify_linear_on_cache=got,
               expect_rms=a.expect_rms, expect_top1=a.expect_top1,
               confirm_mlp=conf_mlp, confirm_linear=conf_lin,
               h18_manifest_sha1=sha12(a.h18 + ".manifest.json"),
               h18_sha1=man["h18_sha1"], q1_cache=a.q1_cache,
               epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd,
               lambda_action=a.lambda_action, device=str(dev),
               gpu_uuid=kc.gpu_uuid(dev, torch), git_head=git_head,
               git_dirty=bool(dirty),
               code_version=kb.code_version([
                   os.path.abspath(__file__),
                   os.path.join(here, "k14_common.py"),
                   os.path.join(here, "depth_rvq_vla.py")]),
               script_sha1=sha12(os.path.abspath(__file__)),
               minutes=float((time.time() - t0) / 60))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
