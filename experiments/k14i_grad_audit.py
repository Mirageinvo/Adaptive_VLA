#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14i: аудит градиентов обучающей цели (§48). Без единого шага обучения.

ВОПРОС. §46.1 дал прямой пример: голова угадывает коды ЛУЧШЕ и при этом даёт
БОЛЬШУЮ ошибку действия. Отсюда гипотеза: слагаемое CE тянет не туда, куда
нужно, а возможно, и мешает.

ПОЧЕМУ ЭТОГО НЕ ВИДНО ПО ЗНАЧЕНИЯМ ПОТЕРЬ. Величина CE 3-4 против малой доли
от потери действия не означает, что CE доминирует в ШАГЕ: значение слагаемого
и его вклад в обновление — разные вещи. Решают нормы градиентов и их взаимное
направление, и меряются они прямо.

ЧТО СЧИТАЕТСЯ на фиксированных батчах, по обучаемым весам головы:

    ||g_CE||                    норма градиента от кросс-энтропии
    ||g_action||                норма градиента от ошибки действия
                                в ВЕСАХ ОБУЧЕНИЯ
    ||g_action_gate||           то же в ВЕСАХ GATE 4
    cos(g_CE, g_action)         направление: конфликтуют ли слагаемые
    cos(g_action, g_action_gate) насколько расходятся две системы весов

ЧТО ЭТО МОЖЕТ ЗАКРЫТЬ. Если косинус около нуля или отрицателен — слагаемые
тянут в разные стороны, и §48 осмыслен. Если градиент от CE на порядок
больше — осмыслен вдвойне. Если ни того, ни другого — переделывать цель
незачем, и ветка закрывается двадцатью минутами вместо семи часов обучения.

    python experiments/k14i_grad_audit.py --device cuda:1 \
        --heads reports/k14f/linear_b8e4.pt
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def flat_cos(a, b):
    """Косинус между двумя наборами градиентов, склеенными в один вектор.

    СКЛЕИВАТЬ ОБЯЗАТЕЛЬНО. Косинус, усреднённый по тензорам, отвечал бы на
    другой вопрос — про согласие по каждому весу отдельно, — и мог бы быть
    близок к нулю при полном согласии шага, и наоборот.
    """
    import torch
    x = torch.cat([g.reshape(-1).double() for g in a])
    y = torch.cat([g.reshape(-1).double() for g in b])
    nx, ny = float(x.norm()), float(y.norm())
    if nx == 0.0 or ny == 0.0:
        return float("nan"), nx, ny
    return float((x @ y) / (nx * ny)), nx, ny


def summarize(rows, keys):
    """МЕДИАНА как основная точка, плюс диапазон и число батчей.

    Какая именно сводная величина решает — определено ДО прогона (§48).
    Иначе после результата можно было бы выбрать среднее, медиану или
    агрегированный градиент в зависимости от того, что удобнее. Медиана
    устойчива к одному выбросу, а диапазон и число батчей печатаются рядом
    всегда: одно число скрыло бы случай, когда на половине батчей слагаемые
    согласны, а на половине конфликтуют.
    """
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        if not len(v):
            out[k] = None
            continue
        out[k] = dict(median=float(np.median(v)), mean=float(v.mean()),
                      min=float(v.min()), max=float(v.max()), n=int(len(v)),
                      n_negative=int((v < 0).sum()))
    return out


COS_CONFLICT, COS_ORTH, RATIO_DOMINATES = 0.0, 0.2, 10.0


def verdict(s, cos_sum=None):
    """Что из чисел следует. Пороги и сводные величины названы до прогона.

    ВСЕ ПОРОГИ ОТНОСЯТСЯ К МЕДИАНЕ по батчам, и отношение норм берётся тем
    же способом — смешивать медиану косинусов со средним отношением нельзя,
    это дало бы свободу выбрать удобную пару.

    Косинус СУММАРНЫХ по батчам градиентов приводится отдельно: он отвечает
    на другой вопрос — про направление накопленного шага, а не про типичный
    батч, — и в пороги не входит.
    """
    c, r = s["cos_ce_act"], s["ratio_ce_over_act"]
    cm, rm = c["median"], r["median"]
    reasons = []
    if cm < COS_CONFLICT:
        reasons.append(f"медианный косинус отрицателен ({cm:+.3f}): "
                       f"слагаемые конфликтуют")
    elif cm < COS_ORTH:
        reasons.append(f"медианный косинус {cm:+.3f} ниже {COS_ORTH}: "
                       f"слагаемые почти ортогональны")
    if rm > RATIO_DOMINATES:
        reasons.append(f"медианное отношение норм {rm:.1f} больше "
                       f"{RATIO_DOMINATES}: CE доминирует в шаге")
    if c["n_negative"] and cm >= COS_CONFLICT:
        reasons.append(f"конфликт на {c['n_negative']} батчах из {c['n']} "
                       f"при неотрицательной медиане")
    txt = ("; ".join(reasons) if reasons else
           "гипотеза §48 не поддержана: по медиане слагаемые согласованы по "
           "направлению и сопоставимы по величине")
    return dict(text=txt, supports=bool(reasons), cos_median=cm,
                ratio_median=rm, n_negative=c["n_negative"], n=c["n"],
                cos_of_summed=cos_sum)


def selftest():
    import torch
    torch.manual_seed(0)
    a = [torch.ones(3), torch.zeros(2)]
    c, na, nb = flat_cos(a, [t.clone() for t in a])
    assert abs(c - 1.0) < 1e-12 and abs(na - 3 ** 0.5) < 1e-9
    c, _, _ = flat_cos(a, [-t for t in a])
    assert abs(c + 1.0) < 1e-12
    x = [torch.tensor([1.0, 0.0])]
    y = [torch.tensor([0.0, 1.0])]
    assert abs(flat_cos(x, y)[0]) < 1e-12
    # НУЛЕВОЙ ГРАДИЕНТ НЕ ДАЁТ ЧИСЛА, А НЕ ДАЁТ НОЛЬ
    c, _, _ = flat_cos(x, [torch.zeros(2)])
    assert c != c, "нулевой градиент обязан давать nan, а не косинус"
    # СКЛЕЙКА, А НЕ СРЕДНЕЕ ПО ТЕНЗОРАМ: здесь потензорные косинусы +1 и −1,
    # а склеенный определяется величинами.
    p = [torch.tensor([10.0]), torch.tensor([1.0])]
    q = [torch.tensor([10.0]), torch.tensor([-1.0])]
    cc, _, _ = flat_cos(p, q)
    assert cc > 0.9, cc

    rows = [dict(cos=0.5, r=1.0), dict(cos=-0.1, r=3.0),
            dict(cos=float("nan"), r=2.0)]
    s = summarize(rows, ("cos", "r"))
    assert s["cos"]["n"] == 2 and abs(s["cos"]["min"] + 0.1) < 1e-12
    assert s["r"]["n"] == 3 and s["r"]["max"] == 3.0

    def mk(cos_vals, ratio_vals):
        rr = [dict(cos_ce_act=c, ratio_ce_over_act=r)
              for c, r in zip(cos_vals, ratio_vals)]
        return summarize(rr, ("cos_ce_act", "ratio_ce_over_act"))

    # ПОРОГИ ОТНОСЯТСЯ К МЕДИАНЕ, и это проверяется прямо: набор, у которого
    # среднее и медиана указывают в разные стороны, обязан читаться по
    # медиане.
    v = verdict(mk([-0.3] * 9 + [0.9] * 7, [2.0] * 16))
    assert v["supports"] and "медианный косинус отрицателен" in v["text"]
    v = verdict(mk([0.8] * 16, [30.0] * 16))
    assert v["supports"] and "доминирует" in v["text"]
    v = verdict(mk([0.7] * 14 + [-0.2] * 2, [2.0] * 16))
    assert v["supports"] and "конфликт на 2 батчах" in v["text"]
    v = verdict(mk([0.8] * 16, [2.0] * 16))
    assert not v["supports"] and "не поддержана" in v["text"]
    # СРЕДНЕЕ СКАЗАЛО БЫ ДРУГОЕ: медиана 0.5, среднее отрицательно
    ss = mk([0.5] * 9 + [-9.0] * 7, [2.0] * 16)
    assert ss["cos_ce_act"]["mean"] < 0 < ss["cos_ce_act"]["median"]
    v = verdict(ss)
    assert "медианный косинус отрицателен" not in v["text"]
    assert v["cos_median"] == 0.5 and v["n_negative"] == 7
    v = verdict(mk([0.8] * 16, [2.0] * 16), cos_sum=-0.4)
    assert v["cos_of_summed"] == -0.4 and not v["supports"], \
        "косинус суммарных градиентов в пороги входить не должен"
    print("самопроверка k14i_grad_audit пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--h18", default="data/k14e/h18")
    ap.add_argument("--q1-cache", default="data/k14b/q1_canonical")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--heads", nargs="*", default=[],
                    help="чекпойнты K-14f; историческая линейная берётся "
                         "из кэша h18 всегда")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-batches", type=int, default=16)
    ap.add_argument("--lambda-action", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="reports/k14i/grad_audit.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} уже существует")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import torch.nn.functional as F
    import k11a_build_hicora_cache as k11a
    import k12b_protocol as kb
    import k14_common as kc
    from depth_rvq_vla import straight_through
    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    sp = importlib.util.spec_from_file_location(
        "_k14f", os.path.join(here, "k14f_mlp_probe.py"))
    k14f = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(k14f)

    H_EXEC = 8
    dev = torch.device(a.device)
    git_head, dirty, _ = kc.check_code_clean(a.allow_dirty)

    man = json.load(open(a.h18 + ".manifest.json"))
    q1_man = json.load(open(a.q1_cache + ".manifest.json"))
    k14f.check_manifest(man, q1_man=q1_man, need_parts=("train",))
    if sha12(a.h18 + ".h18.npy") != man["h18_sha1"]:
        raise SystemExit("кэш h18 не совпал с отпечатком манифеста")
    if sha12(a.h18 + ".meta.npz") != man["meta_sha1"]:
        raise SystemExit("мета кэша не совпала с отпечатком манифеста")
    d_model = int(man["d_model"])
    Hm = np.load(a.h18 + ".h18.npy", mmap_mode="r")
    mt = np.load(a.h18 + ".meta.npz", allow_pickle=True)
    rows, part = np.asarray(mt["rows"], np.int64), mt["part"].astype(str)
    q0_c = np.asarray(mt["q0"], np.int64)
    ACT = np.asarray(mt["action"], np.float32)

    t_rows, t_q1, t_part, _ = kc.load_q1_targets(a.q1_cache, q1_man)
    pos_of = {int(r): i for i, r in enumerate(t_rows)}
    TG = np.ascontiguousarray(t_q1[[pos_of[int(r)] for r in rows]], np.int64)

    E = np.load(f"{a.cache}.codebooks.npy")
    if k14f.arr_sha(np.asarray(E, np.float32)) != man["codebooks_sha1"]:
        raise SystemExit("книги не совпали с теми, на которых снят кэш")
    books = torch.from_numpy(np.asarray(E, np.float32)).to(dev)
    V = int(books.shape[1])

    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    codec = proc.action_processor.to(dev).eval()
    for p_ in codec.parameters():
        p_.requires_grad_(False)
    if str(k11a.state_sha1(codec)) != str(q1_man.get("codec_state_sha1")):
        raise SystemExit("декодер не тот, на котором построены цели")
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    wg = torch.as_tensor(max_act_q[:7], device=dev,
                         dtype=torch.float32).clone()
    wg[-1] = 1.0
    w_tr = torch.ones(7, device=dev, dtype=torch.float32)

    tr = np.where(part == "train")[0]
    # БАТЧИ ФИКСИРОВАНЫ И РАВНОМЕРНЫ ПО ВСЕМУ НАБОРУ: аудит не должен
    # зависеть ни от случайности, ни от того, что первые строки — это
    # определённые эпизоды.
    take = np.linspace(0, len(tr) - a.batch - 1, a.n_batches).astype(int)
    batches = [tr[i:i + a.batch] for i in take]
    # ОТПЕЧАТОК ВЫБОРА БАТЧЕЙ. Вердикт ветвит весь дальнейший план, и по
    # артефакту должно быть видно не только «16 батчей», но и какие именно:
    # иначе повторить измерение можно только на слово.
    batch_rows = np.concatenate([rows[j] for j in batches]).astype(np.int64)
    batches_sha = k14f.arr_sha(batch_rows)
    print(f"  батчей {len(batches)} по {a.batch} строк, равномерно по "
          f"{len(tr)} строкам train; отпечаток выбора {batches_sha}")

    def decode(z, bs=512):
        out = []
        for i in range(0, len(z), bs):
            x, _ = codec._decode(z[i:i + bs].float(), embodiment_ids=0)
            out.append(x[..., :7].float())
        return torch.cat(out)

    def make_heads():
        out = []
        ck0 = torch.load(man["head_ckpt"], map_location="cpu",
                         weights_only=False)
        lin = k14f.make_head(torch, d_model, V, 0,
                             float(man["norm_eps"])).to(dev)
        with torch.no_grad():
            st = ck0["state"]
            lin.norm.weight.copy_(st["depth_rvq_norms.0.weight"].float())
            lin.net.weight.copy_(st["depth_rvq_heads.0.weight"].float())
            if "depth_rvq_heads.0.bias" in st:
                lin.net.bias.copy_(st["depth_rvq_heads.0.bias"].float())
            elif lin.net.bias is not None:
                lin.net.bias.zero_()
        out.append(("линейная (историческая)", lin,
                    dict(checkpoint=man["head_ckpt"],
                         file_sha1=sha12(man["head_ckpt"]),
                         selected_state_sha1=ck0.get("selected_state_sha1"),
                         hidden=0)))
        for p in a.heads:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if str(ck.get("h18_sha1")) != str(man["h18_sha1"]):
                raise SystemExit(f"{p} обучен на другом кэше h18")
            h = k14f.make_head(torch, d_model, V, int(ck["hidden"]),
                               float(ck.get("norm_eps", man["norm_eps"]))
                               ).to(dev)
            h.load_state_dict(ck["state"])
            got = k14f.state_sha_np({k: v.detach().float().cpu().numpy()
                                     for k, v in h.state_dict().items()})
            if got != str(ck["selected_state_sha1"]):
                raise SystemExit(f"{p}: отпечаток после загрузки {got}")
            out.append((os.path.basename(p)[:-3], h,
                        dict(checkpoint=p, file_sha1=sha12(p),
                             selected_state_sha1=ck["selected_state_sha1"],
                             hidden=int(ck["hidden"]),
                             seed=ck.get("seed"), epoch=ck.get("epoch"))))
        return out

    res = {}
    for label, head, prov in make_heads():
        for p_ in head.parameters():
            p_.requires_grad_(True)
        named = list(head.named_parameters())
        par = [p_ for _n, p_ in named]
        # ГРУППЫ ПАРАМЕТРОВ. Общий косинус может скрыть согласие в одной
        # части головы и конфликт в другой; для головы на кэше групп две —
        # норма и сама голова. Ветви обратной связи здесь НЕТ: она вшита в
        # кэш h18 и не обучается, и это ограничение самой постановки, а не
        # недосмотр.
        groups = {"norm": [i for i, (n_, _) in enumerate(named)
                           if n_.startswith("norm.")],
                  "head": [i for i, (n_, _) in enumerate(named)
                           if n_.startswith("net.")]}
        groups = {k: v for k, v in groups.items() if v}
        acc_ce = [torch.zeros_like(p_) for p_ in par]
        acc_ac = [torch.zeros_like(p_) for p_ in par]
        rows_out = []
        for j in batches:
            h = torch.from_numpy(np.asarray(Hm[j])).to(dev).float()
            tg = torch.from_numpy(TG[j]).to(dev)
            e0 = books[0][torch.from_numpy(q0_c[j]).to(dev)]
            at = torch.from_numpy(ACT[j]).to(dev)[..., :7]
            lg = head(h)
            ce = F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1))
            emb, _, _ = straight_through(lg, books[1], tau=1.0)
            ah = decode(e0 + emb)
            dd = ah[:, :H_EXEC] - at[:, :H_EXEC]
            al_tr = ((dd * w_tr) ** 2).mean()
            al_gt = ((dd * wg) ** 2).mean()
            g_ce = torch.autograd.grad(ce, par, retain_graph=True,
                                       allow_unused=False)
            g_tr = torch.autograd.grad(al_tr, par, retain_graph=True,
                                       allow_unused=False)
            g_gt = torch.autograd.grad(al_gt, par, retain_graph=False,
                                       allow_unused=False)
            c_ce_tr, n_ce, n_tr = flat_cos(g_ce, g_tr)
            c_ce_gt, _, n_gt = flat_cos(g_ce, g_gt)
            c_tr_gt, _, _ = flat_cos(g_tr, g_gt)
            for i_, (x_, y_) in enumerate(zip(g_ce, g_tr)):
                acc_ce[i_] += x_.detach()
                acc_ac[i_] += y_.detach()
            by_grp = {}
            for gname, idxs in groups.items():
                cg, ng_ce, ng_ac = flat_cos([g_ce[i] for i in idxs],
                                            [g_tr[i] for i in idxs])
                by_grp[gname] = dict(cos=cg, n_ce=ng_ce, n_act=ng_ac)
            rows_out.append(dict(
                **{f"cos_{k}": v["cos"] for k, v in by_grp.items()},
                **{f"n_ce_{k}": v["n_ce"] for k, v in by_grp.items()},
                **{f"n_act_{k}": v["n_act"] for k, v in by_grp.items()},
                ce=float(ce), act_train=float(al_tr), act_gate=float(al_gt),
                n_ce=n_ce, n_act_train=n_tr, n_act_gate=n_gt,
                # ВКЛАД В ШАГ, а не значение слагаемого: потеря действия
                # входит с весом lambda, и сравнивать надо именно это.
                ratio_ce_over_act=(n_ce / (a.lambda_action * n_tr)
                                   if n_tr > 0 else float("nan")),
                cos_ce_act=c_ce_tr, cos_ce_act_gate=c_ce_gt,
                cos_act_train_gate=c_tr_gt))
        keys = (["ce", "act_train", "act_gate", "n_ce", "n_act_train",
                 "n_act_gate", "ratio_ce_over_act", "cos_ce_act",
                 "cos_ce_act_gate", "cos_act_train_gate"]
                + [f"cos_{k}" for k in groups]
                + [f"n_ce_{k}" for k in groups]
                + [f"n_act_{k}" for k in groups])
        s = summarize(rows_out, keys)
        # КОСИНУС СУММАРНЫХ ГРАДИЕНТОВ: направление накопленного шага, а не
        # типичного батча. В пороги не входит, приводится рядом.
        cos_sum, _, _ = flat_cos(acc_ce, acc_ac)
        v = verdict(s, cos_sum=cos_sum)
        res[label] = dict(summary=s, per_batch=rows_out, verdict=v["text"],
                          supports_hypothesis=bool(v["supports"]),
                          decision=v, groups=sorted(groups), provenance=prov,
                          n_params=int(sum(p_.numel() for p_ in par)))
        txt = v["text"]
        print(f"\\n  === {label} ===")
        print(f"    ||g_CE||           медиана {s['n_ce']['median']:.5f}  "
              f"[{s['n_ce']['min']:.5f}, {s['n_ce']['max']:.5f}]")
        print(f"    ||g_действие||     медиана "
              f"{s['n_act_train']['median']:.5f}  "
              f"[{s['n_act_train']['min']:.5f}, {s['n_act_train']['max']:.5f}]")
        print(f"    отношение CE/дейст медиана "
              f"{s['ratio_ce_over_act']['median']:.2f}  "
              f"[{s['ratio_ce_over_act']['min']:.2f}, "
              f"{s['ratio_ce_over_act']['max']:.2f}]")
        print(f"    cos(CE, действие)  медиана "
              f"{s['cos_ce_act']['median']:+.4f}  "
              f"[{s['cos_ce_act']['min']:+.4f}, "
              f"{s['cos_ce_act']['max']:+.4f}], отрицательных "
              f"{s['cos_ce_act']['n_negative']} из {s['cos_ce_act']['n']}")
        for gname in sorted(groups):
            g = s[f"cos_{gname}"]
            print(f"      по группе {gname:5s} медиана {g['median']:+.4f}  "
                  f"[{g['min']:+.4f}, {g['max']:+.4f}]")
        print(f"    cos суммарных градиентов {cos_sum:+.4f}")
        print(f"    cos(действие в весах обучения, в весах гейта) медиана "
              f"{s['cos_act_train_gate']['median']:+.4f}")
        print(f"    ВЕРДИКТ: {txt}")

    out = dict(kind="k14i_grad_audit", heads=res, batch=int(a.batch),
               n_batches=int(a.n_batches), lambda_action=a.lambda_action,
               batches_sha1=batches_sha,
               batch_row_ids=[[int(x) for x in rows[j]] for j in batches],
               thresholds=dict(cos_conflict=COS_CONFLICT,
                               cos_orthogonal=COS_ORTH,
                               ratio_dominates=RATIO_DOMINATES,
                               main_statistic="median"),
               h18_sha1=man["h18_sha1"], q1_cache_sha1=q1_man["labels_sha1"],
               device=str(dev), gpu_uuid=kc.gpu_uuid(dev, torch),
               git_head=git_head, git_dirty=bool(dirty),
               code_version=kb.code_version([
                   os.path.abspath(__file__),
                   os.path.join(here, "k14_common.py"),
                   os.path.join(here, "depth_rvq_vla.py")]),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"\\n  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
