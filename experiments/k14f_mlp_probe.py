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


def state_sha_np(named):
    """Отпечаток именованных тензоров — та же функция, что в K-14c и K-14e."""
    h = hashlib.sha1()
    for k in sorted(named):
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            np.asarray(named[k], dtype=np.float64)).tobytes())
    return h.hexdigest()[:12]


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
            "gate_r_sha1", "q0_manifest_sha1", "q1_cache_sha1",
            "q1_manifest_sha1")
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
                       "q0_npz_sha1", "gate_r_sha1", "q0_manifest_sha1")
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
               q0_npz_sha1="QN", gate_r_sha1="GR", q0_manifest_sha1="QM")
    man = dict(kind="k14_h18_cache", parts=["train", "val_sel"], n_rows=10,
               n_pos=16, d_model=8, dtype="float16", rows_sha1="R",
               h18_sha1="H", meta_sha1="M", norm_class="RMSNorm",
               norm_eps=1e-5, head_ckpt="c.pt", head_state_sha1="S",
               feedback_baked_in=True, q1_cache_sha1="L",
               q1_manifest_sha1="QMAN", **q1m)
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
    import importlib.util, os as _os
    _h = _os.path.dirname(_os.path.abspath(__file__))
    sp = importlib.util.spec_from_file_location(
        "_k14c", _os.path.join(_h, "k14c_train_q1.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    ref = dict(a=np.arange(6, dtype=np.float32).reshape(2, 3),
               b=np.linspace(-1, 1, 4, dtype=np.float32))
    assert state_sha_np(ref) == m.state_sha(ref), \
        "отпечаток весов считается иначе, чем в K-14c"
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
    ap.add_argument("--objective", default="ce_action",
                    choices=("ce_action", "action_only", "action_only_gate"),
                    help="что оптимизируется. ce_action — как в каноническом "
                         "прогоне: CE + lambda * ошибка действия в весах "
                         "обучения. action_only — только ошибка действия в "
                         "тех же весах, CE убрана. action_only_gate — только "
                         "ошибка действия в ВЕСАХ GATE 4. CE и top-1 "
                         "считаются всегда, но на шаг влияют только в "
                         "ce_action")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="разрешить прогон на незакоммиченном коде")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="reports/k14f/mlp_probe.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    ck_out = (a.out[:-5] + ".pt") if a.out.endswith(".json") else a.out + ".pt"
    if not a.verify_only:
        # КАТАЛОГ И ОТСУТСТВИЕ ОБОИХ ВЫХОДОВ ПРОВЕРЯЮТСЯ ДО ОБУЧЕНИЯ.
        # Прежде `.pt` сохранялся раньше `makedirs`, и девять прогонов
        # падали бы в самом конце; существующий `.pt` перезаписывался молча.
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        for q in (a.out, ck_out):
            if os.path.exists(q):
                raise SystemExit(f"{q} уже существует")

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
    CODE_FILES = [os.path.abspath(__file__),
                  os.path.join(here, "k14_common.py"),
                  os.path.join(here, "depth_rvq_vla.py")]

    def code_state(strict):
        """Снимок кода. `strict` отказывает при грязном дереве, иначе нет.

        В НАЧАЛЕ СТРОГО, В КОНЦЕ НЕТ — и это не послабление, а условие того,
        чтобы отказ вообще случился. Прежде снимок в конце тоже отказывал при
        грязном дереве, то есть падал ДО сравнения версий и до сохранения
        результатов: любой новый файл рядом стоил бы всех часов счёта, ради
        защиты которых сохранение и заводилось.
        """
        h_, d_, _a = kc.check_code_clean(a.allow_dirty if strict else True)
        return h_, kb.code_version(CODE_FILES), bool(d_)

    git_head, code_v0, dirty = code_state(True)
    script_sha_0 = sha12(os.path.abspath(__file__))
    dev = torch.device(a.device)

    man = json.load(open(a.h18 + ".manifest.json"))
    q1_man = json.load(open(a.q1_cache + ".manifest.json"))
    # СВЕРКЕ НУЖНА ТОЛЬКО val_sel. Требовать при ней train значило бы
    # запретить дешёвый смоук: кэш одной части строится за минуты и
    # проверяет ровно ту же цепочку, что полный.
    need_parts = ("val_sel",) if a.verify_only else \
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

    # ЦЕЛИ ЗАГРУЖАЮТСЯ ОБЩИМ СТРОГИМ ЗАГРУЗЧИКОМ. Своя реализация здесь и
    # дала ошибку: отпечаток считался от приведённого к int64 массива, а
    # K-14b хеширует то, что записал, — int32. Правильный кэш отвергался бы
    # всегда. Загрузчик теперь один на всех потребителей.
    t_rows, t_q1, t_part, t_info = kc.load_q1_targets(a.q1_cache, q1_man)
    pos_of = {int(r): i for i, r in enumerate(t_rows)}
    miss = [int(r) for r in rows if int(r) not in pos_of]
    if miss:
        raise SystemExit(f"{len(miss)} строк кэша нет в кэше целей")
    TG = np.ascontiguousarray(t_q1[[pos_of[int(r)] for r in rows]],
                              dtype=np.int64)

    E = np.load(f"{a.cache}.codebooks.npy")
    if arr_sha(np.asarray(E, np.float32)) != man["codebooks_sha1"]:
        raise SystemExit("книги не совпали с теми, на которых снят кэш")
    books = torch.from_numpy(np.asarray(E, np.float32)).to(dev)
    V = int(books.shape[1])

    if int(TG.min()) < 0 or int(TG.max()) >= V:
        raise SystemExit(f"коды целей в диапазоне [{TG.min()}, {TG.max()}] "
                         f"при словаре {V}")
    if int(q0_c.min()) < 0 or int(q0_c.max()) >= V:
        raise SystemExit("коды q0 вне словаря")

    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    codec = proc.action_processor.to(dev).eval()
    for p_ in codec.parameters():
        p_.requires_grad_(False)
    # ЧЕРЕЗ ДЕКОДЕР ИДЁТ ПОТЕРЯ ДЕЙСТВИЯ И ЕЁ ГРАДИЕНТ. Совпадения итогового
    # RMS недостаточно: разные локальные ошибки дают почти одинаковое
    # среднее. Сверяются состояние кодека и проба декодера — те же величины,
    # что заверены в кэше целей.
    import k11a_build_hicora_cache as k11a
    cs_now = k11a.state_sha1(codec)
    dp_now = k11a.decoder_probe(codec, books.float(), dev)
    bad_c = []
    if str(cs_now) != str(q1_man.get("codec_state_sha1")):
        bad_c.append(f"состояние кодека {cs_now} против "
                     f"{q1_man.get('codec_state_sha1')}")
    if str(dp_now) != str(q1_man.get("decoder_probe")):
        bad_c.append(f"проба декодера {dp_now} против "
                     f"{q1_man.get('decoder_probe')}")
    if bad_c:
        raise SystemExit("декодер не тот, на котором построены цели: "
                         + "; ".join(bad_c))
    print(f"  декодер заверен: состояние {cs_now}, проба {dp_now}")
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
    # ОТПЕЧАТОК БАЗОВОЙ ГОЛОВЫ ПЕРЕСЧИТЫВАЕТСЯ И СВЕРЯЕТСЯ С КЭШЕМ h18:
    # именно её обратная связь вшита в кэш, и подмена чекпойнта означала бы,
    # что читается состояние, произведённое другой головой.
    base_sha = state_sha_np({k: v.detach().float().cpu().numpy()
                             for k, v in st.items()})
    if base_sha != str(man["head_state_sha1"]):
        raise SystemExit(
            f"базовая голова {man['head_ckpt']} имеет отпечаток {base_sha}, "
            f"а кэш h18 снят при {man['head_state_sha1']}")
    if str(ck.get("selected_state_sha1")) != base_sha:
        raise SystemExit(f"в чекпойнте записан отпечаток "
                         f"{ck.get('selected_state_sha1')}, фактический "
                         f"{base_sha}")
    # КЭШ ЦЕЛЕЙ СВЕРЯЕТСЯ С ТЕМ, НА КОТОРОМ ОБУЧЕНА БАЗОВАЯ ГОЛОВА И СНЯТ
    # КЭШ СОСТОЯНИЙ. Без этого пару npz+манифест можно было бы подменить
    # между K-14e и пробой, сохранив тот же q0, план и книги: голова читала
    # бы состояния от одной разметки, а целилась бы в другую.
    q1_man_sha = sha12(a.q1_cache + ".manifest.json")
    for who, d_ in (("базовая голова", ck), ("кэш h18", man)):
        for k_, v_ in (("q1_cache_sha1", q1_man["labels_sha1"]),
                       ("q1_manifest_sha1", q1_man_sha)):
            if d_.get(k_) is None:
                raise SystemExit(f"в {who} нет поля {k_}")
            if str(d_[k_]) != str(v_):
                raise SystemExit(
                    f"{who}: {k_} = {d_[k_]}, а поданный кэш целей даёт "
                    f"{v_}. Это другая разметка")
    print(f"  кэш целей сверен с базовой головой и кэшем h18: "
          f"{q1_man['labels_sha1']}, манифест {q1_man_sha}")
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

    def save_partial(res, sel, outcome, extra):
        """Сохранить измеренное, когда подтверждение открывать нельзя."""
        d = dict(kind="k14f_mlp_probe", outcome=outcome,
                 note="подтверждающая половина не открывалась",
                 results=[{k: v for k, v in r.items() if k != "state"}
                          for r in res],
                 selected=({k: v for k, v in sel.items() if k != "state"}
                           if sel else None),
                 history=hist_all, verify_linear_on_cache=got,
                 linear_train_slice=lin_tr, linear_val_sel=got,
                 h18_sha1=man["h18_sha1"], git_head=git_head,
                 script_sha1=script_sha_0, **extra)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        t_ = a.out + f".tmp.{os.getpid()}"
        json.dump(d, open(t_, "w"), ensure_ascii=False, indent=1, default=str)
        os.replace(t_, a.out)
        print(f"  сохранено (без подтверждения): {a.out}")
        if sel is not None and "state" in sel:
            t2 = ck_out + f".tmp.{os.getpid()}"
            sha_ = state_sha_np({k: v.detach().float().cpu().numpy()
                                 for k, v in sel["state"].items()})
            torch.save(dict(kind="k14f_head_unconfirmed", state=sel["state"],
                            hidden=sel["hidden"], seed=sel["seed"],
                            epoch=sel["epoch"], val_sel=sel["val_sel"],
                            selected_state_sha1=sha_,
                            n_params=sel.get("n_params"),
                            d_model=d_model, vocab=V,
                            norm_eps=float(man["norm_eps"]),
                            h18_sha1=man["h18_sha1"],
                            outcome=outcome, **extra), t2)
            os.replace(t2, ck_out)
            print(f"  веса сохранены как неподтверждённые: {ck_out}")

    # --- ОБУЧЕНИЕ ПРОБЫ -----------------------------------------------------
    hiddens = [int(x) for x in a.hidden.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    tr, vc = idx_of["train"], idx_of["val_confirm"]
    # СРЕЗ train РАВНОМЕРНЫЙ ПО ВСЕМУ НАБОРУ, а не первые N строк: строки
    # упорядочены по глобальному индексу, то есть по эпизодам, и первые
    # 5795 были бы другим набором задач, а не случайной частью.
    tr_slice = tr[np.linspace(0, len(tr) - 1, len(vs)).astype(int)]
    tr_slice = np.unique(tr_slice)
    print(f"  срез train для сравнения: {len(tr_slice)} строк из {len(tr)}")
    print(f"  цель обучения: {a.objective}"
          + ("" if a.objective == "ce_action" else
             "; CE считается, но на шаг НЕ влияет"))
    lin_tr = evaluate(lin, tr_slice)
    print(f"  ЛИНЕЙНАЯ ОПОРА на этом срезе: RMS-8 {lin_tr['rms']:.6f}, "
          f"top-1 {100 * lin_tr['top1']:.2f}%, CE {lin_tr['ce']:.5f}")
    print(f"  ЛИНЕЙНАЯ ОПОРА на val_sel:    RMS-8 {got['rms']:.6f}, "
          f"top-1 {100 * got['top1']:.2f}%")
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
                    # ЦЕЛЬ ВЫБИРАЕТСЯ ЯВНО. Аудит §48 показал, что при
                    # ce_action градиент от CE в 68-90 раз больше градиента
                    # от ошибки действия, а направления почти ортогональны:
                    # второе слагаемое вносит около полутора процентов нормы
                    # шага. Варианты ниже убирают CE, чтобы проверить, в ней
                    # ли дело.
                    if a.objective == "ce_action":
                        loss = ce + a.lambda_action * al
                    elif a.objective == "action_only":
                        loss = al
                    else:
                        loss = ((dd * wq) ** 2).mean()
                    if not torch.isfinite(loss):
                        raise SystemExit(f"потеря не число на эпохе {ep}")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if ep == 1 and nb == 0:
                        # ГРАДИЕНТ ОБЯЗАН ДОЙТИ ДО КАЖДОГО ПАРАМЕТРА И БЫТЬ
                        # КОНЕЧНЫМ. Отсутствующий означает, что часть головы
                        # в потере не участвует, и обучается не то, что
                        # заявлено.
                        nog = [n_ for n_, p2 in head.named_parameters()
                               if p2.grad is None]
                        nf = [n_ for n_, p2 in head.named_parameters()
                              if p2.grad is not None
                              and not torch.isfinite(p2.grad).all()]
                        if nog or nf:
                            raise SystemExit(f"градиенты: нет у {nog[:3]}, "
                                             f"нечисловые у {nf[:3]}")
                        print(f"    h={hid} сид {sd}: градиент есть у всех "
                              f"{len(list(head.parameters()))} тензоров")
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
            # ВЕСА ВОЗВРАЩАЮТСЯ К ВЫБРАННОЙ ЭПОХЕ ДО ЛЮБЫХ ИЗМЕРЕНИЙ.
            # После цикла в голове лежит последняя эпоха, а выбрана могла
            # быть ранняя: тогда «вырос ли train» относилось бы не к той
            # модели, которую выбрали.
            head.load_state_dict(best["state"])
            re_val = evaluate(head, vs)
            if abs(re_val["rms"] - best["val_sel"]) > 1e-9:
                raise SystemExit(
                    f"после восстановления эпохи {best['epoch']} val_sel "
                    f"{re_val['rms']:.9f} против сохранённого "
                    f"{best['val_sel']:.9f}")
            best["train_slice"] = evaluate(head, tr_slice)
            hist_all[f"h{hid}_s{sd}"] = hist
            results.append(best)

    pick = select_variant(results)
    print(f"\n  ВЫБРАНО по val_sel: h={pick['hidden']}, сид {pick['seed']}, "
          f"эпоха {pick['epoch']}, val_sel RMS {pick['val_sel']:.6f}, "
          f"top-1 {100 * pick['top1']:.2f}%")
    # ТРИ ЗАРЕГИСТРИРОВАННЫХ ИСХОДА РАЗЛИЧАЮТСЯ ТОЛЬКО ПАРОЙ СРАВНЕНИЙ.
    ts = pick["train_slice"]
    d_tr = lin_tr["top1"] - ts["top1"]
    d_va = got["top1"] - pick["top1"]
    print(f"  top-1: train {100 * lin_tr['top1']:.2f}% -> "
          f"{100 * ts['top1']:.2f}% ({-100 * d_tr:+.2f} п.п.), "
          f"val_sel {100 * got['top1']:.2f}% -> {100 * pick['top1']:.2f}% "
          f"({-100 * d_va:+.2f} п.п.)")
    print(f"  RMS-8: train {lin_tr['rms']:.6f} -> {ts['rms']:.6f}, "
          f"val_sel {got['rms']:.6f} -> {pick['val_sel']:.6f}")

    # ПЕРЕД ОТКРЫТИЕМ val_confirm КОД СВЕРЯЕТСЯ ЗАНОВО. Прогон идёт часы, и
    # правка файла посреди него осталась бы незамеченной.
    git_head1, code_v1, _d1 = code_state(False)
    # СИММЕТРИЧНО ПО КЛЮЧАМ: появившийся или исчезнувший файл — тоже
    # изменение, а односторонний перебор его бы не заметил.
    changed = sorted(k for k in set(code_v0) | set(code_v1)
                     if code_v0.get(k) != code_v1.get(k))
    if changed:
        # ОТКАЗ СТОИТ РОВНО ТОГО, ЧТО ЗАЩИЩАЕТ. Защищается одноразовая
        # подтверждающая половина, а не часы обучения: выбранные веса и все
        # измерения сохраняются, не открывается только она.
        print(f"  КОД ИЗМЕНИЛСЯ ВО ВРЕМЯ ПРОГОНА: файлы {changed}. "
              f"Подтверждающая половина НЕ открывается; результаты "
              f"сохраняются как неподтверждённые")
        save_partial(results, pick, "code_changed_during_run",
                     dict(git_head_start=git_head, git_head_end=git_head1,
                          code_version_start=code_v0,
                          code_version_end=code_v1, changed=changed,
                          script_sha1_start=script_sha_0,
                          script_sha1_end=sha12(os.path.abspath(__file__))))
        return 5
    if git_head1 != git_head:
        # СМЕНА КОММИТА БЕЗ СМЕНЫ ФАЙЛОВ — НЕ ПОВОД ОТКАЗЫВАТЬ. Проверка
        # существует ради того, чтобы результат принадлежал одной версии
        # вычисляющего кода; если ни один такой файл не тронут, он ей и
        # принадлежит. Прежняя версия роняла прогон из-за правки FINDINGS и
        # теряла сто пять минут счёта.
        print(f"  коммит сменился во время прогона ({git_head} -> "
              f"{git_head1}), но ни один отслеживаемый файл не изменился; "
              f"оба коммита записаны в сводку")

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

    # --- ВЕСА ВЫБРАННОЙ ГОЛОВЫ СОХРАНЯЮТСЯ -------------------------------
    # Иначе при хорошем результате от него остались бы только числа в json,
    # а самой головы — нет, и повторить её было бы нечем.
    sel_sha = state_sha_np({k: v.detach().float().cpu().numpy()
                            for k, v in head.state_dict().items()})
    ck_d = dict(kind="k14f_head", stage="q1", variant="probe_mlp",
                state={k: v.detach().cpu()
                       for k, v in head.state_dict().items()},
                hidden=pick["hidden"], seed=pick["seed"], epoch=pick["epoch"],
                n_params=pick["n_params"], selected_state_sha1=sel_sha,
                norm_eps=float(man["norm_eps"]), d_model=d_model, vocab=V,
                val_sel=pick["val_sel"], val_confirm=conf_mlp["rms"],
                confirm_linear=conf_lin["rms"],
                h18_sha1=man["h18_sha1"],
                h18_manifest_sha1=sha12(a.h18 + ".manifest.json"),
                q1_cache=a.q1_cache, q1_cache_sha1=q1_man["labels_sha1"],
                base_head_ckpt=man["head_ckpt"],
                base_head_state_sha1=man["head_state_sha1"],
                epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd,
                lambda_action=a.lambda_action, objective=a.objective,
                note="post-hoc; val_confirm повторно использована",
                git_head=git_head, script_sha1=sha12(os.path.abspath(__file__)))
    tmp_ck = ck_out + f".tmp.{os.getpid()}"
    torch.save(ck_d, tmp_ck)
    os.replace(tmp_ck, ck_out)
    print(f"  веса выбранной головы: {ck_out} (отпечаток {sel_sha})")

    out = dict(kind="k14f_mlp_probe", selected_head=ck_out,
               selected_state_sha1=sel_sha,
               note="post-hoc проверка на повторно "
               "использованной val_confirm, не независимое подтверждение",
               results=[{k: v for k, v in r.items() if k != "state"}
                        for r in results],
               selected={k: v for k, v in pick.items() if k != "state"},
               history=hist_all, verify_linear_on_cache=got,
               expect_rms=a.expect_rms, expect_top1=a.expect_top1,
               confirm_mlp=conf_mlp, confirm_linear=conf_lin,
               linear_train_slice=lin_tr, linear_val_sel=got,
               train_slice_rows=int(len(tr_slice)),
               h18_manifest_sha1=sha12(a.h18 + ".manifest.json"),
               h18_sha1=man["h18_sha1"], q1_cache=a.q1_cache,
               epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd,
               lambda_action=a.lambda_action, objective=a.objective,
               device=str(dev),
               gpu_uuid=kc.gpu_uuid(dev, torch), git_head=git_head,
               git_dirty=bool(dirty), code_version=code_v0,
               code_version_end=code_v1, git_head_end=git_head1,
               script_sha1_end=sha12(os.path.abspath(__file__)),
               script_sha1=script_sha_0,
               minutes=float((time.time() - t0) / 60))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"  сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
