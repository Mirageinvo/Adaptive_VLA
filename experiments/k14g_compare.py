#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14g: попарное сравнение читающих голов на кэше h18, с интервалами.

ЗАЧЕМ. K-14f сравнивает точечные оценки, а различия между бюджетами и
архитектурами оказались порядка третьего знака: выигрыш MLP на val_sel равен
0.0022 при межрепликовом разбросе 0.0019. Решать по таким числам без меры
неопределённости нельзя, а непересечение двух ОТДЕЛЬНЫХ интервалов проверкой
не является: ошибки голов сильно коррелированы, потому что считаются на одних
строках. Нужен интервал для РАЗНОСТИ, считаемой внутри одной реплики.

ДВА НАБОРА С РАЗНЫМИ РОЛЯМИ.

    val_sel      на нём выбирались эпоха, ширина и сид, поэтому интервал
                 выбранного победителя оптимистичен. Годится для инженерного
                 решения «какую ветку продолжать», помечается post-selection
                 и сильным статистическим утверждением не является.

    val_confirm  проверка переноса НАПРАВЛЕНИЯ, строго post-hoc: половина
                 уже открывалась и участвовала в появлении этих гипотез.
                 Совпали направления — согласованное разведочное
                 свидетельство; разошлись — результат неустойчив, и удобную
                 половину выбирать нельзя.

    Формально подтверждённого результата здесь нет ни при каком исходе:
    нетронутого набора в цепочке не осталось.

ДВА НЕЗАВИСИМЫХ ВЕРДИКТА на каждое сравнение. Направление — где интервал
лежит относительно нуля. Практическая величина — помещается ли он целиком в
зону эквивалентности. Они НЕ исключают друг друга: [0.0003, 0.0010] означает
одновременно «статистически лучше» и «практически эквивалентно», и это не
противоречие, а два разных вопроса.

    python experiments/k14g_compare.py --device cuda:1 \
        --heads reports/k14f/mlp_probe.pt reports/k14f/linear_b256e20.pt
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

EQ = 0.0019          # зона практической эквивалентности, см. §47.1


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def arr_sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def verdict(ci, eq=EQ):
    """Два НЕЗАВИСИМЫХ вердикта по интервалу разности.

    Направление отвечает на «различаются ли», практическая величина — на
    «имеет ли различие значение». Они не исключают друг друга: интервал
    [0.0003, 0.0010] означает одновременно «лучше» и «практически
    эквивалентно», и это два ответа на разные вопросы.

    У ПРАКТИЧЕСКОЙ ВЕЛИЧИНЫ ТРИ СОСТОЯНИЯ, А НЕ ДВА. Прежняя версия называла
    «значимым» всё, что не помещалось в зону целиком, — в том числе интервал
    [-0.004, +0.009], который не доказывает ни эквивалентности, ни значимого
    различия. Неопределённость обязана иметь своё имя.
    """
    lo, hi = float(ci[0]), float(ci[1])
    if lo > 0:
        d = "лучше"
    elif hi < 0:
        d = "хуже"
    else:
        d = "неопределённо"
    if lo >= -eq and hi <= eq:
        e = "практически эквивалентно"
    elif lo > eq or hi < -eq:
        e = "практически различаются"
    else:
        e = "величина неопределённа"
    return d, e


def check_alignment(rows_h, part_h, rows_t, part_t):
    """Кэш состояний и кэш целей описывают ОДНИ И ТЕ ЖЕ строки.

    Прежде в основном пути стояло `match_rows(rows, rows)` — массив,
    сравнённый сам с собой, то есть проверка, которая не может не пройти.
    Отрицательный тест при этом существовал и создавал видимость покрытия.

    Порядок здесь НЕ требуется одинаковым: цели сопоставляются по номеру
    строки. Требуется совпадение СОСТАВА, отсутствие дублей и совпадение
    части у каждой строки — иначе голова читала бы состояние строки из
    одной части, а целилась бы в метку из другой.
    """
    rh, rt = np.asarray(rows_h, np.int64), np.asarray(rows_t, np.int64)
    if len(np.unique(rh)) != len(rh):
        raise SystemExit("номера строк кэша состояний повторяются")
    if len(np.unique(rt)) != len(rt):
        raise SystemExit("номера строк кэша целей повторяются")
    miss = np.setdiff1d(rh, rt)
    if len(miss):
        raise SystemExit(f"{len(miss)} строк кэша состояний нет в кэше "
                         f"целей: {list(miss[:5])}")
    pos = {int(r): i for i, r in enumerate(rt)}
    idx = np.array([pos[int(r)] for r in rh])
    bad = np.where(np.asarray(part_t)[idx] != np.asarray(part_h))[0]
    if len(bad):
        i0 = int(bad[0])
        raise SystemExit(
            f"у {len(bad)} строк часть в кэше состояний и в кэше целей "
            f"различается, например строка {int(rh[i0])}: "
            f"{part_h[i0]} против {np.asarray(part_t)[idx][i0]}")
    return idx


def match_rows(rows_cache, rows_need):
    """Позиции нужных строк в кэше. Порядок и состав обязаны совпасть точно.

    Перестановка строк даёт те же суммы по набору и совершенно другое
    сопоставление голов: каждая читала бы своё состояние, а сравнивались бы
    они как одно. Поэтому отказ, а не пересортировка.
    """
    rc = np.asarray(rows_cache, np.int64)
    rn = np.asarray(rows_need, np.int64)
    if rc.shape != rn.shape:
        raise SystemExit(f"строк в кэше {rc.shape}, требуется {rn.shape}")
    if not np.array_equal(rc, rn):
        same = np.array_equal(np.sort(rc), np.sort(rn))
        raise SystemExit(
            "номера строк не совпадают" + (" по порядку (состав тот же): "
                                           "пересортировка запрещена"
                                           if same else " по составу"))
    return np.arange(len(rc))


def load_heads(spec, torch, make_head, d_model, vocab, norm_eps, dev,
               base_state, h18_sha1, state_sha_np, allow_unconfirmed=False):
    """Головы для сравнения: историческая линейная плюс сохранённые пробы.

    КАЖДЫЙ ЧЕКПОЙНТ СВЕРЯЕТСЯ: вид, отпечаток весов после загрузки, кэш h18,
    на котором он обучен. Бутстрапить неверно сопоставленные данные можно
    совершенно корректно и получить совершенно неверный ответ.
    """
    out = []
    lin = make_head(torch, d_model, vocab, 0, norm_eps).to(dev)
    with torch.no_grad():
        lin.norm.weight.copy_(base_state["depth_rvq_norms.0.weight"].float())
        lin.net.weight.copy_(base_state["depth_rvq_heads.0.weight"].float())
        if "depth_rvq_heads.0.bias" in base_state:
            lin.net.bias.copy_(base_state["depth_rvq_heads.0.bias"].float())
        elif lin.net.bias is not None:
            lin.net.bias.zero_()
    out.append(dict(label="линейная (историческая)", head=lin, hidden=0,
                    source="из чекпойнта, вшитого в кэш h18", stored=None))

    for p in spec:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        kind_ = str(ck.get("kind"))
        if kind_ not in ("k14f_head", "k14f_head_unconfirmed"):
            raise SystemExit(f"{p} описывает {kind_}")
        if kind_ == "k14f_head_unconfirmed" and not allow_unconfirmed:
            raise SystemExit(
                f"{p} — голова, для которой подтверждающая половина НЕ "
                f"открывалась. Анализ по ней открыл бы её незаметно; нужен "
                f"явный --allow-unconfirmed")
        for k in ("state", "hidden", "selected_state_sha1", "h18_sha1"):
            if ck.get(k) is None:
                raise SystemExit(f"в {p} нет поля {k}")
        if str(ck["h18_sha1"]) != str(h18_sha1):
            raise SystemExit(f"{p} обучен на кэше {ck['h18_sha1']}, подан "
                             f"{h18_sha1}")
        h = make_head(torch, d_model, vocab, int(ck["hidden"]),
                      float(ck.get("norm_eps", norm_eps))).to(dev)
        h.load_state_dict(ck["state"])
        got = state_sha_np({k: v.detach().float().cpu().numpy()
                            for k, v in h.state_dict().items()})
        if got != str(ck["selected_state_sha1"]):
            raise SystemExit(f"{p}: после загрузки отпечаток {got}, в "
                             f"чекпойнте {ck['selected_state_sha1']}")
        lbl = os.path.basename(p)[:-3]
        out.append(dict(label=f"{lbl} (h={ck['hidden']}, сид {ck.get('seed')},"
                              f" эпоха {ck.get('epoch')})",
                        head=h, hidden=int(ck["hidden"]), source=p,
                        stored=dict(val_sel=ck.get("val_sel"),
                                    val_confirm=ck.get("val_confirm"))))
    return out


def selftest():
    here = os.path.dirname(os.path.abspath(__file__))
    sp = importlib.util.spec_from_file_location(
        "_k14c", os.path.join(here, "k14c_train_q1.py"))
    k14c = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(k14c)

    # --- ВЕРДИКТЫ НЕЗАВИСИМЫ, У ВЕЛИЧИНЫ ТРИ СОСТОЯНИЯ ---------------------
    assert verdict([0.0003, 0.0010]) == ("лучше", "практически эквивалентно")
    assert verdict([0.004, 0.009]) == ("лучше", "практически различаются")
    assert verdict([-0.009, -0.004]) == ("хуже", "практически различаются")
    assert verdict([-0.0005, 0.0005]) == ("неопределённо",
                                          "практически эквивалентно")
    # НЕ ДОКАЗЫВАЕТ НИ ТОГО, НИ ДРУГОГО — и обязан называться так
    assert verdict([-0.004, 0.009]) == ("неопределённо",
                                        "величина неопределённа")
    assert verdict([0.0, 0.002]) == ("неопределённо", "величина неопределённа")
    assert verdict([0.0019, 0.0019]) == ("лучше", "практически эквивалентно")
    assert verdict([0.00191, 0.004])[1] == "практически различаются"

    # --- СООТВЕТСТВИЕ СТРОК ДВУХ КЭШЕЙ ------------------------------------
    rh = np.array([5, 1, 9], np.int64)
    ph = np.array(["train", "val_sel", "train"])
    rt = np.array([1, 5, 9, 12], np.int64)
    pt = np.array(["val_sel", "train", "train", "val_confirm"])
    idx = check_alignment(rh, ph, rt, pt)
    assert list(idx) == [1, 0, 2], idx
    # лишняя строка в целях допустима, недостающая — нет
    try:
        check_alignment(np.array([5, 77], np.int64), np.array(["train"] * 2),
                        rt, pt)
    except SystemExit as e:
        assert "нет в кэше целей" in str(e), e
    else:
        raise AssertionError("принята строка, которой нет в целях")
    # часть обязана совпасть
    try:
        check_alignment(rh, np.array(["train", "train", "train"]), rt, pt)
    except SystemExit as e:
        assert "часть" in str(e), e
    else:
        raise AssertionError("принято расхождение частей")
    for bad_h, bad_t, why in ((np.array([5, 5], np.int64), rt, "состояний"),
                              (rh, np.array([1, 5, 9, 9], np.int64), "целей")):
        try:
            check_alignment(bad_h, np.array(["train"] * len(bad_h)), bad_t,
                            np.array(["train"] * len(bad_t)))
        except SystemExit as e:
            assert "повторяются" in str(e) and why in str(e), (why, e)
        else:
            raise AssertionError(f"приняты дубли в {why}")

    # --- СТРОКИ: ПЕРЕСТАНОВКА ОТВЕРГАЕТСЯ ---------------------------------
    r = np.array([3, 1, 4, 1, 5], np.int64)
    assert list(match_rows(r, r)) == [0, 1, 2, 3, 4]
    try:
        match_rows(r, r[::-1])
    except SystemExit as e:
        assert "по порядку" in str(e), e
    else:
        raise AssertionError("перестановка строк принята")
    try:
        match_rows(r, np.array([3, 1, 4, 1, 9], np.int64))
    except SystemExit as e:
        assert "по составу" in str(e), e
    else:
        raise AssertionError("другой состав строк принят")
    try:
        match_rows(r, r[:3])
    except SystemExit:
        pass
    else:
        raise AssertionError("другая длина принята")

    # --- ПАРНАЯ РАЗНОСТЬ НА СИНТЕТИКЕ --------------------------------------
    rng = np.random.default_rng(0)
    eps = np.repeat(np.arange(50), 6)
    n_el = 56
    base = rng.random(300) * 2 + 3
    same = base.copy()
    better = base * 0.97
    r0 = k14c.cluster_boot(dict(a0=base * 1.2, oracle=base * 0.5, x=base,
                                y=same), eps, n_el, n=300,
                           deltas=[("x", "y")])
    d0 = r0["d_rms:x-y"]
    assert abs(d0[0]) < 1e-12 and abs(d0[1]) < 1e-12, \
        f"одинаковые головы обязаны дать строго нулевую разность: {d0}"
    assert verdict(d0) == ("неопределённо", "практически эквивалентно")
    # ЗНАК. `y` заведомо лучше (меньшая ошибка). По конвенции main для пары
    # (первая=y, вторая=x) берётся d_rms:x-y = RMS(x) − RMS(y) > 0, и это
    # читается как «первая, то есть y, лучше».
    r1 = k14c.cluster_boot(dict(a0=base * 1.2, oracle=base * 0.5, x=base,
                                y=better), eps, n_el, n=300,
                           deltas=[("x", "y")])
    d1 = r1["d_rms:x-y"]
    assert d1[0] > 0, f"интервал {d1} обязан быть выше нуля"
    assert verdict(d1)[0] == "лучше", "y меньше по ошибке, значит лучше"
    # обратная пара обязана дать зеркальный вердикт
    r2 = k14c.cluster_boot(dict(a0=base * 1.2, oracle=base * 0.5, x=base,
                                y=better), eps, n_el, n=300,
                           deltas=[("y", "x")])
    assert verdict(r2["d_rms:y-x"])[0] == "хуже"
    assert "capture:x" in r1 and "capture:y" in r1
    print("самопроверка k14g_compare пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--h18", default="data/k14e/h18")
    ap.add_argument("--q1-cache", default="data/k14b/q1_canonical")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--oracle", default="reports/k14a/oracle_canonical_cuda1.json")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--heads", nargs="+", required=False, default=[],
                    help="чекпойнты голов из K-14f")
    ap.add_argument("--rms-tol", type=float, default=2e-5,
                    help="допуск воспроизведения сохранённого RMS")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="разрешить анализ при незакоммиченном коде")
    ap.add_argument("--allow-unconfirmed", action="store_true",
                    help="принимать головы k14f_head_unconfirmed — те, для "
                         "которых подтверждающая половина НЕ открывалась. "
                         "Без флага такой анализ незаметно открыл бы её")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="reports/k14g/compare.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} уже существует")
    if not a.heads:
        raise SystemExit("нечего сравнивать: укажите --heads")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import k11a_build_hicora_cache as k11a
    import k14_common as kc
    import actioncodec  # noqa: F401
    from utils import ACTION_Q01, ACTION_Q99, VisionLanguageActionProcessor

    sp = importlib.util.spec_from_file_location(
        "_k14c", os.path.join(here, "k14c_train_q1.py"))
    k14c = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(k14c)
    sp2 = importlib.util.spec_from_file_location(
        "_k14f", os.path.join(here, "k14f_mlp_probe.py"))
    k14f = importlib.util.module_from_spec(sp2)
    sp2.loader.exec_module(k14f)

    H_EXEC = 8
    dev = torch.device(a.device)
    git_head, dirty, _ = kc.check_code_clean(a.allow_dirty)

    man = json.load(open(a.h18 + ".manifest.json"))
    q1_man = json.load(open(a.q1_cache + ".manifest.json"))
    k14f.check_manifest(man, q1_man=q1_man,
                        need_parts=("val_sel", "val_confirm"))
    if sha12(a.h18 + ".h18.npy") != man["h18_sha1"]:
        raise SystemExit("кэш h18 не совпал с отпечатком манифеста")
    if sha12(a.h18 + ".meta.npz") != man["meta_sha1"]:
        raise SystemExit("мета кэша не совпала с отпечатком манифеста")

    d_model = int(man["d_model"])
    Hm = np.load(a.h18 + ".h18.npy", mmap_mode="r")
    mt = np.load(a.h18 + ".meta.npz", allow_pickle=True)
    rows, part = np.asarray(mt["rows"], np.int64), mt["part"].astype(str)
    if arr_sha(rows) != man["rows_sha1"]:
        raise SystemExit("номера строк кэша не совпали с отпечатком")
    q0_c, ACT = np.asarray(mt["q0"], np.int64), np.asarray(mt["action"],
                                                           np.float32)
    epi_all = np.asarray(mt["episode"], np.int64)

    t_rows, t_q1, t_part, _ti = kc.load_q1_targets(a.q1_cache, q1_man)
    idx_t = check_alignment(rows, part, t_rows, t_part)
    TG = np.ascontiguousarray(t_q1[idx_t], np.int64)

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
    cs_now, dp_now = k11a.state_sha1(codec), k11a.decoder_probe(
        codec, books.float(), dev)
    if str(cs_now) != str(q1_man.get("codec_state_sha1")) \
            or str(dp_now) != str(q1_man.get("decoder_probe")):
        raise SystemExit("декодер не тот, на котором построены цели")
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    wq = torch.as_tensor(max_act_q[:7], device=dev,
                         dtype=torch.float32).clone()
    wq[-1] = 1.0

    base_ck = torch.load(man["head_ckpt"], map_location="cpu",
                         weights_only=False)
    heads = load_heads(a.heads, torch, k14f.make_head, d_model, V,
                       float(man["norm_eps"]), dev, base_ck["state"],
                       man["h18_sha1"], k14f.state_sha_np,
                       a.allow_unconfirmed)
    print(f"  голов к сравнению: {len(heads)}")
    for h in heads:
        print(f"    {h['label']}")

    def per_row(head, ii, mode="head", bs=512):
        """Суммы квадратов ошибки ПО СТРОКАМ. mode: head | a0 | oracle."""
        out = []
        with torch.no_grad():
            for i in range(0, len(ii), bs):
                j = ii[i:i + bs]
                e0 = books[0][torch.from_numpy(q0_c[j]).to(dev)]
                if mode == "a0":
                    z = e0
                elif mode == "oracle":
                    z = e0 + books[1][torch.from_numpy(TG[j]).to(dev)]
                else:
                    h_ = torch.from_numpy(np.asarray(Hm[j])).to(dev).float()
                    z = e0 + books[1][head(h_).argmax(-1)]
                ah = []
                for k in range(0, len(z), 512):
                    x, _ = codec._decode(z[k:k + 512].float(),
                                         embodiment_ids=0)
                    ah.append(x[..., :7].float())
                ah = torch.cat(ah)
                at = torch.from_numpy(ACT[j]).to(dev)[..., :7]
                dd = (ah[:, :H_EXEC] - at[:, :H_EXEC]) * wq
                out.append((dd ** 2).sum(dim=(1, 2)).cpu().numpy())
        return np.concatenate(out)

    n_el_row = int(H_EXEC * 7)
    res, table = {}, {}
    for part_name in ("val_sel", "val_confirm"):
        ii = np.where(part == part_name)[0]
        eps = epi_all[ii]
        pr = dict(a0=per_row(None, ii, "a0"),
                  oracle=per_row(None, ii, "oracle"))
        names = []
        for k, h in enumerate(heads):
            nm = f"h{k}"
            pr[nm] = per_row(h["head"], ii)
            names.append(nm)
        def rms(v):
            return float(np.sqrt(v.sum() / (len(ii) * n_el_row)))
        a0_r, or_r = rms(pr["a0"]), rms(pr["oracle"])
        print(f"\n  === {part_name} ({len(ii)} строк, "
              f"{len(np.unique(eps))} эпизодов) ===")
        print(f"    опора A0 {a0_r:.6f}, оракул {or_r:.6f}")
        rows_out = {}
        for nm, h in zip(names, heads):
            r_ = rms(pr[nm])
            c_ = (a0_r - r_) / (a0_r - or_r)
            rows_out[h["label"]] = dict(rms=r_, capture=c_)
            # СОХРАНЁННОЕ ЧИСЛО ВОСПРОИЗВОДИТСЯ. Иначе можно совершенно
            # корректно бутстрапить неверно сопоставленные данные.
            st = (h["stored"] or {}).get(part_name)
            mark = ""
            if st is not None:
                if abs(float(st) - r_) > a.rms_tol:
                    raise SystemExit(
                        f"{h['label']}: на {part_name} получено {r_:.6f}, в "
                        f"чекпойнте записано {float(st):.6f}")
                mark = "  (совпало с чекпойнтом)"
            print(f"    {h['label']:56s} RMS {r_:.6f}  C {c_:+.4f}{mark}")
        # ЗНАК: cluster_boot считает RMS(первого) − RMS(второго), а RMS —
        # это ОШИБКА. Чтобы «положительное» означало «первая голова лучше»,
        # в пару передаётся обратный порядок: RMS(второй) − RMS(первой).
        # Прежняя версия печатала подпись «положительная — первая лучше» над
        # величиной с противоположным смыслом.
        pairs = [(names[i], names[j]) for i in range(len(names))
                 for j in range(i + 1, len(names))]
        ci = k14c.cluster_boot(pr, eps, n_el_row, n=a.boot,
                               deltas=[(j_, i_) for i_, j_ in pairs])
        print("\n    попарно, 90% интервал разности RMS(вторая) − RMS(первая)"
              "; положительная означает, что ПЕРВАЯ лучше:")
        cmp_out = {}
        for i_, j_ in pairs:
            d = ci[f"d_rms:{j_}-{i_}"]
            dv, ev = verdict(d)
            la = heads[names.index(i_)]["label"]
            lb = heads[names.index(j_)]["label"]
            cmp_out[f"{la} vs {lb}"] = dict(ci90=d, direction=dv,
                                            equivalence=ev)
            print(f"      {la[:26]:26s} против {lb[:26]:26s} "
                  f"[{d[0]:+.6f}, {d[1]:+.6f}]  {dv}, {ev}")
        table[part_name] = dict(a0=a0_r, oracle=or_r, heads=rows_out,
                                pairs=cmp_out, n_rows=int(len(ii)),
                                n_episodes=int(len(np.unique(eps))))
    res = dict(kind="k14g_compare", table=table,
               role_val_sel="post-selection: на нём выбирались эпоха, ширина "
                            "и сид; сильным статистическим утверждением не "
                            "является",
               role_val_confirm="post-hoc проверка переноса направления; "
                                "половина уже открывалась",
               equivalence_zone=EQ, boot=a.boot,
               h18_sha1=man["h18_sha1"], q1_cache_sha1=q1_man["labels_sha1"],
               heads=[dict(label=h["label"], source=h["source"],
                           hidden=h["hidden"]) for h in heads],
               git_head=git_head, git_dirty=bool(dirty),
               script_sha1=sha12(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    json.dump(res, open(tmp, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, a.out)
    print(f"\n  сохранено: {a.out}")

    same = []
    for k in table["val_sel"]["pairs"]:
        d1 = table["val_sel"]["pairs"][k]["direction"]
        d2 = table["val_confirm"]["pairs"].get(k, {}).get("direction")
        if d1 != "неопределённо" and d2 != "неопределённо" and d1 != d2:
            same.append(k)
    if same:
        print(f"  ВНИМАНИЕ: направление разошлось между половинами у "
              f"{len(same)} пар: {same[:3]}. Результат неустойчив, и выбирать "
              f"удобную половину нельзя")
    else:
        print("  направления на val_sel и val_confirm нигде не противоречат "
              "друг другу")
    return 0


if __name__ == "__main__":
    sys.exit(main())
