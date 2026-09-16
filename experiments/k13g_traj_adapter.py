#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-13g: подключение ТРАЕКТОРНОЙ головы к раскатке K-12d.

ЗАЧЕМ АДАПТЕРОМ, А НЕ КОПИЕЙ РАСКАТКИ. В `k12d_rollout.run` около семисот
строк, и подавляющая их часть — протокол: реплики, печати решений, отпечатки
чекпойнта, сверка сидов, хэширование потока шума, сборка ячейки. Переписать это
заново ради другой формы поправки значит почти наверняка потерять проверку, и
цена такой потери выше, чем цена смены версии файла. Поэтому раскатка получает
ШОВ по виду головы, а вся специфика траекторной лежит здесь, с самопроверкой.

ЧЕМ ОТЛИЧАЕТСЯ ТРАЕКТОРНАЯ ГОЛОВА ОТ ПОЗИЦИОННОЙ НА ЭТОМ ШВЕ:

    действие          [batch, 64] против [batch, 16, 32] — одно решение на
                      чанк, а не по решению на позицию;
    веса чекпойнта    K-13b пишет state головы КАК ЕСТЬ, без префикса
                      `hicora_head.`, которым K-11c помечал свои;
    отпечаток базиса  K-13a пишет sha МАССИВА, K-11c — sha ФАЙЛА .npy. Байты
                      разные: заголовок .npy входит только в файловый;
    сросшийся проход  forward_hicora_t вместо forward_hicora, и голова живёт в
                      model.hicora_t_head после init_hicora_t;
    обучаемое         proj_h., proj_z., net. против proj., net.

ЧЕГО ЗДЕСЬ НЕТ. Никакой логики протокола, раскатки или записи ячеек: всё это
остаётся в K-12d и одинаково для обоих видов головы. Если что-то из этого
понадобится развести по видам — значит шов выбран неверно.
"""
import hashlib
import re

import numpy as np

KIND = "trajectory"
TRAIN_PREFIXES = ("proj_h.", "proj_z.", "net.")
# Буферы, которых нет в обучаемых весах: ставятся через set_basis/set_rho и
# сверяются по отпечаткам отдельно.
BUFFERS = ("basis", "rho", "basis_set", "rho_set", "log_std")


def array_sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def check_traj_ckpt(obj, arm_label, expect_target="coef"):
    """Происхождение чекпойнта траекторной головы.

    ОТДЕЛЬНО ОТ k9h.check_hicora_ckpt, а не вместо: та требует arch == "mlp",
    здесь arch == "trajectory_mlp", и подменять проверку общей означало бы
    ослабить её для обеих.
    """
    need = ("state", "arch", "target", "rank", "n_pos", "d_latent",
            "d_hidden", "proj", "hidden", "cache", "res_norm_sha1",
            "basis_sha1", "rho_sha1", "selected_epoch", "seed", "script_sha1")
    miss = [k for k in need if obj.get(k) is None]
    if miss:
        raise SystemExit(
            f"в чекпойнте траекторной головы нет полей {miss}: он собран "
            f"версией K-13b без записи происхождения, и что исполняется — "
            f"не доказуемо")
    if obj["arch"] != "trajectory_mlp":
        raise SystemExit(
            f"архитектура головы {obj['arch']!r}, ожидалась 'trajectory_mlp'. "
            f"Позиционная голова сюда не подходит: у неё другая форма "
            f"действия, и правдоподобие суммировалось бы по другому числу "
            f"координат")
    if obj["target"] != expect_target:
        raise SystemExit(f"мишень головы {obj['target']!r}, ожидалась "
                         f"{expect_target!r}")
    m = re.search(r"_s(\d+)$", str(arm_label))
    if m is None:
        raise SystemExit(f"метка руки {arm_label!r} не кончается на _s<сид>: "
                         f"сверить её с чекпойнтом нечем")
    if int(m.group(1)) != int(obj["seed"]):
        raise SystemExit(f"метка руки {arm_label!r}, а в чекпойнте сид "
                         f"{obj['seed']}")
    return True


def check_basis_provenance(obj, basis, rho):
    """Базис и rho против чекпойнта — ПО ОТПЕЧАТКУ МАССИВА.

    K-13a и K-13b пишут sha массива. Сверка по файловому отпечатку отвергла бы
    исправный чекпойнт, и это не гипотетическая опасность: ровно на этом
    споткнулась первая версия K-13d.
    """
    for nm, arr, want in (("базис", basis, obj["basis_sha1"]),
                          ("rho", rho, obj["rho_sha1"])):
        got = array_sha(arr)
        if got != want:
            raise SystemExit(f"{nm} на диске {got}, голова обучена на {want}")
    return True


def trainable_state(obj):
    """Обучаемые веса из чекпойнта, без буферов.

    K-13b сохраняет state целиком, включая basis и rho. Грузить их как веса
    нельзя: они ставятся через set_basis/set_rho, где проверяется
    ортонормированность и положительность предела.
    """
    st = {k: v for k, v in obj["state"].items() if k not in BUFFERS}
    stray = [k for k in st if not k.startswith(TRAIN_PREFIXES)]
    if stray:
        raise SystemExit(f"в чекпойнте веса вне {TRAIN_PREFIXES}: {stray[:5]}")
    return st


def build_heads(obj, basis, rho, d_hidden, d_latent, dev, torch):
    """Три головы: исходная, действующая и гауссова. Как в K-12d.

    ДВЕ ДЕТЕРМИНИРОВАННЫЕ, А НЕ ОДНА — по той же причине, что там: одна
    навсегда остаётся стартовой (мера того, насколько обучение ушло), вторая
    несёт действующие веса и участвует в сверке сросшегося прохода.
    """
    import hicora_t_g as htg
    import hicora_t_vla as ht

    if int(obj["d_hidden"]) != int(d_hidden):
        raise SystemExit(f"голова обучена на d_hidden {obj['d_hidden']}, а у "
                         f"модели {d_hidden}")
    if int(obj["d_latent"]) != int(d_latent):
        raise SystemExit(f"голова обучена на d_latent {obj['d_latent']}, а "
                         f"кодовая книга даёт {d_latent}")
    kw = dict(n_pos=int(obj["n_pos"]), rank=int(obj["rank"]),
              proj=int(obj["proj"]), hidden=int(obj["hidden"]))
    st = trainable_state(obj)
    heads = {}
    for which, cls in (("det_d1", ht.make_trajectory_head()),
                       ("det_cur", ht.make_trajectory_head()),
                       ("gau", htg.make_gaussian_trajectory_head())):
        h_ = cls(int(d_hidden), int(d_latent), **kw).to(dev, torch.float32)
        # БАЗИС ПЕРЕДАЁТСЯ НА CPU: set_basis проверяет ортонормированность и
        # переносит его на устройство головы сам.
        h_.set_basis(torch.as_tensor(np.asarray(basis)))
        h_.set_rho(torch.as_tensor(np.asarray(rho)))
        want = {k for k in h_.state_dict() if k.startswith(TRAIN_PREFIXES)}
        if set(st) != want:
            raise SystemExit(
                f"набор весов головы не совпал ({which}): нет в чекпойнте "
                f"{sorted(want - set(st))[:5]}, лишние "
                f"{sorted(set(st) - want)[:5]}")
        with torch.no_grad():
            for k, v in st.items():
                h_.state_dict()[k].copy_(v.to(dev, torch.float32))
        h_.eval()
        heads[which] = h_
    return heads


def bound_norm(dz, torch):
    """‖dz‖ ДЛЯ ТРАЕКТОРНОЙ ПОПРАВКИ: одна норма на чанк, а не на позицию.

    В K-12d предел меряется как `norm(dz, dim=-1).max()` — и это верно для
    позиционной HiCoRA, где rho ограничивает КАЖДУЮ позицию отдельно. У
    траекторной головы коэффициентов 64 на весь чанк, и предел ‖rho‖ относится
    к норме по всему чанку целиком. Применить к ней позиционную формулу значит
    сравнивать величину, меньшую в sqrt(n_pos) раз, с тем же порогом: проверка
    прошла бы всегда и ничего бы не гарантировала.
    """
    return float(torch.linalg.norm(dz.flatten(1), dim=-1).max())


def attach(model, head, obj, q0_depth):
    """Поставить голову в модель для СРОСШЕГОСЯ прохода forward_hicora_t."""
    model.init_hicora_t(q0_depth=int(q0_depth), rank=int(obj["rank"]),
                        proj=int(obj["proj"]), hidden=int(obj["hidden"]),
                        n_pos=int(obj["n_pos"]))
    model.hicora_t_head = head
    return model


def fused_forward(model, **kw):
    return model.forward_hicora_t(**kw)


def meta_fields(*, head_ckpt, head_sha, sigma_json, sigma_obj, k13d_path,
                basis_path, file_sha):
    """Поля провенанса траекторной ветви для ячейки.

    ОДНОГО ЧИСЛА --sigma НЕДОСТАТОЧНО. Оно не говорит, какой мерой подобрано,
    на каком горизонте, от какой головы и какой версией калибратора. Обе
    калибровки дали 0.08817 — по этому числу ячейки s0 и s1 были бы
    неразличимы, и подмена головы не обнаружилась бы.
    """
    need = ("sigma_t", "horizon", "target_rms", "rms_t", "head_t_sha1",
            "script_sha1", "head_seed")
    miss = [k for k in need if sigma_obj.get(k) is None]
    if miss:
        raise SystemExit(f"в артефакте калибровки нет полей {miss}")
    if str(sigma_obj["head_t_sha1"]) != str(head_sha):
        raise SystemExit(
            f"калибровка снята с головы {sigma_obj['head_t_sha1']}, а "
            f"исполняется {head_sha}: sigma относится к другой политике")
    return dict(
        head_kind=KIND,
        head_ckpt=str(head_ckpt), head_sha1=str(head_sha),
        traj_basis=str(basis_path),
        sigma_json=str(sigma_json), sigma_json_sha1=file_sha(sigma_json),
        sigma_t=float(sigma_obj["sigma_t"]),
        calibration_horizon=int(sigma_obj["horizon"]),
        calibration_target_rms=float(sigma_obj["target_rms"]),
        calibration_rms=float(sigma_obj["rms_t"]),
        calibration_head_sha1=str(sigma_obj["head_t_sha1"]),
        k13d_script_sha1=str(sigma_obj["script_sha1"]),
        k13d_path_sha1=file_sha(k13d_path),
        head_seed=int(sigma_obj["head_seed"]))


def selftest():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch

    D_H, D_L, NP_, RK = 24, 32, 4, 6
    q, _ = torch.linalg.qr(torch.randn(NP_ * D_L, RK, dtype=torch.float64))
    B = q.T.to(torch.float32).contiguous().numpy()
    rho = np.full(RK, 0.5, dtype=np.float32)

    import hicora_t_vla as ht
    ref = ht.make_trajectory_head()(D_H, D_L, n_pos=NP_, rank=RK, proj=8,
                                    hidden=16)
    for p_ in ref.net.parameters():
        torch.nn.init.normal_(p_, 0.0, 0.2)
    obj = dict(state={k: v.clone() for k, v in ref.state_dict().items()},
               arch="trajectory_mlp", target="coef", rank=RK, n_pos=NP_,
               d_latent=D_L, d_hidden=D_H, proj=8, hidden=16,
               cache="data/x", res_norm_sha1="aa", basis_sha1=array_sha(B),
               rho_sha1=array_sha(rho), selected_epoch=3, seed=0,
               script_sha1="bb")

    check_traj_ckpt(obj, "hicora_t_s0")
    check_basis_provenance(obj, B, rho)

    # --- позиционный чекпойнт сюда не проходит ---------------------------
    try:
        check_traj_ckpt(dict(obj, arch="mlp"), "hicora_t_s0")
    except SystemExit as e:
        assert "trajectory_mlp" in str(e), e
    else:
        raise AssertionError("позиционная голова принята траекторным швом")

    # --- метка руки против сида ------------------------------------------
    try:
        check_traj_ckpt(obj, "hicora_t_s1")
    except SystemExit as e:
        assert "сид" in str(e), e
    else:
        raise AssertionError("метка руки не сверена с сидом")

    # --- ФАЙЛОВЫЙ отпечаток вместо массивного обязан быть отвергнут -------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "b.npy")
        np.save(fp, B)
        f_sha = hashlib.sha1(open(fp, "rb").read()).hexdigest()[:12]
        assert f_sha != array_sha(B)
        try:
            check_basis_provenance(dict(obj, basis_sha1=f_sha), B, rho)
        except SystemExit as e:
            assert "на диске" in str(e), e
        else:
            raise AssertionError("отпечаток другого вида принят")

    # --- буферы не грузятся как веса --------------------------------------
    st = trainable_state(obj)
    assert all(k.startswith(TRAIN_PREFIXES) for k in st), sorted(st)
    assert not (set(st) & set(BUFFERS))
    try:
        trainable_state(dict(obj, state=dict(obj["state"], **{"чужой": 1})))
    except SystemExit as e:
        assert "вне" in str(e), e
    else:
        raise AssertionError("посторонний ключ пропущен")

    # --- три головы строятся и дают ТО ЖЕ, что эталон ---------------------
    heads = build_heads(obj, B, rho, D_H, D_L, torch.device("cpu"), torch)
    assert set(heads) == {"det_d1", "det_cur", "gau"}
    ref.set_basis(torch.as_tensor(B))
    ref.set_rho(torch.as_tensor(rho))
    ref.eval()
    h = torch.randn(5, NP_, D_H)
    z = torch.randn(5, NP_, D_L)
    with torch.no_grad():
        dz_ref, c_ref = ref(h, z)
        dz_got, c_got = heads["det_cur"](h, z)
        og = heads["gau"](h, z, deterministic=True)
    assert torch.allclose(dz_ref, dz_got, atol=1e-6), "det_cur не та голова"
    assert torch.allclose(dz_ref, og["dz"], atol=1e-6), \
        "среднее гауссовой головы не совпало с детерминированной"
    assert tuple(og["u"].shape) == (5, RK), tuple(og["u"].shape)

    # --- ГЛОБАЛЬНАЯ НОРМА ПРОТИВ ПОЗИЦИОННОЙ ------------------------------
    # Позиционная формула даёт величину меньше в sqrt(n_pos) раз и прошла бы
    # предел всегда. Проверяется, что это действительно разные числа и что
    # ограничение считается по глобальной.
    with torch.no_grad():
        dz_big, _ = heads["det_d1"](h, z)
        dz_big = dz_big + 0.0
    per_pos = float(torch.linalg.norm(dz_big, dim=-1).max())
    glob = bound_norm(dz_big, torch)
    assert glob >= per_pos, (glob, per_pos)
    ones = torch.ones(2, NP_, D_L)
    assert abs(bound_norm(ones, torch)
               - float(np.sqrt(NP_ * D_L))) < 1e-4
    assert abs(float(torch.linalg.norm(ones, dim=-1).max())
               - float(np.sqrt(D_L))) < 1e-4

    # --- несовпадение размерности модели ловится --------------------------
    try:
        build_heads(obj, B, rho, D_H + 1, D_L, torch.device("cpu"), torch)
    except SystemExit as e:
        assert "d_hidden" in str(e), e
    else:
        raise AssertionError("другая ширина ствола принята")
    # --- ПОЛЯ ПРОВЕНАНСА: одного числа sigma недостаточно ----------------
    sobj = dict(sigma_t=0.08817, horizon=8, target_rms=0.0424, rms_t=0.0427,
                head_t_sha1="hhhhhhhhhhhh", script_sha1="dddddddddddd",
                head_seed=0)
    m = meta_fields(head_ckpt="h.pt", head_sha="hhhhhhhhhhhh",
                    sigma_json="s.json", sigma_obj=sobj, k13d_path="k.py",
                    basis_path="b", file_sha=lambda _p: "ffffffffffff")
    assert m["head_kind"] == KIND and m["calibration_horizon"] == 8
    assert m["k13d_script_sha1"] == "dddddddddddd"
    # голова, с которой снята калибровка, обязана совпасть с исполняемой
    try:
        meta_fields(head_ckpt="h.pt", head_sha="ДРУГАЯ",
                    sigma_json="s.json", sigma_obj=sobj, k13d_path="k.py",
                    basis_path="b", file_sha=lambda _p: "f")
    except SystemExit as e:
        assert "другой политике" in str(e), e
    else:
        raise AssertionError("sigma от чужой головы принята")
    for k_ in ("horizon", "head_t_sha1"):
        try:
            meta_fields(head_ckpt="h.pt", head_sha="hhhhhhhhhhhh",
                        sigma_json="s.json",
                        sigma_obj={kk: vv for kk, vv in sobj.items()
                                   if kk != k_},
                        k13d_path="k.py", basis_path="b",
                        file_sha=lambda _p: "f")
        except SystemExit as e:
            assert "нет полей" in str(e) or "другой политике" in str(e), e
        else:
            raise AssertionError(f"артефакт без {k_} принят")

    print("самопроверка k13g_traj_adapter пройдена")


if __name__ == "__main__":
    selftest()
