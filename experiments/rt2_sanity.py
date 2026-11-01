#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, сверка двух путей предсказания baseline (без симулятора и без
переобучения).

Строки: по PER = 16 фиксированных строк train и val каждой из 8 задач
(256), выбор — np.random.default_rng([SEED, задача, split]) без повторов.
Модель — best.pt того же обучения (по умолчанию s0 flat24, сверка SHA и
отпечатка весов через rt2_policy.load_spec).
Путь A (обучение): модель rt2_train.build (как при обучении) с весами
чекпойнта, eval, no_grad; вход — rt1_data.Data.make_batch([строка]) и
rt2_train.forward_loss (bf16 autocast) -> out["action"][0, :8].
Путь B (сервер): настоящий rt2_policy.Model (build_model, как в
PolicyServer); наблюдение собирается из той же строки HDF5: декодированные
кадры трёх камер (vision.<cam>.color, RGB uint8), state.<ключ> из
state/<ключ>s, instruction — та же инструкция, что выбирает загрузчик
обучения; update_obs -> get_action -> 8 словарей команд -> [8, 14].
Оба пути — batch 1, bfloat16 autocast, eval, no_grad, одна карта.
Сверка входов: склейка камер (uint8, побитово), нормированное состояние,
текст, тензоры процессора (input_ids, attention_mask, pixel_values, …) —
побитово; выходы — первые 8 команд по группам: левая рука (0–5), правая
рука (7–12), схваты (6, 13): среднее и максимум |A − B|.
Ошибка к эксперту на первых 8 шагах (действительные позиции чанка), по тем
же группам: модель (путь B) и «удерживать текущую позу» — S[t] на всех
8 шагах (будущие состояния не используются). Рука — радианы, схват — в
единицах action/…_ee_joint_states.
Критерий согласованности (задан до прогона): входы побитово равны,
max |A − B| по всем выходам ≤ TOL_OUT. Отчёт reports/rt2/sanity_s<S>_<arch>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt1_common as rc  # noqa: E402

PER = 16
SEED = 20261102
N8 = 8
TOL_OUT = 1e-4
GROUPS = dict(left_arm=list(range(0, 6)), right_arm=list(range(7, 13)),
              grip=list(rc.GRIP_IDX))
H5_KEYS = tuple((k + "s", d) for k, d in (("left_arm_joint_state", 6),
                                          ("left_ee_joint_state", 1),
                                          ("right_arm_joint_state", 6),
                                          ("right_ee_joint_state", 1)))


def pick_rows(task_ix, split_ix, per=PER, seed=SEED):
    """{(задача, split): строки} — фиксированный выбор без повторов."""
    out = {}
    for ti in sorted(set(task_ix.tolist())):
        for sp, code in (("train", 0), ("val", 1)):
            cand = np.where((task_ix == ti) & (split_ix == code))[0]
            if len(cand) < per:
                raise SystemExit(f"задача {ti} {sp}: строк {len(cand)} < {per}")
            g = np.random.default_rng([seed, int(ti), code])
            out[(int(ti), sp)] = np.sort(g.choice(cand, per, replace=False))
    return out


def obs_of(frames, state14, text):
    """Наблюдение в формате сервера XPolicyLab (кадры уже RGB)."""
    import rt2_policy as rp
    st, o = {}, 0
    for k, d in rp.STATE_KEYS:
        st[k] = np.asarray(state14[o:o + d], np.float64).copy()
        o += d
    return dict(vision={c: dict(color=f) for c, f in zip(rc.CAMS, frames)},
                state=st, instruction=text)


def unpack(acts):
    """8 словарей команд -> [8, 14] в порядке каналов обучения."""
    import rt2_policy as rp
    return np.stack([np.concatenate([np.asarray(a[k], np.float64).reshape(-1)
                                     for k, _d in rp.STATE_KEYS])
                     for a in acts])


def group_err(pred, tgt, valid):
    """Средняя |pred − tgt| по действительным шагам, по группам каналов."""
    v = np.asarray(valid, bool)
    if not v.any():
        return {g: None for g in GROUPS}
    d = np.abs(np.asarray(pred, np.float64) - np.asarray(tgt, np.float64))[v]
    return {g: float(d[:, ix].mean()) for g, ix in GROUPS.items()}


def batch_diff(a, b):
    """Тензоры процессора: побитово равны? -> (равны, расхождения)."""
    import torch
    bad = []
    if set(a) != set(b):
        return False, [f"ключи {sorted(a)} ≠ {sorted(b)}"]
    for k in sorted(a):
        x, y = a[k], b[k]
        if not torch.is_tensor(x):
            if x != y:
                bad.append(k)
            continue
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            bad.append(f"{k} {tuple(x.shape)}/{tuple(y.shape)} {x.dtype}/"
                       f"{y.dtype}")
    return not bad, bad


def run(a):
    import torch
    import k21_train as ktr
    import rt1_data as rd
    import rt1_model as rm
    import rt2_policy as rp
    import rt2_train as r2
    dev = torch.device(a.device)
    D = rd.Data(check=True, check_files=False)
    ck = os.path.join(r2.outdir(a.seed, a.arch), "best.pt")
    rep = r2.report_path(a.seed, a.arch)
    spec = rp.load_spec(torch, ck, rep, D.prep["prep_id"])
    rows = pick_rows(D.ix["task"], D.ix["split"])
    # запись входов процессора и предсказателя
    rec = dict(batch=[], pred=[])
    orig_pb = rc.prompt_batch

    def pb(*x, **k):
        b = orig_pb(*x, **k)
        rec["batch"].append({kk: (v.detach().clone() if torch.is_tensor(v)
                                  else v) for kk, v in b.items()})
        return b
    rc.prompt_batch = pb
    # путь B: настоящий Model (как в PolicyServer)
    model = rp.Model(dict(ckpt=ck, train_report=rep, device=a.device,
                          n_exec=N8, _data=D))
    pred_b = model._predict

    def pred_rec(img, s_n, text):
        rec["pred"].append((np.array(img, copy=True),
                            np.array(s_n, copy=True), text))
        return pred_b(img, s_n, text)
    model._predict = pred_rec
    # путь A: модель обучения с весами чекпойнта
    ns = argparse.Namespace(factory=spec["contract"]["factory"],
                            seed=int(spec["contract"]["seed"]))
    modA, barA, _h = r2.build(torch, ns, D, dev)
    own = dict(modA.named_parameters())
    own.update(dict(modA.named_buffers()))
    with torch.no_grad():
        for k, v in spec["ck"]["weights"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    if ktr.state_hash(ktr.trainable_state(modA)) != spec["ck"]["weights_hash"]:
        raise SystemExit("путь A: веса после загрузки ≠ отпечатку")
    modA.eval()
    for p in modA.parameters():
        p.requires_grad_(False)
    names = list(rc.TASKS)
    per_row, prob = [], []
    t0 = time.time()
    for (ti, sp), rr in sorted(rows.items()):
        for row in rr:
            row = int(row)
            fid, t = int(D.ix["ep"][row]), int(D.ix["t"][row])
            img, s_n, text, tgt, valid = D.sample(row)
            st14 = D._ch[fid][0][t]
            h = D._file(fid)
            frames = [D.decode(h[f"vision/{c}/colors"][t]) for c in rc.CAMS]
            # путь A
            rec["batch"].clear()
            with torch.no_grad():
                out, _l, _la, _lg, tgtA, valA = r2.forward_loss(
                    torch, rm, modA, D, barA, [row], dev)
            bA = rec["batch"][-1]
            outA = out["action"][0, :N8].float().cpu().numpy() \
                .astype(np.float64)
            # путь B
            rec["batch"].clear()
            rec["pred"].clear()
            model.update_obs(obs_of(frames, st14, text))
            outB = unpack(model.get_action())
            bB = rec["batch"][-1]
            imgB, snB, textB = rec["pred"][-1]
            same_b, bad_b = batch_diff(bA, bB)
            r = dict(row=row, task=names[ti], split=sp, fid=fid, t=t,
                     img_equal=bool(np.array_equal(img, imgB)),
                     state_n_max=float(np.abs(np.asarray(s_n) -
                                              np.asarray(snB)).max()),
                     text_equal=text == textB, batch_equal=same_b,
                     batch_bad=bad_b,
                     tgt_equal=bool(np.array_equal(
                         tgtA[0].float().cpu().numpy(), tgt)),
                     out_diff={g: [float(np.abs(outA[:, ix] -
                                                outB[:, ix]).mean()),
                                   float(np.abs(outA[:, ix] -
                                                outB[:, ix]).max())]
                               for g, ix in GROUPS.items()},
                     out_exact=bool(np.array_equal(outA, outB)),
                     err_model=group_err(outB, tgt[:N8], valid[:N8]),
                     err_model_A=group_err(outA, tgt[:N8], valid[:N8]),
                     err_hold=group_err(np.repeat(st14[None], N8, 0),
                                        tgt[:N8], valid[:N8]),
                     n_valid8=int(np.asarray(valid[:N8]).sum()))
            if not (r["img_equal"] and r["state_n_max"] == 0 and
                    r["text_equal"] and r["batch_equal"] and r["tgt_equal"]):
                prob.append(f"{r['task']} {sp} строка {row}: входы различаются "
                            f"(кадры {r['img_equal']}, состояние "
                            f"{r['state_n_max']:.2e}, текст {r['text_equal']}, "
                            f"процессор {bad_b[:3]}, цель {r['tgt_equal']})")
            mx = max(v[1] for v in r["out_diff"].values())
            if not mx <= TOL_OUT:
                prob.append(f"{r['task']} {sp} строка {row}: выходы A/B "
                            f"расходятся на {mx:.2e}")
            per_row.append(r)
        print(f"  {names[ti]:20} {sp:5}: {len(rr)} строк, "
              f"{time.time() - t0:.0f} с", flush=True)
    rc.prompt_batch = orig_pb
    summ = summarize(per_row)
    res = dict(model=spec["meta"], rows=len(per_row), per=PER, seed=SEED,
               tol_out=TOL_OUT, problems=prob, consistent=not prob,
               summary=summ, per_row=per_row,
               note="рука — радианы, схват — единицы action/…_ee_joint_states;"
                    " удержание — S[t] на всех 8 шагах; согласованность "
                    "путей не проверяет исполнение симулятором")
    os.makedirs("reports/rt2", exist_ok=True)
    out = f"reports/rt2/sanity_s{a.seed}_{a.arch}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    report(summ, prob)
    print(f"  отчёт: {out}")
    return 0 if not prob else 3


def summarize(per_row):
    out = {}
    keys = sorted({(r["task"], r["split"]) for r in per_row}) + \
        [("ВСЕ", "train"), ("ВСЕ", "val")]
    for task, sp in keys:
        rs = [r for r in per_row if r["split"] == sp and
              (task == "ВСЕ" or r["task"] == task)]
        o = dict(n=len(rs), exact=sum(r["out_exact"] for r in rs),
                 ab_max={g: max(r["out_diff"][g][1] for r in rs)
                         for g in GROUPS},
                 ab_mean={g: float(np.mean([r["out_diff"][g][0] for r in rs]))
                          for g in GROUPS})
        for nm in ("err_model", "err_hold"):
            o[nm] = {g: float(np.mean([r[nm][g] for r in rs
                                       if r[nm][g] is not None]))
                     for g in GROUPS}
        out[f"{task}|{sp}"] = o
    return out


def report(summ, prob):
    print("\n  задача               split  A=B побитово  max|A−B| (лев/прав/схв)"
          "      модель→эксперт (лев/прав/схв)   удержание→эксперт (лев/прав/схв)")
    for k, o in summ.items():
        task, sp = k.split("|")
        f = lambda d: "/".join(f"{d[g]:.4f}" for g in GROUPS)  # noqa: E731
        fe = lambda d: "/".join(f"{d[g]:.1e}" for g in GROUPS)  # noqa: E731
        print(f"  {task:20} {sp:5}  {o['exact']:3}/{o['n']:<3}       "
              f"{fe(o['ab_max']):26} {f(o['err_model']):30} "
              f"{f(o['err_hold'])}")
    print("  " + ("СОГЛАСОВАНО: входы побитово равны, выходы A/B в пределах "
                  f"{TOL_OUT:g}" if not prob else
                  f"РАСХОЖДЕНИЯ ({len(prob)}): " + "; ".join(prob[:5])))


def selftest():
    import rt2_policy as rp
    # наблюдение -> вектор состояния политики = каналы обучения
    st = np.arange(14, dtype=np.float64) * 0.1
    fr = [np.full((4, 5, 3), i, np.uint8) for i in range(3)]
    ob = obs_of(fr, st, "x")
    assert np.array_equal(rp.state_vector(ob["state"]), st)
    assert [k for k, _ in H5_KEYS] == [k for k, _ in rc.CH_KEYS]
    assert [np.asarray(f).tolist() for f in rp.frames_of(ob["vision"])] == \
        [f.tolist() for f in fr]
    # pack -> unpack = те же 8 строк
    ch = np.random.default_rng(0).normal(size=(rc.T_CHUNK, rc.N_CH))
    assert np.allclose(unpack(rp.pack(ch, N8)), ch[:N8].astype(np.float32))
    # удержание и группы
    tgt = np.zeros((N8, 14))
    tgt[:, 0] = 1.0
    e = group_err(np.zeros((N8, 14)), tgt, np.ones(N8, bool))
    assert abs(e["left_arm"] - 1 / 6) < 1e-12 and e["right_arm"] == 0 and \
        e["grip"] == 0
    v = np.zeros(N8, bool)
    v[:2] = True
    tgt[2:, 7] = 5.0
    assert group_err(np.zeros((N8, 14)), tgt, v)["right_arm"] == 0
    rows = pick_rows(np.repeat(np.arange(2), 40), np.tile(
        np.r_[np.zeros(20), np.ones(20)], 2).astype(int), per=4, seed=1)
    assert len(rows) == 4 and all(len(r) == 4 for r in rows.values())
    assert np.array_equal(rows[(0, "train")], pick_rows(
        np.repeat(np.arange(2), 40), np.tile(np.r_[np.zeros(20), np.ones(20)],
                                             2).astype(int), per=4,
        seed=1)[(0, "train")])
    print("самопроверка rt2_sanity пройдена: наблюдение -> состояние политики "
          "= каналы обучения, кадры, pack/unpack, ошибки по группам и "
          "маска, фиксированный выбор строк")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description="RT: сверка путей предсказания")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="flat24")
    a = ap.parse_args()
    try:
        return run(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(f"  ОТКАЗ: {e.code}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
