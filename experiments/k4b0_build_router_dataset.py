"""K-4b0: датасет для строго причинного router.

ВОПРОС ФАЗЫ B. Способен ли router выбрать позиции для пересчёта, НЕ выполняя
того плотного прохода, который мы хотим сэкономить? Планка: лучший причинный
baseline даёт 0.40, оракулы — 0.79 (независимое ранжирование), 0.91 (жадный),
0.93 (точный).

ЕДИНИЦА ДАННЫХ. Наблюдение + позиция правки p + переход coarse-кода u -> v.
Для неё оцениваются все 16 позиций-кандидатов q.

ПРИЗНАКИ И МЕТКИ РАЗДЕЛЕНЫ ФИЗИЧЕСКИ, тремя файлами. В features.npz попадают
ТОЛЬКО величины, существующие до нового плотного прохода; список ключей
проверяется белым списком и падает при постороннем ключе. Оракульные величины
живут в labels.npz и в признаки не попадают даже косвенно.

ЗАПРЕЩЕНО В ПРИЗНАКАХ: lg_after, JS между распределениями до и после правки,
новые fine-логиты, fine-коды z_ref, h_ref, ||h_ref - h_stale||, датасетное
действие, декодированное действие после ремонта, changed-support, оракульный
лучший набор.

  ОСОБО: ||h_ref - h_stale|| считается ЧЕРЕЗ ЭТАЛОННЫЕ fine-коды и потому
  является утечкой. В K-4a4 эта величина использовалась на стороне МЕТОК, как
  независимый способ разбить пары на слои, и там она законна. В признаках
  допустим только причинный аналог coarse_delta_norm = ||E0[v] - E0[u]||.

ТОЧНОЕ СЖАТИЕ ПЕРЕБОРА. Пусть C — позиции, где stale и z_ref различаются. Для
q вне C латента совпадает ПОБИТОВО, поэтому замена ничего не меняет и

    G(S) = G(S ∩ C)   ТОЧНО, а не приближённо.

Значит достаточно перебрать подмножества C размера <= 4: при среднем |C| = 4.74
это 69 наборов вместо 2517, то есть в 36 раз меньше. Весь датасет на 1000
наблюдений обходится в ~1.1 млн декодирований против 40.3 млн полным перебором
— дешевле, чем K-4a4 на 96 наблюдениях. Поэтому точные таблицы считаются для
ВСЕХ split, без урезания validation и test.

  Сжатие относится только к BAR-прокси и не является предположением о будущем
  дискретном потоке. C вычисляется ТОЛЬКО внутри построителя меток и router'у
  не передаётся.

БЮДЖЕТ «НЕ БОЛЕЕ K». Из установленной немонотонности (K-4a4: нарушения в
29-31% случаев, 2.1% из них съедают весь доступный выигрыш) следует, что router
обязан иметь право остановиться раньше и отозвать выбранную позицию. Поэтому
сохраняются траектории ADD, ADD+STOP и ADD/REMOVE/STOP, а лучший набор — и
ровно K, и не более K.

Запуск:
    python3 experiments/k4b0_build_router_dataset.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \
        --n-obs 1000 --n-ep 400 --out data/k4b0
"""

import argparse
import io
import itertools
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# БЕЛЫЙ СПИСОК признаков. Всё, чего здесь нет, в features.npz не попадёт.
FEATURE_KEYS = {
    # уровень наблюдения
    "obs_pooled_ctx", "obs_task_idx", "obs_state",
    # уровень вмешательства (наблюдение + позиция правки p)
    "int_obs_idx", "int_p", "int_u", "int_v",
    "int_coarse_delta_norm", "int_coarse_cos", "int_logp_diff",
    "int_rank_u", "int_logp_u", "int_logp_v",
    # уровень кандидата q, форма (n_int, P, ...)
    "cand_entropy", "cand_margin", "cand_topk_p", "cand_old_tokens",
    "cand_q", "cand_dq", "cand_absdq", "cand_is_p",
    "cand_latent_norm", "cand_coarse_logp", "cand_coarse_entropy",
}
FORBIDDEN_SUBSTR = ("ref", "after", "js", "oracle", "gain", "label",
                    "changed", "support", "true", "target")


def load_lerobot_b0(n_obs: int, T: int, n_ep: int, seed: int):
    """Загрузка с ГАРАНТИЕЙ числа различных эпизодов.

    Прежний загрузчик брал ceil(n_obs/n_ep) наблюдений с эпизода и обрывался по
    достижении n_obs, поэтому при n_obs=1000, n_ep=400 доходил лишь до ~334
    эпизодов. Здесь эпизоды обходятся до выполнения ОБОИХ условий: набрано
    n_obs наблюдений И не менее n_ep различных эпизодов."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        d = json.loads(line)
        tasks_map[d["task_index"]] = d["task"]

    rng = np.random.default_rng(seed)
    order = rng.permutation(1693)
    per_ep = max(1, n_obs // max(n_ep, 1))
    im1, im2, st, act, tasks, epi = [], [], [], [], [], []

    def png(cell):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))

    n_uniq = 0
    for e in order:
        if len(tasks) >= n_obs and n_uniq >= n_ep:
            break
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        n = t.num_rows
        if n <= T:
            continue
        # пока эпизодов не хватает — берём с каждого поменьше, чтобы места
        # хватило на нужное их число
        k = per_ep
        if n_uniq < n_ep:
            k = min(per_ep, max(1, (n_obs - len(tasks)) // max(1, n_ep - n_uniq)))
        starts = rng.choice(n - T, size=min(k, n - T), replace=False)
        A_ = np.asarray(t.column("actions").to_pylist(), np.float32)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        tset = {ti[int(s)] for s in starts}
        assert len(tset) == 1, f"эпизод {e}: несколько задач {tset}"
        for s0 in starts:
            im1.append(png(c1[s0]))
            im2.append(png(c2[s0]))
            st.append(S_[s0])
            act.append(A_[s0:s0 + T])
            tasks.append(tasks_map[ti[int(s0)]])
            epi.append(int(e))
        n_uniq += 1
        if n_uniq % 50 == 0:
            print(f"  эпизодов {n_uniq}, наблюдений {len(tasks)}", flush=True)

    k = min(n_obs, len(tasks))
    epi = np.array(epi[:k])
    n_real = len(np.unique(epi))
    print(f"LeRobot v2.0: {n_real} различных эпизодов, {k} наблюдений")
    assert n_real >= n_ep, f"эпизодов {n_real} < требуемых {n_ep}"
    import torch
    to_t = (lambda a: torch.from_numpy(np.stack(a[:k])).permute(0, 3, 1, 2))
    return (to_t(im1), to_t(im2), np.stack(st[:k]), np.stack(act[:k]),
            tasks[:k], epi)


def split_by_episode(epi, tasks, fracs=(0.70, 0.15, 0.15), seed: int = 0):
    """Разбиение ПО ЭПИЗОДАМ со стратификацией ПО ЗАДАЧАМ.

    Каждая задача делится в тех же долях, поэтому редкие задачи не оказываются
    целиком в одном split. Все наблюдения эпизода и все его вмешательства
    попадают в одну часть по построению."""
    rng = np.random.default_rng(seed)
    ep_task = {}
    for e, t in zip(epi, tasks):
        ep_task.setdefault(int(e), t)
    out = {}
    for t in sorted(set(ep_task.values())):
        eps = np.array(sorted(e for e, tt in ep_task.items() if tt == t))
        rng.shuffle(eps)
        n = len(eps)
        n_tr = max(1, int(round(fracs[0] * n)))
        n_va = int(round(fracs[1] * n))
        if n_tr + n_va >= n:            # у редких задач тест важнее validation
            n_va = max(0, n - n_tr - 1)
        for e in eps[:n_tr]:
            out[int(e)] = 0
        for e in eps[n_tr:n_tr + n_va]:
            out[int(e)] = 1
        for e in eps[n_tr + n_va:]:
            out[int(e)] = 2
    return np.array([out[int(e)] for e in epi], np.int8)


def subsets_of(C, kmax: int = 4):
    """Канонический порядок подмножеств C размера 0..kmax."""
    C = tuple(sorted(C))
    return [S for k in range(kmax + 1) for S in itertools.combinations(C, k)]


def greedy_paths(gmap, C, tau: float, kmax: int = 4):
    """Три траектории из СЖАТОЙ таблицы, без единого вызова модели.

    ADD          — жадное добавление ровно kmax шагов;
    ADD+STOP     — остановка, когда предельный выигрыш не превышает tau;
    ADD/REM/STOP — на каждом шаге рассматриваются и удаления; отвечает
                   обратимой постановке и мотивирован немонотонностью."""
    def g(S):
        return gmap[tuple(sorted(S))]

    add, S, marg = [], (), []
    for _ in range(kmax):
        cand = [(g(S + (q,)) - g(S), q) for q in C if q not in S]
        if not cand:
            break
        d, q = max(cand)
        add.append(q)
        marg.append(d)
        S = tuple(sorted(S + (q,)))
    stop_k = 0
    for i, d in enumerate(marg):
        if d <= tau:
            break
        stop_k = i + 1

    S, rev = (), []
    while True:
        best = (tau, None, None)
        for q in C:
            if q in S:
                if len(S) > 0:
                    d = g(tuple(x for x in S if x != q)) - g(S)
                    if d > best[0]:
                        best = (d, "rem", q)
            elif len(S) < kmax:
                d = g(tuple(sorted(S + (q,)))) - g(S)
                if d > best[0]:
                    best = (d, "add", q)
        if best[1] is None:
            break
        rev.append((1 if best[1] == "add" else -1) * (best[2] + 1))
        S = (tuple(sorted(S + (best[2],))) if best[1] == "add"
             else tuple(x for x in S if x != best[2]))
        if len(rev) > 2 * kmax:          # защита от зацикливания
            break
    return add, marg, stop_k, rev, tuple(sorted(S))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="data/k4b0")
    ap.add_argument("--n-obs", type=int, default=1000)
    ap.add_argument("--n-ep", type=int, default=400)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--topk-probs", type=int, default=8)
    ap.add_argument("--tau-rel", type=float, default=1e-3,
                    help="порог значимости в долях g1; как в K-4a4")
    ap.add_argument("--gap-rel", type=float, default=1e-2,
                    help="G* ниже этой доли медианного G* -> ремонтировать нечего")
    ap.add_argument("--verify-full", type=int, default=8,
                    help="на скольких вмешательствах сверить сжатие с полным "
                         "перебором 2517 наборов")
    ap.add_argument("--rank-lo", type=int, default=1)
    ap.add_argument("--rank-hi", type=int, default=5)
    ap.add_argument("--pos-offset", type=int, default=3)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--tiled", action="store_true", default=True)
    ap.add_argument("--source", default="lerobot")
    ap.add_argument("--flip", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    from k1_residual_cost import latent_from_codes, projected_codebooks
    from k3_bar_suffix_repair import (MAX_ACTION_Q, STATE_Q01, STATE_Q99,
                                      build_batch)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"

    sys.path.insert(0, os.path.abspath(args.root))
    import copy
    import importlib.util

    import actioncodec  # noqa: F401

    sp = importlib.util.spec_from_file_location(
        "ac_vla_tok", os.path.join(os.path.abspath(args.root), "utils",
                                   "vla_tokenizer.py"))
    mm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mm)
    proc = mm.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tok = proc.action_processor.to(args.device).eval()
    L, P = tok.num_quantizers, tok.n_tokens_per_quantizer
    cfg = tok.config.embodiment_config[
        list(tok.config.embodiment_config.keys())[args.embodiment]]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]
    tok32 = copy.deepcopy(tok).float().eval()
    E = projected_codebooks(tok32, args.device)          # (L, V, D)

    IM1, IM2, ST_RAW, A_, TASKS, EPI = load_lerobot_b0(
        args.n_obs, T, args.n_ep, args.seed)
    N = len(TASKS)
    SPLIT = split_by_episode(EPI, TASKS, seed=args.seed)
    st_all = ((ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0
              - 1.0).astype(np.float32)
    A_ = np.asarray(A_, np.float32).copy()
    A_[..., :-1] = A_[..., :-1] / MAX_ACTION_Q[:-1]
    A_[..., -1] = -A_[..., -1]
    scale = float(np.clip(A_, -1, 1).max() - np.clip(A_, -1, 1).min())

    uniq_tasks = sorted(set(TASKS))
    task_idx = np.array([uniq_tasks.index(t) for t in TASKS], np.int32)
    for nm, s in (("train", 0), ("val", 1), ("test", 2)):
        m = SPLIT == s
        print(f"  {nm:>5}: наблюдений {m.sum():>5}, эпизодов "
              f"{len(np.unique(EPI[m])):>4}, задач {len(set(np.array(TASKS)[m]))}")

    from smolvla.bar import SmolVLABlockwiseAR

    model = SmolVLABlockwiseAR.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
        token_budget=P * L, num_blocks=L,
        action_vocab_size=tok.vocab_size).to(args.device).eval()
    bs, nb = model.block_size, model.num_blocks

    F = {k: [] for k in FEATURE_KEYS}
    Lab = {k: [] for k in
           ("sing_gain", "e_empty", "g_star", "support", "g_flat", "g_off",
            "best_exact_k", "best_le_k", "add_path", "add_marg", "stop_k",
            "rev_path", "rev_set", "tau", "obs_idx", "p")}
    g_off = [0]
    verify = []

    def run_batch(lo, hi):
        nonlocal verify
        B = hi - lo
        args_ns = args
        batch = build_batch(IM1[lo:hi], IM2[lo:hi], TASKS[lo:hi],
                            st_all[lo:hi], proc, args_ns, args.device)
        with torch.no_grad():
            _, vlen, VLM, _ = model._build_vlm_inputs_embeds(
                input_ids=batch["input_ids"], inputs_embeds=None,
                pixel_values=batch.get("pixel_values"),
                pixel_attention_mask=batch.get("pixel_attention_mask"),
                image_hidden_states=None)

            def blk(hist):
                alen = bs + (0 if hist is None else hist.shape[1])
                apos = model._build_action_pos_ids_strided(
                    batch_size=B, base_pos=vlen, action_seq_len=alen,
                    device=VLM.device, position_offset=args.pos_offset)
                pids = model._build_joint_position_ids(
                    batch_size=B, vlm_seq_len=vlen, action_pos_ids=apos,
                    device=VLM.device)
                return model._predict_next_block_logits(
                    vlm_inputs_embeds=VLM,
                    attention_mask=batch.get("attention_mask"),
                    history_tokens=hist, position_ids=pids).float()

            def dec(h):
                out = []
                for i in range(0, len(h), args.chunk):
                    out.append(tok32._decode(h[i:i + args.chunk],
                                             args.embodiment, None)[0][..., :D_act])
                return torch.cat(out)

            def sq(h, ref):
                d = (dec(h)[:, :args.window]
                     - ref[:, :args.window]).abs()[..., :D_act - 1]
                return d.flatten(1).pow(2).mean(-1) / scale ** 2

            # ПРИЗНАК уровня наблюдения: усреднённый контекст VLM
            F["obs_pooled_ctx"].append(VLM.float().mean(1).cpu().numpy())
            F["obs_task_idx"].append(task_idx[lo:hi])
            F["obs_state"].append(st_all[lo:hi])

            hist = None
            for _ in range(nb):
                hist = (blk(hist).argmax(-1) if hist is None
                        else torch.cat([hist, blk(hist).argmax(-1)], 1))
            z_ref = hist.reshape(-1, L, P).transpose(1, 2)
            a_ref = dec(latent_from_codes(E, z_ref))
            lg0 = blk(None)
            lp0 = lg0.log_softmax(-1)
            ar = torch.arange(B, device=args.device)
            rng = torch.Generator(device=args.device).manual_seed(
                1000 + args.seed + lo)

            for p_ in range(P):
                ranks = lg0[:, p_].topk(args.rank_hi, -1).indices
                rk = torch.randint(args.rank_lo, args.rank_hi, (B,),
                                   generator=rng, device=args.device)
                u = ranks[ar, rk]
                v = z_ref[:, p_, 0]
                c0_old = z_ref[:, :, 0].clone()
                c0_old[:, p_] = u

                # ---- ПРИЧИННЫЕ признаки: только этот, уже состоявшийся проход
                lg_before = blk(c0_old)
                pb = lg_before.softmax(-1)
                lpb = lg_before.log_softmax(-1)
                ent = -(pb * lpb).sum(-1)
                t2 = lg_before.topk(2, -1).values
                marg = t2[..., 0] - t2[..., 1]
                topk_p = pb.topk(args.topk_probs, -1).values

                c1_old = lg_before.argmax(-1)
                z_old = torch.stack(
                    [c0_old, c1_old,
                     blk(torch.cat([c0_old, c1_old], 1)).argmax(-1)], -1)
                stale = z_old.clone()
                stale[:, :, 0] = z_ref[:, :, 0]

                eu, ev = E[0][u], E[0][v]
                de = (ev - eu).float()
                F["int_obs_idx"].append(np.arange(lo, hi))
                F["int_p"].append(np.full(B, p_, np.int16))
                F["int_u"].append(u.cpu().numpy().astype(np.int32))
                F["int_v"].append(v.cpu().numpy().astype(np.int32))
                F["int_coarse_delta_norm"].append(de.norm(dim=-1).cpu().numpy())
                F["int_coarse_cos"].append(torch.nn.functional.cosine_similarity(
                    eu.float(), ev.float(), dim=-1).cpu().numpy())
                F["int_logp_u"].append(lp0[ar, p_, u].cpu().numpy())
                F["int_logp_v"].append(lp0[ar, p_, v].cpu().numpy())
                F["int_logp_diff"].append(
                    (lp0[ar, p_, v] - lp0[ar, p_, u]).cpu().numpy())
                F["int_rank_u"].append(rk.cpu().numpy().astype(np.int8))

                qs = torch.arange(P, device=args.device)
                F["cand_entropy"].append(ent.cpu().numpy())
                F["cand_margin"].append(marg.cpu().numpy())
                F["cand_topk_p"].append(topk_p.cpu().numpy())
                F["cand_old_tokens"].append(z_old.cpu().numpy().astype(np.int16))
                F["cand_q"].append(np.tile(np.arange(P, np.int16), (B, 1)))
                F["cand_dq"].append(np.tile((np.arange(P) - p_).astype(np.int16),
                                            (B, 1)))
                F["cand_absdq"].append(np.tile(
                    np.abs(np.arange(P) - p_).astype(np.int16), (B, 1)))
                F["cand_is_p"].append(np.tile(
                    (np.arange(P) == p_).astype(np.int8), (B, 1)))
                h_st = latent_from_codes(E, stale)
                F["cand_latent_norm"].append(
                    h_st.float().norm(dim=-1).cpu().numpy())
                F["cand_coarse_logp"].append(
                    lp0.gather(2, c0_old.unsqueeze(-1)).squeeze(-1).cpu().numpy())
                F["cand_coarse_entropy"].append(
                    (-(lg0.softmax(-1) * lg0.log_softmax(-1)).sum(-1)).cpu().numpy())

                # ---- МЕТКИ: здесь и только здесь появляется z_ref
                h_rf = latent_from_codes(E, z_ref)
                e0 = sq(h_st, a_ref)
                diff = (stale != z_ref).any(-1)          # changed support
                supp = (diff.int() * (1 << qs)).sum(-1)

                # ТОЧНОЕ сжатие: перебираем только подмножества C.
                # Варианты всех примеров батча собираются в ОДИН тензор и
                # декодируются пачками: иначе вышло бы ~16000 мелких вызовов,
                # где накладные расходы больше самой работы.
                subs_all = [subsets_of(torch.nonzero(diff[b]).flatten().tolist(),
                                       args.kmax) for b in range(B)]
                gg_all = [None] * B
                buf_h, buf_a, buf_e, owner = [], [], [], []

                def flush():
                    if not buf_h:
                        return
                    hh = torch.stack(buf_h)
                    aa = torch.stack(buf_a)
                    ee = sq(hh, aa)
                    gv = (torch.stack(buf_e) - ee).cpu().numpy()
                    for (b_, j_), g_ in zip(owner, gv):
                        gg_all[b_][j_] = g_
                    buf_h.clear(); buf_a.clear(); buf_e.clear(); owner.clear()

                for b in range(B):
                    gg_all[b] = np.zeros(len(subs_all[b]), np.float32)
                    for j, S in enumerate(subs_all[b]):
                        h = h_st[b].clone()
                        if S:
                            h[list(S)] = h_rf[b, list(S)]
                        buf_h.append(h)
                        buf_a.append(a_ref[b])
                        buf_e.append(e0[b])
                        owner.append((b, j))
                        if len(buf_h) >= args.chunk:
                            flush()
                flush()

                # ---- ПРОВЕРКА СЖАТИЯ: полный перебор против G(S ∩ C)
                if lo == 0 and p_ == 0 and args.verify_full:
                    full_sets = [S for k in range(args.kmax + 1)
                                 for S in itertools.combinations(range(P), k)]
                    for b in range(min(args.verify_full, B)):
                        gmap_b = {tuple(sorted(S)): float(gg_all[b][j])
                                  for j, S in enumerate(subs_all[b])}
                        Cb = set(torch.nonzero(diff[b]).flatten().tolist())
                        hh, ref = [], []
                        for S in full_sets:
                            h = h_st[b].clone()
                            if S:
                                h[list(S)] = h_rf[b, list(S)]
                            hh.append(h)
                            ref.append(a_ref[b])
                        gf = (e0[b] - sq(torch.stack(hh),
                                         torch.stack(ref))).cpu().numpy()
                        gc = np.array([gmap_b[tuple(sorted(set(S) & Cb))]
                                       for S in full_sets], np.float32)
                        verify.append(float(np.abs(gf - gc).max()))

                # ---- поэкземплярные метки из сжатой таблицы
                blk_lab = {k: [] for k in
                           ("sing_gain", "g_star", "tau", "best_exact_k",
                            "best_le_k", "add_path", "add_marg", "stop_k",
                            "rev_path", "rev_set")}
                for b in range(B):
                    C = torch.nonzero(diff[b]).flatten().tolist()
                    gmap = {tuple(sorted(S)): float(gg_all[b][j])
                            for j, S in enumerate(subs_all[b])}
                    Lab["g_flat"].append(gg_all[b])
                    g_off.append(g_off[-1] + len(gg_all[b]))
                    sing = np.array([gmap.get((q,), 0.0) for q in range(P)],
                                    np.float32)
                    blk_lab["sing_gain"].append(sing)
                    blk_lab["g_star"].append(np.float32(max(0.0, max(gmap.values()))))
                    tau = max(1e-8, args.tau_rel * max(float(sing.max()), 0.0))
                    blk_lab["tau"].append(np.float32(tau))
                    # «ровно K» при |C| < K недостижимо: позиции вне C ничего не
                    # меняют, поэтому берём ровно min(K, |C|)
                    kk = min(args.kmax, len(C))
                    ex = max((S for S in subs_all[b] if len(S) == kk),
                             key=lambda S: gmap[tuple(sorted(S))], default=())
                    le = max(subs_all[b], key=lambda S: gmap[tuple(sorted(S))])
                    blk_lab["best_exact_k"].append(_pad(ex, args.kmax))
                    blk_lab["best_le_k"].append(_pad(le, args.kmax))
                    add, mg, sk, rev, rset = greedy_paths(gmap, C, tau, args.kmax)
                    blk_lab["add_path"].append(_pad(add, args.kmax))
                    blk_lab["add_marg"].append(_pad(mg, args.kmax, 0.0, np.float32))
                    blk_lab["stop_k"].append(np.int8(sk))
                    blk_lab["rev_path"].append(_pad(rev, 2 * args.kmax))
                    blk_lab["rev_set"].append(_pad(rset, args.kmax))

                # все метки блока — массивы формы (B, ...), как и признаки
                Lab["e_empty"].append(e0.cpu().numpy())
                Lab["support"].append(supp.cpu().numpy())
                Lab["obs_idx"].append(np.arange(lo, hi))
                Lab["p"].append(np.full(B, p_, np.int16))
                for k, v in blk_lab.items():
                    Lab[k].append(np.stack(v))
        return

    for lo in range(0, N, args.batch):
        run_batch(lo, min(lo + args.batch, N))
        print(f"наблюдения {min(lo + args.batch, N)}/{N}", flush=True)

    # ---------------- сборка ----------------
    # И признаки, и метки накапливались блоками формы (B, ...) в одном и том же
    # порядке (батч -> позиция p -> пример), поэтому строки соответствуют друг
    # другу по индексу. Единственное исключение — g_flat: рваная таблица,
    # склеиваемая по g_off.
    feats = {k: np.concatenate(v) for k, v in F.items() if v}
    labels = {k: np.concatenate(v) for k, v in Lab.items()
              if k != "g_flat" and v}
    labels["g_flat"] = np.concatenate(Lab["g_flat"]).astype(np.float32)
    labels["g_off"] = np.asarray(g_off, np.int64)
    labels["split"] = SPLIT[labels["obs_idx"]]
    labels["episode"] = EPI[labels["obs_idx"]]
    n_int = len(labels["obs_idx"])
    assert labels["g_off"][-1] == len(labels["g_flat"])
    assert len(labels["g_off"]) == n_int + 1
    for k in ("int_obs_idx", "int_p"):
        assert len(feats[k]) == n_int, f"{k}: {len(feats[k])} против {n_int}"
    assert (feats["int_obs_idx"] == labels["obs_idx"]).all(), \
        "порядок строк признаков и меток разошёлся"
    assert (feats["int_p"] == labels["p"]).all()

    _sanity(feats, labels, EPI, SPLIT, TASKS, args, verify, P)


def _pad(seq, n, fill=-1, dt=np.int16):
    a = np.full(n, fill, dt)
    for i, x in enumerate(list(seq)[:n]):
        a[i] = x
    return a


def _sanity(feats, labels, EPI, SPLIT, TASKS, args, verify, P) -> None:
    """Проверки, падающие громко. Идут ДО записи файлов."""
    print("\n" + "=" * 70)
    print("САНИТАРНЫЕ ПРОВЕРКИ")
    print("=" * 70)

    bad = set(feats) - FEATURE_KEYS
    assert not bad, f"посторонние ключи в признаках: {bad}"
    for k in feats:
        low = k.lower()
        assert not any(s in low for s in FORBIDDEN_SUBSTR), \
            f"подозрительный ключ признака: {k}"
    print(f"  1. белый список признаков: {len(feats)} ключей, посторонних нет")

    eps = {s: set(EPI[SPLIT == s].tolist()) for s in (0, 1, 2)}
    assert not (eps[0] & eps[1]) and not (eps[0] & eps[2]) \
        and not (eps[1] & eps[2]), "пересечение эпизодов между split"
    print(f"  2. пересечения эпизодов между split нет "
          f"({len(eps[0])}/{len(eps[1])}/{len(eps[2])})")

    n_ep = len(np.unique(EPI))
    assert n_ep >= args.n_ep, f"эпизодов {n_ep} < {args.n_ep}"
    print(f"  3. различных эпизодов {n_ep} >= {args.n_ep}")

    T = np.asarray(TASKS)
    miss = [t for t in sorted(set(TASKS))
            if len({int(s) for s in SPLIT[T == t]}) < 2]
    print(f"  4. задач, представленных менее чем в двух split: {len(miss)} "
          f"из {len(set(TASKS))}")

    oi, pp, sp = labels["obs_idx"], labels["p"], labels["split"]
    for o in np.unique(oi)[:200]:
        m = oi == o
        assert len(np.unique(sp[m])) == 1, f"наблюдение {o} в разных split"
        assert len(np.unique(pp[m])) == P, f"наблюдение {o}: не все позиции p"
    print(f"  5. все {P} вмешательств наблюдения лежат в одном split")

    # сверка СЖАТИЯ с полным перебором: verify содержит максимальные
    # расхождения |G_полн(S) - G_сжат(S∩C)|, посчитанные при построении
    if verify:
        w = max(verify)
        g1m = float(np.abs(labels["sing_gain"]).max())
        print(f"  6. сжатие G(S) = G(S∩C) сверено с полным перебором 2517 "
              f"наборов на {len(verify)} примерах:\n"
              f"      максимум расхождения {w:.3e} "
              f"({w / max(g1m, 1e-30):.2e} от максимального одиночного выигрыша)")
        assert w < 1e-6 * max(g1m, 1e-30) + 1e-12, \
            f"сжатие НЕ точное: расхождение {w:.3e}"
    else:
        print("  6. сверка сжатия не проводилась (--verify-full 0)")

    gs = labels["g_star"]
    thr = args.gap_rel * float(np.median(gs))
    print(f"  7. G* <= порога (ремонтировать нечего): "
          f"{(gs <= thr).mean():.2%}, порог {thr:.3e}")
    sg = labels["sing_gain"]
    print(f"  8. доля отрицательных одиночных выигрышей: {(sg < 0).mean():.2%}")
    sz = (labels["best_le_k"] >= 0).sum(1)
    print(f"  9. размер лучшего набора <=K: " + " ".join(
        f"{i}:{(sz == i).mean():.0%}" for i in range(args.kmax + 1)))
    print(f" 10. средняя длина сжатой таблицы: "
          f"{np.diff(labels['g_off']).mean():.1f} наборов "
          f"(полный перебор дал бы 2517)")


if __name__ == "__main__":
    main()
