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
FEATURE_KEYS = (
    # константа кодека: позволяет ВОССТАНОВИТЬ полные векторы, не храня их
    #   delta_emb   = codebook_proj[0][int_v] - codebook_proj[0][int_u]
    #   cand_latent = sum_j codebook_proj[j][cand_old_tokens[..., j]]
    # это точно и на два порядка компактнее, чем хранить сами векторы
    "codebook_proj",
    # уровень наблюдения
    "obs_pooled_ctx", "obs_task_idx", "obs_state",
    # уровень вмешательства (наблюдение + позиция правки p)
    "int_obs_idx", "int_p", "int_u", "int_v",
    "int_coarse_delta_norm", "int_coarse_cos", "int_logp_diff",
    "int_rank_u", "int_logp_u", "int_logp_v",
    # уровень кандидата q, форма (n_int, P, ...)
    "cand_entropy", "cand_margin", "cand_topk_p", "cand_topk_idx",
    "cand_old_tokens", "cand_q", "cand_dq", "cand_absdq", "cand_is_p",
    "cand_latent_norm", "cand_coarse_logp", "cand_coarse_entropy",
)
FEATURE_SET = frozenset(FEATURE_KEYS)
# скрытые состояния действия из СТАРОГО прохода лежат отдельным файлом:
# массив крупный, и в npz он мешал бы ленивой загрузке
HIDDEN_FILE = "features_hidden.npy"
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


def split_by_episode(epi, tasks, fracs=(0.70, 0.15, 0.15), seed: int = 0,
                     task_in_train: bool = True):
    """Разбиение ПО ЭПИЗОДАМ со стратификацией ПО ЗАДАЧАМ.

    Наивное «round(0.15 * n) эпизодов задачи в validation» непригодно: в LIBERO
    около 130 задач, при 400 эпизодах на задачу приходится ~3, и round(0.45)
    даёт НОЛЬ — validation остаётся пустым. Поэтому используется метод
    наибольшего дефицита с переносом остатка ЧЕРЕЗ задачи: каждый следующий
    эпизод уходит в ту часть, которой сейчас больше всего не хватает до целевой
    доли. Глобальные пропорции при этом точные, а эпизоды одной задачи
    расходятся по разным частям.

    Все наблюдения эпизода и все его вмешательства попадают в одну часть по
    построению, потому что решение принимается на уровне эпизода."""
    rng = np.random.default_rng(seed)
    ep_task = {}
    for e, t in zip(epi, tasks):
        ep_task.setdefault(int(e), t)
    out, have = {}, np.zeros(3)
    fr = np.asarray(fracs, float)
    by_task = {}
    for t in sorted(set(ep_task.values())):
        eps = np.array(sorted(e for e, tt in ep_task.items() if tt == t))
        rng.shuffle(eps)
        by_task[t] = eps
    # ПЕРВЫЙ эпизод каждой задачи уходит в train. Иначе задача может целиком
    # оказаться в val/test, и замер незаметно превратится в перенос на НОВЫЕ
    # задачи — другой протокол, который надо объявлять явно, а не получать
    # случайно. Для протокола переноса ставить task_in_train=False.
    if task_in_train:
        for t, eps in by_task.items():
            out[int(eps[0])] = 0
            have[0] += 1
    for t, eps in by_task.items():
        for e in (eps[1:] if task_in_train else eps):
            s = int(np.argmax(fr * (have.sum() + 1.0) - have))
            out[int(e)] = s
            have[s] += 1
    return np.array([out[int(e)] for e in epi], np.int8)


def subsets_of(C, kmax: int = 4):
    """Канонический порядок подмножеств C размера 0..kmax."""
    C = tuple(sorted(C))
    return [S for k in range(kmax + 1) for S in itertools.combinations(C, k)]


def greedy_paths(gmap, C, tau: float, kmax: int = 4):
    """Три траектории из СЖАТОЙ таблицы, без единого вызова модели.

    Порядок жадного отбора одинаков для MSE и RMS: максимум выигрыша — это
    минимум остаточной ошибки, а корень монотонен. Различается только момент
    ОСТАНОВКИ, поэтому tau задаётся на основной метрике (RMS).

    ADD          — жадное добавление ровно kmax шагов;
    ADD+STOP     — остановка, когда предельный выигрыш не превышает tau;
    ADD/REM/STOP — на каждом шаге рассматриваются и удаления. Каждый шаг строго
                   увеличивает G более чем на tau, поэтому состояние не может
                   повториться и искусственный предел длины не нужен; множество
                   посещённых состояний оставлено лишь как страховка."""
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

    S, acts, qs_, seen = (), [], [], {()}
    while len(acts) < 4 * kmax + 4:
        best = (tau, None, None)
        for q in C:
            if q in S:
                Tn = tuple(x for x in S if x != q)
                a_ = 0
            elif len(S) < kmax:
                Tn = tuple(sorted(S + (q,)))
                a_ = 1
            else:
                continue
            d = g(Tn) - g(S)
            if d > best[0]:
                best = (d, a_, q)
        if best[1] is None:
            break
        q = best[2]
        acts.append(best[1])
        qs_.append(q)
        S = (tuple(sorted(S + (q,))) if best[1] == 1
             else tuple(x for x in S if x != q))
        if S in seen:
            break
        seen.add(S)
    return add, marg, stop_k, acts, qs_, tuple(sorted(S))


def derive_labels(raw, split, P, kmax, tau_rel, gap_rel):
    """Все производные метки — ОТДЕЛЬНО от основного цикла.

    Так пороги считаются ТОЛЬКО по train и применяются к val/test без
    изменений, а любая правка правил пересчитывается без GPU.

    Основная метрика — RMS, потому что ворота B1 (0.40 / 0.79 / 0.91 / 0.93)
    получены на ней. Таблица хранится в MSE, перевод точный:

        e_S = e_empty - G_mse(S)
        G_rms(S) = sqrt(e_empty) - sqrt(max(e_S, 0))
    """
    gf, off = raw["g_flat"], raw["g_off"]
    e0 = raw["e_empty"].astype(np.float64)
    supp = raw["support"]
    n = len(e0)
    out = {}

    def tbl(i):
        C = [q for q in range(P) if supp[i] >> q & 1]
        subs = subsets_of(C, kmax)
        v = gf[off[i]:off[i + 1]].astype(np.float64)
        return C, {tuple(sorted(S)): v[j] for j, S in enumerate(subs)}

    def to_rms(i, g_mse):
        return np.sqrt(e0[i]) - np.sqrt(max(e0[i] - g_mse, 0.0))

    sing_mse = np.zeros((n, P), np.float32)
    sing_rms = np.zeros((n, P), np.float32)
    gstar_mse = np.zeros(n, np.float32)
    gstar_rms = np.zeros(n, np.float32)
    for i in range(n):
        C, gm = tbl(i)
        for q in C:
            sing_mse[i, q] = gm[(q,)]
            sing_rms[i, q] = to_rms(i, gm[(q,)])
        bm = max(0.0, max(gm.values()))
        gstar_mse[i] = bm
        gstar_rms[i] = to_rms(i, bm)

    # ПОРОГИ ТОЛЬКО ПО TRAIN, отдельно для каждой позиции вмешательства p.
    # В K-4a4 порог строился по агрегированному масштабу выигрыша, а не по
    # максимуму конкретного примера; здесь то же правило.
    tr = split == 0
    tau_p = np.zeros(P, np.float64)
    for p_ in range(P):
        m = tr & (raw["p"] == p_)
        med = float(np.median(sing_rms[m].max(1))) if m.any() else 0.0
        tau_p[p_] = max(1e-9, tau_rel * med)
    tau = tau_p[raw["p"]]
    gap_thr = gap_rel * float(np.median(gstar_rms[tr]))

    out["sing_gain_mse"] = sing_mse
    out["sing_gain_rms"] = sing_rms
    out["g_star_mse"] = gstar_mse
    out["g_star_rms"] = gstar_rms
    out["tau"] = tau.astype(np.float32)
    out["tau_by_p"] = tau_p.astype(np.float32)
    out["gap_threshold"] = np.float32(gap_thr)
    out["gap_valid"] = (gstar_rms > gap_thr)
    out["no_repair"] = ~out["gap_valid"]
    # нормируем ТОЛЬКО там, где ремонтировать есть что; иначе метка неустойчива
    nrm = np.zeros((n, P), np.float32)
    gv = out["gap_valid"]
    nrm[gv] = sing_rms[gv] / gstar_rms[gv, None]
    out["sing_gain_norm"] = nrm
    # знак по ПОРОГУ, а не по сравнению с нулём: 1 полезна, 0 нейтральна, -1 вредна
    sign = np.zeros((n, P), np.int8)
    sign[sing_rms > tau[:, None]] = 1
    sign[sing_rms < -tau[:, None]] = -1
    out["sing_sign"] = sign

    # ЛУЧШИЙ НАБОР ПО БЮДЖЕТАМ. Для оракула «ровно K» и «не более K» дают
    # ОДИНАКОВЫЙ выигрыш: недостающие слоты добиваются позициями вне support,
    # которые ничего не меняют. Различается ЦЕНА: фиксированный бюджет тратит K
    # всегда, адаптивный — best_size_by_k. Поэтому выигрыш хранится один раз, а
    # рядом лежит фактический размер.
    bs = np.full((n, kmax + 1, kmax), -1, np.int16)
    bg_mse = np.zeros((n, kmax + 1), np.float32)
    bg_rms = np.zeros((n, kmax + 1), np.float32)
    bsz = np.zeros((n, kmax + 1), np.int8)
    add_p = np.full((n, kmax), -1, np.int16)
    add_m = np.zeros((n, kmax), np.float32)
    stop = np.zeros(n, np.int8)
    rev_a = np.full((n, 4 * kmax + 4), -1, np.int8)
    rev_q = np.full((n, 4 * kmax + 4), -1, np.int16)
    rev_l = np.zeros(n, np.int16)
    rev_s = np.full((n, kmax), -1, np.int16)
    for i in range(n):
        C, gm = tbl(i)
        for K in range(kmax + 1):
            best, bv = (), -1e30
            for S, g in gm.items():
                if len(S) <= K and g > bv:
                    best, bv = S, g
            bs[i, K, :len(best)] = best
            bg_mse[i, K] = bv
            bg_rms[i, K] = to_rms(i, bv)
            bsz[i, K] = len(best)
        # траектории считаются на RMS: порядок тот же, но STOP по основной метрике
        gm_r = {S: to_rms(i, g) for S, g in gm.items()}
        a_, m_, sk, ra, rq, rs = greedy_paths(gm_r, C, float(tau[i]), kmax)
        add_p[i, :len(a_)] = a_
        add_m[i, :len(m_)] = m_
        stop[i] = sk
        rev_a[i, :len(ra)] = ra
        rev_q[i, :len(rq)] = rq
        rev_l[i] = len(ra)
        rev_s[i, :len(rs)] = rs
    out.update(best_set_by_k=bs, best_gain_by_k_mse=bg_mse,
               best_gain_by_k_rms=bg_rms, best_size_by_k=bsz,
               add_path=add_p, add_marg_rms=add_m, stop_k=stop,
               rev_action=rev_a, rev_q=rev_q, rev_len=rev_l, rev_set=rev_s)
    return out


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
    ap.add_argument("--vlm-dtype", default="bfloat16",
                    choices=["bfloat16", "float32"],
                    help="точность VLM. В bf16 (8 бит мантиссы) argmax по 2048 "
                         "кодам переворачивается на почти-ничьих при смене "
                         "композиции батча. На V100 bf16 всё равно эмулируется "
                         "на FP32-ядрах (K-4a3), то есть float32 там почти "
                         "бесплатен и заметно устойчивее")
    ap.add_argument("--task-in-train", type=int, default=1,
                    help="1 — гарантировать хотя бы один эпизод каждой задачи "
                         "в train (обобщение на новые эпизоды); 0 — допускать "
                         "задачи только в val/test (перенос на новые задачи)")
    ap.add_argument("--save-hidden", type=int, default=1,
                    help="сохранять скрытые состояния действия старого прохода "
                         "отдельным .npy; это сильнейший причинный признак")
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
    SPLIT = split_by_episode(EPI, TASKS, seed=args.seed,
                         task_in_train=bool(args.task_in_train))
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
        args.ckpt, trust_remote_code=True,
        dtype=getattr(torch, args.vlm_dtype),
        token_budget=P * L, num_blocks=L,
        action_vocab_size=tok.vocab_size).to(args.device).eval()
    bs, nb = model.block_size, model.num_blocks
    print(f"dtype модели по факту: {next(model.parameters()).dtype} "
          f"(запрошен {args.vlm_dtype})")
    vlens = []

    # РАНГИ ВМЕШАТЕЛЬСТВА фиксируются ДО батчевого цикла отдельным numpy RNG.
    # Раньше сид зависел от смещения батча, поэтому --batch 8 и --batch 16 дали
    # бы РАЗНЫЕ вмешательства и несравнимые датасеты.
    rank_table = np.random.default_rng(10_000 + args.seed).integers(
        args.rank_lo, args.rank_hi, size=(N, P))

    # Скрытые состояния позиций действия из УЖЕ СОСТОЯВШЕГОСЯ прохода —
    # сильнейший причинный признак. Берём предвызовным хуком на action_lm_head
    # (третий код при этом не правится).
    _cap = {}

    def _grab(_m, inp):
        _cap["h"] = inp[0].detach()

    model.action_lm_head.register_forward_pre_hook(_grab)
    hid_dim = int(model.action_lm_head.in_features)
    HID = (np.zeros((N * P, P, hid_dim), np.float16)
           if args.save_hidden else None)
    if HID is not None:
        print(f"скрытые состояния: {HID.shape}, "
              f"{HID.nbytes / 1e6:.0f} МБ (float16)")

    # ПЕРВЫЙ ПРОХОД: общий максимум длины промпта по ВСЕЙ выборке.
    # Иначе паддинг идёт до максимума В БАТЧЕ, vlen гуляет между батчами, а он
    # входит в base_pos позиционных id токенов действия. Замер: при --batch 8
    # последний батч дал vlen 174 против 175 у остальных, и 74 вмешательства из
    # 512 сменили changed-support. К точности это отношения не имеет — на
    # float32 числа совпали до последней цифры.
    print("первый проход: общая длина паддинга...")
    pad_to = 0
    for lo in range(0, N, args.batch):
        hi = min(lo + args.batch, N)
        bb = build_batch(IM1[lo:hi], IM2[lo:hi], TASKS[lo:hi], st_all[lo:hi],
                         proc, args, "cpu")
        pad_to = max(pad_to, int(bb["input_ids"].shape[1]))
        del bb
    print(f"  общая длина паддинга: {pad_to} токенов\n")

    row0 = [0]          # счётчик строк: последний батч короче, арифметика по lo неверна

    F = {k: [] for k in FEATURE_KEYS}
    F["codebook_proj"] = [E.float().cpu().numpy().astype(np.float16)]
    # В ЦИКЛЕ копятся только СЫРЫЕ метки. Всё производное (пороги, знаки,
    # нормировки, наборы по бюджетам, траектории) считается потом в
    # derive_labels: тогда пороги берутся только по train и любая правка правил
    # пересчитывается без GPU.
    Lab = {k: [] for k in ("e_empty", "support", "g_flat", "obs_idx", "p")}
    g_off = [0]
    verify = []

    def run_batch(lo, hi):
        nonlocal verify
        B = hi - lo
        args_ns = args
        batch = build_batch(IM1[lo:hi], IM2[lo:hi], TASKS[lo:hi],
                            st_all[lo:hi], proc, args_ns, args.device,
                            pad_to=pad_to)
        with torch.no_grad():
            # «Vocabulary expanded» в логе — предупреждение процессора; здесь
            # явно убеждаемся, что все id помещаются в таблицу эмбеддингов
            emb = model.vlm.text_model.get_input_embeddings()
            assert int(batch["input_ids"].max()) < emb.num_embeddings, (
                f"input_ids до {int(batch['input_ids'].max())} при таблице "
                f"{emb.num_embeddings}")
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

            # vlen — ДОПОЛНЕННАЯ длина префикса. Она входит в base_pos для
            # позиционных id токенов действия, поэтому при padding=True зависит
            # от состава батча. Логируем, чтобы это было видно.
            vlens.append((lo, hi, int(vlen), int(batch["input_ids"].shape[1])))
            print(f"    батч [{lo}:{hi}]: vlen {int(vlen)}, "
                  f"input_ids {int(batch['input_ids'].shape[1])}", flush=True)

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

            for p_ in range(P):
                ranks = lg0[:, p_].topk(args.rank_hi, -1).indices
                rk = torch.as_tensor(rank_table[lo:hi, p_], device=args.device)
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
                tk = pb.topk(args.topk_probs, -1)
                topk_p, topk_i = tk.values, tk.indices
                if HID is not None:
                    HID[row0[0]:row0[0] + B] = \
                        _cap["h"][:, -P:].float().cpu().numpy().astype(np.float16)
                row0[0] += B

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
                F["cand_topk_idx"].append(
                    topk_i.cpu().numpy().astype(np.int16))
                F["cand_old_tokens"].append(z_old.cpu().numpy().astype(np.int16))
                F["cand_q"].append(np.tile(np.arange(P, dtype=np.int16), (B, 1)))
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

                # ---- в цикле копим ТОЛЬКО сырое: таблицу и её смещения
                for b in range(B):
                    # G(пусто) = 0 ПО ОПРЕДЕЛЕНИЮ. Из-за разной композиции
                    # батча порядок редукции в декодере отличается, и выходит
                    # ~1e-9 вместо нуля — на три порядка ниже порога, но
                    # тождество лучше не размывать.
                    gg_all[b][0] = 0.0
                    Lab["g_flat"].append(gg_all[b])
                    g_off.append(g_off[-1] + len(gg_all[b]))
                Lab["e_empty"].append(e0.cpu().numpy())
                Lab["support"].append(supp.cpu().numpy())
                Lab["obs_idx"].append(np.arange(lo, hi))
                Lab["p"].append(np.full(B, p_, np.int16))
        return

    for lo in range(0, N, args.batch):
        run_batch(lo, min(lo + args.batch, N))
        print(f"наблюдения {min(lo + args.batch, N)}/{N}", flush=True)

    # ---------------- сборка ----------------
    # Признаки и сырые метки копились блоками формы (B, ...) в одном порядке
    # (батч -> позиция p -> пример), поэтому строки соответствуют по индексу.
    feats = {k: np.concatenate(v) for k, v in F.items() if v}
    raw = {k: np.concatenate(v) for k, v in Lab.items() if k != "g_flat" and v}
    raw["g_flat"] = np.concatenate(Lab["g_flat"]).astype(np.float32)
    raw["g_off"] = np.asarray(g_off, np.int64)
    n_int = len(raw["obs_idx"])
    assert raw["g_off"][-1] == len(raw["g_flat"])
    assert len(raw["g_off"]) == n_int + 1
    assert (feats["int_obs_idx"] == raw["obs_idx"]).all(), \
        "порядок строк признаков и меток разошёлся"
    assert (feats["int_p"] == raw["p"]).all()
    if HID is not None:
        assert row0[0] == n_int, f"скрытых состояний {row0[0]} против {n_int}"

    # КАНОНИЧЕСКИЙ ПОРЯДОК СТРОК: (наблюдение, позиция p). Порядок накопления
    # зависит от размера батча — при --batch 16 сначала идут все наблюдения
    # батча при p=0, при --batch 8 то же, но батчи короче. Содержимое одно, а
    # раскладка разная, и файлы становятся несравнимыми. Сортировка это снимает.
    perm = np.lexsort((raw["p"], raw["obs_idx"]))
    lens = np.diff(raw["g_off"])
    raw["g_flat"] = np.concatenate(
        [raw["g_flat"][raw["g_off"][i]:raw["g_off"][i + 1]] for i in perm])
    raw["g_off"] = np.concatenate([[0], np.cumsum(lens[perm])]).astype(np.int64)
    for k in list(raw):
        if k not in ("g_flat", "g_off"):
            raw[k] = raw[k][perm]
    for k in list(feats):
        if k.startswith(("int_", "cand_")):
            feats[k] = feats[k][perm]
    if HID is not None:
        HID[:n_int] = HID[:n_int][perm]
    assert (np.diff(raw["obs_idx"] * P + raw["p"]) > 0).all(), \
        "канонический порядок строк нарушен"

    split_int = SPLIT[raw["obs_idx"]]
    print("\nвывод производных меток (пороги — только по train)...")
    labels = dict(raw)
    labels.update(derive_labels(raw, split_int, P, args.kmax,
                                args.tau_rel, args.gap_rel))
    labels["split"] = split_int
    labels["episode"] = EPI[raw["obs_idx"]]

    _sanity(feats, labels, EPI, SPLIT, TASKS, args, verify, P)

    # ---------------- запись и ПОВТОРНАЯ ПРОВЕРКА ----------------
    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, "features.npz")
    lp = os.path.join(args.out, "labels.npz")
    mp = os.path.join(args.out, "metadata.json")
    np.savez_compressed(fp, **feats)
    np.savez_compressed(lp, **labels)
    if HID is not None:
        np.save(os.path.join(args.out, HIDDEN_FILE), HID[:n_int])
    meta = dict(commit=commit, ckpt=args.ckpt, seed=args.seed, n_obs=int(N),
                n_episodes=int(len(np.unique(EPI))), n_interventions=int(n_int),
                P=int(P), L=int(L), kmax=args.kmax, tau_rel=args.tau_rel,
                gap_rel=args.gap_rel, gap_threshold=float(labels["gap_threshold"]),
                tau_by_p=[float(x) for x in labels["tau_by_p"]],
                pos_offset=args.pos_offset, window=args.window,
                vlm_dtype=args.vlm_dtype, pad_to=int(pad_to),
                metric_table="MSE, нормировка на scale**2",
                metric_primary="RMS (как в воротах B1)",
                continuous_channels=int(D_act - 1), scale=scale,
                dataset="physical-intelligence/libero", revision="v2.0",
                feature_keys=sorted(feats), hidden_file=(HIDDEN_FILE
                                                         if HID is not None else None),
                hidden_dim=(int(hid_dim) if HID is not None else None),
                tasks=uniq_tasks, batch=args.batch,
                batch_independent_ranks=True,
                # Ранги вмешательства и порядок строк от батча не зависят. Но
                # логиты VLM в bfloat16 зависят: при почти равных значениях
                # topk/argmax переворачиваются, и выбранный код u может стать
                # другим. Побитовое воспроизведение требует ТОГО ЖЕ --batch.
                bitwise_repro_requires_same_batch=True,
                split_counts={k: int((SPLIT == i).sum())
                              for i, k in enumerate(("train", "val", "test"))})
    with open(mp, "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("ПРОВЕРКА ЗАПИСАННОГО (файлы открываются заново)")
    print("=" * 70)
    import hashlib
    fz, lz = np.load(fp, allow_pickle=True), np.load(lp, allow_pickle=True)
    assert set(fz.files) == set(feats), "ключи признаков не совпали"
    assert set(lz.files) == set(labels), "ключи меток не совпали"
    assert len(fz["int_obs_idx"]) == n_int and len(lz["obs_idx"]) == n_int
    assert (fz["int_obs_idx"] == lz["obs_idx"]).all(), \
        "индексы признаков и меток в файлах разошлись"
    assert np.array_equal(fz["cand_old_tokens"], feats["cand_old_tokens"])
    assert np.allclose(lz["g_flat"], labels["g_flat"])
    bad = set(fz.files) - FEATURE_SET
    assert not bad, f"в записанном файле посторонние ключи: {bad}"
    for path in (fp, lp, mp) + ((os.path.join(args.out, HIDDEN_FILE),)
                                if HID is not None else ()):
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        print(f"  {os.path.basename(path):>22}  "
              f"{os.path.getsize(path) / 1e6:>8.1f} МБ  sha256:{h}")
    print(f"  строк-вмешательств {n_int}, наблюдений {N}, "
          f"ключей признаков {len(fz.files)}, меток {len(lz.files)}")
    print(f"\nготово: {args.out}")


def _sanity(feats, labels, EPI, SPLIT, TASKS, args, verify, P) -> None:
    """Проверки, падающие громко. Идут ДО записи файлов."""
    print("\n" + "=" * 70)
    print("САНИТАРНЫЕ ПРОВЕРКИ")
    print("=" * 70)

    bad = set(feats) - FEATURE_SET
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
    fr = [float((SPLIT == s).mean()) for s in (0, 1, 2)]
    print(f"  4. доли split {[round(x, 3) for x in fr]} (цель 0.70/0.15/0.15); "
          f"задач в одном split: {len(miss)} из {len(set(TASKS))}")
    for s, nm in ((1, "validation"), (2, "test")):
        assert (SPLIT == s).sum() > 0, f"{nm} пуст — разбиение выродилось"

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
        # сверка идёт на MSE-шкале, поэтому и масштаб берётся оттуда
        g1m = float(np.abs(labels["sing_gain_mse"]).max())
        print(f"  6. сжатие G(S) = G(S∩C) сверено с полным перебором 2517 "
              f"наборов на {len(verify)} примерах:\n"
              f"      максимум расхождения {w:.3e} "
              f"({w / max(g1m, 1e-30):.2e} от максимального одиночного выигрыша)")
        assert w < 1e-6 * max(g1m, 1e-30) + 1e-12, \
            f"сжатие НЕ точное: расхождение {w:.3e}"
    else:
        print("  6. сверка сжатия не проводилась (--verify-full 0)")

    gs = labels["g_star_rms"]
    thr = float(labels["gap_threshold"])
    print(f"  7. ремонтировать нечего (G* <= порога, порог по TRAIN "
          f"{thr:.3e}): {labels['no_repair'].mean():.2%}")

    sg, tau = labels["sing_gain_rms"], labels["tau"][:, None]
    chg = np.stack([(labels["support"] >> q & 1).astype(bool)
                    for q in range(P)], 1)
    neg = sg < -tau
    print(f"  8. одиночные выигрыши по ПОРОГУ, а не по сравнению с нулём:")
    print(f"      отрицательных среди всех позиций      {neg.mean():.2%}")
    print(f"      отрицательных ВНУТРИ changed-support  "
          f"{neg[chg].mean():.2%}")
    print(f"      положительных внутри support          "
          f"{(sg > tau)[chg].mean():.2%}")

    sz = labels["best_size_by_k"][:, args.kmax]
    print(f"  9. размер лучшего набора при бюджете {args.kmax}: " + " ".join(
        f"{i}:{(sz == i).mean():.0%}" for i in range(args.kmax + 1)))
    gk = labels["best_gain_by_k_rms"]
    print(f"      средний выигрыш по бюджетам K=0..{args.kmax}: " + " ".join(
        f"{gk[:, K].mean():.2e}" for K in range(args.kmax + 1)))
    print(f" 10. средняя длина сжатой таблицы: "
          f"{np.diff(labels['g_off']).mean():.1f} наборов "
          f"(полный перебор дал бы 2517)")

    # ПОКРЫТИЕ ЗАДАЧ. Если задача val/test отсутствует в train, замер уходит в
    # перенос на НОВЫЕ задачи, а это другой протокол и его надо объявлять явно.
    T = np.asarray(TASKS)
    tr_t = set(T[SPLIT == 0])
    for s_, nm in ((1, "validation"), (2, "test")):
        miss = sorted(set(T[SPLIT == s_]) - tr_t)
        frac = ((~np.isin(T[SPLIT == s_], list(tr_t))).mean()
                if (SPLIT == s_).any() else 0.0)
        print(f" 11. задач в {nm}, отсутствующих в train: {len(miss)} "
              f"({frac:.1%} наблюдений части)")
    print("      если доля заметна, протокол — перенос на НОВЫЕ задачи, "
          "и это надо объявлять отдельно")

    rl = labels["rev_len"]
    print(f" 12. обратимые траектории: средняя длина {rl.mean():.2f}, "
          f"максимум {rl.max()}, доля с хотя бы одним REMOVE "
          f"{((labels['rev_action'] == 0).any(1)).mean():.2%}")


if __name__ == "__main__":
    main()
