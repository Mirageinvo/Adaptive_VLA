"""K-7b: состояния action-запросов на нескольких глубинах, ПЕРВЫЙ блок.

ВОПРОС. K-6h показал, что два прохода башни из трёх не покупают успеха. Значит
остаётся один проход, и вся оставшаяся экономия — в его ГЛУБИНЕ: с какого слоя
грубые коды уже определены. Если с двенадцатого, это ещё примерно вдвое.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ k6i. Тот сохранял полный префикс VLM (181 токен, 2048
каналов, 37 ГиБ) ради вопроса «предсказать тонкие уровни лучше». Этот вопрос
K-6h обесценил: тонкие уровни не влияют на успех. Здесь сохраняются только 16
состояний action-запросов, (N, 16, d) на глубину — при 4000 наблюдений и
четырёх глубинах порядка 0.4 ГБ вместо 37 ГиБ.

ГДЕ СТАВЯТСЯ ХУКИ. Не на башню VLM, а на `action_expert.layers`: именно там
живут 16 позиций действия, и именно их читает `action_lm_head`. Число слоёв
эксперта по конструкции совпадает с VLM (bar.py:41), но здесь оно НЕ
предполагается — берётся из модели, и запрошенные глубины проверяются.

ЧТО СЧИТАЕТСЯ ЦЕЛЬЮ. Грубые коды, которые выдала САМА BAR на полной глубине
(K_bar[:, 0, :]). Вопрос ранней остановки — «можно ли воспроизвести раньше то,
что даёт полная башня», а не «угадать токенизатор». Коды токенизатора (K_true)
сохраняются тоже, как вторичная метрика.

ГЛУБИНА d ОЗНАЧАЕТ ВХОД В СЛОЙ d при нумерации с единицы, то есть состояние
после d-1 слоёв. Полная глубина снимается с самой `action_expert.norm`: её
ВХОД идёт в `h_after_<N>`, а ВЫХОД — в `hn_after_<N>`, и выход обязан побитово
совпадать со входом `action_lm_head` (проверяется). Так полная глубина не
нормируется дважды, в отличие от промежуточных, к которым норму приходится
применять отдельно.

СТРОКИ ВЫРОВНЕНЫ: строка k в любом массиве — наблюдение k. Обход идёт группами
по офсету позиции, но запись ведётся по индексам, а не подряд.

Запуск:
    python3 experiments/k7b_depth_extract.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k7b_depth_extract.py --ckpt <ckpt> \\
        --n-obs 4000 --n-ep 800 --depths 6,12,18,24 \\
        --out data/k7b_depth_4k.npz
"""

import argparse
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20


def hook_index(after_n, total):
    """Куда ставить хук, чтобы снять состояние ПОСЛЕ ровно `after_n` слоёв.

    Вход `input_layernorm` слоя k — это остаточный поток после k-1 слоёв.
    Значит «после 6 слоёв» = вход слоя 7, то есть индекс 6 при нумерации с
    нуля. Раньше здесь было `layers[d-1]`, что давало состояние после d-1
    слоёв при подписи «глубина d»: смещение на один слой, из-за которого
    заявленная экономия раннего выхода была бы систематически завышена.

    Для after_n == total слоя-приёмника не существует: там берётся вход
    action_lm_head, ключ `final`.
    """
    if not 1 <= after_n <= total:
        raise SystemExit(f"«после {after_n} слоёв» вне диапазона 1..{total}")
    return None if after_n == total else after_n


def check_depths(want, total):
    bad = [d for d in want if not 1 <= d <= total]
    if bad:
        raise SystemExit(f"глубины {bad} вне диапазона 1..{total}")
    if len(set(want)) != len(want):
        raise SystemExit(f"повторы в списке глубин: {want}")
    return sorted(want)


def selftest():
    # ГЛАВНАЯ ПРОВЕРКА: «после N слоёв» это вход слоя N+1, а не слоя N.
    # Прежняя версия ставила хук на layers[d-1] и подписывала строку «глубина
    # d» — состояние после d-1 слоёв. Из такой таблицы экономия раннего выхода
    # считалась бы завышенной ровно на один слой.
    assert hook_index(6, 24) == 6, "после 6 слоёв = вход слоя 7 = индекс 6"
    assert hook_index(1, 24) == 1
    assert hook_index(24, 24) is None, "после всех слоёв — только голова"
    for bad in (0, 25, -1):
        try:
            hook_index(bad, 24)
            raise AssertionError(f"{bad} должно падать")
        except SystemExit:
            pass

    assert check_depths([24, 6, 12], 24) == [6, 12, 24]
    for bad, tot in (([0], 24), ([25], 24), ([13], 12)):
        try:
            check_depths(bad, tot)
            raise AssertionError(f"глубина {bad} при {tot} слоях должна падать")
        except SystemExit:
            pass
    try:
        check_depths([6, 6], 24)
        raise AssertionError("повтор должен падать")
    except SystemExit:
        pass

    # ЗАПИСЬ ПО ИНДЕКСАМ при обходе группами по офсету: строка k обязана быть
    # наблюдением k. Тот же отказ, что чинили в k6i.
    N = 7
    dst = np.zeros((N, 2), np.float32)
    offs = np.array([1, 0, 1, 0, 1, 0, 0])
    for po in (0, 1):
        sel = np.where(offs == po)[0]
        dst[sel] = np.stack([np.full(2, float(j)) for j in sel])
    assert (dst[:, 0] == np.arange(N)).all(), dst[:, 0]

    # Раскладка кодов поуровневая: первые 16 токенов — грубый уровень.
    K = np.arange(N_POS * N_LEVEL).reshape(1, N_LEVEL, N_POS)
    assert (K[0, 0] == np.arange(16)).all()

    print("самопроверка k7b пройдена: «после N слоёв» = вход слоя N+1 "
          "(hook_index), после всех слоёв — только голова, запись по индексам "
          "сохраняет соответствие строк, раскладка поуровневая")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-obs", type=int, default=4000)
    ap.add_argument("--n-ep", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--depths", default="6,12,18,24")
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k7b_depth_4k.npz")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       process_state, prompt_template, seed_everything)

    seed_everything(args.seed)
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    n_codes = int(cfg.MODEL.action_processor.vocab_size)

    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    if getattr(model, "action_expert", None) is None:
        raise SystemExit("у модели нет action_expert — глубины снимать не с чего")
    ex_layers = model.action_expert.layers
    depths = check_depths([int(v) for v in args.depths.split(",")], len(ex_layers))
    print(f"слоёв эксперта: {len(ex_layers)}")
    for d in depths:
        hi = hook_index(d, len(ex_layers))
        where = ("вход action_lm_head" if hi is None
                 else f"вход input_layernorm слоя {hi + 1} (индекс {hi})")
        print(f"  after_{d:<3} = состояние ПОСЛЕ {d} слоёв -> {where}")

    keys, grab = [], {}
    for d in depths:
        hi = hook_index(d, len(ex_layers))
        k_ = f"after_{d}"
        keys.append(k_)
        grab[k_] = []
        if hi is None:
            continue          # снимется вместе с `final` ниже
        ex_layers[hi].input_layernorm.register_forward_hook(
            (lambda kk: (lambda m, i, o:
                         grab[kk].append(i[0].detach().float().cpu())))(k_))
    full_key = f"after_{len(ex_layers)}"
    if full_key not in grab:
        keys.append(full_key)
        grab[full_key] = []

    # ПОЛНАЯ ГЛУБИНА СНИМАЕТСЯ С САМОЙ НОРМЫ, А НЕ СО ВХОДА ГОЛОВЫ. Вход
    # action_lm_head — это уже результат action_expert.norm, и если положить
    # его в общий массив, то последующая нормировка (которую мы применяем к
    # промежуточным глубинам) выполнится по второму разу. Тогда строка полной
    # глубины перестанет быть настоящим входом головы, а проверка 100% могла
    # бы даже случайно пройти при неверной постановке.
    # Хук на норме даёт обе величины сразу: вход = состояние до нормы, выход =
    # ровно то, что потребляет голова.
    # ВНИМАНИЕ: этот модуль вызывается НЕ ТОЛЬКО из строки bar.py:1247. При
    # первом прогоне хук сработал шесть раз вместо трёх. Гадать о причине
    # нельзя — поэтому записываем ВСЕ вызовы вместе с формами, отбираем те, у
    # которых 16 позиций (action-запросы), и печатаем формы остальных, чтобы
    # причина стала видна из лога, а не осталась догадкой.
    grab_n = {full_key: []}
    grab["_lm_codes"], grab["_lm_in"], grab["_norm_all"] = [], [], []

    def norm_hook(m, i, o):
        grab["_norm_all"].append((tuple(i[0].shape),
                                  i[0].detach().float().cpu(),
                                  o.detach().float().cpu()))

    def lm_hook(m, i, o):
        grab["_lm_in"].append(i[0].detach().float().cpu())
        grab["_lm_codes"].append(o.detach().argmax(-1).cpu())

    model.action_expert.norm.register_forward_hook(norm_hook)
    model.action_lm_head.register_forward_hook(lm_hook)

    # --- данные: ДАТАСЕТНЫЙ путь предобработки ------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(1693)
    per_ep = max(1, args.n_obs // max(args.n_ep, 1))

    def png(cell):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))

    im1, im2, st, act, tsk, epi = [], [], [], [], [], []
    for e in order:
        if len(tsk) >= args.n_obs:
            break
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        if t.num_rows < T_CHUNK + 1:
            continue
        A_ = np.asarray(t.column("actions").to_pylist(), np.float32)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        n_st = t.num_rows - T_CHUNK + 1
        for s0 in rng.choice(n_st, size=min(per_ep, n_st), replace=False):
            im1.append(png(c1[int(s0)])); im2.append(png(c2[int(s0)]))
            st.append(S_[int(s0)]); act.append(A_[int(s0):int(s0) + T_CHUNK])
            tsk.append(tasks_map[ti[int(s0)]]); epi.append(int(e))
    N = len(tsk)
    epi = np.asarray(epi)
    hw = im1[0].shape[0]
    tf = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])   # ДОЛЯ, не число
    print(f"собрано {N} наблюдений, {len(np.unique(epi))} эпизодов, кадр {hw}")

    ST = np.asarray(st, np.float64)
    if ST.shape[1] == len(STATE_Q01) + 1:
        ST = process_state(ST)
    st_n = (ST - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0

    a_codec = np.asarray(act, np.float64).copy()
    a_codec[..., :-1] /= max_act_q[..., :-1]
    a_codec[..., -1] *= -1
    a_codec = np.clip(a_codec, -1.0, 1.0)
    K_true = np.asarray(proc.action_processor.encode(a_codec),
                        np.int64).reshape(N, N_LEVEL, N_POS)

    off_by_task = None
    if args.pos_offset is None:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; задайте --pos-offset")
        tb = json.load(open(args.offset_table))
        off_by_task = {k: int(v["pos_offset"]) for k, v in tb["tasks"].items()}
        miss = sorted({t for t in tsk if t not in off_by_task})
        if miss:
            raise SystemExit(f"нет офсета для {len(miss)} задач: {miss[:3]}")
    offs = np.array([args.pos_offset if args.pos_offset is not None
                     else off_by_task[tsk[i]] for i in range(N)])

    H = {}
    K_bar = np.zeros((N, N_LEVEL, N_POS), np.int64)
    done = 0
    lm_checked = [False]
    for po in sorted({int(v) for v in offs}):
      idx_po = np.where(offs == po)[0]
      for i0 in range(0, len(idx_po), args.batch):
        sel = idx_po[i0:i0 + args.batch]
        b = len(sel)
        done += b
        i1 = tf(torch.tensor(np.stack([im1[j] for j in sel])).permute(0, 3, 1, 2))
        i2 = tf(torch.tensor(np.stack([im2[j] for j in sel])).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for gi in sel:
            m = prompt_template(
                st_n[gi], None, tsk[gi],
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=n_codes,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        batch = proc(text=texts, images=[[image[k].numpy()] for k in range(b)],
                     return_tensors="pt", padding=True, padding_side="left",
                     action_processor_kwargs={"embodiment_ids": 0})
        batch = dict_apply(lambda x: x.to(dev, dtype), batch)
        for k_ in list(grab):
            grab[k_] = []          # переприсваиваем: full_key заполняется отбором
        for k_ in list(grab_n):
            grab_n[k_] = []
        with torch.no_grad():
            tk = model.generate(**batch, position_offset=po, do_sample=False,
                                initial_position_shift=1)
        K_bar[sel] = tk.cpu().numpy().reshape(b, N_LEVEL, N_POS)

        assert len(grab["_lm_codes"]) == N_LEVEL, (
            f"голова сработала {len(grab['_lm_codes'])} раз, ждали {N_LEVEL}")

        # ОТБОР ВЫЗОВОВ НОРМЫ ПО ФОРМЕ: нас интересуют только те, где 16
        # позиций action-запросов. Прочие вызовы того же модуля к делу не
        # относятся, но их формы печатаются один раз — чтобы было видно, что
        # это, а не осталось необъяснённым.
        # ДЛИНА ПОСЛЕДОВАТЕЛЬНОСТИ РАСТЁТ ПО БЛОКАМ: блок 0 — это 16 позиций
        # BOS, блок 1 добавляет 16 уже сгенерированных токенов, блок 2 ещё 16
        # (bar.py:1196-1199). Поэтому норма вызывается трижды с формами
        # 16/32/48, и нужный нам блок 0 — ЕДИНСТВЕННЫЙ с 16 позициями, а не
        # один из трёх одинаковых. Ровно та же картина у хуков промежуточных
        # глубин, где берётся [0].
        shapes = [t[0] for t in grab["_norm_all"]]
        cand = [t for t in grab["_norm_all"] if t[0][1] == N_POS]
        if not lm_checked[0]:
            print(f"    вызовов action_expert.norm за generate: {len(shapes)} "
                  f"с формами {shapes} — по одному на блок, длина растёт")
        if len(grab["_norm_all"]) != N_LEVEL or len(cand) != 1:
            raise SystemExit(
                f"ждали {N_LEVEL} вызовов нормы, из них ровно один с {N_POS} "
                f"позициями (блок 0). Получено {len(shapes)} вызовов, "
                f"подходящих {len(cand)}. Формы: {shapes}")
        if shapes[0][1] != N_POS:
            raise SystemExit(
                f"первый вызов нормы имеет форму {shapes[0]}, а блок 0 обязан "
                f"идти первым — порядок блоков не тот, что предполагается")
        grab[full_key] = [cand[0][1]]
        grab_n[full_key] = [cand[0][2]]
        # СВЕРКА С НАСТОЯЩИМ ВЫХОДОМ ГОЛОВЫ. Если не совпало — сохранённое
        # состояние относится не к тому вызову, который породил K_bar, и
        # никакой зонд на нём не имеет смысла.
        lm0 = grab["_lm_codes"][0].numpy()
        bad = int((lm0 != K_bar[sel][:, 0, :]).sum())
        if bad:
            raise SystemExit(
                f"argmax выхода action_lm_head на первом блоке расходится с\n"
                f"K_bar[:,0,:] в {bad} из {lm0.size} позиций. Сохранённые\n"
                f"состояния относятся не к тому вызову — зонд был бы ложным.")
        # ВЫХОД НОРМЫ ОБЯЗАН БЫТЬ ВХОДОМ ГОЛОВЫ, побитово. Если нет — между
        # ними есть ещё что-то, и «полная глубина» снята не оттуда.
        dn = float((grab_n[full_key][0] - grab["_lm_in"][0]).abs().max())
        if dn != 0.0:
            raise SystemExit(
                f"выход action_expert.norm не совпал со входом action_lm_head "
                f"(max|Δ| = {dn:.3e}): между ними есть необлюдаемая операция, "
                f"и строка полной глубины снята не оттуда.")
        if not lm_checked[0]:
            lm_checked[0] = True
            print(f"    сверка: argmax головы == K_bar[:,0,:] на всех "
                  f"{lm0.size} позициях батча; выход нормы == вход головы "
                  f"побитово")
        for k_ in keys:
            # ПЕРВЫЙ блок: именно он даёт грубый уровень. Хуки промежуточных
            # глубин срабатывают по разу на блок (формы 16/32/48), а полная
            # глубина уже отобрана выше по форме и содержит ровно один элемент.
            want_n = 1 if k_ == full_key else N_LEVEL
            assert len(grab[k_]) == want_n, (
                f"хук {k_} сработал {len(grab[k_])} раз, ждали {want_n}")
            c = grab[k_][0].numpy()
            assert c.shape[1] == N_POS, (
                f"{k_}: позиций {c.shape[1]}, ждали {N_POS} — снят не тот модуль")
            if k_ not in H:
                H[k_] = np.zeros((N, N_POS, c.shape[-1]), np.float16)
            H[k_][sel] = c.astype(np.float16)
            # НОРМИРОВАННАЯ ВЕРСИЯ ТОЖЕ. Голова кодов принимает вход ПОСЛЕ
            # action_expert.norm (bar.py:1246). Промежуточные глубины
            # снимаются до неё, и подавать их в голову напрямую — сравнивать
            # величины разной природы: полная глубина оказывается в выигрыше
            # просто потому, что уже нормирована.
            if k_ + "_n" not in H:
                H[k_ + "_n"] = np.zeros_like(H[k_])
            if k_ == full_key:
                # Уже снято хуком на самой норме. Нормировать повторно нельзя.
                cn_np = grab_n[full_key][0].numpy()
            else:
                # В ТОМ ЖЕ dtype, ЧТО У МОДЕЛИ: RMSNorm внутри поднимает до
                # fp32 и возвращает во входной тип, поэтому вычисление в fp32
                # дало бы не то, что делает сама модель.
                with torch.no_grad():
                    cn_np = model.action_expert.norm(
                        torch.as_tensor(c, dtype=dtype, device=dev)
                    ).float().cpu().numpy()
            H[k_ + "_n"][sel] = cn_np.astype(np.float16)
        if done % (args.batch * 50) < args.batch:
            print(f"  {done}/{N} (офсет {po})", flush=True)
    assert done == N, f"обработано {done} из {N}"

    print()
    for k_ in keys:
        a = H[k_].astype(np.float32)
        an = H[k_ + "_n"].astype(np.float32)
        if not (np.isfinite(a).all() and np.isfinite(an).all()):
            raise SystemExit(f"не-конечные значения на глубине {k_}")
        print(f"  глубина {str(k_):>9}: сырое sd {a.std():7.3f} "
              f"max|x| {np.abs(a).max():8.1f}   после нормы sd {an.std():7.3f} "
              f"max|x| {np.abs(an).max():7.1f}")
    print(f"\n  совпадение кодов BAR с токенизатором, грубый уровень: "
          f"{(K_bar[:, 0, :] == K_true[:, 0, :]).mean():.1%}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out,
             **{f"h_{k_}": H[k_] for k_ in keys},
             **{f"hn_{k_}": H[k_ + "_n"] for k_ in keys},
             K_true=K_true, K_bar=K_bar, act=a_codec.astype(np.float32),
             episode=epi, task=np.asarray(tsk), pos_offset=offs,
             meta=json.dumps(dict(
                 ckpt=args.ckpt, n_obs=N, n_episodes=int(len(np.unique(epi))),
                 depths=depths, keys=[str(k_) for k_ in keys],
                 normed_keys=[f"hn_{k_}" for k_ in keys],
                 n_expert_layers=len(ex_layers), n_codes=n_codes,
                 d_model=int(H[keys[0]].shape[-1]),
                 source="after_N = состояние ПОСЛЕ N слоёв action_expert, то "
                        "есть вход input_layernorm слоя N+1; для N = числа "
                        "слоёв это вход action_lm_head (после финальной "
                        "нормировки, поэтому эта строка иной природы, чем "
                        "остальные). Первый блок BAR.",
                 lm_head_check="argmax выхода action_lm_head на первом блоке "
                               "совпал с K_bar[:,0,:] во всех батчах",
                 target_note="цель зонда — K_bar[:,0,:], то есть что выдаёт сама "
                             "BAR на полной глубине; K_true вторичен"),
                 ensure_ascii=False))
    sz = sum(H[k_].nbytes + H[k_ + "_n"].nbytes for k_ in keys) / 2 ** 20
    print(f"  сохранено: {args.out}  ({sz:.0f} МиБ)")
    print("  Строка k — наблюдение k во всех массивах, перестановки нет.")


if __name__ == "__main__":
    main()
