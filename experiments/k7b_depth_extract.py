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
после d-1 слоёв. Для последнего слоя дополнительно снимается вход
`action_lm_head` — это состояние ПОСЛЕ всех слоёв и финальной нормировки, то
есть ровно то, из чего BAR и получает коды. Оно сохраняется как глубина
`final`, и служит верхней границей: голова на нём обязана давать почти 100%.

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


def check_depths(want, total):
    """Глубины задаются с единицы и обязаны лежать в диапазоне. Молчаливое
    обрезание сюда пускать нельзя: «слой 24» из 12 означал бы совсем не то,
    что написано в таблице результатов."""
    bad = [d for d in want if not 1 <= d <= total]
    if bad:
        raise SystemExit(f"глубины {bad} вне диапазона 1..{total}")
    if len(set(want)) != len(want):
        raise SystemExit(f"повторы в списке глубин: {want}")
    return sorted(want)


def selftest():
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

    print("самопроверка пройдена: глубины с единицы и проверяются, запись по "
          "индексам сохраняет соответствие строк, раскладка поуровневая")


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
    print(f"слоёв эксперта: {len(ex_layers)}; снимаю входы слоёв {depths} "
          f"плюс `final` (вход action_lm_head)")

    grab = {d: [] for d in depths}
    grab["final"] = []
    model.action_lm_head.register_forward_hook(
        lambda m, i, o: grab["final"].append(i[0].detach().float().cpu()))
    for d in depths:
        ex_layers[d - 1].input_layernorm.register_forward_hook(
            (lambda dd: (lambda m, i, o:
                         grab[dd].append(i[0].detach().float().cpu())))(d))

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

    keys = depths + ["final"]
    H = {}
    K_bar = np.zeros((N, N_LEVEL, N_POS), np.int64)
    done = 0
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
        for k_ in keys:
            grab[k_].clear()
        with torch.no_grad():
            tk = model.generate(**batch, position_offset=po, do_sample=False,
                                initial_position_shift=1)
        K_bar[sel] = tk.cpu().numpy().reshape(b, N_LEVEL, N_POS)

        assert len(grab["final"]) == N_LEVEL, (
            f"голова сработала {len(grab['final'])} раз, ждали {N_LEVEL}")
        for k_ in keys:
            # ПЕРВЫЙ блок: именно он даёт грубый уровень. Хуки эксперта
            # срабатывают по разу на блок, как и голова.
            assert len(grab[k_]) == N_LEVEL, (
                f"хук {k_} сработал {len(grab[k_])} раз, ждали {N_LEVEL}")
            c = grab[k_][0].numpy()
            assert c.shape[1] == N_POS, (
                f"{k_}: позиций {c.shape[1]}, ждали {N_POS} — снят не тот модуль")
            if k_ not in H:
                H[k_] = np.zeros((N, N_POS, c.shape[-1]), np.float16)
            H[k_][sel] = c.astype(np.float16)
        if done % (args.batch * 50) < args.batch:
            print(f"  {done}/{N} (офсет {po})", flush=True)
    assert done == N, f"обработано {done} из {N}"

    print()
    for k_ in keys:
        a = H[k_].astype(np.float32)
        if not np.isfinite(a).all():
            raise SystemExit(f"не-конечные значения на глубине {k_}")
        print(f"  глубина {str(k_):>5}: форма {H[k_].shape}  "
              f"sd {a.std():7.3f}  max|x| {np.abs(a).max():9.1f}")
    print(f"\n  совпадение кодов BAR с токенизатором, грубый уровень: "
          f"{(K_bar[:, 0, :] == K_true[:, 0, :]).mean():.1%}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out,
             **{f"h_{k_}": H[k_] for k_ in keys},
             K_true=K_true, K_bar=K_bar, act=a_codec.astype(np.float32),
             episode=epi, task=np.asarray(tsk), pos_offset=offs,
             meta=json.dumps(dict(
                 ckpt=args.ckpt, n_obs=N, n_episodes=int(len(np.unique(epi))),
                 depths=depths, keys=[str(k_) for k_ in keys],
                 n_expert_layers=len(ex_layers), n_codes=n_codes,
                 d_model=int(H[keys[0]].shape[-1]),
                 source="вход input_layernorm слоёв action_expert (нумерация с "
                        "единицы) плюс `final` = вход action_lm_head; первый блок",
                 target_note="цель зонда — K_bar[:,0,:], то есть что выдаёт сама "
                             "BAR на полной глубине; K_true вторичен"),
                 ensure_ascii=False))
    sz = sum(H[k_].nbytes for k_ in keys) / 2 ** 20
    print(f"  сохранено: {args.out}  ({sz:.0f} МиБ)")
    print("  Строка k — наблюдение k во всех массивах, перестановки нет.")


if __name__ == "__main__":
    main()
