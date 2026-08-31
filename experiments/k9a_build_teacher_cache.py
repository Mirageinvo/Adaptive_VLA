"""K-9a: кэш учителя — coarse-выход полной BAR на первом блоке.

ЗАЧЕМ УЧИТЕЛЬ, А НЕ КОДЫ ТОКЕНИЗАТОРА. BAR совпадает с токенизатором по
грубому уровню лишь на 87%, а успех в замкнутом цикле следует за поведением
BAR: K-6h показал, что coarse-only BAR не теряет успеха, и что
реконструкционный зазор с успехом почти не связан. Цель ученика —
воспроизвести решение УЧИТЕЛЯ вдвое меньшей глубиной; K_true вторичен.

ГЛОБАЛЬНЫЙ ПОРЯДОК СТАРТОВ. Каждой паре (эпизод, шаг) присваивается
воспроизводимый приоритет, зависящий только от сида и самой пары, и берутся
первые n_obs. Отсюда сразу три свойства, которых не было при поэпизодных
квотах через rng.choice:
  * кэш на 20000 есть ПОДМНОЖЕСТВО кэша на 150000 — кривые сравнимы;
  * длинные эпизоды представлены пропорционально числу доступных стартов, без
    ручной раздачи остатка;
  * результат не зависит ни от порядка обхода, ни от состояния общего ГСЧ.

ПАМЯТЬ. Кадры НЕ собираются в списки: 150000 наблюдений по два вида 256x256
это около 55 ГиБ оперативной памяти. Каждый эпизод читается один раз, нужные
кадры сразу приводятся к виду, в котором уходят в процессор, и пишутся в
memmap на диске. Проход учителя читает их оттуда же.

ПАКЕТНОЕ КОДИРОВАНИЕ K_true. `action_processor.encode` переносит весь массив
на GPU; на 150000 чанков это переполнение памяти. Кодируется по частям.

MANIFEST ПРОВЕРЯЕТСЯ ЦЕЛИКОМ И ДО ЧТЕНИЯ ДАННЫХ. Он фиксирует разбиение по
эпизодам, и расхождение набора демонстраций делает сравнение прогонов
недействительным — поэтому здесь отказ, а не предупреждение.

Запуск:
    python3 experiments/k9a_build_teacher_cache.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9a_build_teacher_cache.py --ckpt <ckpt> \\
        --n-obs 150000 --n-ep 1600 --out data/k9_teacher_150k.npz
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20
IMG_H, IMG_W = 224, 448          # два вида, склеенные по ширине
N_EPISODES = 1693                # meta/info.json: 1693 эпизода, 273465 кадров


def priorities(episode, n_steps, seed):
    """Приоритеты стартов эпизода — только от сида и номера эпизода.

    Не зависят ни от порядка обхода, ни от того, сколько наблюдений просят:
    именно поэтому меньший кэш оказывается подмножеством большего.
    """
    return np.random.default_rng([seed, int(episode)]).random(n_steps)


def select(caps, n_obs, seed):
    """Первые n_obs пар (эпизод, шаг) по глобальному приоритету."""
    eps, sts, prs = [], [], []
    for e, c in sorted(caps.items()):
        eps.append(np.full(c, e, np.int64))
        sts.append(np.arange(c, dtype=np.int64))
        prs.append(priorities(e, c, seed))
    eps, sts, prs = np.concatenate(eps), np.concatenate(sts), np.concatenate(prs)
    if len(prs) < n_obs:
        raise SystemExit(f"доступно {len(prs)} стартовых позиций, запрошено "
                         f"{n_obs}: больше в датасете взять неоткуда")
    take = np.argsort(prs, kind="stable")[:n_obs]
    order = np.lexsort((sts[take], eps[take]))     # по эпизодам: для потока
    return eps[take][order], sts[take][order]


def make_split(epi, task, seed=0, frac=(0.8, 0.1)):
    """По эпизодам, стратифицировано по задачам."""
    rng = np.random.default_rng(seed)
    lab = np.full(len(epi), "", dtype=object)
    for t in np.unique(task):
        g = np.where(task == t)[0]
        ep = rng.permutation(np.unique(epi[g]))
        n = len(ep)
        if n < 3:
            raise SystemExit(f"задача «{t}»: {n} эпизодов, не делится на три")
        n_va = max(1, int(round(n * frac[1])))
        n_te = max(1, int(round(n * (1.0 - frac[0] - frac[1]))))
        if n - n_va - n_te < 1:
            n_va, n_te = 1, 1
        for name, p in (("train", ep[:n - n_va - n_te]),
                        ("val", ep[n - n_va - n_te:n - n_te]),
                        ("test", ep[n - n_te:])):
            lab[g[np.isin(epi[g], p)]] = name
    assert (lab != "").all()
    return lab.astype(str)


def check_manifest(mf, want_n_ep, rev):
    """Полная проверка ДО чтения данных: набор эпизодов должен совпадать."""
    eps = [int(e) for e in mf["episodes"]]
    if len(eps) != want_n_ep:
        raise SystemExit(
            f"в manifest {len(eps)} эпизодов, а запрошено {want_n_ep}. "
            f"Manifest фиксирован: либо задайте --n-ep {len(eps)}, либо "
            f"стройте новый файл под другим именем.")
    if len(eps) != len(set(eps)):
        raise SystemExit("в manifest есть повторяющиеся эпизоды")
    bad = sorted(set(mf["splits"]) - {"train", "val", "test"})
    if bad:
        raise SystemExit(f"недопустимые метки частей: {bad}")
    if len(mf["splits"]) != len(eps):
        raise SystemExit("длины episodes и splits не совпадают")
    if mf.get("dataset_revision", rev) != rev:
        raise SystemExit(f"manifest на ревизии {mf.get('dataset_revision')}, "
                         f"сейчас {rev}")
    return eps


def selftest():
    # 1. Приоритеты воспроизводимы и не зависят от порядка обхода.
    a = priorities(7, 50, 3)
    assert np.array_equal(a, priorities(7, 50, 3))
    assert not np.array_equal(a[:20], priorities(8, 20, 3))
    assert np.array_equal(a[:20], priorities(7, 20, 3)), (
        "приоритеты зависят от числа шагов — подмножество сломается")

    # 2. ГЛАВНОЕ СВОЙСТВО: меньший набор есть подмножество большего.
    caps = {e: 30 + (e * 7) % 40 for e in range(20)}
    e1, s1 = select(caps, 50, seed=5)
    e2, s2 = select(caps, 300, seed=5)
    small = set(zip(e1.tolist(), s1.tolist()))
    big = set(zip(e2.tolist(), s2.tolist()))
    assert small <= big, f"{len(small - big)} пар выпали из большего набора"
    assert len(small) == 50 and len(big) == 300, (len(small), len(big))

    # 3. Длинные эпизоды представлены пропорционально, без ручных квот.
    cnt = {e: int((e2 == e).sum()) for e in caps}
    tot = sum(caps.values())
    for e in caps:
        exp = 300 * caps[e] / tot
        assert abs(cnt[e] - exp) < 0.5 * exp + 3, (e, cnt[e], exp)

    # 4. Разбиение: по эпизодам, все задачи всюду, воспроизводимо.
    epi = np.repeat(np.arange(60), 4)
    tsk = np.array([f"t{e % 6}" for e in epi])
    lab = make_split(epi, tsk, seed=0)
    for nm in ("train", "val", "test"):
        m = lab == nm
        assert m.any() and set(tsk[m]) == set(tsk)
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(epi[lab == a_]) & set(epi[lab == b_]))
    assert (make_split(epi, tsk, seed=0) == lab).all()
    assert not (make_split(epi, tsk, seed=1) == lab).all()

    # 5. Проверка manifest отвергает несовпадение числа эпизодов и дубли.
    mf = dict(episodes=[1, 2, 3], splits=["train", "val", "test"])
    assert check_manifest(mf, 3, "v2.0") == [1, 2, 3]
    for bad_mf, n in ((mf, 4),
                      (dict(episodes=[1, 1], splits=["train", "val"]), 2)):
        try:
            check_manifest(bad_mf, n, "v2.0")
            raise AssertionError("должно падать")
        except SystemExit:
            pass

    print("самопроверка k9a пройдена: приоритеты воспроизводимы и не зависят "
          "от числа шагов, меньший набор ВЛОЖЕН в больший, длинные эпизоды "
          "представлены пропорционально, разбиение по эпизодам со "
          "стратификацией, manifest проверяется на число эпизодов и дубли")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-obs", type=int, default=150000)
    ap.add_argument("--n-ep", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--encode-batch", type=int, default=512)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--manifest", default="data/k9_split_manifest.json")
    ap.add_argument("--split-seed", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--out", default="data/k9_teacher_150k.npz")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")
    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k9a sha1 {sha}")

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
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    rid, rev = "physical-intelligence/libero", "v2.0"
    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)) or ".",
                exist_ok=True)

    # --- MANIFEST ПРОВЕРЯЕТСЯ ДО ЧТЕНИЯ ДАННЫХ ------------------------------
    fixed, split_of = None, None
    if os.path.exists(args.manifest):
        mf = json.load(open(args.manifest))
        fixed = check_manifest(mf, args.n_ep, rev)
        if int(mf.get("split_seed", -1)) != args.split_seed:
            raise SystemExit(f"manifest создан с сидом {mf.get('split_seed')}, "
                             f"запрошен {args.split_seed}")
        split_of = {int(e): s for e, s in zip(mf["episodes"], mf["splits"])}
        print(f"manifest прочитан и проверен: {len(fixed)} эпизодов, "
              f"сид {mf['split_seed']}")

    def parquet(e):
        return hf_hub_download(
            rid, f"data/chunk-{int(e) // 1000:03d}/episode_{int(e):06d}.parquet",
            repo_type="dataset", revision=rev)

    # --- вместимости: только метаданные, без разбора таблиц ------------------
    cand = fixed if fixed is not None else list(range(N_EPISODES))
    caps, files = {}, {}
    for i, e in enumerate(cand):
        f = parquet(e)
        c = pq.read_metadata(f).num_rows - T_CHUNK + 1
        if c >= 1:
            caps[int(e)] = int(c); files[int(e)] = f
        if i % 400 == 0:
            print(f"  вместимость {i}/{len(cand)}", flush=True)
    if fixed is not None and set(caps) != set(fixed):
        raise SystemExit(
            f"из manifest непригодны {sorted(set(fixed) - set(caps))[:5]} — "
            f"набор демонстраций отличался бы, и сравнение прогонов было бы "
            f"нечистым")
    if fixed is None:
        keep = sorted(caps, key=lambda e: -caps[e])[:args.n_ep]
        caps = {e: caps[e] for e in keep}
    print(f"эпизодов {len(caps)}, доступных стартов {sum(caps.values())}")

    epi, stp = select(caps, args.n_obs, args.seed)
    N = len(epi)
    print(f"отобрано {N} наблюдений ({N / sum(caps.values()):.0%} доступных), "
          f"эпизодов {len(np.unique(epi))}")

    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    V = int(cfg.MODEL.action_processor.vocab_size)
    need = N * N_POS * V * 2 + (0 if args.no_images else N * 3 * IMG_H * IMG_W)
    free = shutil.disk_usage(outdir).free
    print(f"нужно {need / 2**30:.1f} ГиБ, свободно {free / 2**30:.0f}")
    if free < need * 1.15:
        raise SystemExit("мало места")

    # --- ПОТОКОВОЕ ЧТЕНИЕ: кадры сразу в memmap, в память не собираются ------
    img_tmp = args.out + ".partial.images.npy"
    IMG = None if args.no_images else np.lib.format.open_memmap(
        img_tmp, mode="w+", dtype=np.uint8, shape=(N, 3, IMG_H, IMG_W))
    act = np.zeros((N, T_CHUNK, 7), np.float32)
    tsk = np.empty(N, dtype=object)
    st, tf = None, None
    png = lambda c: np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
    uniq = np.unique(epi)
    for j, e in enumerate(uniq):
        t = pq.read_table(files[int(e)])
        A_ = np.asarray(t.column("actions").to_pylist(), np.float32)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        rows = np.where(epi == e)[0]
        if st is None:
            st = np.zeros((N, S_.shape[1]), np.float64)
        elif st.shape[1] != S_.shape[1]:
            raise SystemExit(f"эпизод {e}: состояние {S_.shape[1]}-мерное")
        if IMG is not None:
            c1 = t.column("image").to_pylist()
            c2 = t.column("wrist_image").to_pylist()
            if tf is None:
                # ДОЛЯ от фактического размера, а не число: кадры датасета 256,
                # кадры среды 224, и путаница между ними уже стоила трёх падений.
                hw = png(c1[0]).shape[0]
                tf = Compose([CenterCrop(int(hw * 0.875)), Resize(IMG_H)])
            i1 = tf(torch.tensor(np.stack([png(c1[int(stp[r])]) for r in rows])
                                 ).permute(0, 3, 1, 2))
            i2 = tf(torch.tensor(np.stack([png(c2[int(stp[r])]) for r in rows])
                                 ).permute(0, 3, 1, 2))
            im = torch.cat([i1, i2], dim=-1).numpy()
            assert im.dtype == np.uint8 and im.shape[1:] == (3, IMG_H, IMG_W), \
                f"кадр {im.dtype} {im.shape[1:]}"
            IMG[rows] = im
            del i1, i2, im, c1, c2
        for r in rows:
            s0 = int(stp[r])
            st[r] = S_[s0]
            act[r] = A_[s0:s0 + T_CHUNK]
            tsk[r] = tasks_map[ti[s0]]
        del t, A_, S_
        if j % 200 == 0:
            print(f"  эпизодов {j}/{len(uniq)}", flush=True)
    tsk_a = tsk.astype(str)
    print("кадры и состояния собраны")

    # --- модель и кодек ------------------------------------------------------
    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None)).to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    if st.shape[1] == len(STATE_Q01) + 1:
        st = process_state(st)
    st_n = (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    a_codec = act.astype(np.float64)
    a_codec[..., :-1] /= max_act_q[..., :-1]
    a_codec[..., -1] *= -1
    a_codec = np.clip(a_codec, -1.0, 1.0)

    # ПАКЕТНО: encode переносит весь массив на GPU, и 150000 чанков не влезут.
    K_true = np.zeros((N, N_LEVEL, N_POS), np.int64)
    for i0 in range(0, N, args.encode_batch):
        j0 = min(i0 + args.encode_batch, N)
        K_true[i0:j0] = np.asarray(
            proc.action_processor.encode(a_codec[i0:j0]),
            np.int64).reshape(j0 - i0, N_LEVEL, N_POS)
        if i0 % (args.encode_batch * 50) == 0:
            print(f"  кодирование {i0}/{N}", flush=True)

    tb = json.load(open(args.offset_table))
    off_by_task = {k: int(v["pos_offset"]) for k, v in tb["tasks"].items()}
    miss = sorted({t for t in tsk_a if t not in off_by_task})
    if miss:
        raise SystemExit(f"нет офсета для задач: {miss[:3]}")
    offs = np.array([off_by_task[t] for t in tsk_a])

    if fixed is None:
        split = make_split(epi, tsk_a, seed=args.split_seed)
        eu = np.unique(epi)
        json.dump(dict(episodes=[int(e) for e in eu],
                       splits=[str(split[np.where(epi == e)[0][0]]) for e in eu],
                       split_seed=args.split_seed, created_by=sha,
                       dataset_revision=rev, dataset_repo=rid,
                       note="разбиение зафиксировано и перезаписи не подлежит"),
                  open(args.manifest, "w"), ensure_ascii=False, indent=1)
        print(f"manifest СОЗДАН: {args.manifest}")
    else:
        split = np.array([split_of[int(e)] for e in epi])
    for nm in ("train", "val", "test"):
        m = split == nm
        print(f"    {nm:<6}{int(m.sum()):>8} наблюдений, "
              f"{len(np.unique(epi[m])):>5} эпизодов, "
              f"{len(np.unique(tsk_a[m])):>3} задач")
        if len(np.unique(tsk_a[m])) != len(np.unique(tsk_a)):
            raise SystemExit(f"в части {nm} представлены не все задачи")

    # --- проход учителя ------------------------------------------------------
    tmp = args.out + ".partial.npy"
    L = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                  shape=(N, N_POS, V))
    Kt = np.zeros((N, N_POS), np.int64)
    checked = [False]
    done = 0
    for po in sorted({int(v) for v in offs}):
      ipo = np.where(offs == po)[0]
      for i0 in range(0, len(ipo), args.batch):
        sel = ipo[i0:i0 + args.batch]
        b = len(sel)
        done += b
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for gi in sel:
            m = prompt_template(st_n[gi], None, tsk_a[gi],
                                mode=cfg.MODEL.vla_processor.kwargs.mode,
                                action_vocab_size=V,
                                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        batch = proc(text=texts, images=[[image[k].numpy()] for k in range(b)],
                     return_tensors="pt", padding=True, padding_side="left",
                     action_processor_kwargs={"embodiment_ids": 0})
        batch = dict_apply(lambda x: x.to(dev, dt), batch)
        with torch.no_grad():
            B_, _, vemb, _ = model._build_vlm_inputs_embeds(
                input_ids=batch.get("input_ids"), inputs_embeds=None,
                pixel_values=batch.get("pixel_values"),
                pixel_attention_mask=batch.get("pixel_attention_mask"),
                image_hidden_states=None)
            apos = model._build_action_pos_ids_strided(
                batch_size=B_, base_pos=vemb.shape[1],
                action_seq_len=model.block_size, device=dev, position_offset=po)
            pos = model._build_joint_position_ids(
                batch_size=B_, vlm_seq_len=vemb.shape[1], action_pos_ids=apos,
                device=dev)
            lg = model._predict_next_block_logits(
                vlm_inputs_embeds=vemb, attention_mask=batch.get("attention_mask"),
                history_tokens=None, position_ids=pos)
        L[sel] = lg.detach().to(torch.float16).cpu().numpy()
        Kt[sel] = lg.argmax(-1).cpu().numpy()

        if not checked[0]:
            checked[0] = True
            assert (L[sel].astype(np.float32).argmax(-1) == Kt[sel]).all()
            with torch.no_grad():
                tk = model.generate(**batch, position_offset=po, do_sample=False)
            k_gen = tk.cpu().numpy().reshape(b, N_LEVEL, N_POS)[:, 0, :]
            if not (k_gen == Kt[sel]).all():
                raise SystemExit("дешёвый путь разошёлся с официальным generate")
            # СБОРКА ПРОВЕРЯЕТСЯ НА ТРЁХ УРОВНЯХ: только для них есть
            # официальный эталон. Индекс 0 в книгах 1-2 — обычное кодовое
            # слово, а не «уровень отсутствует».
            k3 = torch.as_tensor(K_true[sel]).long().to(dev)
            with torch.no_grad():
                z3 = sum(E[j][k3[:, j, :]] for j in range(N_LEVEL))
                x3, _ = codec._decode(z3, embodiment_ids=0)
            ref = np.asarray(proc.action_processor.decode(
                K_true[sel].reshape(b, -1).tolist())[0], np.float64)
            d3 = float(np.abs(x3[..., :7].float().cpu().numpy() - ref).max())
            print(f"    сверка: argmax==коды, коды==generate, сборка трёх "
                  f"уровней против decode max|Δ| = {d3:.3e}")
            if d3 > 1e-3:
                raise SystemExit("сборка латенты расходится с decode")
        if done % (args.batch * 400) < args.batch:
            print(f"  {done}/{N} (офсет {po})", flush=True)
    assert done == N, f"обработано {done} из {N}"

    ag = float((Kt == K_true[:, 0, :]).mean())
    print(f"\nсогласие учителя с токенизатором: {ag:.1%}")
    L.flush(); del L
    if IMG is not None:
        IMG.flush(); del IMG

    np.savez(args.out + ".partial.npz",
             teacher_codes_q0=Kt.astype(np.int16),
             K_true_q0=K_true[:, 0, :].astype(np.int16),
             K_true=K_true.astype(np.int16),
             action=a_codec.astype(np.float32),
             episode=epi, step=stp, task=tsk_a, pos_offset=offs, split=split,
             meta=json.dumps(dict(
                 ckpt=args.ckpt, n_obs=int(N), script_sha1=sha,
                 manifest=args.manifest, split_seed=args.split_seed,
                 vocab=V, seed=args.seed,
                 n_episodes=int(len(np.unique(epi))),
                 available_starts=int(sum(caps.values())),
                 teacher_agreement_with_tokenizer=ag,
                 logits_file=os.path.basename(args.out) + ".logits.npy",
                 images_file=(None if args.no_images
                              else os.path.basename(args.out) + ".images.npy"),
                 image_shape=[3, IMG_H, IMG_W],
                 selection="глобальный приоритет по (эпизод, шаг): меньший "
                           "кэш есть подмножество большего при том же сиде",
                 target="coarse-выход полной BAR, НЕ коды токенизатора"),
                 ensure_ascii=False))
    if not args.no_images:
        os.replace(img_tmp, args.out + ".images.npy")
    os.replace(tmp, args.out + ".logits.npy")
    os.replace(args.out + ".partial.npz", args.out)
    print(f"сохранено: {args.out}, .logits.npy"
          + ("" if args.no_images else ", .images.npy"))
    print("  Строка k — наблюдение k во всех массивах, перестановки нет.")


if __name__ == "__main__":
    main()
