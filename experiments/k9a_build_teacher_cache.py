"""K-9a: кэш учителя — coarse-выход полной BAR на первом блоке.

ЗАЧЕМ УЧИТЕЛЬ, А НЕ КОДЫ ТОКЕНИЗАТОРА. K-8b целился в K_true, но BAR совпадает
с токенизатором по грубому уровню лишь на 87.1%, а успех в замкнутом цикле
следует за поведением BAR, а не за реконструкцией демонстрации: K-6h показал,
что coarse-only BAR не теряет успеха, и что реконструкционный зазор с успехом
почти не связан. Поэтому цель ученика — воспроизвести решение УЧИТЕЛЯ вдвое
меньшей глубиной, а K_true остаётся вторичной метрикой.

ПОЧЕМУ ОДИН ПРОХОД, А НЕ ТРИ. Нужен только блок 0, а его даёт
`_predict_next_block_logits` при пустой истории — это ОДИН проход башни вместо
трёх у полного `generate`. Совпадение с официальным путём проверяется на первом
батче полным `generate`, дальше идёт дешёвый путь. Тождество этих двух путей
уже установлено в K-8a побитово (логиты, max|Δ| = 0.000e+00).

РАЗБИЕНИЕ СОЗДАЁТСЯ ЗАНОВО. Прежний тест использовался при отборе конфигураций
в K-8c и тем самым стал development-выборкой. Новый manifest пишется один раз и
не перезаписывается: если файл существует, он читается, а не создаётся.

Запуск:
    python3 experiments/k9a_build_teacher_cache.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9a_build_teacher_cache.py --ckpt <ckpt> \\
        --n-obs 20000 --n-ep 1600 --out data/k9_teacher_20k.npz
"""

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20


def make_split(epi, task, seed=0, frac=(0.8, 0.1)):
    """По эпизодам, стратифицировано по задачам. Возвращает массив меток."""
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
        parts = (("train", ep[:n - n_va - n_te]),
                 ("val", ep[n - n_va - n_te:n - n_te]),
                 ("test", ep[n - n_te:]))
        for name, p in parts:
            lab[g[np.isin(epi[g], p)]] = name
    assert (lab != "").all(), "остались наблюдения без метки"
    return lab.astype(str)


def selftest():
    epi = np.repeat(np.arange(60), 4)
    tsk = np.array([f"t{e % 6}" for e in epi])
    lab = make_split(epi, tsk, seed=0)
    for nm in ("train", "val", "test"):
        m = lab == nm
        assert m.any(), f"часть {nm} пуста"
        assert set(tsk[m]) == set(tsk), f"{nm}: задачи потеряны"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(epi[lab == a]) & set(epi[lab == b])), \
            f"эпизод попал и в {a}, и в {b}"
    assert 0.7 < (lab == "train").mean() < 0.9, (lab == "train").mean()

    # Разбиение обязано быть ВОСПРОИЗВОДИМЫМ при том же сиде и меняться при
    # другом: manifest пишется один раз и потом только читается.
    assert (make_split(epi, tsk, seed=0) == lab).all()
    assert not (make_split(epi, tsk, seed=1) == lab).all()

    # Раскладка кодов поуровневая: первые 16 токенов — грубый уровень.
    K = np.arange(N_POS * N_LEVEL).reshape(1, N_LEVEL, N_POS)
    assert (K[0, 0] == np.arange(16)).all()

    print("самопроверка k9a пройдена: разбиение по эпизодам со стратификацией "
          "по задачам, без утечки, воспроизводимо по сиду; раскладка кодов "
          "поуровневая")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-obs", type=int, default=20000)
    ap.add_argument("--n-ep", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--manifest", default="data/k9_split_manifest.json")
    ap.add_argument("--split-seed", type=int, default=17,
                    help="сид РАЗБИЕНИЯ, отличный от прежних: старый тест уже "
                         "использовался при отборе в K-8c")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/k9_teacher_20k.npz")
    ap.add_argument("--no-images", action="store_true",
                    help="не сохранять кадры. По умолчанию рядом пишется "
                         "memmap уже подготовленных изображений: при большой "
                         "выборке держать их в оперативной памяти нельзя "
                         "(96000 наблюдений это 38 ГиБ)")
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
    V = int(codec.vocab_size)

    # --- данные: ДАТАСЕТНЫЙ путь предобработки ------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    # РОВНО n_ep ЭПИЗОДОВ И РОВНО n_obs НАБЛЮДЕНИЙ. Прежняя версия считала
    # per_ep = n_obs // n_ep и шла по всем 1693 эпизодам, пока не наберёт
    # n_obs: при 20000/1600 это давало около 1667 эпизодов и 20004 строки, то
    # есть заявленное в команде расходилось с фактическим. Здесь эпизоды
    # отбираются заранее, а остаток наблюдений распределяется по первым из них.
    rng = np.random.default_rng(args.seed)
    png = lambda c: np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
    # КВОТЫ ПРОПОРЦИОНАЛЬНЫ ВМЕСТИМОСТИ ЭПИЗОДА, А НЕ РАВНЫ. Прежняя схема
    # требовала от каждого эпизода одинаковой квоты и отбраковывала короткие.
    # При 12 наблюдениях это ни на что не влияло, но при 94 (150000 на 1600)
    # эпизоду нужно уже 114 кадров, а в среднем их 162 — короткие выпали бы, и
    # 1600 могло не набраться вовсе.
    #
    # ЭПИЗОДЫ БЕРУТСЯ ИЗ MANIFEST, ЕСЛИ ОН ЕСТЬ. Иначе разные по размеру кэши
    # строились бы на разных наборах демонстраций, и сравнивать прогоны было бы
    # нельзя — при том что разбиение идёт как раз по эпизодам.
    def _rows(e):
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        return f, pq.read_metadata(f).num_rows      # без разбора всей таблицы

    fixed = None
    if os.path.exists(args.manifest):
        fixed = [int(e) for e in json.load(open(args.manifest))["episodes"]]
        print(f"эпизоды берутся из manifest: {len(fixed)}")

    chosen, tables, cap = [], {}, {}
    order = fixed if fixed is not None else list(rng.permutation(1693))
    for e in order:
        e = int(e)
        if fixed is None and len(chosen) >= args.n_ep:
            break
        f, nr = _rows(e)
        c = nr - T_CHUNK + 1
        if c < 1:
            continue
        chosen.append(e); tables[e] = f; cap[e] = c
    if fixed is None and len(chosen) < args.n_ep:
        raise SystemExit(f"годных эпизодов {len(chosen)}, нужно {args.n_ep}")
    total_cap = sum(cap.values())
    if total_cap < args.n_obs:
        raise SystemExit(
            f"во всех {len(chosen)} эпизодах {total_cap} стартовых позиций, "
            f"а запрошено {args.n_obs}. Больше в LIBERO взять неоткуда.")

    # Пропорционально вместимости, потом добираем остаток по эпизодам с
    # наибольшим запасом — так суммарно выходит РОВНО n_obs.
    quota = {e: min(cap[e], int(args.n_obs * cap[e] / total_cap))
             for e in chosen}
    left = args.n_obs - sum(quota.values())
    for e in sorted(chosen, key=lambda x: cap[x] - quota[x], reverse=True):
        if left <= 0:
            break
        add = min(left, cap[e] - quota[e])
        quota[e] += add; left -= add
    if left:
        raise SystemExit(f"не удалось разложить {left} наблюдений")
    qs = np.array([quota[e] for e in chosen])
    print(f"отобрано {len(chosen)} эпизодов, квоты {qs.min()}–{qs.max()} "
          f"(медиана {int(np.median(qs))}), суммарно {int(qs.sum())}; "
          f"использовано {qs.sum() / total_cap:.0%} доступных стартов")

    im1, im2, st, act, tsk, epi, stp = [], [], [], [], [], [], []
    for e in chosen:
        if quota[e] == 0:
            continue
        t = pq.read_table(tables[e])
        A_ = np.asarray(t.column("actions").to_pylist(), np.float32)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        n_st = t.num_rows - T_CHUNK + 1
        for s0 in rng.choice(n_st, size=quota[e], replace=False):
            im1.append(png(c1[int(s0)])); im2.append(png(c2[int(s0)]))
            st.append(S_[int(s0)]); act.append(A_[int(s0):int(s0) + T_CHUNK])
            tsk.append(tasks_map[ti[int(s0)]]); epi.append(int(e)); stp.append(int(s0))
    N = len(tsk)
    assert N == args.n_obs and len(set(epi)) == len(chosen), (N, len(set(epi)))
    epi, tsk_a, stp = np.asarray(epi), np.asarray(tsk), np.asarray(stp)
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

    tb = json.load(open(args.offset_table))
    off_by_task = {k: int(v["pos_offset"]) for k, v in tb["tasks"].items()}
    miss = sorted({t for t in tsk if t not in off_by_task})
    if miss:
        raise SystemExit(f"нет офсета для задач: {miss[:3]}")
    offs = np.array([off_by_task[t] for t in tsk])

    # --- MANIFEST: пишется один раз, дальше только читается -----------------
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)) or ".",
                exist_ok=True)
    if os.path.exists(args.manifest):
        mf = json.load(open(args.manifest))
        # ПРОВЕРЯЕТСЯ ВСЁ, ЧТО МОЖЕТ РАЗОЙТИСЬ МОЛЧА. Manifest фиксирован, и
        # несоответствие сида или ревизии датасета означает, что выборка не та,
        # на которой он строился, — а обнаружилось бы это только по странным
        # результатам через несколько часов.
        if int(mf.get("split_seed", -1)) != args.split_seed:
            raise SystemExit(
                f"manifest создан с сидом {mf.get('split_seed')}, а запрошен "
                f"{args.split_seed}. Либо уберите --split-seed, либо возьмите "
                f"новый файл: перезаписывать зафиксированное разбиение нельзя.")
        if mf.get("dataset_revision", rev) != rev:
            raise SystemExit(f"manifest строился на ревизии "
                             f"{mf.get('dataset_revision')}, сейчас {rev}")
        eps, sps = mf["episodes"], mf["splits"]
        if len(eps) != len(set(eps)):
            raise SystemExit("в manifest есть повторяющиеся эпизоды")
        bad = sorted(set(sps) - {"train", "val", "test"})
        if bad:
            raise SystemExit(f"недопустимые метки частей: {bad}")
        key = {str(int(e)): s for e, s in zip(eps, sps)}
        unknown = sorted({int(e) for e in epi if str(int(e)) not in key})
        if unknown:
            raise SystemExit(
                f"в manifest нет {len(unknown)} эпизодов (например {unknown[:3]}). "
                f"Manifest фиксирован и не дополняется: возьмите ту же выборку "
                f"или создайте НОВЫЙ файл под другим именем.")
        extra = sorted(set(key) - {str(int(e)) for e in epi})
        if extra:
            print(f"    ВНИМАНИЕ: в manifest {len(extra)} эпизодов, которых нет "
                  f"в текущей выборке — сравнения с прошлыми прогонами будут "
                  f"на разных данных")
        split = np.array([key[str(int(e))] for e in epi])
        print(f"manifest прочитан: {args.manifest} (сид {mf['split_seed']}, "
              f"{len(eps)} эпизодов)")
    else:
        split = make_split(epi, tsk_a, seed=args.split_seed)
        json.dump(dict(episodes=[int(e) for e in np.unique(epi)],
                       splits=[str(split[np.where(epi == e)[0][0]])
                               for e in np.unique(epi)],
                       split_seed=args.split_seed, created_by=sha,
                       dataset_revision=rev, dataset_repo=rid,
                       n_obs=int(args.n_obs), n_ep=int(args.n_ep),
                       note="прежний тест использовался при отборе в K-8c и "
                            "стал development-выборкой; это разбиение новое и "
                            "перезаписи не подлежит"),
                  open(args.manifest, "w"), ensure_ascii=False, indent=1)
        print(f"manifest СОЗДАН: {args.manifest} (сид {args.split_seed})")
    for nm in ("train", "val", "test"):
        m = split == nm
        print(f"    {nm:<6}{int(m.sum()):>7} наблюдений, "
              f"{len(np.unique(epi[m])):>5} эпизодов, "
              f"{len(np.unique(tsk_a[m])):>3} задач")
        if len(np.unique(tsk_a[m])) != len(np.unique(tsk_a)):
            raise SystemExit(f"в части {nm} представлены не все задачи")

    # --- прогон учителя ------------------------------------------------------
    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    need = N * N_POS * V * 2
    IMG_HW, IMG_W = 224, 448
    if not args.no_images:
        need += N * 3 * IMG_HW * IMG_W
    import shutil
    free = shutil.disk_usage(outdir).free
    if free < need * 1.3:
        raise SystemExit(f"мало места: нужно {need * 1.3 / 2**30:.1f} ГиБ, "
                         f"свободно {free / 2**30:.1f}")
    print(f"логиты учителя плюс кадры: {need / 2**30:.2f} ГиБ")
    tmp = args.out + ".partial.npy"
    L = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                  shape=(N, N_POS, V))
    # КАДРЫ СОХРАНЯЮТСЯ УЖЕ ПОДГОТОВЛЕННЫМИ, ровно в том виде, в каком уходят
    # в процессор: два вида, склеенные по ширине, после кропа и приведения
    # размера. Так между сбором кэша и обучением не остаётся ни одного шага,
    # где предобработка могла бы разойтись, — а она у нас уже трижды
    # расходилась между датасетным и средовым путями.
    IMG = None
    img_tmp = args.out + ".partial.images.npy"
    if not args.no_images:
        IMG = np.lib.format.open_memmap(
            img_tmp, mode="w+", dtype=np.uint8, shape=(N, 3, IMG_HW, IMG_W))
    Kt = np.zeros((N, N_POS), np.int64)
    checked = [False]
    done = 0
    for po in sorted({int(v) for v in offs}):
      ipo = np.where(offs == po)[0]
      for i0 in range(0, len(ipo), args.batch):
        sel = ipo[i0:i0 + args.batch]
        b = len(sel)
        done += b
        i1 = tf(torch.tensor(np.stack([im1[j] for j in sel])).permute(0, 3, 1, 2))
        i2 = tf(torch.tensor(np.stack([im2[j] for j in sel])).permute(0, 3, 1, 2))
        image = torch.cat([i1, i2], dim=-1)
        msgs = []
        for gi in sel:
            m = prompt_template(st_n[gi], None, tsk[gi],
                                mode=cfg.MODEL.vla_processor.kwargs.mode,
                                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
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
        if IMG is not None:
            arr = image.numpy()
            assert arr.dtype == np.uint8 and arr.shape[1:] == (3, IMG_HW, IMG_W), \
                f"кадр {arr.dtype} {arr.shape[1:]}, ждали uint8 (3,{IMG_HW},{IMG_W})"
            IMG[sel] = arr

        if not checked[0]:
            checked[0] = True
            # 1. argmax сохранённых логитов == сохранённые коды.
            assert (L[sel].astype(np.float32).argmax(-1) == Kt[sel]).all(), \
                "argmax сохранённых логитов не совпал с сохранёнными кодами"
            # 2. коды == первые 16 токенов ОФИЦИАЛЬНОГО generate (три прохода).
            with torch.no_grad():
                tk = model.generate(**batch, position_offset=po, do_sample=False)
            k_gen = tk.cpu().numpy().reshape(b, N_LEVEL, N_POS)[:, 0, :]
            if not (k_gen == Kt[sel]).all():
                raise SystemExit(
                    "дешёвый путь (_predict_next_block_logits) разошёлся с "
                    "официальным generate по грубым кодам — кэш недействителен")
            # 3. СБОРКА ЛАТЕНТЫ ПРОВЕРЯЕТСЯ НА ТРЁХ УРОВНЯХ, А НЕ НА ОДНОМ.
            # Прежняя версия сравнивала свою сборку E0[q0] с официальным
            # decode кодов [q0, 0, 0] — но индекс 0 во второй и третьей книгах
            # это обычное кодовое слово, а не «уровень отсутствует», и decode
            # складывал E0[q0]+E1[0]+E2[0]. Совпадать это не обязано, а
            # результат только печатался и ничего не останавливал.
            # Осмысленно проверять то, для чего официальный эталон есть:
            # сборку ВСЕХ трёх уровней. Отсюда coarse-only следует по
            # построению, тем же путём, что в K-6h и K-7a.
            k3 = torch.as_tensor(K_true[sel]).long().to(dev)
            with torch.no_grad():
                z3 = sum(E[j][k3[:, j, :]] for j in range(N_LEVEL))
                x3, _ = codec._decode(z3, embodiment_ids=0)
            ref = np.asarray(proc.action_processor.decode(
                K_true[sel].reshape(b, -1).tolist())[0], np.float64)
            d3 = float(np.abs(x3[..., :7].float().cpu().numpy() - ref).max())
            print(f"    сверка на первом батче: argmax==коды, коды==generate, "
                  f"сборка трёх уровней против официального decode "
                  f"max|Δ| = {d3:.3e}")
            if d3 > 1e-3:
                raise SystemExit(
                    f"своя сборка латенты расходится с официальным decode на "
                    f"{d3:.3e} — кодбуки извлечены неверно, кэш недействителен")
        if done % (args.batch * 100) < args.batch:
            print(f"  {done}/{N} (офсет {po})", flush=True)
    assert done == N, f"обработано {done} из {N}"

    ag = float((Kt == K_true[:, 0, :]).mean())
    print(f"\nсогласие учителя с токенизатором на грубом уровне: {ag:.1%}")
    L.flush(); del L

    np.savez(args.out + ".partial.npz",
             teacher_codes_q0=Kt.astype(np.int16),
             K_true_q0=K_true[:, 0, :].astype(np.int16),
             K_true=K_true.astype(np.int16),
             action=a_codec.astype(np.float32),
             episode=epi, step=stp, task=tsk_a, pos_offset=offs, split=split,
             meta=json.dumps(dict(
                 ckpt=args.ckpt, n_obs=int(N), script_sha1=sha,
                 manifest=args.manifest, split_seed=args.split_seed,
                 vocab=V, image_hw=int(hw),
                 teacher_agreement_with_tokenizer=ag,
                 logits_file=os.path.basename(args.out) + ".logits.npy",
                 images_file=(None if args.no_images
                              else os.path.basename(args.out) + ".images.npy"),
                 image_shape=[3, IMG_HW, IMG_W],
                 target="coarse-выход полной BAR (учитель), НЕ коды токенизатора",
                 preprocessing="датасетный путь: без разворота каналов, кроп "
                               "0.875 от фактического размера, состояние уже 8-мерное"),
                 ensure_ascii=False))
    if IMG is not None:
        IMG.flush(); del IMG
        os.replace(img_tmp, args.out + ".images.npy")
    os.replace(tmp, args.out + ".logits.npy")
    os.replace(args.out + ".partial.npz", args.out)
    print(f"сохранено: {args.out} и {args.out}.logits.npy")
    print("  Строка k — наблюдение k во всех массивах, перестановки нет.")


if __name__ == "__main__":
    main()
