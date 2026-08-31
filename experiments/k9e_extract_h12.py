"""K-9e: снять представление h12 для замороженного ствола.

ЗАЧЕМ. Разложение вклада требует четырёх клеток: ствол исходный или обученный,
голова исходная или обученная. Три из них считаются без единого шага обучения,
если один раз сохранить вход в `action_expert.norm` — то самое h12, которое
голова читает. Обучение головы на кэше идёт минуты вместо часов, и на этом
можно позволить себе перебор шага обучения, чтобы контроль вышел не слабее,
чем он может быть.

РАЗБИЕНИЕ ЗАФИКСИРОВАНО ЗАРАНЕЕ, иначе вклад останется неоднозначным:
  * СТВОЛ — 12 слоёв VLM, 12 слоёв эксперта и `bos_embedding` (это вход потока
    действий на нулевом слое, он участвует во всех двенадцати шагах внимания);
  * ГОЛОВА — `action_expert.norm` и `fast_head` (норма стоит вплотную перед
    головой и читает h12 вместе с ней).
Значит h12 — это вход нормы, а не её выход.

ХУК, А НЕ ПРАВКА joint12_vla.py. Второе было бы проще, но sha этого модуля
записывается в каждую ячейку симуляторного гейта, и правка посреди развёртки
расколола бы её на две несовместимые половины — агрегатор отказался бы их
объединять, и справедливо. Хук на `action_expert.norm` даёт ровно то же и
ничего не меняет. Приём проверен в K-7b.

ХУК НИЧЕГО НЕ ВОЗВРАЩАЕТ. В K-7b возврат кортежа из forward-хука ПОДМЕНИЛ
выход нормы, и расхождение потом искали руками. Здесь функция именованная и
без return.

ХРАНЕНИЕ В FP16 — ЭТО ШУМ, И ОН ИЗМЕРЯЕТСЯ. После записи кэш прогоняется
обратно через ту же голову, и доля разошедшихся токенов печатается. Без этого
числа таблица из четырёх клеток читалась бы так, будто у неё нет собственной
погрешности.

Запуск:
    python3 experiments/k9e_extract_h12.py --selftest

    # исходный ствол
    python3 experiments/k9e_extract_h12.py --ckpt <base> \\
        --cache data/k9_teacher_150k.npz --out data/k9e_orig

    # обученный ствол (эпоха 3)
    python3 experiments/k9e_extract_h12.py --ckpt <base> \\
        --cache data/k9_teacher_150k.npz \\
        --joint-ckpt data/k9d_ep3.pt --out data/k9e_ep3
"""

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8


def plan(n, batch):
    """Границы батчей. Отдельной функцией только ради самопроверки."""
    return [(i, min(i + batch, n)) for i in range(0, n, batch)]


def selftest():
    b = plan(10, 4)
    assert b == [(0, 4), (4, 8), (8, 10)], b
    assert sum(j - i for i, j in b) == 10
    assert plan(0, 4) == []

    # РАЗБИЕНИЕ СТВОЛ/ГОЛОВА. Ошибиться здесь значит получить таблицу, в
    # которой вклад приписан не тому. Проверяется явно, а не комментарием.
    TRUNK = ("vlm.text_model.layers.", "action_expert.layers.", "bos_embedding")
    HEAD = ("action_expert.norm.", "fast_head.")
    for nm in ("bos_embedding", "action_expert.layers.3.mlp.up_proj.weight",
               "vlm.text_model.layers.0.input_layernorm.weight"):
        assert any(nm.startswith(p) or nm == p for p in TRUNK), nm
        assert not any(nm.startswith(p) for p in HEAD), nm
    for nm in ("action_expert.norm.weight", "fast_head.weight",
               "fast_head.bias"):
        assert any(nm.startswith(p) for p in HEAD), nm
        assert not any(nm.startswith(p) or nm == p for p in TRUNK), nm

    # Хук обязан быть БЕЗ возврата: forward-хук, вернувший значение, подменяет
    # выход модуля. Проверяется на настоящем torch, если он есть.
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("самопроверка k9e пройдена частично (torch недоступен): "
              "разбивка батчей и граница ствол/голова")
        return
    lin = nn.Linear(4, 4)
    cap = []

    def grab(mod, inp, out):          # НИЧЕГО НЕ ВОЗВРАЩАЕТ
        cap.append(inp[0].detach().clone())

    h = lin.register_forward_hook(grab)
    x = torch.randn(2, 4)
    y = lin(x)
    h.remove()
    assert torch.equal(cap[0], x), "хук взял не вход"
    assert torch.equal(y, lin(x)), "хук подменил выход"

    # fp16 как хранилище: величина шума должна быть относительной, а не
    # абсолютной, иначе порог не переносится между слоями.
    v = torch.randn(1000, 768) * 3.2
    rel = ((v.half().float() - v).abs() / v.abs().clamp(min=1e-6)).mean()
    assert rel < 1e-2, rel
    print("самопроверка k9e пройдена (версия «h12 = вход нормы»): разбивка "
          "батчей, граница ствол/голова, хук без подмены выхода, шум fp16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--joint-ckpt", default=None,
                    help="если задан — ствол берётся обученный; иначе исходный")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--check-batches", type=int, default=-1,
                    help="сколько батчей сверить кэш против живого прохода; "
                         "-1 значит ВСЕ. Проверка стоит одну норму и одну "
                         "линейную голову на батч — против двенадцати слоёв "
                         "трансформера это ничто, а четыре батча из начала "
                         "первой группы офсета покрывали 0.17% токенов.")
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt or not args.out:
        raise SystemExit("нужны --ckpt и --out (или --selftest)")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k9e sha1 {sha}")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    import joint12_vla as jv
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, process_state, prompt_template,
                       seed_everything)

    seed_everything(0)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    # --- кэш учителя: нужны только индексы и метаданные -----------------------
    z = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    if meta["ckpt"] != args.ckpt:
        raise SystemExit(f"кэш собран на {meta['ckpt']}, а задан {args.ckpt}")
    epi, stp, tsk = z["episode"], z["step"], z["task"]
    offs, split = z["pos_offset"], z["split"]
    N = len(epi)
    print(f"кэш: {N} наблюдений, {len(np.unique(epi))} эпизодов")

    # --- кадры и состояния: тот же путь, что в k9c ----------------------------
    img_path = args.cache + ".images.npy"
    IMG = None
    if os.path.exists(img_path):
        IMG = np.load(img_path, mmap_mode="r")
        assert IMG.shape[0] == N, (IMG.shape, N)
        print(f"кадры из memmap: {IMG.shape}, {IMG.nbytes / 2**30:.1f} ГиБ")
    else:
        raise SystemExit(
            f"нет {img_path}. Сборка кадров из parquet в оперативную память "
            f"на {N} наблюдений уже убивала прогон; пересоберите кэш с "
            f"кадрами.")
    st = None
    rid, rev = "physical-intelligence/libero", "v2.0"
    uniq = np.unique(epi)
    for j, e in enumerate(uniq):
        f = hf_hub_download(rid, f"data/chunk-{int(e) // 1000:03d}/"
                            f"episode_{int(e):06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        if st is None:
            st = np.zeros((N, S_.shape[1]), np.float64)
        elif st.shape[1] != S_.shape[1]:
            raise SystemExit(f"эпизод {e}: состояние {S_.shape[1]}-мерное, "
                             f"а раньше было {st.shape[1]}-мерное")
        for r in np.where(epi == e)[0]:
            st[r] = S_[int(stp[r])]
        if j % 400 == 0:
            print(f"  эпизодов {j}/{len(uniq)}", flush=True)
    if st.shape[1] == len(STATE_Q01) + 1:
        st = process_state(st)
    st_n = (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    print("состояния собраны")

    # --- модель ---------------------------------------------------------------
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=args.depth, head_dtype=dt)

    trunk, weights_sha = "original", None
    if args.joint_ckpt:
        h = hashlib.sha1()
        with open(args.joint_ckpt, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        weights_sha = h.hexdigest()[:12]
        obj = torch.load(args.joint_ckpt, map_location="cpu",
                         weights_only=False)
        if int(obj["depth"]) != args.depth:
            raise SystemExit(f"чекпойнт глубины {obj['depth']}, задано "
                             f"{args.depth}")
        # ЗАГРУЗКА СТРОГАЯ, как в K-9d. Частично применённый чекпойнт даёт
        # правдоподобные, но неверные числа, и таблица этого не покажет.
        state = obj["state"]
        stray = [k for k in state
                 if not any(k.startswith(p) or k == p.rstrip(".")
                            for p in model.trainable_prefixes)]
        if stray:
            raise SystemExit(f"{len(stray)} ключей вне белого списка: "
                             f"{stray[:5]}")
        own = dict(model.named_parameters())
        missing = [k for k in own if own[k].requires_grad and k not in state]
        if missing:
            raise SystemExit(f"нет {len(missing)} обучаемых весов: "
                             f"{missing[:5]}")
        with torch.no_grad():
            for k, v in state.items():
                if tuple(own[k].shape) != tuple(v.shape):
                    raise SystemExit(f"форма {k}: {tuple(own[k].shape)} "
                                     f"против {tuple(v.shape)}")
                if not torch.isfinite(v).all():
                    raise SystemExit(f"в {k} есть nan или inf")
                own[k].data = v.to(dev, torch.float32)
        ca = (obj.get("args") or {}).get("cache")
        if ca and os.path.abspath(ca) != os.path.abspath(args.cache):
            print(f"  ВНИМАНИЕ: чекпойнт обучен на кэше {ca}, а h12 снимается "
                  f"с {args.cache}")
        trunk = "trained"
        print(f"ствол обученный: {args.joint_ckpt}, веса sha {weights_sha}, "
              f"тензоров {len(state)}")
    else:
        print("ствол исходный, из BAR")

    # РЕЖИМ ВЫЧИСЛЕНИЯ ОДИНАКОВ ДЛЯ ОБОИХ СТВОЛОВ И СОВПАДАЕТ С K-9c: там
    # перед оценкой эпохи 0 вызывался to_fp32_trainable, а проход шёл под
    # autocast fp16. Если исходный ствол считать в чистом fp16 без autocast,
    # первая клетка перестанет воспроизводить эпоху 0 буквально — и допуск
    # просто спрячет расхождение вместо того, чтобы его показать.
    n32 = model.to_fp32_trainable()
    use_ac = True
    print(f"режим как в K-9c: {n32} тензоров переведены в fp32, "
          f"проход под autocast {args.dtype}")
    model.eval()

    # --- хук на вход нормы ----------------------------------------------------
    cap = []

    def grab(mod, inp, out):          # НИЧЕГО НЕ ВОЗВРАЩАЕТ, см. докстроку
        cap.append(inp[0].detach())

    hk = model.action_expert.norm.register_forward_hook(grab)

    D = int(model.fast_head.in_features)
    out_h12 = args.out + ".h12.npy"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    need = N * N_POS * D * 2 / 2 ** 30
    # МЕСТО ПРОВЕРЯЕТСЯ ДО ЗАПИСИ. Диск заполнен, а извлечение идёт около часа;
    # падение на последнем батче стоило бы всего прогона.
    st_fs = os.statvfs(os.path.dirname(os.path.abspath(out_h12)) or ".")
    free = st_fs.f_bavail * st_fs.f_frsize / 2 ** 30
    print(f"h12: ({N}, {N_POS}, {D}) fp16 = {need:.2f} ГиБ -> {out_h12}; "
          f"свободно {free:.1f} ГиБ")
    if free < need * 1.1 + 1.0:
        raise SystemExit(
            f"на диске {free:.1f} ГиБ, нужно {need:.2f} ГиБ с запасом. "
            f"Освободите место или снимайте h12 только для val и test.")
    H = np.lib.format.open_memmap(out_h12, mode="w+", dtype=np.float16,
                                  shape=(N, N_POS, D))

    tfm = None
    if IMG is None:
        tfm = Compose([CenterCrop(int(256 * 0.875)), Resize(224)])

    def build(sel):
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for gi in sel:
            m = prompt_template(st_n[gi], None, str(tsk[gi]),
                                mode=cfg.MODEL.vla_processor.kwargs.mode,
                                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts,
                 images=[[image[k].numpy()] for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b)

    # ПОРЯДОК ПО ОФСЕТУ: position_offset задаётся на весь вызов, смешивать в
    # одном батче нельзя. Индексы группируются, а пишутся по своим строкам.
    groups = []
    for po in sorted({int(v) for v in offs}):
        ipo = np.where(offs == po)[0]
        for i, j in plan(len(ipo), args.batch):
            groups.append((po, ipo[i:j]))
    print(f"батчей {len(groups)} по {args.batch}")

    # ПОСЧИТАННЫЙ ШУМ РАЗБИТ ПО ЧАСТЯМ И ПО ОФСЕТАМ. Единственное среднее по
    # первым четырём батчам сказало бы только про начало первой группы; здесь
    # покрытие полное, и разбивка показывает, не сосредоточен ли шум в одной
    # части — валидация решает исход, и её погрешность нужна отдельно.
    n_mis, n_tok, checked = 0, 0, 0
    per_split = {s: [0, 0] for s in ("train", "val", "test")}
    per_off = {}
    worst = (0.0, None)
    done = 0
    for gi, (po, sel) in enumerate(groups):
        b = build(sel)
        cap.clear()
        with torch.no_grad(), torch.autocast(
                device_type=dev.type, dtype=dt, enabled=use_ac):
            v, p = model.build_inputs(position_offset=po, **b)
            o = model.forward_joint_fast(
                vlm_inputs_embeds=v, attention_mask=b.get("attention_mask"),
                position_ids=p)
        if len(cap) != 1:
            raise SystemExit(f"норма вызвана {len(cap)} раз вместо одного — "
                             f"путь изменился, кэш недействителен")
        h12 = cap[0]
        if tuple(h12.shape) != (len(sel), N_POS, D):
            raise SystemExit(f"h12 формы {tuple(h12.shape)}, ожидалось "
                             f"{(len(sel), N_POS, D)}")
        H[sel] = h12.float().cpu().numpy().astype(np.float16)

        # СВЕРКА КЭША ПРОТИВ ЖИВОГО ПРОХОДА. Считается на первых батчах: если
        # fp16-хранение сдвигает argmax, это обязано быть числом в отчёте, а
        # не молчаливым допущением.
        if args.check_batches < 0 or checked < args.check_batches:
            checked += 1
            back = torch.from_numpy(np.asarray(H[sel])).to(dev)
            with torch.no_grad(), torch.autocast(
                    device_type=dev.type, dtype=dt, enabled=use_ac):
                lg = model.fast_head(
                    model.action_expert.norm(back.to(h12.dtype)).to(
                        model.fast_head.weight.dtype))
            bad = (lg.argmax(-1) != o["pred_codes"])
            mis, tok = int(bad.sum()), int(bad.numel())
            n_mis += mis; n_tok += tok
            if tok and mis / tok > worst[0]:
                worst = (mis / tok, int(sel[0]))
            per_off.setdefault(po, [0, 0])
            per_off[po][0] += mis; per_off[po][1] += tok
            rows = bad.sum(-1).cpu().numpy()
            for s in per_split:
                m = split[sel] == s
                if m.any():
                    per_split[s][0] += int(rows[m].sum())
                    per_split[s][1] += int(m.sum()) * N_POS
        done += len(sel)
        if gi % 100 == 0:
            print(f"  {done}/{N} наблюдений", flush=True)

    hk.remove()
    H.flush()
    del H
    rate = (n_mis / n_tok) if n_tok else None
    if not n_tok:
        raise SystemExit("сверка не выполнялась — шум хранения неизвестен, "
                         "кэш непригоден для таблицы")
    cover = n_tok / (N * N_POS)
    print(f"\nсверка кэша против живого прохода: расходится {n_mis} токенов "
          f"из {n_tok} ({rate:.4%}), покрыто {cover:.1%} всех токенов")
    for s, (m, t) in per_split.items():
        if t:
            print(f"    {s:<6}{m / t:.4%}  ({m} из {t})")
    ow = max((v[0] / v[1], k) for k, v in per_off.items() if v[1])
    print(f"    худший офсет {ow[1]}: {ow[0]:.4%}; худший батч "
          f"{worst[0]:.4%} (строка {worst[1]})")
    if rate > 0.005:
        raise SystemExit(
            f"fp16-хранение сдвигает {rate:.2%} токенов — это соизмеримо с "
            f"различиями, которые таблица должна различать. Храните fp32.")

    # ГОЛОВА СОХРАНЯЕТСЯ ЦЕЛИКОМ МОДУЛЯМИ, а не state_dict: восстанавливать
    # норму по имени класса и eps значит завести ещё одно место, где можно
    # ошибиться незаметно. Среда одна и та же, pickle здесь уместен.
    import copy
    rd = args.out + ".readout.pt"
    torch.save(dict(norm=copy.deepcopy(model.action_expert.norm).float().cpu(),
                    head=copy.deepcopy(model.fast_head).float().cpu(),
                    trunk=trunk, depth=args.depth, dim=D,
                    norm_class=type(model.action_expert.norm).__name__), rd)

    md = dict(script_sha1=sha, joint12_vla_sha1=hashlib.sha1(
                  open(jv.__file__, "rb").read()).hexdigest()[:12],
              trunk=trunk, joint_ckpt=args.joint_ckpt,
              joint_weights_sha1=weights_sha, depth=args.depth,
              cache=os.path.abspath(args.cache), n=N, dim=D,
              h12_file=os.path.abspath(out_h12),
              readout_file=os.path.abspath(rd),
              cache_vs_live_token_mismatch=rate, checked_fraction=cover,
              mismatch_by_split={s: (v[0] / v[1] if v[1] else None)
                                 for s, v in per_split.items()},
              mismatch_worst_offset=[ow[1], ow[0]],
              mismatch_worst_batch=worst[0],
              ckpt=args.ckpt, argv=vars(args))
    json.dump(md, open(args.out + ".json", "w"), ensure_ascii=False, indent=1)
    print(f"сохранено: {out_h12}, {rd}, {args.out}.json")
    print(f"  ствол {trunk}, sha скрипта {sha}")


if __name__ == "__main__":
    main()
