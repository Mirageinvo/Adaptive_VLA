"""K-6i: снять состояния префикса с НЕСКОЛЬКИХ глубин VLM за один проход.

ЗАЧЕМ. K-6e закрыл линию уточнения по h: пять параметризаций, сходимость,
ничтожный разброс — ни одна не обошла «только грубый уровень». Но блоки 2-3 у
BAR ПЕРЕЧИТЫВАЮТ изображение, а наш уточнитель видел только сжатое состояние
экспертной башни. Пока доступ к зрительному контексту не проверен, утверждение
«информации нет» неполно.

И проверять надо не одну глубину. Гипотеза, ради которой всё: разные слои VLM
несут разную информацию, и разным параллельным компонентам действия могут быть
нужны разные. Одного финального состояния для проверки мало — отрицательный
результат на нём ничего не докажет, потому что у блоков 2-3 доступ послойный.

ПОЧЕМУ ОТДЕЛЬНЫЙ ФАЙЛ НА СЛОЙ. Четыре слоя при 12000 наблюдений это 33 ГиБ.
Класть в один архив плохо: обучая голову с доступом к ОДНОМУ слою, пришлось бы
поднимать в память все четыре. Каждый слой пишется в свой .npy ЧЕРЕЗ MEMMAP,
то есть на диск по батчам, а не через накопление 33 ГиБ в оперативной памяти.

СТРОКИ ВЫРОВНЕНЫ ВЕЗДЕ ОДИНАКОВО. Строка k в любом файле — это наблюдение k.
Обход идёт группами по офсету позиции, но запись ведётся по индексам, а не
подряд, поэтому перестановки нет. Прежняя версия хранила `row_order` и требовала
применять её вручную: формат, в котором забытая перестановка не падает, а даёт
правдоподобно обучающуюся модель на перепутанных изображениях.

НУМЕРАЦИЯ СЛОЁВ. `--ctx-layers 6,12,18,24` означает остаточный поток НА ВХОДЕ
слоя с этим номером (единичная индексация), то есть состояние после
предыдущих. Для последнего слоя это ровно то, что снимал k6d, поэтому данные
сопоставимы с прежними.

МАСШТАБЫ РАЗНЫХ ГЛУБИН РАЗЛИЧАЮТСЯ. Снимается вход input_layernorm, то есть
поток ДО нормализации: на последнем слое мы намеряли выбросы до 9888 против 8.3
у h. Сохраняем СЫРЫЕ значения и печатаем статистики по каждому слою, а
нормировка делается в голове (там LayerNorm на контексте). Иначе сравнение
глубин рисковало бы превратиться в сравнение масштабов.

ПРЕДОБРАБОТКА ДАТАСЕТНАЯ, НЕ СРЕДОВАЯ — три места, на каждом я ошибался
(FINDINGS §10): изображение БЕЗ разворота каналов, кроп 87.5% от ФАКТИЧЕСКОГО
размера, состояние уже 8-мерное и process_state не применяется.

Запуск:
    python3 experiments/k6i_extract_layers.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k6i_extract_layers.py --ckpt <ckpt> \\
        --n-obs 12000 --n-ep 1200 --ctx-layers 6,12,18,24 \\
        --out-prefix data/k6i_12k
"""

import argparse
import io
import json
import math
import os
import shutil
import sys

import numpy as np


def check_disk(outdir, need_bytes, slack=0.15):
    """Место проверяется ДО создания файлов. Иначе 37 ГиБ пишутся час и
    падают на последнем батче, оставляя мусор и не оставляя признаков."""
    free = shutil.disk_usage(outdir).free
    need = need_bytes * (1.0 + slack)
    if free < need:
        raise SystemExit(
            f"мало места в {outdir}: свободно {free / 2**30:.1f} ГиБ, "
            f"нужно {need / 2**30:.1f} ГиБ (данные {need_bytes / 2**30:.1f} "
            f"плюс {slack:.0%} запаса)")
    return free

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20


def find_vlm_layers(vlm):
    """Найти список слоёв башни, не угадывая путь: самый длинный ModuleList,
    элементы которого имеют input_layernorm."""
    import torch.nn as nn
    best = None
    for name, mod in vlm.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1 and \
                hasattr(mod[0], "input_layernorm"):
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise SystemExit("не нашёл слои VLM: нет ModuleList с input_layernorm")
    return best


def left_pad_into(dst, rows, block, Lmax):
    """Положить батч в memmap ПО ИНДЕКСАМ НАБЛЮДЕНИЙ, с левым паддингом.

    ПОЧЕМУ ПО ИНДЕКСАМ, А НЕ ПОДРЯД. Первая версия писала строки в порядке
    обхода (группами по офсету позиции) и хранила перестановку `row_order`,
    которую потребитель обязан был применить. Формат хрупкий до
    неприемлемого: одна забытая перестановка не падает, а даёт правдоподобно
    обучающуюся модель на перепутанных изображениях. Теперь строка k — это
    наблюдение k во всех массивах разом, и перестановки нет вовсе.

    Паддинг левый, как у процессора: значимая часть прижата вправо, поэтому
    позиция -1 всегда последний реальный токен независимо от длины подсказки.
    """
    b, l = block.shape[0], block.shape[1]
    assert l <= Lmax, (f"префикс длиннее выделенного: {l} > {Lmax}. "
                       f"Поднимите --ctx-max.")
    pad = np.zeros((b, Lmax, block.shape[2]), block.dtype)
    pad[:, Lmax - l:, :] = block
    dst[rows] = pad
    return b


def selftest():
    # 1. ЛЕВЫЙ ПАДДИНГ: значимое прижато вправо, нули слева.
    Lmax = 6
    dst = np.zeros((3, Lmax, 2), np.float16)
    left_pad_into(dst, np.array([0, 1]), np.ones((2, 4, 2), np.float16), Lmax)
    left_pad_into(dst, np.array([2]), np.full((1, 6, 2), 3, np.float16), Lmax)
    assert (dst[0, :2] == 0).all() and (dst[0, 2:] == 1).all(), dst[0]
    assert (dst[2] == 3).all(), "полная длина не должна паддиться"

    # 1б. ЗАПИСЬ ПО ИНДЕКСАМ, а не подряд. Главная защита: строка k обязана
    # быть наблюдением k при ЛЮБОМ порядке обхода. Имитируем обход группами
    # по офсету и проверяем, что каждая строка получила своё значение.
    N = 7
    dst = np.zeros((N, 2, 1), np.float16)
    offs = np.array([1, 0, 1, 0, 1, 0, 0])
    for po in (0, 1):
        sel = np.where(offs == po)[0]
        blk = np.stack([np.full((2, 1), float(j)) for j in sel]).astype(np.float16)
        left_pad_into(dst, sel, blk, 2)
    assert (dst[:, 0, 0] == np.arange(N)).all(), (
        f"строка не совпала с наблюдением: {dst[:, 0, 0]}")

    # 2. НУМЕРАЦИЯ СЛОЁВ единичная: слой 24 из 24 — последний.
    for total, want in ((24, [5, 11, 17, 23]),):
        got = [l - 1 for l in (6, 12, 18, 24)]
        assert got == want and max(got) == total - 1, got

    # 3. Раскладка кодов поуровневая, а не перемежением по времени.
    K = np.arange(N_POS * N_LEVEL)[None].reshape(1, N_LEVEL, N_POS)
    assert (K[0, 0] == np.arange(0, 16)).all()
    assert (K[0, 2] == np.arange(32, 48)).all()

    print("самопроверка пройдена: левый паддинг прижимает значимое вправо, "
          "нумерация слоёв единичная, раскладка кодов поуровневая")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-obs", type=int, default=12000)
    ap.add_argument("--n-ep", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx-layers", default="6,12,18,24",
                    help="номера слоёв (с единицы); снимается вход "
                         "input_layernorm каждого")
    ap.add_argument("--ctx-max", type=int, default=200,
                    help="верхняя граница длины префикса; при превышении "
                         "скрипт падает, а не пишет мусор")
    ap.add_argument("--offset-table", default="data/pos_offset_table.json")
    ap.add_argument("--pos-offset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="data/k6i_12k")
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
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01,  # noqa: E402
                       STATE_Q99, VisionLanguageActionProcessor, dict_apply,
                       get_cfg, process_state, prompt_template, seed_everything)

    seed_everything(args.seed)
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt
    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))
    n_codes = int(cfg.MODEL.action_processor.vocab_size)
    want_layers = [int(v) for v in args.ctx_layers.split(",")]

    off_by_task = None
    if args.pos_offset is None:
        if not os.path.exists(args.offset_table):
            raise SystemExit(f"нет {args.offset_table}; задайте --pos-offset "
                             f"(единый офсет — АБЛЯЦИЯ, а не протокол)")
        tb = json.load(open(args.offset_table))
        off_by_task = {k: int(v["pos_offset"]) for k, v in tb["tasks"].items()}

    model = SmolVLABlockwiseAR.from_pretrained(
        **cfg.MODEL.vlm.kwargs).to(dev, dtype).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    grab_h, grab_ctx = [], {}
    model.action_lm_head.register_forward_hook(
        lambda m, i, o: grab_h.append(i[0].detach().float().cpu()))
    lname, layers = find_vlm_layers(model.vlm)
    print(f"слои VLM: {lname}, всего {len(layers)}")
    for l in want_layers:
        if not 1 <= l <= len(layers):
            raise SystemExit(f"слой {l} вне диапазона 1..{len(layers)}")
        grab_ctx[l] = []
        layers[l - 1].input_layernorm.register_forward_hook(
            (lambda ll: (lambda m, i, o:
                         grab_ctx[ll].append(i[0].detach().to(torch.float16).cpu())))(l))
    print(f"снимаю вход input_layernorm слоёв: {want_layers}")

    # --- данные ---------------------------------------------------------------
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
    print(f"собрано {N} наблюдений, {len(np.unique(epi))} эпизодов, кадр {hw}x{hw}")

    tf = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])   # ДОЛЯ, не число

    ST_RAW = np.asarray(st, np.float64)
    if ST_RAW.shape[1] == len(STATE_Q01) + 1:
        ST_RAW = process_state(ST_RAW)
    assert ST_RAW.shape[1] == len(STATE_Q01), ST_RAW.shape
    st_n = (ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0

    a_codec = np.asarray(act, np.float64).copy()
    a_codec[..., :-1] /= max_act_q[..., :-1]
    a_codec[..., -1] *= -1
    a_codec = np.clip(a_codec, -1.0, 1.0)
    K_true = np.asarray(proc.action_processor.encode(a_codec),
                        np.int64).reshape(N, N_LEVEL, N_POS)

    if args.pos_offset is None:
        miss = sorted({t for t in tsk if t not in off_by_task})
        if miss:
            raise SystemExit(f"в таблице офсетов нет {len(miss)} задач: {miss[:3]}")
    offs = np.array([args.pos_offset if args.pos_offset is not None
                     else off_by_task[tsk[i]] for i in range(N)])

    # --- memmap на диск, по файлу на слой ------------------------------------
    outdir = os.path.dirname(os.path.abspath(args.out_prefix)) or "."
    os.makedirs(outdir, exist_ok=True)
    Lmax = args.ctx_max
    d_guess = int(layers[0].input_layernorm.weight.shape[0])
    need = N * Lmax * d_guess * 2 * len(want_layers) + N * N_POS * 768 * 2
    free = check_disk(outdir, need)
    print(f"место в {outdir}: нужно {need / 2**30:.1f} ГиБ "
          f"(+15% запаса), свободно {free / 2**30:.1f} ГиБ")
    mm, paths, d_vlm = {}, {}, None
    Hm = None
    K_bar = np.zeros((N, N_LEVEL, N_POS), np.int64)
    ctx_len = np.zeros(N, np.int32)
    # ПОТОКОВАЯ СТАТИСТИКА ПО ВСЕМ СТРОКАМ. Считать её по первым 200 значит не
    # увидеть ни выброс, ни NaN в остальных 11800 — а NaN в признаках даёт
    # обучение, которое молча не сходится.
    stat = {l: dict(n=0, s=0.0, s2=0.0, mx=0.0, nonfinite=0) for l in want_layers}

    done_cnt = 0
    for po in sorted({int(v) for v in offs}):
      idx_po = np.where(offs == po)[0]
      for i0 in range(0, len(idx_po), args.batch):
        sel = idx_po[i0:i0 + args.batch]
        b = len(sel)
        done_cnt += b
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
        batch = proc(text=texts, images=[[image[j].numpy()] for j in range(b)],
                     return_tensors="pt", padding=True, padding_side="left",
                     action_processor_kwargs={"embodiment_ids": 0})
        # ДЛИНЫ ДО ПЕРЕНОСА НА GPU: иначе в ctx_len попадёт длина всего батча,
        # и перекрёстное внимание сочтёт паддинг настоящим контекстом.
        real_lens = batch["attention_mask"].sum(dim=1).numpy().astype(np.int32)
        batch = dict_apply(lambda x: x.to(dev, dtype), batch)
        grab_h.clear()
        for l in want_layers:
            grab_ctx[l].clear()
        with torch.no_grad():
            tk = model.generate(**batch, position_offset=po, do_sample=False,
                                initial_position_shift=1)
        assert len(grab_h) == N_LEVEL, f"голова сработала {len(grab_h)} раз"
        g0 = grab_h[0]
        assert g0.shape[1] == N_POS, (
            f"на первом блоке ждали {N_POS} позиций, получено {g0.shape[1]}")
        if Hm is None:
            Hm = np.zeros((N, N_POS, g0.shape[-1]), np.float16)
        Hm[sel] = g0.numpy().astype(np.float16)
        K_bar[sel] = tk.cpu().numpy().reshape(b, N_LEVEL, N_POS)
        ctx_len[sel] = real_lens
        for l in want_layers:
            # ХУК ОБЯЗАН СРАБОТАТЬ РОВНО ПО РАЗУ НА БЛОК. Если срабатываний не
            # три, значит взят не тот модуль (или generate устроен иначе), и
            # брать [0] вслепую нельзя.
            assert len(grab_ctx[l]) == N_LEVEL, (
                f"хук слоя {l} сработал {len(grab_ctx[l])} раз, ждали {N_LEVEL}")
            c = grab_ctx[l][0].numpy()          # ПЕРВЫЙ блок, как и h
            if l not in mm:
                d_vlm = c.shape[-1]
                paths[l] = f"{args.out_prefix}_L{l:02d}_ctx.npy"
                # ПИШЕМ В .partial: упавший прогон иначе оставит внешне
                # валидный .npy, заполненный лишь наполовину, и это невозможно
                # отличить от целого файла.
                mm[l] = np.lib.format.open_memmap(
                    paths[l] + ".partial", mode="w+", dtype=np.float16,
                    shape=(N, Lmax, d_vlm))
                print(f"  создан {paths[l]}.partial: "
                      f"{(N * Lmax * d_vlm * 2) / 2**30:.1f} ГиБ")
            left_pad_into(mm[l], sel, c, Lmax)
            a = c.astype(np.float32)
            fin = np.isfinite(a)
            st_ = stat[l]
            st_["nonfinite"] += int(fin.size - fin.sum())
            af = np.where(fin, a, 0.0)
            st_["n"] += af.size
            st_["s"] += float(af.sum())
            st_["s2"] += float((af * af).sum())
            st_["mx"] = max(st_["mx"], float(np.abs(af).max()))
        # ПРЕФИКС ОДИНАКОВ ВО ВСЕХ ТРЁХ БЛОКАХ — проверяем на первом батче.
        # Это проверяет и правильность хука, и предпосылку кэшируемости, на
        # которой держится весь выигрыш в 1.31x.
        if done_cnt <= args.batch:
            for l in want_layers:
                d01 = float(np.abs(grab_ctx[l][0].numpy().astype(np.float32)
                                   - grab_ctx[l][1].numpy().astype(np.float32)).max())
                print(f"    слой {l:>2}: |префикс блока 1 − блока 2| = {d01:.3e}")
                if d01 > 1e-2:
                    raise SystemExit(
                        f"префикс слоя {l} различается между блоками на {d01:.3e}. "
                        f"Либо хук ловит не префикс, либо контекст не кэшируем — "
                        f"в обоих случаях сохранять признаки бессмысленно.")
        if done_cnt % (args.batch * 50) < args.batch:
            print(f"  {done_cnt}/{N} (офсет {po})", flush=True)
    assert done_cnt == N, f"обработано {done_cnt} из {N}"

    for l in want_layers:
        mm[l].flush()
        del mm[l]
        st_ = stat[l]
        mean = st_["s"] / st_["n"]
        sd = math.sqrt(max(st_["s2"] / st_["n"] - mean * mean, 0.0))
        print(f"  слой {l:>2}: sd {sd:8.3f}  max|x| {st_['mx']:9.1f}  "
              f"не-конечных {st_['nonfinite']}")
        if st_["nonfinite"]:
            raise SystemExit(
                f"в слое {l} {st_['nonfinite']} не-конечных значений — "
                f"обучение на этом молча не сойдётся")
        os.replace(paths[l] + ".partial", paths[l])

    assert ctx_len.max() > ctx_len.min(), \
        "все длины контекста одинаковы — вероятно записана длина батча"
    print(f"значимых токенов {ctx_len.min()}–{ctx_len.max()} "
          f"(медиана {int(np.median(ctx_len))}) из {Lmax}")
    print(f"h: {Hm.shape}; совпадение кодов BAR с истинными: "
          f"{(K_bar == K_true).mean():.1%}")

    base = f"{args.out_prefix}_base.npz"
    np.savez(base, h=Hm, K_true=K_true, K_bar=K_bar,
             act=a_codec.astype(np.float32), episode=epi,
             task=np.asarray(tsk), pos_offset=offs, ctx_len=ctx_len,
             meta=json.dumps(dict(
                 ckpt=args.ckpt, n_obs=N, n_episodes=int(len(np.unique(epi))),
                 n_codes=n_codes, image_hw=int(hw), ctx_max=Lmax,
                 ctx_layers=want_layers, d_vlm=int(d_vlm),
                 ctx_source="вход input_layernorm указанных слоёв (единичная "
                            "нумерация), СЫРОЙ поток до нормализации",
                 bar_code_match=float((K_bar == K_true).mean()),
                 layer_stats={str(l): dict(
                     sd=math.sqrt(max(stat[l]["s2"] / stat[l]["n"]
                                      - (stat[l]["s"] / stat[l]["n"]) ** 2, 0.0)),
                     max_abs=stat[l]["mx"]) for l in want_layers}),
                 ensure_ascii=False))
    print(f"\n  сохранено: {base} + {len(want_layers)} файлов контекста")
    print("  Строка k в *_ctx.npy — это наблюдение k во ВСЕХ массивах. "
          "Перестановки нет.")


if __name__ == "__main__":
    main()
