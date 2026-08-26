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
import os
import sys

import numpy as np

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


def left_pad_into(dst, row0, block, lens, Lmax):
    """Положить батч в memmap с ЛЕВЫМ паддингом.

    Процессор дополняет слева, значимая часть прижата вправо. Кладём так же:
    позиция -1 всегда последний реальный токен независимо от длины подсказки.
    """
    b, l = block.shape[0], block.shape[1]
    assert l <= Lmax, f"батч длиннее выделенного: {l} > {Lmax}"
    dst[row0:row0 + b, Lmax - l:, :] = block
    if Lmax - l:
        dst[row0:row0 + b, :Lmax - l, :] = 0
    return b


def selftest():
    # 1. ЛЕВЫЙ ПАДДИНГ: значимое прижато вправо, нули слева.
    Lmax = 6
    dst = np.zeros((3, Lmax, 2), np.float16)
    left_pad_into(dst, 0, np.ones((2, 4, 2), np.float16), None, Lmax)
    left_pad_into(dst, 2, np.full((1, 6, 2), 3, np.float16), None, Lmax)
    assert (dst[0, :2] == 0).all() and (dst[0, 2:] == 1).all(), dst[0]
    assert (dst[2] == 3).all(), "полная длина не должна паддиться"

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
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".",
                exist_ok=True)
    Lmax = args.ctx_max
    mm, d_vlm = {}, None
    Hm = None
    K_bar = np.zeros((N, N_LEVEL, N_POS), np.int64)
    ctx_len = np.zeros(N, np.int32)
    row = {l: 0 for l in want_layers}
    order_idx = []

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
        order_idx.append(sel)
        for l in want_layers:
            assert grab_ctx[l], f"хук слоя {l} не сработал"
            c = grab_ctx[l][0].numpy()          # ПЕРВЫЙ блок, как и h
            if l not in mm:
                d_vlm = c.shape[-1]
                path = f"{args.out_prefix}_L{l:02d}_ctx.npy"
                mm[l] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float16, shape=(N, Lmax, d_vlm))
                print(f"  создан {path}: {(N * Lmax * d_vlm * 2) / 2**30:.1f} ГиБ")
            left_pad_into(mm[l], row[l], c, real_lens, Lmax)
            row[l] += b
        if done_cnt % (args.batch * 50) < args.batch:
            print(f"  {done_cnt}/{N} (офсет {po})", flush=True)
    assert done_cnt == N, f"обработано {done_cnt} из {N}"

    # ВНИМАНИЕ: memmap заполнялся в порядке обхода по офсетам, а не в порядке
    # наблюдений. Сохраняем перестановку, чтобы строка k в ctx соответствовала
    # наблюдению row_order[k].
    row_order = np.concatenate(order_idx)
    assert len(row_order) == N and len(np.unique(row_order)) == N

    for l in want_layers:
        mm[l].flush()
        a = np.asarray(mm[l][:200], np.float32)
        print(f"  слой {l:>2}: |avg| {np.abs(a).mean():7.3f}  "
              f"max {np.abs(a).max():9.1f}  sd {a.std():7.3f}")
        del mm[l]

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
             row_order=row_order,
             meta=json.dumps(dict(
                 ckpt=args.ckpt, n_obs=N, n_episodes=int(len(np.unique(epi))),
                 n_codes=n_codes, image_hw=int(hw), ctx_max=Lmax,
                 ctx_layers=want_layers, d_vlm=int(d_vlm),
                 ctx_source="вход input_layernorm указанных слоёв (единичная "
                            "нумерация), СЫРОЙ поток до нормализации",
                 bar_code_match=float((K_bar == K_true).mean())),
                 ensure_ascii=False))
    print(f"\n  сохранено: {base} + {len(want_layers)} файлов контекста")
    print("  ВАЖНО: строка k в *_ctx.npy соответствует наблюдению "
          "row_order[k], а не k. Порядок обхода шёл по офсетам.")


if __name__ == "__main__":
    main()
