"""K-5a: сколько стоит трёхкратный пересчёт префикса VLM в BAR.

ВОПРОС. §7б замерил, что три блочных прохода BAR занимают 198.9 мс из 273.3
сквозных, по 66 мс на проход, и что подгонка даёт t(q) = 62.39 + 0.2149*q. То
есть 62.4 мс каждого прохода — постоянная часть, не зависящая от числа позиций
действия. Оценка «убрать два лишних пересчёта = 1.84x» получена из этой
подгонки арифметически и НИКОГДА не измерялась напрямую.

ЧТО ГОВОРИТ КОД. `bar.py:902` ставит -inf на блок «запросы VLM -> ключи
действия», а строка 907 делает VLM causal внутри себя. Значит скрытые состояния
VLM на каждом слое зависят ТОЛЬКО от vlm_inputs_embeds, маски паддинга и
позиционных id префикса. Все три блочных прохода получают их одинаковыми,
поэтому вся башня VLM пересчитывается трижды — не только K/V, а вместе с MLP.

ЗОНД ДЕЛАЕТ ТРИ ВЕЩИ, в порядке возрастания цены ошибки.

1. НЕЗАВИСИМОСТЬ (условие корректности кэша). Прогоняет проход дважды с РАЗНЫМИ
   токенами действия и сверяет скрытые состояния VLM послойно. Если они хоть
   где-то различаются, кэшировать префикс нельзя, и всё остальное не имеет
   смысла. Проверяется эмпирически, а не только чтением маски: маску легко
   прочитать неверно.

2. ЦЕНА ПРОХОДА. Меряет три реальных блочных прохода (длина истории 0, 16, 32)
   и отдельно — прототип прохода с кэшированным префиксом, где q/k/v и MLP
   башни VLM не вычисляются вовсе, а внимание считается только по запросам
   действия против сохранённых ключей. Разность даёт ИЗМЕРЕННЫЙ приз вместо
   арифметической оценки.

3. ЭКВИВАЛЕНТНОСТЬ ПРОТОТИПА. Сверяет логиты и top-1 токены прототипа с
   опорной реализацией. Побитового совпадения не будет: формы матриц другие,
   значит другие ядра и другой порядок суммирования. Требование — совпадение
   top-1 и разность логитов в пределах допуска dtype.

ДАННЫЕ НЕ НУЖНЫ. Время и независимость определяются формами и схемой внимания,
а не содержанием, поэтому vlm_inputs_embeds синтезируется случайным тензором
нужной формы. Для проверки эквивалентности этого тоже достаточно: сравниваются
два вычисления на ОДНОМ И ТОМ ЖЕ входе.

Запуск:
    python3 experiments/k5a_prefix_cost_probe.py --selftest
    python3 experiments/k5a_prefix_cost_probe.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO --out data/k5a.json
"""

import argparse
import json
import os
import statistics
import sys
import time


def selftest():
    """Алгебра, на которой держится весь кэш, с ИЗВЕСТНЫМ ОТВЕТОМ.

    Утверждение: внимание, посчитанное только для ПОДМНОЖЕСТВА запросов, равно
    соответствующим строкам полного внимания — при условии, что ключи, значения
    и строки маски взяты те же. Softmax нормируется по ключам, поэтому строки
    независимы, и урезание запросов ничего не меняет.

    Если это неверно, кэш префикса неверен в принципе. Проверяем численно, а не
    рассуждением, и заодно ловим ошибку в срезе маски — самое лёгкое место, где
    можно перепутать оси.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    B, H, Lv, La, D = 2, 4, 7, 5, 8
    T = Lv + La
    q = rng.normal(size=(B, H, T, D))
    k = rng.normal(size=(B, H, T, D))
    v = rng.normal(size=(B, H, T, D))
    m = np.zeros((B, 1, T, T))
    m[:, :, :Lv, Lv:] = -1e9                      # VLM не видит действия
    m[:, :, :Lv, :Lv] = np.where(
        np.triu(np.ones((Lv, Lv)), 1) > 0, -1e9, 0.0)

    def attn(qq, kk, vv, mm):
        w = qq @ kk.transpose(0, 1, 3, 2) / np.sqrt(D) + mm
        w = np.exp(w - w.max(-1, keepdims=True))
        return (w / w.sum(-1, keepdims=True)) @ vv

    full = attn(q, k, v, m)
    part = attn(q[:, :, Lv:], k, v, m[:, :, Lv:])
    assert np.allclose(full[:, :, Lv:], part, atol=1e-10), \
        "внимание по подмножеству запросов разошлось с полным — срез неверен"

    # И обратное: если бы VLM ВИДЕЛА действия, её выход зависел бы от них,
    # то есть кэш префикса был бы неверен. Проверяем, что тест это ловит.
    m_bad = m.copy()
    m_bad[:, :, :Lv, Lv:] = 0.0
    v2 = v.copy()
    v2[:, :, Lv:] += 5.0                          # меняем ТОЛЬКО действия
    a1 = attn(q[:, :, :Lv], k, v, m_bad[:, :, :Lv])
    a2 = attn(q[:, :, :Lv], k, v2, m_bad[:, :, :Lv])
    assert not np.allclose(a1, a2), \
        "проверка на зависимость VLM от действий не срабатывает"
    a1 = attn(q[:, :, :Lv], k, v, m[:, :, :Lv])
    a2 = attn(q[:, :, :Lv], k, v2, m[:, :, :Lv])
    assert np.allclose(a1, a2, atol=1e-10), \
        "при верной маске выход VLM обязан не зависеть от действий"
    print("самопроверка пройдена: срез запросов эквивалентен полному вниманию; "
          "при маске BAR выход VLM не зависит от токенов действия, и тест "
          "ловит обратный случай")


def bench(fn, warmup, iters, dev):
    import torch
    for _ in range(warmup):
        fn()
    ts = []
    if dev.type == "cuda":
        torch.cuda.synchronize()
        for _ in range(iters):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    g = lambda p: ts[min(len(ts) - 1, int(p * len(ts)))]
    return dict(p50=statistics.median(ts), p90=g(0.90), p99=g(0.99),
                mean=statistics.fmean(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--token-budget", type=int, default=48,
                    help="как в builder: P*L, по умолчанию 16*3")
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--action-vocab", type=int, default=2048)
    ap.add_argument("--vlen", type=int, default=168,
                    help="длина префикса VLM; 168 — измеренная в §7б")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")

    import torch
    sys.path.insert(0, os.path.abspath(args.root))
    from transformers.models.llama.modeling_llama import (  # noqa: E402
        apply_rotary_pos_emb, repeat_kv)
    from smolvla.bar import SmolVLABlockwiseAR  # noqa: E402

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    if dev.type == "cuda":
        cc = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]}), "
              f"torch {torch.__version__}, CUDA {torch.version.cuda}")
        if args.dtype == "bfloat16" and cc[0] < 8:
            print("  ВНИМАНИЕ: bfloat16 без аппаратной поддержки — числа "
                  "не переносимы, используйте float16")
    # ЗАГРУЗКА РОВНО КАК В BUILDER, включая обязательные kwargs. В логе
    # builder значится «dtype модели по факту: torch.float32 (запрошен
    # bfloat16)» — то есть from_pretrained запрошенный dtype ИГНОРИРУЕТ. Для
    # замера времени это решающее обстоятельство, поэтому dtype приводится
    # принудительно, а фактический печатается.
    model = SmolVLABlockwiseAR.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=dtype,
        token_budget=args.token_budget, num_blocks=args.num_blocks,
        action_vocab_size=args.action_vocab).to(dev).eval()
    got = next(model.parameters()).dtype
    if got != dtype:
        print(f"  from_pretrained вернул {got} вместо {dtype} — привожу "
              f"принудительно")
        model = model.to(dtype)
    print(f"  dtype модели по факту: {next(model.parameters()).dtype}")
    n = model.block_size
    cfg = model.vlm.text_model.config
    nL = cfg.num_hidden_layers
    hid = cfg.hidden_size
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nh)
    hd = hid // nh
    B, Lv = args.batch, args.vlen
    print(f"блок {n} токенов, слоёв {nL}, hidden {hid}, голов {nh}/{nkv}, "
          f"префикс {Lv}, batch {B}, dtype {args.dtype}")

    torch.manual_seed(0)
    VLM = torch.randn(B, Lv, hid, device=dev, dtype=dtype) * 0.02
    amask = torch.ones(B, Lv, device=dev, dtype=torch.long)

    def pos_ids(alen, offset=4):
        apos = model._build_action_pos_ids_strided(
            batch_size=B, base_pos=Lv, action_seq_len=alen, device=dev,
            position_offset=offset)
        return model._build_joint_position_ids(
            batch_size=B, vlm_seq_len=Lv, action_pos_ids=apos, device=dev)

    # ---- 1. НЕЗАВИСИМОСТЬ БАШНИ VLM ОТ ТОКЕНОВ ДЕЙСТВИЯ ------------------
    # Оборачиваем метод, чтобы записать скрытые состояния VLM послойно. Третий
    # party не трогаем: обёртка живёт только в этом процессе.
    orig = model._shared_attention_forward
    trace = []

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        trace.append(out[0].detach().float().cpu())
        return out
    model._shared_attention_forward = wrapped

    hist_a = torch.randint(0, model.action_vocab_size, (B, n), device=dev)
    hist_b = torch.randint(0, model.action_vocab_size, (B, n), device=dev)
    assert not torch.equal(hist_a, hist_b), "истории совпали, тест бессмысленен"
    with torch.no_grad():
        trace.clear()
        model._run_action_sequence(vlm_inputs_embeds=VLM, attention_mask=amask,
                                   bos_len=n, action_input_ids=hist_a,
                                   position_ids=pos_ids(n + n))
        tr_a = list(trace)
        trace.clear()
        model._run_action_sequence(vlm_inputs_embeds=VLM, attention_mask=amask,
                                   bos_len=n, action_input_ids=hist_b,
                                   position_ids=pos_ids(n + n))
        tr_b = list(trace)
    model._shared_attention_forward = orig

    print("\n" + "=" * 74)
    print("1. НЕЗАВИСИМОСТЬ БАШНИ VLM ОТ ТОКЕНОВ ДЕЙСТВИЯ")
    print("=" * 74)
    assert len(tr_a) == len(tr_b) == nL, f"слоёв записано {len(tr_a)} из {nL}"
    worst = max(float((x - y).abs().max()) for x, y in zip(tr_a, tr_b))
    print(f"  слоёв сверено: {nL}, максимум расхождения: {worst:.3e}")
    assert worst == 0.0, (
        "скрытые состояния VLM ЗАВИСЯТ от токенов действия — кэшировать "
        "префикс нельзя, дальше идти нет смысла")
    print("  побитово идентичны: префикс кэшируем")

    # ---- 2. ЦЕНА РЕАЛЬНЫХ ПРОХОДОВ ---------------------------------------
    print("\n" + "=" * 74)
    print("2. ЦЕНА БЛОЧНЫХ ПРОХОДОВ (опорная реализация)")
    print("=" * 74)
    ref = {}
    for k_blocks in (0, 1, 2):
        hist = (torch.empty((B, 0), dtype=torch.long, device=dev) if k_blocks == 0
                else torch.randint(0, model.action_vocab_size,
                                   (B, k_blocks * n), device=dev))
        alen = n + k_blocks * n
        pid = pos_ids(alen)

        def fn(h=hist, p=pid):
            with torch.no_grad():
                model._run_action_sequence(
                    vlm_inputs_embeds=VLM, attention_mask=amask, bos_len=n,
                    action_input_ids=h, position_ids=p)
        ref[k_blocks] = bench(fn, args.warmup, args.iters, dev)
        print(f"  блок {k_blocks + 1}, действий {alen:>2}: "
              f"p50 {ref[k_blocks]['p50']:.2f} мс, p90 "
              f"{ref[k_blocks]['p90']:.2f}, p99 {ref[k_blocks]['p99']:.2f}")
    total_ref = sum(ref[i]["p50"] for i in (0, 1, 2))
    print(f"  три прохода суммарно: {total_ref:.2f} мс")

    # ---- 3. ПРОТОТИП С КЭШЕМ ПРЕФИКСА ------------------------------------
    # Пасс 1 идёт как обычно и попутно сохраняет послойные k/v префикса УЖЕ С
    # RoPE. Пассы 2 и 3 не считают ни q/k/v, ни MLP башни VLM: запросы только
    # от токенов действия, ключи — сохранённые плюс собственные.
    def prefill():
        cache = []
        with torch.no_grad():
            vh = VLM
            alen = n
            pid = pos_ids(alen)
            m4 = model._build_joint_attention_mask_blockwise_ar(
                attention_mask=amask, vlm_seq_len=Lv, action_seq_len=alen,
                device=dev,
                action_key_mask=torch.ones(B, alen, device=dev,
                                           dtype=torch.long))
            ah = model.bos_embedding.expand(B, n, -1).to(dev, dtype)
            for li in range(nL):
                vl = model.vlm.text_model.layers[li]
                vn = vl.input_layernorm(vh)
                vk = vl.self_attn.k_proj(vn).view(B, Lv, nkv, hd).transpose(1, 2)
                vv = vl.self_attn.v_proj(vn).view(B, Lv, nkv, hd).transpose(1, 2)
                dummy = torch.empty((B, Lv + alen, hid), device=dev, dtype=dtype)
                cos, sin = model.vlm.text_model.rotary_emb(dummy, position_ids=pid)
                _, vk_r = apply_rotary_pos_emb(vk, vk, cos[:, :Lv], sin[:, :Lv])
                cache.append((vk_r, vv))
                vh, ah = orig(vlm_hidden_states=vh, action_hidden_states=ah,
                              layer_idx=li, attention_mask=m4, position_ids=pid,
                              past_key_values=None, use_cache=False,
                              cache_position=None)
        return cache

    cache = prefill()

    def cached_pass(hist):
        """Проход по действиям при кэшированном префиксе."""
        with torch.no_grad():
            alen = n + hist.shape[1]
            pid = pos_ids(alen)
            m4 = model._build_joint_attention_mask_blockwise_ar(
                attention_mask=amask, vlm_seq_len=Lv, action_seq_len=alen,
                device=dev,
                action_key_mask=torch.ones(B, alen, device=dev,
                                           dtype=torch.long))
            mrow = m4[:, :, Lv:, :]                    # строки запросов действия
            emb = model.bos_embedding.expand(B, n, -1).to(dev, dtype)
            if hist.shape[1]:
                emb = torch.cat(
                    [emb, model.action_token_embedding(hist).to(dtype)], 1)
            ah = emb
            dummy = torch.empty((B, Lv + alen, hid), device=dev, dtype=dtype)
            cos, sin = model.vlm.text_model.rotary_emb(dummy, position_ids=pid)
            ca, sa = cos[:, Lv:], sin[:, Lv:]
            for li in range(nL):
                al = model.action_expert.layers[li]
                an = al.input_layernorm(ah)
                aq = al.self_attn.q_proj(an).view(B, alen, nh, hd).transpose(1, 2)
                ak = al.self_attn.k_proj(an).view(B, alen, nkv, hd).transpose(1, 2)
                av = al.self_attn.v_proj(an).view(B, alen, nkv, hd).transpose(1, 2)
                aq, ak = apply_rotary_pos_emb(aq, ak, ca, sa)
                vk_r, vv = cache[li]
                k = torch.cat([vk_r, ak], 2)
                v = torch.cat([vv, av], 2)
                if nkv != nh:
                    k, v = repeat_kv(k, nh // nkv), repeat_kv(v, nh // nkv)
                w = torch.matmul(aq, k.transpose(-1, -2)) * (hd ** -0.5) + mrow
                w = torch.softmax(w, -1, dtype=torch.float32).to(aq.dtype)
                o = torch.matmul(w, v).transpose(1, 2).contiguous().view(
                    B, alen, -1)
                ah = ah + al.self_attn.o_proj(o)
                ah = ah + al.mlp(al.post_attention_layernorm(ah))
            return model.action_lm_head(model.action_expert.norm(ah))

    print("\n" + "=" * 74)
    print("3. ЭКВИВАЛЕНТНОСТЬ ПРОТОТИПА С КЭШЕМ")
    print("=" * 74)
    ok = True
    for k_blocks in (0, 1, 2):
        hist = (torch.empty((B, 0), dtype=torch.long, device=dev) if k_blocks == 0
                else torch.randint(0, model.action_vocab_size,
                                   (B, k_blocks * n), device=dev))
        with torch.no_grad():
            lg_ref = model._run_action_sequence(
                vlm_inputs_embeds=VLM, attention_mask=amask, bos_len=n,
                action_input_ids=hist, position_ids=pos_ids(n + hist.shape[1]))
        lg_new = cached_pass(hist)
        d = (lg_ref.float() - lg_new.float()).abs()
        same = (lg_ref.argmax(-1) == lg_new.argmax(-1)).float().mean()
        print(f"  блок {k_blocks + 1}: top-1 совпал {same:.4%}, "
              f"логиты макс {float(d.max()):.3e}, медиана "
              f"{float(d.median()):.3e}")
        ok &= bool(same == 1.0)

    print("\n" + "=" * 74)
    print("4. ЦЕНА С КЭШЕМ И ИЗМЕРЕННЫЙ ПРИЗ")
    print("=" * 74)
    cac = {}
    for k_blocks in (1, 2):
        hist = torch.randint(0, model.action_vocab_size,
                             (B, k_blocks * n), device=dev)
        cac[k_blocks] = bench(lambda h=hist: cached_pass(h), args.warmup,
                              args.iters, dev)
        print(f"  блок {k_blocks + 1} по кэшу: p50 {cac[k_blocks]['p50']:.2f} мс"
              f"  против опорных {ref[k_blocks]['p50']:.2f} мс"
              f"  ({ref[k_blocks]['p50'] / cac[k_blocks]['p50']:.2f}x)")
    t_pref = bench(prefill, max(3, args.warmup // 4), max(10, args.iters // 5),
                   dev)
    print(f"  prefill (проход 1 + сохранение кэша): p50 {t_pref['p50']:.2f} мс")
    total_cached = t_pref["p50"] + cac[1]["p50"] + cac[2]["p50"]
    print(f"\n  три прохода: опорно {total_ref:.2f} мс -> с кэшем "
          f"{total_cached:.2f} мс, выигрыш {total_ref - total_cached:.2f} мс "
          f"({total_ref / total_cached:.2f}x на блочной части)")
    print("  ЧИТАТЬ ТАК: это выигрыш на БЛОЧНОЙ части. Сквозной множитель "
          "меньше:\n  зрение и ActionCodec в него не входят и не ускоряются. "
          "Оценку 1.84x\n  из §7б можно заменить измеренной только после "
          "сквозного замера.")
    if not ok:
        print("\n  ВНИМАНИЕ: top-1 совпал не везде — прототип НЕ эквивалентен, "
              "числа времени смысла не имеют до исправления")

    if args.out:
        import hashlib
        import subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             text=True).strip()
        except Exception:
            commit = "unknown"
        json.dump(dict(commit=commit, ckpt=args.ckpt, dtype=args.dtype,
                       gpu=(torch.cuda.get_device_name(0)
                            if dev.type == "cuda" else None),
                       torch=torch.__version__, vlen=Lv, batch=B, layers=nL,
                       block=n, ref={str(k): v for k, v in ref.items()},
                       cached={str(k): v for k, v in cac.items()},
                       prefill=t_pref, total_ref=total_ref,
                       total_cached=total_cached, equivalent=ok,
                       self_sha256=hashlib.sha256(
                           open(__file__, "rb").read()).hexdigest()[:16]),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
