#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-21: полный BAR с кэшем префикса (отдельная версия эталона).

В штатном generate каждый из трёх блоков заново прогоняет весь префикс
(изображение + текст) через 24 слоя VLM вместе с action-токенами. Префикс
не видит action-ключей, поэтому его состояния на каждом слое от блока не
зависят. Здесь: префикс — один раз (causal + padding, как в BAR), K/V
каждого слоя сохраняются (после RoPE); затем для каждого блока через 24
слоя эксперта идут только action-токены [BOS·n + история], читающие
сохранённые K/V префикса и свои; маска action-части — блочная, как в BAR.
Greedy-выбор, позиции (strided, position_offset) и сборка ActionCodec —
как в штатном пути.

--verify: на фиксированных наблюдениях val_sel (батч 1 и батч 5 из одной
задачи) сравнивает коды и действия с model.generate: доля совпавших
токенов и чанков, max|Δ| действий. Совпадение не предполагается заранее:
версия с кэшем — отдельный эталон со своим статусом.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _heads(x, B, n, hd):
    return x.view(B, x.shape[1], n, hd).transpose(1, 2)


def prefix_kv(torch, model, vemb, attn_mask, counters=None):
    """Префикс через 24 слоя VLM; K/V каждого слоя после RoPE."""
    import k21_model as km
    tm = model.vlm.text_model
    cfg = tm.config
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nh)
    hd = cfg.hidden_size // nh
    B, V = vemb.shape[:2]
    dev = vemb.device
    pos = torch.arange(V, device=dev)[None].expand(B, -1)
    cos, sin = tm.rotary_emb(vemb, position_ids=pos)
    m = torch.zeros((B, 1, V, V), device=dev, dtype=torch.float32)
    m.masked_fill_(torch.triu(torch.ones((V, V), device=dev,
                                         dtype=torch.bool), 1), -1e9)
    m = m + (attn_mask == 0).float()[:, None, None, :] * -1e9
    h = vemb
    kv = []
    for l, vl in enumerate(tm.layers):
        n_ = vl.input_layernorm(h)
        q = _heads(vl.self_attn.q_proj(n_), B, nh, hd)
        k = _heads(vl.self_attn.k_proj(n_), B, nkv, hd)
        v = _heads(vl.self_attn.v_proj(n_), B, nkv, hd)
        q, k = km.apply_rope(q, k, cos, sin)
        kv.append((k, v))
        kk = km.repeat_kv(k, nh // nkv)
        vv = km.repeat_kv(v, nh // nkv)
        w = torch.matmul(q, kk.transpose(-1, -2)) * (hd ** -0.5)
        w = torch.softmax(w + m, -1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(w, vv).transpose(1, 2).reshape(B, V, -1)
        h = h + vl.self_attn.o_proj(o)
        h = h + vl.mlp(vl.post_attention_layernorm(h))
        if counters is not None:
            counters["text"][l] += 1
    return kv, (nh, nkv, hd)


def action_logits(torch, model, kv, heads, attn_mask, V, history, po,
                  counters=None):
    """[BOS·n + история] через 24 слоя эксперта с кэшем префикса."""
    import k21_model as km
    nh, nkv, hd = heads
    n = model.block_size
    B = attn_mask.shape[0]
    dev = attn_mask.device
    bos = model.bos_embedding.expand(B, n, -1).to(dev, kv[0][0].dtype)
    if history is None or history.shape[1] == 0:
        ha = bos
    else:
        ha = torch.cat([bos, model.action_token_embedding(history).to(
            bos.dtype)], 1)
    A = ha.shape[1]
    apos = model._build_action_pos_ids_strided(
        batch_size=B, base_pos=V, action_seq_len=A, device=dev,
        position_offset=int(po))
    cos, sin = model.vlm.text_model.rotary_emb(ha, position_ids=apos)
    m = torch.zeros((B, 1, A, V + A), device=dev, dtype=torch.float32)
    m[:, :, :, :V] += (attn_mask == 0).float()[:, None, None, :] * -1e9
    blk = torch.arange(A, device=dev) // n
    fut = blk[None, :] > blk[:, None]
    m[:, :, :, V:] = m[:, :, :, V:].masked_fill(fut, -1e9)
    for l, al in enumerate(model.action_expert.layers):
        an = al.input_layernorm(ha)
        q = _heads(al.self_attn.q_proj(an), B, nh, hd)
        k = _heads(al.self_attn.k_proj(an), B, nkv, hd)
        v = _heads(al.self_attn.v_proj(an), B, nkv, hd)
        q, k = km.apply_rope(q, k, cos, sin)
        K = km.repeat_kv(torch.cat([kv[l][0], k.to(kv[l][0].dtype)], 2),
                         nh // nkv)
        Vv = km.repeat_kv(torch.cat([kv[l][1], v.to(kv[l][1].dtype)], 2),
                          nh // nkv)
        w = torch.matmul(q.to(K.dtype), K.transpose(-1, -2)) * (hd ** -0.5)
        w = torch.softmax(w + m, -1, dtype=torch.float32).to(K.dtype)
        o = torch.matmul(w, Vv).transpose(1, 2).reshape(B, A, -1)
        ha = ha + al.self_attn.o_proj(o)
        ha = ha + al.mlp(al.post_attention_layernorm(ha))
        if counters is not None:
            counters["expert"][l] += 1
    return model.action_lm_head(model.action_expert.norm(ha))


def generate_cached(torch, model, batch, po, counters=None):
    """Тот же greedy blockwise generate, префикс — один раз."""
    with torch.no_grad():
        img = model.vlm.get_image_features(batch["pixel_values"],
                                           batch.get("pixel_attention_mask"))
        if counters is not None:
            counters["vision"] += 1
        _b, _l, vemb, _i = model._build_vlm_inputs_embeds(
            input_ids=batch["input_ids"], inputs_embeds=None,
            pixel_values=None, pixel_attention_mask=None,
            image_hidden_states=img)
        am = batch["attention_mask"]
        V = vemb.shape[1]
        kv, heads = prefix_kv(torch, model, vemb, am, counters)
        n = model.block_size
        hist = torch.empty((vemb.shape[0], 0), device=vemb.device,
                           dtype=torch.long)
        out = []
        for b in range(model.num_blocks):
            lg = action_logits(torch, model, kv, heads, am, V, hist, po,
                               counters)
            s = 0 if b == 0 else n + (b - 1) * n
            tok = lg[:, s:s + n].argmax(-1)
            out.append(tok)
            hist = torch.cat([hist, tok], 1)
        return torch.cat(out, 1)


def decode(torch, E, codec, toks, B):
    K = toks.reshape(B, 3, 16)
    z = sum(E[j][K[:, j, :].long()] for j in range(3))
    x, _ = codec._decode(z, embodiment_ids=0)
    return x[..., :7].float()


VERIFY_REPORT = "reports/k21/barcache_verify.json"


def counters_ok(cnt, n_blocks=3):
    """vision 1, каждый слой VLM 1 (префикс один раз), эксперт n_blocks."""
    return (cnt["vision"] == 1 and len(cnt["text"]) == 24
            and all(x == 1 for x in cnt["text"])
            and all(x == n_blocks for x in cnt["expert"]))


def sha(p):
    import hashlib
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def verify_identity(D):
    """Чем определяется эквивалентность: ревизия BAR, подготовка, код кэша."""
    return dict(revision=D["prep_report"]["bar_pin"]["revision"],
                prep_id=D["prep_report"]["prep_id"],
                barcache_sha1=sha(os.path.join(HERE, "k21_barcache.py")))


def require_verified(D, path=VERIFY_REPORT):
    """Отказ (SystemExit), если нет успешной проверки для этой же ревизии,
    подготовки и кода k21_barcache.py."""
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}: сначала k21_barcache.py --verify")
    r = json.load(open(path))
    want = verify_identity(D)
    got = {k: r.get(k) for k in want}
    if got != want:
        raise SystemExit(f"проверка кэша сделана для другого состояния: "
                         f"{got} != {want}")
    if r.get("ok") is not True or r.get("equivalent") is not True:
        raise SystemExit(f"проверка кэша не пройдена: {r.get('problems')}")
    return r


ADMIT_REPORT = "reports/k21/barcache_admit.json"
NOT_CONFIRMED = "эквивалентность штатному пути не подтверждена"


def admit(a):
    """Допуск fullbar_cached как САМОСТОЯТЕЛЬНОГО эталона (строгий --verify
    не заменяет и не ослабляет): на той же очереди val_sel, батч 1 и 5 —
    конечность действий, форма чанка, счётчики на каждом батче (vision 1,
    каждый слой VLM 1, эксперт num_blocks), закреплённые ревизия, подготовка
    и SHA кода. Статус эквивалентности берётся только из строгой проверки
    той же версии кода; иначе — «не подтверждена». Код 3 при проблемах."""
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_latency as kl
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    model = bar["model"]
    va = np.asarray(D["val_rows"])
    prob, res = [], {}
    for bs in (1, 5):
        queue = kl.make_queue(D["task"], va, bs, a.n // bs)
        n_ok = 0
        for rows in queue:
            batch = kd.prompt_batch(
                bar, [np.asarray(D["IMG"][i]) for i in rows],
                kd.norm_states(np.asarray(D["state8"][rows])),
                D["task"][rows].tolist())
            po = int(D["pos_offset"][rows[0]])
            cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24)
            got = generate_cached(torch, model, batch, po, cnt)
            if not counters_ok(cnt, model.num_blocks):
                prob.append(f"batch {bs}: счётчики {cnt}")
            if tuple(got.shape) != (len(rows), model.num_blocks *
                                    model.block_size):
                prob.append(f"batch {bs}: коды формы {tuple(got.shape)}")
            with torch.no_grad():
                x = decode(torch, bar["E"], bar["codec"], got, len(rows))
            if not bool(torch.isfinite(x).all()):
                prob.append(f"batch {bs}: нечисловые действия")
            if x.shape[0] != len(rows) or x.shape[-1] != 7:
                prob.append(f"batch {bs}: чанк формы {tuple(x.shape)}")
            n_ok += 1
        res[f"batch{bs}"] = dict(batches=n_ok)
    prob = sorted(set(prob))
    ident = verify_identity(D)
    eq = False
    if os.path.exists(VERIFY_REPORT):
        v = json.load(open(VERIFY_REPORT))
        eq = all(v.get(k) == x for k, x in ident.items()) and \
            v.get("ok") is True and v.get("equivalent") is True
    res.update(ident, problems=prob, ok=not prob, kind="standalone",
               equivalence_confirmed=eq,
               equivalence=("подтверждена строгой проверкой" if eq
                            else NOT_CONFIRMED))
    os.makedirs("reports/k21", exist_ok=True)
    json.dump(res, open(ADMIT_REPORT, "w"), ensure_ascii=False, indent=1)
    print(f"  допуск самостоятельного эталона: "
          f"{'пройден' if not prob else 'НЕ пройден'}; {res['equivalence']}; "
          f"отчёт {ADMIT_REPORT}", flush=True)
    if prob:
        print("  ОТКАЗ: " + "; ".join(prob[:6]), flush=True)
        return 3
    return 0


def require_admitted(D, path=ADMIT_REPORT):
    """Отказ, если нет пройденного допуска самостоятельного эталона для этой
    же ревизии, подготовки и кода. Возвращает отчёт допуска."""
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}: сначала k21_barcache.py --admit")
    r = json.load(open(path))
    want = verify_identity(D)
    got = {k: r.get(k) for k in want}
    if got != want:
        raise SystemExit(f"допуск сделан для другого состояния: {got} != "
                         f"{want}")
    if r.get("ok") is not True or r.get("kind") != "standalone":
        raise SystemExit(f"допуск не пройден: {r.get('problems')}")
    return r


def verify(a):
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_latency as kl
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    model = bar["model"]
    va = np.asarray(D["val_rows"])
    res, prob = {}, []
    for bs in (1, 5):
        queue = kl.make_queue(D["task"], va, bs, a.n // bs)
        tok_eq = chunk_eq = 0
        n_tok = n_chunk = 0
        maxd = 0.0
        t_std, t_cache = [], []
        cnt = None
        for rows in queue:
            batch = kd.prompt_batch(
                bar, [np.asarray(D["IMG"][i]) for i in rows],
                kd.norm_states(np.asarray(D["state8"][rows])),
                D["task"][rows].tolist())
            po = int(D["pos_offset"][rows[0]])
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                ref = model.generate(**batch, position_offset=po,
                                     do_sample=False, initial_position_shift=1)
            torch.cuda.synchronize(dev)
            t_std.append(time.perf_counter() - t0)
            cnt = dict(vision=0, text=[0] * 24, expert=[0] * 24)
            t0 = time.perf_counter()
            got = generate_cached(torch, model, batch, po, cnt)
            torch.cuda.synchronize(dev)
            t_cache.append(time.perf_counter() - t0)
            if not counters_ok(cnt, model.num_blocks):
                prob.append(f"batch {bs}: счётчики {cnt}")
            tok_eq += int((ref == got).sum())
            n_tok += ref.numel()
            chunk_eq += int((ref == got).all(1).sum())
            n_chunk += ref.shape[0]
            with torch.no_grad():
                d = (decode(torch, bar["E"], bar["codec"], ref, len(rows))
                     - decode(torch, bar["E"], bar["codec"], got, len(rows)))
            if not bool(torch.isfinite(d).all()):
                prob.append(f"batch {bs}: нечисловые действия")
            maxd = max(maxd, float(d.abs().max()))
        res[f"batch{bs}"] = dict(
            observations=int(n_chunk), token_match=tok_eq / n_tok,
            chunk_match=chunk_eq / n_chunk, max_abs_action_diff=maxd,
            counters_last=cnt, seconds_std_median=float(np.median(t_std)),
            seconds_cached_median=float(np.median(t_cache)))
        r = res[f"batch{bs}"]
        print(f"  batch {bs}: наблюдений {r['observations']}, совпало токенов "
              f"{100 * r['token_match']:.2f}%, чанков "
              f"{100 * r['chunk_match']:.1f}%, max|Δ действия| "
              f"{r['max_abs_action_diff']:.2e}; vision {cnt['vision']}×, "
              f"слои VLM {set(cnt['text'])}×, эксперт {set(cnt['expert'])}×; "
              f"generate {1e3 * r['seconds_std_median']:.0f} мс против "
              f"кэша {1e3 * r['seconds_cached_median']:.0f} мс (грубо, без "
              f"прогрева)", flush=True)
    res["equivalent"] = all(res[f"batch{b}"]["chunk_match"] == 1.0
                            for b in (1, 5))
    if not res["equivalent"]:
        prob.append("коды не совпали полностью")
    prob = sorted(set(prob))
    res.update(verify_identity(D), problems=prob, ok=not prob)
    os.makedirs("reports/k21", exist_ok=True)
    json.dump(res, open(VERIFY_REPORT, "w"), ensure_ascii=False, indent=1)
    print("  " + ("коды совпали полностью: версия с кэшем эквивалентна "
                  "штатной" if res["equivalent"] else
                  "коды НЕ совпали полностью: версия с кэшем — отдельный "
                  "эталон со своим статусом") + "; отчёт "
          "reports/k21/barcache_verify.json")
    if prob:
        print("  ОТКАЗ: " + "; ".join(prob), flush=True)
        return 3
    return 0


def noise_floor(a):
    """Описательно (критерий --verify не меняет): уровень шума самого
    эталона и близость к ничьей в местах расхождения.
      1) generate на батче 5 против generate по одному наблюдению — насколько
         штатный путь согласован сам с собой при другом составе батча;
      2) то же для пути с кэшем;
      3) на первом расходящемся блоке (история — коды эталона) разрыв
         логитов пути с кэшем между его argmax и токеном эталона, рядом —
         распределение разрыва top1 − top2 на всех токенах."""
    import torch
    import k20_data as kd
    import k21_data as d21
    import k21_latency as kl
    dev = torch.device(a.device)
    D = d21.load_prepared(require_teacher=False)
    bar = d21.load_bar(dev, D["prep_report"]["bar_pin"])
    model = bar["model"]
    n = model.block_size
    va = np.asarray(D["val_rows"])
    queue = kl.make_queue(D["task"], va, 5, a.n // 5)
    agree = dict(std_b5_vs_b1=[0, 0, 0, 0], cached_b5_vs_b1=[0, 0, 0, 0],
                 cached_vs_std_b5=[0, 0, 0, 0], cached_vs_std_b1=[0, 0, 0, 0])
    gaps_mis, gaps_all = [], []

    def acc(key, x, y):
        v = agree[key]
        v[0] += int((x == y).sum())
        v[1] += x.numel()
        v[2] += int((x == y).all(1).sum())
        v[3] += x.shape[0]

    def std(b, po):
        with torch.no_grad():
            return model.generate(**b, position_offset=po, do_sample=False,
                                  initial_position_shift=1)

    for rows in queue:
        imgs = [np.asarray(D["IMG"][i]) for i in rows]
        st8 = kd.norm_states(np.asarray(D["state8"][rows]))
        tasks = D["task"][rows].tolist()
        po = int(D["pos_offset"][rows[0]])
        b5 = kd.prompt_batch(bar, imgs, st8, tasks)
        s5 = std(b5, po)
        c5 = generate_cached(torch, model, b5, po)
        s1, c1 = [], []
        for j in range(len(rows)):
            b1 = kd.prompt_batch(bar, imgs[j:j + 1], st8[j:j + 1],
                                 tasks[j:j + 1])
            s1.append(std(b1, po))
            c1.append(generate_cached(torch, model, b1, po))
        s1, c1 = torch.cat(s1), torch.cat(c1)
        acc("std_b5_vs_b1", s5, s1)
        acc("cached_b5_vs_b1", c5, c1)
        acc("cached_vs_std_b5", c5, s5)
        acc("cached_vs_std_b1", c1, s1)
        # разрыв логитов на первом расходящемся блоке (батч 5)
        with torch.no_grad():
            img = model.vlm.get_image_features(
                b5["pixel_values"], b5.get("pixel_attention_mask"))
            _b, _l, vemb, _i = model._build_vlm_inputs_embeds(
                input_ids=b5["input_ids"], inputs_embeds=None,
                pixel_values=None, pixel_attention_mask=None,
                image_hidden_states=img)
            am = b5["attention_mask"]
            kv, heads = prefix_kv(torch, model, vemb, am)
            for blk in range(model.num_blocks):
                lg = action_logits(torch, model, kv, heads, am,
                                   vemb.shape[1], s5[:, :blk * n], po).float()
                s_ = 0 if blk == 0 else n + (blk - 1) * n
                lg = lg[:, s_:s_ + n]
                top2 = lg.topk(2, -1).values
                gaps_all.append((top2[..., 0] - top2[..., 1]).flatten()
                                .cpu().numpy())
                ref = s5[:, blk * n:(blk + 1) * n]
                mis = lg.argmax(-1) != ref
                if mis.any():
                    g = (lg.max(-1).values - lg.gather(-1, ref[..., None])
                         [..., 0])[mis]
                    gaps_mis.append(g.cpu().numpy())
    out = {k: dict(token_match=v[0] / v[1], chunk_match=v[2] / v[3],
                   chunks=v[3]) for k, v in agree.items()}
    ga = np.concatenate(gaps_all)
    gm = np.concatenate(gaps_mis) if gaps_mis else np.zeros(0)
    out["logit_gap"] = dict(
        mismatch_n=int(gm.size),
        mismatch_gap_quantiles=(np.percentile(gm, [50, 90, 100]).tolist()
                                if gm.size else None),
        all_top1_top2_quantiles=np.percentile(ga, [1, 5, 50]).tolist(),
        share_all_below_max_mismatch_gap=(float((ga <= gm.max()).mean())
                                          if gm.size else None))
    out["note"] = ("описательно; критерий эквивалентности --verify не "
                   "меняется")
    out.update(verify_identity(D))
    for k in agree:
        r = out[k]
        print(f"  {k:18}: совпало токенов {100 * r['token_match']:.2f}%, "
              f"чанков {100 * r['chunk_match']:.1f}% ({r['chunks']})")
    lgp = out["logit_gap"]
    print(f"  расхождений на первом блоке с историей эталона: "
          f"{lgp['mismatch_n']}; разрыв логитов (медиана, 90%, макс) "
          f"{lgp['mismatch_gap_quantiles']}; top1−top2 по всем токенам "
          f"(1%, 5%, 50%) {lgp['all_top1_top2_quantiles']}; доля токенов с "
          f"разрывом ≤ максимального у расхождений "
          f"{lgp['share_all_below_max_mismatch_gap']}", flush=True)
    os.makedirs("reports/k21", exist_ok=True)
    json.dump(out, open("reports/k21/barcache_noise.json", "w"),
              ensure_ascii=False, indent=1)
    print("  отчёт reports/k21/barcache_noise.json")
    return 0


def selftest():
    """Игрушечная модель: кэш префикса = совместный проход BAR-подобной
    схемы (префикс не видит действий) на всех трёх блоках."""
    import torch
    import k21_model as km
    torch.manual_seed(0)
    bar = km.fake_bar(torch)
    nn = torch.nn

    class Fake(nn.Module):
        pass
    m = bar
    m.block_size, m.num_blocks = 4, 3
    m.action_token_embedding = nn.Embedding(20, 48)
    m.action_lm_head = nn.Linear(48, 20, bias=False)

    def strided(*, batch_size, base_pos, action_seq_len, device,
                position_offset=0):
        s = position_offset + 1
        p = torch.arange(action_seq_len, device=device)[None].expand(
            batch_size, -1)
        return base_pos + position_offset + s * p
    m._build_action_pos_ids_strided = strided
    b = km.fake_batch(torch, B=2, pad=2)
    with torch.no_grad():
        img = m.vlm.get_image_features(b["pixel_values"])
        _b, _l, vemb, _i = m._build_vlm_inputs_embeds(
            input_ids=b["input_ids"], inputs_embeds=None, pixel_values=None,
            pixel_attention_mask=None, image_hidden_states=img)
        am = b["attention_mask"]
        V = vemb.shape[1]
        kv, heads = prefix_kv(torch, m, vemb, am)
        hist = torch.randint(0, 20, (2, 8))
        lg = action_logits(torch, m, kv, heads, am, V, hist, 1)
        # эталон: совместный проход (как _run_action_sequence BAR) через
        # собственный слой K21 с блочной маской
        M = km.build(torch)
        k21 = M["K21"](m, "flat24", -np.ones(6), np.ones(6), lora=False)
        A = 4 + 8
        ha = torch.cat([m.bos_embedding.expand(2, 4, -1),
                        m.action_token_embedding(hist)], 1)
        mask = k21._mask(am, V, A, "cpu")
        blk = torch.arange(A) // 4
        mask[:, :, V:, V:] = mask[:, :, V:, V:].masked_fill(
            blk[None, :] > blk[:, None], -1e9)
        apos = strided(batch_size=2, base_pos=V, action_seq_len=A,
                       device="cpu", position_offset=1)
        pos = torch.cat([torch.arange(V)[None].expand(2, -1), apos], 1)
        cos, sin = m.vlm.text_model.rotary_emb(vemb, position_ids=pos)
        hv = vemb
        for l in range(24):
            hv, ha = k21._layer(l, hv, ha, mask, cos, sin)
        ref = m.action_lm_head(m.action_expert.norm(ha))
    assert torch.allclose(lg, ref, atol=1e-5), float((lg - ref).abs().max())
    _ = Fake
    good = dict(vision=1, text=[1] * 24, expert=[3] * 24)
    assert counters_ok(good)
    for bad in (dict(good, vision=2), dict(good, text=[1] * 23 + [3]),
                dict(good, expert=[3] * 23 + [2])):
        assert not counters_ok(bad)
    import tempfile
    DD = dict(prep_report=dict(bar_pin=dict(revision="r"), prep_id="p"))
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "v.json")
        base = dict(verify_identity(DD), ok=True, equivalent=True)
        json.dump(base, open(fp, "w"))
        require_verified(DD, fp)
        for bad in (dict(base, ok=False), dict(base, equivalent=False),
                    dict(base, prep_id="q"), dict(base, barcache_sha1="x"),
                    dict(base, revision="s")):
            json.dump(bad, open(fp, "w"))
            try:
                require_verified(DD, fp)
                raise AssertionError(bad)
            except SystemExit:
                pass
        ap_ = os.path.join(td, "a.json")
        okr = dict(verify_identity(DD), ok=True, kind="standalone")
        json.dump(okr, open(ap_, "w"))
        require_admitted(DD, ap_)
        for bad in (dict(okr, ok=False), dict(okr, kind="strict"),
                    dict(okr, barcache_sha1="x"), dict(okr, prep_id="q")):
            json.dump(bad, open(ap_, "w"))
            try:
                require_admitted(DD, ap_)
                raise AssertionError(bad)
            except SystemExit:
                pass
    print("самопроверка k21_barcache пройдена: кэш префикса = совместному "
          "проходу BAR-подобной схемы (блочная маска, padding, strided "
          "позиции); счётчики и допуск латентности по отчёту проверки")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: BAR с кэшем префикса")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--noise-floor", action="store_true")
    ap.add_argument("--admit", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.verify:
        return verify(a)
    if a.noise_floor:
        return noise_floor(a)
    if a.admit:
        return admit(a)
    raise SystemExit("--selftest, --verify, --admit или --noise-floor")


if __name__ == "__main__":
    sys.exit(main())
