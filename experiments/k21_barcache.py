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
    res = {}
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
            tok_eq += int((ref == got).sum())
            n_tok += ref.numel()
            chunk_eq += int((ref == got).all(1).sum())
            n_chunk += ref.shape[0]
            with torch.no_grad():
                d = (decode(torch, bar["E"], bar["codec"], ref, len(rows))
                     - decode(torch, bar["E"], bar["codec"], got, len(rows)))
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
    res["revision"] = D["prep_report"]["bar_pin"]["revision"]
    os.makedirs("reports/k21", exist_ok=True)
    json.dump(res, open("reports/k21/barcache_verify.json", "w"),
              ensure_ascii=False, indent=1)
    print("  " + ("коды совпали полностью: версия с кэшем эквивалентна "
                  "штатной" if res["equivalent"] else
                  "коды НЕ совпали полностью: версия с кэшем — отдельный "
                  "эталон со своим статусом") + "; отчёт "
          "reports/k21/barcache_verify.json")
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
    print("самопроверка k21_barcache пройдена: кэш префикса = совместному "
          "проходу BAR-подобной схемы (блочная маска, padding, strided "
          "позиции)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="K-21: BAR с кэшем префикса")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.verify:
        return verify(a)
    raise SystemExit("--selftest или --verify")


if __name__ == "__main__":
    sys.exit(main())
