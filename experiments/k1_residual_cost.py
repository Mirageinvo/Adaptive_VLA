"""K-1: цена устаревшего суффикса RVQ, и настоящая ли это цена.

ПОВОД. План (§4.1) предлагает при смене грубого кода переквантовать зависимый
суффикс. Обоснование было — «остаточная семантика ломается». Но чтение кода
ActionCodec показало (FINDINGS.md §1):

    modeling_actioncodec.py:271  z_q = self.vq.from_codes(indices)[0]   # СУММА
    modeling_actioncodec.py:303  x_recon, _ = self.decoder(z_q, ...)

Декодер получает ТОЛЬКО сумму и не знает, каким кортежем кодов она порождена.
Значит «нарушить остаточную семантику» на его уровне нельзя.

Остаётся другой вопрос, и он настоящий: **не ушла ли сумма из области, на
которой декодер обучался.**

ДВА ОБЪЯСНЕНИЯ, КОТОРЫЕ НАДО РАЗДЕЛИТЬ. Пусть A — подмена кода со старым
суффиксом, B — она же с переквантованным суффиксом.

  (i)  A хуже, чем следует из его ошибки латенты. Декодер чувствителен к
       области, куда ушла сумма. Механизм есть, и он про иерархию.
  (ii) A и B ложатся на одну зависимость «ошибка действия от ошибки латенты».
       Тогда A просто худшее приближение, помогает ЛЮБОЕ сближение с целевой
       латентой, а иерархическая переквантизация — лишь один из способов.
       Вклад тонкий, формулировку надо менять.

Наивное сравнение «B лучше A» не различает их вовсе: B по построению есть
наилучшее RVQ-приближение цели при данном префиксе, так что его превосходство
гарантировано арифметически, а не установлено измерением.

ПОЭТОМУ СРАВНИВАЕМ ПРИ РАВНОЙ ОШИБКЕ ЛАТЕНТЫ. Раскладываем оба варианта по
корзинам ошибки латенты и внутри корзины смотрим ошибку действия. Та же
дисциплина, что везде: сравнение только при уравненном условии.

ПРАВИЛО ЧТЕНИЯ, зафиксировано до запуска:
  внутри корзин A заметно хуже B (скажем, в 1.3 раза и более, устойчиво по
      корзинам) -> объяснение (i), механизм есть, §4 плана обоснован;
  внутри корзин A и B совпадают -> объяснение (ii), «остаточная
      несогласованность» как отдельное явление отсутствует, и заявку надо
      переписывать на проекцию в пространстве латент.

Запуск:
    python3 experiments/k1_residual_cost.py \
        --zarr /path/libero10_N500.zarr [--device cuda]
"""

import argparse
import os
import sys

import einops
import numpy as np
import torch


def load_codec(root: str, model_id: str, device: str):
    sys.path.insert(0, root)
    from actioncodec.modeling_actioncodec import ActionCodec

    m = ActionCodec.from_pretrained(model_id).to(device).eval()
    print(f"словарь {m.vocab_size}, уровней {m.num_quantizers}, "
          f"позиций {m.n_tokens_per_quantizer}")
    return m


def projected_codebooks(model, device: str) -> torch.Tensor:
    """(L, V, D) — вектор, который каждый код ДОБАВЛЯЕТ к сумме.

    Именно в этом пространстве живёт остаток: from_codes складывает
    out_project(decode_code(c)) по уровням, значит и жадная переквантизация
    должна искать ближайший код здесь, а не в пространстве до проекции."""
    V = model.vocab_size
    idx = torch.arange(V, device=device).unsqueeze(0)          # (1, V)
    out = []
    with torch.no_grad():
        for q in model.vq.quantizers:
            e = q.out_project(q.decode_code(idx))              # (1, V, D)
            out.append(e[0])
    E = torch.stack(out)                                       # (L, V, D)
    assert E.shape[0] == model.num_quantizers, E.shape
    return E


def latent_from_codes(E: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """codes (B, P, L) -> сумма (B, P, D)."""
    L = E.shape[0]
    return sum(E[j][codes[:, :, j]] for j in range(L))


def requantize_suffix(E: torch.Tensor, codes: torch.Tensor, target: torch.Tensor,
                      level: int) -> torch.Tensor:
    """Жадно переквантовать уровни > level так, чтобы сумма приблизилась к
    target. Это ровно процедура кодировщика RVQ, применённая к предсказанной
    (здесь — истинной) чистой латенте."""
    out = codes.clone()
    L = E.shape[0]
    r = target - sum(E[j][out[:, :, j]] for j in range(level + 1))
    for j in range(level + 1, L):
        d = torch.cdist(r, E[j])                               # (B, P, V)
        out[:, :, j] = d.argmin(-1)
        r = r - E[j][out[:, :, j]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--model", default="ZibinDong/ActionCodec-Base-RVQft")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--n-chunks", type=int, default=64)
    ap.add_argument("--n-cand", type=int, default=24,
                    help="подстановок на (чанк, уровень)")
    ap.add_argument("--embodiment", type=int, default=0, help="franka_libero_20hz")
    ap.add_argument("--fix-gripper", choices=["auto", "yes", "no", "invert"],
                    default="auto",
                    help="перевести захват: auto/yes = (x+1)/2, invert = (1-x)/2")
    ap.add_argument("--no-gripper", action="store_true",
                    help="исключить канал захвата из нормы ошибки действия")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import zarr

    model = load_codec(os.path.abspath(args.root), args.model, args.device)
    V, L, P = model.vocab_size, model.num_quantizers, model.n_tokens_per_quantizer
    emb_cfg = model.config.embodiment_config
    name = list(emb_cfg.keys())[args.embodiment]
    cfg = emb_cfg[name]
    T = int(cfg["freq"] * cfg["duration"])
    D_act = cfg["action_dim"]
    print(f"эмбодимент {args.embodiment} = {name}: чанк {T}x{D_act}\n")

    # ---------- данные ----------
    z = zarr.open(os.path.abspath(args.zarr), mode="r")
    acts = np.asarray(z["data"]["action"])
    ends = np.asarray(z["meta"]["episode_ends"])
    assert acts.shape[1] == D_act, f"действие {acts.shape[1]}-мерное, ожидалось {D_act}"
    chunks, start = [], 0
    for e in ends:
        ep = acts[start:e]
        for i in range(len(ep) // T):
            chunks.append(ep[i * T:(i + 1) * T])
        start = e
    A = np.stack(chunks).astype(np.float32)

    # ---- диагностика формата: она ловит несовпадение соглашений ----
    print("каналы действия в данных:")
    for d in range(A.shape[2]):
        c = A[:, :, d]
        u = np.unique(c)
        print(f"  {d}: [{c.min():+.3f}, {c.max():+.3f}] "
              f"среднее {c.mean():+.3f}"
              + (f", значений всего {len(u)}: {u[:4]}" if len(u) <= 4 else ""))

    # ActionCodec описывает эмбодимент как «gripper position (1 open/0 close)»,
    # а LIBERO обычно даёт -1/+1. При несовпадении канал захвата
    # реконструируется плохо, и это утащит все числа.
    g = A[:, :, -1]
    if args.fix_gripper == "auto":
        need = g.min() < -0.5
    else:
        need = args.fix_gripper == "yes"
    if args.fix_gripper == "invert":
        print(f"\nЗАХВАТ: диапазон [{g.min():+.2f}, {g.max():+.2f}] -> (1-x)/2 "
              f"(перевёрнутое соглашение).")
        A[:, :, -1] = (1.0 - A[:, :, -1]) / 2.0
    elif need:
        print(f"\nЗАХВАТ: в данных диапазон [{g.min():+.2f}, {g.max():+.2f}], "
              f"кодек ждёт [0, 1] — перевожу (x+1)/2.")
        A[:, :, -1] = (A[:, :, -1] + 1.0) / 2.0
    else:
        print(f"\nЗАХВАТ: диапазон [{g.min():+.2f}, {g.max():+.2f}], "
              f"перевод не нужен.")

    idx = np.random.default_rng(0).choice(len(A), size=min(args.n_chunks, len(A)),
                                          replace=False)
    a = torch.from_numpy(A[idx]).to(args.device)
    scale = float(a.max() - a.min())
    print(f"\nчанков {len(a)}, размах действий {scale:.2f}")

    E = projected_codebooks(model, args.device)
    Dz = E.shape[-1]

    with torch.no_grad():
        flat = torch.as_tensor(np.asarray(model.encode(a, embodiment_ids=args.embodiment)),
                               device=args.device, dtype=torch.long)
        assert flat.shape[1] == P * L, f"{flat.shape} против P*L={P*L}"

        # ВНИМАНИЕ. Докстринг model.decode утверждает, что токены перемежаются
        # ([q0_t0, q1_t0, ...]), но код (modeling_actioncodec.py:595) делает
        #     einops.rearrange(tokens, "b (n m) -> b m n", m=n_tokens_per_quantizer)
        # где n — уровень и он ВНЕШНИЙ. То есть раскладка ПОУРОВНЕВАЯ:
        # [q0_t0..q0_t15, q1_t0..q1_t15, ...]. Докстринг противоречит коду.
        # Пользуемся их же вызовом, чтобы соглашение совпадало по построению.
        codes = einops.rearrange(flat, "b (n m) -> b m n", m=P)      # (B, P, L)

        def unflatten_back(c):
            return einops.rearrange(c, "b m n -> b (n m)")

        assert torch.equal(unflatten_back(codes), flat), "разворот не обратим"

        # ---- САНИТАРНАЯ ПРОВЕРКА: наша таблица против их from_codes ----
        h_ours = latent_from_codes(E, codes)                       # (B, P, D)
        h_theirs = model.vq.from_codes(codes)[0]
        if h_theirs.shape != h_ours.shape:                          # (B, D, P)?
            h_theirs = h_theirs.transpose(1, 2)
        gap = (h_ours - h_theirs).abs().max().item()
        print(f"сверка латенты: макс. расхождение {gap:.2e} "
              f"({'совпадает' if gap < 1e-4 else 'РАЗОШЛОСЬ'})")
        assert gap < 1e-4, ("наша сумма не совпала с from_codes — дальше всё "
                            "будет считаться в неверном пространстве")

        def decode(c):
            # numpy, а не torch: в их torch-ветке (modeling_actioncodec.py:511)
            # вызывается tokens.dtype.is_integer, чего у torch.dtype нет.
            t = unflatten_back(c).cpu().numpy().astype(np.int64)
            rec, _ = model.decode(t, embodiment_ids=args.embodiment)
            return torch.as_tensor(np.asarray(rec)[..., :D_act],
                                   device=args.device, dtype=torch.float32)

        base = decode(codes)
        floor = ((base - a).abs().median() / scale).item()
        per_ch = [((base - a)[:, :, d].abs().median() / scale).item()
                  for d in range(D_act)]
        print(f"пол кодека (encode-decode): {floor:.4f} размаха")
        print("  по каналам: " + " ".join(f"{e:.4f}" for e in per_ch))
        if floor > 0.05:
            print("  ВНИМАНИЕ: велик. Скорее всего не сошлись формат действий,\n"
                  "  частота или нормализация. Смотреть, какой канал виноват —\n"
                  "  дальнейшие числа при таком поле недостоверны.")
        print()

        # ---------- перебор подстановок ----------
        rng = np.random.default_rng(1)
        rows = []          # (уровень, ошибка латенты, ошибка действия, вариант)
        ch_dev = []        # вклад каждого канала в отклонение
        for lev in range(L):
            for _ in range(args.n_cand):
                p = int(rng.integers(P))
                v = int(rng.integers(V))
                cA = codes.clone()
                cA[:, p, lev] = v
                cB = requantize_suffix(E, cA, h_ours, lev)

                for tag, c in (("A", cA), ("B", cB)):
                    h = latent_from_codes(E, c)
                    le = (h - h_ours).norm(dim=-1).amax(-1) / np.sqrt(Dz)   # (B,)
                    d_ = decode(c)
                    dev = (d_ - base).abs()                       # (B, T, D)
                    ch_dev.append(dev.amax(1).median(0).values.cpu().numpy())
                    if args.no_gripper:
                        dev = dev[..., :-1]
                    ae = dev.flatten(1).amax(-1) / scale                     # (B,)
                    for i in range(len(a)):
                        rows.append((lev, float(le[i]), float(ae[i]), tag))

    R = np.array([(r[0], r[1], r[2], 0 if r[3] == "A" else 1) for r in rows])
    print(f"замеров {len(R)}: A {(R[:,3]==0).sum()}, B {(R[:,3]==1).sum()}")
    cd = np.median(np.stack(ch_dev), axis=0) / scale
    print("медианное отклонение по каналам: " + " ".join(f"{v:.4f}" for v in cd)
          + ("   (захват исключён из нормы)" if args.no_gripper else ""))
    if cd[-1] > 2 * cd[:-1].max():
        print("  ВНИМАНИЕ: захват доминирует над остальными каналами — "
              "перепроверить\n  вывод с --no-gripper.")

    # ---------- наивное сравнение (то, что НЕ доказывает ничего) ----------
    print("\n" + "=" * 78)
    print("НАИВНОЕ СРАВНЕНИЕ — привожу, чтобы показать его бесполезность")
    print("=" * 78)
    print(f"{'уровень':>8}{'ошибка действия A':>20}{'B':>12}{'A/B':>8}")
    for lev in range(L):
        m = R[:, 0] == lev
        eA = np.median(R[m & (R[:, 3] == 0), 2])
        eB = np.median(R[m & (R[:, 3] == 1), 2])
        print(f"{lev:>8}{eA:>20.4f}{eB:>12.4f}{eA/max(eB,1e-9):>8.2f}")
    print("""
B по построению есть наилучшее RVQ-приближение цели при данном префиксе,
поэтому его превосходство здесь гарантировано арифметикой. Ниже — сравнение
ПРИ РАВНОЙ ОШИБКЕ ЛАТЕНТЫ, только оно и различает объяснения.""")

    # ---------- сравнение при равной ошибке латенты ----------
    print("\n" + "=" * 78)
    print("ПРИ РАВНОЙ ОШИБКЕ ЛАТЕНТЫ")
    print("=" * 78)
    q = np.quantile(R[:, 1], np.linspace(0, 1, 9))
    print(f"{'корзина ошибки латенты':>26}{'n(A)':>7}{'n(B)':>7}"
          f"{'действие A':>13}{'действие B':>13}{'A/B':>8}")
    ratios = []
    for k in range(len(q) - 1):
        m = (R[:, 1] >= q[k]) & (R[:, 1] < q[k + 1])
        mA, mB = m & (R[:, 3] == 0), m & (R[:, 3] == 1)
        nA, nB = int(mA.sum()), int(mB.sum())
        if nA < 20 or nB < 20:
            print(f"{f'[{q[k]:.3f}, {q[k+1]:.3f})':>26}{nA:>7}{nB:>7}"
                  f"{'мало данных':>13}")
            continue
        eA, eB = np.median(R[mA, 2]), np.median(R[mB, 2])
        ratios.append(eA / max(eB, 1e-9))
        print(f"{f'[{q[k]:.3f}, {q[k+1]:.3f})':>26}{nA:>7}{nB:>7}"
              f"{eA:>13.4f}{eB:>13.4f}{ratios[-1]:>8.2f}")

    if ratios:
        med = float(np.median(ratios))
        print(f"\nмедианное A/B по корзинам: {med:.2f}")
        print("ВЫВОД:", "объяснение (i) — механизм есть, §4 плана обоснован"
              if med >= 1.3 else
              "объяснение (ii) — отдельного явления нет, заявку переписывать")
    print(f"\nДля сверки: пол кодека {floor:.4f} размаха. Если ошибки действия "
          f"в корзинах\nсопоставимы с ним, эффект тонет в собственной "
          f"погрешности кодека.")


if __name__ == "__main__":
    main()
