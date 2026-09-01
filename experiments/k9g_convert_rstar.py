"""K-9g: собрать чекпойнт Frozen-12 + R* в формате, который принимает гейт.

ЧТО ЭТО ЗА КОНФИГУРАЦИЯ. Таблица K-9f показала, что считыватель в 1.6 млн
параметров, обученный на ЗАМОРОЖЕННОМ исходном стволе, воспроизводит 102%
прироста Joint-12: 32.9% против 32.8% согласия, поза8 0.1209 против 0.1199,
знак 2.64% против 2.51%. То есть дообучение 880 млн параметров магистрали
оказалось излишним. Прежде чем заявлять это, конфигурацию надо провести через
симулятор — офлайновое согласие только что само себя дискредитировало как
предиктор успеха (33% против 87% у полной глубины при неотличимом успехе).

ПОЧЕМУ ОТДЕЛЬНЫЙ СКРИПТ, А НЕ ФЛАГ В ГЕЙТЕ. Гейт принимает чекпойнт формата
k9c: словарь весов по белому списку обучаемых префиксов. R* хранится иначе —
это state_dict модуля `Readout` из k9f с ключами `norm.*` и `head.*`. Перенос
ключей — операция, где ошибиться легко и незаметно: веса лягут, глубина
совпадёт, гейт отработает, а исполняться будет не то. Поэтому перенос
изолирован и обвешан проверками.

ЧЕТЫРЕ ПРОВЕРКИ, И НИ ОДНА НЕ ЗАМЕНЯЕТ ДРУГУЮ.
  1. Ствол в собранном чекпойнте ПОБИТОВО равен исходной модели. Не «близок»,
     а равен: конфигурация называется Frozen-12, и любое отличие означает, что
     она называется неверно.
  2. Изменились РОВНО norm и fast_head, и они действительно изменились —
     иначе мы бы собрали исходную модель и не заметили.
  3. Собранный считыватель на кэше h12 исходного ствола воспроизводит
     согласие из table.json. Это сквозная проверка: если ключи перепутаны,
     число развалится.
  4. Полнота по белому списку в обе стороны — как в K-9d.

СОГЛАСИЕ, А НЕ ПОЗА8, в третьей проверке намеренно: поза8 требует кодека и
ничего не добавляет к вопросу «легли ли веса туда, куда надо». Кодек уже
проверен в K-9e, где расхождение кэша с живым проходом составило ноль токенов
из 2.4 млн.

Запуск:
    python3 experiments/k9g_convert_rstar.py --selftest

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9g_convert_rstar.py \\
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO \\
        --rstar data/k9f/head_only.pt --table data/k9f/table.json \\
        --orig data/k9e_orig --cache data/k9_teacher_150k.npz \\
        --out data/k9g_frozen12_rstar.pt
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

N_POS = 16


def map_key(k):
    """Ключ модуля Readout -> имя параметра в модели.

    Разбиение ствол/голова зафиксировано в K-9e: `action_expert.norm` относится
    к ГОЛОВЕ (стоит вплотную перед ней), `bos_embedding` — к СТВОЛУ (вход
    потока действий на нулевом слое). Здесь переносится только голова.
    """
    if k.startswith("norm."):
        return "action_expert.norm." + k[len("norm."):]
    if k.startswith("head."):
        return "fast_head." + k[len("head."):]
    raise KeyError(f"неожиданный ключ считывателя: {k}")


def selftest():
    assert map_key("norm.weight") == "action_expert.norm.weight"
    assert map_key("head.weight") == "fast_head.weight"
    assert map_key("head.bias") == "fast_head.bias"
    for bad in ("proj.weight", "norm", "fast_head.weight"):
        try:
            map_key(bad)
        except KeyError:
            pass
        else:
            raise AssertionError(f"«{bad}» обязан быть отвергнут")

    # Белый список: что относится к стволу, а что к голове. Ошибка здесь даёт
    # чекпойнт, который загрузится и будет исполнять не ту сеть.
    HEAD = ("action_expert.norm.", "fast_head.")
    for nm in ("vlm.text_model.layers.0.mlp.up_proj.weight",
               "action_expert.layers.11.self_attn.k_proj.weight",
               "bos_embedding"):
        assert not any(nm.startswith(p) for p in HEAD), nm
    for nm in ("action_expert.norm.weight", "fast_head.bias"):
        assert any(nm.startswith(p) for p in HEAD), nm

    # Побитовое равенство — это equal, а не allclose. Проверка на настоящем
    # torch, если он доступен: разница в один ULP обязана быть замечена.
    try:
        import torch
    except ImportError:
        print("самопроверка k9g пройдена частично (torch недоступен): "
              "перенос ключей и граница ствол/голова")
        return
    a = torch.tensor([1.0, 2.0], dtype=torch.float16)
    b = a.clone()
    b[0] = torch.nextafter(b[0].float(), torch.tensor(2.0)).half()
    assert torch.equal(a, a.clone())
    assert not torch.equal(a, b), "один ULP обязан ломать равенство"
    assert torch.allclose(a, b), "allclose такую разницу пропускает"
    print("самопроверка k9g пройдена (версия «побитовый ствол»): перенос "
          "ключей, граница ствол/голова, equal против allclose")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--rstar", help="head_only.pt от k9f")
    ap.add_argument("--table", help="table.json от k9f")
    ap.add_argument("--orig", help="префикс кэша k9e для исходного ствола")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--tol-acc", type=float, default=0.001,
                    help="допуск на воспроизведение согласия из table.json")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    for need in ("ckpt", "rstar", "table", "orig", "out"):
        if not getattr(args, need):
            raise SystemExit(f"нужен --{need} (или --selftest)")

    sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"k9g sha1 {sha}")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch

    import actioncodec  # noqa: F401
    import joint12_vla as jv
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import get_cfg

    def file_sha(p):
        h = hashlib.sha1()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""):
                h.update(c)
        return h.hexdigest()[:12]

    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    model.init_joint_fast(depth=args.depth, head_dtype=dt)
    own = dict(model.named_parameters())
    trainable = [k for k in own if own[k].requires_grad]
    print(f"исходная модель загружена, обучаемых имён {len(trainable)}")

    # --- исходные веса ствола -------------------------------------------------
    HEAD_PREFIX = ("action_expert.norm.", "fast_head.")
    trunk_keys = [k for k in trainable
                  if not any(k.startswith(p) for p in HEAD_PREFIX)]
    head_keys = [k for k in trainable
                 if any(k.startswith(p) for p in HEAD_PREFIX)]
    print(f"  ствол {len(trunk_keys)} имён, голова {len(head_keys)}")
    orig_trunk = {k: own[k].detach().cpu().clone() for k in trunk_keys}
    orig_head = {k: own[k].detach().cpu().clone() for k in head_keys}

    # --- считыватель R* -------------------------------------------------------
    rs = torch.load(args.rstar, map_location="cpu", weights_only=False)
    if not isinstance(rs, dict):
        raise SystemExit(f"{args.rstar}: ожидался state_dict")
    new_head = {}
    for k, v in rs.items():
        nk = map_key(k)
        if nk not in own:
            raise SystemExit(f"{k} -> {nk}: такого параметра в модели нет")
        if tuple(own[nk].shape) != tuple(v.shape):
            raise SystemExit(f"{nk}: форма {tuple(own[nk].shape)} против "
                             f"{tuple(v.shape)}")
        if not torch.isfinite(v).all():
            raise SystemExit(f"{k}: есть nan или inf")
        new_head[nk] = v.detach().cpu().clone()
    missing_head = [k for k in head_keys if k not in new_head]
    if missing_head:
        raise SystemExit(f"считыватель не покрывает {missing_head}")
    print(f"  R* из {args.rstar} (sha {file_sha(args.rstar)}): "
          f"{len(new_head)} тензоров")

    # --- сборка ---------------------------------------------------------------
    # СТВОЛ КЛАДЁТСЯ В ИСХОДНОМ DTYPE. Переход fp16 -> fp32 точен, обратный —
    # нет; гейт всё равно приводит к fp32 при загрузке. Хранить исходное
    # значение — единственный способ сохранить побитовое равенство.
    state = {}
    for k in trunk_keys:
        state[k] = orig_trunk[k].clone()
    for k, v in new_head.items():
        state[k] = v.clone()

    # ПРОВЕРКА 1: ствол побитово исходный.
    diff = [k for k in trunk_keys if not torch.equal(state[k], orig_trunk[k])]
    if diff:
        raise SystemExit(f"ствол изменился в {len(diff)} тензорах: {diff[:5]}")
    print(f"  ствол побитово совпадает с исходным: {len(trunk_keys)} тензоров")

    # ПРОВЕРКА 2: голова изменилась, и изменилась вся.
    same = [k for k in head_keys if torch.equal(state[k].float(),
                                                orig_head[k].float())]
    if same:
        raise SystemExit(
            f"голова не изменилась в {same}: собран исходный чекпойнт, а не "
            f"R*. Проверьте, тот ли файл подан в --rstar.")
    for k in head_keys:
        d = float((state[k].float() - orig_head[k].float()).abs().max())
        print(f"    {k}: max|Δ| к исходной = {d:.4e}")

    # ПРОВЕРКА 3: полнота по белому списку в обе стороны, как в K-9d.
    stray = [k for k in state
             if not any(k.startswith(p) or k == p.rstrip(".")
                        for p in model.trainable_prefixes)]
    if stray:
        raise SystemExit(f"вне белого списка: {stray[:5]}")
    missing = [k for k in trainable if k not in state]
    if missing:
        raise SystemExit(f"не покрыто {len(missing)} обучаемых: {missing[:5]}")

    # --- ПРОВЕРКА 4: сквозная, против table.json ------------------------------
    tab = json.load(open(args.table))
    want = tab["test_cells"]["T0Rstar"]["acc_teacher"]
    md_o = json.load(open(args.orig + ".json"))
    if md_o["trunk"] != "original":
        raise SystemExit(f"{args.orig}: ствол «{md_o['trunk']}», нужен "
                         f"«original»")
    if md_o["depth"] != args.depth:
        raise SystemExit(f"{args.orig}: глубина {md_o['depth']}")
    H = np.load(md_o["h12_file"], mmap_mode="r")
    z = np.load(args.cache, allow_pickle=True)
    q_teach = z["teacher_codes_q0"].astype(np.int64)
    ite = np.where(z["split"] == "test")[0]
    if H.shape[0] != len(q_teach):
        raise SystemExit("кэш h12 и кэш учителя разной длины")

    # Считыватель собирается ИЗ ЧЕКПОЙНТА, а не из файла R*: проверяется то,
    # что будет исполнено, а не то, что лежало на входе.
    import copy
    norm = copy.deepcopy(model.action_expert.norm).float().cpu()
    head = copy.deepcopy(model.fast_head).float().cpu()
    norm.load_state_dict({k[len("action_expert.norm."):]: state[k].float()
                          for k in state if k.startswith("action_expert.norm.")})
    head.load_state_dict({k[len("fast_head."):]: state[k].float()
                          for k in state if k.startswith("fast_head.")})
    norm, head = norm.to(dev), head.to(dev)
    acc, n = 0.0, 0
    with torch.no_grad():
        for i0 in range(0, len(ite), args.batch):
            sel = ite[i0:i0 + args.batch]
            h = torch.from_numpy(np.asarray(H[sel])).to(dev).float()
            pc = head(norm(h)).argmax(-1).cpu().numpy()
            acc += float((pc == q_teach[sel]).mean()) * len(sel)
            n += len(sel)
    got = acc / n
    print(f"  согласие на test из собранного чекпойнта: {got:.4%}, "
          f"в table.json {want:.4%}, расхождение {abs(got - want) * 100:.3f} пп")
    if abs(got - want) > args.tol_acc:
        raise SystemExit(
            f"собранный чекпойнт не воспроизводит table.json "
            f"({abs(got - want) * 100:.3f} пп при допуске "
            f"{args.tol_acc * 100:.1f} пп). Ключи перенесены неверно.")

    # --- запись ---------------------------------------------------------------
    obj = dict(state=state, depth=args.depth, sha1=sha,
               source="frozen12_rstar",
               base_ckpt=args.ckpt,
               rstar_file=os.path.abspath(args.rstar),
               rstar_sha1=file_sha(args.rstar),
               table_file=os.path.abspath(args.table),
               table_sha1=file_sha(args.table),
               joint12_vla_sha1=hashlib.sha1(
                   open(jv.__file__, "rb").read()).hexdigest()[:12],
               control_cfg=tab.get("control_cfg"),
               test_acc_teacher=got,
               args=dict(vars(args), cache=os.path.abspath(args.cache)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    torch.save(obj, args.out)
    print(f"\nсохранено: {args.out} (sha весов {file_sha(args.out)})")
    print(f"  source=frozen12_rstar, глубина {args.depth}, "
          f"lr/сид/эпоха: {tab.get('control_cfg')}")
    print("  ствол исходный побитово; обучены только norm и fast_head")


if __name__ == "__main__":
    main()
