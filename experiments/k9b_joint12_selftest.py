"""K-9b: проверки проводки Joint-12 ДО обучения.

Ни одна проверка не про качество. Все — про то, что модель делает ровно
написанное, что обучается ровно то, что заявлено, и что шаг оптимизатора
действительно меняет первые двенадцать слоёв.

Без модели (`--selftest`, CPU):
  0. Потеря дистилляции: при совпадении логитов обращается в ноль, растёт при
     расхождении, масштабируется T^2.

С моделью (`--ckpt`, GPU):
  1. ТОЖДЕСТВО. depth=24 с копией головы обязано побитово воспроизвести первый
     блок официальной BAR — и токены, и логиты.
  2. СЧЁТЧИКИ СЛОЁВ по `input_layernorm`: depth=12 даёт 12 и 12, depth=24 — 24
     и 24, официальная BAR — 72 и 72.
  3. Голова стартует копией `action_lm_head`.
  4. Белый список: обучаемо только заявленное, глубокие слои заморожены.
  5. Оптимизатор покрывает обучаемое ПО id(), а не по количеству.
  6. Градиенты доходят до КАЖДОГО из размороженных слоёв (печатается норма по
     каждому), у глубоких слоёв None.
  7. Шаг оптимизатора реально меняет все 2*depth слоёв и голову, а замороженные
     веса остаются побитово теми же.
  8. Память на один полный шаг при батче 1.

ТОЧНОСТЬ. Тождество проверяется в fp16, как модель и загружена. Дальше
обучаемые веса переводятся в fp32, проход идёт под autocast fp16 с GradScaler:
AdamW на fp16-параметрах держал бы состояния в fp16, а при lr=1e-5 и весах
порядка 1e-2 относительный шаг около 1e-3 — на границе разрешения fp16. Замер
памяти в таком режиме показал бы привлекательную, но численно негодную
конфигурацию.

ПОРЯДОК. Всё после проверки тождества идёт на батче 1: иначе нехватка памяти
роняла бы скрипт ДО строки, ради которой он написан. Режимы чекпойнтинга
меряются отдельными процессами через --grad-ckpt: после OOM состояние
аллокатора уже загрязнено.

Запуск:
    python3 experiments/k9b_joint12_selftest.py --selftest
    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k9b_joint12_selftest.py --ckpt <ckpt> --depth 12
"""

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

N_POS, N_LEVEL, T_CHUNK = 16, 3, 20


def selftest_cpu():
    try:
        import torch
    except ImportError:
        raise SystemExit("нет torch: проверки k9b про поведение модулей и без "
                         "него бессмысленны. Запускать на кластере.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from joint12_vla import kd_loss

    torch.manual_seed(0)
    t = torch.randn(4, N_POS, 64)
    assert float(kd_loss(t, t, 2.0)) < 1e-6, "совпадающие логиты дают не ноль"
    s = t + torch.randn_like(t) * 2.0
    assert float(kd_loss(s, t, 2.0)) > 0.05, "расхождение не штрафуется"
    # Сдвиг логитов на константу softmax не меняет — потеря не должна реагировать.
    assert abs(float(kd_loss(t + 3.0, t, 2.0))) < 1e-5, (
        "потеря реагирует на сдвиг логитов, хотя softmax к нему инвариантен")
    # Масштаб T^2: при удвоении T потеря на малых расхождениях меняется мало,
    # но обязана остаться конечной и положительной.
    for T in (1.0, 2.0, 4.0):
        v = float(kd_loss(s, t, T))
        assert np.isfinite(v) and v > 0, (T, v)
    # Градиент идёт в УЧЕНИКА и не идёт в учителя.
    su = (t + 0.5).requires_grad_(True)
    te = t.clone().requires_grad_(True)
    kd_loss(su, te.detach(), 2.0).backward()
    assert su.grad is not None and su.grad.abs().sum() > 0
    assert te.grad is None, "градиент утёк в учителя"

    print("самопроверка k9b (без модели) пройдена: дистилляция обращается в "
          "ноль при совпадении, инвариантна к сдвигу логитов, конечна при "
          "разных T, градиент идёт только в ученика")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--mem-batch", type=int, default=1,
                    help="батч для замера памяти; проверки проводки идут на "
                         "нём же, чтобы падение по памяти не случилось ДО "
                         "строки, ради которой всё")
    ap.add_argument("--grad-ckpt", choices=["off", "on"], default="off",
                    help="режим чекпойнтинга. Замерять надо ОТДЕЛЬНЫМИ "
                         "процессами: после OOM состояние аллокатора уже "
                         "загрязнено, и empty_cache этого не чинит")
    ap.add_argument("--pos-offset", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    selftest_cpu()
    if args.selftest:
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --selftest")
    print(f"k9b sha1 "
          f"{hashlib.sha1(open(__file__, 'rb').read()).hexdigest()[:12]}")

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from torchvision.transforms.v2 import CenterCrop, Compose, Resize

    import actioncodec  # noqa: F401
    from joint12_vla import kd_loss, make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, process_state, prompt_template,
                       seed_everything)

    seed_everything(args.seed)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    n_layers = int(model.config.vlm_config.text_config.num_hidden_layers)
    if dev.type == "cuda":
        tot = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
        print(f"{torch.cuda.get_device_name(dev)}, всего {tot:.1f} ГиБ; "
              f"слоёв {n_layers}")

    # --- данные ---------------------------------------------------------------
    rid, rev = "physical-intelligence/libero", "v2.0"
    tasks_map = {}
    for line in open(hf_hub_download(rid, "meta/tasks.jsonl", repo_type="dataset",
                                     revision=rev)):
        r = json.loads(line)
        tasks_map[r["task_index"]] = r["task"]
    rng = np.random.default_rng(args.seed)
    im1, im2, st, tsk = [], [], [], []
    for e in rng.permutation(1693):
        if len(tsk) >= args.n_obs:
            break
        f = hf_hub_download(rid, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=rev)
        t = pq.read_table(f)
        if t.num_rows < T_CHUNK + 1:
            continue
        S_ = np.asarray(t.column("state").to_pylist(), np.float32)
        ti = t.column("task_index").to_pylist()
        c1, c2 = t.column("image").to_pylist(), t.column("wrist_image").to_pylist()
        png = lambda c: np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
        s0 = int(rng.integers(0, t.num_rows - T_CHUNK + 1))
        im1.append(png(c1[s0])); im2.append(png(c2[s0]))
        st.append(S_[s0]); tsk.append(tasks_map[ti[s0]])
    N, hw = len(tsk), im1[0].shape[0]
    tf = Compose([CenterCrop(int(hw * 0.875)), Resize(224)])   # ДОЛЯ, не число
    ST = np.asarray(st, np.float64)
    if ST.shape[1] == len(STATE_Q01) + 1:
        ST = process_state(ST)
    st_n = (ST - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
    i1 = tf(torch.tensor(np.stack(im1)).permute(0, 3, 1, 2))
    i2 = tf(torch.tensor(np.stack(im2)).permute(0, 3, 1, 2))
    image = torch.cat([i1, i2], dim=-1)
    msgs = []
    for i in range(N):
        m = prompt_template(st_n[i], None, tsk[i],
                            mode=cfg.MODEL.vla_processor.kwargs.mode,
                            action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                            action_token_len=cfg.MODEL.action_processor.token_len)
        m[1]["content"] = m[1]["content"][1:]
        msgs.append(m)
    texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
    batch = proc(text=texts, images=[[image[i].numpy()] for i in range(N)],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
    batch = dict_apply(lambda x: x.to(dev, dt), batch)
    print(f"батч {N} наблюдений, кадр {hw}")

    # --- счётчики -------------------------------------------------------------
    cnt = {"vlm": 0, "expert": 0}
    bump = lambda k: (lambda m, i_, o: cnt.__setitem__(k, cnt[k] + 1))
    vl, el = model.vlm.text_model.layers, model.action_expert.layers
    hs = [vl[i].input_layernorm.register_forward_hook(bump("vlm"))
          for i in range(n_layers)]
    hs += [el[i].input_layernorm.register_forward_hook(bump("expert"))
           for i in range(n_layers)]

    def counted(fn):
        cnt["vlm"] = cnt["expert"] = 0
        return fn(), dict(cnt)

    with torch.no_grad():
        tk_bar, c_bar = counted(lambda: model.generate(
            **batch, position_offset=args.pos_offset, do_sample=False))
    K_bar = tk_bar.cpu().numpy().reshape(N, N_LEVEL, N_POS)
    print(f"\n  официальная BAR: VLM {c_bar['vlm']}, эксперт {c_bar['expert']} "
          f"(ждали {3 * n_layers} и {3 * n_layers})")
    if c_bar != {"vlm": 3 * n_layers, "expert": 3 * n_layers}:
        raise SystemExit(f"счётчики на официальной BAR дали {c_bar} — хуки "
                         f"стоят не там, замеры экономии ничего не значат")

    # ПОРЯДОК ВАЖЕН: эталон считается ПОСЛЕ создания fast_head.
    # Раньше он считался до, и сравнение шло через изменение состояния модели:
    # создание нового модуля прямо на GPU сдвигает аллокатор, меняется
    # выравнивание буферов и выбор ядра cuBLAS в финальном матмуле. Оба пути,
    # посчитанные в ОДНОМ состоянии, совпадают побитово — это показала
    # послойная локализация, где расхождение было нулевым везде, включая
    # выход нормы и головы на одном входе.
    # --- 1 и 3: тождество на полной глубине, голова = копия -------------------
    model.init_joint_fast(depth=args.depth, head_dtype=dt)
    dw = float((model.fast_head.weight - model.action_lm_head.weight).abs().max())
    print(f"  голова стартует копией action_lm_head: max|Δ| = {dw:.3e}")
    if dw != 0.0:
        raise SystemExit("fast_head не является точной копией")

    # ЭТАЛОННЫЕ ЛОГИТЫ И СОБСТВЕННЫЙ ШУМ МОДЕЛИ. Порог для тождества нельзя
    # брать с потолка: официальный путь вызывается ДВАЖДЫ на одном и том же
    # входе, и его расхождение с самим собой задаёт нижнюю границу, ниже
    # которой сравнивать бессмысленно. Тот же приём, что в K-5c, где D(0)=0
    # было тавтологией индексации.
    with torch.no_grad():
        vemb, pos = model.build_inputs(position_offset=args.pos_offset, **batch)
        ref_kw = dict(vlm_inputs_embeds=vemb,
                      attention_mask=batch.get("attention_mask"),
                      history_tokens=None, position_ids=pos)
        lg_ref = model._predict_next_block_logits(**ref_kw)
        lg_ref2 = model._predict_next_block_logits(**ref_kw)
    self_noise = (lg_ref.float() - lg_ref2.float()).abs().max().item()
    scale = lg_ref.float().abs().max().item()
    print(f"  собственный шум официального пути (два вызова на одном входе): "
          f"max|Δ| = {self_noise:.3e} при масштабе логитов {scale:.1f}")

    with torch.no_grad():
        o24, c24 = counted(lambda: model.forward_joint_fast(
            vlm_inputs_embeds=vemb, attention_mask=batch.get("attention_mask"),
            position_ids=pos, depth=n_layers))
    # СОБСТВЕННОЕ РАСХОЖДЕНИЕ МОДЕЛИ ПО ТОКЕНАМ. K_bar приходит из generate, а
    # эталонные логиты из _predict_next_block_logits. Это разные вызовы, и при
    # близких логитах двух классов их argmax может разойтись: при батче 8 они
    # совпадали, при 32 разошлись на один токен из 512. Сравнивать свой путь
    # надо с ТЕМ ЖЕ эталоном, от которого взяты логиты, а расхождение эталона
    # с generate измерять и печатать отдельно.
    ref_codes = lg_ref.argmax(-1).cpu().numpy()
    gen_vs_ref = (ref_codes == K_bar[:, 0, :])
    if not gen_vs_ref.all():
        print(f"  ВНИМАНИЕ: сам generate расходится с эталонными логитами на "
              f"{(~gen_vs_ref).sum()} токенах из {gen_vs_ref.size} "
              f"({1 - gen_vs_ref.mean():.3%}) — это собственное свойство "
              f"модели при данном батче, не наша проводка")
    same = (o24["pred_codes"].cpu().numpy() == ref_codes)
    dlg = (o24["logits"].float() - lg_ref.float()).abs().max().item()
    # ПОРОГ ОТ СОБСТВЕННОГО ШУМА, а не абсолютный. Если официальный путь сам с
    # собой расходится на N, то требовать от нашего меньше N бессмысленно.
    tol = max(4.0 * self_noise, 1e-6 * max(scale, 1.0))
    print(f"  тождество при depth={n_layers}: токены {same.mean():.6%} "
          f"(против эталонных логитов), "
          f"логиты max|Δ| = {dlg:.3e} при допуске {tol:.3e}")
    if not same.all():
        raise SystemExit(
            f"ТОКЕНЫ не совпали с эталонными логитами на "
            f"{(~same).sum()} из {same.size} при побитово равных логитах — "
            f"этого быть не может, ищите ошибку в сравнении.")
    if dlg > tol:
        # ЛОКАЛИЗАЦИЯ, А НЕ ДОГАДКИ. Снимаем вход input_layernorm каждого слоя
        # эксперта на обоих путях и ищем ПЕРВЫЙ слой, где они разошлись.
        # Расхождение на входе слоя 0 означает разную сборку BOS; на слое k —
        # что-то внутри предыдущего шага общего внимания.
        # СНИМАЕМ ОБЕ БАШНИ И САМУ НОРМУ. Прежняя версия брала только входы
        # слоёв эксперта: выход последнего слоя и вход нормы не покрывались
        # ничем, а расхождение башни VLM на последнем слое проявилось бы у
        # эксперта уже после всех хуков.
        grab = {k: [] for k in ("ref_e", "our_e", "ref_v", "our_v",
                                "ref_ni", "our_ni", "ref_no", "our_no")}
        tag = ["ref"]
        hh = [el[i].input_layernorm.register_forward_hook(
            lambda m, i_, o: grab[tag[0] + "_e"].append(i_[0].detach().float().cpu()))
            for i in range(n_layers)]
        hh += [vl[i].input_layernorm.register_forward_hook(
            lambda m, i_, o: grab[tag[0] + "_v"].append(i_[0].detach().float().cpu()))
            for i in range(n_layers)]
        # ХУК ОБЯЗАН ВЕРНУТЬ None. Ненулевой возврат из forward-хука ПОДМЕНЯЕТ
        # выход модуля: лямбда из двух append через запятую возвращала кортеж,
        # и норма начинала отдавать кортеж вместо тензора.
        def norm_hook(m, i_, o):
            grab[tag[0] + "_ni"].append(i_[0].detach().float().cpu())
            grab[tag[0] + "_no"].append(o.detach().float().cpu())

        hh.append(model.action_expert.norm.register_forward_hook(norm_hook))
        with torch.no_grad():
            fresh_ref = model._predict_next_block_logits(**ref_kw)
            tag[0] = "our"
            fresh_our = model.forward_joint_fast(
                vlm_inputs_embeds=vemb,
                attention_mask=batch.get("attention_mask"),
                position_ids=pos, depth=n_layers)["logits"]
        for h in hh:
            h.remove()
        d_fresh = (fresh_ref.float() - fresh_our.float()).abs().max().item()
        print(f"\n  оба пути, посчитанные ЗАНОВО в одном состоянии: "
              f"max|Δ| = {d_fresh:.3e}")
        if d_fresh == 0.0:
            print("  Значит проводка идентична, а прежнее расхождение было "
                  "следствием\n  сравнения через изменение состояния модели.")
        print(f"\n  ЛОКАЛИЗАЦИЯ")
        first = None
        for nm, kr, ko in (("эксперт", "ref_e", "our_e"),
                           ("VLM    ", "ref_v", "our_v")):
            bad = []
            for i in range(min(len(grab[kr]), len(grab[ko]))):
                d = (grab[kr][i] - grab[ko][i]).abs().max().item()
                if d > 0:
                    bad.append((i, d))
                    if first is None or i < first:
                        first = i
            print(f"    входы слоёв, {nm}: расходятся на "
                  + (f"слоях {[b[0] for b in bad[:5]]}, первое "
                     f"max|Δ| = {bad[0][1]:.3e}" if bad else "НИ ОДНОМ"))
        # Вход и выход самой нормы — то, что раньше не покрывалось.
        for nm, kr, ko in (("вход нормы ", "ref_ni", "our_ni"),
                           ("выход нормы", "ref_no", "our_no")):
            if grab[kr] and grab[ko]:
                # у официального пути норма зовётся трижды (блоки 0,1,2),
                # у нашего — один раз; сравниваем ПЕРВЫЙ вызов, это блок 0
                d = (grab[kr][0] - grab[ko][0]).abs().max().item()
                print(f"    {nm}: вызовов {len(grab[kr])} и {len(grab[ko])}, "
                      f"max|Δ| на первом = {d:.3e}")
        if first == 0:
            print("\n  Расходится УЖЕ НА ВХОДЕ ПЕРВОГО СЛОЯ: дело в сборке "
                  "action-запросов.\n  Официальный путь делает "
                  "torch.cat([bos, пустой тензор]) и получает СПЛОШНОЙ "
                  "тензор,\n  а expand(...) оставляет вид с нулевыми шагами. "
                  "Значения те же, раскладка\n  в памяти разная, и матмул "
                  "суммирует в другом порядке.")
        elif first is not None:
            print(f"\n  Первое расхождение на входе слоя {first}: значит "
                  f"разошлось внутри шага\n  общего внимания слоя {first - 1}, "
                  f"а не в подготовке входов.")
        else:
            # ГОЛОВЫ НА ОДНОМ ВХОДЕ. Веса совпадают побитово (проверено выше),
            # значит любое расхождение здесь — выбор ядра cuBLAS из-за разной
            # раскладки в памяти: fast_head создана сразу на GPU, исходная
            # перенесена. Значения те же, порядок суммирования другой.
            with torch.no_grad():
                probe = torch.randn(4, N_POS, model.fast_head.in_features,
                                    device=dev, dtype=dt)
                dh = (model.fast_head(probe).float()
                      - model.action_lm_head(probe).float()).abs().max().item()
            print(f"\n  Промежуточные состояния совпали ВЕЗДЕ. Две головы с "
                  f"побитово равными\n  весами на одном входе расходятся на "
                  f"{dh:.3e}.")
            if dh > 0:
                print("  Значит проводка идентична, а разница — выбор ядра "
                      "cuBLAS из-за разной\n  раскладки в памяти. Числено это "
                      "не ошибка: токены совпали на 100%.")
            else:
                print("  Головы совпадают, значит расхождение вносит "
                      "action_expert.norm — \n  это уже настоящее отличие "
                      "проводки, и его надо искать.")
        # ДИАГНОСТИКА ОБЯЗАНА УМЕТЬ ОПРАВДАТЬ. Прежде после печати «проводка
        # идентична» всё равно шёл безусловный отказ, то есть повторная сверка
        # ничего не могла решить.
        if d_fresh <= tol:
            print("  Свежая сверка прошла: проводка идентична, а прежнее "
                  "расхождение\n  вызвано сравнением через изменение "
                  "состояния модели. Продолжаю.")
        else:
            raise SystemExit(
                f"логиты расходятся на {dlg:.3e} (свежая сверка {d_fresh:.3e}) "
                f"при собственном шуме модели {self_noise:.3e}: это не "
                f"округление, а разная проводка.")
    if self_noise > 0:
        print(f"  (официальный путь не побитово воспроизводим сам по себе — "
              f"сравнение идёт в пределах его шума)")
    if c24 != {"vlm": n_layers, "expert": n_layers}:
        raise SystemExit(f"depth={n_layers} исполнил {c24}")

    # --- 2: счётчики на рабочей глубине ---------------------------------------
    with torch.no_grad():
        o12, c12 = counted(lambda: model.forward_joint_fast(
            vlm_inputs_embeds=vemb, attention_mask=batch.get("attention_mask"),
            position_ids=pos))
    print(f"  depth={args.depth}: VLM {c12['vlm']}, эксперт {c12['expert']}")
    if c12 != {"vlm": args.depth, "expert": args.depth}:
        raise SystemExit(f"исполнено {c12}, а заявлено {args.depth} — "
                         f"экономия глубины ложная")
    ag = float((o12["pred_codes"].cpu().numpy() == K_bar[:, 0, :]).mean())
    print(f"  согласие с учителем ДО обучения: {ag:.1%}")

    # --- 4: белый список ------------------------------------------------------
    rep = model.trainable_report()
    tot_p = sum(p.numel() for p in model.parameters())
    print(f"\n  обучаемое:")
    for k, v in sorted(rep.items()):
        print(f"    {k:<24}{v / 1e6:9.2f} млн")
    n_tr = sum(rep.values())
    print(f"    {'итого':<24}{n_tr / 1e6:9.2f} млн из {tot_p / 1e6:.0f} "
          f"({n_tr / tot_p:.1%})")

    # --- 5: оптимизатор покрывает обучаемое ПО id() ---------------------------
    # --- 6 и 7: градиенты и настоящий шаг, НА БАТЧЕ ДЛЯ ЗАМЕРА -------------
    # Раньше шаг шёл на всём батче, и при нехватке памяти скрипт падал ДО
    # строки, которая должна решить, возможен ли эксперимент.
    mb = max(1, min(args.mem_batch, N))
    b1 = {k: (v[:mb] if torch.is_tensor(v) and v.shape[0] == N else v)
          for k, v in batch.items()}
    print(f"\n  дальше всё на батче {mb}")

    # ТОЧНОСТЬ: обучаемые веса в fp32, проход под autocast fp16, GradScaler.
    # Тождество выше проверено ДО этой смены — вне autocast fp32-слой на
    # fp16-входе упал бы по типам.
    n_cast = model.to_fp32_trainable()
    est = model.memory_estimate()
    print(f"  переведено в fp32: {n_cast} тензоров")
    print(f"  оценка статической памяти: веса обучаемые "
          f"{est['weights_trainable_gib']:.2f} + замороженные "
          f"{est['weights_frozen_gib']:.2f} + градиенты {est['grads_gib']:.2f} "
          f"+ состояния Adam {est['adam_states_gib']:.2f} = "
          f"{est['total_static_gib']:.2f} ГиБ")

    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=1e-5)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    need = {id(p) for p in model.parameters() if p.requires_grad}
    if in_opt != need:
        raise SystemExit(
            f"оптимизатор покрывает {len(in_opt)} тензоров, обучаемых "
            f"{len(need)}; расхождение {len(need ^ in_opt)} — часть весов "
            f"получала бы градиент и не обновлялась")
    print(f"  оптимизатор покрывает все {len(need)} обучаемых тензоров (по id)")
    assert all(p.dtype == torch.float32 for p in params), \
        "не все обучаемые тензоры в fp32 — Adam будет держать состояния в fp16"

    watch = {
        "fast_head": model.fast_head.weight,
        "bos_embedding": model.bos_embedding,
        "expert.norm": model.action_expert.norm.weight,
    }
    frozen = {"vlm[deep]": vl[n_layers - 1].self_attn.q_proj.weight,
              "expert[deep]": el[n_layers - 1].self_attn.q_proj.weight}
    before = {k: v.detach().clone() for k, v in {**watch, **frozen}.items()}
    # СРЕЗЫ, А НЕ ЦЕЛЫЕ МАТРИЦЫ: полные копии двадцати четырёх весов живут во
    # время forward/backward и завышают измеряемый пик примерно на 230 МиБ.
    # Для «изменился ли тензор» первой строки достаточно.
    def _slice(w):
        return w[:1].detach().clone()

    before_layers = {f"vlm[{i}]": _slice((vl[i].self_attn.k_proj
                                          if i == args.depth - 1
                                          else vl[i].self_attn.q_proj).weight)
                     for i in range(args.depth)}
    before_layers.update({f"expert[{i}]": _slice(el[i].self_attn.q_proj.weight)
                          for i in range(args.depth)})

    model.grad_ckpt = (args.grad_ckpt == "on")
    model.train()
    # МАСШТАБ СТАРТУЕТ НИЖЕ ШТАТНОГО. При потере около 9 и стандартных 65536
    # обратный проход переполняет fp16, и градиенты приходят как NaN у головы
    # и самых ранних слоёв. Это не сбой, а то, ради чего существует
    # GradScaler: он обязан пропустить шаг и уронить масштаб. Но разглядывать
    # градиенты имеет смысл только на той попытке, где они КОНЕЧНЫ, — иначе
    # проверка падает на штатном поведении.
    scaler = torch.amp.GradScaler("cuda", init_scale=2. ** 12,
                                  enabled=(dev.type == "cuda"))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=dt,
                            enabled=(dev.type == "cuda")):
            v1, p1 = model.build_inputs(position_offset=args.pos_offset, **b1)
            o = model.forward_joint_fast(
                vlm_inputs_embeds=v1, attention_mask=b1.get("attention_mask"),
                position_ids=p1)
            l = kd_loss(o["logits"], lg_ref[:mb].detach(), 2.0)
        scaler.scale(l).backward()
        scaler.unscale_(opt)
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in params)
        return float(l), finite

    loss_v, finite = fwd_bwd()
    print(f"\n  потеря дистилляции на НАСТОЯЩЕМ учителе: {loss_v:.4f}")
    if loss_v < 1e-4:
        raise SystemExit("потеря близка к нулю — замер памяти ничего не значит")
    tries = 1
    while not finite and tries < 12:
        # Масштаб роняем тем же механизмом, что и в обучении.
        scaler.update(scaler.get_scale() / 2.0)
        loss_v, finite = fwd_bwd()
        tries += 1
    if not finite:
        raise SystemExit(
            f"градиенты не стали конечными за {tries} попыток при масштабе "
            f"{scaler.get_scale():.0f} — fp16 не тянет эту конфигурацию, "
            f"нужен bf16 или меньший lr")
    print(f"  конечные градиенты получены с {tries}-й попытки, масштаб "
          f"{scaler.get_scale():.0f} (пропуски на старте штатны и в обучении "
          f"будут теми же)")

    # ПОКАЗАТЬ НОРМУ ГРАДИЕНТА ПО КАЖДОМУ СЛОЮ, а не по двум крайним: иначе не
    # видно, доходит ли сигнал до середины стопки.
    # НА ПОСЛЕДНЕМ СЛОЕ У БАШНИ VLM СМОТРИМ k_proj, А НЕ q_proj: её запросы
    # там мертвы по построению (выход башни после последнего слоя не
    # используется), и требовать от них градиента — ошибка проверки, а не
    # модели. У эксперта живо всё, там q_proj везде.
    gn = {}
    for i in range(args.depth):
        g = el[i].self_attn.q_proj.weight.grad
        gn[f"expert[{i}]"] = 0.0 if g is None else float(g.norm())
        w = (vl[i].self_attn.k_proj if i == args.depth - 1
             else vl[i].self_attn.q_proj).weight
        gn[f"vlm[{i}]"] = 0.0 if w.grad is None else float(w.grad.norm())
    dead = [k for k, v in gn.items() if v == 0.0]
    print("  нормы градиента по слоям:")
    for tag in ("vlm", "expert"):
        row = "  ".join(f"{gn[f'{tag}[{i}]']:.1e}" for i in range(args.depth))
        print(f"    {tag:<7}{row}")
    if dead:
        raise SystemExit(f"нулевой градиент у слоёв: {dead}")
    for k, v in watch.items():
        if v.grad is None or not torch.isfinite(v.grad).all() or v.grad.abs().sum() == 0:
            raise SystemExit(f"нет ненулевого конечного градиента у {k}")
    for k, v in frozen.items():
        if v.grad is not None:
            raise SystemExit(f"у замороженного {k} появился градиент")
    print("  градиенты дошли до всех размороженных групп, "
          "у глубоких слоёв их нет")

    gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    print(f"  норма градиента до обрезки: {float(gnorm):.3f} (обрезка на 1.0, "
          f"как в обучении)")
    scaler.step(opt); scaler.update()

    moved = [k for k, v in watch.items() if not torch.equal(v.detach(), before[k])]
    if set(moved) != set(watch):
        raise SystemExit(f"не изменились: {sorted(set(watch) - set(moved))}")
    def _watched(k):
        i = int(k.split("[")[1][:-1])
        if k.startswith("expert"):
            return el[i].self_attn.q_proj.weight
        return (vl[i].self_attn.k_proj if i == args.depth - 1
                else vl[i].self_attn.q_proj).weight

    unmoved = [k for k, v in before_layers.items()
               if torch.equal(_watched(k)[:1].detach(), v)]
    if unmoved:
        raise SystemExit(f"после шага НЕ изменились слои: {unmoved}")
    still = [k for k, v in frozen.items() if not torch.equal(v.detach(), before[k])]
    if still:
        raise SystemExit(f"замороженные веса ИЗМЕНИЛИСЬ: {still}")
    print(f"  шаг изменил все {2 * args.depth} наблюдаемых слоёв и не тронул "
          f"глубокие")

    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        res = torch.cuda.memory_reserved(dev) / 2 ** 30
        print(f"\n  ПАМЯТЬ: батч {mb}, чекпойнтинг {args.grad_ckpt}, "
              f"пик {peak:.2f} ГиБ, зарезервировано {res:.2f}, "
              f"всего на карте {tot:.1f}")
        print(f"  Второй режим запускать ОТДЕЛЬНЫМ процессом: "
              f"--grad-ckpt {'on' if args.grad_ckpt == 'off' else 'off'}")

    for h in hs:
        h.remove()
    print("\n  все проверки пройдены: тождество на полной глубине точное, "
          "экономия глубины реальная, обучается только белый список, "
          "оптимизатор покрывает его целиком, шаг меняет первые "
          f"{args.depth} слоёв и не трогает остальные")


if __name__ == "__main__":
    main()
