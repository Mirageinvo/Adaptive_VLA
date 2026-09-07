"""K-11b: проверки тождества HiCoRA до какого бы то ни было обучения.

ЗАЧЕМ. Голова поправки инициализирована нулями, поэтому ДО обучения HiCoRA
обязана быть неотличима от базового coarse-выхода головы q0. Если это не так,
всё последующее сравнение идёт не с той опорой, а «улучшение» может оказаться
просто расхождением конвейеров. Тождество проверяется ДО обучения, потому что
после него отличить одно от другого уже нечем.

ЧТО ПРОВЕРЯЕТСЯ.
   1. Токены q0 из `forward_taps` совпадают ПОБИТОВО с `forward_joint_fast`
      на глубине 12 — то есть черновик HiCoRA есть в точности черновик
      Joint12, а не похожая на него величина.
   2. При нулевой голове z == z0 побитово.
   3. Декодированные действия совпадают в пределах численного допуска.
   4. Знак схвата совпадает на 100%.
   5. Каждый из 24 слоёв исполняется РОВНО ОДИН раз за проход.
   6. `generate` не вызывается ни разу — «один проход» не должно оказаться
      двумя, из которых второй спрятан внутри.
   7. Декодер ActionCodec вызывается ровно один раз на выдачу действия.
   8. Всё перечисленное — и на батче 1, и на батче 10.
   9. После одного шага обучения у головы поправки НЕНУЛЕВОЙ градиент.
  10. В замороженном режиме градиент не попадает в ствол, голову q0 и кодек.

ПЛЮС КОНТРФАКТИЧЕСКАЯ ПРОВЕРКА. Если подменить предсказанный q0, обязаны
измениться и вход головы поправки, и целевой остаток. Без неё «поправка к
реально предсказанному черновику» осталась бы заявлением: голова, полностью
игнорирующая z0, прошла бы все девять проверок выше.

ПЛЮС ПОВТОРНАЯ СВЕРКА ПРОИСХОЖДЕНИЯ. K-11a уже проверил манифест, книги и
декодер; здесь то же самое сверяется с ЗАПИСАННЫМ в кэш, чтобы между сбором
и обучением ничего не подменилось. K-11b не первое место проверки, а второе.

ПОЧЕМУ НУЛЕВОЙ ГРАДИЕНТ БЫЛ БЫ ПЛОХ. Нулевая инициализация последнего слоя
даёт нулевую поправку — это и требуется. Но если она же даёт нулевой
градиент, обучение никогда не стронется, а тождество будет достигнуто мёртвой
головой. Проверка 9 отличает одно от другого; ровно этой проверки не хватало
в первой версии `hicora_vla.py`, и она прошла бы на мёртвой голове.

Запуск:
    python3 experiments/k11b_hicora_identity.py --selftest

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k11b_hicora_identity.py --ckpt <base> \\
        --cache data/k11a_joint12 --joint-ckpt data/k9d_ep3.pt \\
        --device cuda:1 --out data/k11b_identity.json
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k11a_build_hicora_cache as k11a                        # noqa: E402

N_POS, N_LEVEL, T_CHUNK, H_EXEC = 16, 3, 20, 8
TAPS = (12, 18, 24)
# Допуск на совпадение действий. Не ноль: путь HiCoRA собирает z как z0 + dz,
# где dz строго нулевой, но сложение с нулём в fp16 всё же проходит через
# другой порядок операций. Порог на три порядка ниже характерной величины
# действия (около 1e-2), то есть заведомо ниже любой содержательной разницы.
ACT_TOL = 1e-5


class CallCounter:
    """Счётчик вызовов метода. Оборачивает, не подменяя поведение.

    Возврат обёртки ОБЯЗАН быть выходом исходного метода без изменений: в
    K-7b возврат кортежа из хука однажды подменил выход нормы, и расхождение
    потом искали руками.
    """

    def __init__(self, obj, name):
        self.obj, self.name, self.n = obj, name, 0
        self.orig = getattr(obj, name)

        def wrapped(*a, **kw):
            self.n += 1
            return self.orig(*a, **kw)

        setattr(obj, name, wrapped)

    def reset(self):
        self.n = 0
        return self

    def close(self):
        setattr(self.obj, self.name, self.orig)


class LayerCounter:
    """Сколько раз исполнился КАЖДЫЙ слой за проход, а не сколько всего.

    ХУК НА `action_expert.layers[i]` НЕ СРАБАТЫВАЕТ. `_shared_attention_forward`
    НЕ вызывает `layer.forward()`: он дёргает дочерние модули напрямую —
    `input_layernorm`, проекции, `mlp`. Forward-хук родительского модуля при
    этом молчит, и словарь счётчиков оставался бы ПУСТЫМ, а проверка «каждый
    слой по разу» падала бы с «слои не исполнились» на исправном коде.
    Поэтому считаем по `layer_idx`, который передаётся в сам метод.

    Общее число вызовов не годится: оно не различает «24 слоя по разу» и
    «12 слоёв по два», а это ровно то, что утверждает фраза «один проход».
    """

    def __init__(self, model, name="_shared_attention_forward"):
        self.model, self.name = model, name
        self.counts = {}
        self.orig = getattr(model, name)

        def wrapped(*a, **kw):
            idx = kw.get("layer_idx")
            if idx is None and a:
                idx = a[2] if len(a) > 2 else None
            self.counts[int(idx)] = self.counts.get(int(idx), 0) + 1
            return self.orig(*a, **kw)

        setattr(model, name, wrapped)

    def reset(self):
        self.counts = {}
        return self

    def close(self):
        setattr(self.model, self.name, self.orig)


def check_layer_counts(counts, n_layers):
    """Ровно один вызов у каждого слоя, и ни одного пропущенного."""
    missing = [i for i in range(n_layers) if i not in counts]
    extra = {i: c for i, c in counts.items() if c != 1}
    if missing:
        raise SystemExit(f"слои {missing[:5]} не исполнились: проход неполный")
    if extra:
        raise SystemExit(f"слои исполнены не по разу: {dict(list(extra.items())[:5])}")
    if len(counts) != n_layers:
        raise SystemExit(f"исполнено {len(counts)} слоёв из {n_layers}")
    return True


def act_diff(a, b):
    """Максимальное расхождение действий и доля несовпавших знаков схвата."""
    A, B = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return dict(max_abs=float(np.abs(A[..., :6] - B[..., :6]).max()),
                grip_mismatch=float((np.sign(A[..., 6])
                                     != np.sign(B[..., 6])).mean()))


# Поля кэша, которые обязаны совпасть с диагностикой и с ожиданиями K-11b.
CACHE_FIELDS = ("q0_source", "depth", "taps")


def check_artifacts(diag, cache_prefix, basis_sha, rho_sha, rank,
                    cache_meta_sha):
    """Строгая привязка basis.npy и rho.npy к породившей их диагностике.

    ХЕШ, ПОСЧИТАННЫЙ ПОСЛЕ ПРИНЯТИЯ ФАЙЛА, — ЭТО ОТЧЁТ, А НЕ ПРОВЕРКА.
    Прежняя версия сверяла только ранг, поэтому любой ортонормированный
    базис той же формы — например с переставленными столбцами — проходил
    все проверки и обучение шло бы в чужих координатах с чужой rho.
    """
    need = ("basis_sha1", "rho_sha1", "rank", "prefix", "cache_meta_sha1")
    miss = [k for k in need if not diag.get(k) and diag.get(k) != 0]
    if miss:
        raise SystemExit(
            f"в диагностике нет полей {miss}: она собрана версией, которая не "
            f"записывала отпечатки артефактов. Перезапустите шаг --diagnose "
            f"текущей версией K-11a — он читает готовый кэш и стоит минуты")
    if diag["prefix"] != cache_prefix:
        raise SystemExit(f"диагностика собрана для {diag['prefix']}, а кэш "
                         f"{cache_prefix}")
    if int(diag["rank"]) != int(rank):
        raise SystemExit(f"ранг в диагностике {diag['rank']}, а в basis.npy "
                         f"{rank}")
    if diag["basis_sha1"] != basis_sha:
        raise SystemExit(
            f"basis.npy sha {basis_sha}, а диагностика породила "
            f"{diag['basis_sha1']}: это ДРУГОЙ базис той же формы")
    if diag["rho_sha1"] != rho_sha:
        raise SystemExit(f"rho.npy sha {rho_sha}, а в диагностике "
                         f"{diag['rho_sha1']}")
    if diag["cache_meta_sha1"] != cache_meta_sha:
        raise SystemExit("meta кэша изменилась после диагностики: базис и "
                         "rho относятся к другому кэшу")
    return True


def check_cache_fields(meta, diag, expect, modules, allow_drift=False):
    """Поля кэша сверяются ДО загрузки модели, а не после.

    `hicora_vla_sha1` и `joint12_vla_sha1` определяют исполняемую сеть не
    меньше, чем веса. Расхождение по умолчанию — отказ; разрешить его можно
    только явным флагом, и тогда оно печатается и попадает в отчёт.
    """
    for k, want in expect.items():
        got = meta.get(k)
        if want is not None and got != want:
            raise SystemExit(f"кэш собран с {k}={got!r}, а здесь {want!r}")
    if diag is not None:
        for k in CACHE_FIELDS:
            if diag.get(k) is not None and diag.get(k) != meta.get(k):
                raise SystemExit(
                    f"диагностика и кэш расходятся по {k}: "
                    f"{diag.get(k)!r} против {meta.get(k)!r}")
    # ОТСУТСТВИЕ ОТПЕЧАТКА МОДУЛЯ — ОТКАЗ, а не «нечего сравнивать»: условие
    # `if meta.get(k)` было fail-open, и кэш без этих полей проходил молча.
    miss_mod = [k for k in modules if not meta.get(k)]
    if miss_mod:
        raise SystemExit(
            f"в meta кэша нет версий модулей {miss_mod}: подтвердить, что "
            f"кэш собран той же сетью, нечем. Пересоберите кэш текущей "
            f"версией K-11a")
    drift = {k: (meta.get(k), v) for k, v in modules.items()
             if meta[k] != v}
    if drift and not allow_drift:
        lines = "; ".join(f"{k}: кэш {a}, сейчас {b}"
                          for k, (a, b) in drift.items())
        raise SystemExit(
            f"версии модулей разошлись с кэшем — {lines}. Если правка "
            f"заведомо не затрагивает путь сбора, повторите с "
            f"--allow-module-drift: расхождение будет напечатано и записано "
            f"в отчёт")
    return drift


def dataset_source(meta):
    """Репозиторий и ревизия данных — из meta, а не из литерала в коде.

    Прежде оба были зашиты строками. Тогда состояния для входов читались бы
    из набора, никак не связанного с тем, на котором собран кэш: тождество
    подтверждалось бы на чужих данных, а расхождение выглядело бы как
    расхождение моделей. Отсутствие поля — отказ, а не подстановка
    умолчания: угаданное происхождение хуже отсутствующего.
    """
    man = meta.get("manifest") or {}
    repo, rev = man.get("dataset_repo"), man.get("dataset_revision")
    if not repo or not rev:
        raise SystemExit(
            "в meta нет manifest.dataset_repo/dataset_revision — кэш собран "
            "версией K-11a без записи происхождения данных; восстановить "
            "поля угадыванием нельзя, нужен пересбор")
    top = meta.get("dataset_revision")
    if top is not None and top != rev:
        raise SystemExit(
            f"ревизия в meta противоречива: dataset_revision={top}, "
            f"manifest.dataset_revision={rev}")
    return str(repo), str(rev)


def read_identity(res):
    """Пре-регистрированное чтение: тождество либо есть, либо его нет."""
    bad = [k for k, v in res.items() if v is False]
    if bad:
        return (f"ТОЖДЕСТВО НЕ ВЫПОЛНЕНО: {bad}. Обучение начинать нельзя — "
                f"сравнение шло бы не с той опорой, и «улучшение» могло бы "
                f"оказаться расхождением конвейеров")
    return ("тождество выполнено по всем пунктам: до обучения HiCoRA "
            "неотличима от черновика Joint12, голова живая, градиент в "
            "замороженное не течёт — можно переходить к K-11c")


def selftest():
    # --- счётчик вызовов не меняет поведение -------------------------------
    class Obj:
        def f(self, x):
            return x * 2

    o = Obj()
    c = CallCounter(o, "f")
    assert o.f(3) == 6 and o.f(4) == 8, "обёртка изменила выход"
    assert c.n == 2, c.n
    c.close()
    assert o.f(5) == 10 and c.n == 2, "снятие обёртки не вернуло метод"

    # --- подсчёт слоёв различает «24 по разу» и «12 по два» ----------------
    assert check_layer_counts({i: 1 for i in range(24)}, 24) is True
    for bad, frag in (({i: 1 for i in range(12)}, "не исполнились"),
                      ({i: 2 for i in range(24)}, "не по разу"),
                      ({**{i: 1 for i in range(23)}, 23: 3}, "не по разу")):
        try:
            check_layer_counts(bad, 24)
            raise AssertionError(f"пропущено: {frag}")
        except SystemExit as e:
            assert frag in str(e), (frag, str(e))
    # ДВЕНАДЦАТЬ СЛОЁВ ПО ДВА РАЗА дают те же 24 вызова суммарно — и обязаны
    # быть отвергнуты, иначе счётчик не проверял бы «один проход».
    two_passes = {i: 2 for i in range(12)}
    try:
        check_layer_counts(two_passes, 24)
        raise AssertionError("двенадцать слоёв по два раза прошли")
    except SystemExit:
        pass

    # --- САМ МЕХАНИЗМ ПОДСЧЁТА, А НЕ ТОЛЬКО ГОТОВЫЙ СЛОВАРЬ ----------------
    # Прежняя версия ставила forward-хуки на `action_expert.layers[i]`, но
    # `_shared_attention_forward` их не вызывает — он дёргает дочерние модули
    # напрямую. Словарь оставался ПУСТЫМ, и проверка падала бы на исправном
    # коде. Здесь проверяется именно обёртка метода.
    class FakeModel:
        def __init__(self):
            self.calls = []

        def _shared_attention_forward(self, *, vlm_hidden_states,
                                      action_hidden_states, layer_idx, **kw):
            self.calls.append(layer_idx)
            return vlm_hidden_states, action_hidden_states

    fm = FakeModel()
    lc = LayerCounter(fm)
    for i in range(24):
        fm._shared_attention_forward(vlm_hidden_states=1,
                                     action_hidden_states=2, layer_idx=i)
    assert lc.counts == {i: 1 for i in range(24)}, lc.counts
    assert check_layer_counts(lc.counts, 24) is True
    assert fm.calls == list(range(24)), "обёртка не пропустила вызовы"
    # ДВА ПРОХОДА ПО ДВЕНАДЦАТИ СЛОЯМ обязаны быть отвергнуты.
    lc.reset()
    for _ in range(2):
        for i in range(12):
            fm._shared_attention_forward(vlm_hidden_states=1,
                                         action_hidden_states=2, layer_idx=i)
    try:
        check_layer_counts(lc.counts, 24)
        raise AssertionError("два прохода по 12 слоям прошли")
    except SystemExit:
        pass
    lc.close()
    lc.reset()
    fm._shared_attention_forward(vlm_hidden_states=1, action_hidden_states=2,
                                 layer_idx=0)
    assert lc.counts == {}, "снятие обёртки не вернуло метод"

    # --- сравнение действий -------------------------------------------------
    a = np.zeros((2, T_CHUNK, 7)); b = np.zeros((2, T_CHUNK, 7))
    a[..., 6] = 1.0; b[..., 6] = 1.0
    d = act_diff(a, b)
    assert d["max_abs"] == 0.0 and d["grip_mismatch"] == 0.0
    b[0, 0, 0] = 0.5
    assert act_diff(a, b)["max_abs"] == 0.5
    b2 = b.copy(); b2[..., 6] = -1.0
    assert act_diff(a, b2)["grip_mismatch"] == 1.0
    # ЗНАК СЧИТАЕТСЯ ОТДЕЛЬНО ОТ ПОЗЫ: идеальная поза при перевёрнутом схвате
    # не есть тождество.
    b3 = a.copy(); b3[..., 6] = -1.0
    assert act_diff(a, b3)["max_abs"] == 0.0
    assert act_diff(a, b3)["grip_mismatch"] == 1.0

    # --- ПРИВЯЗКА АРТЕФАКТОВ К ДИАГНОСТИКЕ ---------------------------------
    # Прежде сверялся только ранг, и переставленный базис той же формы
    # проходил. Здесь мутации: отсутствие поля, изменённый sha, чужой prefix,
    # другой ранг, изменившаяся meta.
    diag_ok = dict(basis_sha1="B1", rho_sha1="R1", rank=16, prefix="/c",
                   cache_meta_sha1="M1")
    assert check_artifacts(diag_ok, "/c", "B1", "R1", 16, "M1") is True

    def art_fail(diag, *a, frag=""):
        try:
            check_artifacts(diag, *a)
        except SystemExit as e:
            assert frag in str(e), (frag, str(e))
            return
        raise AssertionError(f"пропущено: {frag}")

    # ПЕРЕСТАВЛЕННЫЙ БАЗИС той же формы: ранг совпадает, sha другой.
    art_fail(diag_ok, "/c", "B2", "R1", 16, "M1", frag="ДРУГОЙ базис")
    art_fail(diag_ok, "/c", "B1", "R2", 16, "M1", frag="rho.npy sha")
    art_fail(diag_ok, "/c", "B1", "R1", 8, "M1", frag="ранг")
    art_fail(diag_ok, "/other", "B1", "R1", 16, "M1", frag="собрана для")
    art_fail(diag_ok, "/c", "B1", "R1", 16, "M2", frag="meta кэша изменилась")
    for k in ("basis_sha1", "rho_sha1", "rank", "prefix", "cache_meta_sha1"):
        art_fail({x: v for x, v in diag_ok.items() if x != k},
                 "/c", "B1", "R1", 16, "M1", frag="нет полей")

    # --- ПОЛЯ КЭША И ВЕРСИИ МОДУЛЕЙ ----------------------------------------
    meta_ok = dict(depth=12, taps=[12, 18, 24], q0_source="joint12",
                   hicora_vla_sha1="H1", joint12_vla_sha1="J1")
    mods = dict(hicora_vla_sha1="H1", joint12_vla_sha1="J1")
    assert check_cache_fields(meta_ok, None, dict(depth=12,
                                                  taps=[12, 18, 24],
                                                  q0_source="joint12"),
                              mods) == {}
    # ИСТОЧНИК ЧЕРНОВИКА СВЕРЯЕТСЯ СТРОГО.
    try:
        check_cache_fields(meta_ok, None, dict(q0_source="readout"), mods)
        raise AssertionError("чужой источник q0 прошёл")
    except SystemExit as e:
        assert "q0_source" in str(e)
    # ОТСУТСТВИЕ ОТПЕЧАТКА МОДУЛЯ — отказ, а не молчаливый пропуск.
    for k in ("hicora_vla_sha1", "joint12_vla_sha1"):
        gone = {x: v for x, v in meta_ok.items() if x != k}
        try:
            check_cache_fields(gone, None, dict(depth=12), mods)
            raise AssertionError(f"отсутствие {k} прошло")
        except SystemExit as e:
            assert "нет версий модулей" in str(e), str(e)
    try:
        check_cache_fields(meta_ok, None, dict(depth=18), mods)
        raise AssertionError("другая глубина прошла")
    except SystemExit as e:
        assert "depth" in str(e)
    # ДРЕЙФ ВЕРСИИ МОДУЛЯ по умолчанию отказ, с флагом — возвращается наружу.
    try:
        check_cache_fields(meta_ok, None, dict(depth=12),
                           dict(hicora_vla_sha1="H2", joint12_vla_sha1="J1"))
        raise AssertionError("дрейф модуля прошёл молча")
    except SystemExit as e:
        assert "разошлись с кэшем" in str(e)
    d = check_cache_fields(meta_ok, None, dict(depth=12),
                           dict(hicora_vla_sha1="H2", joint12_vla_sha1="J1"),
                           allow_drift=True)
    assert d == {"hicora_vla_sha1": ("H1", "H2")}, d
    # Расхождение диагностики с кэшем по полю тоже отказ.
    try:
        check_cache_fields(meta_ok, dict(depth=18), dict(depth=12), mods)
        raise AssertionError("расхождение диагностики и кэша прошло")
    except SystemExit as e:
        assert "расходятся по depth" in str(e)

    # --- происхождение данных берётся из meta -------------------------------
    good_meta = dict(manifest=dict(dataset_repo="physical-intelligence/libero",
                                   dataset_revision="v2.0"),
                     dataset_revision="v2.0")
    assert dataset_source(good_meta) == ("physical-intelligence/libero", "v2.0")
    # поле отсутствует — отказ, а не подстановка умолчания
    for bad in (dict(manifest={}, dataset_revision="v2.0"),
                dict(manifest=dict(dataset_repo="physical-intelligence/libero"),
                     dataset_revision="v2.0"),
                dict(manifest=dict(dataset_revision="v2.0")),
                dict(dataset_revision="v2.0")):
        try:
            dataset_source(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"неполное происхождение принято: {bad}")
    # верхнее поле противоречит манифесту — отказ
    try:
        dataset_source(dict(manifest=dict(
            dataset_repo="physical-intelligence/libero",
            dataset_revision="v2.0"), dataset_revision="v1.0"))
    except SystemExit:
        pass
    else:
        raise AssertionError("противоречивая ревизия принята")
    # контроль: другой (но согласованный) источник проходит и возвращает себя,
    # то есть проверка сверяет поля, а не сравнивает с зашитым литералом
    assert dataset_source(dict(manifest=dict(dataset_repo="someone/else",
                                             dataset_revision="v9.9"))) \
        == ("someone/else", "v9.9")

    # --- чтение вердикта ----------------------------------------------------
    assert "можно переходить" in read_identity({"a": True, "b": True})
    txt = read_identity({"a": True, "q0_bitwise": False})
    assert "НЕ ВЫПОЛНЕНО" in txt and "q0_bitwise" in txt

    print("самопроверка k11b пройдена (версия «происхождение данных из meta»): "
          "обёртка не меняет выход и снимается, "
          "счётчик слоёв считает по layer_idx и отвергает двенадцать по два "
          "при тех же 24 вызовах, знак схвата считается отдельно от позы, "
          "вердикт называет провалившийся пункт, переставленный базис той "
          "же формы отвергается, дрейф версии модуля требует явного флага, "
          "репозиторий и ревизия читаются из манифеста и отвергают неполное "
          "и противоречивое происхождение")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", help="префикс кэша K-11a")
    ap.add_argument("--joint-ckpt")
    ap.add_argument("--basis", default=None,
                    help="basis.npy от диагностики; по умолчанию <cache>.basis.npy")
    ap.add_argument("--rho", default=None)
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--q0-source", default="joint12",
                    help="источник черновика, ожидаемый от кэша. Сверяется "
                         "строго: кэш с q0 от другого источника описывает "
                         "другую модель")
    ap.add_argument("--batches", default="1,10")
    ap.add_argument("--allow-module-drift", action="store_true",
                    help="разрешить расхождение версий hicora_vla/joint12_vla "
                         "с кэшем. Расхождение печатается и попадает в отчёт")
    ap.add_argument("--out", default="data/k11b_identity.json")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    for f in ("cache", "joint_ckpt", "basis", "rho", "out"):
        v = getattr(args, f)
        if v:
            setattr(args, f, os.path.abspath(v))
    args.root = os.path.abspath(args.root)
    sys.path.insert(0, args.root)
    sha = k11a.file_sha1(__file__)
    print(f"k11b sha1 {sha}")
    for need, why in ((args.ckpt, "--ckpt"), (args.cache, "--cache"),
                      (args.joint_ckpt, "--joint-ckpt")):
        if not need:
            raise SystemExit(f"нужен {why}")
    basis_p = args.basis or (args.cache + ".basis.npy")
    rho_p = args.rho or (args.cache + ".rho.npy")
    for p_ in (basis_p, rho_p):
        if not os.path.exists(p_):
            raise SystemExit(
                f"нет {p_}: сначала диагностика K-11a выберет ранг и rho. "
                f"Без них голова отказывается работать, и это правильно")

    import torch
    import actioncodec  # noqa: F401  регистрирует action_codec
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "action_codec" not in CONFIG_MAPPING:
        raise SystemExit("тип «action_codec» не зарегистрирован")
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (VisionLanguageActionProcessor, dict_apply, get_cfg,
                       prompt_template, seed_everything)
    from joint12_vla import make_joint12_class
    import joint12_vla as jv
    import hicora_vla as hv

    seed_everything(0)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(args.root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    meta = json.load(open(args.cache + ".meta.json"))
    if meta.get("ckpt") != args.ckpt:
        raise SystemExit(f"кэш собран чекпойнтом {meta.get('ckpt')}")
    src = meta.get("source") or {}
    if src.get("weights_sha1") != k11a.file_sha1(args.joint_ckpt):
        raise SystemExit(
            f"кэш собран весами sha {src.get('weights_sha1')}, а здесь "
            f"{k11a.file_sha1(args.joint_ckpt)}: черновик был бы другой")

    # --- ПОЛЯ КЭША И ВЕРСИИ МОДУЛЕЙ СВЕРЯЮТСЯ ДО ЗАГРУЗКИ МОДЕЛИ -----------
    diag_p = args.cache + ".diag.json"
    if not os.path.exists(diag_p):
        raise SystemExit(f"нет {diag_p}: сначала шаг --diagnose K-11a")
    diag = json.load(open(diag_p))
    hv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hicora_vla.py")
    jv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "joint12_vla.py")
    drift = check_cache_fields(
        meta, diag,
        expect=dict(depth=args.depth, taps=list(TAPS),
                    q0_source=args.q0_source),
        modules=dict(hicora_vla_sha1=k11a.file_sha1(hv_path),
                     joint12_vla_sha1=k11a.file_sha1(jv_path)),
        allow_drift=args.allow_module_drift)
    if drift:
        print("  ВНИМАНИЕ, версии модулей разошлись с кэшем (разрешено "
              "явно):")
        for k, (a, b) in drift.items():
            print(f"    {k}: кэш {a}, сейчас {b}")
    print(f"  кэш: {meta['n_obs']} наблюдений, источник q0 "
          f"«{meta['q0_source']}», отводы {meta['taps']}, глубина "
          f"{meta['depth']}")

    # --- модель ровно как в K-11a -------------------------------------------
    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=args.depth, head_dtype=dt)

    ac = proc.action_processor
    codec = ac if hasattr(ac, "vq") else getattr(ac, "codec", None)
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        idx = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(idx))[0]
                         for q in codec.vq.quantizers]).float()

    # ПОВТОРНАЯ СВЕРКА ПРОИСХОЖДЕНИЯ: между сбором и обучением декодер мог
    # смениться под тем же именем чекпойнта.
    k11a.check_fingerprints(meta, dict(
        codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
            E.cpu().numpy().astype(np.float32)).tobytes()).hexdigest()[:12],
        decoder_probe=k11a.decoder_probe(codec, E, dev),
        codec_state_sha1=k11a.state_sha1(codec)))
    print("  отпечатки декодера совпали с записанными в кэш")

    import copy
    res_norm = copy.deepcopy(model.action_expert.norm)
    obj = torch.load(args.joint_ckpt, map_location="cpu", weights_only=False)
    state = obj["state"]
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            own[k].data = v.to(dev, torch.float32)
    model.to_fp32_trainable()
    not32 = [k for k in state if own[k].dtype != torch.float32]
    if not32:
        raise SystemExit(f"{len(not32)} загруженных весов не в fp32")
    model.eval()

    model.__class__ = hv.make_hicora_class(type(model))
    model.set_codebooks(E)
    model.set_res_norm(res_norm.to(dev))
    model.taps, model.q0_depth = TAPS, args.depth
    model.n_layers_total = len(model.action_expert.layers)
    B = np.load(basis_p)
    rho = np.load(rho_p)
    model.init_hicora(q0_depth=args.depth, rank=int(B.shape[1]), taps=TAPS)
    model.hicora_head.set_basis(torch.as_tensor(B))
    model.hicora_head.set_rho(torch.as_tensor(rho))
    model.hicora_head.to(dev)
    model.hicora_head.check_ready()
    # БАЗИС И rho СВЕРЯЮТСЯ СО СВОИМИ ФАЙЛАМИ. Загрузка молча не туда — самый
    # тихий способ обучить голову с чужим пределом амплитуды.
    got_B = model.hicora_head.basis.detach().cpu().numpy()
    got_r = model.hicora_head.rho.detach().cpu().numpy()
    if not (np.allclose(got_B, B, atol=1e-6)
            and np.allclose(got_r, rho, atol=1e-6)):
        raise SystemExit("базис или rho в модели не совпали с файлами")
    # ОТПЕЧАТКИ ФАЙЛОВ СЧИТАЮТСЯ ДО ИХ ПРИНЯТИЯ и сверяются с диагностикой,
    # которая их породила. Совпадение содержимого с загруженным массивом
    # подтверждало бы только копирование.
    basis_sha, rho_sha = k11a.file_sha1(basis_p), k11a.file_sha1(rho_p)
    check_artifacts(diag, args.cache, basis_sha, rho_sha, int(B.shape[1]),
                    k11a.file_sha1(args.cache + ".meta.json"))
    if not np.allclose(np.asarray(diag["rho"], np.float64), rho, atol=1e-6):
        raise SystemExit("rho.npy не совпадает с rho из диагностики")
    detail_prov = dict(diag_sha1=k11a.file_sha1(diag_p),
                       basis_sha1=basis_sha, rho_sha1=rho_sha,
                       module_drift={k: list(v) for k, v in drift.items()},
                       gain_target=diag.get("gain_target"),
                       grip_delta=diag.get("grip_delta"))
    print(f"  происхождение базиса подтверждено диагностикой: ранг "
          f"{diag['rank']}, порог {diag.get('gain_target')}, допуск схвата "
          f"{diag.get('grip_delta')}")
    print(f"  базис r={B.shape[1]} и rho загружены и сверены с файлами; "
          f"||rho|| = {float(np.linalg.norm(rho)):.4f}")

    # --- входы из кэша K-9a --------------------------------------------------
    cache_npz = meta["cache"]
    d = np.load(cache_npz, allow_pickle=True)
    IMG = np.load(cache_npz + ".images.npy", mmap_mode="r")
    q0hat = np.load(args.cache + ".q0hat.npy")
    offs = d["pos_offset"].astype(np.int64)
    tsk = d["task"]
    from utils import STATE_Q01, STATE_Q99, process_state
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    epi, stp = d["episode"], d["step"]

    sizes = [int(x) for x in str(args.batches).split(",") if x]
    # ОДИН ОФСЕТ НА ВЫЗОВ: position_offset задаётся на весь батч, смешивать
    # в одном нельзя.
    po = int(offs[0])
    pool = np.where(offs == po)[0][:max(sizes)]

    # ПРОИСХОЖДЕНИЕ ДАННЫХ БЕРЁТСЯ ИЗ meta, А НЕ ИЗ ЛИТЕРАЛА. Прежде
    # репозиторий и ревизия были зашиты строками: тождество подтверждалось
    # бы на любом наборе, даже если кэш собран на другом, и расхождение
    # выглядело бы как расхождение модели.
    ds_repo, ds_rev = dataset_source(meta)
    print(f"  данные: {ds_repo}@{ds_rev} (из meta, не из литерала)")
    st = np.zeros((len(pool), len(STATE_Q01)), np.float64)
    for j, gi in enumerate(pool):
        e = int(epi[gi])
        f = hf_hub_download(ds_repo,
                            f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                            repo_type="dataset", revision=ds_rev)
        S_ = np.asarray(pq.read_table(f).column("state").to_pylist(),
                        np.float32)
        st[j] = S_[int(stp[gi])] if S_.shape[1] == len(STATE_Q01) \
            else process_state(S_[int(stp[gi])][None])[0]
    st_n = (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0

    def build(k):
        sel = pool[:k]
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for j, gi in enumerate(sel):
            m = prompt_template(
                st_n[j], None, str(tsk[gi]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[i].numpy()] for i in range(k)],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b), sel

    res, detail = {}, {}
    gen_cnt = CallCounter(model, "generate")
    dec_cnt = CallCounter(codec, "_decode")
    lay_cnt = LayerCounter(model)

    for k in sizes:
        tag = f"b{k}"
        b, sel = build(k)
        lay_cnt.reset()
        gen_cnt.reset(); dec_cnt.reset()
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dt):
            v, pp = model.build_inputs(position_offset=po, **b)
            out = model.forward_hicora(
                vlm_inputs_embeds=v, attention_mask=b.get("attention_mask"),
                position_ids=pp)
        # 7. ДЕКОДИРОВАНИЙ ВНУТРИ forward_hicora БЫТЬ НЕ ДОЛЖНО, а на выдачу
        # действия — ровно одно. Счётчик снимается СРАЗУ после прохода:
        # прежняя версия сбрасывала его ПОСЛЕ forward_hicora и потом считала
        # собственную явную строку `_decode`, то есть проверяла тавтологию и
        # забывала любые скрытые декодирования внутри прохода.
        dec_inside = dec_cnt.n
        detail[f"decode_inside_forward_{tag}"] = dec_inside
        # 5. каждый слой ровно один раз
        res[f"layers_once_{tag}"] = check_layer_counts(lay_cnt.counts,
                                                       model.n_layers_total)
        # 6. никакого спрятанного generate
        res[f"no_generate_{tag}"] = (gen_cnt.n == 0)
        detail[f"generate_calls_{tag}"] = gen_cnt.n

        # 1. q0 побитово совпадает с Joint12
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dt):
            jf = model.forward_joint_fast(
                vlm_inputs_embeds=v, attention_mask=b.get("attention_mask"),
                position_ids=pp, depth=args.depth)
        same = bool(torch.equal(out["q0"], jf["pred_codes"]))
        res[f"q0_bitwise_{tag}"] = same
        detail[f"q0_mismatch_{tag}"] = float(
            (out["q0"] != jf["pred_codes"]).float().mean())
        # и совпадает с тем, что записано в кэш
        res[f"q0_matches_cache_{tag}"] = bool(
            (out["q0"].cpu().numpy().astype(np.int16) == q0hat[sel]).all())

        # 2. нулевая голова даёт z == z0 побитово
        res[f"zero_delta_{tag}"] = bool(torch.equal(out["z"], out["z0"]))
        detail[f"dz_max_{tag}"] = float(out["dz"].abs().max())

        # 3-4. действия и знак совпадают
        with torch.no_grad():
            A_h = codec._decode(out["z"], embodiment_ids=0)[0][..., :7].float()
        # ЧИСЛО ФИКСИРУЕТСЯ ДО базового декодирования: иначе оно неизбежно
        # станет двойкой, и печать «decode 2» уже наблюдалась.
        total_dec = dec_cnt.n
        res[f"decode_once_{tag}"] = (dec_inside == 0 and total_dec == 1)
        detail[f"decode_calls_{tag}"] = total_dec
        with torch.no_grad():
            A_0 = codec._decode(out["z0"], embodiment_ids=0)[0][..., :7].float()
        dd = act_diff(A_h.cpu().numpy(), A_0.cpu().numpy())
        res[f"actions_match_{tag}"] = bool(dd["max_abs"] <= ACT_TOL)
        res[f"grip_match_{tag}"] = bool(dd["grip_mismatch"] == 0.0)
        detail[f"act_max_abs_{tag}"] = dd["max_abs"]
        detail[f"grip_mismatch_{tag}"] = dd["grip_mismatch"]
        print(f"  {tag}: слои по разу, generate {gen_cnt.n}, decode внутри "
              f"прохода {dec_inside} и на выдачу {total_dec}, |dz| "
              f"{detail[f'dz_max_{tag}']:.2e}, |ΔA| {dd['max_abs']:.2e}, "
              f"знак {dd['grip_mismatch']:.1%}")

    # --- КОНТРФАКТИЧЕСКАЯ ПРОВЕРКА ------------------------------------------
    # Подменяем черновик и требуем, чтобы изменились И вход головы, И цель.
    # Голова, игнорирующая z0, прошла бы все проверки выше.
    b, sel = build(min(sizes))
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dt):
        v, pp = model.build_inputs(position_offset=po, **b)
        tp = model.forward_taps(vlm_inputs_embeds=v,
                                attention_mask=b.get("attention_mask"),
                                position_ids=pp)
        _, q_real = model.q0_from(tp[args.depth])
    h24 = model.res_norm(tp[max(TAPS)]).float()
    z_real = model.codebooks[0][q_real]
    q_fake = (q_real + 1) % int(model.codebooks.shape[1])
    z_fake = model.codebooks[0][q_fake]
    with torch.no_grad():
        c_real = model.hicora_head.coeffs(h24, z_real)
        c_fake = model.hicora_head.coeffs(h24, z_fake)
    # При нулевом последнем слое коэффициенты нулевые у обоих, поэтому
    # различие проверяется на ПРЕДПОСЛЕДНЕМ слое: он уже читает z0.
    with torch.no_grad():
        pre = model.hicora_head.net[:-1]
        p_real = pre(torch.cat([h24, model.hicora_head.proj(z_real)], -1))
        p_fake = pre(torch.cat([h24, model.hicora_head.proj(z_fake)], -1))
    res["counterfactual_input"] = bool(
        float((p_real - p_fake).abs().max()) > 0)
    detail["counterfactual_input_delta"] = float((p_real - p_fake).abs().max())
    # ЦЕЛЬ СТРОИТСЯ ТЕМ ЖЕ ВЫРАЖЕНИЕМ, ЧТО ПОЙДЁТ В ОБУЧЕНИЕ: r = z* - z0.
    # Прежняя версия сравнивала только z_real и z_fake — вывод верен
    # математически, но код построения actual-draft target не проверялся.
    Kt_all = np.load(args.cache + ".ktrue.npy")
    # ДОЛЯ СЧИТАЕТСЯ ПО ВСЕМУ КЭШУ, А НЕ ПО ОДНОМУ НАБЛЮДЕНИЮ. При batch=1
    # прежняя величина опиралась на 16 токенов и о вырождении задачи не
    # говорила ничего.
    frac_diff = float((q0hat != Kt_all[:, 0, :]).mean())
    detail["draft_vs_true_q0_frac"] = frac_diff
    res["draft_differs_from_true_q0"] = bool(frac_diff > 0)
    print(f"  черновик расходится с истинным q0* на {frac_diff:.1%} токенов "
          f"по всему кэшу ({len(q0hat)} наблюдений)")
    if frac_diff == 0:
        print("    ноль означает, что задача HiCoRA вырождается в задачу "
              "кодека:\n    исправлять ошибку предсказания нечего.")
    Kt = Kt_all[sel]
    kt = torch.as_tensor(Kt).long().to(dev)
    z_star = sum(model.codebooks[l][kt[:, l, :]] for l in range(N_LEVEL))
    r_real = z_star - z_real
    r_fake = z_star - z_fake
    res["counterfactual_target"] = bool(
        float((r_real - r_fake).abs().max()) > 0)
    detail["counterfactual_target_delta"] = float((r_real - r_fake).abs().max())
    # И ЦЕЛЬ ДОЛЖНА ОТЛИЧАТЬСЯ ОТ ОБЫЧНОГО RVQ-ОСТАТКА: если q0_hat совпадает
    # с q0* на всех позициях, задача HiCoRA вырождается в задачу кодека.
    z0_true = model.codebooks[0][kt[:, 0, :]]
    detail["draft_vs_true_q0_delta"] = float((z_real - z0_true).abs().max())
    print(f"  контрфакт: вход головы меняется на "
          f"{detail['counterfactual_input_delta']:.3e}, цель тоже")

    # --- 9-10. ГРАДИЕНТЫ -----------------------------------------------------
    n_par = model.configure_hicora_d1(verbose=True)
    # КОДЕК ЗАМОРАЖИВАЕТСЯ ЯВНО. `codec.eval()` параметры не замораживает, а
    # прежняя проверка вела градиент от `dz.sum()`, где декодера в графе нет
    # вовсе: «в кодек не течёт» выполнялось бы и при полностью обучаемом
    # кодеке. Здесь потеря идёт ЧЕРЕЗ декодер, поэтому проверка настоящая.
    codec.requires_grad_(False)
    unfrozen = [n for n, p_ in codec.named_parameters() if p_.requires_grad]
    res["codec_frozen"] = (len(unfrozen) == 0)
    detail["codec_unfrozen"] = unfrozen[:5]
    model.zero_grad(set_to_none=True)
    codec.zero_grad(set_to_none=True)
    b, sel = build(min(sizes))
    with torch.autocast(device_type=dev.type, dtype=dt):
        v, pp = model.build_inputs(position_offset=po, **b)
        out = model.forward_hicora(
            vlm_inputs_embeds=v, attention_mask=b.get("attention_mask"),
            position_ids=pp)
    # ПОТЕРЯ В ПРОСТРАНСТВЕ ДЕЙСТВИЙ, ЧЕРЕЗ ЕДИНСТВЕННОЕ ДЕКОДИРОВАНИЕ — та
    # самая цепочка, по которой пойдёт обучение D1. И цель НЕНУЛЕВАЯ: ни
    # `sum(y^2)` при нулевом выходе (градиент ноль по построению), ни
    # `dz.sum()` (может сократиться, если столбцы базиса ортогональны вектору
    # из единиц) надёжной проверкой не являются.
    A = codec._decode(out["z"], embodiment_ids=0)[0][..., :7].float()
    target = torch.full_like(A, 0.1)
    loss = torch.nn.functional.smooth_l1_loss(A, target)
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise SystemExit("потеря не конечна: проверка градиентов "
                         "недействительна")
    g_head = sum(float(p_.grad.abs().sum()) for p_ in
                 model.hicora_head.parameters() if p_.grad is not None)
    res["head_grad_nonzero"] = bool(g_head > 0 and np.isfinite(g_head))
    detail["head_grad_sum"] = g_head
    detail["loss"] = float(loss)
    frozen_grad = [n for n, p_ in model.named_parameters()
                   if not n.startswith(("hicora_head.proj.",
                                        "hicora_head.net."))
                   and p_.grad is not None
                   and float(p_.grad.abs().sum()) > 0]
    res["no_grad_into_frozen"] = (len(frozen_grad) == 0)
    detail["frozen_with_grad"] = frozen_grad[:5]
    cod_grad = [n for n, p_ in codec.named_parameters()
                if p_.grad is not None and float(p_.grad.abs().sum()) > 0]
    res["no_grad_into_codec"] = (len(cod_grad) == 0)
    detail["codec_with_grad"] = cod_grad[:5]
    print(f"  потеря через декодер {float(loss):.4e}, градиент головы "
          f"{g_head:.3e}, обучаемых параметров {n_par}; в замороженное "
          f"течёт: {len(frozen_grad)}, в кодек: {len(cod_grad)}, "
          f"незамороженных весов кодека: {len(unfrozen)}")

    gen_cnt.close(); dec_cnt.close(); lay_cnt.close()

    print(f"\n  {read_identity(res)}")
    ok = all(v is not False for v in res.values())
    out_j = dict(ok=ok, checks=res, detail=detail, batches=sizes,
                 provenance=detail_prov,
                 rank=int(B.shape[1]), rho_norm=float(np.linalg.norm(rho)),
                 act_tol=ACT_TOL, cache=args.cache,
                 cache_meta_sha1=k11a.file_sha1(args.cache + ".meta.json"),
                 joint_ckpt_sha1=k11a.file_sha1(args.joint_ckpt),
                 hicora_vla_sha1=k11a.file_sha1(hv.__file__),
                 # SHA joint12_vla ОБЯЗАТЕЛЕН: этот модуль определяет
                 # исполняемую сеть не меньше, чем веса, и его версия
                 # записана в каждой ячейке симуляторного гейта K-9.
                 joint12_vla_sha1=k11a.file_sha1(jv.__file__),
                 k11a_sha1=k11a.file_sha1(k11a.__file__),
                 script_sha1=sha)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out_j, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"  сохранено: {args.out}")
    if not ok:
        raise SystemExit("тождество не выполнено — обучение не начинать")


if __name__ == "__main__":
    main()
