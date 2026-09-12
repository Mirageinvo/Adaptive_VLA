#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-12b: протокол обучения HiCoRA-G с подкреплением. Модуль, а не JSON.

ЗАЧЕМ МОДУЛЬ. Три раза за эту серию обычный JSON-протокол оказывался
fail-open: он перечислял поля, но никто не проверял, что прогон им
соответствует, и расхождение обнаруживалось уже в интерпретации чисел (K-11h
оказался блокировкой прогона, а не регистрацией; k12a.load_rates не сверял
источник). Здесь регистрация, проверка и сверка прогона — функции, у каждой
есть причина отказа, и все они закрыты самопроверкой.

ЧТО РЕГИСТРИРУЕТСЯ

  1. ТРИ НЕПЕРЕСЕКАЮЩИХСЯ НАБОРА НАЧАЛЬНЫХ СОСТОЯНИЙ.
     train — раскатки RL и банк отказов; dev — выбор sigma и чекпойнта;
     final — открывается ОДИН раз. Пересечение train и dev с final означало
     бы оценку на обучающей выборке, а это ровно тот дефект, который делает
     любой выигрыш неинтерпретируемым.

  2. СЕТКА sigma ОТДЕЛЬНО ОТ K-11g. Регистрация K-11g выбирала sigma по RMS
     изменения действия, и измерение переходов показало, что этот критерий
     смотрел не туда: восстановления почти не растут от 0.03 к 0.30, а потери
     растут с 2 до 17. Поэтому сетка здесь ОБЯЗАНА содержать 0.03, а критерий
     выбора — баланс восстановлений и потерь на dev, не RMS.

  3. ПРАВИЛА ОСТАНОВКИ. По нижней границе чистого эффекта на dev с терпением,
     а не по одной доле потерь: у неизменившейся политики потерь нет вовсе,
     так что малое r_loss само по себе ничего не значит.

  4. ГЕЙТ УДЕРЖАНИЯ. Предел доли потерь считается из доли отказов базовой
     политики, r_loss_max = (p_fail - delta) / (1 - p_fail), и ПЕРЕСЧИТЫВАЕТСЯ
     при проверке: записанное число ничем не лучше вычисленного.

  5. ОДИН ПОЛНОБАТЧЕВЫЙ ШАГ С ОТКАТОМ. Откат обязан включать состояние
     оптимизатора: у Adam моменты переживают возврат параметров, и следующий
     шаг после «отката» пошёл бы по устаревшему направлению.

  6. ЧЕТЫРЕ РЕПЛИКИ 2 x 2 И ПРАВИЛО ГЕЙТА. Зарегистрировано `mean`;
     `all` остаётся вторичным наблюдением, потому что по расчёту мощности оно
     обваливается с 0.95 до 0.23-0.58 от одного лишнего пункта
     дискордантности.

  7. ЧИСЛО ЭПИЗОДОВ — НЕ СВОБОДНОЕ ЧИСЛО. Оно проверяется против измеренного
     числа СВЕЖИХ начальных состояний (артефакт K-12c). Повтор состояния с
     другим сидом раскатки независимым эпизодом не является.
"""
import argparse
import hashlib
import json
import os
import sys
import time

GATE_RULES = ("mean", "mean_rep", "all")
REGISTERED_RULE = "mean"
STAGES = ("train", "dev", "final")
SPLITS = ("train", "dev", "final")
ROLLBACK_REQUIRED = ("params", "optimizer_state")
SIGMA_REQUIRED = 0.03          # см. пункт 2


class ProtocolError(Exception):
    pass


def _sha12(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def canon(obj):
    """Канонический текст объекта: по нему считается sha протокола.

    Нужен именно канонический вид, иначе переупорядочивание ключей меняло бы
    sha, и «тот же протокол» выглядел бы подменённым.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def proto_sha(proto):
    body = {k: v for k, v in proto.items() if k != "sha1"}
    return hashlib.sha1(canon(body).encode()).hexdigest()[:12]


def parse_ids(spec):
    """"0-39,44" -> [0..39,44]. Дубликаты и пересечения внутри — отказ."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-")
            a, b = int(a), int(b)
            if b < a:
                raise ProtocolError(f"диапазон {part} пуст")
            out += list(range(a, b + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out):
        dup = sorted({i for i in out if out.count(i) > 1})
        raise ProtocolError(f"повторяющиеся id в {spec}: {dup}")
    return sorted(out)


def r_loss_max(p_fail, delta):
    """Предел доли потерь, при котором чистый эффект delta ещё достижим.

    Выводится из delta = p_fail*r_rec - (1-p_fail)*r_loss при r_rec = 1:
    даже полное восстановление всех отказов не компенсирует больших потерь.
    """
    p_fail, delta = float(p_fail), float(delta)
    if not 0.0 < p_fail < 1.0:
        raise ProtocolError(f"p_fail={p_fail} вне (0,1)")
    if delta <= 0:
        raise ProtocolError("delta должна быть положительной")
    if delta >= p_fail:
        raise ProtocolError(
            f"эффект {delta} не меньше доли отказов {p_fail}: потолок эффекта "
            f"равен p_fail*r_rec <= p_fail, поэтому он недостижим даже при "
            f"восстановлении ВСЕХ отказов, независимо от потерь")
    return (p_fail - delta) / (1.0 - p_fail)


# ------------------------------ регистрация -------------------------------

def init_protocol(*, splits, sigma_grid, tasks, n_episodes_final,
                  delta_target, discord_max, p_fail_by_head, replicas,
                  gate_rule=REGISTERED_RULE, stop=None, step=None,
                  bootstrap=None, states_json=None, power_json=None,
                  scripts=None, k11g_protocol=None, k11e_protocol=None,
                  notes=""):
    """Собрать протокол. НЕ пишет файл: запись — отдельное решение."""
    stop = dict(stop or {})
    step = dict(step or {})
    proto = dict(
        version=1,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        splits={k: parse_ids(v) for k, v in splits.items()},
        sigma_grid=sorted({round(float(s), 6) for s in sigma_grid}),
        sigma_criterion="net_effect_dev",
        tasks=sorted({int(t) for t in tasks}),
        n_episodes_final=int(n_episodes_final),
        delta_target=float(delta_target),
        discord_max=float(discord_max),
        p_fail_by_head={str(k): float(v) for k, v in p_fail_by_head.items()},
        replicas=[dict(d1_seed=int(a), rl_seed=int(b)) for a, b in replicas],
        gate_rule=str(gate_rule),
        gate_rules_secondary=[r for r in GATE_RULES if r != gate_rule],
        stop=stop, step=step,
        bootstrap=dict(bootstrap or dict(n_boot=20000, alpha=0.05, seed=0)),
        states_json=states_json, power_json=power_json,
        k11g_protocol=k11g_protocol, k11e_protocol=k11e_protocol,
        scripts=dict(scripts or {}), notes=str(notes))
    proto["r_loss_max_by_head"] = {
        h: round(r_loss_max(p, proto["delta_target"]), 6)
        for h, p in proto["p_fail_by_head"].items()}
    proto["sha1"] = proto_sha(proto)
    return proto


def check_protocol(proto):
    """Все отказы сразу, а не первый: иначе регистрация правится по одному
    пункту и каждый раз запускается заново."""
    bad = []
    if proto.get("version") != 1:
        bad.append(f"версия протокола {proto.get('version')}, ожидалась 1")
    if proto.get("sha1") != proto_sha(proto):
        bad.append("sha1 не совпадает с содержимым: протокол изменён после "
                   "регистрации")

    sp = proto.get("splits") or {}
    for name in SPLITS:
        if not sp.get(name):
            bad.append(f"нет набора состояний '{name}'")
    for a in SPLITS:
        for b in SPLITS:
            if a < b and sp.get(a) and sp.get(b):
                inter = sorted(set(sp[a]) & set(sp[b]))
                if inter:
                    bad.append(f"наборы {a} и {b} пересекаются по {inter[:8]}"
                               f" ({len(inter)} шт.): это оценка на обучающей "
                               f"выборке")
    if sp.get("final") and len(sp["final"]) < proto.get("n_episodes_final", 0):
        bad.append(f"в final {len(sp['final'])} состояний, а эпизодов на "
                   f"задачу заявлено {proto.get('n_episodes_final')}: разница "
                   f"набиралась бы повторами одного состояния")

    grid = proto.get("sigma_grid") or []
    if not grid:
        bad.append("пустая сетка sigma")
    if SIGMA_REQUIRED not in [round(float(s), 6) for s in grid]:
        bad.append(f"в сетке sigma нет {SIGMA_REQUIRED}: измерение переходов "
                   f"K-11g показало там те же восстановления при втрое "
                   f"меньших потерях, и исключать её нельзя")
    if any(float(s) <= 0 for s in grid):
        bad.append("в сетке sigma есть неположительное значение: нулевая "
                   "sigma — это детерминированная D1, а не политика")
    if proto.get("sigma_criterion") != "net_effect_dev":
        bad.append("критерий выбора sigma не по чистому эффекту на dev: RMS "
                   "изменения действия уже один раз измерял не ту величину")

    if not proto.get("tasks"):
        bad.append("не перечислены задачи")
    if proto.get("delta_target", 0) <= 0:
        bad.append("delta_target должна быть положительной")
    dm, dt = proto.get("discord_max"), proto.get("delta_target")
    if dm is None:
        bad.append("не задан предел дискордантности")
    elif dt and float(dm) > float(dt) + 0.01 + 1e-9:
        bad.append(f"предел дискордантности {dm} выше эффекта {dt} более чем "
                   f"на 1 пункт: по расчёту мощности это уже нерабочая область")

    pf = proto.get("p_fail_by_head") or {}
    if not pf:
        bad.append("не заданы доли отказов базовой политики по головам")
    rl = proto.get("r_loss_max_by_head") or {}
    for h, p in pf.items():
        try:
            want = round(r_loss_max(p, proto["delta_target"]), 6)
        except (ProtocolError, KeyError, TypeError) as e:
            bad.append(f"предел потерь для {h} не вычисляется: {e}")
            continue
        got = rl.get(h)
        if got is None or abs(float(got) - want) > 1e-6:
            bad.append(f"предел потерь для {h}: записано {got}, из p_fail="
                       f"{p} и delta={proto['delta_target']} следует {want}")

    reps = proto.get("replicas") or []
    keys = [(r.get("d1_seed"), r.get("rl_seed")) for r in reps]
    if len(keys) < 2:
        bad.append("меньше двух реплик: разброс по сидам тогда не измерен, а "
                   "в K-11e он был равен самому эффекту")
    if len(set(keys)) != len(keys):
        bad.append(f"повторяющиеся реплики {keys}: это один прогон, "
                   f"посчитанный дважды")
    if len({k[0] for k in keys}) < 2:
        bad.append("все реплики на одном сиде D1: обобщения на метод нет")
    if len({k[1] for k in keys}) < 2:
        bad.append("все реплики на одном сиде RL")

    if proto.get("gate_rule") not in GATE_RULES:
        bad.append(f"правило гейта {proto.get('gate_rule')} не из {GATE_RULES}")
    if proto.get("gate_rule") == "all":
        bad.append("правило 'all' зарегистрировано как основное: по расчёту "
                   "мощности оно падает до 0.23-0.58 от одного лишнего пункта "
                   "дискордантности, ему место во вторичных")

    st = proto.get("stop") or {}
    for k in ("max_steps", "patience", "criterion"):
        if k not in st:
            bad.append(f"в правиле остановки нет '{k}'")
    if st.get("criterion") not in (None, "net_effect_lower_dev"):
        bad.append("критерий остановки не по нижней границе чистого эффекта "
                   "на dev: по одной доле потерь останавливаться нельзя, у "
                   "неизменившейся политики она равна нулю")
    if st.get("max_steps") is not None and int(st["max_steps"]) < 1:
        bad.append("max_steps меньше одного шага")

    sg = proto.get("step") or {}
    if not sg.get("full_batch"):
        bad.append("шаг не полнобатчевый: K-11i показал, что минибатчевый PPO "
                   "здесь непригоден при любом lr")
    if float(sg.get("lr", 1e9)) > 3e-6 + 1e-12:
        bad.append(f"lr={sg.get('lr')} выше измеренного предела 3e-6")
    bt = sg.get("backtrack") or {}
    if int(bt.get("max_halvings", 0)) < 1:
        bad.append("нет дробления шага: единственный шаг без отката можно "
                   "сделать только один раз и нельзя исправить")
    tr = sg.get("trust") or {}
    if not tr.get("kl_max") or not tr.get("ratio_max"):
        bad.append("в шаге нет области доверия с kl_max И ratio_max: K-11i "
                   "показал, что опасность — хвост отношения правдоподобий на "
                   "СОХРАНЁННЫХ сэмплах, а KL на свежих его не видит; "
                   "ограничивать нужно обе величины")
    if "train_log_std" not in sg:
        bad.append("не сказано, учится ли log_std: если учится, sigma перестаёт"
                   " быть зарегистрированной константой, и следующая раскатка "
                   "идёт не под той sigma, что записана")
    if sg.get("train_log_std"):
        bad.append("log_std обучается на первом этапе: sigma выбирается на dev "
                   "и фиксируется, иначе выбор sigma и обучение смешиваются")
    if sg.get("head_precision") != "fp32":
        bad.append("точность головы не fp32: под autocast fp16 разрешение mu "
                   "(~1e-3 относительных) крупнее самого шага при lr 3e-6, и "
                   "шаг был бы НЕОТЛИЧИМ от отсутствия шага — отношение "
                   "правдоподобий осталось бы единицей из-за округления")
    miss = [k for k in ROLLBACK_REQUIRED if k not in (sg.get("rollback") or [])]
    if miss:
        bad.append(f"откат не включает {miss}: моменты Adam переживают возврат "
                   f"параметров, и следующий шаг пошёл бы по устаревшему "
                   f"направлению")

    bs = proto.get("bootstrap") or {}
    if int(bs.get("n_boot", 0)) < 2000:
        bad.append(f"n_boot={bs.get('n_boot')}: квантиль бутстрапа при таком "
                   f"числе повторов шумит сильнее, чем сам эффект")
    if not 0.0 < float(bs.get("alpha", 0)) < 0.5:
        bad.append(f"alpha={bs.get('alpha')} вне (0, 0.5): односторонний "
                   f"уровень должен быть зарегистрирован явно")
    if "seed" not in bs:
        bad.append("не зарегистрирован сид бутстрапа: без него границу можно "
                   "пересчитывать до нужного результата")
    for key in ("states_json", "power_json", "k11g_protocol", "k11e_protocol"):
        ref = proto.get(key)
        if not ref or not isinstance(ref, dict) or not ref.get("sha1"):
            bad.append(f"нет привязки к {key} с sha1: без неё протокол ссылается"
                       f" на числа, происхождение которых не проверить")
    if not proto.get("scripts"):
        bad.append("не записаны sha скриптов")

    # ЧИСЛО ЭПИЗОДОВ ПРОТИВ ИЗМЕРЕННЫХ СОСТОЯНИЙ
    sj = proto.get("states_json") or {}
    mf = sj.get("min_fresh")
    if mf is not None and proto.get("n_episodes_final") is not None:
        if int(proto["n_episodes_final"]) > int(mf):
            bad.append(f"эпизодов на задачу {proto['n_episodes_final']}, а "
                       f"свежих состояний измерено {mf}: остальное были бы "
                       f"повторы одного состояния")
    if sj.get("any_silent_clamp"):
        bad.append("в артефакте состояний отмечен молчаливый кламп id: часть "
                   "'разных' эпизодов — копии")
    if bad:
        raise ProtocolError("протокол не проходит проверку:\n  - "
                            + "\n  - ".join(bad))
    return True


def write_protocol(path, proto, *, allow_same=True):
    """Атомарная запись. Перезапись ДРУГИМ содержимым — отказ.

    Регистрация, которую можно молча переписать после первого результата,
    регистрацией не является.
    """
    check_protocol(proto)
    if os.path.exists(path):
        old = json.load(open(path))
        if old.get("sha1") == proto.get("sha1") and allow_same:
            return False
        raise ProtocolError(
            f"{path} уже существует с sha1={old.get('sha1')}, а новый "
            f"{proto.get('sha1')}: перерегистрация запрещена")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(proto, open(tmp, "w"), ensure_ascii=False, indent=1,
              sort_keys=True)
    os.replace(tmp, path)
    return True


def load_protocol(path):
    proto = json.load(open(path))
    check_protocol(proto)
    return proto


# ------------------ решения на dev и одноразовый final ---------------------

def seal_decisions(path, proto, *, sigma, checkpoints, dev_evidence):
    """Запечатать решения, принятые на dev: sigma и чекпойнт КАЖДОЙ реплики.

    Почему sha, а не просто файл: final открывается по этой печати, и если
    решения можно дописать после, то «выбрано на dev» становится «выбрано на
    final». Требуются все реплики: выбор чекпойнта по одной и применение ко
    всем — это утечка между репликами.
    """
    sig = round(float(sigma), 6)
    if sig not in [round(float(s), 6) for s in proto["sigma_grid"]]:
        raise ProtocolError(f"sigma={sig} вне зарегистрированной сетки "
                            f"{proto['sigma_grid']}")
    want = {replica_key(r) for r in proto["replicas"]}
    got = set(checkpoints)
    if got != want:
        raise ProtocolError(f"чекпойнты заданы для {sorted(got)}, а реплики "
                            f"{sorted(want)}: выбор по части реплик и перенос "
                            f"на остальные — утечка")
    for k, c in checkpoints.items():
        if not isinstance(c, dict) or not c.get("sha1") or not c.get("path"):
            raise ProtocolError(f"чекпойнт {k} без path и sha1")
    if not dev_evidence or not dev_evidence.get("source"):
        raise ProtocolError("нет ссылки на измерение на dev, по которому "
                            "выбраны sigma и чекпойнт")
    dec = dict(protocol_sha1=proto["sha1"], sigma=sig,
               checkpoints=dict(checkpoints), dev_evidence=dict(dev_evidence),
               sealed=time.strftime("%Y-%m-%dT%H:%M:%S"))
    dec["sha1"] = hashlib.sha1(
        canon({k: v for k, v in dec.items() if k != "sha1"}).encode()
    ).hexdigest()[:12]
    if os.path.exists(path):
        old = json.load(open(path))
        if old.get("sha1") == dec["sha1"]:
            return dec
        raise ProtocolError(f"решения уже запечатаны (sha1={old.get('sha1')}) "
                            f"и отличаются от новых {dec['sha1']}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(dec, open(tmp, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return dec


def open_final(path, proto, decisions):
    """Одноразовое открытие final. Второе открытие — отказ.

    Печать решений обязана предшествовать: открытие final до фиксации sigma и
    чекпойнта превращает финальную оценку в ещё один dev.
    """
    if decisions.get("protocol_sha1") != proto["sha1"]:
        raise ProtocolError("решения запечатаны под другой протокол")
    if os.path.exists(path):
        old = json.load(open(path))
        raise ProtocolError(
            f"final уже открывался {old.get('opened')} под решения "
            f"{old.get('decisions_sha1')}: повторное открытие запрещено")
    rec = dict(protocol_sha1=proto["sha1"], decisions_sha1=decisions["sha1"],
               sigma=decisions["sigma"],
               checkpoints={k: v["sha1"] for k, v
                            in decisions["checkpoints"].items()},
               state_ids=proto["splits"]["final"], tasks=proto["tasks"],
               opened=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return rec


def replica_key(r):
    return f"d1{int(r['d1_seed'])}_rl{int(r['rl_seed'])}"


# ---------------------------- сверка прогона ------------------------------

def check_run(proto, record, *, decisions=None, final_open=None,
              partial=False):
    """Соответствует ли прогон протоколу. Отказов сразу все.

    `partial=True` — запись ОДНОЙ ячейки (блок состояний одной задачи), а не
    прогона целиком: тогда проверяется включение в набор, но не полнота. Полнота
    финальной оценки проверяется отдельно, `check_final_complete`: иначе
    «final на одном состоянии из 80» прошёл бы как финальный прогон.

    ОБЯЗАТЕЛЬНЫЕ ПОЛЯ ТРЕБУЮТСЯ, А НЕ ПРОВЕРЯЮТСЯ ПРИ НАЛИЧИИ. Проверка вида
    «если поле есть, оно должно быть верным» пропускает запись без поля, то
    есть ровно тот случай, когда сверять нечего.
    """
    bad = []
    stage = record.get("stage")
    if stage not in STAGES:
        bad.append(f"этап '{stage}' не из {STAGES}")
    if record.get("protocol_sha1") != proto.get("sha1"):
        bad.append(f"прогон помечен протоколом {record.get('protocol_sha1')}, "
                   f"а проверяется против {proto.get('sha1')}")
    for need in ("replica", "sigma", "state_ids", "task_ids"):
        if record.get(need) in (None, "", [], {}):
            bad.append(f"в записи прогона нет обязательного поля '{need}'")
    ids = sorted({int(i) for i in record.get("state_ids") or []})
    if not ids:
        bad.append("в записи прогона нет начальных состояний")
    if stage in SPLITS:
        allowed = set(proto["splits"].get(stage, []))
        leaked = sorted(set(ids) - allowed)
        if leaked:
            bad.append(f"состояния {leaked[:8]} ({len(leaked)} шт.) не из "
                       f"набора '{stage}'")
        for other in SPLITS:
            if other == stage:
                continue
            both = sorted(set(ids) & set(proto["splits"].get(other, [])))
            if both:
                bad.append(f"состояния {both[:8]} принадлежат набору "
                           f"'{other}': перенос между наборами")
    tasks = sorted({int(t) for t in record.get("task_ids") or []})
    unknown = sorted(set(tasks) - set(proto["tasks"]))
    if unknown:
        bad.append(f"задачи {unknown} не зарегистрированы")
    if not partial and stage == "final" and set(tasks) != set(proto["tasks"]):
        bad.append(f"финальная оценка по задачам {tasks}, а зарегистрированы "
                   f"{proto['tasks']}: выбор подмножества задач после "
                   f"обучения — это выбор результата")

    sig = record.get("sigma")
    if sig is None:
        bad.append("в записи нет sigma")
    elif stage in ("train", "dev"):
        if round(float(sig), 6) not in [round(float(s), 6)
                                        for s in proto["sigma_grid"]]:
            bad.append(f"sigma={sig} вне сетки {proto['sigma_grid']}")
    if record.get("replica") not in [replica_key(r) for r in proto["replicas"]]:
        bad.append(f"реплика {record.get('replica')} не зарегистрирована")

    if stage == "final":
        if not record.get("ckpt_sha1"):
            bad.append("финальный прогон без ckpt_sha1: нечего сверять с "
                       "запечатанным решением, и подменённый чекпойнт прошёл "
                       "бы незамеченным")
        if not partial:
            want_ids = set(proto["splits"]["final"])
            if set(ids) != want_ids:
                miss = sorted(want_ids - set(ids))[:6]
                extra = sorted(set(ids) - want_ids)[:6]
                bad.append(f"финальная оценка не на зарегистрированном наборе: "
                           f"не хватает {len(want_ids - set(ids))} "
                           f"({miss}), лишних "
                           f"{len(set(ids) - want_ids)} ({extra})")
            n_per = record.get("n_episodes_per_task")
            if n_per is None:
                bad.append("нет n_episodes_per_task: число эпизодов на задачу "
                           "зарегистрировано и обязано быть записано")
            elif int(n_per) != int(proto["n_episodes_final"]):
                bad.append(f"эпизодов на задачу {n_per}, зарегистрировано "
                           f"{proto['n_episodes_final']}")
        if decisions is None or final_open is None:
            bad.append("финальный прогон без запечатанных решений или без "
                       "записи об открытии final")
        else:
            if decisions.get("protocol_sha1") != proto.get("sha1"):
                bad.append("решения от другого протокола")
            if final_open.get("decisions_sha1") != decisions.get("sha1"):
                bad.append("final открывался под другие решения")
            if sig is None or round(float(sig), 6) != round(
                    float(decisions["sigma"]), 6):
                bad.append(f"в финальном прогоне sigma={sig}, а запечатана "
                           f"{decisions['sigma']}: это выбор после открытия "
                           f"final")
            rk = record.get("replica")
            ck = (decisions.get("checkpoints") or {}).get(rk)
            if ck is None:
                bad.append(f"для реплики {rk} нет запечатанного чекпойнта")
            elif record.get("ckpt_sha1") != ck["sha1"]:
                bad.append(f"чекпойнт реплики {rk} ({record.get('ckpt_sha1')}) "
                           f"не тот, что запечатан ({ck['sha1']})")
    if bad:
        raise ProtocolError("прогон не соответствует протоколу:\n  - "
                            + "\n  - ".join(bad))
    return True


def check_final_complete(proto, records, *, decisions, final_open):
    """Полнота финальной оценки по набору ЯЧЕЕК.

    Каждая ячейка проверяется как частичная (`partial=True`), а полнота — здесь:
    для каждой реплики объединение состояний по каждой задаче обязано в точности
    совпасть с зарегистрированным набором final, без повторов. Без этой функции
    «финальный прогон» на одном состоянии одной задачи выглядел бы законным.
    """
    bad = []
    want_ids = set(proto["splits"]["final"])
    want_reps = {replica_key(r) for r in proto["replicas"]}
    seen = {}
    for rec in records:
        try:
            check_run(proto, rec, decisions=decisions, final_open=final_open,
                      partial=True)
        except ProtocolError as e:
            bad.append(f"ячейка {rec.get('replica')}/{rec.get('task_ids')}: {e}")
            continue
        if rec.get("stage") != "final":
            bad.append(f"ячейка на этапе {rec.get('stage')}, а не final")
            continue
        for t in rec["task_ids"]:
            key = (rec["replica"], int(t))
            for i in rec["state_ids"]:
                dup = seen.setdefault(key, [])
                if int(i) in dup:
                    bad.append(f"{key}: состояние {i} посчитано дважды — "
                               f"эпизод вошёл бы в оценку с двойным весом")
                dup.append(int(i))
    missing_reps = sorted(want_reps - {k[0] for k in seen})
    if missing_reps:
        bad.append(f"нет ячеек для реплик {missing_reps}: гейт по среднему "
                   f"четырёх реплик нельзя считать по трём")
    for rk in sorted(want_reps & {k[0] for k in seen}):
        for t in proto["tasks"]:
            got = set(seen.get((rk, int(t)), []))
            if got != want_ids:
                bad.append(
                    f"{rk}, задача {t}: состояний {len(got)} из "
                    f"{len(want_ids)}, не хватает "
                    f"{sorted(want_ids - got)[:6]}")
    if bad:
        raise ProtocolError("финальная оценка неполна:\n  - "
                            + "\n  - ".join(bad))
    return dict(replicas=sorted(want_reps), tasks=list(proto["tasks"]),
                n_states=len(want_ids),
                n_episodes=len(want_reps) * len(proto["tasks"]) * len(want_ids))


def gate(proto, diffs, *, n_boot=None, seed=None, alpha=None):
    """Итог гейта по зарегистрированному правилу. СОВМЕСТНЫЙ бутстрап.

    `diffs` — {реплика: {задача: (средняя парная разность, число пар)}}.

    ПОЧЕМУ НЕ СРЕДНЕЕ НИЖНИХ ГРАНИЦ. Среднее четырёх односторонних границ —
    не граница среднего: каждая из них уже сдвинута вниз на свой запас, и их
    среднее не имеет заявленного покрытия. Нижняя граница СРЕДНЕГО получается
    только совместным ресэмплированием задач, когда на каждой итерации
    бутстрапа берутся ОДНИ И ТЕ ЖЕ задачи у всех реплик и усредняется уже
    результат. Поэтому вычисление делегировано `k12b_power_hier.lower_mean`,
    тому же коду, которым считалась мощность: расчёт мощности и гейт обязаны
    быть одной процедурой, иначе зарегистрированная мощность относится не к
    тому правилу, по которому принимается решение.

    Наивное среднее границ всё же считается и возвращается — но под именем
    `naive_mean_of_lowers` и с пометкой, что границей оно не является: иначе
    это число однажды снова прочитают как вывод.

    ЗАДАЧИ ВЫРАВНИВАЮТСЯ МЕЖДУ РЕПЛИКАМИ. Совместный бутстрап индексирует
    столбцы, и если у реплик разный набор задач, то ресэмплирование смешало бы
    разные задачи в один столбец.
    """
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import k12b_power_hier as ph

    bs = proto.get("bootstrap") or {}
    n_boot = int(n_boot if n_boot is not None else bs["n_boot"])
    seed = int(seed if seed is not None else bs["seed"])
    alpha = float(alpha if alpha is not None else bs["alpha"])

    keys = [replica_key(r) for r in proto["replicas"]]
    if not isinstance(diffs, dict):
        raise ProtocolError(
            "gate ожидает {реплика: {задача: (разность, число пар)}}, а не "
            "список нижних границ: среднее границ границей среднего не "
            "является")
    if set(diffs) != set(keys):
        raise ProtocolError(f"поданы реплики {sorted(diffs)}, "
                            f"зарегистрированы {sorted(keys)}")
    tasks = [int(t) for t in proto["tasks"]]
    reps = []
    for rk in keys:
        d = {int(k): v for k, v in diffs[rk].items()}
        if set(d) != set(tasks):
            raise ProtocolError(
                f"реплика {rk}: задачи {sorted(d)} вместо {tasks} — "
                f"совместный бутстрап смешал бы разные задачи в один столбец")
        row = []
        for t in tasks:
            diff, w = float(d[t][0]), float(d[t][1])
            if not w > 0:
                raise ProtocolError(f"реплика {rk}, задача {t}: вес {w} не "
                                    f"положителен")
            row.append((diff, w))
        reps.append(row)

    lo_mean = ph.lower_mean(reps, n_boot, np.random.default_rng(seed),
                            resample_replicas=False, alpha=alpha)
    lo_rep = ph.lower_mean(reps, n_boot, np.random.default_rng(seed + 1),
                           resample_replicas=True, alpha=alpha)
    lo_all = ph.lower_all(reps, n_boot, np.random.default_rng(seed + 2),
                          alpha=alpha)
    per_rep = [ph.cluster_lower(r, n_boot,
                                np.random.default_rng(seed + 10 + i), alpha)
               for i, r in enumerate(reps)]
    point = [sum(d * w for d, w in r) / sum(w for _d, w in r) for r in reps]
    res = dict(registered=proto["gate_rule"],
               lower=dict(mean=lo_mean, mean_rep=lo_rep, all=lo_all),
               values=dict(mean=bool(lo_mean > 0), mean_rep=bool(lo_rep > 0),
                           all=bool(lo_all > 0)),
               per_replica_lower=per_rep, per_replica_point=point,
               naive_mean_of_lowers=sum(per_rep) / len(per_rep),
               naive_is_not_a_bound=True,
               n_boot=n_boot, alpha=alpha, seed=seed)
    res["passed"] = bool(res["values"][proto["gate_rule"]])
    return res


# ------------------------------ самопроверка ------------------------------

def _proto_ok(**over):
    kw = dict(
        splits={"train": "0-29", "dev": "30-44", "final": "45-124"},
        sigma_grid=[0.03, 0.05, 0.10],
        tasks=range(10), n_episodes_final=80,
        delta_target=0.05, discord_max=0.05,
        p_fail_by_head={"s0": 0.12, "s1": 0.10},
        replicas=[(0, 0), (0, 1), (1, 0), (1, 1)],
        stop=dict(max_steps=8, patience=2, criterion="net_effect_lower_dev"),
        step=dict(full_batch=True, lr=3e-6, head_precision="fp32",
                  train_log_std=False,
                  trust=dict(kl_max=0.02, ratio_max=1.5),
                  backtrack=dict(max_halvings=4, accept="net_effect_dev"),
                  rollback=["params", "optimizer_state"]),
        states_json=dict(path="data/k12c_states.json", sha1="aaaaaaaaaaaa",
                         min_fresh=80, any_silent_clamp=False),
        power_json=dict(path="data/k12b/power_hier.json", sha1="bbbbbbbbbbbb",
                        power_mean=1.0),
        k11g_protocol=dict(path="data/k11g/protocol.json", sha1="cccccccccccc"),
        k11e_protocol=dict(path="data/k11e/protocol.json", sha1="dddddddddddd"),
        scripts={"k12d_pg_step.py": "eeeeeeeeeeee"})
    kw.update(over)
    # _proto_ok = «собрать И проверить»: иначе тесты неверных конфигураций
    # молча проходили бы, ведь init_protocol сам ничего не проверяет
    p = init_protocol(**kw)
    check_protocol(p)
    return p


def _expect(fn, needle):
    try:
        fn()
    except ProtocolError as e:
        assert needle in str(e), f"ожидал «{needle}», получил: {e}"
        return str(e)
    raise AssertionError(f"отказа «{needle}» не было")


def selftest(tmpdir=None):
    import tempfile
    tmp = tmpdir or tempfile.mkdtemp(prefix="k12b_")

    # --- базовый протокол проходит, и его sha устойчив к порядку ключей ----
    p = _proto_ok()
    assert check_protocol(p)
    p2 = json.loads(json.dumps(p, sort_keys=True))
    assert proto_sha(p2) == p["sha1"], "sha зависит от порядка ключей"
    # предел потерь пересчитывается, а не берётся на веру
    # сравнение с допуском округления самого поля: предел хранится с шестью
    # знаками, и сверка по 1e-9 падала бы на самом округлении, а не на ошибке
    assert abs(p["r_loss_max_by_head"]["s0"]
               - (0.12 - 0.05) / 0.88) < 1e-6, p["r_loss_max_by_head"]
    bad = dict(p)
    bad["r_loss_max_by_head"] = {"s0": 0.05, "s1": 0.05}
    bad["sha1"] = proto_sha(bad)
    _expect(lambda: check_protocol(bad), "предел потерь для")

    # --- 1. пересекающиеся наборы состояний -------------------------------
    _expect(lambda: _proto_ok(splits={"train": "0-29", "dev": "25-44",
                                      "final": "45-124"}),
            "пересекаются")
    _expect(lambda: _proto_ok(splits={"train": "0-29", "dev": "30-44",
                                      "final": "20-99"}),
            "пересекаются")
    # final короче заявленного числа эпизодов
    _expect(lambda: _proto_ok(splits={"train": "0-29", "dev": "30-44",
                                      "final": "45-84"},
                              n_episodes_final=80),
            "набиралась бы повторами")
    # повтор id внутри одного набора
    _expect(lambda: _proto_ok(splits={"train": "0-29,5", "dev": "30-44",
                                      "final": "45-124"}),
            "повторяющиеся id")

    # --- 2. сетка sigma ----------------------------------------------------
    _expect(lambda: _proto_ok(sigma_grid=[0.05, 0.10, 0.30]), "нет 0.03")
    _expect(lambda: _proto_ok(sigma_grid=[0.0, 0.03]), "неположительное")
    pc = dict(_proto_ok())
    pc["sigma_criterion"] = "rms_action_change"
    pc["sha1"] = proto_sha(pc)
    _expect(lambda: check_protocol(pc), "измерял не ту величину")

    # --- 3-4. остановка и гейт удержания ----------------------------------
    _expect(lambda: _proto_ok(stop=dict(max_steps=8, patience=2)), "'criterion'")
    _expect(lambda: _proto_ok(stop=dict(max_steps=8, patience=2,
                                        criterion="r_loss")),
            "по одной доле потерь")
    # эффект выше доли отказов недостижим в принципе
    _expect(lambda: _proto_ok(delta_target=0.20,
                              p_fail_by_head={"s0": 0.12, "s1": 0.10},
                              discord_max=0.20),
            "недостижим даже при")

    # --- 5. шаг и откат ----------------------------------------------------
    def _step(**over):
        base = dict(full_batch=True, lr=3e-6, head_precision="fp32",
                    train_log_std=False,
                    trust=dict(kl_max=0.02, ratio_max=1.5),
                    backtrack=dict(max_halvings=4, accept="net_effect_dev"),
                    rollback=["params", "optimizer_state"])
        base.update(over)
        return dict(base)
    _expect(lambda: _proto_ok(step=_step(full_batch=False)), "не полнобатчевый")
    _expect(lambda: _proto_ok(step=_step(lr=1e-5)), "выше измеренного предела")
    _expect(lambda: _proto_ok(step=_step(backtrack=dict(max_halvings=0))),
            "нет дробления шага")
    _expect(lambda: _proto_ok(step=_step(rollback=["params"])), "моменты Adam")
    # область доверия: одного KL недостаточно
    _expect(lambda: _proto_ok(step=_step(trust=dict(kl_max=0.02))),
            "хвост отношения правдоподобий")
    _expect(lambda: _proto_ok(step=_step(trust=dict(ratio_max=1.5))),
            "хвост отношения правдоподобий")
    st_no = _step()
    st_no.pop("train_log_std")
    _expect(lambda: _proto_ok(step=st_no), "учится ли log_std")
    _expect(lambda: _proto_ok(step=_step(train_log_std=True)),
            "log_std обучается")
    _expect(lambda: _proto_ok(step=_step(head_precision="fp16")),
            "НЕОТЛИЧИМ от отсутствия шага")

    # --- 6. реплики и правило гейта ---------------------------------------
    _expect(lambda: _proto_ok(replicas=[(0, 0), (0, 1)]), "одном сиде D1")
    _expect(lambda: _proto_ok(replicas=[(0, 0), (1, 0)]), "одном сиде RL")
    _expect(lambda: _proto_ok(replicas=[(0, 0), (0, 0), (1, 1)]),
            "повторяющиеся реплики")
    _expect(lambda: _proto_ok(replicas=[(0, 0)]), "меньше двух реплик")
    _expect(lambda: _proto_ok(gate_rule="all"), "'all' зарегистрировано")
    _expect(lambda: _proto_ok(gate_rule="median"), "не из")

    # --- 7. эпизоды против измеренных состояний ---------------------------
    _expect(lambda: _proto_ok(
        states_json=dict(path="s", sha1="x", min_fresh=45,
                         any_silent_clamp=False),
        splits={"train": "0-29", "dev": "30-44", "final": "45-124"}),
        "свежих состояний измерено 45")
    _expect(lambda: _proto_ok(
        states_json=dict(path="s", sha1="x", min_fresh=80,
                         any_silent_clamp=True)), "молчаливый кламп")
    _expect(lambda: _proto_ok(states_json=dict(path="s", min_fresh=80)),
            "states_json с sha1")
    _expect(lambda: _proto_ok(power_json=None), "power_json")
    _expect(lambda: _proto_ok(scripts={}), "sha скриптов")
    # дискордантность заметно выше эффекта
    _expect(lambda: _proto_ok(discord_max=0.08), "нерабочая область")

    # --- запись: перерегистрация запрещена, повтор того же — нет ----------
    path = os.path.join(tmp, "protocol.json")
    assert write_protocol(path, p) is True
    assert write_protocol(path, p) is False, "идемпотентная запись отказала"
    p_other = _proto_ok(stop=dict(max_steps=9, patience=2,
                                  criterion="net_effect_lower_dev"))
    _expect(lambda: write_protocol(path, p_other), "перерегистрация запрещена")
    assert load_protocol(path)["sha1"] == p["sha1"]
    # правка файла после регистрации ловится по sha
    obj = json.load(open(path))
    obj["delta_target"] = 0.03
    json.dump(obj, open(path, "w"))
    _expect(lambda: load_protocol(path), "изменён после регистрации")
    json.dump(p, open(path, "w"))

    # --- решения на dev ---------------------------------------------------
    cks = {replica_key(r): dict(path=f"ck_{replica_key(r)}.pt",
                                sha1=f"s{i:011d}")
           for i, r in enumerate(p["replicas"])}
    dpath = os.path.join(tmp, "decisions.json")
    ev = dict(source="data/k12b/dev_sigma.json", sha1="f" * 12)
    _expect(lambda: seal_decisions(dpath, p, sigma=0.07, checkpoints=cks,
                                   dev_evidence=ev), "вне зарегистрированной")
    part = {k: v for k, v in list(cks.items())[:2]}
    _expect(lambda: seal_decisions(dpath, p, sigma=0.03, checkpoints=part,
                                   dev_evidence=ev), "утечка")
    _expect(lambda: seal_decisions(dpath, p, sigma=0.03, checkpoints=cks,
                                   dev_evidence={}), "выбраны sigma")
    dec = seal_decisions(dpath, p, sigma=0.03, checkpoints=cks,
                         dev_evidence=ev)
    assert seal_decisions(dpath, p, sigma=0.03, checkpoints=cks,
                          dev_evidence=ev)["sha1"] == dec["sha1"]
    _expect(lambda: seal_decisions(dpath, p, sigma=0.05, checkpoints=cks,
                                   dev_evidence=ev), "уже запечатаны")

    # --- final открывается один раз --------------------------------------
    fpath = os.path.join(tmp, "final_open.json")
    fo = open_final(fpath, p, dec)
    assert fo["sigma"] == 0.03 and len(fo["state_ids"]) == 80
    _expect(lambda: open_final(fpath, p, dec), "повторное открытие")
    other_dec = dict(dec, protocol_sha1="zzzzzzzzzzzz")
    _expect(lambda: open_final(os.path.join(tmp, "f2.json"), p, other_dec),
            "другой протокол")

    # --- сверка прогонов --------------------------------------------------
    base = dict(stage="train", protocol_sha1=p["sha1"],
                state_ids=list(range(30)), task_ids=list(range(10)),
                sigma=0.03, replica="d10_rl0")
    assert check_run(p, base)
    _expect(lambda: check_run(p, dict(base, state_ids=[0, 1, 35])),
            "принадлежат набору 'dev'")
    _expect(lambda: check_run(p, dict(base, stage="dev")),
            "принадлежат набору 'train'")
    _expect(lambda: check_run(p, dict(base, sigma=0.07)), "вне сетки")
    _expect(lambda: check_run(p, dict(base, task_ids=[0, 42])),
            "не зарегистрированы")
    _expect(lambda: check_run(p, dict(base, replica="d17_rl9")),
            "не зарегистрирована")
    _expect(lambda: check_run(p, dict(base, protocol_sha1="0" * 12)),
            "помечен протоколом")
    fin = dict(stage="final", protocol_sha1=p["sha1"],
               state_ids=p["splits"]["final"], task_ids=p["tasks"],
               sigma=0.03, replica="d10_rl0", n_episodes_per_task=80,
               ckpt_sha1=cks["d10_rl0"]["sha1"])
    assert check_run(p, fin, decisions=dec, final_open=fo)
    _expect(lambda: check_run(p, fin), "без запечатанных решений")
    _expect(lambda: check_run(p, dict(fin, sigma=0.05), decisions=dec,
                              final_open=fo), "выбор после открытия final")
    _expect(lambda: check_run(p, dict(fin, ckpt_sha1="9" * 12), decisions=dec,
                              final_open=fo), "не тот, что запечатан")
    _expect(lambda: check_run(p, dict(fin, n_episodes_per_task=40),
                              decisions=dec, final_open=fo),
            "зарегистрировано 80")
    _expect(lambda: check_run(p, fin, decisions=dec,
                              final_open=dict(fo, decisions_sha1="0" * 12)),
            "под другие решения")

    # --- ПОЛНОТА ФИНАЛЬНОЙ ОЦЕНКИ -----------------------------------------
    cells = [dict(stage="final", protocol_sha1=p["sha1"], replica=rk,
                  task_ids=[t], state_ids=p["splits"]["final"], sigma=0.03,
                  ckpt_sha1=cks[rk]["sha1"])
             for rk in [replica_key(r) for r in p["replicas"]]
             for t in p["tasks"]]
    info = check_final_complete(p, cells, decisions=dec, final_open=fo)
    assert info["n_episodes"] == 4 * 10 * 80, info
    # одна задача без части состояний — отказ
    short = [dict(c) for c in cells]
    short[3] = dict(short[3], state_ids=p["splits"]["final"][:40])
    _expect(lambda: check_final_complete(p, short, decisions=dec,
                                         final_open=fo), "состояний 40 из 80")
    # целой реплики нет — отказ
    three = [c for c in cells if c["replica"] != "d11_rl1"]
    _expect(lambda: check_final_complete(p, three, decisions=dec,
                                         final_open=fo),
            "нельзя считать по трём")
    # состояние посчитано дважды
    dbl = cells + [cells[0]]
    _expect(lambda: check_final_complete(p, dbl, decisions=dec, final_open=fo),
            "посчитано дважды")

    # --- ГЕЙТ: совместный бутстрап, а не среднее границ -------------------
    rng_tasks = list(p["tasks"])
    # РАЗБРОС ПО ЗАДАЧАМ ОБЯЗАТЕЛЕН В ТЕСТЕ: при одинаковых задачах у бутстрапа
    # нулевая дисперсия, граница совпадает с точечной оценкой, и разница между
    # правильным и наивным правилом исчезает — тест проверял бы тождество
    _jit = [0.04, -0.03, 0.02, -0.01, 0.03, -0.02, 0.01, 0.0, -0.04, 0.02]

    def _diffs(by_rep):
        # СДВИГ УЗОРА НА КАЖДУЮ РЕПЛИКУ: при одинаковом разбросе по задачам у
        # всех реплик среднее границ тождественно равно границе среднего, и
        # разница правил снова была бы не видна. Независимые узоры — это и есть
        # то, за счёт чего усреднение реплик снижает дисперсию
        return {replica_key(r): {t: (by_rep[i] + _jit[(j + 3 * i) % len(_jit)],
                                     40.0)
                                 for j, t in enumerate(rng_tasks)}
                for i, r in enumerate(p["replicas"])}
    # список границ больше не принимается: именно так и возникла ошибка
    _expect(lambda: gate(p, [0.01, 0.02, -0.005, 0.03]),
            "границей среднего не является")
    g = gate(p, _diffs([0.05, 0.05, 0.05, 0.05]), n_boot=2000, seed=1)
    assert g["registered"] == "mean" and g["passed"] is True, g
    assert g["lower"]["mean"] > 0 and g["lower"]["all"] > 0, g["lower"]
    # нулевой эффект у всех: граница не выше нуля
    g0 = gate(p, {replica_key(r): {t: (0.0, 40.0) for t in rng_tasks}
                  for r in p["replicas"]}, n_boot=2000, seed=1)
    assert g0["passed"] is False and abs(g0["lower"]["mean"]) < 1e-9, g0
    # mean и mean_rep РАЗЛИЧАЮТСЯ: вторая обобщает на метод и потому шире
    spread = gate(p, _diffs([0.10, 0.02, 0.01, -0.02]), n_boot=4000, seed=2)
    assert spread["lower"]["mean_rep"] < spread["lower"]["mean"], spread["lower"]
    assert spread["lower"]["all"] <= spread["lower"]["mean"], spread["lower"]
    # наивное среднее границ НЕ совпадает с границей среднего и помечено.
    # Направление расхождения тоже содержательно: усреднение снижает
    # дисперсию, поэтому настоящая граница среднего ВЫШЕ среднего границ, и
    # наивное правило занижало бы результат, а не завышало
    assert spread["naive_is_not_a_bound"]
    assert spread["lower"]["mean"] > spread["naive_mean_of_lowers"] + 1e-6, (
        spread["lower"], spread["naive_mean_of_lowers"])
    # разнобой задач между репликами — отказ выравнивания
    bad_al = _diffs([0.05] * 4)
    bad_al["d10_rl0"] = {t: (0.05, 40.0) for t in rng_tasks[:-1]}
    _expect(lambda: gate(p, bad_al, n_boot=1000), "в один столбец")
    _expect(lambda: gate(p, {"d10_rl0": {t: (0.0, 40.0) for t in rng_tasks}},
                         n_boot=1000), "поданы реплики")
    bad_w = _diffs([0.05] * 4)
    bad_w["d10_rl0"][rng_tasks[0]] = (0.05, 0.0)
    _expect(lambda: gate(p, bad_w, n_boot=1000), "не положителен")

    # --- ТО, ЧТО РАНЬШЕ ПРОХОДИЛО МОЛЧА ----------------------------------
    # запись без реплики, без задач, без sigma
    for miss in ("replica", "task_ids", "sigma"):
        rec = dict(base)
        rec.pop(miss)
        _expect(lambda r=rec: check_run(p, r), f"нет обязательного поля "
                                               f"'{miss}'")
    # final на одном состоянии одной задачи больше не «финальный прогон»
    tiny = dict(stage="final", protocol_sha1=p["sha1"], replica="d10_rl0",
                task_ids=[0], state_ids=[p["splits"]["final"][0]], sigma=0.03,
                ckpt_sha1=cks["d10_rl0"]["sha1"], n_episodes_per_task=80)
    _expect(lambda: check_run(p, tiny, decisions=dec, final_open=fo),
            "не на зарегистрированном наборе")
    _expect(lambda: check_run(p, tiny, decisions=dec, final_open=fo),
            "выбор подмножества задач")
    # но как ЯЧЕЙКА она законна
    assert check_run(p, tiny, decisions=dec, final_open=fo, partial=True)
    # final без ckpt_sha1 — отказ на любом уровне
    no_ck = dict(tiny)
    no_ck.pop("ckpt_sha1")
    _expect(lambda: check_run(p, no_ck, decisions=dec, final_open=fo,
                              partial=True), "без ckpt_sha1")
    # final без n_episodes_per_task
    no_n = dict(fin)
    no_n.pop("n_episodes_per_task")
    _expect(lambda: check_run(p, no_n, decisions=dec, final_open=fo),
            "нет n_episodes_per_task")
    # реплика, которой нет в протоколе, и чекпойнт чужой реплики
    _expect(lambda: check_run(p, dict(fin, replica="d19_rl9"), decisions=dec,
                              final_open=fo), "не зарегистрирована")
    _expect(lambda: check_run(p, dict(fin, ckpt_sha1=cks["d11_rl1"]["sha1"]),
                              decisions=dec, final_open=fo),
            "не тот, что запечатан")

    # --- параметры бутстрапа обязаны быть зарегистрированы ----------------
    _expect(lambda: _proto_ok(bootstrap=dict(n_boot=100, alpha=0.05, seed=0)),
            "шумит сильнее")
    _expect(lambda: _proto_ok(bootstrap=dict(n_boot=20000, alpha=0.9, seed=0)),
            "вне (0, 0.5)")
    _expect(lambda: _proto_ok(bootstrap=dict(n_boot=20000, alpha=0.05)),
            "сид бутстрапа")

    print("самопроверка k12b_protocol пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--out", default="data/k12b/protocol.json")
    ap.add_argument("--train-ids", default="0-29")
    ap.add_argument("--dev-ids", default="30-44")
    ap.add_argument("--final-ids", required=False)
    ap.add_argument("--sigma-grid", default="0.03,0.05,0.10")
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episodes-final", type=int, default=80)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--discord-max", type=float, default=0.05)
    ap.add_argument("--p-fail-s0", type=float, default=0.12)
    ap.add_argument("--p-fail-s1", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--halvings", type=int, default=4)
    ap.add_argument("--kl-max", type=float, default=0.02)
    ap.add_argument("--ratio-max", type=float, default=1.5)
    ap.add_argument("--states-json", default="data/k12c_states.json")
    ap.add_argument("--power-json", default="data/k12b/power_hier.json")
    ap.add_argument("--k11g-protocol", default="data/k11g/protocol.json")
    ap.add_argument("--k11e-protocol", default="data/k11e/protocol.json")
    ap.add_argument("--script", action="append", default=[],
                    help="путь к скрипту, чей sha войдёт в протокол")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.init:
        ap.error("нужен --selftest или --init")
    if not args.final_ids:
        ap.error("--final-ids обязателен: набор final регистрируется явно, "
                 "а не выводится из остальных")

    st = json.load(open(args.states_json))
    summ = st.get("summary") or {}
    if "min_fresh" not in summ:
        raise SystemExit(f"{args.states_json} без summary.min_fresh: сначала "
                         f"измерьте состояния (k12c_states_probe.py)")
    states_ref = dict(path=args.states_json, sha1=_sha12(args.states_json),
                      min_fresh=int(summ["min_fresh"]),
                      min_distinct=int(summ.get("min_distinct", -1)),
                      any_silent_clamp=bool(summ.get("any_silent_clamp")))
    pj = json.load(open(args.power_json))
    power_ref = dict(path=args.power_json, sha1=_sha12(args.power_json),
                     power_mean=pj.get("registered", {}).get("power_mean"))
    scripts = {os.path.basename(s): _sha12(s) for s in args.script}
    scripts[os.path.basename(__file__)] = _sha12(os.path.abspath(__file__))

    proto = init_protocol(
        splits={"train": args.train_ids, "dev": args.dev_ids,
                "final": args.final_ids},
        sigma_grid=[float(x) for x in args.sigma_grid.split(",")],
        tasks=[int(x) for x in args.tasks.split(",")],
        n_episodes_final=args.episodes_final,
        delta_target=args.delta, discord_max=args.discord_max,
        p_fail_by_head={"s0": args.p_fail_s0, "s1": args.p_fail_s1},
        replicas=[(0, 0), (0, 1), (1, 0), (1, 1)],
        stop=dict(max_steps=args.max_steps, patience=args.patience,
                  criterion="net_effect_lower_dev"),
        step=dict(full_batch=True, lr=args.lr, head_precision="fp32",
                  train_log_std=False,
                  trust=dict(kl_max=args.kl_max, ratio_max=args.ratio_max),
                  backtrack=dict(max_halvings=args.halvings,
                                 accept="net_effect_dev"),
                  rollback=["params", "optimizer_state"]),
        states_json=states_ref, power_json=power_ref,
        k11g_protocol=dict(path=args.k11g_protocol,
                           sha1=_sha12(args.k11g_protocol)),
        k11e_protocol=dict(path=args.k11e_protocol,
                           sha1=_sha12(args.k11e_protocol)),
        scripts=scripts, notes=args.notes)
    fresh = write_protocol(args.out, proto)
    print(f"{'записан' if fresh else 'уже был'} {args.out}, "
          f"sha1={proto['sha1']}")
    print(f"  наборы: train {len(proto['splits']['train'])}, dev "
          f"{len(proto['splits']['dev'])}, final "
          f"{len(proto['splits']['final'])} состояний")
    print(f"  sigma {proto['sigma_grid']}, эффект {proto['delta_target']}, "
          f"предел потерь {proto['r_loss_max_by_head']}")


if __name__ == "__main__":
    sys.exit(main())
