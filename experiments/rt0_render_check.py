#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboTwin, этап 0: рендер SAPIEN в этом окружении (без кода RoboTwin).

Сцена: пол, красный куб, свет, камера 320×240. Шаг физики, рендер, снимок
цвета и глубины; проверки: снимок не пустой (есть и куб, и фон — красный
канал различается), глубина конечна; затем время N снимков (кадр/с).
Работает с SAPIEN 3 и (запасной путь) SAPIEN 2. Запускать в окружении
RoboTwin (conda env с sapien). Отчёт: reports/rt0/render_check.json, кадр —
reports/rt0/render_check.png (если есть imageio или PIL). Код 3 при отказе.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def build_scene(sapien, device):
    v = getattr(sapien, "__version__", "?")
    major = 3 if str(v)[:1] not in ("1", "2") else int(str(v)[0])
    if major < 3 and not hasattr(sapien, "Engine"):
        import sapien.core as sapien          # SAPIEN 2: API в sapien.core
    if major >= 3:
        scene = sapien.Scene()
        scene.set_timestep(1 / 250)
        scene.add_ground(0)
        b = scene.create_actor_builder()
        b.add_box_collision(half_size=[0.1, 0.1, 0.1])
        b.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[1, 0, 0])
        box = b.build(name="box")
        box.set_pose(sapien.Pose([0, 0, 0.1]))
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        cam = scene.add_camera(name="cam", width=320, height=240, fovy=1.0,
                               near=0.01, far=10.0)
        cam.set_pose(sapien.Pose([-0.8, 0, 0.4], [0.9659, 0, 0.2588, 0]))

        def shot():
            scene.step()
            scene.update_render()
            cam.take_picture()
            rgba = np.asarray(cam.get_picture("Color"))
            pos = np.asarray(cam.get_picture("Position"))
            return rgba, pos[..., 2]
        return major, v, shot
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(offscreen_only=True, device=device or None)
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[0.1, 0.1, 0.1])
    b.add_box_visual(half_size=[0.1, 0.1, 0.1], color=[1, 0, 0])
    box = b.build(name="box")
    box.set_pose(sapien.Pose([0, 0, 0.1]))
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    cam = scene.add_camera(name="cam", width=320, height=240, fovy=1.0,
                           near=0.01, far=10.0)
    cam.set_pose(sapien.Pose([-0.8, 0, 0.4], [0.9659, 0, 0.2588, 0]))

    def shot():
        scene.step()
        scene.update_render()
        cam.take_picture()
        rgba = np.asarray(cam.get_float_texture("Color"))
        pos = np.asarray(cam.get_float_texture("Position"))
        return rgba, pos[..., 2]
    return major, v, shot


def render_devices(sapien):
    """Перечень устройств рендера SAPIEN (строка) или None, если API нет:
    отличает GPU NVIDIA от программного рендера (llvmpipe/lavapipe)."""
    for mod in (getattr(sapien, "render", None), sapien):
        fn = getattr(mod, "get_device_summary", None) if mod else None
        if fn is not None:
            try:
                return str(fn())
            except Exception as e:  # noqa: BLE001
                return f"ошибка get_device_summary: {e}"
    return None


def checks(rgba, depth):
    rgb = np.asarray(rgba, np.float64)[..., :3]
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    red = rgb[..., 0] - rgb[..., 1:].mean(-1)
    prob = []
    if not np.isfinite(rgb).all():
        prob.append("цвет не конечен")
    if rgb.std() < 1e-3:
        prob.append("кадр однотонный (рендер пустой)")
    if not (red > 0.3).any():
        prob.append("красного куба в кадре нет")
    if not np.isfinite(depth).all():
        prob.append("глубина не конечна")
    return prob, dict(mean=float(rgb.mean()), std=float(rgb.std()),
                      red_pixels=int((red > 0.3).sum()))


def save_png(rgba, path):
    img = np.asarray(rgba)[..., :3]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    try:
        import imageio.v2 as iio
        iio.imwrite(path, img)
        return True
    except Exception:  # noqa: BLE001
        try:
            from PIL import Image
            Image.fromarray(img).save(path)
            return True
        except Exception:  # noqa: BLE001
            return False


def selftest():
    h, w = 24, 32
    img = np.zeros((h, w, 4), np.float32)
    img[..., :3] = 0.4
    img[8:16, 10:20, 0] = 0.9
    img[8:16, 10:20, 1:3] = 0.1
    p, s = checks(img, np.ones((h, w)))
    assert p == [] and s["red_pixels"] == 80, (p, s)
    p, _ = checks(np.zeros((h, w, 4)), np.ones((h, w)))
    assert "кадр однотонный (рендер пустой)" in p
    p, _ = checks(img, np.full((h, w), np.nan))
    assert "глубина не конечна" in p
    print("самопроверка rt0_render_check пройдена: проверки кадра")
    return 0


def main():
    ap = argparse.ArgumentParser(description="RoboTwin этап 0: рендер SAPIEN")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="", help="напр. cuda:0 (SAPIEN 2)")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    os.makedirs("reports/rt0", exist_ok=True)
    rep = dict(python=sys.version.split()[0])
    try:
        import sapien
        major, ver, shot = build_scene(sapien, a.device)
        rep.update(sapien_version=str(ver), api_major=major)
        dev = render_devices(sapien)
        rep["render_devices"] = dev
        if dev is not None and "nvidia" not in dev.lower():
            raise RuntimeError(f"рендер не на NVIDIA (устройства: {dev})")
        t0 = time.perf_counter()
        rgba, depth = shot()
        rep["first_frame_s"] = time.perf_counter() - t0
        prob, stat = checks(rgba, depth)
        rep.update(frame=stat, problems=prob)
        rep["png_saved"] = save_png(rgba, "reports/rt0/render_check.png")
        t0 = time.perf_counter()
        for _ in range(a.n):
            shot()
        dt = time.perf_counter() - t0
        rep["fps_step_render_320x240"] = a.n / dt
    except Exception as e:  # noqa: BLE001
        rep["problems"] = [f"{type(e).__name__}: {e}"]
    rep["ok"] = not rep["problems"]
    json.dump(rep, open("reports/rt0/render_check.json", "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"  устройства рендера: {rep.get('render_devices')}")
    print("  " + ("рендер SAPIEN работает" if rep["ok"] else
                  "ОТКАЗ: " + "; ".join(rep["problems"])))
    return 0 if rep["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
