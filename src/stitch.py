import numpy as np
import cv2
import glob
import os
import time
import json

BASE = r"C:\Users\79211\Desktop\seabed-mosaic"
out_dir = os.path.join(BASE, "output")
os.makedirs(out_dir, exist_ok=True)

def save_map(mosaic, name, out_dir, meta=None):
    """PNG с прозрачным фоном + JSON-паспорт."""
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    alpha = (gray > 0).astype(np.uint8) * 255
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    bgra = cv2.cvtColor(mosaic, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    png_path = os.path.join(out_dir, name + ".png")
    cv2.imwrite(png_path, bgra)
    info = {"file": name + ".png", "width": int(mosaic.shape[1]),
            "height": int(mosaic.shape[0]), "crs": None, "bounds": None,
            "meters_per_pixel": None}
    if meta:
        info.update(meta)
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return png_path

START = 0          # с какого по порядку кадра в папке мы будет взять
COUNT = 50         # сколько кадров взять
STRIDE = 3         # брать каждый кадр
WIDTH = 480        # рабочая ширина кадра
LIT_THRESHOLD = 0.6   # больше число, значит мы обрезаем еще сильнее
USE_CLAHE = False  # усиление контраста (на Aqualoc не нужно)
CROP_DARK = True     # обрезать тёмные края кадров
FLAT_FIELD = False   # короче убирает затемнение по краям

paths = sorted(glob.glob(os.path.join(BASE, "data", "frames", "*.jpg")))
print("Кадров в папке:", len(paths))
paths = paths[START:START + COUNT * STRIDE:STRIDE]

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
images = []
for p in paths:
    img = cv2.imread(p)
    if img is None:
        continue
    k = WIDTH / img.shape[1]
    img = cv2.resize(img, None, fx=k, fy=k)
    if USE_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    images.append(img)

if len(images) < 2:
    raise SystemExit("Мало кадров: проверь data\\frames")

if CROP_DARK or FLAT_FIELD:
    stack = np.stack([im.astype(np.float32) for im in images])
    flat = cv2.GaussianBlur(stack.mean(axis=0), (0, 0), 25)
    flat = np.maximum(flat, 10)
    light = flat.mean(axis=2)
    cv2.imwrite(os.path.join(out_dir, "light_profile.png"),
                (255 * light / light.max()).astype(np.uint8))

    if CROP_DARK:
        lit = light > LIT_THRESHOLD * np.percentile(light, 90)
        ys, xs = np.where(lit)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        print(f"Освещённая часть кадра: x {x0}-{x1} из {light.shape[1]}, "
              f"y {y0}-{y1} из {light.shape[0]}")
        images = [im[y0:y1, x0:x1] for im in images]
        flat = flat[y0:y1, x0:x1]

    if FLAT_FIELD:
        gain = np.clip(flat.mean() / flat, 0.6, 2.5)
        images = [np.clip(im.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                  for im in images]

print("Кадров для сшивки:", len(images))

start = time.time()
stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
status, mosaic = stitcher.stitch(images)
print(f"Сшивка заняла {time.time() - start:.1f} сек, код {int(status)} (0 = успех)")

if int(status) == 0 and mosaic is not None:
    used = len(stitcher.component())
    print("Использовано кадров:", used, "из", len(images))
    name = f"aqualoc_s{START}_c{COUNT}_k{STRIDE}"
    png = save_map(mosaic, name, out_dir, {
        "source": "Aqualoc", "start": START, "count": COUNT, "stride": STRIDE,
        "frames_used": used, "frames_taken": len(images)})
    print("Размер мозаики:", mosaic.shape)
    print("Сохранено:", png)
else:
    print("Сшивка не удалась")

    #   Коды: 1 мало кадров
    #   2нет общих точек
    #   3 сбой подгонки