import cv2
import glob
import os
import re

BASE = r"C:\Users\79211\Desktop\seabed-mosaic"
SRC = os.path.join(BASE, "dataset")
DST = os.path.join(BASE, "data", "frames")
STEP = 4
EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

if not os.path.isdir(SRC):
    raise SystemExit(f"Папки нет: {SRC}")

files = []
for root, _, names in os.walk(SRC):
    for n in names:
        if n.lower().endswith(EXT):
            files.append(os.path.join(root, n))
files.sort(key=natural_key)

print("Найдено кадров:", len(files))
if not files:
    print("В папке лежит (первые 20):", os.listdir(SRC)[:20])
    raise SystemExit("Кадров не нашли, старые кадры не тронуты.")
print("Первый:", files[0])
print("Последний:", files[-1])

os.makedirs(DST, exist_ok=True)
for old in glob.glob(os.path.join(DST, "*.jpg")):
    os.remove(old)

saved = 0
for i, path in enumerate(files):
    if i % STEP:
        continue
    img = cv2.imread(path)
    if img is None:
        print("Не открылся:", path)
        continue
    cv2.imwrite(os.path.join(DST, f"frame_{i:06d}.jpg"), img)
    saved += 1
print("Скопировано:", saved)