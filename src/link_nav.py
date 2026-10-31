import glob
import os
import re
import numpy as np
import pandas as pd

# Папка проекта = на уровень выше папки src, где лежит этот скрипт
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = 3                                        # номер последовательности Aqualoc
NAV = os.path.join(BASE, "data", "nav")        # CSV с навигацией
FRAMES_DIR = os.path.join(BASE, "dataset")     # та же папка, что SRC в mosaic_orb.py
OUT = os.path.join(BASE, "output")


def read(name, cols):
    path = os.path.join(NAV, f"{name}_sequence_{SEQ}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Не найден файл: {path}")
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = cols
    return df


# ---------- 1. таблицы навигации ----------
img = read("img", ["t_ns", "frame_id"])
dep = read("depth", ["t_ns", "depth_m"])
mag = read("mag", ["t_ns", "bx", "by", "bz"])

img["num"] = img.frame_id.astype(str).str.extract(r"(\d+)")[0].astype(int)
print("В таблице кадров:", len(img), "(номера", img.num.min(), "-", img.num.max(), ")")

img["t_sec"] = (img.t_ns - img.t_ns.iloc[0]) / 1e9
img["depth_m"] = np.interp(img.t_ns, dep.t_ns, dep.depth_m)
bx = np.interp(img.t_ns, mag.t_ns, mag.bx)
by = np.interp(img.t_ns, mag.t_ns, mag.by)
# ВНИМАНИЕ: курс по магнитометру почти не меняется и ненадёжен, только для справки
img["heading_deg"] = (np.degrees(np.arctan2(by, bx)) + 360) % 360

# ---------- 2. кадры из папки ----------
files = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
    files += glob.glob(os.path.join(FRAMES_DIR, ext))
files = sorted(files)

print("Ищу кадры в:", FRAMES_DIR)
print("Найдено файлов:", len(files))
if not files:
    raise SystemExit("Кадров нет. Проверь папку FRAMES_DIR и расширение файлов.")
print("Пример имени:", os.path.basename(files[0]))

# номер кадра = последнее число в имени (так же, как fnum() в mosaic_orb.py)
frames = pd.DataFrame(
    [(f, int(re.findall(r"\d+", os.path.basename(f))[-1])) for f in files],
    columns=["file", "num"])

# ---------- 3. объединение ----------
out = frames.merge(img[["num", "t_sec", "depth_m", "heading_deg"]],
                   on="num", how="left")
missing = int(out.t_sec.isna().sum())
print("Кадров в папке:", len(out), ", без данных в таблице:", missing)
if missing:
    print("Часть номеров не совпала: проверь SEQ и формат имён кадров.")
out = out.dropna(subset=["t_sec"])

os.makedirs(OUT, exist_ok=True)
out.to_csv(os.path.join(OUT, "frames_nav.csv"), index=False)
print("Сохранено кадров с данными:", len(out))
print(out[["num", "t_sec", "depth_m", "heading_deg"]].head())