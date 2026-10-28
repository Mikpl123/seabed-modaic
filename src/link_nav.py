import glob
import os
import re
import numpy as np
import pandas as pd

BASE = r"C:\Users\79211\Desktop\seabed-modaic"
SEQ = 3                                   # номер последовательности Aqualoc
NAV = os.path.join(BASE, "data", "nav")

def read(name, cols):
    df = pd.read_csv(os.path.join(NAV, f"{name}_sequence_{SEQ}.csv"),
                     skipinitialspace=True)
    df.columns = cols
    return df

img = read("img", ["t_ns", "frame_id"])
dep = read("depth", ["t_ns", "depth_m"])
mag = read("mag", ["t_ns", "bx", "by", "bz"])

img["num"] = img.frame_id.str.extract(r"(\d+)").astype(int)
print("В таблице кадров:", len(img), "(номера", img.num.min(), "-", img.num.max(), ")")

img["t_sec"] = (img.t_ns - img.t_ns.iloc[0]) / 1e9
img["depth_m"] = np.interp(img.t_ns, dep.t_ns, dep.depth_m)
bx = np.interp(img.t_ns, mag.t_ns, mag.bx)
by = np.interp(img.t_ns, mag.t_ns, mag.by)
img["heading_deg"] = (np.degrees(np.arctan2(by, bx)) + 360) % 360   # грубо

files = sorted(glob.glob(os.path.join(BASE, "data", "frames", "*.jpg")))
rows = []
for f in files:
    n = int(re.search(r"(\d+)", os.path.basename(f)).group(1))
    rows.append((f, n))
frames = pd.DataFrame(rows, columns=["file", "num"])
out = frames.merge(img[["num", "t_sec", "depth_m", "heading_deg"]], on="num", how="left")
missing = out.t_sec.isna().sum()
print("Кадров в папке:", len(out), ", без данных в таблице (уберём):", missing)
out = out.dropna(subset=["t_sec"])

out.to_csv(os.path.join(BASE, "output", "frames_nav.csv"), index=False)
print("Сохранено кадров с данными:", len(out))
print(out[["num", "t_sec", "depth_m", "heading_deg"]].head())

missing = out.t_sec.isna().sum()
print("Кадров в папке:", len(out), ", без данных в таблице:", missing)
if missing:
    print("Номера в папке и в таблице не совпали: проверь номер последовательности SEQ.")

out.to_csv(os.path.join(BASE, "output", "frames_nav.csv"), index=False)
print(out.head())