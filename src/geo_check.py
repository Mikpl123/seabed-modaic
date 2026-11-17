"""Проверка: совпадает ли траектория COLMAP с положениями кадров, которые посчитал mosaic_orb.py.

Что делает: берёт координаты камеры из файла COLMAP (кадр, x, y, z, кватернион), берёт центры кадров на карте
(poses_part0.json от mosaic_orb.py), перебирает пары осей (xy, xz, yz), для каждой подгоняет сдвиг + поворот + масштаб
(с зеркалом и без) и печатает, какая пара осей ложится лучше. Так видно: (1) тот ли это файл траектории,
(2) в какой плоскости лежит дно, (3) сколько пикселей карты приходится на единицу COLMAP,
(4) на каких кадрах цепочка ORB уходит от траектории (там дрейф)."""
import json
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(BASE, "colmap", "colmap_traj_sequence_3.txt")      # поменяйте на нужный файл траектории
POSES = os.path.join(BASE, "output", "poses_part0.json")                # сохраняет mosaic_orb.py
LOOPS = os.path.join(BASE, "colmap", "colmap_detected_loop_sequence_3.txt")   # матрица петель (0/1 через запятую)
LOOP_STEP = 20                                                           # кадров на одну ячейку матрицы


def fit_similarity(X, Y):
    """Y ~ z*X + c (комплексные числа): поворот, масштаб, сдвиг по методу наименьших квадратов."""
    x, y = X[:, 0] + 1j * X[:, 1], Y[:, 0] + 1j * Y[:, 1]
    x0, y0 = x - x.mean(), y - y.mean()
    z = np.sum(np.conj(x0) * y0) / np.sum(np.abs(x0) ** 2)
    c = y.mean() - z * x.mean()
    r = np.abs(z * x + c - y)
    return z, c, r


def main(traj_path=TRAJ, poses_path=POSES):
    T = np.loadtxt(traj_path)
    P = json.load(open(poses_path, encoding="utf-8"))
    fr = np.array([p["frame"] for p in P], float)
    Y = np.array([[p["cx"], p["cy"]] for p in P])
    ok = (fr >= T[0, 0]) & (fr <= T[-1, 0])
    fr, Y = fr[ok], Y[ok]
    xyz = np.c_[[np.interp(fr, T[:, 0], T[:, i]) for i in (1, 2, 3)]].T
    print(f"Траектория: {os.path.basename(traj_path)}, кадров в ней {len(T)} (до №{int(T[-1, 0])}); "
          f"ключевых кадров для сравнения: {len(fr)}")
    span = float(np.ptp(Y, axis=0).max())
    best = None
    for name, (a, b) in {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items():
        for mirror in (False, True):
            X = xyz[:, [a, b]].copy()
            if mirror:
                X[:, 0] = -X[:, 0]
            z, c, r = fit_similarity(X, Y)
            rms = float(np.sqrt(np.mean(r ** 2)))
            print(f"  оси {name}{' (зеркало)' if mirror else ''}: невязка {rms:7.1f} px "
                  f"({100 * rms / span:4.1f}% размера карты), масштаб {abs(z):8.1f} px на единицу COLMAP")
            if best is None or rms < best[0]:
                best = (rms, name, mirror, z, r)
    rms, name, mirror, z, r = best
    print(f"Лучшее: оси {name}{', зеркало' if mirror else ''}, невязка {rms:.1f} px.")
    if rms > 0.1 * span:
        print("Невязка больше 10% размера карты: либо не тот файл траектории, либо не та последовательность, "
              "либо дрейф цепочки ORB очень большой. Привязывать картинку к этим координатам пока нельзя.")
    worst = np.argsort(-r)[:8]
    print("Кадры, где цепочка ORB дальше всего от траектории (там дрейф):")
    for i in sorted(worst):
        print(f"  кадр {int(fr[i])}: расхождение {r[i]:.0f} px")


def loop_report(poses_path, loops_path=LOOPS, label=""):
    """Главная проверка петель: COLMAP говорит, что кадры i и j смотрят на одно место. Если петля замкнута,
    на нашей карте эти кадры лежат рядом. Печатаем расстояние между ними (px рабочего размера; кадр шириной 480 px)."""
    try:
        L = np.loadtxt(loops_path, delimiter=",")
    except (ValueError, OSError) as e:
        print(f"Матрица петель не прочитана ({str(e)[:60]}).")
        return
    P = json.load(open(poses_path, encoding="utf-8"))
    fr = np.array([p["frame"] for p in P], float)
    Y = np.array([[p["cx"], p["cy"]] for p in P])
    seen, d = set(), []
    for i, j in zip(*np.where(np.triu(L, 25) > 0)):
        a, b = int(np.argmin(np.abs(fr - i * LOOP_STEP))), int(np.argmin(np.abs(fr - j * LOOP_STEP)))
        if a != b and (a, b) not in seen and abs(fr[a] - i * LOOP_STEP) <= 40 and abs(fr[b] - j * LOOP_STEP) <= 40:
            seen.add((a, b))
            d.append(float(np.linalg.norm(Y[a] - Y[b])))
    d = np.array(d)
    if not len(d):
        print("Пар для проверки петель не нашлось.")
        return
    print(f"Петли ({label or os.path.basename(poses_path)}): пар {len(d)}; расстояние между кадрами одного места на карте: "
          f"медиана {np.median(d):.0f} px, 90% ниже {np.percentile(d, 90):.0f} px; "
          f"ближе 100 px: {100 * np.mean(d < 100):.0f}% пар (замкнуто, если почти все)")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args[:2])
    loop_report(args[1] if len(args) > 1 else POSES)
    ref = os.path.join(os.path.dirname(POSES), "poses_refined_part0.json")
    if os.path.exists(ref):                                                  # положения после уточнения по изображению
        loop_report(ref, label="после уточнения")