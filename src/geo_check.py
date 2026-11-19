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


def best_fit(xyz, Y):
    """Перебирает пары осей COLMAP (с зеркалом и без) и возвращает лучшую подгонку к положениям на карте."""
    best = None
    for name, (a, b) in {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items():
        for mirror in (False, True):
            X = xyz[:, [a, b]].copy()
            if mirror:
                X[:, 0] = -X[:, 0]
            z, c, r = fit_similarity(X, Y)
            rms = float(np.sqrt(np.mean(r ** 2)))
            if best is None or rms < best[0]:
                best = (rms, name, mirror, z, c, r, X)
    return best


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
    if all("scale" in p for p in P):
        sc = np.array([p["scale"] for p in P])[ok]
        zc = np.interp(fr, T[:, 0], T[:, 3])
        if np.std(sc) > 1e-6 and np.std(zc) > 1e-9:
            rr = float(np.corrcoef(np.log(sc), zc)[0, 1])
            print(f"Масштаб кадров у нас и высота камеры COLMAP (z): корреляция {rr:+.2f}, масштаб от {sc.min():.2f} до {sc.max():.2f}. "
                  f"{'Масштаб следует за высотой: его нельзя держать равным 1.' if abs(rr) > 0.6 else 'Связь слабая.'}")
    worst = np.argsort(-r)[:8]
    print("Кадры, где цепочка ORB дальше всего от траектории (там дрейф):")
    for i in sorted(worst):
        print(f"  кадр {int(fr[i])}: расхождение {r[i]:.0f} px")


def loop_report(poses_path, loops_path=LOOPS, traj_path=TRAJ, label=""):
    """Проверка петель. Кадры, которые COLMAP считает похожими, смотрят на одно место, но центры у них не совпадают
    (достаточно общего перекрытия). Поэтому сравнивать надо не с нулём, а с тем, что даёт сама траектория COLMAP
    для тех же пар. Печатаем: расстояние между кадрами пары у нас, у COLMAP, и разницу векторов (это и есть ошибка петли)."""
    try:
        L = np.loadtxt(loops_path, delimiter=",")
        T = np.loadtxt(traj_path)
    except (ValueError, OSError) as e:
        print(f"Матрица петель или траектория не прочитаны ({str(e)[:60]}).")
        return
    P = json.load(open(poses_path, encoding="utf-8"))
    fr = np.array([p["frame"] for p in P], float)
    Y = np.array([[p["cx"], p["cy"]] for p in P])
    ok = (fr >= T[0, 0]) & (fr <= T[-1, 0])
    xyz = np.c_[[np.interp(fr[ok], T[:, 0], T[:, k]) for k in (1, 2, 3)]].T
    _, _, _, z, c, _, X = best_fit(xyz, Y[ok])
    Z = z * (X[:, 0] + 1j * X[:, 1]) + c
    Pc = np.full((len(fr), 2), np.nan)
    Pc[ok] = np.c_[Z.real, Z.imag]                           # где кадры должны лежать по COLMAP (в пикселях карты)
    seen, ours, colm, err = set(), [], [], []
    for i, j in zip(*np.where(np.triu(L, 25) > 0)):
        a, b = int(np.argmin(np.abs(fr - i * LOOP_STEP))), int(np.argmin(np.abs(fr - j * LOOP_STEP)))
        if a == b or (a, b) in seen or abs(fr[a] - i * LOOP_STEP) > 40 or abs(fr[b] - j * LOOP_STEP) > 40:
            continue
        if np.isnan(Pc[a]).any() or np.isnan(Pc[b]).any():
            continue
        seen.add((a, b))
        ours.append(float(np.linalg.norm(Y[a] - Y[b])))
        colm.append(float(np.linalg.norm(Pc[a] - Pc[b])))
        err.append(float(np.linalg.norm((Y[a] - Y[b]) - (Pc[a] - Pc[b]))))
    if not ours:
        print("Пар для проверки петель не нашлось.")
        return
    ours, colm, err = np.array(ours), np.array(colm), np.array(err)
    name = label or os.path.basename(poses_path)
    print(f"Петли ({name}), пар {len(ours)} (кадр шириной 480 px; расстояния в px рабочего размера):")
    print(f"    у нас:    медиана {np.median(ours):.0f}, 90% ниже {np.percentile(ours, 90):.0f}, ближе 100 px: {100 * np.mean(ours < 100):.0f}%")
    print(f"    у COLMAP: медиана {np.median(colm):.0f}, 90% ниже {np.percentile(colm, 90):.0f}, ближе 100 px: {100 * np.mean(colm < 100):.0f}%   <- так выглядит идеально замкнутая петля")
    print(f"    ошибка петли (разница векторов пары у нас и у COLMAP): медиана {np.median(err):.0f} px, 90% ниже {np.percentile(err, 90):.0f} px")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args[:2])
    tp = args[0] if args else TRAJ
    loop_report(args[1] if len(args) > 1 else POSES, traj_path=tp)
    ref = os.path.join(os.path.dirname(POSES), "poses_refined_part0.json")
    if os.path.exists(ref):                                                  # положения после уточнения по изображению
        loop_report(ref, traj_path=tp, label="после уточнения")