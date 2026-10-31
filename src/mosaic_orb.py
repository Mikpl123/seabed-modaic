import cv2
import glob
import json
import os
import re
import time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = os.path.join(BASE, "dataset")        # папка со всеми кадрами
OUT = os.path.join(BASE, "output")
START, END = 0, None                       # с какого по порядку кадра в папке мы будет взять (END= None - до конца)
WIDTH = 480                                # рабочая ширина кадра
MIN_SHIFT = 0.15                           # новый кадр берём, когда сдвиг >= 20% меньшей стороны
CHECK_EVERY = 2                            # проверять каждый 2-й кадр
NEIGHBORS = 5                              # с сколькими прошлыми ключевыми кадрами сопоставлять
MIN_INLIERS = 25                           # минимум надёжных совпадений для связи
MAX_SIDE = 4000                            # максимальный размер карты в пикселях
LIGHT_POWER = 2.0                          # насколько сильно тёмные края теряют вес при смешивании
DIST_POWER = 24.0                           # чем больше, тем меньше двоения (побеждает кадр, где точка ближе к центру)
MAX_PARTS = 3                              # сколько самых больших кусков сохранять
FLAT_FIELD = True                          # выравнивание света
SCALE_PRIOR = 1e5                          # держать масштаб кадров близким к 1 (0 = выключить если срочно :D высота аппарата сильно меняется)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def fnum(path):
    """Номер кадра из имени файла (frame000123.png -> 123)."""
    m = re.findall(r"\d+", os.path.basename(path))
    return int(m[-1]) if m else -1

def load(path):
    img = cv2.imread(path)
    k = WIDTH / img.shape[1]
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)

def hp_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 15)          # убираем плавный свет

# отбор кадров (New??????) надо как-то отдельно создать функцию в отедльном файле???? для скорости
def select_keyframes(paths):
    last = hp_gray(load(paths[0]))
    win = cv2.createHanningWindow((last.shape[1], last.shape[0]), cv2.CV_32F)
    keep, since = [0], 0
    for i in range(1, len(paths), CHECK_EVERY):
        cur = hp_gray(load(paths[i]))
        (dx, dy), resp = cv2.phaseCorrelate(last, cur, win)
        shift = float(np.hypot(dx, dy)) / min(last.shape)
        since += CHECK_EVERY
        if shift >= MIN_SHIFT or (resp < 0.05 and since >= 10):
            keep.append(i)
            last, since = cur, 0
    return keep

# зацепки ORB и связи между кадрами (New!!)
clahe = cv2.createCLAHE(2.0, (8, 8))
orb = cv2.ORB_create(4000, fastThreshold=5)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def features(img, k=12, gain=25):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(g, (0, 0), k)                                   # плавный фон (муть)
    std = np.sqrt(cv2.GaussianBlur((g - mean) ** 2, (0, 0), k)) + 2         # местный контраст
    g = np.clip(128 + gain * (g - mean) / std, 0, 255).astype(np.uint8)     # выравниваем контраст
    kp, des = orb.detectAndCompute(g, None)
    pts = np.float32([x.pt for x in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, des

def match(fa, fb):
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return None
    pairs = [p for p in bf.knnMatch(da, db, k=2) if len(p) == 2]
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    A = pa[[m.queryIdx for m in good]]
    B = pb[[m.trainIdx for m in good]]
    M, inl = cv2.estimateAffinePartial2D(B, A, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < MIN_INLIERS:
        return None
    return A[inl], B[inl]

def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for j, k, _, _ in edges:
        parent[find(j)] = find(k)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)

def solve(nodes, edges, size):
    """Подбираем положение каждого кадра (поворот, масштаб, сдвиг) так,
    чтобы одни и те же камни во всех кадрах легли в одно место."""
    pos = {n: i for i, n in enumerate(nodes)}
    m = len(nodes)
    cx, cy = size[0] / 2, size[1] / 2
    H = np.zeros((4 * m, 4 * m))
    rhs = np.zeros(4 * m)
    rng = np.random.default_rng(0)
    used = []
    for j, k, A, B in edges:
        if j not in pos or k not in pos:
            continue
        if len(A) > 80:
            sel = rng.choice(len(A), 80, replace=False)
            A, B = A[sel], B[sel]
        u, v = A - (cx, cy), B - (cx, cy)
        J = np.zeros((2 * len(u), 8))
        J[0::2] = np.c_[u[:, 0], -u[:, 1], np.ones(len(u)), np.zeros(len(u)),
                        -v[:, 0], v[:, 1], -np.ones(len(u)), np.zeros(len(u))]
        J[1::2] = np.c_[u[:, 1], u[:, 0], np.zeros(len(u)), np.ones(len(u)),
                        -v[:, 1], -v[:, 0], np.zeros(len(u)), -np.ones(len(u))]
        ids = np.r_[4 * pos[j]:4 * pos[j] + 4, 4 * pos[k]:4 * pos[k] + 4]
        H[np.ix_(ids, ids)] += J.T @ J
        used.append((ids, J))
    W = 1e6                                  # первый кадр закреплён
    H[:4, :4] += W * np.eye(4)
    rhs[:4] += W * np.array([1.0, 0, 0, 0])
    p = np.linalg.solve(H, rhs)
    # второй проход: мягко возвращаем масштаб кадров к 1 (высота аппарата почти не меняется)
    PRIOR = SCALE_PRIOR
    H2, rhs2 = H.copy(), rhs.copy()
    for i in range(m if SCALE_PRIOR > 0 else 0):
        a, b = p[4 * i], p[4 * i + 1]
        s = max(float(np.hypot(a, b)), 1e-6)
        for col, target in ((4 * i, a / s), (4 * i + 1, b / s)):
            H2[col, col] += PRIOR
            rhs2[col] += PRIOR * target
    p = np.linalg.solve(H2, rhs2)
    sq, cnt = 0.0, 0
    for ids, J in used:
        r = J @ p[ids]
        sq += float((r ** 2).sum())
        cnt += len(r)
    rms = (sq / max(cnt, 1)) ** 0.5
    poses = []
    for i in range(m):
        a, b, tx, ty = p[4 * i:4 * i + 4]
        poses.append(np.array([[a, -b, tx - (a * cx - b * cy)],
                               [b, a, ty - (b * cx + a * cy)]]))
    return poses, rms

def weight_map(h, w, L):
    m = np.zeros((h, w), np.uint8)
    m[2:-2, 2:-2] = 255
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    d = d / d.max()
    return (d ** DIST_POWER) * (L ** LIGHT_POWER)

# ---------- отбраковка связей, которые спорят с остальными ----------
PRUNE_FACTOR = 2.5                         # выбрасываем, если невязка в столько раз больше медианной
PRUNE_MIN_PX = 5.0                         # ...и больше этого числа пикселей

def edge_rms(e, poses, pos):
    j, k, A, B = e
    Mj, Mk = poses[pos[j]], poses[pos[k]]
    r = np.linalg.norm((A @ Mj[:, :2].T + Mj[:, 2]) - (B @ Mk[:, :2].T + Mk[:, 2]), axis=1)
    return float(np.sqrt(np.mean(r ** 2)))

def prune_links(best, edges, size, rounds=3):
    """Убираем связи, которые сильно не согласуются с остальными (но не рвём цепочку)."""
    bset = set(best)
    edges = [e for e in edges if e[0] in bset and e[1] in bset]
    pos = {n: i for i, n in enumerate(best)}
    removed = 0
    for _ in range(rounds):
        poses, rms = solve(best, edges, size)
        er = [edge_rms(e, poses, pos) for e in edges]
        limit = max(PRUNE_MIN_PX, PRUNE_FACTOR * float(np.median(er)))
        order = sorted(range(len(edges)), key=lambda i: -er[i])
        drop = set()
        for i in order:
            if er[i] <= limit:
                break
            rest = [e for t, e in enumerate(edges) if t not in drop and t != i]
            comp0 = components(max(best) + 1, rest)[0]
            if bset <= set(comp0):
                drop.add(i)                                      # связность сохраняется
        if not drop:
            break
        edges = [e for t, e in enumerate(edges) if t not in drop]
        removed += len(drop)
    print(f"Отбраковано связей: {removed} (осталось {len(edges)})")
    return edges
def render(paths, poses):
    h, w = load(paths[0]).shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    allp = np.concatenate([cv2.transform(corners[None], M)[0] for M in poses])
    minx, miny = allp.min(0)
    maxx, maxy = allp.max(0)
    sc = min(1.0, MAX_SIDE / max(maxx - minx, maxy - miny))
    Wc, Hc = int(np.ceil((maxx - minx) * sc)) + 1, int(np.ceil((maxy - miny) * sc)) + 1
    print(f"Размер карты: {Wc} x {Hc}, масштаб {sc:.2f}")

    sample = [load(paths[i]) for i in np.linspace(0, len(paths) - 1, min(60, len(paths))).astype(int)]
    L = cv2.GaussianBlur(np.mean([cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) for s in sample], axis=0), (0, 0), 25)
    L = np.maximum(L / L.max(), 0.05)
    gain = np.clip(L.mean() / L, 0.6, 2.5) if FLAT_FIELD else np.ones_like(L)
    wmap = weight_map(h, w, L).astype(np.float32)

    acc = np.zeros((Hc, Wc, 3), np.float32)
    wsum = np.zeros((Hc, Wc), np.float32)
    for p, M in zip(paths, poses):
        Mc = np.zeros((2, 3))
        Mc[:, :2] = sc * M[:, :2]
        Mc[:, 2] = sc * (M[:, 2] - (minx, miny))
        pts = cv2.transform(corners[None], Mc)[0]
        x0, x1 = max(int(np.floor(pts[:, 0].min())), 0), min(int(np.ceil(pts[:, 0].max())) + 1, Wc)
        y0, y1 = max(int(np.floor(pts[:, 1].min())), 0), min(int(np.ceil(pts[:, 1].max())) + 1, Hc)
        if x1 <= x0 or y1 <= y0:
            continue
        Ml = Mc.copy()
        Ml[0, 2] -= x0
        Ml[1, 2] -= y0
        img = load(p).astype(np.float32) * gain[..., None]
        wi = cv2.warpAffine(img, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        ww = cv2.warpAffine(wmap, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        acc[y0:y1, x0:x1] += wi * ww[..., None]
        wsum[y0:y1, x0:x1] += ww
    valid = wsum > 1e-3
    mosaic = acc / np.maximum(wsum, 1e-6)[..., None]
    mosaic[~valid] = 0
    return np.clip(mosaic, 0, 255).astype(np.uint8), valid

def save_map(mosaic, valid, name, meta):
    bgra = cv2.cvtColor(mosaic, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = valid.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT, name + ".png"), bgra)
    info = {"file": name + ".png", "width": int(mosaic.shape[1]), "height": int(mosaic.shape[0]),
            "crs": None, "bounds": None, "meters_per_pixel": None}
    info.update(meta)
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

# ---------- проверка связей (для тестировщика) ----------
CHECK_AT = 5200                            # номер кадра, около которого цепочка раньше рвалась

def draw_link(path_a, path_b, A, B, out_path):
    """Картинка: два кадра рядом и линии между совпавшими точками."""
    ia, ib = load(path_a), load(path_b)
    canvas = np.hstack([ia, ib])
    off = ia.shape[1]
    for (x1, y1), (x2, y2) in list(zip(A, B))[:60]:
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + off, int(y2)), (0, 255, 0), 1)
        cv2.circle(canvas, (int(x1), int(y1)), 3, (0, 0, 255), -1)
        cv2.circle(canvas, (int(x2) + off, int(y2)), 3, (0, 0, 255), -1)
    cv2.imwrite(out_path, canvas)

def diagnose(best, edges, poses, kpaths, top=6):
    pos = {n: i for i, n in enumerate(best)}
    rows = []
    for j, k, A, B in edges:
        if j not in pos or k not in pos:
            continue
        Mj, Mk = poses[pos[j]], poses[pos[k]]
        gj = A @ Mj[:, :2].T + Mj[:, 2]
        gk = B @ Mk[:, :2].T + Mk[:, 2]
        r = np.linalg.norm(gj - gk, axis=1)
        rows.append((float(np.sqrt(np.mean(r ** 2))), len(A), fnum(kpaths[j]), fnum(kpaths[k]), j, k, A, B))
    print("Самые слабые связи (мало совпадений):")
    for rms, n, a, b, *_ in sorted(rows, key=lambda x: x[1])[:top]:
        print(f"  кадры {a} -> {b}: совпадений {n}, невязка {rms:.1f} px")
    print("Самые большие невязки:")
    for rms, n, a, b, *_ in sorted(rows, key=lambda x: -x[0])[:top]:
        print(f"  кадры {a} -> {b}: совпадений {n}, невязка {rms:.1f} px")
    span = [r for r in rows if r[2] < CHECK_AT <= r[3]]
    print(f"Связи, перекрывающие кадр {CHECK_AT}: {len(span)} шт.")
    for rms, n, a, b, j, k, A, B in sorted(span, key=lambda x: x[1])[:3]:
        print(f"  кадры {a} -> {b}: совпадений {n}, невязка {rms:.1f} px  (картинка link_{a}_{b}.png)")
        draw_link(kpaths[j], kpaths[k], A, B, os.path.join(OUT, f"link_{a}_{b}.png"))
def main():
    os.makedirs(OUT, exist_ok=True)
    allp = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        allp += glob.glob(os.path.join(SRC, ext))
    allp = sorted(allp, key=natural_key)[START:END]
    if not allp:
        raise SystemExit(f"Кадров не найдено в {SRC}. Проверь путь и что в папке лежат png/jpg.")
    print("Кадров в папке:", len(allp))
    t0 = time.time()

    keys = select_keyframes(allp)
    print(f"Ключевых кадров: {len(keys)} из {len(allp)} ({time.time() - t0:.0f} с)")
    kpaths = [allp[i] for i in keys]

    feats = [features(load(p)) for p in kpaths]
    edges = []
    for k in range(1, len(kpaths)):
        for j in range(max(0, k - NEIGHBORS), k):
            r = match(feats[j], feats[k])
            if r is not None:
                edges.append((j, k, r[0], r[1]))
    print("Связей между кадрами:", len(edges))

    linked = {k for j, k, _, _ in edges}
    breaks = [fnum(kpaths[k]) for k in range(1, len(keys)) if k not in linked]
    print("Цепочка рвётся перед кадрами №:", breaks)

    comps = components(len(kpaths), edges)
    print("Кусков (связных групп):", len(comps), ", размеры:", [len(c) for c in comps[:8]])
    size = (load(kpaths[0]).shape[1], load(kpaths[0]).shape[0])
    results = []
    for part, comp in enumerate(comps[:MAX_PARTS]):
        if len(comp) < 5:
            break
        best = sorted(comp)
        cedges = prune_links(best, edges, size)
        poses, rms = solve(best, cedges, size)
        print(f"Кусок {part}: ключевых кадров {len(best)}, "
              f"кадры №{fnum(kpaths[best[0]])}-{fnum(kpaths[best[-1]])}, невязка {rms:.2f} px")
        mosaic, valid = render([kpaths[i] for i in best], poses)
        name = f"orb_s{START}_part{part}_n{len(best)}"
        save_map(mosaic, valid, name, {"source": "ORB + общая подгонка", "keyframes_used": len(best),
                                       "first_frame": fnum(kpaths[best[0]]), "last_frame": fnum(kpaths[best[-1]]),
                                       "frames_total": len(allp), "rms_px": round(rms, 3)})
        print(f"  Файл: {name}.png, размер {mosaic.shape[1]}x{mosaic.shape[0]}")
        if part == 0:
            diagnose(best, cedges, poses, kpaths)
        results.append((best, poses))
    print(f"Готово за {time.time() - t0:.0f} с.")
    return keys, results

if __name__ == "__main__":
    main()