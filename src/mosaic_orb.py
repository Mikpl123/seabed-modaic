import cv2
import glob
import json
import os
import re
import time
from functools import lru_cache
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- вход и выход ----------
SRC = os.path.join(BASE, "dataset")        # папка со всеми кадрами
OUT = os.path.join(BASE, "output")
START, END = 0, None                       # с какого по порядку кадра брать (END = None: до конца)

# ---------- ключевые кадры и связи (ORB) ----------
WIDTH = 480                                # рабочая ширина кадра для поиска положений
MIN_SHIFT = 0.15                           # новый ключевой кадр берём, когда сдвиг >= 15% меньшей стороны
CHECK_EVERY = 2                            # проверять каждый 2-й кадр
NEIGHBORS = 5                              # с сколькими прошлыми ключевыми кадрами сопоставлять
MIN_INLIERS = 25                           # минимум надёжных совпадений для связи
MAX_PARTS = 3                              # сколько самых больших кусков сохранять
SCALE_PRIOR = 1e5                          # держать масштаб кадров близким к 1 (0 = выключить: карта схлопывается)
LOOP_SAMPLE = 20                           # сшивка кусков: сколько ключевых кадров от каждого куска сверяем (0 = выключить)
LOOP_MIN_INLIERS = 40                      # сшивка кусков: минимум совпадений (строже, чем MIN_INLIERS)
LOOP_STRONG = 120                          # сшивка кусков: одна связь принимается без подтверждения, если совпадений не меньше

# ---------- сборка карты ----------
RENDER_WIDTH = 960                         # ширина кадров при сборке (не больше исходной): больше = чётче, но дольше
MAX_SIDE = 4000                            # максимальный размер карты в пикселях
TILE_SIZE = 1024                            # размер блока финальной сборки (меньше = меньше пиковая память)
CACHE_SIZE = 24                              # сколько уменьшенных кадров держать в кеше
REFINE = True                              # уточнять положение кадров по изображению (убирает двоение камней и рыб)
REFINE_SIDE = 3000                         # размер рабочей карты уточнения, px
REFINE_MIN_CC = 0.30                       # минимальная корреляция, чтобы принять уточнение кадра
REFINE_MAX_SHIFT = 40.0                    # максимальный сдвиг кадра при уточнении, px (рабочего размера)
REFINE_MIN_OVERLAP = 0.10                  # минимальная доля перекрытия с уже собранным, чтобы уточнять кадр
SELECT_MIN = 0.30                          # пиксель заполнен, если лучший кадр лежит на нём не ближе этой доли к краю (0..1)
SEAM_SCALE = 0.5                           # масштаб, на котором ищем швы (меньше = быстрее и меньше памяти)
FEATHER = 4.0                              # ширина размытия шва для мелких деталей, px (мало = нет двоения)
FEATHER_LOW = 25.0                         # ширина смешивания плавного света на шве, px (больше = плавнее переходы яркости)
SPLIT_SIGMA = 8.0                          # граница между «светом» и «деталями», px
EXPOSURE_MATCH = True                      # выравнивать яркость кадров между собой (убирает ступеньки на швах)

# ---------- улучшение готовой карты ----------
ENHANCE = True                             # улучшение подводного вида (сырую карту всё равно сохраняем рядом: *_raw.png)
BG_SIGMA = 40.0                            # размер "мути", px: что плавнее этого, считаем фоном
BG_STRENGTH = 0.7                          # 0..1: насколько вычитаем фон (0 = не вычитать)
DENOISE_H = 3.0                            # сила шумоподавления (0 = выключить)
TARGET_MEAN = 145                          # средняя яркость готовой карты, 0..255 (None = оставить как есть)
CONTRAST = 1.8                             # контраст вокруг средней яркости (1 = не менять)
CLARITY = 0.2                              # локальный контраст среднего масштаба (0 = выключить)
SHARPEN = 0.35                             # резкость мелких деталей (0 = выключить)

# ---------- flat-field correction: выравнивание неравномерного освещения ----------
FLAT_FIELD = True                            # включить коррекцию плавного виньетирования/пятен света
FLAT_SIGMA = 80.0                           # радиус оценки освещения в пикселях готовой карты
FLAT_STRENGTH = 0.55                         # 0..1: сила коррекции (начать примерно с 0.3-0.5)
FLAT_GAIN_MIN = 0.70                           # ограничение снизу: не затемнять сильно
FLAT_GAIN_MAX = 1.60                           # ограничение сверху: не вытягивать шум слишком сильно

# ---------- GeoTIFF ----------
GEO_EPSG = None                            # код проекции в метрах, напр. 32755 (UTM 55S). None = локальные координаты
GEO_ORIGIN = (0.0, 0.0)                    # координаты (X, Y) левого верхнего угла карты, м
GEO_MPP = None                             # метров на пиксель карты. None = неизвестно (1 пиксель = 1 единица)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def fnum(path):
    """Номер кадра из имени файла (frame000123.png -> 123)."""
    m = re.findall(r"\d+", os.path.basename(path))
    return int(m[-1]) if m else -1


@lru_cache(maxsize=CACHE_SIZE)
def _load_cached(path, width):
    """Кеширует уменьшенный кадр. Ключ: путь + ширина; кеш ограничен CACHE_SIZE."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать кадр: {path}")
    k = width / img.shape[1]
    if abs(k - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def load(path, width=None):
    """Загрузка с LRU-кешем; width включён в ключ кеша."""
    return _load_cached(os.path.abspath(path), int(width or WIDTH))


def hp_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 15)          # убираем плавный свет


# отбор кадров
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


# зацепки ORB и связи между кадрами
detector = cv2.ORB_create(4000, fastThreshold=5)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
RATIO = 0.75


def features(img, k=12, gain=25):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(g, (0, 0), k)                                   # плавный фон (муть)
    std = np.sqrt(cv2.GaussianBlur((g - mean) ** 2, (0, 0), k)) + 2         # местный контраст
    g = np.clip(128 + gain * (g - mean) / std, 0, 255).astype(np.uint8)     # выравниваем контраст
    kp, des = detector.detectAndCompute(g, None)
    pts = np.float32([x.pt for x in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, des


def match(fa, fb, min_inl=None):
    mi = min_inl or MIN_INLIERS
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return None
    pairs = [p for p in bf.knnMatch(da, db, k=2) if len(p) == 2]
    good = [m for m, n in pairs if m.distance < RATIO * n.distance]
    if len(good) < mi:
        return None
    A = pa[[m.queryIdx for m in good]]
    B = pb[[m.trainIdx for m in good]]
    M, inl = cv2.estimateAffinePartial2D(B, A, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < mi:
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


def _apply(M, P):
    return P @ M[:, :2].T + M[:, 2]


def similarity(X, Y):
    """Y = z*X + c по точкам (комплексные числа): масштаб, поворот и сдвиг."""
    x = X[:, 0] + 1j * X[:, 1]
    y = Y[:, 0] + 1j * Y[:, 1]
    x0, y0 = x - x.mean(), y - y.mean()
    den = float(np.sum(np.abs(x0) ** 2))
    if den < 1e-9:
        return None
    z = np.sum(np.conj(x0) * y0) / den
    return z, y.mean() - z * x.mean()


def link_pieces(feats, edges, size):
    """Куски цепочки могут перекрываться на местности: аппарат вернулся или смотрел на то же место, хотя соседние
    во времени кадры не сопоставились. Сверяем кадры разных кусков между собой. Связь принимаем, только если её
    подтверждают минимум две независимые (или одна очень сильная) и они дают одно и то же взаимное положение кусков."""
    cx, cy = size[0] / 2, size[1] / 2
    for _ in range(3):
        comps = components(len(feats), edges)
        if len(comps) < 2:
            break
        pose = {}                                          # предварительные положения кадров внутри каждого куска
        for c in comps:
            cs = set(c)
            nodes = sorted(c)
            poses, _r = solve(nodes, [e for e in edges if e[0] in cs and e[1] in cs], size)
            pose.update(zip(nodes, poses))
        added = []
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                ids_a, ids_b = sorted(comps[a]), sorted(comps[b])
                pick = lambda ids: [ids[i] for i in np.unique(np.linspace(0, len(ids) - 1, min(LOOP_SAMPLE, len(ids))).astype(int))]
                cands = []
                for i in pick(ids_a):
                    for k in pick(ids_b):
                        r = match(feats[i], feats[k], LOOP_MIN_INLIERS)
                        if r is None:
                            continue
                        s = similarity(_apply(pose[k], r[1]), _apply(pose[i], r[0]))
                        if s is not None and 0.7 < abs(s[0]) < 1.4:
                            cands.append((len(r[0]), i, k, s[0], s[1], r[0], r[1]))
                if not cands:
                    continue
                p0 = np.mean([_apply(pose[n], np.array([[cx, cy]]))[0] for n in ids_b], axis=0)
                p0 = p0[0] + 1j * p0[1]
                q = [c[3] * p0 + c[4] for c in cands]       # куда попадает центр куска B в системе куска A
                best = []
                for ci, c in enumerate(cands):
                    mem = [oi for oi, o in enumerate(cands)
                           if abs(np.log(abs(c[3] / o[3]))) < 0.06 and abs(np.angle(c[3] / o[3])) < 0.06 and abs(q[ci] - q[oi]) < 40]
                    if len(mem) > len(best) or (len(mem) == len(best) and sum(cands[m][0] for m in mem) > sum(cands[m][0] for m in best)):
                        best = mem
                if len(best) >= 2 or (best and cands[best[0]][0] >= LOOP_STRONG):
                    top = sorted(best, key=lambda m: -cands[m][0])[:6]
                    added += [(cands[m][1], cands[m][2], cands[m][5], cands[m][6]) for m in top]
                    print(f"  Сшивка: куски из {len(ids_a)} и {len(ids_b)} кадров связаны через общее место "
                          f"(подтверждений {len(best)}, макс. совпадений {max(cands[m][0] for m in best)})")
        if not added:
            break
        edges = edges + added
    return edges


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


def illumination(paths, ld):
    """Карта света по выборке кадров: L (яркость), gain (поправка на виньетирование), C (средний уровень кадров)."""
    idx = np.linspace(0, len(paths) - 1, min(60, len(paths))).astype(int)
    sample = [ld(paths[i]) for i in idx]
    L = cv2.GaussianBlur(np.mean([cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) for s in sample], axis=0), (0, 0), 25)
    L = np.maximum(L / L.max(), 0.05)
    gain = np.clip(L.mean() / L, 0.6, 2.5)
    return L, gain


def exposure_gains(paths, edges, gain, ld):
    """Подгоняем яркость каждого кадра к соседям: по совпавшим точкам сравниваем сглаженную яркость
    (в логарифмах, метод наименьших квадратов). Убирает ступеньки яркости на швах."""
    m = len(paths)
    sm = []
    for p in paths:
        g = np.clip(ld(p).astype(np.float32) * gain[..., None], 0, 255).mean(axis=2) + 1.0
        sm.append(cv2.GaussianBlur(g, (0, 0), 6 * g.shape[1] / WIDTH))

    def samp(img, P):
        x = np.clip(np.rint(P[:, 0]).astype(int), 0, img.shape[1] - 1)
        y = np.clip(np.rint(P[:, 1]).astype(int), 0, img.shape[0] - 1)
        return img[y, x]
    rows, rhs, wts = [], [], []
    for j, k, A, B in edges:
        d = float(np.median(np.log(samp(sm[j], A)) - np.log(samp(sm[k], B))))
        r = np.zeros(m)
        r[j], r[k] = 1.0, -1.0
        rows.append(r)
        rhs.append(-d)                                     # g_j - g_k = -(лог яркости в j - лог яркости в k)
        wts.append(np.sqrt(min(len(A), 100)))
    if not rows:
        return np.ones(m)
    wts = np.array(wts)
    Aeq = np.vstack([np.array(rows) * wts[:, None], 0.3 * np.eye(m)])    # слабо держим поправки около 1
    beq = np.r_[np.array(rhs) * wts, np.zeros(m)]
    g = np.linalg.lstsq(Aeq, beq, rcond=None)[0]
    return np.exp(np.clip(g - g.mean(), -0.5, 0.5))


# ---------- уточнение положения кадров по изображению ----------
def ecc_shift(ref, frm, ov):
    """Сдвиг кадра относительно уже собранной карты по самому изображению (ECC, от грубого масштаба к точному)."""
    ys, xs = np.where(ov)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tpl, inp = (ref * ov)[y0:y1, x0:x1], (frm * ov)[y0:y1, x0:x1]
    msk = ov[y0:y1, x0:x1].astype(np.uint8) * 255
    if min(tpl.shape) < 48:
        return None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
    t, cc = np.zeros(2), None
    for lvl in (0.125, 0.25, 0.5, 1.0):
        if lvl < 1:
            a = cv2.resize(tpl, None, fx=lvl, fy=lvl, interpolation=cv2.INTER_AREA)
            b = cv2.resize(inp, None, fx=lvl, fy=lvl, interpolation=cv2.INTER_AREA)
            m = cv2.resize(msk, None, fx=lvl, fy=lvl, interpolation=cv2.INTER_NEAREST)
        else:
            a, b, m = tpl, inp, msk
        if min(a.shape) < 20:
            continue
        warp = np.array([[1, 0, t[0] * lvl], [0, 1, t[1] * lvl]], np.float32)
        try:
            cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_TRANSLATION, crit, m, 5)
        except cv2.error:
            return None
        t = np.array([warp[0, 2], warp[1, 2]]) / lvl
    return None if cc is None else (t, float(cc))


def refine_poses(paths, poses):
    """Каждый кадр по очереди прижимаем к уже собранной части карты (по камням, а не по точкам ORB).
    Общий расчёт кладёт кадры с ошибкой в несколько пикселей, из-за этого камни и рыбы двоятся; здесь соседи
    приводятся в точное совпадение. Где текстуры нет (гладкий песок), кадр не трогаем, но переносим на него
    поправку предыдущего, чтобы цепочка не рвалась."""
    ld = load
    h, w = ld(paths[0]).shape[:2]
    _, gain = illumination(paths, ld)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    allp = np.concatenate([cv2.transform(corners[None], M)[0] for M in poses])
    pad = 2 * REFINE_MAX_SHIFT
    mn, mx = allp.min(0) - pad, allp.max(0) + pad
    s = min(1.0, REFINE_SIDE / float((mx - mn).max()))
    W, H = int(np.ceil((mx[0] - mn[0]) * s)) + 1, int(np.ceil((mx[1] - mn[1]) * s)) + 1
    canvas = np.zeros((H, W), np.float32)
    filled = np.zeros((H, W), bool)
    cen = np.zeros((h, w), np.uint8)                       # в карту пишем и по ней сверяем только середину кадра
    cen[int(.12 * h):int(.88 * h), int(.12 * w):int(.88 * w)] = 255
    n_cen = float((cen > 0).sum()) * s * s

    def place(M):
        Mc = np.zeros((2, 3))
        Mc[:, :2] = s * M[:, :2]
        Mc[:, 2] = s * (M[:, 2] - mn)
        pts = cv2.transform(corners[None], Mc)[0]
        x0, x1 = max(int(np.floor(pts[:, 0].min())), 0), min(int(np.ceil(pts[:, 0].max())) + 1, W)
        y0, y1 = max(int(np.floor(pts[:, 1].min())), 0), min(int(np.ceil(pts[:, 1].max())) + 1, H)
        Ml = Mc.copy()
        Ml[0, 2] -= x0
        Ml[1, 2] -= y0
        return x0, x1, y0, y1, Ml

    out, stat, carry = [], {"moved": 0, "rejected": 0, "free": 0, "shifts": []}, np.zeros(2)
    for p, M in zip(paths, poses):
        M = M.copy()
        M[:, 2] += carry
        g = np.clip(ld(p).astype(np.float32) * gain[..., None], 0, 255).mean(axis=2)
        g = g - cv2.GaussianBlur(g, (0, 0), 40)            # убираем плавный свет, остаются камни и текстура
        x0, x1, y0, y1, Ml = place(M)
        fw = cv2.warpAffine(g, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        cw = cv2.warpAffine(cen, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST) > 0
        ov = cw & filled[y0:y1, x0:x1]
        if ov.sum() >= REFINE_MIN_OVERLAP * n_cen:
            r = ecc_shift(canvas[y0:y1, x0:x1], fw, ov)
            if r is not None and r[1] >= REFINE_MIN_CC and np.hypot(*(r[0] / s)) <= REFINE_MAX_SHIFT:
                d = -r[0] / s                              # ECC даёт, на сколько содержимое сдвинуто; кадр двигаем обратно
                M[:, 2] += d
                carry = carry + d
                stat["moved"] += 1
                stat["shifts"].append(float(np.hypot(*d)))
                x0, x1, y0, y1, Ml = place(M)
                fw = cv2.warpAffine(g, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
                cw = cv2.warpAffine(cen, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST) > 0
            else:
                stat["rejected"] += 1
        else:
            stat["free"] += 1
        canvas[y0:y1, x0:x1][cw] = fw[cw]
        filled[y0:y1, x0:x1][cw] = True
        out.append(M)
    return out, stat


def render(paths, poses, edges=None, width=None):
    """Сборка карты. Положения poses заданы для кадров шириной WIDTH, а картинку берём из кадров шириной width
    (чётче). На каждом пикселе берём ОДИН кадр: швы ищем там, где соседние кадры лучше всего совпадают (в обход
    камней и рыб), поэтому объекты не двоятся. Свет на шве смешиваем широко, детали узко."""
    width = width or WIDTH
    kk = width / WIDTH
    ld = lambda p: load(p, width)
    h, w = ld(paths[0]).shape[:2]
    h0, w0 = load(paths[0]).shape[:2]
    crn = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    crn0 = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]])
    poses_r = [np.c_[M[:, :2], M[:, 2] * kk] for M in poses]
    allp = np.concatenate([cv2.transform(crn[None], M)[0] for M in poses_r])
    minx, miny = allp.min(0)
    maxx, maxy = allp.max(0)
    sc = min(1.0, MAX_SIDE / max(maxx - minx, maxy - miny))
    Wc, Hc = int(np.ceil((maxx - minx) * sc)) + 1, int(np.ceil((maxy - miny) * sc)) + 1
    px = kk * sc                                           # пикселей карты на пиксель рабочего кадра
    print(f"Размер карты: {Wc} x {Hc}, кадры шириной {width} px")

    def place(M, corners, scale, W, H, mnx, mny):
        Mc = np.zeros((2, 3))
        Mc[:, :2] = scale * M[:, :2]
        Mc[:, 2] = scale * (M[:, 2] - (mnx, mny))
        pts = cv2.transform(corners[None], Mc)[0]
        x0, x1 = max(int(np.floor(pts[:, 0].min())), 0), min(int(np.ceil(pts[:, 0].max())) + 1, W)
        y0, y1 = max(int(np.floor(pts[:, 1].min())), 0), min(int(np.ceil(pts[:, 1].max())) + 1, H)
        if x1 <= x0 or y1 <= y0:
            return None
        Ml = Mc.copy()
        Ml[0, 2] -= x0
        Ml[1, 2] -= y0
        return x0, x1, y0, y1, Ml

    L, gain = illumination(paths, ld)
    if EXPOSURE_MATCH and edges:
        expo = exposure_gains(paths, [(j, k, A * kk, B * kk) for j, k, A, B in edges], gain, ld)
    else:
        expo = np.ones(len(paths))
    expo = np.asarray(expo, dtype=np.float32)              # float32, иначе OpenCV (поиск швов) не примет кадры
    print(f"  Поправки яркости кадров: от {expo.min():.2f} до {expo.max():.2f}")

    def prep(p, idx):
        return (ld(p).astype(np.float32) * gain[..., None] * expo[idx]).astype(np.float32)

    m0 = np.zeros((h, w), np.uint8)
    m0[2:-2, 2:-2] = 255
    d = cv2.distanceTransform(m0, cv2.DIST_L2, 3)
    score = ((d / d.max()) * L).astype(np.float32)         # лучшее место кадра: центр и не тёмный угол
    cover = (m0 > 0).astype(np.float32)
    boxes = [place(M, crn, sc, Wc, Hc, minx, miny) for M in poses_r]

    label = None
    try:                                                   # швы ищем на рабочем масштабе (экономит память)
        allp0 = np.concatenate([cv2.transform(crn0[None], M)[0] for M in poses])
        mnx0, mny0 = allp0.min(0)
        mxx0, mxy0 = allp0.max(0)
        ss = SEAM_SCALE
        Ws, Hs = int(np.ceil((mxx0 - mnx0) * ss)) + 1, int(np.ceil((mxy0 - mny0) * ss)) + 1
        good = cv2.resize(((score > SELECT_MIN) * 255).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
        imgs, masks, cors, ids = [], [], [], []
        for idx, (p, M) in enumerate(zip(paths, poses)):
            pl = place(M, crn0, ss, Ws, Hs, mnx0, mny0)
            if pl is None:
                continue
            x0, x1, y0, y1, Ml = pl
            osz = (x1 - x0, y1 - y0)
            small = cv2.resize(prep(p, idx), (w0, h0), interpolation=cv2.INTER_AREA)
            imgs.append(cv2.warpAffine(small, Ml, osz, flags=cv2.INTER_LINEAR))
            masks.append(cv2.warpAffine(good, Ml, osz, flags=cv2.INTER_NEAREST))
            cors.append((x0, y0))
            ids.append(idx)
        finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
        res = finder.find(imgs, cors, masks)
        if res is not None:
            masks = res
        labs = np.full((Hs, Ws), -1, np.int16)
        for (x0, y0), mk, idx in zip(cors, masks, ids):
            mk = np.asarray(mk.get() if hasattr(mk, "get") else mk)
            labs[y0:y0 + mk.shape[0], x0:x0 + mk.shape[1]][mk > 0] = idx
        del imgs, masks
        label = cv2.resize(labs, (Wc, Hc), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        print("  Швы: поиск по совпадению соседних кадров")
    except (AttributeError, cv2.error) as e:
        print(f"  Поиск швов недоступен ({e}), беру кадр с лучшим положением")
    if label is None:
        best = np.zeros((Hc, Wc), np.float32)
        label = np.full((Hc, Wc), -1, np.int32)
        for idx, pl in enumerate(boxes):
            if pl is None:
                continue
            x0, x1, y0, y1, Ml = pl
            s = cv2.warpAffine(score, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            bsub, lsub = best[y0:y1, x0:x1], label[y0:y1, x0:x1]
            better = s > bsub
            bsub[better] = s[better]
            lsub[better] = idx
        label[best <= SELECT_MIN] = -1
    valid = label >= 0

    # Финальная сборка по блокам: не держим сразу четыре полноразмерных float32-массива.
    # label и итоговые uint8/valid остаются глобальными, но временные накопители ограничены TILE_SIZE.
    valid = label >= 0
    mosaic = np.zeros((Hc, Wc, 3), np.uint8)
    tile = max(256, int(TILE_SIZE))
    print(f"  Тайловая сборка: блок {tile}x{tile}")

    # Подготовленные кадры кешируются только на время render; глобальный LRU load() также помогает
    # повторным чтениям между этапами программы.
    prep_cache = {}
    def prep_cached(p, idx):
        if idx not in prep_cache:
            prep_cache[idx] = (ld(p).astype(np.float32) * gain[..., None] * expo[idx]).astype(np.float32)
        return prep_cache[idx]

    for ty in range(0, Hc, tile):
        yb, ye = ty, min(ty + tile, Hc)
        for tx in range(0, Wc, tile):
            xb, xe = tx, min(tx + tile, Wc)
            th, tw = ye - yb, xe - xb
            acc_h = np.zeros((th, tw, 3), np.float32)
            acc_l = np.zeros((th, tw, 3), np.float32)
            ws_h = np.zeros((th, tw), np.float32)
            ws_l = np.zeros((th, tw), np.float32)

            for idx, (p, pl) in enumerate(zip(paths, boxes)):
                if pl is None:
                    continue
                x0, x1, y0, y1, Ml = pl
                ix0, ix1 = max(x0, xb), min(x1, xe)
                iy0, iy1 = max(y0, yb), min(y1, ye)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                # Кадр не выбран для этого участка — не тратим время на warp.
                sel_full = (label[iy0:iy1, ix0:ix1] == idx).astype(np.float32)
                if not sel_full.any():
                    continue

                # Mc = глобальное преобразование кадра; переводим его в координаты текущего тайла.
                out_size = (tw, th)
                Mtile = Ml.copy()
                Mtile[0, 2] += x0 - xb
                Mtile[1, 2] += y0 - yb
                img = prep_cached(p, idx)
                low = cv2.GaussianBlur(img, (0, 0), SPLIT_SIGMA * kk)
                hi_w = cv2.warpAffine(img - low, Mtile, out_size, flags=cv2.INTER_CUBIC)
                lo_w = cv2.warpAffine(low, Mtile, out_size, flags=cv2.INTER_LINEAR)

                # Метка шва уже находится в координатах итоговой карты.
                sel = (label[yb:ye, xb:xe] == idx).astype(np.float32)
                cov = cv2.warpAffine(cover, Mtile, out_size, flags=cv2.INTER_LINEAR) > 0.999
                wh = (cv2.GaussianBlur(sel, (0, 0), FEATHER) if FEATHER > 0 else sel) * cov
                wl = (cv2.GaussianBlur(sel, (0, 0), FEATHER_LOW) if FEATHER_LOW > 0 else sel) * cov
                acc_h += hi_w * wh[..., None]
                ws_h += wh
                acc_l += lo_w * wl[..., None]
                ws_l += wl

            tile_valid = (ws_h > 1e-3) & (ws_l > 1e-3) & valid[yb:ye, xb:xe]
            out_tile = acc_h / np.maximum(ws_h, 1e-6)[..., None] + acc_l / np.maximum(ws_l, 1e-6)[..., None]
            out_tile[~tile_valid] = 0
            mosaic[yb:ye, xb:xe] = np.clip(out_tile, 0, 255).astype(np.uint8)
            valid[yb:ye, xb:xe] &= tile_valid
        print(f"  Готово строка тайлов y={yb}:{ye} / {Hc}")

    return mosaic, valid, px


# ---------- улучшение подводного вида готовой карты ----------
def masked_blur(x, valid, sigma):
    """Размытие только по заполненным пикселям (пустые не затемняют края). Считаем на уменьшенной копии."""
    k = max(1, int(sigma // 6))
    small = cv2.resize(x * valid, None, fx=1.0 / k, fy=1.0 / k, interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid.astype(np.float32), None, fx=1.0 / k, fy=1.0 / k, interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(small, (0, 0), sigma / k)
    den = cv2.GaussianBlur(vs, (0, 0), sigma / k)
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    return cv2.resize(out, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)


def flat_field_correction(mosaic, valid, px=1.0):
    """Исправляет плавную неравномерность освещения по яркости готовой мозаики.

    Оценивает фон освещения большим masked Gaussian blur, затем применяет
    ограниченную мультипликативную поправку. Детали дна сохраняются лучше,
    чем при простом вычитании размытого фона. valid=True обозначает реальные
    пиксели карты; пустые области не участвуют в оценке.
    """
    if not valid.any() or FLAT_STRENGTH <= 0:
        return mosaic

    lab = cv2.cvtColor(mosaic, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    # Плавная карта освещения: исключаем пустые области из оценки.
    light = masked_blur(L, valid, FLAT_SIGMA * px)
    good = valid & np.isfinite(light) & (light > 1.0)
    if not good.any():
        return mosaic

    # Целевой уровень света — среднее по заполненной области.
    target = float(np.median(light[good]))
    gain = target / np.maximum(light, 1.0)
    gain = np.clip(gain, FLAT_GAIN_MIN, FLAT_GAIN_MAX)
    # Смешиваем исходный gain с единицей, чтобы регулировать силу эффекта.
    gain = 1.0 + FLAT_STRENGTH * (gain - 1.0)
    corrected = L * gain
    lab[:, :, 0] = np.clip(corrected, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out[~valid] = 0
    return out


def enhance(mosaic, valid, px=1.0):
    """Улучшение подводного вида: муть вычитаем, шум гасим, подбираем яркость и контраст, добавляем чёткость.
    px: во сколько раз карта крупнее рабочего размера (от него зависят радиусы фильтров)."""
    lab = cv2.cvtColor(mosaic, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    m0 = float(L[valid].mean()) if valid.any() else 128.0
    near = masked_blur(valid.astype(np.float32), valid, 25 * px) > 0
    L = np.where(valid, L, np.where(near, masked_blur(L, valid, 25 * px), m0))   # пустое заполняем соседним, чтобы края не искажались
    if BG_STRENGTH > 0:
        bg = masked_blur(L, valid, BG_SIGMA * px)                                # плавный фон = муть и неровный свет
        L = L - BG_STRENGTH * (bg - float(bg[valid].mean()))
    L8 = np.clip(L, 0, 255).astype(np.uint8)
    if DENOISE_H > 0:
        L8 = cv2.fastNlMeansDenoising(L8, None, h=DENOISE_H, templateWindowSize=5, searchWindowSize=15)
    Lf = L8.astype(np.float32)
    mid = float(np.mean(Lf[valid][::7])) if valid.any() else 128.0
    z = np.clip((mid + CONTRAST * (Lf - mid)) / 255.0, 0, 1)                     # контраст вокруг средней яркости
    target = (TARGET_MEAN if TARGET_MEAN else m0) / 255.0
    zs = z[valid][::7] if valid.any() else z.ravel()[::7]
    g_lo, g_hi = 0.2, 4.0                                  # подбираем гамму так, чтобы средняя яркость стала target
    for _ in range(30):
        g = float(np.sqrt(g_lo * g_hi))
        if float(np.mean(zs ** g)) > target:
            g_lo = g
        else:
            g_hi = g
    y = 255.0 * z ** float(np.sqrt(g_lo * g_hi))
    if CLARITY > 0:
        y = y + CLARITY * (y - cv2.GaussianBlur(y, (0, 0), 8 * px))             # контраст среднего масштаба
    if SHARPEN > 0:
        y = y + SHARPEN * (y - cv2.GaussianBlur(y, (0, 0), 1.2 * px))           # резкость мелких деталей
    lab[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out[~valid] = 0
    return out


# ---------- сохранение: PNG и GeoTIFF ----------
def geo_info(width, height):
    mpp = float(GEO_MPP) if GEO_MPP else 1.0
    x0, y0 = GEO_ORIGIN
    return {"crs": f"EPSG:{int(GEO_EPSG)}" if GEO_EPSG else None,
            "units": "m" if GEO_MPP else "px (масштаб неизвестен)",
            "meters_per_pixel": mpp if GEO_MPP else None,
            "origin_top_left": [x0, y0],
            "bounds": [x0, y0 - height * mpp, x0 + width * mpp, y0]}


def write_tiff_rgba(path, rgba, scale, tiepoint, geokeys=None):
    """Минимальная запись GeoTIFF (RGBA, без сжатия) без сторонних библиотек."""
    import struct
    H, W = rgba.shape[:2]
    body = bytearray(np.ascontiguousarray(rgba, dtype=np.uint8).tobytes())      # данные лежат сразу после заголовка (смещение 8)

    def blob(b):
        if len(body) % 2:
            body.append(0)
        off = 8 + len(body)
        body.extend(b)
        return off

    def short(tag, vals):
        if len(vals) <= 2:
            v = struct.pack('<%dH' % len(vals), *vals).ljust(4, b'\0')
        else:
            v = struct.pack('<I', blob(struct.pack('<%dH' % len(vals), *vals)))
        return (tag, 3, len(vals), v)

    def long_(tag, val):
        return (tag, 4, 1, struct.pack('<I', val))

    def dbl(tag, vals):
        return (tag, 12, len(vals), struct.pack('<I', blob(struct.pack('<%dd' % len(vals), *vals))))

    entries = [long_(256, W), long_(257, H), short(258, [8, 8, 8, 8]), short(259, [1]), short(262, [2]),
               long_(273, 8), short(277, [4]), long_(278, H), long_(279, W * H * 4), short(284, [1]),
               short(338, [2]),                                                 # ExtraSamples: неассоциированная прозрачность
               dbl(33550, scale), dbl(33922, tiepoint)]                         # ModelPixelScale, ModelTiepoint
    if geokeys:
        entries.append(short(34735, geokeys))                                   # GeoKeyDirectory
    entries.sort(key=lambda e: e[0])
    if len(body) % 2:
        body.append(0)
    ifd = struct.pack('<H', len(entries)) + b''.join(struct.pack('<HHI', t, ty, c) + v for t, ty, c, v in entries) + struct.pack('<I', 0)
    with open(path, 'wb') as f:
        f.write(b'II' + struct.pack('<HI', 42, 8 + len(body)))
        f.write(body)
        f.write(ifd)


def save_geotiff(mosaic, valid, path):
    """GeoTIFF (RGBA, прозрачность = пустые места). Левый верхний угол GEO_ORIGIN, размер пикселя GEO_MPP, Y вверх.
    Без GEO_EPSG файл содержит только локальную привязку."""
    if GEO_EPSG and 4000 <= int(GEO_EPSG) < 5000:
        raise ValueError("GEO_EPSG должен быть проекцией в метрах (например UTM 326xx/327xx), а не градусами.")
    mpp = float(GEO_MPP) if GEO_MPP else 1.0
    x0, y0 = GEO_ORIGIN
    rgba = np.dstack([cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB), valid.astype(np.uint8) * 255])
    keys = None
    if GEO_EPSG:
        keys = [1, 1, 0, 3,
                1024, 0, 1, 1,                                                  # GTModelType = Projected
                1025, 0, 1, 1,                                                  # GTRasterType = PixelIsArea
                3072, 0, 1, int(GEO_EPSG)]                                      # ProjectedCSType = EPSG
    write_tiff_rgba(path, rgba, [mpp, mpp, 0.0], [0.0, 0.0, 0.0, float(x0), float(y0), 0.0], keys)


def save_map(mosaic, valid, name, meta, raw=None):
    bgra = cv2.cvtColor(mosaic, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = valid.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT, name + ".png"), bgra)
    if raw is not None:                                    # карта без улучшения, для сравнения
        braw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
        braw[:, :, 3] = valid.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(OUT, name + "_raw.png"), braw)
    info = {"file": name + ".png", "width": int(mosaic.shape[1]), "height": int(mosaic.shape[0])}
    info.update(geo_info(mosaic.shape[1], mosaic.shape[0]))
    info.update(meta)
    try:
        save_geotiff(mosaic, valid, os.path.join(OUT, name + ".tif"))
        info["geotiff"] = name + ".tif"
        print(f"  GeoTIFF: {name}.tif")
    except ValueError as e:
        print(f"  GeoTIFF не записан: {e}")
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


# ---------- проверка связей (для тестировщика) ----------


def draw_link(path_a, path_b, A, B, out_path):
    """Картинка: два кадра рядом и линии между совпавшими точками."""
    ia, ib = load(path_a), load(path_b)
    canvas = np.hstack([ia, ib])
    off = ia.shape[1]
    idx = np.linspace(0, len(A) - 1, min(len(A), 60)).astype(int)     # равномерно по всем точкам, не первые 60
    for (x1, y1), (x2, y2) in zip(A[idx], B[idx]):
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
    worst = sorted(rows, key=lambda x: -x[0])[:top]
    for rms, n, a, b, j, k, A, B in worst:
        print(f"  кадры {a} -> {b}: совпадений {n}, невязка {rms:.1f} px")
    for rms, n, a, b, j, k, A, B in worst[:3]:                  # картинки для 3 худших связей
        draw_link(kpaths[j], kpaths[k], A, B, os.path.join(OUT, f"link_{a}_{b}.png"))
        print(f"  картинка: link_{a}_{b}.png")


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

    size = (load(kpaths[0]).shape[1], load(kpaths[0]).shape[0])
    comps = components(len(kpaths), edges)
    print("Кусков (связных групп):", len(comps), ", размеры:", [len(c) for c in comps[:8]])
    if LOOP_SAMPLE and len(comps) > 1:
        t3 = time.time()
        edges = link_pieces(feats, edges, size)
        comps = components(len(kpaths), edges)
        print(f"После сшивки кусков между собой: {len(comps)} шт., размеры: {[len(c) for c in comps[:8]]} ({time.time() - t3:.0f} с)")
    orig_w = cv2.imread(allp[0]).shape[1]
    rw = int(min(max(RENDER_WIDTH or WIDTH, WIDTH), orig_w))
    print(f"Ширина исходных кадров: {orig_w} px; карта собирается из кадров шириной {rw} px")
    results = []
    for part, comp in enumerate(comps[:MAX_PARTS]):
        if len(comp) < 5:
            break
        best = sorted(comp)
        cedges = prune_links(best, edges, size)
        poses, rms = solve(best, cedges, size)
        print(f"Кусок {part}: ключевых кадров {len(best)}, "
              f"кадры №{fnum(kpaths[best[0]])}-{fnum(kpaths[best[-1]])}, невязка {rms:.2f} px")
        posd = {n: i for i, n in enumerate(best)}
        er = [edge_rms(e, poses, posd) for e in cedges]
        print(f"  Невязка по связям: медиана {np.median(er):.2f} px, 90% связей ниже {np.percentile(er, 90):.2f} px, "
              f"больше {PRUNE_MIN_PX:.0f} px: {sum(x > PRUNE_MIN_PX for x in er)} из {len(er)}")
        sc_all = [float(np.hypot(M[0, 0], M[1, 0])) for M in poses]
        print(f"  Масштаб кадров: от {min(sc_all):.2f} до {max(sc_all):.2f} (первый = 1.00)")
        if REFINE:
            t2 = time.time()
            poses, st = refine_poses([kpaths[i] for i in best], poses)
            sh = st["shifts"]
            er2 = [edge_rms(e, poses, posd) for e in cedges]
            print(f"  Уточнение по изображению: сдвинуто {st['moved']} из {len(best)} кадров, отклонено {st['rejected']}, "
                  f"без перекрытия {st['free']}; сдвиг медиана {np.median(sh) if sh else 0:.1f} px, макс {max(sh) if sh else 0:.1f} px "
                  f"({time.time() - t2:.0f} с)")
            print(f"  Невязка по связям ORB после уточнения: медиана {np.median(er2):.2f} px (было {np.median(er):.2f})")
        t1 = time.time()
        loc = {n: i for i, n in enumerate(best)}
        ledges = [(loc[j], loc[k], A, B) for j, k, A, B in cedges]
        raw, valid, px = render([kpaths[i] for i in best], poses, ledges, width=rw)
        # Сначала выравниваем плавный свет, затем выполняем обычное улучшение.
        flat = flat_field_correction(raw, valid, px) if FLAT_FIELD else raw
        mosaic = enhance(flat, valid, px) if ENHANCE else flat
        print(f"  Сборка и улучшение: {time.time() - t1:.0f} с")
        name = f"orb_s{START}_part{part}_n{len(best)}"
        save_map(mosaic, valid, name, {"source": "ORB + общая подгонка", "keyframes_used": len(best),
                                       "first_frame": fnum(kpaths[best[0]]), "last_frame": fnum(kpaths[best[-1]]),
                                       "frames_total": len(allp), "rms_px": round(rms, 3),
                                       "render_width": rw, "refined": bool(REFINE)},
                 raw=raw if ENHANCE else None)
        print(f"  Файл: {name}.png, размер {mosaic.shape[1]}x{mosaic.shape[0]}")
        if part == 0:
            diagnose(best, cedges, poses, kpaths)
        results.append((best, poses))
    print(f"Готово за {time.time() - t0:.0f} с.")
    return keys, results


if __name__ == "__main__":
    main()