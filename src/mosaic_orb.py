import cv2
import glob
import json
import os
import re
import time
from functools import lru_cache
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG = print                                # куда писать сообщения; интерфейс может подставить свою функцию: mosaic_orb.LOG = f


def log(*args, **kwargs):
    LOG(*args, **kwargs)


# ---------- вход и выход ----------
SRC = os.path.join(BASE, "dataset")        # папка со всеми кадрами
OUT = os.path.join(BASE, "output")
START, END = 0, None                       # с какого по порядку кадра брать (END = None: до конца)
MAP_NAME = None                            # название карты: префикс имён выходных файлов (None = "orb_s<START>"); пробелы и знаки вроде / \ : * ? заменяются на _

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

# ---------- замыкание петель (аппарат вернулся на то же место) ----------
CROSS_ENABLE = True                        # False = выключить (для сравнения "с" и "без")
CROSS_MIN_GAP = 12                         # пара кандидат, если номера ключевых кадров различаются больше чем на это
CROSS_RETRIEVE_DESC = 1500                 # сколько дескрипторов брать для грубого поиска похожих кадров
CROSS_TOP = 450                            # сколько лучших по виду пар проверять полным сопоставлением
CROSS_MIN_INLIERS = 40                     # минимум совпадений для связи-петли (строже, чем MIN_INLIERS)
CROSS_ACCEPT_RMS = 15.0                    # связь-петля остаётся, если после общего расчёта её невязка не больше, px рабочего размера
LOOP_FILE = None                           # проверочный режим: путь к colmap_detected_loop_sequence_3.txt (кандидаты из COLMAP)
LOOP_FILE_STEP = 20                        # номеров кадров на одну строку/столбец этого файла

# ---------- траектория аппарата на карте ----------
TRAJ_DRAW = True                           # рисовать путь аппарата: сохраняется отдельная картинка *_traj.png
TRAJ_START_HUE = 95                        # цвет линии плавно меняется от начала к концу пути: оттенок (0-179, OpenCV) в начале...
TRAJ_END_HUE = 8                           # ...и в конце (95 = голубой, 8 = оранжевый)
TRAJ_SMOOTH = 2                            # сглаживание ломаной (0 = без сглаживания, 2-3 = мягко)
TRAJ_ARROW_SPACING = 240                   # расстояние между стрелками направления, px карты
TRAJ_START_TEXT = "старт"                  # подписи (для кириллицы нужен Pillow: pip install pillow; без него будет латиница)
TRAJ_END_TEXT = "конец"

# ---------- навигация как подсказка положениям (необязательно) ----------
NAV_FILE = None                            # файл с координатами аппарата: строки "кадр x y ..." (пробелы, запятые или ;). None = не использовать
NAV_COL_FRAME = 0                          # номер столбца с номером кадра (с нуля)
NAV_COLS = (1, 2)                          # столбцы координат на плоскости дна (для COLMAP из Aqualoc это x и y)
NAV_LAGS = (24, 72)                        # навигация подсказывает смещение между ключевыми кадрами на таком расстоянии друг от друга
NAV_WEIGHT = 1.0                           # 1 = навигация держит так же жёстко, как цепочка ORB; 0.3 = мягко, 3 = жёстко
LOOP_WEIGHT = 3                           # во сколько раз усилить связи-петли в общем расчёте (1 = как есть; пробуйте 2-3)
TRAJ_EXPORT = True                         # сохранять путь аппарата в *_traj.geojson и *_traj.csv (для QGIS и интерфейса)

# ---------- модель положения кадра ----------
FRAME_MODEL = "similarity"                 # "similarity" = поворот+масштаб+сдвиг; "affine" = ещё и неравный масштаб по осям/перекос (наклон камеры)
AFFINE_REG = 3e5                           # насколько жёстко держать аффинную модель близкой к подобию (больше = ближе к "similarity")

# ---------- швы между проходами ----------
EXCLUSIVE_PASSES = True                    # там, где аппарат прошёл второй раз, брать картинку только из первого прохода (нет двойных камней)
PASS_GAP = 12                              # проходом считаем кадры, далёкие по номеру ключевого кадра больше чем на это
FILL_HOLES = True                          # закрывать «дыры», где поиск шва ничего не выбрал, а кадры это место видят
RENDER_STEP = 0.45                         # для сборки брать не каждый ключевой кадр, а через шаг в долях размера кадра (0 = все; 0.4-0.5 = реже и шире полосы)
RENDER_STEP_ANGLE = 0.12                   # ...но и при повороте больше этого угла (рад) кадр тоже берём

# ---------- отладка ----------
DEBUG_LABELS = True                        # сохранять debug_labels.png: карта раскрашена по кадрам-источникам

# ---------- сборка карты ----------
RENDER_WIDTH = 960                         # ширина кадров при сборке (не больше исходной): больше = чётче, но дольше
MAX_SIDE = 4000                            # максимальный размер карты в пикселях
TILE_SIZE = 1024                           # размер блока финальной сборки (меньше = меньше пиковая память)
CACHE_SIZE = 24                            # сколько уменьшенных кадров держать в кеше
REFINE = True                              # уточнять положение кадров по изображению (убирает двоение камней и рыб)
REFINE_SIDE = 3000                         # размер рабочей карты уточнения, px
REFINE_MIN_CC = 0.30                       # минимальная корреляция, чтобы принять уточнение кадра
REFINE_MAX_SHIFT = 20.0                    # максимальный сдвиг кадра при уточнении, px (рабочего размера)
REFINE_CARRY = False                       # переносить сдвиг кадра на все следующие (накапливает ошибку и разрывает петли: оставь False)
REFINE_MIN_OVERLAP = 0.10                  # минимальная доля перекрытия с уже собранным, чтобы уточнять кадр
SELECT_MIN = 0.05                          # пиксель заполнен, если лучший кадр лежит на нём не ближе этой доли к краю (0..1)
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
FLAT_FIELD = True                          # включить коррекцию плавного виньетирования/пятен света
FLAT_SIGMA = 80.0                          # радиус оценки освещения в пикселях готовой карты
FLAT_STRENGTH = 0.55                       # 0..1: сила коррекции
FLAT_GAIN_MIN = 0.70                       # ограничение снизу: не затемнять сильно
FLAT_GAIN_MAX = 1.60                       # ограничение сверху: не вытягивать шум слишком сильно

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


def solve(nodes, edges, size, nav_pairs=None):
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
    for j, k, dx, dy, w in (nav_pairs or []):  # подсказка навигации: смещение между центрами кадров j и k
        if j in pos and k in pos:
            for off, d in ((2, dx), (3, dy)):
                ia, ib = 4 * pos[k] + off, 4 * pos[j] + off
                H[ia, ia] += w
                H[ib, ib] += w
                H[ia, ib] -= w
                H[ib, ia] -= w
                rhs[ia] += w * d
                rhs[ib] -= w * d
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
                    log(f"  Сшивка: куски из {len(ids_a)} и {len(ids_b)} кадров связаны через общее место "
                          f"(подтверждений {len(best)}, макс. совпадений {max(cands[m][0] for m in best)})")
        if not added:
            break
        edges = edges + added
    return edges


# ---------- отбраковка связей, которые спорят с остальными ----------
PRUNE_FACTOR = 2.5                         # выбрасываем, если невязка в столько раз больше медианной
PRUNE_MIN_PX = 5.0                         # ...и больше этого числа пикселей


def solve_affine(nodes, edges, size):
    """Как solve, но положение кадра аффинное (6 чисел): масштаб по осям может быть разным и допустим небольшой перекос,
    так что наклонная камера описывается точнее. От произвольной деформации держит мягкая привязка к подобию (AFFINE_REG),
    масштаб в среднем тянется к 1, как и в обычном solve."""
    pos = {n: i for i, n in enumerate(nodes)}
    m = len(nodes)
    cx, cy = size[0] / 2, size[1] / 2
    N = 6 * m
    H = np.zeros((N, N))
    rhs = np.zeros(N)
    rng = np.random.default_rng(0)
    used = []
    for j, k, A, B in edges:
        if j not in pos or k not in pos:
            continue
        if len(A) > 80:
            sel = rng.choice(len(A), 80, replace=False)
            A, B = A[sel], B[sel]
        u, v = A - (cx, cy), B - (cx, cy)
        n = len(u)
        z, o = np.zeros(n), np.ones(n)
        J = np.zeros((2 * n, 12))
        J[0::2] = np.c_[u[:, 0], u[:, 1], z, z, o, z, -v[:, 0], -v[:, 1], z, z, -o, z]
        J[1::2] = np.c_[z, z, u[:, 0], u[:, 1], z, o, z, z, -v[:, 0], -v[:, 1], z, -o]
        ids = np.r_[6 * pos[j]:6 * pos[j] + 6, 6 * pos[k]:6 * pos[k] + 6]
        H[np.ix_(ids, ids)] += J.T @ J
        used.append((ids, J))
    W = 1e6                                  # первый кадр закреплён
    H[:6, :6] += W * np.eye(6)
    rhs[:6] += W * np.array([1.0, 0, 0, 1.0, 0, 0])
    for i in range(m):                       # мягко: a22 = a11, a12 = -a21 (подобие)
        ia, ib, ic, idd = 6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3
        H[ia, ia] += AFFINE_REG
        H[idd, idd] += AFFINE_REG
        H[ia, idd] -= AFFINE_REG
        H[idd, ia] -= AFFINE_REG
        H[ib, ib] += AFFINE_REG
        H[ic, ic] += AFFINE_REG
        H[ib, ic] += AFFINE_REG
        H[ic, ib] += AFFINE_REG
    p = np.linalg.solve(H, rhs)
    H2, rhs2 = H.copy(), rhs.copy()
    if SCALE_PRIOR > 0:                      # средний масштаб кадра мягко к 1, поворот сохраняем
        w = 2 * SCALE_PRIOR
        for i in range(m):
            ia, ib, ic, idd = 6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3
            s, r = 0.5 * (p[ia] + p[idd]), 0.5 * (p[ic] - p[ib])
            nrm = max(float(np.hypot(s, r)), 1e-6)
            for ids2, coef, target in (((ia, idd), (0.5, 0.5), s / nrm), ((ib, ic), (-0.5, 0.5), r / nrm)):
                for x_, cx_ in zip(ids2, coef):
                    for y_, cy_ in zip(ids2, coef):
                        H2[x_, y_] += w * cx_ * cy_
                    rhs2[x_] += w * cx_ * target
    p = np.linalg.solve(H2, rhs2)
    sq, cnt = 0.0, 0
    for ids, J in used:
        r = J @ p[ids]
        sq += float((r ** 2).sum())
        cnt += len(r)
    rms = (sq / max(cnt, 1)) ** 0.5
    poses = []
    for i in range(m):
        a_, b_, c_, d_, tx, ty = p[6 * i:6 * i + 6]
        poses.append(np.array([[a_, b_, tx - (a_ * cx + b_ * cy)],
                               [c_, d_, ty - (c_ * cx + d_ * cy)]]))
    return poses, rms


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
    log(f"Отбраковано связей: {removed} (осталось {len(edges)})")
    return edges


# ---------- замыкание петель ----------
def retrieve_pairs(best, feats, linked):
    """Грубый поиск кадров, похожих по виду, независимо от текущих положений (при дрейфе положения врут).
    Считаем число совпавших дескрипторов на случайной выборке; лучшие пары потом проверяем полностью."""
    rng = np.random.default_rng(0)
    small = {}
    for n in best:
        d = feats[n][1]
        if d is not None and len(d) >= 30:
            small[n] = d[rng.choice(len(d), min(len(d), CROSS_RETRIEVE_DESC), replace=False)]
    ns = sorted(small)
    out = []
    for a, j in enumerate(ns):
        for k in ns[a + 1:]:
            if k - j <= CROSS_MIN_GAP or (j, k) in linked:
                continue
            pr = bf.knnMatch(small[j], small[k], k=2)
            sc = sum(1 for p in pr if len(p) == 2 and p[0].distance < RATIO * p[1].distance)
            out.append((sc, j, k))
    out.sort(reverse=True)
    return [(j, k) for _, j, k in out[:CROSS_TOP]]


def loop_file_pairs(best, frame_of, linked):
    """Кандидаты из файла COLMAP (только для проверки): ячейка (i, j) = кадры i*STEP и j*STEP смотрят на одно место."""
    try:
        L = np.loadtxt(LOOP_FILE, delimiter=",")
    except (ValueError, OSError) as e:
        log(f"  LOOP_FILE не прочитан ({str(e)[:60]}): нужен файл с матрицей из 0 и 1 через запятую. Беру поиск по виду.")
        return None
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        log("  LOOP_FILE не квадратная матрица. Беру поиск по виду.")
        return None
    fr = np.array([frame_of[n] for n in best])
    pairs = set()
    for i, j in zip(*np.where(np.triu(L, 25) > 0)):
        a = best[int(np.argmin(np.abs(fr - i * LOOP_FILE_STEP)))]
        b = best[int(np.argmin(np.abs(fr - j * LOOP_FILE_STEP)))]
        a, b = min(a, b), max(a, b)
        if b - a > CROSS_MIN_GAP and (a, b) not in linked:
            pairs.add((a, b))
    pairs = sorted(pairs)
    if len(pairs) > CROSS_TOP:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, CROSS_TOP).astype(int)]
    return pairs


def add_loops(best, edges, feats, size, frame_of):
    """Добавляем связи между кадрами, далёкими по времени. Положения тут не используем: после дрейфа они неверны.
    Каждую связь проверяет RANSAC (40+ совпадений, масштаб 0.7-1.4), потом общий расчёт: связи-петли, которые
    не укладываются вместе с остальными, выбрасываем по одной (самую плохую первой)."""
    linked = {(min(j, k), max(j, k)) for j, k, _, _ in edges}
    if LOOP_FILE:
        cand = loop_file_pairs(best, frame_of, linked)
        if cand is not None:
            log(f"  Петли: кандидатов из файла COLMAP {len(cand)}")
    else:
        cand = None
    if cand is None:
        cand = retrieve_pairs(best, feats, linked)
        log(f"  Петли: кандидатов по внешнему виду {len(cand)}")
    new = []
    for j, k in cand:
        r = match(feats[j], feats[k], CROSS_MIN_INLIERS)
        if r is None:
            continue
        s = similarity(r[1], r[0])
        if s is not None and 0.7 < abs(s[0]) < 1.4:
            new.append((j, k, r[0], r[1]))
    log(f"  Петли: прошли проверку RANSAC {len(new)}")
    if not new:
        return edges, 0
    pos = {n: i for i, n in enumerate(best)}
    keep = list(new)
    while keep:
        poses, _ = solve(best, edges + keep, size)
        er = [edge_rms(e, poses, pos) for e in keep]
        w = int(np.argmax(er))
        if er[w] <= CROSS_ACCEPT_RMS:
            break
        keep.pop(w)
    hist = {}
    for e in keep:
        key = (frame_of[e[0]] // 1000 * 1000, frame_of[e[1]] // 1000 * 1000)
        hist[key] = hist.get(key, 0) + 1
    log(f"  Петли: принято {len(keep)} из {len(new)}; по участкам (кадры от-до, тысячи): {hist}")
    return edges + keep * max(1, int(LOOP_WEIGHT)), len(keep)


def load_nav(path):
    """Читает таблицу из текстового файла: числа через пробелы, запятые или ';'; нечисловые строки пропускает."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in re.split(r"[,\s;]+", line) if x != ""])
            except ValueError:
                continue
    if not rows:
        raise ValueError("в файле навигации нет числовых строк")
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], float)


def nav_pairs(best, poses, kpaths, size, nav):
    """Подсказка положениям от навигации. Подгоняем координаты навигации к цепочке ORB (поворот, масштаб, зеркало:
    ось и единицы в файле нам неизвестны), потом берём из навигации смещения между кадрами на расстоянии NAV_LAGS
    ключевых кадров. Абсолютное положение не задаём: у цепочки нет привязки, важны только смещения (они и копят дрейф)."""
    frames = np.array([fnum(kpaths[n]) for n in best], float)
    fn = nav[:, NAV_COL_FRAME]
    ok = np.where((frames >= fn.min()) & (frames <= fn.max()))[0]
    if len(ok) < 2 * min(NAV_LAGS):
        if len(best) >= 2 * min(NAV_LAGS):
            log("  Навигация: номера кадров в файле не совпадают с кадрами карты, подсказка не используется")
        return None                                        # маленький кусок: подгонка ненадёжна, подсказку не берём
    c0, c1 = NAV_COLS
    X = np.c_[np.interp(frames[ok], fn, nav[:, c0]), np.interp(frames[ok], fn, nav[:, c1])]
    cx, cy = size[0] / 2, size[1] / 2
    Y = np.array([_apply(poses[i], np.array([[cx, cy]]))[0] for i in ok])
    fit = None
    for mirror in (False, True):
        Xm = X.copy()
        if mirror:
            Xm[:, 0] = -Xm[:, 0]
        s = similarity(Xm, Y)
        if s is None:
            continue
        z, c = s
        Zc = z * (Xm[:, 0] + 1j * Xm[:, 1]) + c
        rms = float(np.sqrt(np.mean(np.abs(Zc - (Y[:, 0] + 1j * Y[:, 1])) ** 2)))
        if fit is None or rms < fit[0]:
            fit = (rms, mirror, z, Zc)
    if fit is None:
        return None
    rms, mirror, z, Zc = fit
    log(f"  Навигация: подгонка к цепочке ORB: расхождение {rms:.0f} px, масштаб {abs(z):.1f} px на единицу"
          f"{', зеркало' if mirror else ''}")
    pairs = []
    for L in NAV_LAGS:
        for a in range(0, len(ok) - L):
            d = Zc[a + L] - Zc[a]
            pairs.append((best[ok[a]], best[ok[a + L]], float(d.real), float(d.imag), NAV_WEIGHT * 80.0 / L))
    return pairs


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
        if REFINE_CARRY:
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


def render(paths, poses, edges=None, width=None, tag="map"):
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
    log(f"Размер карты: {Wc} x {Hc}, кадры шириной {width} px")

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
    log(f"  Поправки яркости кадров: от {expo.min():.2f} до {expo.max():.2f}")

    def prep(p, idx):
        return (ld(p).astype(np.float32) * gain[..., None] * expo[idx]).astype(np.float32)

    m0 = np.zeros((h, w), np.uint8)
    m0[2:-2, 2:-2] = 255
    d = cv2.distanceTransform(m0, cv2.DIST_L2, 3)
    ok = L > (L.mean() / 2.0)                              # убираем только по-настоящему тёмные углы кадра
    score = ((d / d.max()) * ok).astype(np.float32)        # лучшее место кадра: центр и не тёмный угол
    cover = (m0 > 0).astype(np.float32)
    boxes = [place(M, crn, sc, Wc, Hc, minx, miny) for M in poses_r]
    centers = [sc * (M @ np.array([w / 2.0, h / 2.0, 1.0]) - np.array([minx, miny])) for M in poses_r]   # центры кадров в пикселях карты

    label = None
    try:                                                   # швы ищем на рабочем масштабе (экономит память)
        allp0 = np.concatenate([cv2.transform(crn0[None], M)[0] for M in poses])
        mnx0, mny0 = allp0.min(0)
        mxx0, mxy0 = allp0.max(0)
        ss = SEAM_SCALE
        Ws, Hs = int(np.ceil((mxx0 - mnx0) * ss)) + 1, int(np.ceil((mxy0 - mny0) * ss)) + 1
        good = cv2.resize(((score > SELECT_MIN) * 255).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
        imgs, masks, cors, ids = [], [], [], []
        covered = np.zeros((Hs, Ws), np.uint8)             # что уже занято кадрами, далёкими по времени (другой проход)
        pending, done = {}, -1
        use = set(range(len(paths)))
        if RENDER_STEP > 0:                                # реже кадры = шире полосы между швами, меньше повторяющегося узора на неровном дне
            cen0 = lambda Mx: Mx @ np.array([w0 / 2.0, h0 / 2.0, 1.0])
            use, last = {0}, 0
            for k in range(1, len(paths)):
                dist = float(np.linalg.norm(cen0(poses[k]) - cen0(poses[last])))
                da = np.arctan2(poses[k][1, 0], poses[k][0, 0]) - np.arctan2(poses[last][1, 0], poses[last][0, 0])
                if dist >= RENDER_STEP * min(w0, h0) or abs(float(np.angle(np.exp(1j * da)))) >= RENDER_STEP_ANGLE:
                    use.add(k)
                    last = k
            use.add(len(paths) - 1)
            log(f"  Для швов отобрано кадров: {len(use)} из {len(paths)} (шаг {RENDER_STEP:.2f} размера кадра)")
        for idx, (p, M) in enumerate(zip(paths, poses)):
            if idx not in use:
                continue
            pl = place(M, crn0, ss, Ws, Hs, mnx0, mny0)
            if pl is None:
                continue
            x0, x1, y0, y1, Ml = pl
            osz = (x1 - x0, y1 - y0)
            mk = cv2.warpAffine(good, Ml, osz, flags=cv2.INTER_NEAREST)
            if EXCLUSIVE_PASSES:
                while done < idx - PASS_GAP - 1:           # кадры, ушедшие далеко назад по времени, переходят в «занятое»
                    done += 1
                    if done in pending:
                        ox, oy, om = pending.pop(done)
                        covered[oy:oy + om.shape[0], ox:ox + om.shape[1]][om > 0] = 255
                pending[idx] = (x0, y0, mk.copy())
                mk[covered[y0:y1, x0:x1] > 0] = 0          # второй проход не перекрывает первый, берёт только новые места
                if not mk.any():
                    continue
            small = cv2.resize(prep(p, idx), (w0, h0), interpolation=cv2.INTER_AREA)
            imgs.append(cv2.warpAffine(small, Ml, osz, flags=cv2.INTER_LINEAR))
            masks.append(mk)
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
        log("  Швы: поиск по совпадению соседних кадров")
    except (AttributeError, cv2.error) as e:
        log(f"  Поиск швов недоступен ({e}), беру кадр с лучшим положением")
    def label_by_score():
        """На каждом пикселе берём кадр, у которого это место ближе к центру и не в тёмном углу."""
        best = np.zeros((Hc, Wc), np.float32)
        lab = np.full((Hc, Wc), -1, np.int32)
        for idx, pl in enumerate(boxes):
            if pl is None:
                continue
            x0, x1, y0, y1, Ml = pl
            s = cv2.warpAffine(score, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            bsub, lsub = best[y0:y1, x0:x1], lab[y0:y1, x0:x1]
            better = s > bsub
            bsub[better] = s[better]
            lsub[better] = idx
        lab[best <= SELECT_MIN] = -1
        return lab

    if label is None:
        label = label_by_score()
    elif FILL_HOLES:
        fb = label_by_score()
        holes = (label < 0) & (fb >= 0)
        if holes.any():
            label[holes] = fb[holes]
            log(f"  Закрыто пикселей в дырах карты: {int(holes.sum())}")
        del fb, holes
    valid = label >= 0

    if DEBUG_LABELS:                                       # карта, раскрашенная по кадрам-источникам
        pal = np.random.default_rng(1).integers(40, 255, (len(paths) + 1, 3), dtype=np.uint8)
        dbg = pal[label]
        dbg[label < 0] = 255
        cv2.imwrite(os.path.join(OUT, f"debug_labels_{tag}.png"), dbg)
        del dbg

    # Финальная сборка по блокам: временные накопители ограничены TILE_SIZE.
    mosaic = np.zeros((Hc, Wc, 3), np.uint8)
    tile = max(256, int(TILE_SIZE))
    log(f"  Тайловая сборка: блок {tile}x{tile}")

    prep_cache = {}

    def prep_cached(p, idx):
        if idx not in prep_cache:
            prep_cache[idx] = prep(p, idx)
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
                if not (label[iy0:iy1, ix0:ix1] == idx).any():
                    continue                               # кадр не выбран для этого участка

                out_size = (tw, th)
                Mtile = Ml.copy()
                Mtile[0, 2] += x0 - xb
                Mtile[1, 2] += y0 - yb
                img = prep_cached(p, idx)
                low = cv2.GaussianBlur(img, (0, 0), SPLIT_SIGMA * kk)
                hi_w = cv2.warpAffine(img - low, Mtile, out_size, flags=cv2.INTER_CUBIC)
                lo_w = cv2.warpAffine(low, Mtile, out_size, flags=cv2.INTER_LINEAR)

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
        log(f"  Готово строка тайлов y={yb}:{ye} / {Hc}")

    return mosaic, valid, px, centers


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
    """Исправляет плавную неравномерность освещения по яркости готовой мозаики (масштабная поправка
    по размытой карте света; пустые области в оценке не участвуют)."""
    if not valid.any() or FLAT_STRENGTH <= 0:
        return mosaic

    lab = cv2.cvtColor(mosaic, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    light = masked_blur(L, valid, FLAT_SIGMA * px)
    good = valid & np.isfinite(light) & (light > 1.0)
    if not good.any():
        return mosaic

    target = float(np.median(light[good]))
    gain = target / np.maximum(light, 1.0)
    gain = np.clip(gain, FLAT_GAIN_MIN, FLAT_GAIN_MAX)
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
        log(f"  GeoTIFF: {name}.tif")
    except ValueError as e:
        log(f"  GeoTIFF не записан: {e}")
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def _font(size):
    """Шрифт с кириллицей через Pillow. None, если Pillow или шрифт недоступны."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "arialbd.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return None


def _put_label(img, text, anchor, size):
    """Подпись на тёмной полупрозрачной плашке справа от точки anchor. Возвращает прямоугольник (x0, y0, x1, y1)."""
    H, W = img.shape[:2]
    font = _font(size)
    if font is None:                                       # без Pillow: латиница через OpenCV
        text = {TRAJ_START_TEXT: "start", TRAJ_END_TEXT: "end"}.get(text, text)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, 2)
        x0, y0 = int(anchor[0]), int(anchor[1] - th)
        cv2.putText(img, text, (x0, y0 + th), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, text, (x0, y0 + th), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, (255, 255, 255), 2, cv2.LINE_AA)
        return x0, y0, x0 + tw, y0 + th
    from PIL import Image, ImageDraw
    pad = max(4, size // 3)
    tw = int(font.getlength(text)) if hasattr(font, "getlength") else font.getsize(text)[0]
    w, h = tw + 2 * pad, size + 2 * pad
    x0 = int(min(max(anchor[0], 0), W - w - 1))
    y0 = int(min(max(anchor[1] - h / 2, 0), H - h - 1))
    crop = Image.fromarray(cv2.cvtColor(img[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    box = (0, 0, w - 1, h - 1)
    if hasattr(d, "rounded_rectangle"):
        d.rounded_rectangle(box, radius=h // 3, fill=(20, 24, 28, 190))
    else:
        d.rectangle(box, fill=(20, 24, 28, 190))
    d.text((pad, pad - size // 8), text, font=font, fill=(255, 255, 255, 255))
    out = Image.alpha_composite(crop, layer).convert("RGB")
    img[y0:y0 + h, x0:x0 + w] = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return x0, y0, x0 + w, y0 + h


def _smooth_path(pts, iters):
    """Лёгкое сглаживание ломаной: усреднение соседей и срезание углов (Чайкин). Концы остаются на месте."""
    p = np.asarray(pts, np.float64)
    if iters <= 0 or len(p) < 4:
        return p
    q = p.copy()
    for i in range(1, len(p) - 1):
        q[i] = (p[i - 1] + p[i] + p[i + 1]) / 3.0
    p = q
    for _ in range(iters):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q.append(0.75 * a + 0.25 * b)
            q.append(0.25 * a + 0.75 * b)
        q.append(p[-1])
        p = np.array(q)
    return p


def draw_trajectory(mosaic, valid, centers, out_path):
    """Карта с путём аппарата: плавная линия, цвет меняется от начала к концу, стрелки направления, плашки «старт» и «конец».
    Путь строится по центрам ключевых кадров, между ними он сглажен и не является точной траекторией."""
    if len(centers) < 2:
        return
    img = mosaic.copy()
    mask = np.zeros(valid.shape, np.uint8)
    S = min(img.shape[:2])
    th = max(3, int(round(S / 260)))
    path = _smooth_path(centers, TRAJ_SMOOTH)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    total = max(float(arc[-1]), 1.0)
    ipts = np.rint(path).astype(np.int32)

    halo = img.copy()                                      # мягкая тёмная подложка: линия читается на светлом и тёмном дне
    cv2.polylines(halo, [ipts.reshape(-1, 1, 2)], False, (15, 15, 15), th + 5, cv2.LINE_AA)
    img = cv2.addWeighted(halo, 0.55, img, 0.45, 0)
    cv2.polylines(mask, [ipts.reshape(-1, 1, 2)], False, 255, th + 6)

    def color(t):
        hue = TRAJ_START_HUE + (TRAJ_END_HUE - TRAJ_START_HUE) * t
        return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])

    for i in range(len(path) - 1):                         # линия с плавным переходом цвета
        cv2.line(img, tuple(ipts[i]), tuple(ipts[i + 1]), color(arc[i] / total), th, cv2.LINE_AA)

    tri = th * 3.2                                         # стрелки: белые треугольники вдоль пути с равным шагом
    for s in np.arange(TRAJ_ARROW_SPACING * 0.5, total - tri, TRAJ_ARROW_SPACING):
        i = min(int(np.searchsorted(arc, s)), len(path) - 2)
        d = path[i + 1] - path[i]
        n = float(np.hypot(*d))
        if n < 1e-6:
            continue
        d /= n
        q = path[i] + d * (s - arc[i])
        nrm = np.array([-d[1], d[0]])
        poly = np.array([q + d * tri, q - d * tri * 0.7 + nrm * tri * 0.8, q - d * tri * 0.7 - nrm * tri * 0.8])
        cv2.fillConvexPoly(img, np.rint(poly).astype(np.int32), (255, 255, 255), cv2.LINE_AA)
        cv2.polylines(img, [np.rint(poly).astype(np.int32).reshape(-1, 1, 2)], True, (30, 30, 30), 1, cv2.LINE_AA)

    r = th * 3
    size = max(22, int(S / 38))
    for p, hue_t, txt in ((path[0], 0.0, TRAJ_START_TEXT), (path[-1], 1.0, TRAJ_END_TEXT)):
        c = (int(p[0]), int(p[1]))
        cv2.circle(img, c, r + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, r, color(hue_t), -1, cv2.LINE_AA)
        x0, y0, x1, y1 = _put_label(img, txt, (c[0] + r + 8, c[1]), size)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        cv2.circle(mask, c, r + 4, 255, -1)

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = np.where(valid | (mask > 0), 255, 0).astype(np.uint8)
    cv2.imwrite(out_path, bgra)


def save_trajectory(name, frames, centers):
    """Путь аппарата в координатах карты (те же, что у GeoTIFF: левый верхний угол GEO_ORIGIN, Y вверх, пиксель = GEO_MPP).
    Без GEO_MPP это пиксели карты. GeoJSON открывается в QGIS поверх .tif, CSV читает любой интерфейс."""
    mpp = float(GEO_MPP) if GEO_MPP else 1.0
    x0, y0 = GEO_ORIGIN
    pts = [[round(x0 + float(c[0]) * mpp, 3), round(y0 - float(c[1]) * mpp, 3)] for c in centers]
    gj = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {"name": "trajectory", "keyframes": len(pts)},
                        "geometry": {"type": "LineString", "coordinates": pts}}] +
                      [{"type": "Feature", "properties": {"frame": int(f)}, "geometry": {"type": "Point", "coordinates": p}}
                       for f, p in zip(frames, pts)]}
    if GEO_EPSG:
        gj["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{int(GEO_EPSG)}"}}
    else:
        gj["units"] = "px (локальные координаты, без привязки)"
    with open(os.path.join(OUT, name + "_traj.geojson"), "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)
    with open(os.path.join(OUT, name + "_traj.csv"), "w", encoding="utf-8") as f:
        f.write("frame,x,y,map_px_x,map_px_y\n")
        for fr_, p, c in zip(frames, pts, centers):
            f.write(f"{int(fr_)},{p[0]},{p[1]},{float(c[0]):.1f},{float(c[1]):.1f}\n")


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
    log("Самые слабые связи (мало совпадений):")
    for rms, n, a, b, *_ in sorted(rows, key=lambda x: x[1])[:top]:
        log(f"  кадры {a} -> {b}: совпадений {n}, невязка {rms:.1f} px")

    log("Самые большие невязки:")
    worst = sorted(rows, key=lambda x: -x[0])[:top]
    for rms, n, a, b, j, k, A, B in worst:
        log(f"  кадры {a} -> {b}: совпадений {n}, невязка {rms:.1f} px")
    for rms, n, a, b, j, k, A, B in worst[:3]:                  # картинки для 3 худших связей
        draw_link(kpaths[j], kpaths[k], A, B, os.path.join(OUT, f"link_{a}_{b}.png"))
        log(f"  картинка: link_{a}_{b}.png")


def dump_poses(path, nodes, poses, kpaths, size):
    """Центры кадров на карте (px рабочего размера) для geo_check.py."""
    cx, cy = size[0] / 2, size[1] / 2
    rows = []
    for n, M in zip(nodes, poses):
        c = M @ np.array([cx, cy, 1.0])
        rows.append({"frame": fnum(kpaths[n]), "cx": float(c[0]), "cy": float(c[1]),
                     "scale": float(np.hypot(M[0, 0], M[1, 0]))})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def cycle_check(edges, size, max_triplets=4000):
    """Решающая проверка модели положения кадра. Берём тройки кадров i<j<k, у которых есть все три связи (i,j), (j,k), (i,k).
    Если собрать связи (i,j) и (j,k) в цепочку, получится положение k относительно i; оно должно совпасть с прямой связью (i,k).
    Расхождение считаем по углам кадра в пикселях. Делаем это для трёх моделей: поворот+масштаб+сдвиг (то, что мы используем),
    аффинной и перспективной. Если перспективная/аффинная сходится заметно лучше, модель кадра надо усложнять;
    если все три одинаково плохи, дело в рельефе дна (параллакс), и 2D-модель тут не поможет."""
    def to3(M):
        M = np.asarray(M, np.float64)
        return np.vstack([M, [0, 0, 1]]) if M.shape == (2, 3) else M

    fits = {}
    for j, k, A, B in edges:
        if (j, k) in fits or len(A) < 8:
            continue
        try:
            s, _ = cv2.estimateAffinePartial2D(B, A, method=cv2.LMEDS)
            af, _ = cv2.estimateAffine2D(B, A, method=cv2.LMEDS)
            hm, _ = cv2.findHomography(B, A, cv2.LMEDS)
        except cv2.error:
            continue
        if s is None or af is None or hm is None:
            continue
        fits[(j, k)] = (to3(s), to3(af), to3(hm))
    w, h = size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64).T

    def corner_err(P, Q):
        a, b = P @ corners, Q @ corners
        a, b = a[:2] / a[2], b[:2] / b[2]
        return float(np.mean(np.linalg.norm(a - b, axis=0)))

    errs = [[], [], []]
    nodes = sorted({n for e in fits for n in e})
    succ = {}
    for (j, k) in fits:
        succ.setdefault(j, []).append(k)
    cnt = 0
    for i in nodes:
        for j in succ.get(i, []):
            for k in succ.get(j, []):
                if (i, k) in fits and k - i <= 3 * NEIGHBORS:
                    for m_ in range(3):
                        errs[m_].append(corner_err(fits[(i, j)][m_] @ fits[(j, k)][m_], fits[(i, k)][m_]))
                    cnt += 1
                    if cnt >= max_triplets:
                        break
            if cnt >= max_triplets:
                break
        if cnt >= max_triplets:
            break
    if not cnt:
        log("  Проверка модели кадра: троек связей не нашлось")
        return
    names = ("поворот+масштаб+сдвиг (сейчас)", "аффинная", "перспективная")
    log(f"  Проверка модели кадра по {cnt} тройкам связей (расхождение цепочки и прямой связи по углам кадра, px):")
    for nm, e in zip(names, errs):
        log(f"    {nm}: медиана {np.median(e):.1f}, 90% ниже {np.percentile(e, 90):.1f}")


# ---------- настройки из файла config.json (для интерфейса и для запуска из командной строки) ----------
CONFIG_PATH = os.path.join(BASE, "config.json")            # по умолчанию лежит в корне проекта, рядом с папкой src
SCHEMA_PATH = os.path.join(BASE, "params_schema.json")     # описание параметров для интерфейса: подписи, границы, значения (необязателен)
NOT_CONFIGURABLE = {"BASE", "CACHE_SIZE", "RATIO", "CONFIG_PATH", "SCHEMA_PATH"}
PATH_PARAMS = ("SRC", "OUT", "LOOP_FILE", "NAV_FILE")


class ConfigError(Exception):
    """Ошибки в настройках. Список понятных сообщений лежит в .problems (интерфейс может показать их пользователю)."""
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("Ошибки в настройках:\n" + "\n".join("  - " + p for p in self.problems))


def _snapshot_defaults():
    ok = (bool, int, float, str, type(None), tuple)
    return {k: v for k, v in globals().items() if k.isupper() and k not in NOT_CONFIGURABLE and isinstance(v, ok)}


DEFAULTS = _snapshot_defaults()                            # значения, заданные в коде выше


def _load_schema():
    try:
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            return {p["key"]: p for p in json.load(f)["params"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def read_config(path):
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except OSError as e:
        raise ConfigError([f"файл настроек не открывается: {path} ({e.strerror or e})"])
    except ValueError as e:
        raise ConfigError([f"файл настроек не разобран как JSON: {path} ({e})"])
    if not isinstance(cfg, dict):
        raise ConfigError([f"в файле настроек должен быть словарь вида {{\"параметр\": значение}}: {path}"])
    return cfg


def check_loop_file(path):
    """None, если файл похож на матрицу петель (квадратная таблица чисел через запятую), иначе понятное сообщение."""
    try:
        L = np.loadtxt(path, delimiter=",")
    except (ValueError, OSError):
        return "это не матрица петель: ожидается таблица из нулей и единиц через запятую (возможно, выбран файл траектории)"
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 10:
        return "это не квадратная матрица петель"
    return None


def check_nav_file(path):
    try:
        nav = load_nav(path)
    except (ValueError, OSError) as e:
        return f"файл навигации не прочитан ({e})"
    if nav.shape[0] < 10 or nav.shape[1] < 3:
        return "в файле навигации нужно не меньше 10 строк и 3 столбцов: кадр, x, y"
    if not np.all(np.diff(nav[:, NAV_COL_FRAME]) > 0):
        return "первый столбец должен быть номером кадра по возрастанию (похоже, это не файл навигации)"
    return None


def apply_config(cfg):
    """Сначала возвращает все параметры к значениям по умолчанию, потом применяет cfg (словарь параметров).
    Значения проверяются; при ошибках ничего не применяется и поднимается ConfigError. Возвращает список предупреждений.
    Относительные пути считаются от корня проекта."""
    schema = _load_schema()
    g = globals()
    for k, v in DEFAULTS.items():
        g[k] = v
    errors, warnings, new = [], [], {}
    for key, val in (cfg or {}).items():
        if key.startswith("_"):                            # "_comment" и подобное пропускаем
            continue
        if key not in DEFAULTS:
            warnings.append(f"неизвестный параметр «{key}» пропущен")
            continue
        d, sp = DEFAULTS[key], schema.get(key, {})
        name = f"{sp.get('label', key)} ({key})"
        if key in PATH_PARAMS:
            if val in (None, ""):
                if key in ("SRC", "OUT"):
                    errors.append(f"{name}: путь не может быть пустым")
                else:
                    new[key] = None
                continue
            if not isinstance(val, str):
                errors.append(f"{name}: нужен путь в виде текста")
                continue
            p = val if os.path.isabs(val) else os.path.join(BASE, val)
            if key == "SRC" and not os.path.isdir(p):
                errors.append(f"{name}: папка не найдена: {p}")
                continue
            if key in ("LOOP_FILE", "NAV_FILE"):
                if not os.path.isfile(p):
                    errors.append(f"{name}: файл не найден: {p}")
                    continue
                msg = check_loop_file(p) if key == "LOOP_FILE" else check_nav_file(p)
                if msg:
                    errors.append(f"{name}: {msg}")
                    continue
            new[key] = p
            continue
        is_num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
        if isinstance(d, bool):
            if not isinstance(val, bool):
                errors.append(f"{name}: нужно да/нет (true или false)")
                continue
        elif isinstance(d, tuple):
            need = None if key == "NAV_LAGS" else len(d)
            if not isinstance(val, (list, tuple)) or not val or (need and len(val) != need) or not all(is_num(x) for x in val):
                errors.append(f"{name}: нужен набор из {need or 'нескольких'} чисел")
                continue
            val = tuple(val)
        elif isinstance(d, str):
            if not isinstance(val, str):
                errors.append(f"{name}: нужен текст")
                continue
            if sp.get("choices") and val not in sp["choices"]:
                errors.append(f"{name}: допустимы только значения {sp['choices']}")
                continue
        else:                                              # число или пусто
            if val is None:
                if d is not None and not sp.get("nullable"):
                    errors.append(f"{name}: значение не может быть пустым")
                    continue
            elif key == "MAP_NAME":
                val = str(val)
            elif not is_num(val):
                errors.append(f"{name}: нужно число")
                continue
            else:
                if sp.get("type") == "int":
                    if isinstance(val, float) and not val.is_integer():
                        errors.append(f"{name}: нужно целое число")
                        continue
                    val = int(val)
                if sp.get("min") is not None and val < sp["min"]:
                    errors.append(f"{name}: не меньше {sp['min']}")
                    continue
                if sp.get("max") is not None and val > sp["max"]:
                    errors.append(f"{name}: не больше {sp['max']}")
                    continue
        if key == "GEO_EPSG" and val is not None and 4000 <= int(val) < 5000:
            errors.append(f"{name}: нужна проекция в метрах (например UTM 326xx или 327xx), а не градусы")
            continue
        new[key] = val
    if new.get("END") is not None and new.get("START", DEFAULTS["START"]) >= new["END"]:
        errors.append("Диапазон кадров: первый кадр должен быть меньше последнего")
    if errors:
        raise ConfigError(errors)
    g.update(new)
    return warnings


def main(cfg=None):
    """cfg: словарь параметров (интерфейс передаёт его сам). Если не задан, берётся config.json из корня проекта (если он есть)."""
    if cfg is None:
        cfg = read_config(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else {}
    for w in apply_config(cfg):
        log("  Настройки: " + w)
    os.makedirs(OUT, exist_ok=True)
    allp = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        allp += glob.glob(os.path.join(SRC, ext))
    allp = sorted(allp, key=natural_key)[START:END]
    if not allp:
        raise SystemExit(f"Кадров не найдено в {SRC}. Проверь путь и что в папке лежат png/jpg.")
    log("Кадров в папке:", len(allp))
    t0 = time.time()

    keys = select_keyframes(allp)
    log(f"Ключевых кадров: {len(keys)} из {len(allp)} ({time.time() - t0:.0f} с)")
    kpaths = [allp[i] for i in keys]

    feats = [features(load(p)) for p in kpaths]
    edges = []
    for k in range(1, len(kpaths)):
        for j in range(max(0, k - NEIGHBORS), k):
            r = match(feats[j], feats[k])
            if r is not None:
                edges.append((j, k, r[0], r[1]))
    log("Связей между кадрами:", len(edges))

    linked = {k for j, k, _, _ in edges}
    breaks = [fnum(kpaths[k]) for k in range(1, len(keys)) if k not in linked]
    log("Цепочка рвётся перед кадрами №:", breaks)

    size = (load(kpaths[0]).shape[1], load(kpaths[0]).shape[0])
    comps = components(len(kpaths), edges)
    log("Кусков (связных групп):", len(comps), ", размеры:", [len(c) for c in comps[:8]])
    if LOOP_SAMPLE and len(comps) > 1:
        t3 = time.time()
        edges = link_pieces(feats, edges, size)
        comps = components(len(kpaths), edges)
        log(f"После сшивки кусков между собой: {len(comps)} шт., размеры: {[len(c) for c in comps[:8]]} ({time.time() - t3:.0f} с)")
    orig_w = cv2.imread(allp[0]).shape[1]
    rw = int(min(max(RENDER_WIDTH or WIDTH, WIDTH), orig_w))
    log(f"Ширина исходных кадров: {orig_w} px; карта собирается из кадров шириной {rw} px")
    results = []
    for part, comp in enumerate(comps[:MAX_PARTS]):
        if len(comp) < 5:
            break
        best = sorted(comp)
        cedges = prune_links(best, edges, size)
        poses, rms = solve(best, cedges, size)
        if CROSS_ENABLE:
            cedges, n_new = add_loops(best, cedges, feats, size, [fnum(p) for p in kpaths])
            if n_new:
                poses, rms = solve(best, cedges, size)
        if NAV_FILE:
            try:
                nav = load_nav(NAV_FILE)
            except (OSError, ValueError) as e:
                nav = None
                log(f"  Навигация не прочитана ({str(e)[:70]}), подсказка не используется")
            if nav is not None:
                npairs = nav_pairs(best, poses, kpaths, size, nav)
                if npairs:
                    cc = np.array([[size[0] / 2, size[1] / 2]])
                    before = [_apply(M, cc)[0] for M in poses]
                    poses, rms = solve(best, cedges, size, npairs)
                    mv = [float(np.linalg.norm(_apply(M, cc)[0] - b0)) for M, b0 in zip(poses, before)]
                    log(f"  Навигация учтена: положения сдвинулись на {np.median(mv):.0f} px (медиана), до {max(mv):.0f} px")
        if FRAME_MODEL == "affine":
            try:
                pa, ra = solve_affine(best, cedges, size)
                posd_a = {n: k_ for k_, n in enumerate(best)}
                e_s = float(np.median([edge_rms(e, poses, posd_a) for e in cedges]))
                e_a = float(np.median([edge_rms(e, pa, posd_a) for e in cedges]))
                sv = [np.linalg.svd(M[:, :2], compute_uv=False) for M in pa]
                aniso = max(float(s_[0] / s_[1]) for s_ in sv)
                ok_a = np.isfinite(e_a) and aniso < 1.25
                log(f"  Аффинная модель кадров: невязка по связям медиана {e_a:.2f} px (подобие: {e_s:.2f} px), "
                      f"наибольшая неравномерность масштаба по осям {aniso:.2f}" + ("" if ok_a else "; отклонена, оставляю подобие"))
                if ok_a:
                    poses, rms = pa, ra
            except np.linalg.LinAlgError:
                log("  Аффинная модель не рассчиталась, оставляю подобие")
        poses_pre = poses                                  # положения ДО уточнения по изображению (для диагностики)
        dump_poses(os.path.join(OUT, f"poses_part{part}.json"), best, poses_pre, kpaths, size)
        posd0 = {n: i for i, n in enumerate(best)}
        lo = [edge_rms(e, poses, posd0) for e in cedges if e[1] - e[0] > CROSS_MIN_GAP]
        ch = [edge_rms(e, poses, posd0) for e in cedges if e[1] - e[0] <= CROSS_MIN_GAP]
        if lo and ch:
            log(f"  Невязка связей-петель: медиана {np.median(lo):.1f} px (обычных связей {np.median(ch):.1f} px), "
                  f"петель {len({(e[0], e[1]) for e in cedges if e[1] - e[0] > CROSS_MIN_GAP})}")
        log(f"Кусок {part}: ключевых кадров {len(best)}, "
              f"кадры №{fnum(kpaths[best[0]])}-{fnum(kpaths[best[-1]])}, невязка {rms:.2f} px")
        posd = {n: i for i, n in enumerate(best)}
        er = [edge_rms(e, poses, posd) for e in cedges]
        log(f"  Невязка по связям: медиана {np.median(er):.2f} px, 90% связей ниже {np.percentile(er, 90):.2f} px, "
              f"больше {PRUNE_MIN_PX:.0f} px: {sum(x > PRUNE_MIN_PX for x in er)} из {len(er)}")
        sc_all = [float(np.hypot(M[0, 0], M[1, 0])) for M in poses]
        log(f"  Масштаб кадров: от {min(sc_all):.2f} до {max(sc_all):.2f} (первый = 1.00)")
        if part == 0:
            diagnose(best, cedges, poses_pre, kpaths)      # невязки общего расчёта, а не после refine
            cycle_check(cedges, size)
        if REFINE:
            t2 = time.time()
            poses, st = refine_poses([kpaths[i] for i in best], poses)
            sh = st["shifts"]
            er2 = [edge_rms(e, poses, posd) for e in cedges]
            log(f"  Уточнение по изображению: сдвинуто {st['moved']} из {len(best)} кадров, отклонено {st['rejected']}, "
                  f"без перекрытия {st['free']}; сдвиг медиана {np.median(sh) if sh else 0:.1f} px, макс {max(sh) if sh else 0:.1f} px "
                  f"({time.time() - t2:.0f} с)")
            log(f"  Невязка по связям ORB после уточнения: медиана {np.median(er2):.2f} px (было {np.median(er):.2f})")
        dump_poses(os.path.join(OUT, f"poses_refined_part{part}.json"), best, poses, kpaths, size)
        t1 = time.time()
        loc = {n: i for i, n in enumerate(best)}
        ledges = [(loc[j], loc[k], A, B) for j, k, A, B in cedges]
        raw, valid, px, centers = render([kpaths[i] for i in best], poses, ledges, width=rw, tag=f"part{part}")
        # Сначала выравниваем плавный свет, затем выполняем обычное улучшение.
        flat = flat_field_correction(raw, valid, px) if FLAT_FIELD else raw
        mosaic = enhance(flat, valid, px) if ENHANCE else flat
        log(f"  Сборка и улучшение: {time.time() - t1:.0f} с")
        base = re.sub(r'[\\/:*?"<>|\s]+', "_", str(MAP_NAME).strip()).strip("_") if MAP_NAME else ""
        name = f"{base or f'orb_s{START}'}_part{part}_n{len(best)}"
        save_map(mosaic, valid, name, {"source": "ORB + общая подгонка", "keyframes_used": len(best),
                                       "first_frame": fnum(kpaths[best[0]]), "last_frame": fnum(kpaths[best[-1]]),
                                       "frames_total": len(allp), "rms_px": round(rms, 3),
                                       "title": str(MAP_NAME) if MAP_NAME else None,
                                       "render_width": rw, "refined": bool(REFINE),
                                       "crossovers": bool(CROSS_ENABLE),
                                       "trajectory_px": [{"frame": fnum(kpaths[n]), "x": round(float(c[0]), 1), "y": round(float(c[1]), 1)}
                                                         for n, c in zip(best, centers)]},
                 raw=raw if ENHANCE else None)
        log(f"  Файл: {name}.png, размер {mosaic.shape[1]}x{mosaic.shape[0]}")
        if TRAJ_DRAW:
            draw_trajectory(mosaic, valid, centers, os.path.join(OUT, name + "_traj.png"))
            log(f"  Путь аппарата: {name}_traj.png")
        if TRAJ_EXPORT:
            save_trajectory(name, [fnum(kpaths[n]) for n in best], centers)
            log(f"  Путь аппарата: {name}_traj.geojson, {name}_traj.csv")
        results.append((best, poses))
    log(f"Готово за {time.time() - t0:.0f} с.")
    return keys, results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Сборка мозаики морского дна из кадров")
    ap.add_argument("--config", help="путь к файлу настроек (по умолчанию config.json в корне проекта)")
    args = ap.parse_args()
    try:
        main(read_config(args.config) if args.config else None)
    except ConfigError as e:
        raise SystemExit(str(e))