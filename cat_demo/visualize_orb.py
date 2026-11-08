import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
IMG1 = os.path.join(BASE, "cat1.jpg")
IMG2 = os.path.join(BASE, "cat2.jpg")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

WIDTH = 480
RATIO = 0.75
MIN_INLIERS = 25
RANSAC_THR = 3.0
MAX_MATCHES = 80
MAX_KP_DRAW = 100

LINE_INLIER_THICK = 1
DOT_RADIUS = 2
DOT_THICK = -1
HUE_START = 70
HUE_END = 110
SATURATION = 180
VALUE = 210

KP1_RADIUS = 4
KP2_RADIUS = 4
KP1_COLOR = (60, 200, 60)
KP2_COLOR = (60, 60, 220)


def load(path, width):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать: {path}")
    if img.shape[1] == width:
        return img
    k = width / img.shape[1]
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


img1 = load(IMG1, WIDTH)
img2 = load(IMG2, WIDTH)


def preprocess(img, k=12, gain=25):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(g, (0, 0), k)
    std = np.sqrt(cv2.GaussianBlur((g - mean) ** 2, (0, 0), k)) + 2
    return np.clip(128 + gain * (g - mean) / std, 0, 255).astype(np.uint8)


gray1 = preprocess(img1)
gray2 = preprocess(img2)

detector = cv2.ORB_create(4000, fastThreshold=5)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

kp1, des1 = detector.detectAndCompute(gray1, None)
kp2, des2 = detector.detectAndCompute(gray2, None)
print(f"Точек на фото 1: {len(kp1)}")
print(f"Точек на фото 2: {len(kp2)}")


def match_orb(des1, des2):
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return [], np.zeros(0, dtype=bool), None

    pairs = [p for p in bf.knnMatch(des1, des2, k=2) if len(p) == 2]
    good = [m for m, n in pairs if m.distance < RATIO * n.distance]
    print(f"  Прошло ratio-фильтр: {len(good)}")

    if len(good) < 8:
        return good, np.zeros(len(good), dtype=bool), None

    A = np.float32([kp1[m.queryIdx].pt for m in good])
    B = np.float32([kp2[m.trainIdx].pt for m in good])

    M, inl = cv2.estimateAffinePartial2D(
        B, A, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THR
    )
    if M is None or inl is None:
        return good, np.zeros(len(good), dtype=bool), None

    inl = inl.ravel().astype(bool)
    print(f"  Inliers после RANSAC: {int(inl.sum())}")
    return good, inl, M


matches, mask, matrix = match_orb(des1, des2)
inliers_count = int(mask.sum())
accepted = inliers_count >= MIN_INLIERS


def hsv_to_bgr(h, s=SATURATION, v=VALUE):
    px = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def draw_matches_clean(kp1, kp2, matches, mask, max_draw=MAX_MATCHES):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    inliers = [m for m, ok in zip(matches, mask) if ok][:max_draw]
    if not inliers:
        return canvas

    n = len(inliers)
    for i, m in enumerate(inliers):
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        x2 += w1

        t = i / max(n - 1, 1)
        hue = int(HUE_START + (HUE_END - HUE_START) * t) % 180
        color = hsv_to_bgr(hue)

        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, LINE_INLIER_THICK, cv2.LINE_AA)
        cv2.circle(canvas, (int(x1), int(y1)), DOT_RADIUS, color, DOT_THICK, cv2.LINE_AA)
        cv2.circle(canvas, (int(x2), int(y2)), DOT_RADIUS, color, DOT_THICK, cv2.LINE_AA)

    return canvas


def draw_keypoints(img, kps, color, radius, max_draw=MAX_KP_DRAW):
    out = img.copy()
    top = sorted(kps, key=lambda kp: kp.response, reverse=True)[:max_draw]
    for kp in top:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(out, (x, y), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
    return out


def create_overlay(matrix, img1, img2):
    h, w = img1.shape[:2]

    if matrix is None:
        overlay = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
        return overlay

    warped = cv2.warpAffine(img2, matrix, (w, h))

    mask2 = np.ones_like(img2) * 255
    warped_mask = cv2.warpAffine(mask2, matrix, (w, h))
    _, bin_mask = cv2.threshold(cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

    base = cv2.addWeighted(img1, 0.5, np.zeros_like(img1), 0, 0)

    blended = cv2.addWeighted(base, 0.4, warped, 0.6, 0)
    base[bin_mask > 0] = blended[bin_mask > 0]

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.drawContours(base, [box], 0, (200, 200, 200), 1, cv2.LINE_AA)

        for pt in box:
            cv2.drawMarker(base, tuple(pt), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=10, thickness=1, line_type=cv2.LINE_AA)

        label = "Совмещение: Фото 1 (база) + Фото 2 (сопоставлено)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.55, 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

        x0, y0 = 10, 10
        overlay_box = base.copy()
        cv2.rectangle(overlay_box, (x0, y0), (x0 + tw + 16, y0 + th + baseline + 10), (20, 20, 20), -1)
        cv2.addWeighted(overlay_box, 0.6, base, 0.4, 0, base)

        cv2.putText(base, label, (x0 + 8, y0 + th + 4),
                    font, scale, (240, 240, 240), thick, cv2.LINE_AA)

    return base


vis1 = draw_keypoints(img1, kp1, KP1_COLOR, KP1_RADIUS)
vis2 = draw_keypoints(img2, kp2, KP2_COLOR, KP2_RADIUS)
pair = np.vstack([vis1, vis2])
cv2.imwrite(os.path.join(OUT, "orb_01_points.png"), pair)

match_img = draw_matches_clean(kp1, kp2, matches, mask)
cv2.imwrite(os.path.join(OUT, "orb_02_matches.png"), match_img)

overlay_img = create_overlay(matrix, img1, img2)
cv2.imwrite(os.path.join(OUT, "orb_03_overlay.png"), overlay_img)

print("\n========== ИТОГИ (ORB) ==========")
print(f"Ключевых точек: {len(kp1)} / {len(kp2)}")
print(f"Совпадений после ratio: {len(matches)}")
print(f"Надёжных (RANSAC): {inliers_count}")
print(f"Порог MIN_INLIERS: {MIN_INLIERS}")

if accepted:
    print("Вердикт: ОК — связь принята")
    print("Матрица положения (фото 2 -> фото 1):")
    for row in matrix:
        print(f"  {row[0]:.4f}  {row[1]:.4f}  {row[2]:.2f}")
else:
    print("Вердикт: слабо — связь отклонена")
    print("Совмещение наложено без выравнивания.")

print(f"\nРезультаты сохранены в: {OUT}")