import cv2
import os
import glob
import numpy as np

VIDEO = r"C:\Users\79211\Desktop\seabed-mosaic\data\videous\моё_видео1.mp4"   # поправь имя
OUT = r"C:\Users\79211\Desktop\seabed-mosaic\data\frames"

def extract_frames(video_path, out_dir, step=10, keep_ratio=0.6):
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*.jpg")):
        os.remove(old)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    candidates = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            candidates.append((i, cv2.Laplacian(gray, cv2.CV_64F).var()))
        i += 1

    values = np.array([s for _, s in candidates])
    threshold = keep_ratio * np.median(values)
    print(f"Прочитано: {i}, кандидатов: {len(candidates)}")
    print(f"Резкость: мин {values.min():.1f}, медиана {np.median(values):.1f}, макс {values.max():.1f}")
    print(f"Порог: {threshold:.1f}")

    saved = 0
    for idx, sharp in candidates:
        if sharp >= threshold:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                name = f"frame_{idx:06d}_t{idx / fps:.2f}.jpg"
                cv2.imwrite(os.path.join(out_dir, name), frame)
                saved += 1
    cap.release()
    print(f"Сохранено кадров: {saved}")

if __name__ == "__main__":
    extract_frames(VIDEO, OUT)