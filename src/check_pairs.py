import cv2
import glob
import os

BASE = r"C:\Users\79211\Desktop\seabed-modaic"
paths = sorted(glob.glob(os.path.join(BASE, "data", "frames", "*.jpg")))[:30]

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

prev_des = None
for i, p in enumerate(paths):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    k = 480 / img.shape[1]
    img = cv2.resize(img, None, fx=k, fy=k)
    kp, des = orb.detectAndCompute(img, None)
    if prev_des is not None and des is not None:
        pairs = bf.knnMatch(prev_des, des, k=2)
        good = [pr[0] for pr in pairs
                if len(pr) == 2 and pr[0].distance < 0.75 * pr[1].distance]
        print(f"кадры {i-1}->{i}: зацепок {len(kp)}, хороших совпадений {len(good)}")
    prev_des = des