"""Make a synthetic 'approach' clip from one still: a slowly shrinking crop resized to 1920x1080.
Only for pipeline smoke tests; not a substitute for real footage."""
import sys
import cv2
import numpy as np

src, out = sys.argv[1], sys.argv[2]
n, fps = int(sys.argv[3]) if len(sys.argv) > 3 else 240, 30
img = cv2.imread(src)
H, W = img.shape[:2]
wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
for i in range(n):
    z = 1.0 + 0.3 * i / n                     # zoom 1.0 -> 1.3
    cw, ch = int(W / z), int(W / z * 1080 / 1920)
    cx, cy = W // 2, int(H * 0.55)
    x0, y0 = max(0, cx - cw // 2), max(0, min(H - ch, cy - ch // 2))
    crop = img[y0:y0 + ch, x0:x0 + cw]
    frame = cv2.resize(crop, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    noise = np.random.default_rng(i).normal(0, 2, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    wr.write(frame)
wr.release()
print("wrote", out, n, "frames")
