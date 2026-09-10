import sys
import time
import cv2
import numpy as np
from perception.models.lanes_clrernet import CLRerNetDetector

img = cv2.imread(sys.argv[1])
print("img", img.shape)
det = CLRerNetDetector()
for _ in range(5):
    det.detect(img)
t = time.perf_counter()
N = 30
for _ in range(N):
    lanes = det.detect(img)
print(f"avg {(time.perf_counter() - t) / N * 1000:.1f} ms/frame  (pre {det.pre_ms:.1f} ms, model {det.last_ms:.1f} ms)")
print("lanes:", len(lanes))
for uv, conf in lanes:
    print(f"  conf={conf:.2f} n={len(uv)} bottom={uv[0].round(0).tolist()} top={uv[-1].round(0).tolist()}")
for uv, conf in lanes:
    cv2.polylines(img, [uv.astype(np.int32)], False, (0, 255, 0), 3)
cv2.imwrite("/tmp/lanes_wrapper.jpg", img)
