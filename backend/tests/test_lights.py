import sys, time, cv2
from perception.models.lights_openlenda import OpenLendaDetector
det = OpenLendaDetector()
img = cv2.imread(sys.argv[1])
for _ in range(5): det.detect(img)
t = time.perf_counter(); N = 50
for _ in range(N): hs = det.detect(img)
print(f"avg {(time.perf_counter()-t)/N*1000:.1f} ms/frame (incl. pre/post), model {det.last_ms:.1f} ms")
for h in hs:
    print(h["bbox"], h["orientation"], [(l["label"], round(l["conf"],2)) for l in h["lamps"]])
