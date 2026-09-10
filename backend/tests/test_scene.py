import sys, time, cv2, numpy as np
from perception.models.scene_mapanything import MapAnythingScene
cap = cv2.VideoCapture(sys.argv[1]); frames = []
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i in (0, 5, 10): frames.append(f)
    i += 1
print("frames", len(frames), frames[0].shape)
sc = MapAnythingScene()
for n in (1, 2):
    for _ in range(2): res = sc.infer(frames[:n])
    ts = []
    for _ in range(5): res = sc.infer(frames[:n]); ts.append(sc.last_ms)
    r0 = res[0]; size, ratio, scale = r0["size"], r0["ratio"], r0["scale"]
    print(f"views={n} model {np.mean(ts):.0f} ms; depth {size} ratio {ratio} scale {scale:.3f}")
    print("  K=", r0["K"].round(1).tolist())
    d = r0["depth"]; m = r0["mask"]
    print("  depth center/median:", d[size[0]//2, size[1]//2].round(2), np.median(d[m]).round(2), "valid%", m.mean().round(2))
    if n == 2:
        T = res[1]["cam2world"]
        print("  view1 cam2world t=", T[:3, 3].round(3).tolist(), "R diag", np.diag(T[:3,:3]).round(4).tolist())
