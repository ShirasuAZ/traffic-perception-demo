from perception.calib import Calib
from perception.fusion.lights import LightFusion

c = Calib(fx=1500, fy=1500, cx=960, cy=540, width=1920, height=1080, cam_height_m=1.5)
fu = LightFusion(c)


def housing(color, arrows=(), x=900, w=30):
    lamps = [{"label": color, "conf": 0.8}] + [{"label": a, "conf": 0.7} for a in arrows]
    return {"bbox": [x, 300, x + w, 300 + w * 0.4], "orientation": "horizontal", "conf": 0.8, "lamps": lamps}


timeline = (
    [("red", ("left",))] * 20      # red + green left arrow
    + [("green", ())] * 20         # legal: red -> green
    + [("red", ())] * 6            # illegal green -> red (missed yellow), short glitch: should be rejected
    + [("green", ())] * 10
    + [("red", ())] * 15           # illegal but persistent: forced after 10 frames
)
for i, (col, arr) in enumerate(timeline):
    ts = i / 30
    hs = [housing(col, arr, x=900 + i * 0.5, w=30 + i * 0.1)]
    out = fu.update(hs, ts)
    if i in (2, 5, 19, 22, 25, 42, 45, 50, 55, 62, 66, 70):
        for o in out:
            lamps = ", ".join(f"{l['shape']}={l['color']}{'(F)' if l['forced'] else ''}" for l in o["lamps"])
            print(f"f{i:02d} obs={col}{list(arr)} -> id={o['track_id']} [{lamps}] dist={o['distance_m']}m dconf={o['distance_confidence']} pos={o['position_ego']}")
# coasting / drop
for j in range(30):
    out = fu.update([], (len(timeline) + j) / 30)
    if j in (0, 5, 25):
        print(f"coast+{j}: tracks={len(fu.tracks)} published={len(out)} coasting={[o['coasting'] for o in out]}")
