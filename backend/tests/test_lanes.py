import numpy as np
from perception.calib import Calib
from perception.fusion.lanes import LaneFusion

c = Calib(fx=1000, fy=1000, cx=960, cy=540, width=1920, height=1080, cam_height_m=1.5, pitch_deg=2.0)
fu = LaneFusion(c)
rng = np.random.default_rng(0)


def frame(t):
    lanes = []
    for y0 in (-5.4, -1.8, 1.8, 5.4):
        xs = np.linspace(4, 45, 20)
        ys = y0 + 0.002 * xs ** 2 + 0.05 * np.sin(t)
        uv, _ = c.ego_to_pixel(np.stack([xs, ys, np.zeros_like(xs)], 1))
        uv += rng.normal(0, 1.5, uv.shape)
        if not (2.0 < t < 2.3 and y0 == 1.8):  # occlusion of ego-left
            lanes.append((uv, 0.9))
    return lanes


for i in range(90):
    t = i / 30
    out = fu.update(frame(t), t)
    if i in (0, 2, 3, 30, 62, 66, 89):
        desc = " | ".join(f"{l['id']}:{l['role']} c={l['poly_bev'][2]:.2f}{' coast' if l['coasting'] else ''}" for l in out)
        print(f"t={t:.2f} n={len(out)} {desc}")
print("ego_lane_id", fu.ego_lane_id(out), "pts_3d[0..2]", out[0]["points_3d"][:2])
