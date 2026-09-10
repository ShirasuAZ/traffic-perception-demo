"""Synthetic check: a flat road at the calibrated height, rendered as a depth map with a wrong scale,
must be recovered by the scale alignment, and a known camera motion must come out as odometry."""
import numpy as np
from perception.calib import Calib
from perception.fusion.scene import SceneFusion

calib = Calib(fx=1000, fy=1000, cx=960, cy=540, width=1920, height=1080, cam_height_m=1.5, pitch_deg=2.0)
h, w = 336, 518
ratio = (w / 1920, h / 1080)
Ks = np.array([[1000 * ratio[0], 0, 960 * ratio[0]], [0, 1000 * ratio[1], 540 * ratio[1]], [0, 0, 1]])
vs, us = np.mgrid[0:h, 0:w]
uv_full = np.stack([us.ravel() / ratio[0], vs.ravel() / ratio[1]], 1)
g = calib.pixel_to_ground(uv_full)
_, z_true = calib.ego_to_pixel(np.nan_to_num(g))
depth_true = z_true.reshape(h, w)
mask = np.isfinite(g[:, 0]).reshape(h, w) & (depth_true < 80)
WRONG = 0.7                                   # model returns depth at 70 % of true metric scale


def view(T):
    return {"depth": depth_true * WRONG, "K": Ks, "mask": mask, "cam2world": T, "size": (h, w), "ratio": ratio, "conf": None}


fu = SceneFusion(calib)
# camera moves 1.0 m forward (cam +z) per step; in the unscaled model frame that is 0.7 m
T0 = np.eye(4)
for i in range(6):
    T1 = np.eye(4)
    T1[2, 3] = 1.0 * WRONG
    out = fu.update([view(T0), view(T1)], ts=i * 0.1)
    print(f"step {i}: scale={out['scale_factor']} conf={out['scale_confidence']} speed={out['speed_mps']} m/s "
          f"pos_world={np.array(out['T_world_ego'])[:3, 3].round(2).tolist()} n_pts={len(out['points_xyz'])}")
p = out["points_xyz"]
print("cloud z (should be ~0 for road):", np.percentile(p[:, 2], [5, 50, 95]).round(3).tolist())
print("expected scale", round(1 / WRONG, 3), "expected speed 10.0 m/s")
