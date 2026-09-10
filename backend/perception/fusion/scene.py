"""Scene fusion: MapAnything output -> metric point cloud in the ego frame + ego odometry.

Scale alignment: MapAnything depth is "metric-ish"; we refine it with the calibrated camera height by
fitting the road plane to the points that IPM says are on the ego lane at 5-25 m ahead. The ratio of
calibrated height to the fitted plane height is the scale factor (low-passed, outlier-rejected).
Odometry: the relative pose between the two most recent views (scaled) is integrated into a world
trajectory (world = ego frame at start). Only the last HISTORY_S seconds are kept.
"""
import numpy as np

HISTORY_S = 5.0
VOXEL_M = 0.15
MAX_POINTS = 50000
MAX_RANGE_M = 60.0
SCALE_ALPHA = 0.2
SCALE_OUTLIER = 0.25


def voxel_downsample(xyz, rgb, voxel):
    if len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx]


class SceneFusion:
    def __init__(self, calib):
        self.calib = calib
        self.scale = 1.0
        self.scale_conf = 0.0
        self.T_world_ego = np.eye(4)
        self.prev_cam2world = None      # MapAnything pose of the previous *current* view (unscaled, window frame)
        self.trajectory = []            # (ts, T_world_ego)
        self.speed_mps = 0.0
        self.yaw_rate_rps = 0.0
        self.last = None

    # ---- helpers ------------------------------------------------------
    def _cloud_cam(self, view, img_bgr_small=None, stride=2):
        """Back-project a view's depth map into camera-frame points (unscaled)."""
        depth, K, mask = view["depth"], view["K"], view["mask"]
        h, w = depth.shape
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        d = depth[::stride, ::stride]
        m = mask[::stride, ::stride] & np.isfinite(d) & (d > 0.1)
        us, vs, d = us[m], vs[m], d[m]
        x = (us - K[0, 2]) / K[0, 0] * d
        y = (vs - K[1, 2]) / K[1, 1] * d
        pts = np.stack([x, y, d], 1)
        rgb = None
        if img_bgr_small is not None:
            rgb = img_bgr_small[::stride, ::stride][m][:, ::-1]
        return pts, rgb, (us, vs)

    def _fit_scale(self, pts_cam, uv_small, ratio):
        """Return the plane-height ratio calibrated/observed for road points ahead, or None."""
        us, vs = uv_small
        uv_full = np.stack([us / ratio[0], vs / ratio[1]], 1)
        g = self.calib.pixel_to_ground(uv_full)
        road = np.isfinite(g[:, 0]) & (g[:, 0] > 5) & (g[:, 0] < 25) & (np.abs(g[:, 1]) < 1.6)
        if road.sum() < 200:
            return None
        p = pts_cam[road] @ self.calib.R_cam2ego.T          # rotate to ego axes, camera at origin
        # robust plane: z = n.x + d ; here we only need the camera height above the plane => use median of -z
        # after removing tilt with a least squares plane
        A = np.c_[p[:, 0], p[:, 1], np.ones(len(p))]
        coef, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
        resid = p[:, 2] - A @ coef
        inl = np.abs(resid) < 3 * max(np.median(np.abs(resid)), 0.02)
        if inl.sum() < 100:
            return None
        coef, *_ = np.linalg.lstsq(A[inl], p[inl, 2], rcond=None)
        h_obs = -coef[2]                                    # plane height below the camera at x=y=0
        if h_obs <= 0.2:
            return None
        return self.calib.cam_height_m / h_obs

    # ---- per MapAnything result -------------------------------------
    def update(self, views, ts, img_bgr=None):
        """views: MapAnythingScene.infer output (last element = current frame). Returns scene dict."""
        cur = views[-1]
        ratio = cur["ratio"]
        small = None
        if img_bgr is not None:
            import cv2
            small = cv2.resize(img_bgr, (cur["size"][1], cur["size"][0]), interpolation=cv2.INTER_AREA)
        pts_cam, rgb, uv_small = self._cloud_cam(cur, small)

        s = self._fit_scale(pts_cam, uv_small, ratio)
        if s is not None:
            if self.scale_conf == 0 or abs(s / self.scale - 1) < SCALE_OUTLIER:
                self.scale = (1 - SCALE_ALPHA) * self.scale + SCALE_ALPHA * s if self.scale_conf > 0 else s
                self.scale_conf = min(1.0, self.scale_conf + 0.2)
            else:
                self.scale_conf = max(0.0, self.scale_conf - 0.1)

        # odometry from relative pose between the previous view and the current one inside this window
        if len(views) >= 2:
            T_prev = views[-2]["cam2world"]
            T_cur = cur["cam2world"]
            T_rel_cam = np.linalg.inv(T_prev) @ T_cur          # prev_cam <- cur_cam
            T_rel_cam[:3, 3] *= self.scale
            Tc = self.calib.T_cam2ego
            T_rel_ego = Tc @ T_rel_cam @ np.linalg.inv(Tc)
            dt = ts - self.trajectory[-1][0] if self.trajectory else None
            plausible = True
            if dt and dt > 1e-3:
                v = float(np.linalg.norm(T_rel_ego[:3, 3]) / dt)
                yaw = float(np.arctan2(T_rel_ego[1, 0], T_rel_ego[0, 0]))
                # reject physically implausible steps (bad pose window / scale glitch): > 45 m/s, > 6 m/s^2 jump, > 60 deg/s
                if v > 45.0 or abs(v - self.speed_mps) / dt > 6.0 and self.trajectory and len(self.trajectory) > 3 or abs(yaw / dt) > 1.0:
                    plausible = False
                else:
                    self.speed_mps = 0.6 * self.speed_mps + 0.4 * v
                    self.yaw_rate_rps = 0.6 * self.yaw_rate_rps + 0.4 * yaw / dt
            if plausible:
                self.T_world_ego = self.T_world_ego @ T_rel_ego
            elif dt:
                # coast with the filtered speed along the current heading
                T_coast = np.eye(4)
                T_coast[0, 3] = self.speed_mps * dt
                self.T_world_ego = self.T_world_ego @ T_coast
                self.scale_conf = max(0.0, self.scale_conf - 0.1)
        self.trajectory.append((ts, self.T_world_ego.copy()))
        self.trajectory = [(t, T) for t, T in self.trajectory if ts - t <= HISTORY_S]

        # cloud into ego frame (metres)
        p = (pts_cam * self.scale) @ self.calib.R_cam2ego.T + self.calib.t_cam2ego
        keep = (p[:, 0] > 0.5) & (p[:, 0] < MAX_RANGE_M) & (np.abs(p[:, 1]) < 40) & (p[:, 2] > -3) & (p[:, 2] < 15)
        p = p[keep]
        c = rgb[keep] if rgb is not None else np.full((len(p), 3), 180, np.uint8)
        p, c = voxel_downsample(p, c, VOXEL_M)
        if len(p) > MAX_POINTS:
            sel = np.random.default_rng(0).choice(len(p), MAX_POINTS, replace=False)
            p, c = p[sel], c[sel]
        self.last = {
            "ts": ts,
            "points_xyz": p.astype(np.float32),
            "points_rgb": c.astype(np.uint8),
            "scale_factor": round(float(self.scale), 4),
            "scale_confidence": round(float(self.scale_conf), 2),
            "T_world_ego": self.T_world_ego.round(4).tolist(),
            "speed_mps": round(float(self.speed_mps), 2),
            "yaw_rate_rps": round(float(self.yaw_rate_rps), 4),
            "odom_confidence": round(float(min(1.0, self.scale_conf)), 2),
            "depth": cur["depth"] * self.scale,
            "depth_ratio": ratio,
        }
        return self.last
