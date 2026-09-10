"""Lane fusion in BEV (ego frame, metres).

Per frame: 2D lane point sets -> IPM -> quadratic y = a x^2 + b x + c -> Hungarian association on lateral
offset -> per-track Kalman filter on [a, b, c]. Elevation z is added from a depth map when available.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

X_MIN, X_MAX = 3.0, 40.0     # longitudinal fit range (m)
X_REF = 5.0                  # where lateral offsets are compared for association
GATE_M = 0.9                 # association gate on lateral offset
MIN_HITS = 3                 # frames before a track is published
MAX_MISS_S = 0.5             # coast time before a track is dropped


def fit_bev(pts_xy, w=None):
    """Least squares y = a x^2 + b x + c. pts_xy (N,2). Returns (a,b,c) or None."""
    if len(pts_xy) < 4:
        return None
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    A = np.stack([x * x, x, np.ones_like(x)], 1)
    if w is not None:
        A, y = A * w[:, None], y * w
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def eval_poly(coef, x):
    a, b, c = coef
    return a * x * x + b * x + c


class LaneTrack:
    _next_id = 1

    def __init__(self, coef, conf, ts):
        self.id = LaneTrack._next_id
        LaneTrack._next_id += 1
        self.x = np.asarray(coef, float)                   # [a,b,c]
        self.P = np.diag([1e-4, 1e-2, 0.5])
        self.Q = np.diag([2e-7, 2e-5, 4e-3])               # per-frame random-walk noise
        self.R = np.diag([1e-6, 1e-4, 2e-2])               # measurement noise of a single fit
        self.conf = conf
        self.hits, self.miss = 1, 0
        self.last_seen, self.x_max_seen = ts, X_MAX

    def predict(self):
        self.P = self.P + self.Q

    def update(self, coef, conf, ts, x_max):
        z = np.asarray(coef, float)
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(3) - K) @ self.P
        self.conf = 0.7 * self.conf + 0.3 * conf
        self.hits += 1
        self.miss = 0
        self.last_seen = ts
        self.x_max_seen = x_max

    def offset_at(self, x=X_REF):
        return eval_poly(self.x, x)


class LaneFusion:
    def __init__(self, calib):
        self.calib = calib
        self.tracks = []

    # ---- per frame ----------------------------------------------------
    def update(self, lanes_2d, ts, depth=None, depth_ratio=None):
        """lanes_2d: list of (pts_uv (N,2), conf). ts: seconds. depth: optional (h,w) cam z-depth, depth_ratio
        (w_ratio, h_ratio) mapping full-res pixels -> depth-map pixels. Returns list of published lanes."""
        dets = []
        for uv, conf in lanes_2d:
            uv = np.asarray(uv, float)
            g = self.calib.pixel_to_ground(uv)
            ok = np.isfinite(g[:, 0]) & (g[:, 0] >= X_MIN) & (g[:, 0] <= X_MAX)
            coef = fit_bev(g[ok, :2])
            if coef is None:
                continue
            dets.append((coef, float(conf), float(g[ok, 0].max())))

        for t in self.tracks:
            t.predict()

        # association on lateral offset at X_REF
        if self.tracks and dets:
            cost = np.array([[abs(t.offset_at() - eval_poly(d[0], X_REF)) for d in dets] for t in self.tracks])
            ri, ci = linear_sum_assignment(cost)
            matched_d = set()
            for r, c in zip(ri, ci):
                if cost[r, c] < GATE_M:
                    coef, conf, xmax = dets[c]
                    self.tracks[r].update(coef, conf, ts, xmax)
                    matched_d.add(c)
            for c, d in enumerate(dets):
                if c not in matched_d:
                    self.tracks.append(LaneTrack(d[0], d[1], ts))
        elif dets:
            self.tracks = [LaneTrack(d[0], d[1], ts) for d in dets]

        for t in self.tracks:
            if t.last_seen != ts:
                t.miss += 1
        self.tracks = [t for t in self.tracks if ts - t.last_seen <= MAX_MISS_S]
        # merge duplicates that converged onto the same line
        self.tracks.sort(key=lambda t: -t.hits)
        kept = []
        for t in self.tracks:
            if all(abs(t.offset_at() - k.offset_at()) > GATE_M * 0.6 for k in kept):
                kept.append(t)
        self.tracks = kept
        return self.publish(depth, depth_ratio)

    # ---- output ---------------------------------------------------------
    def publish(self, depth=None, depth_ratio=None):
        pub = [t for t in self.tracks if t.hits >= MIN_HITS]
        offs = {t.id: t.offset_at() for t in pub}
        left = [t for t in pub if offs[t.id] > 0]
        right = [t for t in pub if offs[t.id] <= 0]
        ego_left = min(left, key=lambda t: offs[t.id]) if left else None
        ego_right = max(right, key=lambda t: offs[t.id]) if right else None
        out = []
        for t in pub:
            if t is ego_left:
                role = "ego_left"
            elif t is ego_right:
                role = "ego_right"
            else:
                role = "left" if offs[t.id] > 0 else "right"
            xs = np.arange(X_MIN, min(t.x_max_seen, X_MAX) + 1e-6, 1.0)
            ys = eval_poly(t.x, xs)
            pts = np.stack([xs, ys, np.zeros_like(xs)], 1)
            if depth is not None:
                pts[:, 2] = self._elevation(pts, depth, depth_ratio)
            uv, _ = self.calib.ego_to_pixel(pts)
            out.append({
                "id": t.id, "role": role, "confidence": round(float(t.conf), 3), "age_frames": t.hits,
                "coasting": t.miss > 0,
                "poly_bev": [float(v) for v in t.x],
                "points_3d": pts.round(3).tolist(),
                "points_2d": uv.round(1).tolist(),
            })
        return out

    @staticmethod
    def ego_lane_id(lanes):
        for lane in lanes:
            if lane["role"] == "ego_left":
                return lane["id"]
        return None

    def _elevation(self, pts, depth, ratio):
        """Refine z of ground points using a metric depth map; falls back to 0 where depth is missing/implausible."""
        uv, z_plane = self.calib.ego_to_pixel(pts)
        h, w = depth.shape
        u = np.clip(np.nan_to_num(uv[:, 0] * ratio[0]).round().astype(int), 0, w - 1)
        v = np.clip(np.nan_to_num(uv[:, 1] * ratio[1]).round().astype(int), 0, h - 1)
        d = depth[v, u]
        ok = np.isfinite(uv[:, 0]) & (d > 0) & (np.abs(d - z_plane) / np.maximum(z_plane, 1) < 0.3) & (pts[:, 0] <= 30)
        z = np.zeros(len(pts))
        if ok.any():
            p3 = self.calib.pixel_depth_to_ego(uv[ok], d[ok])
            z[ok] = p3[:, 2]
            z = np.convolve(np.pad(z, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        return np.clip(z, -1.0, 1.0)
