"""Camera calibration + ego-frame geometry.

Frames:
  cam : OpenCV, x right, y down, z forward, origin at optical center
  ego : x forward, y left, z up, origin = camera ground projection (we ignore rear-axle offset for the demo)
Ground plane: ego z = 0. Camera sits at ego (0, 0, h), pitched down by `pitch_deg` (positive = looking down).
"""
from dataclasses import dataclass, field
import numpy as np
import yaml

# cam axes expressed in ego axes for zero rotation: cam x -> -ego y, cam y -> -ego z, cam z -> ego x
_R_CAM2EGO_0 = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])


def _rot_y(a):  # ego yaw (about ego z) / cam pitch (about cam x) helpers below use explicit axes
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Calib:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    cam_height_m: float = 1.4
    pitch_deg: float = 0.0     # + looks down
    yaw_deg: float = 0.0       # + looks left (ego +y)
    roll_deg: float = 0.0
    dist: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    # ---- construction -------------------------------------------------
    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(self.__dict__, f, sort_keys=False)

    # ---- matrices -----------------------------------------------------
    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def R_cam2ego(self):
        # rotation of the camera body in ego frame: yaw about ego z, pitch about ego y (nose down = +), roll about ego x
        p, y, r = np.deg2rad(self.pitch_deg), np.deg2rad(self.yaw_deg), np.deg2rad(self.roll_deg)
        R_body = _rot_z(y) @ _rot_y(p) @ _rot_x(r)
        return R_body @ _R_CAM2EGO_0

    @property
    def t_cam2ego(self):
        return np.array([0.0, 0.0, self.cam_height_m])

    @property
    def T_cam2ego(self):
        T = np.eye(4)
        T[:3, :3] = self.R_cam2ego
        T[:3, 3] = self.t_cam2ego
        return T

    # ---- projections --------------------------------------------------
    def pixel_rays_ego(self, uv):
        """uv: (N,2) pixels -> (N,3) unit ray directions in ego frame."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        d = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy, np.ones(len(uv))], 1)
        d = d @ self.R_cam2ego.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def pixel_to_ground(self, uv):
        """IPM: (N,2) pixels -> (N,3) ego points on z=0. Points above the horizon get NaN."""
        d = self.pixel_rays_ego(uv)
        o = self.t_cam2ego
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -o[2] / d[:, 2]
        s[(d[:, 2] >= -1e-6)] = np.nan
        return o[None, :] + s[:, None] * d

    def ego_to_pixel(self, xyz):
        """(N,3) ego -> (N,2) pixels and (N,) depth along cam z. Points behind camera get NaN."""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        pc = (xyz - self.t_cam2ego) @ self.R_cam2ego  # R^T applied on the right
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pc[:, 0] / z + self.cx
            v = self.fy * pc[:, 1] / z + self.cy
        bad = z <= 1e-6
        u[bad] = np.nan; v[bad] = np.nan
        return np.stack([u, v], 1), z

    def pixel_depth_to_ego(self, uv, depth_z):
        """Back-project pixels with cam z-depth -> ego points."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        z = np.asarray(depth_z, dtype=np.float64).reshape(-1)
        pc = np.stack([(uv[:, 0] - self.cx) / self.fx * z, (uv[:, 1] - self.cy) / self.fy * z, z], 1)
        return pc @ self.R_cam2ego.T + self.t_cam2ego

    def horizon_v(self):
        """Image row of the horizon for the current pitch (u at cx)."""
        d_ego = np.array([1.0, 0.0, 0.0])  # forward, level
        pc = d_ego @ self.R_cam2ego
        return self.fy * pc[1] / pc[2] + self.cy

    # ---- estimation helpers -------------------------------------------
    def set_pitch_from_vanishing_point(self, vp_uv):
        """Given the vanishing point of the straight road, solve pitch (and yaw)."""
        u, v = vp_uv
        self.pitch_deg = float(np.rad2deg(np.arctan2(self.cy - v, self.fy)))
        self.yaw_deg = float(np.rad2deg(np.arctan2(u - self.cx, self.fx)))

    def set_height_from_lane_width(self, left_uv, right_uv, lane_width_m=3.75):
        """left_uv/right_uv: (N,2) pixel samples of ego-lane boundaries at similar rows.
        Scales cam height so that their IPM separation equals lane_width_m."""
        gl, gr = self.pixel_to_ground(left_uv), self.pixel_to_ground(right_uv)
        sep = np.nanmedian(np.abs(gl[:, 1] - gr[:, 1]))
        self.cam_height_m *= lane_width_m / sep
        return sep
