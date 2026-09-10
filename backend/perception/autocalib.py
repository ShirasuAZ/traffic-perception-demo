"""Automatic calibration from frames (no checkerboard):
  intrinsics : median of MapAnything's per-frame estimate, rescaled to full resolution
  pitch/yaw  : vanishing point of the ego-lane pair from CLRerNet
  height     : ego-lane IPM separation forced to lane_width_m
"""
import numpy as np
from .calib import Calib


def line_fit(uv):
    v, u = uv[:, 1], uv[:, 0]
    sel = v > np.percentile(v, 40)
    A = np.c_[v[sel], np.ones(sel.sum())]
    (m, b), *_ = np.linalg.lstsq(A, u[sel], rcond=None)
    return m, b


def calibrate_frames(frames, scene, lane_det, lane_width_m=3.75, height_prior=1.4, log=print):
    """frames: list of BGR frames (same size). scene: MapAnythingScene, lane_det: CLRerNetDetector.
    Returns (Calib, info dict). Falls back to a generic prior when no ego lane pair is found."""
    H, W = frames[0].shape[:2]
    Ks = []
    for f in frames:
        v = scene.infer([f])[0]
        K, (rw, rh) = v["K"], v["ratio"]
        Ks.append([K[0, 0] / rw, K[1, 1] / rh, K[0, 2] / rw, K[1, 2] / rh])
    fx, fy, cx, cy = np.median(np.array(Ks), axis=0)
    info = {"fx_spread": float(np.std([k[0] for k in Ks])), "n_frames": len(frames)}
    log(f"intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    vps, pairs = [], []
    for f in frames:
        lanes = lane_det.detect(f)
        if len(lanes) < 2:
            continue
        bottoms = [(uv[0][0], uv, c) for uv, c in lanes]
        left = [b for b in bottoms if b[0] < W / 2]
        right = [b for b in bottoms if b[0] >= W / 2]
        if not left or not right:
            continue
        L, R = max(left, key=lambda b: b[0]), min(right, key=lambda b: b[0])
        ml, bl = line_fit(L[1])
        mr, br = line_fit(R[1])
        if abs(ml - mr) < 1e-6:
            continue
        v = (br - bl) / (ml - mr)
        u = ml * v + bl
        if 0 < u < W and 0 < v < H:
            vps.append((u, v))
            pairs.append((L[1], R[1]))
    calib = Calib(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=W, height=H, cam_height_m=height_prior)
    if not vps:
        log("no ego lane pair found; using prior pitch 0 / height prior")
        info.update({"vp_frames": 0, "fallback": True})
        return calib, info
    vp = np.median(np.array(vps), axis=0)
    calib.set_pitch_from_vanishing_point(vp)
    seps = []
    for L, R in pairs:
        rows = np.linspace(vp[1] + 0.25 * (H - vp[1]), H - 1, 12)
        ml, bl = line_fit(L)
        mr, br = line_fit(R)
        gl = calib.pixel_to_ground(np.c_[ml * rows + bl, rows])
        gr = calib.pixel_to_ground(np.c_[mr * rows + br, rows])
        seps.append(np.nanmedian(np.abs(gl[:, 1] - gr[:, 1])))
    sep = float(np.median(seps))
    calib.cam_height_m = calib.cam_height_m * lane_width_m / sep
    info.update({"vp": [float(vp[0]), float(vp[1])], "vp_frames": len(vps), "vp_spread": np.std(np.array(vps), axis=0).tolist(),
                 "fallback": False})
    log(f"vp {vp.round(1).tolist()} from {len(vps)} frames; pitch {calib.pitch_deg:.2f} yaw {calib.yaw_deg:.2f} height {calib.cam_height_m:.2f} m")
    return calib, info


def sample_frames_from_video(path, n=20):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def sample_frames_from_stream(cap, n=12, spacing_s=0.5):
    """Grab n frames from an open live capture, spaced in time."""
    import time
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        time.sleep(spacing_s)
    return frames
