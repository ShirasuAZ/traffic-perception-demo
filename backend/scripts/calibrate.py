"""Estimate calib.yaml for a dashcam video without a checkerboard.

  intrinsics : median of MapAnything's per-frame intrinsics estimate (rescaled to full resolution)
  pitch/yaw  : vanishing point of the two ego-lane boundaries from CLRerNet on straight-road frames
  height     : IPM separation of the ego-lane boundaries forced to --lane-width

  python -m scripts.calibrate --video data/x.mp4 --out data/x.calib.yaml [--frames 20] [--lane-width 3.75]
"""
import argparse
import cv2
import numpy as np

from perception.calib import Calib
from perception.models.scene_mapanything import MapAnythingScene
from perception.models.lanes_clrernet import CLRerNetDetector


def line_fit(uv):
    """Fit u = m v + b to the lower half of a lane (least squares). Returns (m, b)."""
    v, u = uv[:, 1], uv[:, 0]
    sel = v > np.percentile(v, 40)
    A = np.c_[v[sel], np.ones(sel.sum())]
    (m, b), *_ = np.linalg.lstsq(A, u[sel], rcond=None)
    return m, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--lane-width", type=float, default=3.75)
    ap.add_argument("--height-prior", type=float, default=1.4)
    ap.add_argument("--band-bottom", type=float, default=1.0, help="fraction of the frame height where the road band ends (exclude the hood)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = np.linspace(0, max(n - 1, 0), args.frames).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    print(f"video {W}x{H}, {n} frames, sampled {len(frames)}")

    # 1) intrinsics
    scene = MapAnythingScene()
    Ks = []
    for f in frames:
        v = scene.infer([f])[0]
        K, (rw, rh) = v["K"], v["ratio"]
        Ks.append([K[0, 0] / rw, K[1, 1] / rh, K[0, 2] / rw, K[1, 2] / rh])
    fx, fy, cx, cy = np.median(np.array(Ks), axis=0)
    print(f"intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  (spread fx {np.std([k[0] for k in Ks]):.1f})")
    del scene

    # 2) vanishing point from ego lane pair
    det = CLRerNetDetector(band_bottom_frac=args.band_bottom)
    vps, pairs = [], []
    for f in frames:
        lanes = det.detect(f)
        if len(lanes) < 2:
            continue
        # ego pair: lanes whose bottom x straddle the image centre, closest to it
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
    if not vps:
        raise SystemExit("no ego lane pair found; pick a clip with a straight road and visible lane lines")
    vp = np.median(np.array(vps), axis=0)
    print(f"vanishing point {vp.round(1).tolist()} from {len(vps)} frames (spread {np.std(np.array(vps), axis=0).round(1).tolist()})")

    calib = Calib(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=W, height=H, cam_height_m=args.height_prior)
    calib.set_pitch_from_vanishing_point(vp)

    # 3) height from lane width: use lower parts of the ego pair (rows below the vanishing point + margin)
    seps = []
    for L, R in pairs:
        rows = np.linspace(vp[1] + 0.25 * (H - vp[1]), H - 1, 12)
        ml, bl = line_fit(L)
        mr, br = line_fit(R)
        lu = np.c_[ml * rows + bl, rows]
        ru = np.c_[mr * rows + br, rows]
        gl, gr = calib.pixel_to_ground(lu), calib.pixel_to_ground(ru)
        seps.append(np.nanmedian(np.abs(gl[:, 1] - gr[:, 1])))
    sep = float(np.median(seps))
    calib.cam_height_m = calib.cam_height_m * args.lane_width / sep
    print(f"pitch {calib.pitch_deg:.2f} deg, yaw {calib.yaw_deg:.2f} deg, height {calib.cam_height_m:.2f} m "
          f"(lane sep at prior height was {sep:.2f} m)")
    calib.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
