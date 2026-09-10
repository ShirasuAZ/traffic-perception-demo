"""Sanity-check a calib.yaml against dumped lane detections: after IPM the ego-lane boundaries should be
parallel (small heading difference) and about 3.5-3.8 m apart, consistently over the clip.

  python -m scripts.check_calib --calib data/x.calib.yaml --replay data/x.replay.pkl
"""
import argparse
import pickle
import numpy as np
from perception.calib import Calib
from perception.fusion.lanes import fit_bev, eval_poly, X_MIN, X_MAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--replay", required=True)
    args = ap.parse_args()
    calib = Calib.load(args.calib)
    with open(args.replay, "rb") as f:
        lanes = pickle.load(f)["lane"]
    seps, heads, far_seps = [], [], []
    for seq, dets in lanes.items():
        fits = []
        for uv, conf in dets:
            g = calib.pixel_to_ground(np.asarray(uv, float))
            ok = np.isfinite(g[:, 0]) & (g[:, 0] >= X_MIN) & (g[:, 0] <= X_MAX)
            c = fit_bev(g[ok, :2])
            if c is not None and ok.sum() >= 6:
                fits.append(c)
        offs = [eval_poly(c, 5.0) for c in fits]
        left = [c for c, o in zip(fits, offs) if o > 0]
        right = [c for c, o in zip(fits, offs) if o <= 0]
        if not left or not right:
            continue
        L = min(left, key=lambda c: eval_poly(c, 5.0))
        R = max(right, key=lambda c: eval_poly(c, 5.0))
        seps.append(eval_poly(L, 5.0) - eval_poly(R, 5.0))
        far_seps.append(eval_poly(L, 30.0) - eval_poly(R, 30.0))
        heads.append(np.degrees(np.arctan(2 * L[0] * 15 + L[1]) - np.arctan(2 * R[0] * 15 + R[1])))
    seps, far_seps, heads = map(np.array, (seps, far_seps, heads))
    print(f"frames with ego pair: {len(seps)} / {len(lanes)}")
    print(f"ego lane width @5m : median {np.median(seps):.2f} m  (p10 {np.percentile(seps, 10):.2f}, p90 {np.percentile(seps, 90):.2f})")
    print(f"ego lane width @30m: median {np.median(far_seps):.2f} m  (p10 {np.percentile(far_seps, 10):.2f}, p90 {np.percentile(far_seps, 90):.2f})")
    print(f"heading diff L-R   : median {np.median(np.abs(heads)):.2f} deg (p90 {np.percentile(np.abs(heads), 90):.2f})")
    print("verdict:", "OK" if abs(np.median(seps) - 3.7) < 0.4 and abs(np.median(far_seps) - np.median(seps)) < 0.5 else "CHECK pitch/height")


if __name__ == "__main__":
    main()
