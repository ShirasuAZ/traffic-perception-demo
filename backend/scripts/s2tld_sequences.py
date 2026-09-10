"""Group S2TLD frames into contiguous sequences by timestamp and build mp4 clips from the longest ones.

  python -m scripts.s2tld_sequences --root data/s2tld --out data --top 3 --fps 5 --interp 20
"""
import argparse
import os
import re
import subprocess
from datetime import datetime

import cv2


def parse_ts(name):
    m = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2})_(\d{2})_(\d{2})\.(\d+)", name)
    d, h, mi, s, frac = m.groups()
    return datetime.strptime(d, "%Y-%m-%d").timestamp() + int(h) * 3600 + int(mi) * 60 + int(s) + float("0." + frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/s2tld")
    ap.add_argument("--out", default="data")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--fps", type=float, default=5.0, help="native rate of the frames")
    ap.add_argument("--interp", type=int, default=0, help="if >0, motion-interpolate to this fps with ffmpeg")
    ap.add_argument("--gap", type=float, default=0.6, help="max seconds between frames of one sequence")
    args = ap.parse_args()
    img_dir = next(p for p, d, f in os.walk(args.root) if os.path.basename(p) == "JPEGImages")
    names = sorted(os.listdir(img_dir), key=parse_ts)
    seqs, cur = [], [names[0]]
    for a, b in zip(names, names[1:]):
        if parse_ts(b) - parse_ts(a) <= args.gap:
            cur.append(b)
        else:
            seqs.append(cur)
            cur = [b]
    seqs.append(cur)
    seqs.sort(key=len, reverse=True)
    print(f"{len(names)} frames, {len(seqs)} sequences; longest: {[len(s) for s in seqs[:10]]}")
    for s in seqs[:10]:
        dts = [parse_ts(b) - parse_ts(a) for a, b in zip(s, s[1:])]
        print(f"  {len(s):4d} frames  {s[0][:19]} -> {s[-1][:19]}  median dt {sorted(dts)[len(dts) // 2]:.3f}s")
    for i, s in enumerate(seqs[:args.top]):
        raw = os.path.join(args.out, f"s2tld_seq{i}_{len(s)}f_{int(args.fps)}fps.mp4")
        h, w = cv2.imread(os.path.join(img_dir, s[0])).shape[:2]
        wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for n in s:
            wr.write(cv2.imread(os.path.join(img_dir, n)))
        wr.release()
        with open(raw.replace(".mp4", ".frames.txt"), "w") as f:
            f.write("\n".join(s))
        print("wrote", raw)
        if args.interp:
            out = raw.replace(f"_{int(args.fps)}fps.mp4", f"_{args.interp}fps.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf",
                            f"minterpolate=fps={args.interp}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out], check=True)
            print("wrote", out)


if __name__ == "__main__":
    main()
