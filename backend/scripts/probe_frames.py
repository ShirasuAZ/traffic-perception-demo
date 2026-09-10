"""Run OpenLenda + CLRerNet on selected timestamps of a video and write annotated frames.

  python -m scripts.probe_frames --video data/raw/x.webm --times 2 66 70 --out logs/probe_x
"""
import argparse
import os
import cv2
import numpy as np
from perception.models.lights_openlenda import OpenLendaDetector
from perception.models.lanes_clrernet import CLRerNetDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--light-conf", type=float, default=0.3)
    ap.add_argument("--tsize", type=int, default=1280)
    ap.add_argument("--band-bottom", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    lights = OpenLendaDetector(conf=args.light_conf, tsize=args.tsize)
    lanes = CLRerNetDetector(band_bottom_frac=args.band_bottom)
    for t in args.times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            print(f"t={t}: no frame")
            continue
        hs = lights.detect(img)
        ls = lanes.detect(img)
        print(f"t={t:6.1f}s  lights={len(hs)}  lanes={len(ls)}  " +
              "; ".join(f"[{','.join(l['label'] for l in h['lamps'])}]{h['conf']:.2f}@{int(h['bbox'][2]-h['bbox'][0])}px" for h in hs))
        for uv, conf in ls:
            cv2.polylines(img, [uv.astype(np.int32)], False, (0, 255, 0), 3)
        for h in hs:
            x1, y1, x2, y2 = map(int, h["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(img, ",".join(l["label"] for l in h["lamps"]), (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(args.out, f"t{t:06.1f}.jpg"), img)


if __name__ == "__main__":
    main()
