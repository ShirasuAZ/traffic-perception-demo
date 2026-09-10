"""Evaluate traffic-light detectors on S2TLD (Chinese lights) ground truth.

Reports per detector: recall of GT boxes (IoU>=0.3), precision, colour accuracy on matched boxes, ms/frame.
  python -m scripts.eval_s2tld --n 150 --detectors openlenda_s openlenda_x rtdetr_r50
"""
import argparse
import os
import random
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

COLOUR_MAP = {"red": "red", "green": "green", "yellow": "yellow", "off": "empty", "wait_on": "red"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def load_gt(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.findall("object"):
        b = o.find("bndbox")
        out.append(([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")], o.find("name").text))
    return out


def build(name, conf):
    if name.startswith("openlenda"):
        from perception.models.lights_openlenda import OpenLendaDetector
        return OpenLendaDetector(variant=name.split("_")[1], conf=conf, tsize=1280)
    if name.startswith("rtdetr"):
        from perception.models.lights_rtdetr import RTDetrLightDetector
        return RTDetrLightDetector(size=name.split("_")[1], conf=conf)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/s2tld")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--detectors", nargs="+", default=["openlenda_s", "openlenda_x", "rtdetr_r50"])
    ap.add_argument("--save", default=None, help="dir to save annotated samples of the first detector")
    args = ap.parse_args()
    ann_dir = next(p for p, d, f in os.walk(args.root) if os.path.basename(p) == "Annotations")
    img_dir = os.path.join(os.path.dirname(ann_dir), "JPEGImages")
    files = sorted(os.listdir(ann_dir))
    random.Random(0).shuffle(files)
    files = files[:args.n]
    samples = []
    for x in files:
        gt = load_gt(os.path.join(ann_dir, x))
        img = cv2.imread(os.path.join(img_dir, x.replace(".xml", ".jpg")))
        if img is None:
            continue
        samples.append((x, img, gt))
    n_gt = sum(len(g) for _, _, g in samples)
    widths = [b[2] - b[0] for _, _, g in samples for b, _ in g]
    print(f"{len(samples)} images, {n_gt} GT lights, GT width px: p10 {np.percentile(widths, 10):.0f} median {np.median(widths):.0f} p90 {np.percentile(widths, 90):.0f}")
    for name in args.detectors:
        det = build(name, args.conf)
        det.detect(samples[0][1])
        tp = fp = 0
        colour_ok = colour_n = 0
        t_total = 0.0
        from collections import Counter
        confusion = Counter()
        for i, (x, img, gt) in enumerate(samples):
            t0 = time.perf_counter()
            hs = det.detect(img)
            t_total += time.perf_counter() - t0
            used = set()
            for h in hs:
                best, bi = 0.0, -1
                for j, (gb, gc) in enumerate(gt):
                    if j in used:
                        continue
                    v = iou(h["bbox"], gb)
                    if v > best:
                        best, bi = v, j
                if best >= 0.3:
                    tp += 1
                    used.add(bi)
                    cols = [l["label"] for l in h["lamps"] if l["label"] in ("red", "green", "yellow", "empty")]
                    if cols:
                        colour_n += 1
                        colour_ok += int(cols[0] == COLOUR_MAP.get(gt[bi][1], gt[bi][1]))
                        confusion[(gt[bi][1], cols[0])] += 1
                else:
                    fp += 1
            if args.save and name == args.detectors[0] and i < 12:
                os.makedirs(args.save, exist_ok=True)
                vis = img.copy()
                for gb, gc in gt:
                    cv2.rectangle(vis, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (255, 255, 255), 1)
                for h in hs:
                    b = [int(v) for v in h["bbox"]]
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 2)
                    cv2.putText(vis, ",".join(l["label"] for l in h["lamps"]), (b[0], max(12, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(args.save, f"{i:02d}.jpg"), vis)
        rec = tp / max(n_gt, 1)
        prec = tp / max(tp + fp, 1)
        print(f"{name:12s} recall {rec:.3f}  precision {prec:.3f}  colour acc {colour_ok / max(colour_n, 1):.3f} (n={colour_n})  {t_total / len(samples) * 1000:.0f} ms/img")
        print("   confusion (gt -> pred):", ", ".join(f"{g}->{p}:{n}" for (g, p), n in sorted(confusion.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
