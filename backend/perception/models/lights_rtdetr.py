"""Fallback traffic-light detector: RT-DETR (COCO class "traffic light") + HSV colour voting.

Same output contract as OpenLendaDetector.detect(): list of housings with bbox / lamps / conf / orientation.
Arrow shapes are not recognised (boxes are usually too small); lamps carry the colour only.
"""
import time
import cv2
import numpy as np
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

MODELS = {"r18": "PekingU/rtdetr_r18vd_coco_o365", "r50": "PekingU/rtdetr_r50vd_coco_o365"}


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def classify_colour(patch_bgr):
    """Return (colour, confidence) from hue voting over bright, saturated pixels."""
    if patch_bgr.size == 0:
        return "empty", 0.0
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    lit = (s > 90) & (v > 140)
    n = int(lit.sum())
    if n < 3:
        return "empty", 0.2
    hh = h[lit]
    red = int(((hh < 10) | (hh > 165)).sum())
    yellow = int(((hh >= 10) & (hh < 35)).sum())
    green = int(((hh >= 40) & (hh < 100)).sum())
    votes = {"red": red, "yellow": yellow, "green": green}
    best = max(votes, key=votes.get)
    if votes[best] == 0:
        return "empty", 0.2
    conf = votes[best] / n
    # position prior for vertical housings: red top, green bottom
    H = patch_bgr.shape[0]
    ys = np.nonzero(lit)[0]
    if H >= 12 and len(ys) > 0:
        rel = ys.mean() / H
        if best == "red" and rel > 0.66:
            conf *= 0.6
        if best == "green" and rel < 0.33:
            conf *= 0.6
    return best, float(min(1.0, conf))


class RTDetrLightDetector:
    def __init__(self, size="r50", conf=0.4, device="cuda", fp16=True, min_px=5, colour="apollo", pad=0.15):
        """colour: 'apollo' (Baidu Apollo CNN recognisers, Chinese lights) or 'hsv' (hue voting)."""
        name = MODELS[size]
        self.colour_mode, self.pad = colour, pad
        self.apollo = None
        if colour == "apollo":
            from .apollo_tl_classifier import ApolloTLClassifier
            self.apollo = ApolloTLClassifier(device)
        self.proc = RTDetrImageProcessor.from_pretrained(name)
        self.model = RTDetrForObjectDetection.from_pretrained(name).to(device).eval()
        if fp16:
            self.model = self.model.half()
        self.fp16, self.device, self.conf, self.min_px = fp16, device, conf, min_px
        self.tl_id = next(i for i, n in self.model.config.id2label.items() if n == "traffic light")
        self.last_ms = 0.0

    @torch.no_grad()
    def detect(self, img_bgr):
        """GPU-side preprocessing (RT-DETR: 640x640 resize, /255, no mean/std) and traffic-light-only postprocess."""
        h, w = img_bgr.shape[:2]
        t0 = time.perf_counter()
        x = torch.from_numpy(img_bgr).to(self.device, non_blocking=True)
        x = x[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float()
        x = torch.nn.functional.interpolate(x, size=(640, 640), mode="bilinear", align_corners=False) / 255.0
        x = x.half() if self.fp16 else x
        out = self.model(pixel_values=x)
        probs = out.logits[0].float().sigmoid()[:, self.tl_id]          # (300,) traffic-light score per query
        keep = probs > self.conf
        boxes = out.pred_boxes[0].float()[keep]                          # cx, cy, w, h normalised
        scores = probs[keep]
        torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - t0) * 1000
        cands, crops = [], []
        for s, b in zip(scores.tolist(), boxes.tolist()):
            cx, cy, bw, bh = b
            x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            if x2 - x1 < self.min_px or y2 - y1 < self.min_px:
                continue
            pw, ph = (x2 - x1) * self.pad, (y2 - y1) * self.pad
            xi1, yi1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            xi2, yi2 = min(w, int(np.ceil(x2 + pw))), min(h, int(np.ceil(y2 + ph)))
            cands.append((float(s), x1, y1, x2, y2))
            crops.append(img_bgr[yi1:yi2, xi1:xi2])
        # RT-DETR has no NMS; drop near-duplicate queries on the same light
        order = sorted(range(len(cands)), key=lambda i: -cands[i][0])
        kept = []
        for i in order:
            bi = cands[i][1:]
            if all(_iou(bi, cands[j][1:]) < 0.5 for j in kept):
                kept.append(i)
        cands = [cands[i] for i in kept]
        crops = [crops[i] for i in kept]
        if self.apollo is not None:
            colours = self.apollo.classify(crops)
        else:
            colours = [classify_colour(c) for c in crops]
        housings = []
        for (score, x1, y1, x2, y2), (colour, cconf) in zip(cands, colours):
            housings.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "lamps": [{"label": colour, "conf": round(score * (0.5 + 0.5 * cconf), 3)}],
                "colors": [{"label": colour, "conf": round(score * (0.5 + 0.5 * cconf), 3)}],
                "shapes": [],
                "conf": score,
                "orientation": "horizontal" if (x2 - x1) >= (y2 - y1) else "vertical",
            })
        housings.sort(key=lambda hh: -hh["conf"])
        return housings
