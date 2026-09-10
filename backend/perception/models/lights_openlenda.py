"""OpenLenda (YOLOX) traffic-light detector wrapper.

Output: list of housings. One housing = one physical light box with a list of lamps.
OpenLenda emits one detection per (box, label) pair; labels are colors
(red/green/yellow/empty) and shapes (straight/left/right/other). We merge
detections whose boxes overlap (IoU > 0.6) into one housing.
"""
import sys, time
import numpy as np
import torch
from ..paths import THIRD, WEIGHTS

_OL = THIRD / "openlenda"
if str(_OL) not in sys.path:
    sys.path.insert(0, str(_OL))

from yolox.exp import get_exp            # noqa: E402
from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.utils import postprocess      # noqa: E402

CLASSES = ("red", "green", "yellow", "empty", "straight", "left", "right", "other")
COLORS = {"red", "green", "yellow", "empty"}
SHAPES = {"straight", "left", "right", "other"}


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class OpenLendaDetector:
    def __init__(self, variant="s", conf=0.4, nms=0.01, fp16=True, device="cuda", tsize=1280):
        """tsize: network input side. OpenLenda was trained at 1280; 640 loses small/far lights on 1080p video."""
        exp = get_exp(str(_OL / "exps" / f"openlenda_{variant}.py"), None)
        self.exp = exp
        self.num_classes = exp.num_classes
        self.test_size = (tsize, tsize)
        self.conf, self.nms = conf, nms
        self.device, self.fp16 = device, fp16
        model = exp.get_model()
        ckpt = torch.load(WEIGHTS / f"openlenda_{variant}.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        if fp16:
            model = model.half()
        self.model = model
        self.preproc = ValTransform(legacy=False)
        self.last_ms = 0.0

    @torch.no_grad()
    def detect(self, img_bgr):
        """img_bgr: HxWx3 uint8. Returns list of housings (dicts)."""
        h, w = img_bgr.shape[:2]
        ratio = min(self.test_size[0] / h, self.test_size[1] / w)
        x, _ = self.preproc(img_bgr, None, self.test_size)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        x = x.half() if self.fp16 else x.float()
        t0 = time.perf_counter()
        out = self.model(x)
        out = postprocess(out.float(), self.num_classes, self.conf, self.nms)[0]
        torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - t0) * 1000
        if out is None:
            return []
        out = out.cpu().numpy()
        dets = []
        for row in out:
            box = (row[:4] / ratio).tolist()
            dets.append((box, float(row[4] * row[5]), CLASSES[int(row[6])]))
        return self._group(dets)

    @staticmethod
    def _group(dets):
        dets.sort(key=lambda d: -d[1])
        housings = []
        for box, score, label in dets:
            for hs in housings:
                if _iou(hs["bbox"], box) > 0.6:
                    hs["lamps"].append({"label": label, "conf": score})
                    break
            else:
                housings.append({"bbox": [round(v, 1) for v in box], "lamps": [{"label": label, "conf": score}]})
        for hs in housings:
            hs["colors"] = [l for l in hs["lamps"] if l["label"] in COLORS]
            hs["shapes"] = [l for l in hs["lamps"] if l["label"] in SHAPES]
            hs["conf"] = max(l["conf"] for l in hs["lamps"])
            bw, bh = hs["bbox"][2] - hs["bbox"][0], hs["bbox"][3] - hs["bbox"][1]
            hs["orientation"] = "horizontal" if bw >= bh else "vertical"
        return housings
