"""CLRerNet (mmdet 3.x) lane detector wrapper: BGR frame -> list of (points_uv (N,2), conf)."""
import os
import sys
import time
import tempfile
import numpy as np
import torch
from ..paths import THIRD, WEIGHTS

_CL = THIRD / "CLRerNet"
if str(_CL) not in sys.path:
    sys.path.insert(0, str(_CL))

from mmengine.config import Config          # noqa: E402
from mmdet.apis import init_detector        # noqa: E402
from libs.datasets.pipelines import Compose  # noqa: E402  (registers custom transforms)


class CLRerNetDetector:
    # CULane geometry the model was trained on
    CULANE_W, CULANE_H = 1640, 590

    def __init__(self, config="configs/clrernet/culane/clrernet_culane_dla34_ema.py",
                 weights="clrernet_culane_dla34_ema.pth", device="cuda:0", fp16=True,
                 letterbox=True, band_bottom_frac=1.0):
        """letterbox: convert any frame to CULane geometry first (resize to 1640 wide, keep the bottom 590
        rows ending at band_bottom_frac*height). Points are mapped back to the original frame."""
        self.letterbox = letterbox
        self.band_bottom_frac = band_bottom_frac
        cfg = Config.fromfile(str(_CL / config))
        # init_detector builds the test dataset for metainfo; point it at an empty list so no data is needed
        empty = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        empty.close()
        cfg.test_dataloader.dataset.data_list = empty.name
        self.model = init_detector(cfg, str(WEIGHTS / weights), device=device)
        self.model.bbox_head.test_cfg.as_lanes = False
        self.pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
        self.fp16 = fp16
        self.last_ms = 0.0
        self.pre_ms = 0.0
        os.unlink(empty.name)

    def _to_culane(self, img_bgr):
        """Returns (img_1640x590, scale, y_offset) such that orig = (u / scale, (v + y_offset) / scale)."""
        import cv2
        h, w = img_bgr.shape[:2]
        scale = self.CULANE_W / w
        rs = cv2.resize(img_bgr, (self.CULANE_W, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        bottom = int(round(rs.shape[0] * self.band_bottom_frac))
        top = max(0, bottom - self.CULANE_H)
        band = rs[top:bottom]
        if band.shape[0] < self.CULANE_H:  # frame shorter than the band: pad on top
            pad = np.zeros((self.CULANE_H - band.shape[0], self.CULANE_W, 3), band.dtype)
            band = np.concatenate([pad, band], 0)
            top -= pad.shape[0]
        return np.ascontiguousarray(band), scale, top

    @torch.no_grad()
    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        t0 = time.perf_counter()
        scale, y_off = 1.0, 0
        if self.letterbox:
            img_bgr, scale, y_off = self._to_culane(img_bgr)
            h, w = img_bgr.shape[:2]
        data = dict(filename="", sub_img_name=None, img=img_bgr, gt_points=[], id_classes=[], id_instances=[],
                    img_shape=img_bgr.shape, ori_shape=img_bgr.shape)
        data = self.pipeline(data)
        batch = dict(inputs=[data["inputs"]], data_samples=[data["data_samples"]])
        self.pre_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            results = self.model.test_step(batch)
        torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - t1) * 1000
        lanes = results[0]["lanes"]
        scores = results[0].get("scores", None)
        out = []
        for i, lane in enumerate(lanes):
            lane = lane.cpu().numpy()
            ok = (lane[:, 0] >= 0) & (lane[:, 0] < 1)
            if ok.sum() < 2:
                continue
            uv = np.stack([lane[ok, 0] * w, lane[ok, 1] * h], 1)[::-1]  # bottom -> top order
            if self.letterbox:
                uv = np.stack([uv[:, 0] / scale, (uv[:, 1] + y_off) / scale], 1)
            conf = float(scores[i]) if scores is not None else 1.0
            out.append((uv, conf))
        return out
