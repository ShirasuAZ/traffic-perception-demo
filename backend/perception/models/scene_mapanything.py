"""MapAnything wrapper: N in-memory BGR frames -> per-view metric depth, intrinsics, cam2world pose (world = view 0)."""
import time
import numpy as np
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs


class MapAnythingScene:
    def __init__(self, repo="facebook/map-anything", device="cuda", resolution_set=518):
        self.device = device
        self.resolution_set = resolution_set
        self.model = MapAnything.from_pretrained(repo).to(device).eval()
        self.last_ms = 0.0

    @torch.no_grad()
    def infer(self, frames_bgr, intrinsics=None):
        """frames_bgr: list of HxWx3 uint8 (same size). intrinsics: optional 3x3 for the ORIGINAL image size.
        Returns list of dicts with numpy arrays: depth (h,w), K (3,3) for the resized image, cam2world (4,4),
        conf (h,w), mask (h,w), scale (float), size (h,w), plus resize ratio info."""
        views = []
        for f in frames_bgr:
            rgb = torch.from_numpy(np.ascontiguousarray(f[:, :, ::-1])).to(self.device)
            v = {"img": rgb}
            if intrinsics is not None:
                v["intrinsics"] = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device)
            views.append(v)
        proc = preprocess_inputs(views, resolution_set=self.resolution_set)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        preds = self.model.infer(proc, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                                 apply_mask=True, mask_edges=False, apply_confidence_mask=False)
        torch.cuda.synchronize(); self.last_ms = (time.perf_counter() - t0) * 1000
        out = []
        H0, W0 = frames_bgr[0].shape[:2]
        for p in preds:
            depth = p["depth_z"][0, :, :, 0].float().cpu().numpy()
            h, w = depth.shape
            out.append({
                "depth": depth,
                "K": p["intrinsics"][0].float().cpu().numpy(),
                "cam2world": p["camera_poses"][0].float().cpu().numpy(),
                "conf": p["conf"][0].float().cpu().numpy(),
                "mask": p["mask"][0, :, :, 0].bool().cpu().numpy(),
                "scale": float(p["metric_scaling_factor"].flatten()[0]) if "metric_scaling_factor" in p else 1.0,
                "size": (h, w),
                "ratio": (w / W0, h / H0),
            })
        return out
