"""Baidu Apollo traffic-light colour recognisers (horizontal / vertical / quadrate), re-implemented in PyTorch
and loaded straight from the released caffemodel files (Apache-2.0, trained on Chinese roads).

Architecture (all three): 5x [conv3x3 -> BN -> scale -> ReLU -> maxpool3x3/2] -> global avg -> fc128 -> BN -> ReLU -> fc4 -> softmax
Input: BGR crop resized to (W,H) = (96,32) horizontal, (32,96) vertical, (64,64) quadrate; (x - mean_bgr) * 0.01.
Output classes: 0 black/off, 1 red, 2 yellow, 3 green (Apollo TLColor order).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ..paths import THIRD

_ROOT = THIRD / "apollo_tl"
SPECS = {  # name: (input (W,H), mean BGR, caffemodel)
    "horizontal": ((96, 32), (69.06, 66.58, 66.56), "horizontal_caffe/horizontal_caffe/baidu_iter_200000.caffemodel"),
    "vertical": ((32, 96), (69.06, 66.58, 66.56), "vertical_caffe/vertical_caffe/baidu_iter_250000.caffemodel"),
    "quadrate": ((64, 64), (125.68, 128.93, 109.56), "quadrate_caffe/quadrate_caffe/baidu_iter_200000.caffemodel"),
}
CLASSES = ("empty", "red", "yellow", "green")
SCALE = 0.01


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [3, 32, 64, 128, 128, 128]
        self.convs = nn.ModuleList([nn.Conv2d(chans[i], chans[i + 1], 3, padding=1) for i in range(5)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(chans[i + 1]) for i in range(5)])
        self.ft = nn.Linear(128, 128)
        self.ft_bn = nn.BatchNorm1d(128)
        self.logits = nn.Linear(128, 4)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
            x = F.max_pool2d(x, 3, stride=2, padding=1, ceil_mode=False)
        x = x.mean(dim=(2, 3))
        x = F.relu(self.ft_bn(self.ft(x)))
        return F.softmax(self.logits(x), dim=1)


def _load_caffe(path):
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from caffe2onnx.proto import caffe_upsample_pb2 as pb
    net = pb.NetParameter()
    with open(path, "rb") as f:
        net.ParseFromString(f.read())
    blobs = {}
    for l in net.layer:
        if len(l.blobs):
            blobs[l.name] = [np.array(b.data, np.float32).reshape(tuple(b.shape.dim)) for b in l.blobs]
    return blobs


def _fill_bn(bn, bn_blobs, scale_blobs, eps=1e-5):
    mean, var, factor = bn_blobs
    factor = float(factor[0]) if factor[0] != 0 else 1.0
    bn.running_mean.copy_(torch.from_numpy(mean / factor))
    bn.running_var.copy_(torch.from_numpy(var / factor))
    bn.weight.copy_(torch.from_numpy(scale_blobs[0]))
    bn.bias.copy_(torch.from_numpy(scale_blobs[1]) if len(scale_blobs) > 1 else torch.zeros_like(bn.bias))
    bn.eps = eps


def build(name):
    blobs = _load_caffe(str(_ROOT / SPECS[name][2]))
    net = _Net()
    with torch.no_grad():
        for i in range(5):
            w, b = blobs[f"conv{i + 1}"]
            net.convs[i].weight.copy_(torch.from_numpy(w))
            net.convs[i].bias.copy_(torch.from_numpy(b))
            _fill_bn(net.bns[i], blobs[f"conv{i + 1}_bn"], blobs[f"conv{i + 1}_bn_scale"])
        w, b = blobs["ft"]
        net.ft.weight.copy_(torch.from_numpy(w))
        net.ft.bias.copy_(torch.from_numpy(b))
        _fill_bn(net.ft_bn, blobs["ft_bn"], blobs["ft_bn_scale"])
        w, b = blobs["logits"]
        net.logits.weight.copy_(torch.from_numpy(w))
        net.logits.bias.copy_(torch.from_numpy(b))
    return net.eval()


class ApolloTLClassifier:
    def __init__(self, device="cuda"):
        self.device = device
        self.nets = {n: build(n).to(device) for n in SPECS}

    @staticmethod
    def pick(w, h):
        if w >= 1.5 * h:
            return "horizontal"
        if h >= 1.5 * w:
            return "vertical"
        return "quadrate"

    @torch.no_grad()
    def classify(self, crops_bgr):
        """crops_bgr: list of HxWx3 uint8 crops. Returns list of (colour, prob)."""
        out = [None] * len(crops_bgr)
        groups = {}
        for i, c in enumerate(crops_bgr):
            if c is None or c.size == 0:
                out[i] = ("empty", 0.0)
                continue
            groups.setdefault(self.pick(c.shape[1], c.shape[0]), []).append(i)
        for name, idx in groups.items():
            (W, H), mean, _ = SPECS[name]
            batch = []
            for i in idx:
                x = cv2.resize(crops_bgr[i], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                x = (x - np.array(mean, np.float32)) * SCALE
                batch.append(x.transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(batch)).to(self.device)
            p = self.nets[name](x).cpu().numpy()
            for i, pi in zip(idx, p):
                k = int(pi.argmax())
                out[i] = (CLASSES[k], float(pi[k]))
        return out
