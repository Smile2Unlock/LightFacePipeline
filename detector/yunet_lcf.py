"""Ultra-light fallback detector: LCF + YuNet-S (<1 MB FP32 ONNX) with identical 5-pt contract."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detector.lcf import LearnableCurveFrontend
from detector.yolo11_face import ConvBNAct, DecoupledFaceHead


class YuNetDWBlock(nn.Module):
    """Lightweight YuNet-style depthwise-separable conv block."""

    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 3, stride=stride, padding=1, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish(inplace=True)
        self.use_res = stride == 1 and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.dw(x)))
        y = self.act(self.bn2(self.pw(y)))
        return x + y if self.use_res else y


class LCFYuNetS(nn.Module):
    """Sub-1MB FP32 ONNX fallback detector (`LCF + YuNet-S`) sharing exact P3/P4/P5/5-pt contract."""

    def __init__(self, use_lcf: bool = True, width_scale: float = 1.5, reg_max: int = 8) -> None:
        super().__init__()
        self.use_lcf = use_lcf
        self.lcf = LearnableCurveFrontend(in_ch=3, hidden_ch=16, num_iters=4) if use_lcf else nn.Identity()

        def cw(c: int) -> int:
            return max(16, int(round(c * width_scale / 8.0) * 8))

        c0 = cw(16)
        c1 = cw(24)
        c2 = cw(48)
        c3 = cw(64)

        self.stem = ConvBNAct(3, c0, 3, 2)  # /2
        self.s1 = nn.Sequential(YuNetDWBlock(c0, c1, 2), YuNetDWBlock(c1, c1, 1))  # /4
        self.s2 = nn.Sequential(YuNetDWBlock(c1, c2, 2), YuNetDWBlock(c2, c2, 1))  # /8 P3
        self.s3 = nn.Sequential(YuNetDWBlock(c2, c3, 2), YuNetDWBlock(c3, c3, 1))  # /16 P4
        self.s4 = nn.Sequential(YuNetDWBlock(c3, c3, 2), YuNetDWBlock(c3, c3, 1))  # /32 P5

        self.lat3 = ConvBNAct(c2, c2, 1, 1)
        self.lat4 = ConvBNAct(c3, c2, 1, 1)
        self.lat5 = ConvBNAct(c3, c2, 1, 1)

        self.fpn_channels = [c2, c2, c2]
        self.head = DecoupledFaceHead([c2, c2, c2], strides=[8, 16, 32], reg_max=reg_max, num_kpts=5)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        x_enh = self.lcf(x)
        x1 = self.s1(self.stem(x_enh))
        p3 = self.s2(x1)
        p4 = self.s3(p3)
        p5 = self.s4(p4)

        n5 = self.lat5(p5)
        n4 = self.lat4(p4) + F.interpolate(n5, scale_factor=2.0, mode="nearest")
        n3 = self.lat3(p3) + F.interpolate(n4, scale_factor=2.0, mode="nearest")

        cls_outs, box_outs, kpt_outs = self.head.forward_raw([n3, n4, n5])
        boxes_xyxy, scores, kpts, cls_logits, dfl_dist = self.head.decode(cls_outs, box_outs, kpt_outs)
        return {
            "enhanced_rgb": x_enh,
            "boxes_xyxy": boxes_xyxy,
            "scores": scores,
            "kpts": kpts,
            "cls_logits": cls_logits,
            "dfl_dist": dfl_dist,
            "fpn_features": [n3, n4, n5],
        }
