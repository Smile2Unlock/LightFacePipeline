"""Pure-PyTorch LCF + YOLO11n-face-5pt (Anchor-Free + C3k2 + SPPF + C2PSA + DFL + 5-pt)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detector.lcf import LearnableCurveFrontend


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 3, 1)
        self.cv2 = ConvBNAct(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2(nn.Module):
    """Lightweight CSP-style multi-branch block (YOLO11 C3k2 pattern)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * c_, 1, 1)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut=shortcut, e=1.0) for _ in range(n))
        self.cv2 = ConvBNAct((2 + n) * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.m:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class C2PSA(nn.Module):
    """Lightweight Spatial Self-Attention block on low-res P5 feature map (SDPA-safe)."""

    def __init__(self, c: int, num_heads: int = 4) -> None:
        super().__init__()
        self.c_half = c // 2
        self.num_heads = num_heads
        self.head_dim = self.c_half // num_heads
        self.cv1 = ConvBNAct(c, c, 1, 1)
        self.qkv = nn.Conv2d(self.c_half, self.c_half * 3, 1, bias=False)
        self.proj = ConvBNAct(self.c_half, self.c_half, 1, 1)
        self.cv2 = ConvBNAct(c, c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c_half, self.c_half), dim=1)
        bs, c_h, h, w = b.shape
        qkv = self.qkv(b).view(bs, 3, self.num_heads, self.head_dim, h * w).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        b_out = attn.permute(0, 1, 3, 2).reshape(bs, c_h, h, w)
        b_out = b + self.proj(b_out)
        return self.cv2(torch.cat((a, b_out), dim=1))


class DecoupledFaceHead(nn.Module):
    """Anchor-free decoupled classification, DFL bbox (4 * reg_max), and 5-pt keypoint (5 * 3) head."""

    def __init__(self, ch_list: List[int], strides: List[int], reg_max: int = 16, num_kpts: int = 5) -> None:
        super().__init__()
        self.strides = strides
        self.reg_max = reg_max
        self.num_kpts = num_kpts
        self.no_box = 4 * reg_max
        self.no_kpt = num_kpts * 3

        c2 = max(24, ch_list[0] // 2)
        c3 = max(32, ch_list[0] // 2)
        self.cls_convs = nn.ModuleList(
            nn.Sequential(ConvBNAct(ch, c2, 3), ConvBNAct(c2, c2, 3), nn.Conv2d(c2, 1, 1)) for ch in ch_list
        )
        self.box_convs = nn.ModuleList(
            nn.Sequential(ConvBNAct(ch, c3, 3), ConvBNAct(c3, c3, 3), nn.Conv2d(c3, self.no_box, 1)) for ch in ch_list
        )
        self.kpt_convs = nn.ModuleList(
            nn.Sequential(ConvBNAct(ch, c3, 3), ConvBNAct(c3, c3, 3), nn.Conv2d(c3, self.no_kpt, 1)) for ch in ch_list
        )
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32), persistent=False)
        self._init_biases()

    def _init_biases(self) -> None:
        for conv_seq in self.cls_convs:
            nn.init.constant_(conv_seq[-1].bias, -4.5)  # ~0.01 prior prob

    def forward_raw(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        cls_outs = [m(f) for m, f in zip(self.cls_convs, feats)]
        box_outs = [m(f) for m, f in zip(self.box_convs, feats)]
        kpt_outs = [m(f) for m, f in zip(self.kpt_convs, feats)]
        return cls_outs, box_outs, kpt_outs

    def decode(
        self,
        cls_outs: List[torch.Tensor],
        box_outs: List[torch.Tensor],
        kpt_outs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode multi-level outputs into `[B, N, ...]` tensors in input pixel coordinates.

        Returns:
            boxes_xyxy: `[B, N, 4]`
            scores: `[B, N]` (sigmoid probs)
            kpts: `[B, N, 5, 3]` (`x, y` in pixels + visibility logit)
            cls_logits_flat: `[B, N]`
            dfl_dist_flat: `[B, N, 4, reg_max]`
        """
        b = cls_outs[0].shape[0]
        device = cls_outs[0].device
        dtype = cls_outs[0].dtype

        all_boxes: List[torch.Tensor] = []
        all_cls: List[torch.Tensor] = []
        all_kpts: List[torch.Tensor] = []
        all_dfl: List[torch.Tensor] = []
        all_anchors: List[torch.Tensor] = []
        all_strides: List[torch.Tensor] = []

        proj = self.proj.to(device=device, dtype=torch.float32)

        for cls_l, box_l, kpt_l, stride in zip(cls_outs, box_outs, kpt_outs, self.strides):
            _, _, fh, fw = cls_l.shape
            n_loc = fh * fw

            sy = torch.arange(fh, device=device, dtype=torch.float32) + 0.5
            sx = torch.arange(fw, device=device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_xy = torch.stack((gx, gy), dim=-1).reshape(1, n_loc, 2)  # grid units
            all_anchors.append(anchor_xy)
            all_strides.append(torch.full((1, n_loc, 1), float(stride), device=device, dtype=torch.float32))

            cls_flat = cls_l.permute(0, 2, 3, 1).reshape(b, n_loc)
            box_dfl = box_l.permute(0, 2, 3, 1).reshape(b, n_loc, 4, self.reg_max)
            dfl_probs = F.softmax(box_dfl.float(), dim=-1)
            ltrb = (dfl_probs * proj).sum(dim=-1) * float(stride)  # [B, n_loc, 4]

            cx = anchor_xy[..., 0] * float(stride)
            cy = anchor_xy[..., 1] * float(stride)
            x1 = cx - ltrb[..., 0]
            y1 = cy - ltrb[..., 1]
            x2 = cx + ltrb[..., 2]
            y2 = cy + ltrb[..., 3]
            boxes = torch.stack((x1, y1, x2, y2), dim=-1)

            kpt_raw = kpt_l.permute(0, 2, 3, 1).reshape(b, n_loc, self.num_kpts, 3).float()
            kx = (kpt_raw[..., 0] * 2.0 + (anchor_xy[..., 0:1] - 0.5)) * float(stride)
            ky = (kpt_raw[..., 1] * 2.0 + (anchor_xy[..., 1:2] - 0.5)) * float(stride)
            kv = kpt_raw[..., 2]
            kpts = torch.stack((kx, ky, kv), dim=-1)

            all_boxes.append(boxes)
            all_cls.append(cls_flat.float())
            all_kpts.append(kpts)
            all_dfl.append(box_dfl.float())

        boxes_xyxy = torch.cat(all_boxes, dim=1)
        cls_logits = torch.cat(all_cls, dim=1)
        scores = torch.sigmoid(cls_logits)
        kpts_cat = torch.cat(all_kpts, dim=1)
        dfl_cat = torch.cat(all_dfl, dim=1)
        return boxes_xyxy, scores, kpts_cat, cls_logits, dfl_cat


class LCFYOLO11Face5pt(nn.Module):
    """LCF + pure-PyTorch YOLO11-style Anchor-Free Face + 5-pt Detector (~5-6 MB FP32 ONNX)."""

    def __init__(
        self,
        use_lcf: bool = True,
        use_p2: bool = True,
        width_mult: float = 0.22,
        reg_max: int = 8,
    ) -> None:
        super().__init__()
        self.use_lcf = use_lcf
        self.use_p2 = use_p2
        self.lcf = LearnableCurveFrontend(in_ch=3, hidden_ch=16, num_iters=4) if use_lcf else nn.Identity()

        def ch(base: int) -> int:
            return max(16, int(round(base * width_mult / 8.0) * 8))

        c1 = ch(64)   # P1 (/2)
        c2 = ch(128)  # P2 (/4)
        c3 = ch(256)  # P3 (/8)
        c4 = ch(512)  # P4 (/16)
        c5 = ch(768)  # P5 (/32)

        # Backbone
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), C3k2(c2, c2, n=1))
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), C3k2(c3, c3, n=2))
        self.stage4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), C3k2(c4, c4, n=2))
        self.stage5 = nn.Sequential(ConvBNAct(c4, c5, 3, 2), C3k2(c5, c5, n=1), SPPF(c5, c5), C2PSA(c5, num_heads=4))

        # Top-down FPN
        self.up_p5_p4 = ConvBNAct(c5 + c4, c4, 1, 1)
        self.fpn_p4 = C3k2(c4, c4, n=1, shortcut=False)

        self.up_p4_p3 = ConvBNAct(c4 + c3, c3, 1, 1)
        self.fpn_p3 = C3k2(c3, c3, n=1, shortcut=False)

        if self.use_p2:
            self.up_p3_p2 = ConvBNAct(c3 + c2, c2, 1, 1)
            self.fpn_p2 = C3k2(c2, c2, n=1, shortcut=False)
            self.down_p2_p3 = ConvBNAct(c2, c2, 3, 2)
            self.pan_p3 = C3k2(c2 + c3, c3, n=1, shortcut=False)
        else:
            self.pan_p3 = nn.Identity()

        # Bottom-up PAN
        self.down_p3_p4 = ConvBNAct(c3, c3, 3, 2)
        self.pan_p4 = C3k2(c3 + c4, c4, n=1, shortcut=False)
        self.down_p4_p5 = ConvBNAct(c4, c4, 3, 2)
        self.pan_p5 = C3k2(c4 + c5, c5, n=1, shortcut=False)

        if self.use_p2:
            head_ch = [c2, c3, c4, c5]
            strides = [4, 8, 16, 32]
        else:
            head_ch = [c3, c4, c5]
            strides = [8, 16, 32]

        self.fpn_channels = [c3, c4, c5]
        self.head = DecoupledFaceHead(head_ch, strides=strides, reg_max=reg_max, num_kpts=5)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        """Forward RGB `[B, 3, H, W]` in `[0, 1]` returning decoded predictions and P3/P4/P5 features."""
        x_enh = self.lcf(x)
        p1 = self.stage2(self.stem(x_enh))
        p2 = p1  # stride 4
        p3 = self.stage3(p2)  # stride 8
        p4 = self.stage4(p3)  # stride 16
        p5 = self.stage5(p4)  # stride 32

        n4 = self.fpn_p4(self.up_p5_p4(torch.cat((F.interpolate(p5, scale_factor=2.0, mode="nearest"), p4), dim=1)))
        n3 = self.fpn_p3(self.up_p4_p3(torch.cat((F.interpolate(n4, scale_factor=2.0, mode="nearest"), p3), dim=1)))

        if self.use_p2:
            n2 = self.fpn_p2(self.up_p3_p2(torch.cat((F.interpolate(n3, scale_factor=2.0, mode="nearest"), p2), dim=1)))
            o3 = self.pan_p3(torch.cat((self.down_p2_p3(n2), n3), dim=1))
        else:
            n2 = None
            o3 = n3

        o4 = self.pan_p4(torch.cat((self.down_p3_p4(o3), n4), dim=1))
        o5 = self.pan_p5(torch.cat((self.down_p4_p5(o4), p5), dim=1))

        head_feats = [n2, o3, o4, o5] if self.use_p2 else [o3, o4, o5]
        cls_outs, box_outs, kpt_outs = self.head.forward_raw(head_feats)
        boxes_xyxy, scores, kpts, cls_logits, dfl_dist = self.head.decode(cls_outs, box_outs, kpt_outs)

        return {
            "enhanced_rgb": x_enh,
            "boxes_xyxy": boxes_xyxy,
            "scores": scores,
            "kpts": kpts,
            "cls_logits": cls_logits,
            "dfl_dist": dfl_dist,
            "fpn_features": [o3, o4, o5],  # always P3, P4, P5 for HLA distillation
        }


class DetectorONNXWrapper(nn.Module):
    """Clean ONNX-exportable wrapper outputting `(boxes_xyxy, scores, kpts5_xy)`."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, rgb_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.detector(rgb_01)
        return out["boxes_xyxy"], out["scores"], out["kpts"][..., :2]


def build_detector_teacher(use_p2: bool = True) -> LCFYOLO11Face5pt:
    """Build wider normal-light anchor teacher (YOLO11l-scale pure PyTorch, no LCF)."""
    return LCFYOLO11Face5pt(use_lcf=False, use_p2=use_p2, width_mult=0.75, reg_max=16)
