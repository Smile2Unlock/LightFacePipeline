"""Task-driven Learnable Curve Frontend (LCF, Zero-DCE style, <30K params, zero-init identity)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWSepConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, act: bool = True) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=True)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class LearnableCurveFrontend(nn.Module):
    """Lightweight 3-layer depthwise-separable curve estimator at 1/4 resolution.

    Predicts per-iteration pixel-wise curves `A_k(x, y) in [-1, 1]^3` at 1/4 resolution,
    upsamples bilinearly, and applies `num_iters` Zero-DCE curve updates:
        I_{k+1} = I_k + A_k * I_k * (1 - I_k)

    Zero-initialized output projection ensures exact identity (`I_out == I_in`) at init
    and preserves bright normal-light images when `A ~ 0`.
    """

    def __init__(self, in_ch: int = 3, hidden_ch: int = 16, num_iters: int = 4) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.net = nn.Sequential(
            DWSepConv(in_ch, hidden_ch, act=True),
            DWSepConv(hidden_ch, hidden_ch, act=True),
            DWSepConv(hidden_ch, in_ch * num_iters, act=False),
        )
        # Exact zero initialization -> A == 0 at init -> identity mapping
        nn.init.zeros_(self.net[-1].pw.weight)
        if self.net[-1].pw.bias is not None:
            nn.init.zeros_(self.net[-1].pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance raw RGB `[B, 3, H, W]` in `[0, 1]` for downstream detection/landmarks."""
        low = F.avg_pool2d(x, kernel_size=4, stride=4)
        curves_low = torch.tanh(self.net(low))  # [B, 3*num_iters, H/4, W/4]
        curves = F.interpolate(curves_low, scale_factor=4.0, mode="bilinear", align_corners=False)

        img = x
        for k in range(self.num_iters):
            a_k = curves[:, 3 * k : 3 * (k + 1)]
            img = img + a_k * img * (1.0 - img)
        return img.clamp(0.0, 1.0)
