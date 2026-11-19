"""Pure-PyTorch EdgeFace-S / EdgeFace-XS (SDTA + LoRaLin) and MobileFaceNet fallback."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRaLin(nn.Module):
    """Low-Rank Linear layer (`y = W_b(W_a(x)) + b`) from EdgeFace (IEEE T-BIOM 2024)."""

    def __init__(self, in_features: int, out_features: int, rank_ratio: float = 0.5, bias: bool = True) -> None:
        super().__init__()
        rank = max(8, int(round(min(in_features, out_features) * rank_ratio)))
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm safe for static & ONNX opset 17 export."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_norm + self.bias.view(1, -1, 1, 1)


class ConvEncoderBlock(nn.Module):
    """EdgeNeXt/EdgeFace convolutional encoder block with optional LoRaLin inverted bottleneck."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expan_ratio: float = 4.0,
        use_lora: bool = True,
        lora_rank_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * expan_ratio)
        if use_lora:
            self.pwconv1 = LoRaLin(dim, hidden, rank_ratio=lora_rank_ratio)
            self.pwconv2 = LoRaLin(hidden, dim, rank_ratio=lora_rank_ratio)
        else:
            self.pwconv1 = nn.Linear(dim, hidden)
            self.pwconv2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return residual + x


class XCA(nn.Module):
    """Cross-Covariance Attention across channel dimension (linear in spatial tokens, ONNX-friendly)."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        head_dim = c // self.num_heads
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, head_dim).permute(2, 0, 3, 4, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, hd, N]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, H, hd, hd]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 3, 1, 2).reshape(b, n, c)
        return self.proj(out)


class SDTABlock(nn.Module):
    """Split Depth-wise Transpose Attention block from EdgeNeXt / EdgeFace."""

    def __init__(
        self,
        dim: int,
        num_scales: int = 2,
        num_heads: int = 8,
        use_lora: bool = True,
        lora_rank_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        width = max(int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales)))
        self.width = width
        self.nums = max(1, num_scales - 1)
        convs = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)

        self.norm_xca = nn.LayerNorm(dim, eps=1e-6)
        self.xca = XCA(dim, num_heads=num_heads)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * 4
        if use_lora:
            self.ffn1 = LoRaLin(dim, hidden, rank_ratio=lora_rank_ratio)
            self.ffn2 = LoRaLin(hidden, dim, rank_ratio=lora_rank_ratio)
        else:
            self.ffn1 = nn.Linear(dim, hidden)
            self.ffn2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.gamma_xca = nn.Parameter(torch.ones(dim) * 1e-6)
        self.gamma_ffn = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        spx = torch.split(x, self.width, dim=1)
        out_splits: List[torch.Tensor] = []
        sp = spx[0]
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            out_splits.append(sp)
        if len(spx) > self.nums:
            out_splits.append(spx[-1])
        x = torch.cat(out_splits, dim=1)

        b, c, h, w = x.shape
        tok = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        tok = tok + self.gamma_xca * self.xca(self.norm_xca(tok))
        tok = tok + self.gamma_ffn * self.ffn2(self.act(self.ffn1(self.norm_ffn(tok))))
        x = tok.transpose(1, 2).reshape(b, c, h, w)
        return residual + x


class EdgeFaceBackbone(nn.Module):
    """EdgeFace recognition model (`112x112` RGB in `[-1, 1]` -> `(l2_emb [B, 512], feat_norm [B, 1])`)."""

    def __init__(
        self,
        dims: Tuple[int, int, int, int],
        depths: Tuple[int, int, int, int],
        num_heads: Tuple[int, int, int, int] = (4, 4, 8, 8),
        emb_dim: int = 512,
        use_lora: bool = True,
        lora_rank_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.dims = dims

        # Downsample stem + stages (112 -> 28 -> 14 -> 7 -> 4)
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0]),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks: List[nn.Module] = []
            for j in range(depths[i]):
                if i >= 1 and j == depths[i] - 1:
                    blocks.append(
                        SDTABlock(
                            dims[i],
                            num_scales=2 if i == 1 else 3,
                            num_heads=num_heads[i],
                            use_lora=use_lora,
                            lora_rank_ratio=lora_rank_ratio,
                        )
                    )
                else:
                    ks = 3 if i == 0 else (5 if i == 1 else 7)
                    blocks.append(
                        ConvEncoderBlock(
                            dims[i],
                            kernel_size=ks,
                            expan_ratio=4.0,
                            use_lora=use_lora,
                            lora_rank_ratio=lora_rank_ratio,
                        )
                    )
            self.stages.append(nn.Sequential(*blocks))

        self.head_norm = nn.LayerNorm(dims[-1])
        self.head_fc = nn.Linear(dims[-1], emb_dim, bias=False)
        self.head_bn = nn.BatchNorm1d(emb_dim, affine=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward normalized RGB `[-1, 1]` `[B, 3, 112, 112]`.

        Returns:
            l2_emb: `[B, 512]` L2-normalized embedding (in FP32 for cosine stability).
            feat_norm: `[B, 1]` raw pre-normalization embedding norm `||z||`.
        """
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        pooled = self.head_norm(x.mean(dim=(-2, -1)))
        raw = self.head_bn(self.head_fc(pooled)).float()
        feat_norm = torch.norm(raw, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        l2_emb = raw / feat_norm
        return l2_emb, feat_norm


def EdgeFaceXS(emb_dim: int = 512) -> EdgeFaceBackbone:
    """EdgeFace-XS (~1.77M params, ~7 MB FP32 ONNX)."""
    return EdgeFaceBackbone(
        dims=(36, 72, 136, 224),
        depths=(2, 2, 5, 2),
        num_heads=(4, 4, 8, 8),
        emb_dim=emb_dim,
        use_lora=True,
        lora_rank_ratio=0.5,
    )


def EdgeFaceS(emb_dim: int = 512) -> EdgeFaceBackbone:
    """EdgeFace-S (~3.65M params, ~14 MB FP32 ONNX)."""
    return EdgeFaceBackbone(
        dims=(48, 96, 184, 320),
        depths=(3, 3, 7, 3),
        num_heads=(4, 4, 8, 8),
        emb_dim=emb_dim,
        use_lora=True,
        lora_rank_ratio=0.5,
    )


class MobileFaceNetFallback(nn.Module):
    """Pure-PyTorch MobileFaceNet (w=1.5) ultra-compatible CPU fallback."""

    def __init__(self, width_mult: float = 1.5, emb_dim: int = 512) -> None:
        super().__init__()

        def c(v: int) -> int:
            return max(16, int(round(v * width_mult / 8.0) * 8))

        def dw_block(inc: int, outc: int, s: int = 1, exp: int = 2) -> nn.Sequential:
            hid = inc * exp
            return nn.Sequential(
                nn.Conv2d(inc, hid, 1, bias=False),
                nn.BatchNorm2d(hid),
                nn.PReLU(hid),
                nn.Conv2d(hid, hid, 3, stride=s, padding=1, groups=hid, bias=False),
                nn.BatchNorm2d(hid),
                nn.PReLU(hid),
                nn.Conv2d(hid, outc, 1, bias=False),
                nn.BatchNorm2d(outc),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, c(64), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(64)),
            nn.PReLU(c(64)),
            dw_block(c(64), c(64), 2, 2),
            dw_block(c(64), c(128), 2, 4),
            dw_block(c(128), c(128), 1, 2),
            dw_block(c(128), c(128), 2, 4),
            nn.Conv2d(c(128), emb_dim, 1, bias=False),
            nn.BatchNorm2d(emb_dim),
            nn.PReLU(emb_dim),
            nn.Conv2d(emb_dim, emb_dim, 7, groups=emb_dim, bias=False),
            nn.BatchNorm2d(emb_dim),
        )
        self.fc = nn.Linear(emb_dim, emb_dim, bias=False)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x).flatten(1)
        raw = self.bn(self.fc(feat)).float()
        norm = torch.norm(raw, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        return raw / norm, norm


def build_recognizer_student(arch: str = "edgeface_s", emb_dim: int = 512) -> nn.Module:
    if arch == "edgeface_s":
        return EdgeFaceS(emb_dim=emb_dim)
    if arch == "edgeface_xs":
        return EdgeFaceXS(emb_dim=emb_dim)
    if arch == "mobilefacenet_w15":
        return MobileFaceNetFallback(width_mult=1.5, emb_dim=emb_dim)
    raise ValueError(f"Unknown recognizer arch: {arch}")
