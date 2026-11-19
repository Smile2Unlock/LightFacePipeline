"""ST-FAS-Lite: Dual-scale, FIQA-FAS quality-aware, streaming-equivalent temporal liveness (~3-5 MB FP32)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDC2d(nn.Module):
    """Pure-PyTorch 3x3 Central Difference Convolution high-pass operator (ONNX-friendly)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, theta: float = 0.7) -> None:
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_normal = self.conv(x)
        if self.theta == 0.0:
            return out_normal
        # Sum of kernel weights over 3x3 neighborhood applied as 1x1 conv
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(x, kernel_diff, bias=None, stride=1, padding=0)
        return out_normal - self.theta * out_diff


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXtV2."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.sqrt((x * x).sum(dim=(2, 3), keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 3.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)  # channels-first LayerNorm equivalent, very fast on ROCm & ONNX
        hidden = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pwconv2 = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(self.dwconv(x))
        x = self.pwconv2(self.grn(self.act(self.pwconv1(x))))
        return residual + x


class ConvNeXtV2LiteBranch(nn.Module):
    """Lightweight ConvNeXtV2-Atto-style encoder (`128x128` -> `8x8` feature map + pooled vector)."""

    def __init__(self, in_ch: int, dims: Tuple[int, int, int], depths: Tuple[int, int, int]) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], kernel_size=4, stride=4),  # 128 -> 32
            nn.GroupNorm(1, dims[0]),
        )
        self.stage1 = nn.Sequential(*(ConvNeXtV2Block(dims[0]) for _ in range(depths[0])))
        self.down1 = nn.Sequential(nn.GroupNorm(1, dims[0]), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))  # 32 -> 16
        self.stage2 = nn.Sequential(*(ConvNeXtV2Block(dims[1]) for _ in range(depths[1])))
        self.down2 = nn.Sequential(nn.GroupNorm(1, dims[1]), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))  # 16 -> 8
        self.stage3 = nn.Sequential(*(ConvNeXtV2Block(dims[2]) for _ in range(depths[2])))
        self.out_norm = nn.GroupNorm(1, dims[2])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down1(x))
        fmap = self.out_norm(self.stage3(self.down2(x)))  # [B, dims[2], 8, 8]
        vec = fmap.mean(dim=(-2, -1))
        return vec, fmap


class STFASSpatialEncoder(nn.Module):
    """Single-frame dual-scale spatial encoder (`tex_curr, tex_prev, ctx_curr -> spatial_feat, quality, fmap`)."""

    def __init__(self, feat_dim: int = 192) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.cdc = CDC2d(3, 3, theta=0.7)

        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        # 1.5x Texture branch: [norm_RGB_t(3), |tex_curr - tex_prev|(3), CDC(tex_curr)(3)] = 9ch
        self.tex_branch = ConvNeXtV2LiteBranch(in_ch=9, dims=(40, 80, 144), depths=(1, 2, 2))
        # 2.7x Context branch (0.5x width): [norm_RGB_t(3)] = 3ch
        self.ctx_branch = ConvNeXtV2LiteBranch(in_ch=3, dims=(20, 40, 72), depths=(1, 1, 1))

        self.fusion = nn.Sequential(
            nn.Linear(144 + 72, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        # FIQA-FAS frame quality scorer
        self.quality_head = nn.Sequential(
            nn.Linear(feat_dim, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        # Train-only Pseudo-3D depth map head (automatically excluded in ONNX deploy wrappers)
        self.depth_head = nn.Sequential(
            nn.Conv2d(144, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _norm_rgb(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.im_mean.to(device=x.device, dtype=x.dtype)) / self.im_std.to(device=x.device, dtype=x.dtype)

    def forward_train(
        self,
        tex_curr: torch.Tensor,
        tex_prev: torch.Tensor,
        ctx_curr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tex_norm = self._norm_rgb(tex_curr)
        ctx_norm = self._norm_rgb(ctx_curr)
        diff = torch.abs(tex_curr - tex_prev)
        cdc_hi = self.cdc(tex_curr)
        tex_in = torch.cat((tex_norm, diff, cdc_hi), dim=1)

        v_tex, fmap_tex = self.tex_branch(tex_in)
        v_ctx, _ = self.ctx_branch(ctx_norm)

        feat = self.fusion(torch.cat((v_tex, v_ctx), dim=-1))
        quality_logit = self.quality_head(feat)
        depth_map = self.depth_head(fmap_tex)
        return feat, quality_logit, fmap_tex, depth_map

    def forward(
        self,
        tex_curr: torch.Tensor,
        tex_prev: torch.Tensor,
        ctx_curr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deployment forward (prunes depth head & dense fmap)."""
        tex_norm = self._norm_rgb(tex_curr)
        ctx_norm = self._norm_rgb(ctx_curr)
        diff = torch.abs(tex_curr - tex_prev)
        cdc_hi = self.cdc(tex_curr)
        tex_in = torch.cat((tex_norm, diff, cdc_hi), dim=1)

        v_tex, _ = self.tex_branch(tex_in)
        v_ctx, _ = self.ctx_branch(ctx_norm)
        feat = self.fusion(torch.cat((v_tex, v_ctx), dim=-1))
        quality_logit = self.quality_head(feat)
        return feat, quality_logit


class TemporalAttentionBlock(nn.Module):
    """Lightweight Temporal Self-Attention (dim=192, heads=4, pure PyTorch SDPA)."""

    def __init__(self, dim: int = 192, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class STFASTemporalHead(nn.Module):
    """1D-DWConv(k=3) + 2-layer Temporal Self-Attention + FIQA-FAS Quality Pooling + 5-cls/Clarity Head."""

    def __init__(self, feat_dim: int = 192, clip_len: int = 8, num_classes: int = 5) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.clip_len = clip_len
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_len, feat_dim))
        self.dw_temp_conv = nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        self.temp_norm = nn.LayerNorm(feat_dim)
        self.attn_blocks = nn.ModuleList([TemporalAttentionBlock(dim=feat_dim, num_heads=4) for _ in range(2)])
        self.out_norm = nn.LayerNorm(feat_dim)
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.clarity_head = nn.Sequential(nn.Linear(feat_dim, 48), nn.GELU(), nn.Linear(48, 1), nn.Sigmoid())

    def forward_with_feat(
        self,
        spatial_seq: torch.Tensor,
        quality_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1D-DWConv over time [B, T, D] -> [B, D, T] -> [B, T, D]
        conv_res = self.dw_temp_conv(spatial_seq.transpose(1, 2)).transpose(1, 2)
        x = self.temp_norm(spatial_seq + conv_res) + self.pos_embed[:, : spatial_seq.shape[1], :]
        for blk in self.attn_blocks:
            x = blk(x)
        x = self.out_norm(x)

        # FIQA-FAS quality-weighted temporal pooling
        weights = F.softmax(quality_seq, dim=1)  # [B, T, 1]
        pooled = (x * weights).sum(dim=1)  # [B, D]

        spoof_logits = self.cls_head(pooled).float()
        live_prob = F.softmax(spoof_logits, dim=-1)[:, 0:1]
        clarity_score = self.clarity_head(pooled).float()
        return live_prob, spoof_logits, clarity_score, pooled

    def forward(
        self,
        spatial_seq: torch.Tensor,
        quality_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        live_prob, spoof_logits, clarity_score, _ = self.forward_with_feat(spatial_seq, quality_seq)
        return live_prob, spoof_logits, clarity_score


class STFASLiteClip(nn.Module):
    """Full clip ST-FAS-Lite composing the exact `STFASSpatialEncoder` + `STFASTemporalHead`."""

    def __init__(self, feat_dim: int = 192, clip_len: int = 8, num_classes: int = 5) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.clip_len = clip_len
        self.spatial_enc = STFASSpatialEncoder(feat_dim=feat_dim)
        self.temporal_head = STFASTemporalHead(feat_dim=feat_dim, clip_len=clip_len, num_classes=num_classes)

    def forward_train(
        self,
        clip_tex: torch.Tensor,
        clip_ctx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, t, c, h, w = clip_tex.shape
        tex_prev = torch.cat((clip_tex[:, 0:1], clip_tex[:, :-1]), dim=1)
        feat, q_logit, fmap_tex, depth_map = self.spatial_enc.forward_train(
            clip_tex.reshape(b * t, c, h, w),
            tex_prev.reshape(b * t, c, h, w),
            clip_ctx.reshape(b * t, c, h, w),
        )
        spatial_seq = feat.view(b, t, self.feat_dim)
        quality_seq = q_logit.view(b, t, 1)
        live_prob, spoof_logits, clarity_score, pooled = self.temporal_head.forward_with_feat(spatial_seq, quality_seq)
        return {
            "live_prob": live_prob,
            "spoof_logits": spoof_logits,
            "clarity_score": clarity_score,
            "pooled_feat": pooled,
            "tex_fmap": fmap_tex.view(b, t, fmap_tex.shape[1], fmap_tex.shape[2], fmap_tex.shape[3]),
            "depth_pred": depth_map.view(b, t, 1, depth_map.shape[2], depth_map.shape[3]),
            "frame_quality_logits": quality_seq,
        }

    def forward(
        self,
        clip_tex: torch.Tensor,
        clip_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact inference forward matching streaming (`fas_spatial_enc` + `fas_temporal_head`)."""
        b, t, c, h, w = clip_tex.shape
        tex_prev = torch.cat((clip_tex[:, 0:1], clip_tex[:, :-1]), dim=1)
        feat, q_logit = self.spatial_enc(
            clip_tex.reshape(b * t, c, h, w),
            tex_prev.reshape(b * t, c, h, w),
            clip_ctx.reshape(b * t, c, h, w),
        )
        spatial_seq = feat.view(b, t, self.feat_dim)
        quality_seq = q_logit.view(b, t, 1)
        return self.temporal_head(spatial_seq, quality_seq)
