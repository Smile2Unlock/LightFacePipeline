"""GPU-vectorized physical low-light / backlight / UVC degradation engine & FAS physical augmentor."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.contracts import (
    DET_MIN_FACE_LUMA_255,
    REC_SAFE_LUMA_CEIL_255,
    REC_SAFE_LUMA_FLOOR_255,
    REC_TRIGGER_LUMA_FLOOR_255,
)


def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3, H, W] RGB in [0, 1] to YCbCr (Y in [0, 1], Cb/Cr in [-0.5, 0.5])."""
    r, g, b = x.unbind(dim=-3)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.stack((y, cb, cr), dim=-3)


def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert [..., 3, H, W] YCbCr back to RGB clamped in [0, 1]."""
    y, cb, cr = ycbcr.unbind(dim=-3)
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack((r, g, b), dim=-3).clamp(0.0, 1.0)


def compute_rgb_luma_255(x: torch.Tensor) -> torch.Tensor:
    """Return per-image mean BT.601 luma in [0, 255] for [..., 3, H, W] tensor in [0, 1]."""
    r, g, b = x.unbind(dim=-3)
    y = (0.299 * r + 0.587 * g + 0.114 * b) * 255.0
    return y.mean(dim=(-2, -1))


def _build_dct_matrix(n: int = 8) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(n, dtype=torch.float32)[None, :]
    alpha = torch.where(k == 0, math.sqrt(1.0 / n), math.sqrt(2.0 / n))
    return alpha * torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n))


# Standard JPEG luminance quantization table (Q=50 baseline)
_JPEG_LUMA_Q50 = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

_JPEG_CHROMA_Q50 = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


class PhysicalLowLightDegrader(nn.Module):
    """5-step GPU-vectorized physical low-light, backlight, noise, blur & UVC/MJPEG simulator.

    Supports both image batches `[B, 3, H, W]` and video clip batches `[B, T, 3, H, W]` in RGB `[0, 1]`.
    Across temporal dimension `T`, illumination/exposure/AWB stay coherent (with optional gentle AE drift)
    while shot/read sensor noise and DCT compression jitter are sampled i.i.d. per frame.
    """

    def __init__(
        self,
        mode: str = "detector",
        gamma_range: Tuple[float, float] = (2.0, 2.4),
        exp_range: Tuple[float, float] = (0.04, 0.50),
        shot_sigma_range: Tuple[float, float] = (0.003, 0.015),
        read_sigma_range: Tuple[float, float] = (0.0005, 0.004),
        blur_prob: float = 0.40,
        uvc_prob: float = 0.85,
        jpeg_quality_range: Tuple[int, int] = (45, 85),
    ) -> None:
        super().__init__()
        if mode not in ("detector", "recognizer", "liveness"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.gamma_range = gamma_range
        self.exp_range = exp_range
        self.shot_sigma_range = shot_sigma_range
        self.read_sigma_range = read_sigma_range
        self.blur_prob = blur_prob
        self.uvc_prob = uvc_prob
        self.jpeg_quality_range = jpeg_quality_range

        self.register_buffer("dct_mat", _build_dct_matrix(8), persistent=False)
        self.register_buffer("q_luma_50", _JPEG_LUMA_Q50, persistent=False)
        self.register_buffer("q_chroma_50", _JPEG_CHROMA_Q50, persistent=False)

    def _build_face_mask(
        self,
        b: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
        bboxes_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create soft elliptical face mask [B, 1, H, W] in [0, 1]."""
        yy = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)[None, None, :, None]
        xx = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)[None, None, None, :]

        if bboxes_norm is None:
            cx = torch.full((b, 1, 1, 1), 0.5, device=device, dtype=dtype)
            cy = torch.full((b, 1, 1, 1), 0.48, device=device, dtype=dtype)
            rx = torch.full((b, 1, 1, 1), 0.28, device=device, dtype=dtype)
            ry = torch.full((b, 1, 1, 1), 0.34, device=device, dtype=dtype)
        else:
            bb = bboxes_norm.to(device=device, dtype=dtype)
            cx = ((bb[:, 0] + bb[:, 2]) * 0.5).view(b, 1, 1, 1)
            cy = ((bb[:, 1] + bb[:, 3]) * 0.5).view(b, 1, 1, 1)
            rx = ((bb[:, 2] - bb[:, 0]).clamp(min=0.08) * 0.65).view(b, 1, 1, 1)
            ry = ((bb[:, 3] - bb[:, 1]).clamp(min=0.08) * 0.65).view(b, 1, 1, 1)

        dist2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        return torch.exp(-0.85 * dist2).clamp(0.0, 1.0)

    def _sample_spatial_field(
        self,
        b: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
        bboxes_norm: Optional[torch.Tensor] = None,
        field_probs: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mult_field [B,1,H,W], add_bg_field [B,1,H,W], face_mask [B,1,H,W])."""
        face_mask = self._build_face_mask(b, h, w, device, dtype, bboxes_norm)
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)[None, None, :, None]
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)[None, None, None, :]

        probs = torch.tensor(field_probs, device=device, dtype=torch.float32)
        kind = torch.multinomial(probs, b, replacement=True)  # 0:backlight, 1:screen, 2:shadow, 3:uniform

        mult = torch.ones((b, 1, h, w), device=device, dtype=dtype)
        bg_boost = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

        for i in range(b):
            k = int(kind[i].item())
            if k == 0:
                # Strong window backlight: dim face strongly, brighten exterior background without leaking white fog onto face
                face_atten = float(torch.empty(1, device=device).uniform_(0.25, 0.65).item())
                bg_gain = float(torch.empty(1, device=device).uniform_(1.4, 2.8).item())
                bg_lift = float(torch.empty(1, device=device).uniform_(0.22, 0.55).item())
                fm = face_mask[i : i + 1]
                bg_mask = torch.sigmoid((0.28 - fm) * 16.0)
                mult[i : i + 1] = (1.0 - bg_mask) * face_atten + bg_mask * bg_gain
                bg_boost[i : i + 1] = bg_mask * bg_lift
            elif k == 1:
                # Cold monitor glow from lower-center onto face, very dark background
                sx = float(torch.empty(1, device=device).uniform_(-0.15, 0.15).item())
                sy = float(torch.empty(1, device=device).uniform_(0.25, 0.65).item())
                rad2 = (xx - sx) ** 2 + (yy - sy) ** 2
                glow = torch.exp(-1.8 * rad2)
                mult[i : i + 1] = 0.12 + 1.15 * glow
            elif k == 2:
                # Half-face directional shadow / overhead cap shadow
                ang = float(torch.empty(1, device=device).uniform_(0.0, 2.0 * math.pi).item())
                proj = xx * math.cos(ang) + yy * math.sin(ang)
                grad = torch.sigmoid(proj * float(torch.empty(1, device=device).uniform_(2.5, 4.5).item()))
                mult[i : i + 1] = 0.25 + 0.85 * grad
            else:
                # Uniform dark
                mult[i : i + 1] = 1.0

        return mult, bg_boost, face_mask

    def _apply_optical_blur(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Apply slight motion or Gaussian defocus blur (3x3 - 7x7) on a subset of frames."""
        n, c, h, w = x_flat.shape
        out = x_flat.clone()
        apply_mask = torch.rand(n, device=x_flat.device) < self.blur_prob
        if not apply_mask.any():
            return out

        ksize = int(torch.tensor([3, 5, 7])[torch.randint(0, 3, (1,)).item()].item())
        pad = ksize // 2
        if torch.rand(1).item() < 0.6:
            # Directional motion blur kernel
            angle = float(torch.empty(1).uniform_(0.0, math.pi).item())
            kernel = torch.zeros((ksize, ksize), device=x_flat.device, dtype=x_flat.dtype)
            coords = torch.linspace(-pad, pad, ksize * 3, device=x_flat.device)
            xs = torch.round(coords * math.cos(angle)).long() + pad
            ys = torch.round(coords * math.sin(angle)).long() + pad
            valid = (xs >= 0) & (xs < ksize) & (ys >= 0) & (ys < ksize)
            kernel[ys[valid], xs[valid]] += 1.0
            kernel = kernel / kernel.sum().clamp(min=1.0)
        else:
            # Gaussian defocus blur
            sigma = float(torch.empty(1).uniform_(0.6, 1.4).item())
            ax = torch.arange(-pad, pad + 1, device=x_flat.device, dtype=x_flat.dtype)
            g = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
            kernel = torch.outer(g, g)
            kernel = kernel / kernel.sum()

        weight = kernel.view(1, 1, ksize, ksize).repeat(c, 1, 1, 1)
        idx = apply_mask.nonzero(as_tuple=False).squeeze(1)
        blurred = F.conv2d(F.pad(x_flat[idx], (pad, pad, pad, pad), mode="reflect"), weight, groups=c)
        out[idx] = blurred.clamp(0.0, 1.0)
        return out

    def _apply_uvc_and_dct_jpeg(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Simulate NV12/YUY2 4:2:0 chroma subsampling + 8x8 DCT quantization artifacts."""
        n, c, h, w = x_flat.shape
        out = x_flat
        if torch.rand(1, device=x_flat.device).item() >= self.uvc_prob:
            return out

        # 1) Chroma 2x subsampling (NV12/YUY2)
        ycbcr = _rgb_to_ycbcr(out)
        y = ycbcr[:, 0:1]
        chroma = ycbcr[:, 1:3]
        ch_h = max(1, h // 2)
        ch_w = max(1, w // 2)
        chroma_ds = F.interpolate(chroma, size=(ch_h, ch_w), mode="bilinear", align_corners=False)
        chroma_us = F.interpolate(chroma_ds, size=(h, w), mode="bilinear", align_corners=False)
        ycbcr = torch.cat((y, chroma_us), dim=1)

        # 2) Vectorized 8x8 DCT quantization (MJPEG blocking)
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h or pad_w:
            ycbcr_p = F.pad(ycbcr, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            ycbcr_p = ycbcr

        hp, wp = ycbcr_p.shape[-2:]
        q_val = int(torch.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1] + 1, (1,)).item())
        scale = (5000.0 / q_val) if q_val < 50 else (200.0 - 2.0 * q_val)
        q_luma = ((self.q_luma_50 * scale + 50.0) / 100.0).clamp(min=1.0).to(dtype=ycbcr_p.dtype)
        q_chroma = ((self.q_chroma_50 * scale + 50.0) / 100.0).clamp(min=1.0).to(dtype=ycbcr_p.dtype)

        d = self.dct_mat.to(device=ycbcr_p.device, dtype=ycbcr_p.dtype)
        dt = d.t()

        # Scale from [0, 1] / [-0.5, 0.5] to JPEG [-128, 127] space
        scaled = torch.empty_like(ycbcr_p)
        scaled[:, 0] = ycbcr_p[:, 0] * 255.0 - 128.0
        scaled[:, 1:3] = ycbcr_p[:, 1:3] * 255.0

        # Reshape into 8x8 blocks: [N, 3, hp//8, 8, wp//8, 8] -> [N, 3, nb_h, nb_w, 8, 8]
        nb_h, nb_w = hp // 8, wp // 8
        blocks = scaled.view(n, 3, nb_h, 8, nb_w, 8).permute(0, 1, 2, 4, 3, 5)
        dct_coeff = d @ blocks @ dt

        q_table = torch.stack((q_luma, q_chroma, q_chroma), dim=0).view(1, 3, 1, 1, 8, 8)
        # Straight-through round approximation + small uniform quant noise for differentiability if needed
        quant = torch.round(dct_coeff / q_table) * q_table
        rec_blocks = dt @ quant @ d
        rec = rec_blocks.permute(0, 1, 2, 4, 3, 5).reshape(n, 3, hp, wp)

        rec_ycbcr = torch.empty_like(rec)
        rec_ycbcr[:, 0] = (rec[:, 0] + 128.0) / 255.0
        rec_ycbcr[:, 1:3] = rec[:, 1:3] / 255.0
        if pad_h or pad_w:
            rec_ycbcr = rec_ycbcr[:, :, :h, :w]

        return _ycbcr_to_rgb(rec_ycbcr)

    def _enforce_luma_bounds(
        self,
        lin_exposed: torch.Tensor,
        inv_gamma: torch.Tensor,
        face_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Keep recognizer / detector face luma above physically valid lower bounds."""
        srgb_est = lin_exposed.clamp(0.0, 1.0).pow(1.0 / inv_gamma)
        r, g, b = srgb_est.unbind(dim=-3)
        luma_map = (0.299 * r + 0.587 * g + 0.114 * b) * 255.0  # [..., H, W]

        if face_mask is not None:
            fm = face_mask.squeeze(-3)
            denom = fm.sum(dim=(-2, -1)).clamp(min=1e-5)
            face_luma = (luma_map * fm).sum(dim=(-2, -1)) / denom
        else:
            face_luma = luma_map.mean(dim=(-2, -1))

        if self.mode == "recognizer":
            too_dark = face_luma < REC_TRIGGER_LUMA_FLOOR_255
            if too_dark.any():
                target_luma = REC_SAFE_LUMA_FLOOR_255 + 3.0
                ratio_srgb = (target_luma / face_luma.clamp(min=2.0)).clamp(min=1.0, max=12.0)
                boost = ratio_srgb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).pow(inv_gamma)
                boost = torch.where(
                    too_dark.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    boost,
                    torch.ones_like(boost),
                )
                lin_exposed = lin_exposed * boost
        elif self.mode == "detector":
            too_dark = face_luma < DET_MIN_FACE_LUMA_255
            if too_dark.any():
                ratio_srgb = (DET_MIN_FACE_LUMA_255 / face_luma.clamp(min=1.5)).clamp(min=1.0, max=8.0)
                boost = ratio_srgb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).pow(inv_gamma)
                boost = torch.where(
                    too_dark.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    boost,
                    torch.ones_like(boost),
                )
                lin_exposed = lin_exposed * boost
        return lin_exposed

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        bboxes_norm: Optional[torch.Tensor] = None,
        field_probs: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
    ) -> torch.Tensor:
        """Apply physical degradation to `[B, 3, H, W]` or `[B, T, 3, H, W]` RGB `[0, 1]` tensor."""
        is_clip = x.ndim == 5
        if not is_clip and x.ndim != 4:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

        orig_dtype = x.dtype
        work = x.float().clamp(0.0, 1.0)
        if not is_clip:
            work = work.unsqueeze(1)  # [B, 1, 3, H, W]

        b, t, c, h, w = work.shape
        device = work.device

        # 1) Inverse ISP parameters (shared across clip T, with smooth mild temporal drift)
        inv_gamma = torch.empty((b, 1, 1, 1, 1), device=device).uniform_(*self.gamma_range)
        alpha_exp = torch.empty((b, 1, 1, 1, 1), device=device).uniform_(*self.exp_range)
        if t > 1:
            # Smooth slow AE/head motion drift within ±8%
            t_axis = torch.linspace(-1.0, 1.0, t, device=device).view(1, t, 1, 1, 1)
            drift_slope = torch.empty((b, 1, 1, 1, 1), device=device).uniform_(-0.08, 0.08)
            alpha_t = alpha_exp * (1.0 + drift_slope * t_axis).clamp(min=0.5)
        else:
            alpha_t = alpha_exp

        lin = work.clamp(min=1e-4).pow(inv_gamma)

        # 2) Spatial non-uniform illumination field (shared across clip T)
        mult, bg_boost, face_mask = self._sample_spatial_field(
            b, h, w, device, torch.float32, bboxes_norm=bboxes_norm, field_probs=field_probs
        )
        lin_exposed = lin * alpha_t * mult.unsqueeze(1) + bg_boost.unsqueeze(1)

        # Enforce safety luma floor in linear domain
        lin_exposed = self._enforce_luma_bounds(
            lin_exposed,
            inv_gamma.expand(b, t, 1, 1, 1),
            face_mask.unsqueeze(1).expand(b, t, 1, h, w) if bboxes_norm is not None else None,
        )

        # 3) Sensor shot + read noise (STRICTLY i.i.d. per frame!) + AWB color drift (consistent across T)
        sigma_s = torch.empty((b, 1, 1, 1, 1), device=device).uniform_(*self.shot_sigma_range)
        sigma_r = torch.empty((b, 1, 1, 1, 1), device=device).uniform_(*self.read_sigma_range)

        # Warm (2800K: boost R, drop B) vs Cool (7500K: drop R, boost B)
        awb = torch.ones((b, 1, 3, 1, 1), device=device)
        warm_sign = torch.where(torch.rand((b, 1, 1, 1), device=device) < 0.5, 1.0, -1.0)
        shift = torch.empty((b, 1, 1, 1), device=device).uniform_(0.0, 0.16) * warm_sign
        awb[:, :, 0] = (1.0 + shift).clamp(0.80, 1.22)
        awb[:, :, 2] = (1.0 - shift).clamp(0.80, 1.22)

        shot_std = torch.sqrt(lin_exposed.clamp(min=0.0)) * sigma_s
        shot_noise = torch.randn_like(lin_exposed) * shot_std
        read_noise = torch.randn_like(lin_exposed) * sigma_r
        lin_noisy = (lin_exposed * awb + shot_noise + read_noise).clamp(0.0, 1.0)

        srgb = lin_noisy.pow(1.0 / inv_gamma).clamp(0.0, 1.0)

        # 4) Optical blur + 5) UVC 4:2:0 + MJPEG 8x8 DCT
        flat = srgb.view(b * t, c, h, w)
        flat = self._apply_optical_blur(flat)
        flat = self._apply_uvc_and_dct_jpeg(flat)

        out = flat.view(b, t, c, h, w)
        if not is_clip:
            out = out.squeeze(1)
        return out.to(dtype=orig_dtype)


class LivenessPhysicalAugmentor(nn.Module):
    """Differentiated low-light FAS augmentor (Strategy A) + Frozen-Rigid hard negatives (Strategy B)."""

    def __init__(self) -> None:
        super().__init__()
        self.live_degrader = PhysicalLowLightDegrader(
            mode="liveness",
            exp_range=(0.06, 0.55),
            blur_prob=0.35,
            uvc_prob=0.80,
        )
        self.print_degrader = PhysicalLowLightDegrader(
            mode="liveness",
            exp_range=(0.08, 0.50),
            blur_prob=0.25,
            uvc_prob=0.80,
        )
        self.sensor_tail = PhysicalLowLightDegrader(
            mode="liveness",
            exp_range=(0.98, 1.02),
            shot_sigma_range=(0.008, 0.03),
            read_sigma_range=(0.004, 0.02),
            blur_prob=0.20,
            uvc_prob=0.85,
        )

    @torch.no_grad()
    def augment_live(self, clips: torch.Tensor) -> torch.Tensor:
        """Apply realistic low-light/backlight/screen-glow + i.i.d. per-frame sensor noise to live clips."""
        return self.live_degrader(clips, field_probs=(0.35, 0.30, 0.15, 0.20))

    @torch.no_grad()
    def augment_print(self, clips: torch.Tensor) -> torch.Tensor:
        """Dim paper print clips with matte diffuse low-contrast haze + i.i.d. sensor noise."""
        b, t, c, h, w = clips.shape
        x = clips.float().clamp(0.0, 1.0)
        # Paper diffuse haze & reduced local contrast
        contrast = torch.empty((b, 1, 1, 1, 1), device=x.device).uniform_(0.55, 0.82)
        haze = torch.empty((b, 1, 1, 1, 1), device=x.device).uniform_(0.04, 0.14)
        mean_val = x.mean(dim=(-2, -1), keepdim=True)
        hazed = ((x - mean_val) * contrast + mean_val * 0.85 + haze).clamp(0.0, 1.0)
        return self.print_degrader(hazed, field_probs=(0.15, 0.15, 0.25, 0.45)).to(dtype=clips.dtype)

    @torch.no_grad()
    def augment_replay(self, clips: torch.Tensor, dark_dim_ratio: float = 0.30) -> torch.Tensor:
        """Preserve dark-room self-luminous screen physics (70% glowing cool screen, 30% 20%-dim screen)."""
        orig_dtype = clips.dtype
        x = clips.float().clamp(0.0, 1.0)
        b, t, c, h, w = x.shape
        device = x.device

        yy = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, 1, w)
        t_idx = torch.arange(t, device=device, dtype=torch.float32).view(1, t, 1, 1, 1)

        out = torch.empty_like(x)
        for i in range(b):
            clip = x[i : i + 1]
            is_dim_screen = torch.rand(1, device=device).item() < dark_dim_ratio

            # Dim peripheral border simulating dark room around phone/tablet screen
            border_dist = torch.maximum(xx.abs(), yy.abs())
            bezel_mask = torch.sigmoid((0.88 - border_dist) * 14.0)  # 1 inside screen, ~0 near edges

            if is_dim_screen:
                # 20% screen brightness replay in dark room
                screen_gain = float(torch.empty(1, device=device).uniform_(0.22, 0.42).item())
                clip = clip * screen_gain
            else:
                # Self-luminous screen: keeps medium-bright face + cool cyan-blue tint
                screen_gain = float(torch.empty(1, device=device).uniform_(0.65, 1.05).item())
                cool_gain = torch.tensor([0.92, 1.00, 1.12], device=device).view(1, 1, 3, 1, 1)
                clip = (clip * screen_gain * cool_gain).clamp(0.0, 1.0)

            # Subtle high-frequency RGB moire stripes + rolling 60/120Hz flicker across T
            freq = float(torch.empty(1, device=device).uniform_(18.0, 42.0).item())
            theta = float(torch.empty(1, device=device).uniform_(0.0, math.pi).item())
            moire_amp = float(torch.empty(1, device=device).uniform_(0.02, 0.06).item())
            moire = 1.0 + moire_amp * torch.sin(freq * (xx * math.cos(theta) + yy * math.sin(theta)))

            roll_speed = float(torch.empty(1, device=device).uniform_(0.4, 1.2).item())
            flicker_amp = float(torch.empty(1, device=device).uniform_(0.03, 0.09).item())
            flicker = 1.0 + flicker_amp * torch.sin(5.0 * yy + roll_speed * t_idx)

            halo = 0.04 * torch.exp(-((border_dist - 0.85) ** 2) / 0.01)
            clip = (clip * moire * flicker * (0.15 + 0.85 * bezel_mask) + halo).clamp(0.0, 1.0)
            out[i : i + 1] = clip

        # Camera sensor capture noise (i.i.d. per frame) + UVC/MJPEG
        out = self.sensor_tail(out, field_probs=(0.0, 0.0, 0.0, 1.0))
        return out.to(dtype=orig_dtype)

    @torch.no_grad()
    def synthesize_frozen_rigid_negative(self, live_clips: torch.Tensor) -> torch.Tensor:
        """Strategy B: Freeze a single low-light live frame, apply 2D rigid jitter or bend, add fresh i.i.d. noise."""
        orig_dtype = live_clips.dtype
        x = live_clips.float().clamp(0.0, 1.0)
        b, t, c, h, w = x.shape
        device = x.device

        # First ensure source frames match realistic low-light distribution if not already dark
        k_idx = torch.randint(0, t, (b,), device=device)
        frozen_frames = x[torch.arange(b, device=device), k_idx]  # [B, C, H, W]

        # Build smooth 2D rigid affine or gentle cylindrical paper-bend flow across T
        syn = torch.empty((b, t, c, h, w), device=device, dtype=torch.float32)
        base_yy, base_xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )
        for i in range(b):
            src = frozen_frames[i : i + 1].expand(t, -1, -1, -1)
            use_bend = torch.rand(1, device=device).item() < 0.35
            t_norm = torch.linspace(-1.0, 1.0, t, device=device).view(t, 1, 1)

            if not use_bend:
                # Smooth sinusoidal / random-walk rigid 2D hand-shake (translation + small rot + tiny scale)
                phase = float(torch.empty(1, device=device).uniform_(0.0, 2.0 * math.pi).item())
                tx = float(torch.empty(1, device=device).uniform_(0.01, 0.035).item()) * torch.sin(math.pi * t_norm + phase)
                ty = float(torch.empty(1, device=device).uniform_(0.01, 0.035).item()) * torch.cos(math.pi * t_norm + phase)
                ang = float(torch.empty(1, device=device).uniform_(0.01, 0.04).item()) * torch.sin(1.3 * math.pi * t_norm)
                sc = 1.0 + float(torch.empty(1, device=device).uniform_(0.005, 0.02).item()) * t_norm
                ca, sa = torch.cos(ang), torch.sin(ang)
                gx = (base_xx.unsqueeze(0) * ca - base_yy.unsqueeze(0) * sa) * sc + tx
                gy = (base_xx.unsqueeze(0) * sa + base_yy.unsqueeze(0) * ca) * sc + ty
            else:
                # Smooth cylindrical paper bend + micro shift
                amp = float(torch.empty(1, device=device).uniform_(0.02, 0.05).item()) * torch.sin(math.pi * t_norm)
                gx = base_xx.unsqueeze(0) + amp * (base_xx.unsqueeze(0) ** 2 - 0.3)
                gy = base_yy.unsqueeze(0) + 0.5 * amp * base_xx.unsqueeze(0)

            grid = torch.stack((gx, gy), dim=-1)
            warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
            syn[i] = warped

        # Apply fresh per-frame i.i.d. live camera sensor noise + YUV/MJPEG artifacts
        syn = self.sensor_tail(syn, field_probs=(0.0, 0.0, 0.0, 1.0))
        return syn.to(dtype=orig_dtype)

    @torch.no_grad()
    def _apply_joint_isp(
        self,
        tex_i: torch.Tensor,
        ctx_i: torch.Tensor,
        inv_gamma: float,
        exp_gain: float,
        drift_slope: float,
        field_kind: str,
        awb_shift: float,
        shot_sigma: float,
        read_sigma: float,
        contrast_scale: float = 1.0,
        haze_add: float = 0.0,
        is_replay: bool = False,
        replay_dim: bool = False,
        moire_freq: float = 28.0,
        moire_theta: float = 0.7,
        moire_amp: float = 0.03,
        roll_speed: float = 0.7,
        flicker_amp: float = 0.04,
        do_blur: bool = False,
        blur_ksize: int = 3,
        blur_angle: float = 0.5,
        is_motion_blur: bool = True,
        do_uvc: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply synchronized ISP / illumination / motion blur / UVC pipeline to a single (tex, ctx) clip pair."""
        _, t, c, h, w = tex_i.shape
        device = tex_i.device
        scale_ratio = 1.5 / 2.7  # tex FOV / ctx FOV

        if contrast_scale != 1.0 or haze_add != 0.0:
            mean_t = tex_i.mean(dim=(-2, -1), keepdim=True)
            mean_c = ctx_i.mean(dim=(-2, -1), keepdim=True)
            tex_i = ((tex_i - mean_t) * contrast_scale + mean_t * 0.88 + haze_add).clamp(0.0, 1.0)
            ctx_i = ((ctx_i - mean_c) * contrast_scale + mean_c * 0.88 + haze_add).clamp(0.0, 1.0)

        lin_t = tex_i.clamp(min=1e-4).pow(inv_gamma)
        lin_c = ctx_i.clamp(min=1e-4).pow(inv_gamma)

        t_axis = torch.linspace(-1.0, 1.0, t, device=device).view(1, t, 1, 1, 1)
        alpha_t = exp_gain * (1.0 + drift_slope * t_axis).clamp(min=0.4)

        yy = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, 1, w)
        rad2_t = ((xx * scale_ratio) / 0.35) ** 2 + ((yy * scale_ratio) / 0.42) ** 2
        fm_t = torch.exp(-0.85 * rad2_t).clamp(0.0, 1.0)
        rad2_c = (xx / 0.35) ** 2 + (yy / 0.42) ** 2
        fm_c = torch.exp(-0.85 * rad2_c).clamp(0.0, 1.0)

        mult_t = torch.ones((1, 1, 1, h, w), device=device)
        mult_c = torch.ones((1, 1, 1, h, w), device=device)
        bg_t = torch.zeros_like(mult_t)
        bg_c = torch.zeros_like(mult_c)

        if field_kind == "backlight":
            face_atten = 0.42
            bg_gain = 2.0
            bg_lift = 0.32
            bgm_t = torch.sigmoid((0.28 - fm_t) * 16.0)
            bgm_c = torch.sigmoid((0.28 - fm_c) * 16.0)
            mult_t = (1.0 - bgm_t) * face_atten + bgm_t * bg_gain
            mult_c = (1.0 - bgm_c) * face_atten + bgm_c * bg_gain
            bg_t = bgm_t * bg_lift
            bg_c = bgm_c * bg_lift
        elif field_kind == "screen_glow":
            r2_t = (xx * scale_ratio) ** 2 + (yy * scale_ratio - 0.35) ** 2
            r2_c = xx ** 2 + (yy - 0.35) ** 2
            mult_t = 0.16 + 1.10 * torch.exp(-1.8 * r2_t)
            mult_c = 0.16 + 1.10 * torch.exp(-1.8 * r2_c)
        elif field_kind == "shadow":
            proj_t = (xx * scale_ratio) * math.cos(blur_angle) + (yy * scale_ratio) * math.sin(blur_angle)
            proj_c = xx * math.cos(blur_angle) + yy * math.sin(blur_angle)
            mult_t = 0.30 + 0.85 * torch.sigmoid(proj_t * 3.2)
            mult_c = 0.30 + 0.85 * torch.sigmoid(proj_c * 3.2)

        lin_exp_t = lin_t * alpha_t * mult_t + bg_t
        lin_exp_c = lin_c * alpha_t * mult_c + bg_c

        awb = torch.ones((1, 1, 3, 1, 1), device=device)
        awb[:, :, 0] = max(0.80, min(1.22, 1.0 + awb_shift))
        awb[:, :, 2] = max(0.80, min(1.22, 1.0 - awb_shift))

        shot_t = torch.sqrt(lin_exp_t.clamp(min=0.0)) * shot_sigma
        shot_c = torch.sqrt(lin_exp_c.clamp(min=0.0)) * shot_sigma
        srgb_t = (lin_exp_t * awb + torch.randn_like(lin_exp_t) * shot_t + torch.randn_like(lin_exp_t) * read_sigma).clamp(0.0, 1.0).pow(1.0 / inv_gamma).clamp(0.0, 1.0)
        srgb_c = (lin_exp_c * awb + torch.randn_like(lin_exp_c) * shot_c + torch.randn_like(lin_exp_c) * read_sigma).clamp(0.0, 1.0).pow(1.0 / inv_gamma).clamp(0.0, 1.0)

        if is_replay:
            t_idx = torch.arange(t, device=device, dtype=torch.float32).view(1, t, 1, 1, 1)
            if not replay_dim:
                cool = torch.tensor([0.92, 1.00, 1.12], device=device).view(1, 1, 3, 1, 1)
                srgb_t = (srgb_t * cool).clamp(0.0, 1.0)
                srgb_c = (srgb_c * cool).clamp(0.0, 1.0)
            moire_t = 1.0 + moire_amp * torch.sin(moire_freq * (xx * math.cos(moire_theta) + yy * math.sin(moire_theta)))
            moire_c = 1.0 + moire_amp * torch.sin((moire_freq * (2.7 / 1.5)) * (xx * math.cos(moire_theta) + yy * math.sin(moire_theta)))
            flicker_t = 1.0 + flicker_amp * torch.sin(5.0 * (yy * scale_ratio) + roll_speed * t_idx)
            flicker_c = 1.0 + flicker_amp * torch.sin(5.0 * yy + roll_speed * t_idx)
            bdist_c = torch.maximum(xx.abs(), yy.abs())
            bezel_c = torch.sigmoid((0.88 - bdist_c) * 14.0)
            halo_c = 0.04 * torch.exp(-((bdist_c - 0.85) ** 2) / 0.01)
            srgb_t = (srgb_t * moire_t * flicker_t).clamp(0.0, 1.0)
            srgb_c = (srgb_c * moire_c * flicker_c * (0.20 + 0.80 * bezel_c) + halo_c).clamp(0.0, 1.0)

        flat_t = srgb_t.view(t, c, h, w)
        flat_c = srgb_c.view(t, c, h, w)
        if do_blur:
            pad = blur_ksize // 2
            if is_motion_blur:
                ker = torch.zeros((blur_ksize, blur_ksize), device=device, dtype=torch.float32)
                coords = torch.linspace(-pad, pad, blur_ksize * 3, device=device)
                xs = (torch.round(coords * math.cos(blur_angle)).long() + pad).clamp(0, blur_ksize - 1)
                ys = (torch.round(coords * math.sin(blur_angle)).long() + pad).clamp(0, blur_ksize - 1)
                ker[ys, xs] += 1.0
                ker = ker / ker.sum().clamp(min=1.0)
            else:
                ax = torch.arange(-pad, pad + 1, device=device, dtype=torch.float32)
                g = torch.exp(-(ax ** 2) / (2.0 * 1.0 * 1.0))
                ker = torch.outer(g, g)
                ker = ker / ker.sum()
            w_k = ker.view(1, 1, blur_ksize, blur_ksize).repeat(c, 1, 1, 1)
            flat_t = F.conv2d(F.pad(flat_t, (pad, pad, pad, pad), mode="reflect"), w_k, groups=c).clamp(0.0, 1.0)
            flat_c = F.conv2d(F.pad(flat_c, (pad, pad, pad, pad), mode="reflect"), w_k, groups=c).clamp(0.0, 1.0)

        if do_uvc:
            both = torch.cat((flat_t, flat_c), dim=0)
            both = self.sensor_tail._apply_uvc_and_dct_jpeg(both)
            flat_t, flat_c = both[:t], both[t:]

        return flat_t.view(1, t, c, h, w), flat_c.view(1, t, c, h, w)

    @torch.no_grad()
    def _freeze_pair_synchronized(
        self,
        tex_i: torch.Tensor,
        ctx_i: torch.Tensor,
        k_idx: int,
        use_bend: bool,
        phase: float,
        tx_amp: float,
        ty_amp: float,
        ang_amp: float,
        sc_amp: float,
        bend_amp: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strategy B with shared frozen frame index k_idx and scale-matched 2D trajectory on (tex, ctx)."""
        _, t, c, h, w = tex_i.shape
        device = tex_i.device
        scale_ratio = 1.5 / 2.7
        t_norm = torch.linspace(-1.0, 1.0, t, device=device).view(t, 1, 1)
        base_yy, base_xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )

        src_t = tex_i[0, k_idx : k_idx + 1].expand(t, -1, -1, -1)
        src_c = ctx_i[0, k_idx : k_idx + 1].expand(t, -1, -1, -1)

        if not use_bend:
            tx = tx_amp * torch.sin(math.pi * t_norm + phase)
            ty = ty_amp * torch.cos(math.pi * t_norm + phase)
            ang = ang_amp * torch.sin(1.3 * math.pi * t_norm)
            sc = 1.0 + sc_amp * t_norm
            ca, sa = torch.cos(ang), torch.sin(ang)
            gx_t = (base_xx.unsqueeze(0) * ca - base_yy.unsqueeze(0) * sa) * sc + tx
            gy_t = (base_xx.unsqueeze(0) * sa + base_yy.unsqueeze(0) * ca) * sc + ty
            gx_c = (base_xx.unsqueeze(0) * ca - base_yy.unsqueeze(0) * sa) * sc + tx * scale_ratio
            gy_c = (base_xx.unsqueeze(0) * sa + base_yy.unsqueeze(0) * ca) * sc + ty * scale_ratio
        else:
            amp = bend_amp * torch.sin(math.pi * t_norm)
            gx_t = base_xx.unsqueeze(0) + amp * (base_xx.unsqueeze(0) ** 2 - 0.3)
            gy_t = base_yy.unsqueeze(0) + 0.5 * amp * base_xx.unsqueeze(0)
            gx_c = base_xx.unsqueeze(0) + (amp * scale_ratio) * (base_xx.unsqueeze(0) ** 2 - 0.3)
            gy_c = base_yy.unsqueeze(0) + 0.5 * (amp * scale_ratio) * base_xx.unsqueeze(0)

        w_t = F.grid_sample(src_t, torch.stack((gx_t, gy_t), dim=-1), mode="bilinear", padding_mode="border", align_corners=False)
        w_c = F.grid_sample(src_c, torch.stack((gx_c, gy_c), dim=-1), mode="bilinear", padding_mode="border", align_corners=False)
        n_t = (w_t + torch.randn_like(w_t) * 0.012).clamp(0.0, 1.0)
        n_c = (w_c + torch.randn_like(w_c) * 0.012).clamp(0.0, 1.0)
        return n_t.unsqueeze(0), n_c.unsqueeze(0)

    @torch.no_grad()
    def augment_pair(
        self,
        tex: torch.Tensor,
        ctx: torch.Tensor,
        labels: torch.Tensor,
        depth_target: Optional[torch.Tensor] = None,
        fine_known: Optional[torch.Tensor] = None,
        p_normal: float = 0.30,
        p_frozen: float = 0.20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Synchronized Strategy A (Normal + Differentiated Low-Light) & Strategy B (Frozen-Rigid) on (tex, ctx)."""
        orig_dtype = tex.dtype
        tex_w = tex.float().clamp(0.0, 1.0)
        ctx_w = ctx.float().clamp(0.0, 1.0)
        b, t = tex_w.shape[:2]
        device = tex_w.device

        labels_out = labels.clone()
        depth_out = depth_target.clone() if depth_target is not None else None
        fk_out = fine_known.clone() if fine_known is not None else torch.ones(b, dtype=torch.bool, device=device)

        tex_out = torch.empty_like(tex_w)
        ctx_out = torch.empty_like(ctx_w)

        def _u(lo: float, hi: float) -> float:
            return float(torch.empty(1, device=device).uniform_(lo, hi).item())

        for i in range(b):
            lbl = int(labels[i].item())
            fk = bool(fk_out[i].item())
            is_normal = torch.rand(1, device=device).item() < p_normal

            if is_normal:
                exp_g = _u(0.85, 1.15)
                inv_g = _u(0.92, 1.12)
                drift = _u(-0.04, 0.04)
                fkind = "uniform"
                awb = _u(-0.05, 0.05)
                shot_s = _u(0.002, 0.008)
                read_s = _u(0.002, 0.007)
                c_scale = 1.0
                haze = 0.0
                is_rep = False
                rep_dim = False
                do_blur = torch.rand(1, device=device).item() < 0.15
                do_uvc = torch.rand(1, device=device).item() < 0.50
            else:
                inv_g = _u(1.45, 2.25)
                drift = _u(-0.08, 0.08)
                awb = _u(-0.15, 0.15)
                shot_s = _u(0.008, 0.032)
                read_s = _u(0.005, 0.022)
                do_blur = torch.rand(1, device=device).item() < 0.30
                do_uvc = torch.rand(1, device=device).item() < 0.80

                eff_lbl = lbl if (lbl == 0 or fk) else int([1, 2, 2][int(torch.randint(0, 3, (1,)).item())])
                r_f = torch.rand(1, device=device).item()
                if eff_lbl == 0:
                    exp_g = _u(0.10, 0.55)
                    fkind = "uniform" if r_f < 0.55 else ("backlight" if r_f < 0.80 else "screen_glow")
                    c_scale, haze, is_rep, rep_dim = 1.0, 0.0, False, False
                    if fkind == "screen_glow":
                        awb = _u(-0.16, -0.04)
                elif eff_lbl == 1:
                    exp_g = _u(0.10, 0.52)
                    fkind = "uniform" if r_f < 0.50 else ("backlight" if r_f < 0.75 else ("shadow" if r_f < 0.90 else "screen_glow"))
                    c_scale = _u(0.60, 0.85)
                    haze = _u(0.03, 0.12)
                    is_rep, rep_dim = False, False
                elif eff_lbl == 2:
                    is_rep = True
                    rep_dim = torch.rand(1, device=device).item() < 0.50
                    exp_g = _u(0.14, 0.48) if rep_dim else _u(0.55, 1.02)
                    inv_g = _u(1.15, 1.85) if not rep_dim else _u(1.40, 2.15)
                    fkind = "uniform" if r_f < 0.60 else ("backlight" if r_f < 0.85 else "shadow")
                    c_scale, haze = 1.0, 0.0
                else:
                    exp_g = _u(0.10, 0.55)
                    fkind = "uniform" if r_f < 0.55 else ("backlight" if r_f < 0.80 else "screen_glow")
                    c_scale, haze, is_rep, rep_dim = _u(0.88, 1.12), 0.0, False, False

            ti, ci = self._apply_joint_isp(
                tex_w[i : i + 1],
                ctx_w[i : i + 1],
                inv_gamma=inv_g,
                exp_gain=exp_g,
                drift_slope=drift,
                field_kind=fkind,
                awb_shift=awb,
                shot_sigma=shot_s,
                read_sigma=read_s,
                contrast_scale=c_scale,
                haze_add=haze,
                is_replay=is_rep,
                replay_dim=rep_dim,
                moire_freq=_u(18.0, 42.0),
                moire_theta=_u(0.0, math.pi),
                moire_amp=_u(0.02, 0.06),
                roll_speed=_u(0.4, 1.2),
                flicker_amp=_u(0.03, 0.08),
                do_blur=do_blur,
                blur_ksize=int([3, 5][int(torch.randint(0, 2, (1,)).item())]),
                blur_angle=_u(0.0, math.pi),
                is_motion_blur=torch.rand(1, device=device).item() < 0.6,
                do_uvc=do_uvc,
            )

            if lbl == 0 and torch.rand(1, device=device).item() < p_frozen:
                k_idx = int(torch.randint(0, t, (1,), device=device).item())
                ti, ci = self._freeze_pair_synchronized(
                    ti,
                    ci,
                    k_idx=k_idx,
                    use_bend=torch.rand(1, device=device).item() < 0.35,
                    phase=_u(0.0, 2.0 * math.pi),
                    tx_amp=_u(0.01, 0.035),
                    ty_amp=_u(0.01, 0.035),
                    ang_amp=_u(0.01, 0.04),
                    sc_amp=_u(0.005, 0.02),
                    bend_amp=_u(0.02, 0.05),
                )
                labels_out[i] = 4
                fk_out[i] = True
                if depth_out is not None:
                    depth_out[i] = 0.0

            tex_out[i : i + 1] = ti
            ctx_out[i : i + 1] = ci

        return tex_out.to(dtype=orig_dtype), ctx_out.to(dtype=orig_dtype), labels_out, depth_out, fk_out

    @torch.no_grad()
    def apply_eval_lighting(
        self,
        tex: torch.Tensor,
        ctx: torch.Tensor,
        mode: str,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic synchronized lighting transforms for Phase 3 ISO 30107-3 evaluation matrix."""
        if mode == "normal":
            return tex, ctx

        orig_dtype = tex.dtype
        tex_w = tex.float().clamp(0.0, 1.0)
        ctx_w = ctx.float().clamp(0.0, 1.0)
        b, t = tex_w.shape[:2]
        tex_out = torch.empty_like(tex_w)
        ctx_out = torch.empty_like(ctx_w)

        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(tex.device) if tex.is_cuda else None
        try:
            torch.manual_seed(seed)
            if tex.is_cuda:
                torch.cuda.manual_seed(seed)

            for i in range(b):
                if mode == "low_light":
                    ti, ci = self._apply_joint_isp(
                        tex_w[i : i + 1], ctx_w[i : i + 1],
                        inv_gamma=2.0, exp_gain=0.18, drift_slope=0.03,
                        field_kind="uniform", awb_shift=-0.06,
                        shot_sigma=0.018, read_sigma=0.012, do_blur=False, do_uvc=True,
                    )
                elif mode == "backlight":
                    ti, ci = self._apply_joint_isp(
                        tex_w[i : i + 1], ctx_w[i : i + 1],
                        inv_gamma=1.9, exp_gain=0.35, drift_slope=0.02,
                        field_kind="backlight", awb_shift=0.02,
                        shot_sigma=0.014, read_sigma=0.009, do_blur=False, do_uvc=True,
                    )
                elif mode == "screen_glow":
                    ti, ci = self._apply_joint_isp(
                        tex_w[i : i + 1], ctx_w[i : i + 1],
                        inv_gamma=1.8, exp_gain=0.28, drift_slope=0.02,
                        field_kind="screen_glow", awb_shift=-0.12,
                        shot_sigma=0.012, read_sigma=0.008, do_blur=False, do_uvc=True,
                    )
                elif mode == "dim_screen":
                    ti, ci = self._apply_joint_isp(
                        tex_w[i : i + 1], ctx_w[i : i + 1],
                        inv_gamma=1.8, exp_gain=0.25, drift_slope=0.02,
                        field_kind="uniform", awb_shift=-0.04,
                        shot_sigma=0.012, read_sigma=0.008,
                        is_replay=True, replay_dim=True, do_blur=False, do_uvc=True,
                    )
                elif mode == "frozen_rigid":
                    ti, ci = self._apply_joint_isp(
                        tex_w[i : i + 1], ctx_w[i : i + 1],
                        inv_gamma=1.9, exp_gain=0.22, drift_slope=0.02,
                        field_kind="uniform", awb_shift=-0.05,
                        shot_sigma=0.014, read_sigma=0.010, do_blur=False, do_uvc=True,
                    )
                    ti, ci = self._freeze_pair_synchronized(
                        ti, ci, k_idx=t // 2, use_bend=False, phase=0.8 + i * 0.3,
                        tx_amp=0.024, ty_amp=0.022, ang_amp=0.025, sc_amp=0.012, bend_amp=0.03,
                    )
                else:
                    raise ValueError(f"Unknown eval lighting mode: {mode}")
                tex_out[i : i + 1] = ti
                ctx_out[i : i + 1] = ci
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, tex.device)

        return tex_out.to(dtype=orig_dtype), ctx_out.to(dtype=orig_dtype)
