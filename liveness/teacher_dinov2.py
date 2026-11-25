"""Dual-stream DINOv2-with-Registers (LoRA) + ConvNeXtV2-Laplacian High-Freq Teacher."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from liveness.st_fas_lite import ConvNeXtV2LiteBranch, TemporalAttentionBlock


def compute_laplacian_residual(rgb: torch.Tensor) -> torch.Tensor:
    """Fixed 3x3 Laplacian high-pass filter across RGB channels."""
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=rgb.device, dtype=rgb.dtype)
    weight = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return F.conv2d(F.pad(rgb, (1, 1, 1, 1), mode="reflect"), weight, groups=3)


class DualStreamDINOv2Teacher(nn.Module):
    """ICCVW 2025 / AAAI 2026 inspired Dual-Stream Teacher:

    1. Global semantic stream: `Dinov2WithRegistersModel` (`vit_base_patch14_reg4`) + LoRA (r=16)
    2. High-frequency micro-texture stream: `ConvNeXtV2-Tiny`-style 6-ch (RGB + Laplacian)
    3. Gated fusion + 2-layer Temporal Transformer + Patch Affinity output for Patch-Affinity-KD
    """

    DEFAULT_REPO = "facebook/dinov2-with-registers-base"

    def __init__(
        self,
        model_id_or_path: Optional[str] = None,
        hf_endpoint: str = "https://hf-mirror.com",
        lora_r: int = 16,
        num_classes: int = 5,
        offline_config_only: bool = False,
        clip_len: int = 8,
    ) -> None:
        super().__init__()
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        if offline_config_only:
            # Lightweight offline instantiation matching exact Dinov2WithRegistersModel + PEFT LoRA mechanics
            cfg = Dinov2WithRegistersConfig(
                hidden_size=192,
                num_hidden_layers=2,
                num_attention_heads=4,
                mlp_ratio=2,
                image_size=112,
                patch_size=14,
                num_register_tokens=4,
            )
            base_vit = Dinov2WithRegistersModel(cfg)
            self.vit_input_size = 112
        else:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
            target = model_id_or_path or self.DEFAULT_REPO
            base_vit = Dinov2WithRegistersModel.from_pretrained(target)
            self.vit_input_size = int(getattr(base_vit.config, "image_size", 224))

        linear_leaf_names = {
            name.split(".")[-1]
            for name, mod in base_vit.named_modules()
            if isinstance(mod, nn.Linear)
        }
        target_modules = ["q_proj", "v_proj"] if {"q_proj", "v_proj"}.issubset(linear_leaf_names) else ["query", "value"]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        self.vit = get_peft_model(base_vit, lora_cfg)
        self.num_reg = int(getattr(self.vit.config, "num_register_tokens", 4))
        vit_dim = int(self.vit.config.hidden_size)

        # High-frequency ConvNeXtV2-Tiny stream (RGB + Laplacian = 6ch)
        self.hf_branch = ConvNeXtV2LiteBranch(in_ch=6, dims=(64, 128, 256), depths=(2, 3, 2))

        fuse_dim = 256
        self.vit_proj = nn.Linear(vit_dim, fuse_dim)
        self.hf_proj = nn.Linear(256, fuse_dim)
        self.gate = nn.Sequential(nn.Linear(fuse_dim * 2, fuse_dim), nn.Sigmoid())

        self.pos_embed = nn.Parameter(torch.zeros(1, clip_len, fuse_dim))
        self.temp_blocks = nn.ModuleList([TemporalAttentionBlock(dim=fuse_dim, num_heads=4) for _ in range(2)])
        self.cls_head = nn.Linear(fuse_dim, num_classes)

    def forward(self, clip_tex: torch.Tensor, clip_ctx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward `[B, T, 3, 128, 128]` RGB `[0, 1]` -> logits, pooled feat, and patch affinity matrix."""
        b, t, c, h, w = clip_tex.shape
        tex_flat = clip_tex.reshape(b * t, c, h, w)
        ctx_flat = clip_ctx.reshape(b * t, c, h, w) if clip_ctx is not None else tex_flat

        norm_tex = (tex_flat - self.im_mean.to(tex_flat.device, tex_flat.dtype)) / self.im_std.to(
            tex_flat.device, tex_flat.dtype
        )
        norm_ctx = (ctx_flat - self.im_mean.to(ctx_flat.device, ctx_flat.dtype)) / self.im_std.to(
            ctx_flat.device, ctx_flat.dtype
        )

        # Global DINOv2-Reg stream on context/texture view resized to ViT patch grid
        vit_in = F.interpolate(norm_ctx, size=(self.vit_input_size, self.vit_input_size), mode="bilinear", align_corners=False)
        vit_out = self.vit(pixel_values=vit_in)
        seq = vit_out.last_hidden_state  # [B*T, 1 + num_reg + P, vit_dim]
        cls_tok = seq[:, 0]
        patch_toks = seq[:, 1 + self.num_reg :]  # strip CLS & 4 register tokens -> pure spatial patches!

        # High-frequency micro-texture stream on 1.5x texture crop
        lap = compute_laplacian_residual(tex_flat)
        hf_vec, _ = self.hf_branch(torch.cat((norm_tex, lap), dim=1))

        # Gated bimodal fusion
        gv = self.vit_proj(cls_tok)
        gh = self.hf_proj(hf_vec)
        g = self.gate(torch.cat((gv, gh), dim=-1))
        fused = (g * gv + (1.0 - g) * gh).view(b, t, -1)

        x = fused + self.pos_embed[:, :t, :]
        for blk in self.temp_blocks:
            x = blk(x)
        pooled = x.mean(dim=1)
        logits = self.cls_head(pooled).float()

        # Patch self-affinity matrix [B, T, P, P] for Patch-Affinity-KD
        p_norm = F.normalize(patch_toks.float(), dim=-1)
        patch_affinity = torch.bmm(p_norm, p_norm.transpose(1, 2)).view(b, t, p_norm.shape[1], p_norm.shape[1])

        return {
            "logits": logits,
            "pooled_feat": pooled,
            "patch_affinity": patch_affinity,
        }
