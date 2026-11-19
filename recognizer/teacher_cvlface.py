"""Frozen CVLface ViT-Base-KP-RPE teacher wrapper (plus pure-PyTorch test proxy)."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTKPRPETeacherProxy(nn.Module):
    """Lightweight offline ViT+KP-RPE proxy matching CVLface interface for smoke/unit tests."""

    def __init__(self, emb_dim: int = 512, depth: int = 2, dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=14, stride=14)
        self.kpt_embed = nn.Linear(10, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Linear(dim, emb_dim)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_112: torch.Tensor, kpts5_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        tok = self.patch(x_112).flatten(2).transpose(1, 2)  # [B, 64, dim]
        if kpts5_norm is not None:
            bias = self.kpt_embed(kpts5_norm.reshape(x_112.shape[0], 10)).unsqueeze(1)
            tok = tok + bias
        out = self.encoder(tok).mean(dim=1)
        return F.normalize(self.proj(out).float(), dim=1)


class CVLfaceTeacherWrapper(nn.Module):
    """Frozen CVLface `adaface_vit_base_kprpe_webface12m` wrapper (`HF_ENDPOINT=https://hf-mirror.com`)."""

    DEFAULT_REPO = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"

    def __init__(
        self,
        model_id_or_path: Optional[str] = None,
        hf_endpoint: str = "https://hf-mirror.com",
        offline_proxy: bool = False,
    ) -> None:
        super().__init__()
        self.offline_proxy = offline_proxy
        if offline_proxy:
            self.model = ViTKPRPETeacherProxy()
            return

        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
        from transformers import AutoModel

        target = model_id_or_path or self.DEFAULT_REPO
        self.model = AutoModel.from_pretrained(target, trust_remote_code=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_112: torch.Tensor, kpts5_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.offline_proxy:
            return self.model(x_112, kpts5_norm)
        # CVLface KP-RPE expects normalized 112x112 RGB in [-1, 1] and optional 5-pt landmarks in [0, 1]
        try:
            out = self.model(x_112, keypoints=kpts5_norm) if kpts5_norm is not None else self.model(x_112)
        except TypeError:
            out = self.model(x_112)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return F.normalize(out.float(), dim=1)
