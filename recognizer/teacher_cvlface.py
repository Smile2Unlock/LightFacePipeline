"""Frozen CVLface ViT-Base-KP-RPE / IR101 teacher wrapper (plus pure-PyTorch test proxy)."""

from __future__ import annotations

import inspect
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from common.contracts import ARCFACE_DST_112


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


def _patch_cvlface_rpe_auto_install(local_dir: Path) -> None:
    """Disable upstream `setup.py install --user` + `sys.exit()` inside CVLface RPE init so pure PyTorch runs cleanly."""
    rpe_init = local_dir / "models" / "vit_kprpe" / "RPE" / "__init__.py"
    if not rpe_init.is_file():
        return
    txt = rpe_init.read_text(encoding="utf-8")
    if "setup.py" not in txt and "sys.exit" not in txt:
        return
    marker = "def build_rpe("
    idx = txt.find(marker)
    if idx < 0:
        return
    safe_header = (
        "from .KPRPE import kprpe_shared\n"
        "try:\n"
        "    from .rpe_ops.rpe_index import RPEIndexFunction\n"
        "except Exception:\n"
        "    RPEIndexFunction = None\n\n\n"
    )
    rpe_init.write_text(safe_header + txt[idx:], encoding="utf-8")


@contextmanager
def _cvlface_import_context(local_dir: Path) -> Iterator[None]:
    prev_cwd = os.getcwd()
    str_dir = str(local_dir.resolve())
    saved_modules = {
        name: sys.modules.pop(name)
        for name in list(sys.modules.keys())
        if name == "models" or name.startswith("models.")
    }
    sys.path.insert(0, str_dir)
    os.chdir(str_dir)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        if sys.path and sys.path[0] == str_dir:
            sys.path.pop(0)
        for name in list(sys.modules.keys()):
            if name == "models" or name.startswith("models."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _load_local_cvlface_model(local_dir: Path) -> nn.Module:
    from omegaconf import OmegaConf

    resolved_dir = local_dir.resolve()
    _patch_cvlface_rpe_auto_install(resolved_dir)
    with _cvlface_import_context(resolved_dir):
        from models import get_model

        with open("pretrained_model/model.yaml", "r", encoding="utf-8") as f:
            raw_conf = yaml.safe_load(f)
        model_conf = OmegaConf.create(raw_conf)
        model = get_model(model_conf)
        pt_path = resolved_dir / "pretrained_model" / "model.pt"
        if pt_path.is_file():
            model.load_state_dict_from_path(str(pt_path))
        else:
            from safetensors.torch import load_file

            sd = load_file(str(resolved_dir / "model.safetensors"))
            cleaned = {re.sub(r"^model\.", "", k): v for k, v in sd.items()}
            model.load_state_dict(cleaned, strict=False)
    return model


class CVLfaceTeacherWrapper(nn.Module):
    """Frozen CVLface ViT-Base-KP-RPE / IR101 teacher wrapper with pure-PyTorch fallback."""

    DEFAULT_REPO = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"

    def __init__(
        self,
        model_id_or_path: Optional[str] = None,
        hf_endpoint: str = "https://hf-mirror.com",
        offline_proxy: bool = False,
        cache_dir: Optional[str] = None,
        use_amp: bool = True,
    ) -> None:
        super().__init__()
        self.offline_proxy = offline_proxy
        self.use_amp = use_amp
        self.register_buffer(
            "canonical_kpts_norm",
            torch.from_numpy((ARCFACE_DST_112 / 112.0).astype("float32")).unsqueeze(0),
            persistent=False,
        )
        if offline_proxy:
            self.model = ViTKPRPETeacherProxy()
            self._accepts_keypoints = True
            return

        target = model_id_or_path or self.DEFAULT_REPO
        target_path = Path(target)
        if target_path.is_dir():
            local_dir = target_path
        else:
            if hf_endpoint and "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = hf_endpoint
            from huggingface_hub import snapshot_download

            slug = target.replace("/", "__")
            local_dir = Path(cache_dir) if cache_dir else Path("datasets/pretrained") / slug
            snapshot_download(target, local_dir=str(local_dir))

        self.model = _load_local_cvlface_model(local_dir)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._accepts_keypoints = hasattr(self.model, "net") and (
            "keypoints" in inspect.signature(self.model.net.forward).parameters
        )

    @torch.no_grad()
    def forward(self, x_112: torch.Tensor, kpts5_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.offline_proxy:
            return self.model(x_112, kpts5_norm)

        kpts = kpts5_norm
        if self._accepts_keypoints:
            if kpts is None:
                kpts = self.canonical_kpts_norm.to(device=x_112.device, dtype=torch.float32).expand(
                    x_112.shape[0], -1, -1
                )
            elif kpts.ndim == 2 and kpts.shape[1] == 10:
                kpts = kpts.reshape(-1, 5, 2)

        amp_enabled = bool(x_112.is_cuda and self.use_amp)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            if self._accepts_keypoints:
                out = self.model(x_112, kpts)
            else:
                out = self.model(x_112)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return F.normalize(out.float(), dim=1)
