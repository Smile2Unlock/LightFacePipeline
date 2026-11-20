"""Runtime helpers: ROCm/DTK precision & SDPA contexts, ModelEMA, checkpoints, provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def configure_cpu_threads(max_threads: int = 4) -> int:
    """Clamp OpenMP/PyTorch CPU threads to avoid cgroup oversubscription livelocks."""
    n = int(os.environ.get("OMP_NUM_THREADS", str(max_threads)))
    n = max(1, min(n, max_threads)) if "OMP_NUM_THREADS" not in os.environ else max(1, n)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(min(2, n))
    except RuntimeError:
        pass
    return n


def seed_everything(seed: int = 42) -> None:
    configure_cpu_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    configure_cpu_threads(4)
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_amp_autocast(device: torch.device, precision: str = "fp16"):
    """Return a fresh autocast context manager (single-use per forward pass)."""
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported precision: {precision}")


def get_grad_scaler(device: torch.device, precision: str = "fp16") -> torch.amp.GradScaler:
    """Return GradScaler enabled only for CUDA/ROCm FP16."""
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))


def get_amp_context(device: torch.device, precision: str = "fp16"):
    """Return (autocast_ctx, grad_scaler) configured from ROCm stress test findings."""
    return get_amp_autocast(device, precision), get_grad_scaler(device, precision)


_FLASH_ATTN_AVAILABLE: Optional[bool] = None


def _has_working_flash_sdpa() -> bool:
    """Probe once whether ROCm/CUDA FlashAttention SDPA kernels load cleanly."""
    global _FLASH_ATTN_AVAILABLE
    if _FLASH_ATTN_AVAILABLE is not None:
        return _FLASH_ATTN_AVAILABLE
    if not torch.cuda.is_available():
        _FLASH_ATTN_AVAILABLE = False
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.empty((1, 4, 8, 48), device="cuda", dtype=torch.float16)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        _FLASH_ATTN_AVAILABLE = True
    except Exception:
        _FLASH_ATTN_AVAILABLE = False
    return _FLASH_ATTN_AVAILABLE


def get_sdpa_context(backend: str = "flash_attention"):
    """Select verified SDPA backend (`flash_attention` or `math` on Hygon ROCm/DTK)."""
    if not torch.cuda.is_available():
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not _has_working_flash_sdpa():
        return sdpa_kernel([SDPBackend.MATH])

    mapping = {
        "flash_attention": [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
        "math": [SDPBackend.MATH],
        "auto": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    }
    backends = mapping.get(backend, [SDPBackend.MATH])
    return sdpa_kernel(backends)


class ModelEMA:
    """Exponential Moving Average copy of model weights + buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.9995, warmup_tau: float = 500.0) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.warmup_tau = warmup_tau
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.warmup_tau)) if self.warmup_tau > 0 else self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.detach(), alpha=1.0 - d)
            else:
                v.copy_(src)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[ModelEMA] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    backup_dir: Optional[Path] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = ema.ema.state_dict()
    torch.save(payload, path)

    if backup_dir is not None:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / path.name).write_bytes(path.read_bytes())
    return path


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[ModelEMA] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.ema.load_state_dict(ckpt["ema"])
    return ckpt


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def collect_provenance(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_rev = "uninitialized"
    info: Dict[str, Any] = {
        "git_commit": git_rev,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }
    if extra:
        info.update(extra)
    return info
