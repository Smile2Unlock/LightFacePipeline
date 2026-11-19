#!/usr/bin/env python3
"""Phase 0 ROCm / DTK stress test & hardware profiler using only stdlib + PyTorch."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _mem_mib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    return {
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
    }


def _check_pkg(name: str) -> Optional[str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "installed")
    except Exception as exc:
        return f"import_error: {exc}"


def audit_hardware_and_storage() -> Dict[str, Any]:
    meminfo: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    val = parts[1].strip().split()[0]
                    if val.isdigit():
                        meminfo[parts[0]] = int(val)
    except OSError:
        pass

    gpu_props: Dict[str, Any] = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu_props = {
            "name": p.name,
            "gcnArchName": getattr(p, "gcnArchName", None),
            "total_memory_mib": round(p.total_memory / (1024 ** 2), 2),
            "multi_processor_count": p.multi_processor_count,
            "major": p.major,
            "minor": p.minor,
        }

    storage: Dict[str, Any] = {}
    for path_str in [
        "/workspace",
        "/public/home/ation_ciger",
        "/public/share/ation_ciger",
        "/work1/share/ation_ciger",
        os.getcwd(),
    ]:
        p = Path(path_str)
        entry: Dict[str, Any] = {"exists": p.exists()}
        if p.exists():
            try:
                usage = shutil.disk_usage(p)
                entry["total_gib"] = round(usage.total / (1024 ** 3), 2)
                entry["used_gib"] = round(usage.used / (1024 ** 3), 2)
                entry["free_gib"] = round(usage.free / (1024 ** 3), 2)
            except OSError as exc:
                entry["usage_error"] = str(exc)
            test_file = p / f".rocm_write_probe_{os.getpid()}"
            try:
                test_file.write_text("ok", encoding="utf-8")
                test_file.unlink()
                entry["writable"] = True
            except OSError as exc:
                entry["writable"] = False
                entry["write_error"] = str(exc)
        storage[path_str] = entry

    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        "cpu_count": os.cpu_count(),
        "ram_total_gib": round(meminfo.get("MemTotal", 0) / (1024 ** 2), 2),
        "ram_available_gib": round(meminfo.get("MemAvailable", 0) / (1024 ** 2), 2),
        "gpu": gpu_props,
        "allocator_env": {
            "PYTORCH_HIP_ALLOC_CONF": os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
            "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        },
        "preinstalled_packages": {
            pkg: _check_pkg(pkg)
            for pkg in ["flash_attn", "triton", "transformers", "accelerate", "safetensors", "huggingface_hub", "einops"]
        },
        "storage": storage,
    }


class DWSepConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class ConvProxyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWSepConvBlock(32, 64, stride=2),
            DWSepConvBlock(64, 64, stride=1),
            DWSepConvBlock(64, 128, stride=2),
            DWSepConvBlock(128, 128, stride=1),
            DWSepConvBlock(128, 256, stride=2),
            DWSepConvBlock(256, 256, stride=1),
            DWSepConvBlock(256, 512, stride=2),
        )
        self.head = nn.Linear(512, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        x = x.mean(dim=(-2, -1))
        return self.head(x)


class ViTProxyBlock(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0) -> None:
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
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTProxyNet(nn.Module):
    def __init__(self, depth: int = 4, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ViTProxyBlock(dim=dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


def _run_precision_case(
    mode: str,
    model_fn,
    input_shape: Tuple[int, ...],
    warmup: int = 10,
    steps: int = 30,
) -> Dict[str, Any]:
    _reset_mem()
    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[mode]
    use_amp = mode in ("bf16", "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=(mode == "fp16"))

    try:
        model = model_fn().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(*input_shape, device=device, dtype=torch.float32)
        finite = True
        loss_val = 0.0

        for step in range(warmup + steps):
            if step == warmup:
                _sync()
                t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                out = model(x)
                loss = (out.float() ** 2).mean()
            if not torch.isfinite(loss).item():
                finite = False
                break
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_val = float(loss.detach().item())

        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(1, steps)
        res = {
            "ok": finite,
            "ms_per_step": round(elapsed_ms, 3),
            "last_loss": round(loss_val, 6),
            **_mem_mib(),
        }
        del model, opt, x, scaler
        _reset_mem()
        return res
    except Exception as exc:
        _reset_mem()
        return {"ok": False, "error": str(exc)}


def benchmark_precisions() -> Dict[str, Any]:
    conv_res = {
        mode: _run_precision_case(mode, ConvProxyNet, (64, 3, 320, 320))
        for mode in ("bf16", "fp16", "fp32")
    }
    vit_res = {
        mode: _run_precision_case(mode, lambda: ViTProxyNet(depth=4, dim=768, num_heads=12), (64, 257, 768))
        for mode in ("bf16", "fp16", "fp32")
    }

    # AdaFace Cosine FP32-cast numerical stability probe under autocast
    adaface_check: Dict[str, Any] = {}
    device = torch.device("cuda")
    for mode, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        try:
            feats = torch.randn(512, 512, device=device, dtype=torch.float32, requires_grad=True)
            centers = torch.randn(4096, 512, device=device, dtype=torch.float32)
            labels = torch.randint(0, 4096, (512,), device=device)
            with torch.amp.autocast("cuda", dtype=dtype):
                f_raw = feats * 2.0
                # Explicit FP32 normalization + cosine similarity as planned
                norms = f_raw.float().norm(dim=1, keepdim=True)
                z = f_raw.float() / norms.clamp(min=1e-6)
                w = F.normalize(centers.float(), dim=1)
                cos = torch.matmul(z, w.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                loss = F.cross_entropy(cos * 64.0, labels)
            loss.backward()
            adaface_check[mode] = {
                "ok": bool(torch.isfinite(loss).item() and torch.isfinite(feats.grad).all().item()),
                "loss": round(float(loss.item()), 5),
            }
        except Exception as exc:
            adaface_check[mode] = {"ok": False, "error": str(exc)}
    _reset_mem()

    bf16_ok = conv_res["bf16"].get("ok") and vit_res["bf16"].get("ok") and adaface_check["bf16"].get("ok")
    fp16_ok = conv_res["fp16"].get("ok") and vit_res["fp16"].get("ok") and adaface_check["fp16"].get("ok")

    if bf16_ok and fp16_ok:
        bf16_ms = conv_res["bf16"]["ms_per_step"] + vit_res["bf16"]["ms_per_step"]
        fp16_ms = conv_res["fp16"]["ms_per_step"] + vit_res["fp16"]["ms_per_step"]
        rec = "bf16" if bf16_ms <= 1.25 * fp16_ms else "fp16"
    elif bf16_ok:
        rec = "bf16"
    elif fp16_ok:
        rec = "fp16"
    else:
        rec = "fp32"

    return {
        "conv_dwsep_64x3x320x320": conv_res,
        "vit_b_4blk_64x257x768": vit_res,
        "adaface_cosine_fp32_cast": adaface_check,
        "recommended_precision": rec,
    }


def probe_sdpa_backends() -> Dict[str, Any]:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = [
        ("flash_attention", SDPBackend.FLASH_ATTENTION),
        ("efficient_attention", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]
    shapes = {
        "vit_257": (64, 12, 257, 64),
        "temporal_8": (64, 4, 8, 48),
    }
    results: Dict[str, Any] = {}
    device = torch.device("cuda")

    for dtype_name, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        results[dtype_name] = {}
        for shape_name, (b, h, n, d) in shapes.items():
            shape_res: Dict[str, Any] = {}
            q = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
            for b_name, backend in backends:
                try:
                    with sdpa_kernel(backend):
                        for _ in range(5):
                            out = F.scaled_dot_product_attention(q, k, v)
                            out.sum().backward()
                            q.grad = k.grad = v.grad = None
                        _sync()
                        t0 = time.perf_counter()
                        for _ in range(25):
                            out = F.scaled_dot_product_attention(q, k, v)
                            out.sum().backward()
                            q.grad = k.grad = v.grad = None
                        _sync()
                        ms = (time.perf_counter() - t0) * 1000.0 / 25
                    shape_res[b_name] = {"ok": True, "ms_per_step": round(ms, 3)}
                except Exception as exc:
                    shape_res[b_name] = {"ok": False, "error": str(exc).splitlines()[0]}
            results[dtype_name][shape_name] = shape_res
            del q, k, v
    _reset_mem()

    # Determine fastest working backend for vit_257 under bf16/fp16
    best: Dict[str, str] = {}
    for dtype_name in ("bf16", "fp16"):
        candidates = [
            (v["ms_per_step"], k)
            for k, v in results[dtype_name]["vit_257"].items()
            if v.get("ok")
        ]
        if candidates:
            candidates.sort()
            best[dtype_name] = candidates[0][1]
    results["recommended_backend"] = best
    return results


def _phase1_detector_stress(dtype: torch.dtype) -> Dict[str, Any]:
    device = torch.device("cuda")
    use_amp = dtype in (torch.bfloat16, torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    for bs in (128, 64):
        _reset_mem()
        try:
            student = ConvProxyNet().to(device)
            ema = ConvProxyNet().to(device)
            teacher = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.SiLU(inplace=True),
                DWSepConvBlock(64, 128, 2),
                DWSepConvBlock(128, 256, 2),
                DWSepConvBlock(256, 512, 2),
                DWSepConvBlock(512, 512, 2),
            ).to(device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            for p in ema.parameters():
                p.requires_grad_(False)
            opt = torch.optim.AdamW(student.parameters(), lr=2e-4)

            scales = [320, 480, 640] * 3
            t0 = time.perf_counter()
            for s in scales:
                x_s = torch.randn(bs, 3, s, s, device=device)
                x_t = torch.randn(bs, 3, s, s, device=device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                    with torch.no_grad():
                        t_feat = teacher(x_t).mean(dim=(-2, -1))
                        _ = ema(x_s)
                    s_out = student(x_s)
                    loss = (s_out[:, :128] ** 2).mean() + (t_feat[:, :128].detach() - s_out).pow(2).mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            _sync()
            elapsed = time.perf_counter() - t0
            res = {
                "ok": True,
                "batch_size": bs,
                "steps": len(scales),
                "elapsed_s": round(elapsed, 3),
                "imgs_per_sec": round(bs * len(scales) / elapsed, 1),
                **_mem_mib(),
            }
            del student, ema, teacher, opt
            _reset_mem()
            return res
        except torch.cuda.OutOfMemoryError:
            _reset_mem()
            if bs == 64:
                return {"ok": False, "error": "OOM even at batch_size=64"}
        except Exception as exc:
            _reset_mem()
            return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": "unexpected"}


def _phase2_recognizer_stress(dtype: torch.dtype) -> Dict[str, Any]:
    device = torch.device("cuda")
    use_amp = dtype in (torch.bfloat16, torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    for pairs in (512, 256):
        _reset_mem()
        try:
            # Teacher ViT-B-like (12 blocks) + Student CNN/SDTA proxy + PartialFC (30,000 centers)
            teacher_proj = nn.Conv2d(3, 768, kernel_size=14, stride=8, padding=3).to(device).eval()
            teacher_vit = ViTProxyNet(depth=12, dim=768, num_heads=12).to(device).eval()
            for m in (teacher_proj, teacher_vit):
                for p in m.parameters():
                    p.requires_grad_(False)

            student = ConvProxyNet().to(device)
            student_head = nn.Linear(128, 512).to(device)
            partial_fc_centers = nn.Parameter(torch.randn(30000, 512, device=device))
            opt = torch.optim.AdamW(list(student.parameters()) + list(student_head.parameters()) + [partial_fc_centers], lr=2e-4)

            steps = 6
            t0 = time.perf_counter()
            for _ in range(steps):
                x_clean = torch.randn(pairs, 3, 112, 112, device=device)
                x_dark = torch.randn(pairs, 3, 112, 112, device=device)
                labels = torch.randint(0, 30000, (pairs,), device=device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                    with torch.no_grad():
                        tok = teacher_proj(x_clean).flatten(2).transpose(1, 2)
                        t_feat = teacher_vit(tok)[:, :512]
                    s_clean = student_head(student(x_clean))
                    s_dark = student_head(student(x_dark))
                # FP32 AdaFace cosine + AdaDistill cosine
                zc = F.normalize(s_clean.float(), dim=1)
                zd = F.normalize(s_dark.float(), dim=1)
                zt = F.normalize(t_feat.float(), dim=1)
                w = F.normalize(partial_fc_centers.float(), dim=1)
                logits = torch.matmul(zd, w.t()) * 32.0
                loss = F.cross_entropy(logits, labels) + (1.0 - (zd * zt).sum(dim=1)).mean() + (1.0 - (zd * zc.detach()).sum(dim=1)).mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            _sync()
            elapsed = time.perf_counter() - t0
            res = {
                "ok": True,
                "pairs_per_step": pairs,
                "images_per_step": pairs * 2,
                "partial_fc_classes": 30000,
                "steps": steps,
                "elapsed_s": round(elapsed, 3),
                "pairs_per_sec": round(pairs * steps / elapsed, 1),
                **_mem_mib(),
            }
            del teacher_proj, teacher_vit, student, student_head, partial_fc_centers, opt
            _reset_mem()
            return res
        except torch.cuda.OutOfMemoryError:
            _reset_mem()
            if pairs == 256:
                return {"ok": False, "error": "OOM even at pairs=256"}
        except Exception as exc:
            _reset_mem()
            return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": "unexpected"}


def _phase3_liveness_stress(dtype: torch.dtype) -> Dict[str, Any]:
    device = torch.device("cuda")
    use_amp = dtype in (torch.bfloat16, torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    for clips in (64, 32):
        _reset_mem()
        try:
            # Frozen DINOv2-Reg proxy (12 blocks) + ConvNeXtV2-T proxy + Student dual-scale + temporal attn
            teacher_patch = nn.Conv2d(3, 768, kernel_size=14, stride=14).to(device).eval()
            teacher_vit = ViTProxyNet(depth=12, dim=768, num_heads=12).to(device).eval()
            for m in (teacher_patch, teacher_vit):
                for p in m.parameters():
                    p.requires_grad_(False)

            stu_tex = ConvProxyNet().to(device)
            stu_ctx = ConvProxyNet().to(device)
            stu_proj = nn.Linear(256, 192).to(device)
            stu_temp = ViTProxyNet(depth=2, dim=192, num_heads=4).to(device)
            stu_cls = nn.Linear(192, 5).to(device)
            opt = torch.optim.AdamW(
                list(stu_tex.parameters())
                + list(stu_ctx.parameters())
                + list(stu_proj.parameters())
                + list(stu_temp.parameters())
                + list(stu_cls.parameters()),
                lr=3e-4,
            )

            t_len = 8
            steps = 6
            t0 = time.perf_counter()
            for _ in range(steps):
                tex = torch.randn(clips, t_len, 3, 128, 128, device=device)
                ctx = torch.randn(clips, t_len, 3, 128, 128, device=device)
                labels = torch.randint(0, 5, (clips,), device=device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                    tex_flat = tex.view(clips * t_len, 3, 128, 128)
                    ctx_flat = ctx.view(clips * t_len, 3, 128, 128)
                    with torch.no_grad():
                        t_in = F.interpolate(tex_flat, size=(224, 224), mode="bilinear", align_corners=False)
                        t_tok = teacher_patch(t_in).flatten(2).transpose(1, 2)
                        t_feat = teacher_vit(t_tok).view(clips, t_len, 768).mean(dim=1)
                    f_tex = stu_tex(tex_flat)
                    f_ctx = stu_ctx(ctx_flat)
                    seq = stu_proj(torch.cat([f_tex, f_ctx], dim=-1)).view(clips, t_len, 192)
                    pooled = stu_temp(seq)
                    logits = stu_cls(pooled)
                    loss = F.cross_entropy(logits, labels) + (pooled - t_feat[:, :192]).pow(2).mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            _sync()
            elapsed = time.perf_counter() - t0
            res = {
                "ok": True,
                "clips_per_step": clips,
                "frames_per_step": clips * t_len,
                "steps": steps,
                "elapsed_s": round(elapsed, 3),
                "clips_per_sec": round(clips * steps / elapsed, 1),
                **_mem_mib(),
            }
            del teacher_patch, teacher_vit, stu_tex, stu_ctx, stu_proj, stu_temp, stu_cls, opt
            _reset_mem()
            return res
        except torch.cuda.OutOfMemoryError:
            _reset_mem()
            if clips == 32:
                return {"ok": False, "error": "OOM even at clips=32"}
        except Exception as exc:
            _reset_mem()
            return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": "unexpected"}


class SyntheticUint8Dataset(Dataset):
    def __init__(self, n: int = 256) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Pure CPU uint8 tensor simulating decoded RGB frame (zero GPU/HIP calls in workers)
        gen = torch.Generator().manual_seed(idx + 17)
        return torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8, generator=gen)


def stress_dataloader_and_gpu_degrade() -> Dict[str, Any]:
    _reset_mem()
    device = torch.device("cuda")
    try:
        ds = SyntheticUint8Dataset(256)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
        total_frames = 0
        t0 = time.perf_counter()
        for batch in dl:
            x = batch.to(device, non_blocking=True).float().div_(255.0)
            # Vectorized GPU inverse ISP + exposure + shot/read noise
            lin = x.clamp(1e-4, 1.0).pow(2.2) * 0.18
            noise = torch.randn_like(lin) * torch.sqrt(lin * 0.02 ** 2 + 0.01 ** 2)
            out = (lin + noise).clamp(0.0, 1.0).pow(1.0 / 2.2)
            total_frames += out.shape[0]
        _sync()
        elapsed = time.perf_counter() - t0
        _reset_mem()
        return {
            "ok": True,
            "num_workers": 8,
            "total_frames": total_frames,
            "elapsed_s": round(elapsed, 3),
            "frames_per_sec": round(total_frames / elapsed, 1),
        }
    except Exception as exc:
        _reset_mem()
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 ROCm/DTK stress test")
    parser.add_argument("--output", default="reports/rocm_stress_report.json", help="Output JSON report path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {}
    report["hardware_and_storage"] = audit_hardware_and_storage()

    if not torch.cuda.is_available():
        report["error"] = "torch.cuda.is_available() is False"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(1)

    prec = benchmark_precisions()
    report["precision_benchmark"] = prec
    report["sdpa_probe"] = probe_sdpa_backends()

    rec_name = prec["recommended_precision"]
    rec_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[rec_name]

    report["vram_stress_64gb"] = {
        "dtype_used": rec_name,
        "phase1_detector": _phase1_detector_stress(rec_dtype),
        "phase2_recognizer": _phase2_recognizer_stress(rec_dtype),
        "phase3_liveness": _phase3_liveness_stress(rec_dtype),
    }
    report["dataloader_gpu_degrade_stress"] = stress_dataloader_and_gpu_degrade()

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
