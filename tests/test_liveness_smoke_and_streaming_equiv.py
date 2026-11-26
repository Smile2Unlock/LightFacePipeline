"""Tests for Phase 3 ST-FAS-Lite, Streaming Equivalence, DINOv2-Reg LoRA Teacher, and Redline Checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from common.degradations import LivenessPhysicalAugmentor
from common.metrics import audit_luma_anti_shortcut, compute_iso30107_metrics
from common.runtime import ModelEMA, get_device
from liveness.dataset import SyntheticLivenessSmokeDataset, VideoClipLivenessDataset
from liveness.losses import FASCombinedLoss
from liveness.st_fas_lite import STFASLiteClip
from liveness.teacher_dinov2 import DualStreamDINOv2Teacher
from liveness.train import train_one_epoch_liveness, train_one_epoch_teacher


def test_st_fas_lite_clip_vs_streaming_pytorch_equiv() -> None:
    device = get_device()
    model = STFASLiteClip().eval().to(device)
    clip_tex = torch.rand(2, 8, 3, 128, 128, device=device)
    clip_ctx = torch.rand(2, 8, 3, 128, 128, device=device)

    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32 if device.type == "cuda" else None
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            c_live, c_logits, c_clarity = model(clip_tex, clip_ctx)

            feats = []
            quals = []
            for t in range(8):
                prev = clip_tex[:, t - 1] if t > 0 else clip_tex[:, 0]
                f_t, q_t = model.spatial_enc(clip_tex[:, t], prev, clip_ctx[:, t])
                feats.append(f_t)
                quals.append(q_t)
            s_live, s_logits, s_clarity = model.temporal_head(torch.stack(feats, dim=1), torch.stack(quals, dim=1))
    finally:
        if device.type == "cuda":
            assert prev_matmul_tf32 is not None and prev_cudnn_tf32 is not None
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    assert torch.max(torch.abs(c_live - s_live)) < 1e-5
    assert torch.max(torch.abs(c_logits - s_logits)) < 1e-5
    assert torch.max(torch.abs(c_clarity - s_clarity)) < 1e-5


def test_static_frame_repetition_rejected_for_live_clips(tmp_path: Path) -> None:
    manifest = tmp_path / "bad_clips.jsonl"
    row = {
        "clip_id": "bad_live_001",
        "frame_paths": ["/tmp/same_frame.png"] * 8,
        "bboxes": [[10, 10, 100, 100]] * 8,
        "label": 0,
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="REDLINE VIOLATION"):
        VideoClipLivenessDataset(manifest)


def test_dinov2_reg_teacher_and_liveness_gpu_smoke() -> None:
    device = get_device()
    student = STFASLiteClip().to(device)
    teacher = DualStreamDINOv2Teacher(offline_config_only=True, lora_r=8).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    ema = ModelEMA(student, decay=0.999)
    criterion = FASCombinedLoss().to(device)
    augmentor = LivenessPhysicalAugmentor().to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)

    ds = SyntheticLivenessSmokeDataset(n=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    stats = train_one_epoch_liveness(
        student, teacher, ema, loader, criterion, augmentor, opt, device, precision="fp16", max_steps=2
    )
    assert torch.isfinite(torch.tensor(stats["loss"]))

    iso = compute_iso30107_metrics(
        np.array([0.92, 0.85, 0.78, 0.15, 0.22, 0.08]),
        np.array([1, 1, 1, 0, 0, 0]),
    )
    assert iso["auc"] == 1.0
    assert iso["operating_point"]["ACER"] == 0.0

    audit = audit_luma_anti_shortcut(
        np.full((4, 8, 3, 32, 32), 0.25, dtype=np.float32),
        np.full((4, 8, 3, 32, 32), 0.27, dtype=np.float32),
    )
    assert not audit["warning_shortcut_risk"]


def test_synchronized_pair_augmentor_and_teacher_smoke() -> None:
    device = get_device()
    aug = LivenessPhysicalAugmentor().to(device)
    tex = torch.rand(6, 8, 3, 128, 128, device=device)
    ctx = torch.rand(6, 8, 3, 128, 128, device=device)
    labels = torch.tensor([0, 0, 1, 2, 3, 2], device=device)
    fk = torch.tensor([True, True, True, True, True, False], device=device)
    depth = torch.zeros(6, 8, 1, 8, 8, device=device)

    t_out, c_out, l_out, d_out, fk_out = aug.augment_pair(
        tex, ctx, labels, depth_target=depth, fine_known=fk, p_normal=0.33, p_frozen=0.50
    )
    assert t_out.shape == tex.shape and c_out.shape == ctx.shape
    assert d_out is not None and torch.isfinite(t_out).all() and torch.isfinite(c_out).all()

    criterion = FASCombinedLoss().to(device)
    student = STFASLiteClip().to(device)
    s_out = student.forward_train(t_out, c_out)
    ld = criterion(s_out, l_out, depth_target=d_out, clarity_target=torch.full((6,), 0.6, device=device), fine_known=fk_out)
    assert torch.isfinite(ld["loss"])

    teacher = DualStreamDINOv2Teacher(offline_config_only=True, lora_r=4, enable_grad_checkpointing=True).to(device)
    opt = torch.optim.AdamW([p for p in teacher.parameters() if p.requires_grad], lr=1e-4)
    ds = SyntheticLivenessSmokeDataset(n=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    t_stats = train_one_epoch_teacher(
        teacher, loader, aug, opt, device, precision="fp16", sdpa_backend="math", max_steps=2
    )
    assert torch.isfinite(torch.tensor(t_stats["loss"]))
