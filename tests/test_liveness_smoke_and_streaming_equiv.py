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
from liveness.train import train_one_epoch_liveness


def test_st_fas_lite_clip_vs_streaming_pytorch_equiv() -> None:
    device = get_device()
    model = STFASLiteClip().eval().to(device)
    clip_tex = torch.rand(2, 8, 3, 128, 128, device=device)
    clip_ctx = torch.rand(2, 8, 3, 128, 128, device=device)

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

    # Allow 1e-4 for CUDA SM 8.9 cuDNN TF32 reduction order between batched clip vs sequential frames
    assert torch.max(torch.abs(c_live - s_live)) < 1e-4
    assert torch.max(torch.abs(c_logits - s_logits)) < 1e-4
    assert torch.max(torch.abs(c_clarity - s_clarity)) < 1e-4


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
