"""Tests for Phase 2 EdgeFace-S/XS, AdaFace, PartialFC, AdaDistill, and evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.degradations import PhysicalLowLightDegrader
from common.runtime import ModelEMA, get_device
from recognizer.dataset import SyntheticRecognizerSmokeDataset
from recognizer.edgeface import EdgeFaceS, EdgeFaceXS
from recognizer.eval_rec import evaluate_clean_to_dark_pairs
from recognizer.losses import AdaDistillLoss
from recognizer.teacher_cvlface import CVLfaceTeacherWrapper
from recognizer.train import train_one_epoch_recognizer


def test_edgeface_param_counts_and_outputs() -> None:
    device = get_device()
    xs = EdgeFaceXS().to(device)
    s = EdgeFaceS()
    p_xs = sum(p.numel() for p in xs.parameters())
    p_s = sum(p.numel() for p in s.parameters())
    assert 1_000_000 <= p_xs <= 2_600_000, f"Unexpected EdgeFaceXS params: {p_xs}"
    assert 2_800_000 <= p_s <= 4_800_000, f"Unexpected EdgeFaceS params: {p_s}"

    xs.eval()
    x = torch.randn(4, 3, 112, 112, device=device)
    emb, fnorm = xs(x)
    assert emb.shape == (4, 512)
    assert fnorm.shape == (4, 1)
    norms = torch.norm(emb, dim=1)
    assert torch.max(torch.abs(norms - 1.0)) < 1e-5
    assert (fnorm > 0).all()


def test_recognizer_adface_partialfc_adadistill_gpu_smoke() -> None:
    device = get_device()
    student = EdgeFaceXS().to(device)
    teacher = CVLfaceTeacherWrapper(offline_proxy=True).to(device)
    ema = ModelEMA(student, decay=0.999)
    criterion = AdaDistillLoss(num_classes=32, partial_fc_rate=0.5).to(device)
    degrader = PhysicalLowLightDegrader(mode="recognizer").to(device)
    opt = torch.optim.AdamW(list(student.parameters()) + list(criterion.parameters()), lr=2e-4)

    ds = SyntheticRecognizerSmokeDataset(n=16, num_classes=16)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    stats = train_one_epoch_recognizer(
        student, teacher, ema, loader, criterion, degrader, opt, device, precision="fp16", max_steps=2
    )
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())

    # Clean-to-Dark evaluation check
    g = torch.rand(12, 3, 112, 112)
    p = torch.rand(12, 3, 112, 112)
    same = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    rec_metrics = evaluate_clean_to_dark_pairs(ema.ema, g, p, same, device)
    assert "tar_at_far" in rec_metrics and "auc" in rec_metrics
