"""Tests for Phase 1 LCF, YOLO11n-face-5pt, YuNet-S, masked WingLoss, HLA distill & Mean-Teacher."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from common.degradations import PhysicalLowLightDegrader
from common.runtime import ModelEMA, get_device
from detector.dataset import SyntheticDetectorSmokeDataset
from detector.lcf import LearnableCurveFrontend
from detector.losses import DetectorLoss, HLAFeatureDistillLoss
from detector.train import train_one_epoch_detector
from detector.yolo11_face import LCFYOLO11Face5pt, build_detector_teacher
from detector.yunet_lcf import LCFYuNetS


def test_lcf_param_budget_and_zero_init_identity() -> None:
    device = get_device()
    lcf = LearnableCurveFrontend(in_ch=3, hidden_ch=16, num_iters=4).to(device)
    n_params = sum(p.numel() for p in lcf.parameters())
    assert n_params < 30_000, f"LCF too large: {n_params}"

    x = torch.rand(2, 3, 128, 128, device=device)
    y = lcf(x)
    assert torch.max(torch.abs(y - x)) < 1e-6


def test_has_landmarks_zero_gives_zero_kpt_grad() -> None:
    device = get_device()
    model = LCFYOLO11Face5pt(use_p2=True).to(device)
    loss_fn = DetectorLoss(strides=model.head.strides, reg_max=model.head.reg_max).to(device)

    x = torch.rand(2, 3, 320, 320, device=device)
    gt_boxes = torch.tensor(
        [[[80.0, 80.0, 160.0, 170.0]], [[100.0, 90.0, 180.0, 180.0]]], device=device
    )
    gt_valid = torch.ones((2, 1), dtype=torch.bool, device=device)
    gt_kpts = torch.zeros((2, 1, 5, 2), device=device)
    has_lm = torch.zeros((2,), dtype=torch.bool, device=device)  # DarkFace style: NO landmarks!

    model.zero_grad(set_to_none=True)
    out = model(x)
    losses = loss_fn(out, (320, 320), gt_boxes, gt_valid, gt_kpts, has_lm)
    assert float(losses["kpt_loss"].item()) == 0.0
    losses["loss"].backward()

    for p in model.head.kpt_convs.parameters():
        assert p.grad is None or float(p.grad.abs().max().item()) == 0.0


def test_detector_stage1_and_stage2_smoke_on_gpu() -> None:
    device = get_device()
    ds = SyntheticDetectorSmokeDataset(n=8, img_size=320)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    for student in (LCFYOLO11Face5pt(use_p2=True).to(device), LCFYuNetS().to(device)):
        teacher = build_detector_teacher(use_p2=True).to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        ema = ModelEMA(student, decay=0.999)
        det_loss = DetectorLoss(strides=student.head.strides, reg_max=student.head.reg_max).to(device)
        hla_loss = HLAFeatureDistillLoss(student.fpn_channels, teacher.fpn_channels).to(device)
        degrader = PhysicalLowLightDegrader(mode="detector").to(device)
        opt = torch.optim.AdamW(list(student.parameters()) + list(hla_loss.parameters()), lr=2e-4)

        m1 = train_one_epoch_detector(
            student, teacher, ema, loader, det_loss, hla_loss, degrader, opt, device,
            precision="fp16", stage2_semi=False, multiscales=(320,), max_steps=2
        )
        m2 = train_one_epoch_detector(
            student, teacher, ema, loader, det_loss, hla_loss, degrader, opt, device,
            precision="fp16", stage2_semi=True, pseudo_conf_thresh=0.0, multiscales=(320,), max_steps=2
        )
        assert all(torch.isfinite(torch.tensor(v)) for v in m1.values())
        assert all(torch.isfinite(torch.tensor(v)) for v in m2.values())
