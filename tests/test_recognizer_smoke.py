"""Tests for Phase 2 EdgeFace-S/XS, AdaFace, PartialFC, AdaDistill, and evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.degradations import PhysicalLowLightDegrader
from common.runtime import ModelEMA, get_device
from eval_e2e_onnx import export_onnx_model, verify_pytorch_vs_onnx
from recognizer.dataset import SyntheticRecognizerSmokeDataset
from recognizer.edgeface import EdgeFaceS, EdgeFaceXS, OfficialEdgeFace
from recognizer.eval_rec import evaluate_clean_to_dark_pairs
from recognizer.losses import AdaDistillLoss
from recognizer.teacher_cvlface import CVLfaceTeacherWrapper
from recognizer.train import (
    align_student_head_procrustes,
    set_recognizer_warmup_trainable,
    train_one_epoch_recognizer,
)


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


def test_official_edgeface_onnx_and_vectorized_adadistill(tmp_path) -> None:
    device = get_device()
    model = OfficialEdgeFace("edgeface_s_gamma_05", emb_dim=512).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert 3_000_000 <= n_params <= 4_500_000, f"Unexpected OfficialEdgeFace params: {n_params}"

    # Verify warmup freeze toggles early stages cleanly
    set_recognizer_warmup_trainable(model, freeze_early=True)
    assert not any(p.requires_grad for p in model.backbone.model.stages[0].parameters())
    assert all(p.requires_grad for p in model.backbone.model.stages[2].parameters())
    set_recognizer_warmup_trainable(model, freeze_early=False)
    assert all(p.requires_grad for p in model.parameters())

    # Verify ONNX export + parity < 1e-4
    onnx_path = tmp_path / "official_edgeface_s.onnx"
    dummy = np.random.uniform(-1.0, 1.0, (1, 3, 112, 112)).astype(np.float32)
    export_onnx_model(
        model,
        (torch.from_numpy(dummy),),
        onnx_path,
        input_names=["rgb_norm"],
        output_names=["l2_emb", "feat_norm"],
    )
    ver = verify_pytorch_vs_onnx(model.cpu(), onnx_path, {"rgb_norm": dummy})
    assert ver["passed"], f"ONNX parity failed: {ver}"
    assert ver["size_mb"] <= 16.0

    # Verify vectorized teacher center update, PartialFC init from teacher & unseen negative masking
    crit = AdaDistillLoss(num_classes=64, partial_fc_rate=0.5, lambda_clean_kd=1.5).to(device)
    zc = torch.nn.functional.normalize(torch.randn(8, 512, device=device, requires_grad=True), dim=1)
    zd = torch.nn.functional.normalize(torch.randn(8, 512, device=device, requires_grad=True), dim=1)
    zt = torch.nn.functional.normalize(torch.randn(8, 512, device=device), dim=1)
    nc = torch.full((8, 1), 20.0, device=device)
    nd = torch.full((8, 1), 15.0, device=device)
    labels = torch.tensor([0, 1, 2, 0, 1, 3, 4, 5], device=device)
    out = crit(zc, nc, zd, nd, zt, labels)
    assert crit.teacher_seen[:6].all() and not crit.teacher_seen[6:].any()
    assert all(torch.isfinite(v) for v in out.values())


def test_procrustes_alignment_and_precomputed_centers() -> None:
    device = get_device()
    model = OfficialEdgeFace("edgeface_s_gamma_05", emb_dim=512).to(device).eval()
    ds = SyntheticRecognizerSmokeDataset(n=128, num_classes=16)

    # Synthetic orthogonal rotation as proxy teacher mapping
    rng = np.random.default_rng(7)
    q_np, _ = np.linalg.qr(rng.normal(size=(512, 512)).astype(np.float32))
    q_mat = torch.from_numpy(q_np.astype(np.float32)).to(device)

    class _RotProxyTeacher(torch.nn.Module):
        def __init__(self, ref_model: OfficialEdgeFace, rot: torch.Tensor) -> None:
            super().__init__()
            self.ref = ref_model
            self.rot = rot

        @torch.no_grad()
        def forward(self, x_clean: torch.Tensor, kpts: torch.Tensor | None = None) -> torch.Tensor:
            z, _ = self.ref(x_clean)
            return torch.nn.functional.normalize(z @ self.rot, dim=1)

    teacher = _RotProxyTeacher(model, q_mat).to(device).eval()
    # Freeze a detached snapshot reference for teacher before model weights are rotated in-place
    import copy

    ref_copy = copy.deepcopy(model).to(device).eval()
    teacher.ref = ref_copy

    res = align_student_head_procrustes(model, ds, device, teacher=teacher, n_samples=128, batch_size=64)
    assert res["applied"], f"Procrustes failed: {res}"
    assert res["max_pair_cos_diff"] < 1e-5, f"Pairwise cosine changed: {res['max_pair_cos_diff']}"
    assert res["cos_teacher_after"] > 0.70, f"Unexpected low Procrustes cosine: {res}"

    crit = AdaDistillLoss(num_classes=16, partial_fc_rate=0.5).to(device)
    raw_ctrs = torch.nn.functional.normalize(torch.randn(16, 512), dim=1)
    crit.load_precomputed_teacher_centers(raw_ctrs)
    assert crit.teacher_seen.all()


def test_rank_compatible_kl_kd_and_ref_anchor() -> None:
    import copy

    device = get_device()
    student = OfficialEdgeFace("edgeface_s_gamma_05", emb_dim=512).to(device)
    ref_student = copy.deepcopy(student).eval()
    for p in ref_student.parameters():
        p.requires_grad_(False)
    teacher = CVLfaceTeacherWrapper(offline_proxy=True).to(device)
    ema = ModelEMA(student, decay=0.997, warmup_tau=0.0)
    crit = AdaDistillLoss(
        num_classes=32,
        partial_fc_rate=0.5,
        lambda_ada_clean=0.25,
        lambda1_dark_adaface=0.5,
        lambda2_adadistill=1.0,
        lambda3_teacher_cos=1.5,
        lambda4_student_cos=1.0,
        lambda_clean_kd=0.5,
        lambda_rel_kd=15.0,
        lambda_ref_clean=3.0,
        lambda_dark_ref=5.0,
        use_kd_projector=True,
        adadistill_mode="kl",
    ).to(device)
    crit.load_precomputed_teacher_centers(
        torch.nn.functional.normalize(torch.randn(32, 512), dim=1),
        init_partial_fc=False,
    )
    assert crit.teacher_seen.all() and not crit.partial_fc_seen.any()

    degrader = PhysicalLowLightDegrader(mode="recognizer").to(device)
    opt = torch.optim.AdamW(list(student.parameters()) + list(crit.parameters()), lr=1e-4)
    ds = SyntheticRecognizerSmokeDataset(n=16, num_classes=16)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    stats = train_one_epoch_recognizer(
        student=student,
        teacher=teacher,
        ema=ema,
        loader=loader,
        criterion=crit,
        degrader=degrader,
        optimizer=opt,
        device=device,
        precision="fp16",
        max_steps=2,
        ref_student=ref_student,
    )
    assert crit.partial_fc_seen.any()
    assert all(np.isfinite(float(v)) for k, v in stats.items() if isinstance(v, (int, float)))

