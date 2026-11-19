"""Tests for GPU-vectorized PhysicalLowLightDegrader and LivenessPhysicalAugmentor."""

from __future__ import annotations

import torch

from common.contracts import REC_TRIGGER_LUMA_FLOOR_255
from common.degradations import LivenessPhysicalAugmentor, PhysicalLowLightDegrader, compute_rgb_luma_255
from common.runtime import get_device


def test_physical_degrader_image_and_recognizer_luma_floor() -> None:
    device = get_device()
    x = torch.full((6, 3, 112, 112), 0.75, device=device)

    deg_det = PhysicalLowLightDegrader(mode="detector").to(device)
    out_det = deg_det(x)
    assert out_det.shape == x.shape
    assert torch.isfinite(out_det).all()
    assert out_det.min() >= 0.0 and out_det.max() <= 1.0

    # Extreme dark exposure setting must still trigger safety boost in recognizer mode (>= 22/255)
    deg_rec = PhysicalLowLightDegrader(mode="recognizer", exp_range=(0.01, 0.02)).to(device)
    out_rec = deg_rec(x)
    lumas = compute_rgb_luma_255(out_rec)
    assert float(lumas.min().item()) >= REC_TRIGGER_LUMA_FLOOR_255 - 2.0


def test_temporal_clip_noise_is_iid_per_frame() -> None:
    device = get_device()
    # Constant gray static clip across T=8 frames
    clip = torch.full((2, 8, 3, 64, 64), 0.5, device=device)
    deg = PhysicalLowLightDegrader(mode="liveness", exp_range=(0.25, 0.30), blur_prob=0.0, uvc_prob=0.0).to(device)
    out = deg(clip)
    assert out.shape == clip.shape
    # Because shot+read noise is sampled i.i.d. per frame, adjacent frame diffs must be strictly non-zero
    frame_diffs = (out[:, 1:] - out[:, :-1]).abs().mean(dim=(2, 3, 4))
    assert (frame_diffs > 1e-3).all()


def test_liveness_differentiated_and_frozen_rigid_augmentor() -> None:
    device = get_device()
    aug = LivenessPhysicalAugmentor().to(device)
    clips = torch.rand((4, 8, 3, 128, 128), device=device) * 0.5 + 0.25

    live_out = aug.augment_live(clips)
    print_out = aug.augment_print(clips)
    replay_out = aug.augment_replay(clips)
    frozen_out = aug.synthesize_frozen_rigid_negative(live_out)

    for tensor in (live_out, print_out, replay_out, frozen_out):
        assert tensor.shape == clips.shape
        assert torch.isfinite(tensor).all()
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0
