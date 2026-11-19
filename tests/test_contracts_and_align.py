"""Tests for shared contracts, Umeyama 5-pt alignment, dual-scale FAS crops, and DarkFace score.py parity."""

from __future__ import annotations

import math

import numpy as np

from common.align import (
    align_face_112,
    crop_dual_scale_fas,
    estimate_umeyama_similarity,
    letterbox_rgb,
    unletterbox_coords,
)
from common.contracts import ARCFACE_DST_112, FAS_CROP_SIZE, REC_INPUT_SIZE
from common.metrics import compute_nme_5pt, evaluate_darkface_rows


def test_umeyama_similarity_recovery() -> None:
    theta = math.radians(18.0)
    scale = 1.75
    tx, ty = 120.0, 85.0
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    # Construct source points via inverse mapping so mapping src -> ARCFACE_DST_112 recovers ARCFACE_DST_112
    src = (ARCFACE_DST_112 @ rot.T) * scale + np.array([tx, ty], dtype=np.float32)
    m = estimate_umeyama_similarity(src, ARCFACE_DST_112)
    hom = np.hstack([src, np.ones((5, 1), dtype=np.float32)])
    reproj = hom @ m.T
    assert np.max(np.abs(reproj - ARCFACE_DST_112)) < 1e-3


def test_align_112_and_jitter() -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[60:180, 60:180] = [200, 150, 120]
    kpts = ARCFACE_DST_112 + np.array([64.0, 64.0], dtype=np.float32)

    clean, _ = align_face_112(img, kpts)
    rng = np.random.default_rng(123)
    jittered, _ = align_face_112(img, kpts, jitter_sigma_px=3.0, scale_jitter=0.05, angle_jitter_deg=4.0, rng=rng)
    assert clean.shape == (REC_INPUT_SIZE, REC_INPUT_SIZE, 3)
    assert jittered.shape == (REC_INPUT_SIZE, REC_INPUT_SIZE, 3)
    assert not np.array_equal(clean, jittered)


def test_dual_scale_fas_crop_and_letterbox_roundtrip() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Put bright border outside 1.5x box but inside 2.7x box so ctx sees it and tex mostly doesn't
    img[60:420, 140:500] = [10, 220, 10]
    img[180:300, 260:380] = [220, 80, 80]
    bbox = np.array([260.0, 180.0, 380.0, 300.0], dtype=np.float32)

    tex, ctx = crop_dual_scale_fas(img, bbox)
    assert tex.shape == (FAS_CROP_SIZE, FAS_CROP_SIZE, 3)
    assert ctx.shape == (FAS_CROP_SIZE, FAS_CROP_SIZE, 3)
    # Texture crop sees predominantly red center; context crop sees more green surround
    assert tex[..., 0].mean() > ctx[..., 0].mean()
    assert ctx[..., 1].mean() > tex[..., 1].mean()

    lb, scale, pad = letterbox_rgb(img, (640, 640))
    assert lb.shape == (640, 640, 3)
    scaled_box = bbox.copy()
    scaled_box[[0, 2]] = scaled_box[[0, 2]] * scale + pad[0]
    scaled_box[[1, 3]] = scaled_box[[1, 3]] * scale + pad[1]
    kpts = np.array([[290.0, 210.0], [350.0, 210.0], [320.0, 240.0], [300.0, 270.0], [340.0, 270.0]], dtype=np.float32)
    scaled_kpts = kpts.copy()
    scaled_kpts[:, 0] = scaled_kpts[:, 0] * scale + pad[0]
    scaled_kpts[:, 1] = scaled_kpts[:, 1] * scale + pad[1]

    rec_box, rec_kpts = unletterbox_coords(scaled_box[None], scaled_kpts[None], scale, pad, (480, 640))
    assert np.max(np.abs(rec_box[0] - bbox)) < 1e-3
    assert np.max(np.abs(rec_kpts[0] - kpts)) < 1e-3


def test_darkface_metrics_and_nme() -> None:
    rows = [
        {
            "kind": "image",
            "gt": [[10, 10, 20, 20, 1], [100, 100, 30, 30, 1]],
            "raw_native": [[10, 10, 20, 20, 0.95], [100, 100, 30, 30, 0.85]],
            "mean_luma": 15.0,
            "face_luma": [12.0, 18.0],
        },
        {
            "kind": "image",
            "gt": [],
            "raw_native": [[5, 5, 10, 10, 0.2]],
            "mean_luma": 10.0,
            "face_luma": [],
        },
    ]
    res = evaluate_darkface_rows(rows, pred_key="raw_native")
    assert res["matched_iou50"] == 2
    assert res["recall_iou50"] == 1.0
    assert res["AP50_VOC_greedy"] > 0.99
    assert res["AP50_official_tools_style"] > 0.99
    assert res["empty_frame_fppi"] == 1.0

    gt_k = np.array([[[30.0, 50.0], [70.0, 50.0], [50.0, 70.0], [35.0, 85.0], [65.0, 85.0]]])
    pred_k = gt_k + np.array([[[0.8, 0.0], [0.8, 0.0], [0.8, 0.0], [0.8, 0.0], [0.8, 0.0]]])
    nme = compute_nme_5pt(pred_k, gt_k)
    assert abs(nme - 0.8 / 40.0) < 1e-6
