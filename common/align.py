"""Authoritative 5-point Umeyama 112x112 alignment, dual-scale FAS cropping, and letterboxing."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from common.contracts import ARCFACE_DST_112, FAS_CROP_SIZE, FAS_SCALES, REC_INPUT_SIZE


def estimate_umeyama_similarity(
    src_pts: np.ndarray,
    dst_pts: np.ndarray = ARCFACE_DST_112,
) -> np.ndarray:
    """Compute 2x3 similarity transform matrix (rotation + uniform scale + translation) via Umeyama SVD."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != (5, 2) or dst.shape != (5, 2):
        raise ValueError(f"Expected (5, 2) landmarks, got {src.shape} and {dst.shape}")

    num = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = (dst_demean.T @ src_demean) / float(num)
    d = np.ones((2,), dtype=np.float64)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0

    u, s, vt = np.linalg.svd(cov)
    rank = np.linalg.matrix_rank(cov)
    if rank == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    t_mat = np.eye(3, dtype=np.float64)
    if rank == 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            t_mat[:2, :2] = u @ vt
        else:
            s_tmp = d[1]
            d[1] = -1.0
            t_mat[:2, :2] = u @ np.diag(d) @ vt
            d[1] = s_tmp
    else:
        t_mat[:2, :2] = u @ np.diag(d) @ vt

    var_src = float((src_demean ** 2).sum() / float(num))
    scale = float(np.dot(s, d) / max(var_src, 1e-12))
    t_mat[:2, :2] *= scale
    t_mat[:2, 2] = dst_mean - scale * (t_mat[:2, :2] / max(scale, 1e-12)) @ src_mean
    # Cleaner exact translation using scaled linear part:
    t_mat[:2, 2] = dst_mean - t_mat[:2, :2] @ src_mean
    return t_mat[:2, :].astype(np.float32)


def _perturb_landmarks(
    kpts5: np.ndarray,
    jitter_sigma_px: float = 0.0,
    scale_jitter: float = 0.0,
    angle_jitter_deg: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    pts = np.asarray(kpts5, dtype=np.float32).copy()
    if rng is None:
        rng = np.random.default_rng()
    if jitter_sigma_px > 0.0:
        pts += rng.normal(0.0, jitter_sigma_px, size=pts.shape).astype(np.float32)
    if scale_jitter > 0.0 or angle_jitter_deg > 0.0:
        center = pts.mean(axis=0, keepdims=True)
        sc = 1.0 + (rng.uniform(-scale_jitter, scale_jitter) if scale_jitter > 0.0 else 0.0)
        ang = np.deg2rad(rng.uniform(-angle_jitter_deg, angle_jitter_deg) if angle_jitter_deg > 0.0 else 0.0)
        c, s = float(np.cos(ang)), float(np.sin(ang))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = ((pts - center) @ rot.T) * np.float32(sc) + center
    return pts


def align_face_112(
    rgb: np.ndarray,
    kpts5: np.ndarray,
    out_size: int = REC_INPUT_SIZE,
    jitter_sigma_px: float = 0.0,
    scale_jitter: float = 0.0,
    angle_jitter_deg: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align raw RGB HWC image to 112x112 RGB via Umeyama similarity transform."""
    pts = _perturb_landmarks(
        kpts5,
        jitter_sigma_px=jitter_sigma_px,
        scale_jitter=scale_jitter,
        angle_jitter_deg=angle_jitter_deg,
        rng=rng,
    )
    dst = ARCFACE_DST_112 * (float(out_size) / float(REC_INPUT_SIZE))
    m = estimate_umeyama_similarity(pts, dst)
    aligned = cv2.warpAffine(
        rgb,
        m,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, m


def crop_square_scale(
    rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    scale: float,
    out_size: int = FAS_CROP_SIZE,
    box_jitter_frac: float = 0.0,
    scale_jitter_frac: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Crop a square patch centered on bbox with side = scale * max(w, h) using edge/zero padding."""
    h_img, w_img = rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(bw, bh)

    if rng is None and (box_jitter_frac > 0.0 or scale_jitter_frac > 0.0):
        rng = np.random.default_rng()
    if box_jitter_frac > 0.0 and rng is not None:
        cx += float(rng.uniform(-box_jitter_frac, box_jitter_frac)) * bw
        cy += float(rng.uniform(-box_jitter_frac, box_jitter_frac)) * bh
    if scale_jitter_frac > 0.0 and rng is not None:
        side *= 1.0 + float(rng.uniform(-scale_jitter_frac, scale_jitter_frac))

    crop_side = max(side * float(scale), 2.0)
    half = 0.5 * crop_side

    # Warp affine maps [cx - half, cy - half] -> (0, 0) and [cx + half, cy + half] -> (out_size, out_size)
    src_x0 = cx - half
    src_y0 = cy - half
    sf = float(out_size) / crop_side
    m = np.array([[sf, 0.0, -src_x0 * sf], [0.0, sf, -src_y0 * sf]], dtype=np.float32)
    patch = cv2.warpAffine(
        rgb,
        m,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return patch


def crop_dual_scale_fas(
    rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    out_size: int = FAS_CROP_SIZE,
    scales: Tuple[float, float] = FAS_SCALES,
    box_jitter_frac: float = 0.0,
    scale_jitter_frac: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop (1.5x texture, 2.7x context) 128x128 RGB patches from original frame coordinates."""
    if rng is None and (box_jitter_frac > 0.0 or scale_jitter_frac > 0.0):
        rng = np.random.default_rng()
    # Apply the same detector frame jitter to both scales for spatial coherence within frame t
    h_img, w_img = rgb.shape[:2]
    _ = (h_img, w_img)
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(bw, bh)
    if rng is not None:
        if box_jitter_frac > 0.0:
            cx += float(rng.uniform(-box_jitter_frac, box_jitter_frac)) * bw
            cy += float(rng.uniform(-box_jitter_frac, box_jitter_frac)) * bh
        if scale_jitter_frac > 0.0:
            side *= 1.0 + float(rng.uniform(-scale_jitter_frac, scale_jitter_frac))
    jittered_box = np.array([cx - 0.5 * side, cy - 0.5 * side, cx + 0.5 * side, cy + 0.5 * side], dtype=np.float32)
    tex = crop_square_scale(rgb, jittered_box, scale=scales[0], out_size=out_size)
    ctx = crop_square_scale(rgb, jittered_box, scale=scales[1], out_size=out_size)
    return tex, ctx


def letterbox_rgb(
    rgb: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Aspect-preserving resize + gray letterbox padding to (target_h, target_w)."""
    h, w = rgb.shape[:2]
    th, tw = target_size
    r = min(th / float(h), tw / float(w))
    nh = int(round(h * r))
    nw = int(round(w * r))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR)
    dw = (tw - nw) / 2.0
    dh = (th - nh) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    # Guarantee exact output shape even on odd deltas
    bottom = th - nh - top
    right = tw - nw - left
    out = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return out, float(r), (float(left), float(top))


def unletterbox_coords(
    boxes_xyxy: Optional[np.ndarray],
    kpts_5: Optional[np.ndarray],
    scale: float,
    pad_xy: Tuple[float, float],
    orig_hw: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Map letterboxed boxes [N, 4] and 5-pt landmarks [N, 5, 2] back to original RGB coords."""
    pad_x, pad_y = pad_xy
    oh, ow = orig_hw
    inv_s = 1.0 / max(scale, 1e-12)

    out_boxes = None
    if boxes_xyxy is not None:
        b = np.asarray(boxes_xyxy, dtype=np.float32).copy()
        b[..., 0::2] = (b[..., 0::2] - pad_x) * inv_s
        b[..., 1::2] = (b[..., 1::2] - pad_y) * inv_s
        b[..., 0::2] = np.clip(b[..., 0::2], 0.0, float(ow))
        b[..., 1::2] = np.clip(b[..., 1::2], 0.0, float(oh))
        out_boxes = b

    out_kpts = None
    if kpts_5 is not None:
        k = np.asarray(kpts_5, dtype=np.float32).copy()
        k[..., 0] = (k[..., 0] - pad_x) * inv_s
        k[..., 1] = (k[..., 1] - pad_y) * inv_s
        out_kpts = k

    return out_boxes, out_kpts


def composite_tight_face_on_canvas(
    face_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    kpts_5: np.ndarray,
    bg_rgb: Optional[np.ndarray] = None,
    canvas_size: int = 480,
    target_patch_size: Optional[int] = None,
    center_xy: Optional[Tuple[int, int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Softly feather-blend a tight face crop (e.g. WFLW 112x112) onto a full-scene background canvas."""
    if rng is None:
        rng = np.random.default_rng(42)
    fh, fw = face_rgb.shape[:2]

    if bg_rgb is not None and bg_rgb.size > 0:
        canvas = cv2.resize(bg_rgb, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR).copy()
    else:
        base_val = int(rng.integers(90, 175))
        canvas = np.full((canvas_size, canvas_size, 3), base_val, dtype=np.uint8)

    if target_patch_size is None:
        target_patch_size = int(rng.integers(int(canvas_size * 0.24), int(canvas_size * 0.60)))
    target_patch_size = max(48, min(canvas_size - 16, target_patch_size))

    scale = float(target_patch_size) / float(max(fh, fw))
    nh = max(16, int(round(fh * scale)))
    nw = max(16, int(round(fw * scale)))
    patch = cv2.resize(face_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    if center_xy is None:
        x0 = int(rng.integers(8, max(9, canvas_size - nw - 8)))
        y0 = int(rng.integers(8, max(9, canvas_size - nh - 8)))
    else:
        x0 = max(0, min(canvas_size - nw, center_xy[0] - nw // 2))
        y0 = max(0, min(canvas_size - nh, center_xy[1] - nh // 2))

    # 8px border feather mask to prevent sharp rectangle boundary shortcuts
    feather = max(4, min(12, nw // 10))
    mask = np.ones((nh, nw, 1), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
    mask[:, :feather, 0] *= ramp[None, :]
    mask[:, -feather:, 0] *= ramp[::-1][None, :]
    mask[:feather, :, 0] *= ramp[:, None]
    mask[-feather:, :, 0] *= ramp[::-1][:, None]

    roi = canvas[y0 : y0 + nh, x0 : x0 + nw].astype(np.float32)
    blended = patch.astype(np.float32) * mask + roi * (1.0 - mask)
    canvas[y0 : y0 + nh, x0 : x0 + nw] = np.clip(blended, 0, 255).astype(np.uint8)

    new_boxes = boxes_xyxy.copy()
    new_kpts = kpts_5.copy()
    if len(new_boxes):
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]] * (float(nw) / float(fw)) + x0
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]] * (float(nh) / float(fh)) + y0
    if len(new_kpts):
        new_kpts[..., 0] = new_kpts[..., 0] * (float(nw) / float(fw)) + x0
        new_kpts[..., 1] = new_kpts[..., 1] * (float(nh) / float(fh)) + y0
    return canvas, new_boxes, new_kpts
