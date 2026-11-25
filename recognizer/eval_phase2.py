"""Phase 2 Offline Acceptance Evaluator: 5 Public Benchmarks (Clean / Dark-GT / Dark-Jitter) + Closed-Loop Phase-1 Detector 5pt."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common.align import _perturb_landmarks, align_face_112, estimate_umeyama_similarity
from common.contracts import ARCFACE_DST_112
from common.degradations import PhysicalLowLightDegrader
from common.metrics import compute_nme_5pt, compute_tar_at_far
from common.runtime import get_device, seed_everything
from detector.eval_phase1 import load_detector_checkpoint, predict_image_rgb
from recognizer.edgeface import OfficialEdgeFace, build_recognizer_student
from recognizer.teacher_cvlface import CVLfaceTeacherWrapper


DEFAULT_BINS = ("lfw", "cfp_fp", "agedb_30", "cplfw", "calfw")


def load_insightface_bin(bin_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Decode an InsightFace `.bin` verification pack into `(rgb_u8 [2*N, 112, 112, 3], is_same [N])`."""
    with open(bin_path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")
    n_pairs = len(issame_list)
    n_imgs = len(bins)
    if n_imgs != 2 * n_pairs:
        raise ValueError(f"{bin_path}: expected {2 * n_pairs} images, found {n_imgs}")

    imgs = np.empty((n_imgs, 112, 112, 3), dtype=np.uint8)
    for i, buf in enumerate(bins):
        arr = np.frombuffer(buf, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode image {i} in {bin_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (112, 112):
            rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        imgs[i] = rgb
    return imgs, np.asarray(issame_list, dtype=bool)


def evaluate_10fold_verification(
    scores: np.ndarray,
    is_same: np.ndarray,
    n_folds: int = 10,
) -> Dict[str, float]:
    """Standard LFW/InsightFace 10-fold CV verification accuracy across cosine similarity thresholds."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(is_same, dtype=bool)
    n = len(s)
    thresholds = np.linspace(-1.0, 1.0, 4001)
    fold_size = n // n_folds
    accs: List[float] = []
    best_ths: List[float] = []

    for fi in range(n_folds):
        val_start = fi * fold_size
        val_end = (fi + 1) * fold_size if fi < n_folds - 1 else n
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True
        tr_mask = ~val_mask

        # Vectorized accuracy sweep on training folds: [T, N_tr] -> [T]
        tr_preds = s[tr_mask][None, :] >= thresholds[:, None]
        tr_accs = (tr_preds == y[tr_mask][None, :]).mean(axis=1)
        best_idx = int(np.argmax(tr_accs))
        th = float(thresholds[best_idx])
        val_acc = float(((s[val_mask] >= th) == y[val_mask]).mean())
        accs.append(val_acc)
        best_ths.append(th)

    return {
        "acc_mean_pct": round(float(np.mean(accs) * 100.0), 3),
        "acc_std_pct": round(float(np.std(accs) * 100.0), 3),
        "best_thresh_mean": round(float(np.mean(best_ths)), 4),
    }


@torch.no_grad()
def embed_rgb_u8_batch(
    model: torch.nn.Module,
    imgs_rgb_u8: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    flip_avg: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model / teacher on `[N, 112, 112, 3]` RGB uint8 -> `(l2_embs [N, 512], norms [N])`."""
    model.eval()
    embs: List[np.ndarray] = []
    norms: List[np.ndarray] = []

    def _run_batch(x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = model(x_norm)
        if isinstance(out, (tuple, list)):
            z, fn = out[0], out[1].squeeze(-1)
        else:
            z = out
            fn = torch.ones((x_norm.shape[0],), device=x_norm.device, dtype=torch.float32)
        return F.normalize(z.float(), dim=1), fn.float()

    for i in range(0, len(imgs_rgb_u8), batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(imgs_rgb_u8[i : i + batch_size].transpose(0, 3, 1, 2))).to(
            device=device, dtype=torch.float32
        )
        x = (chunk / 255.0 - 0.5) / 0.5
        z, fn = _run_batch(x)
        if flip_avg:
            z_f, _ = _run_batch(torch.flip(x, dims=[3]))
            z = F.normalize(z + z_f, dim=1)
        embs.append(z.cpu().numpy())
        norms.append(fn.cpu().numpy())

    return np.concatenate(embs, axis=0), np.concatenate(norms, axis=0)


def _apply_jitter_batch_112(
    imgs_rgb_u8: np.ndarray,
    seed: int,
    jitter_sigma_px: float = 2.0,
    scale_jitter: float = 0.03,
    angle_jitter_deg: float = 3.0,
) -> np.ndarray:
    """Apply deterministic Phase-1 empirical 5-pt alignment perturbation to `[N, 112, 112, 3]`."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(imgs_rgb_u8)
    for i in range(len(imgs_rgb_u8)):
        pts = _perturb_landmarks(
            ARCFACE_DST_112,
            jitter_sigma_px=jitter_sigma_px,
            scale_jitter=scale_jitter,
            angle_jitter_deg=angle_jitter_deg,
            rng=rng,
        )
        m = estimate_umeyama_similarity(pts, ARCFACE_DST_112)
        out[i] = cv2.warpAffine(
            imgs_rgb_u8[i],
            m,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return out


@torch.no_grad()
def apply_deterministic_dark_112(
    imgs_rgb_u8: np.ndarray,
    degrader: PhysicalLowLightDegrader,
    device: torch.device,
    seed: int,
    batch_size: int = 128,
) -> Tuple[np.ndarray, float]:
    """Deterministically apply GPU `PhysicalLowLightDegrader(mode='recognizer')` to `[N, 112, 112, 3]`."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        out = np.empty_like(imgs_rgb_u8)
        luma_vals: List[np.ndarray] = []
        for i in range(0, len(imgs_rgb_u8), batch_size):
            t = torch.from_numpy(np.ascontiguousarray(imgs_rgb_u8[i : i + batch_size].transpose(0, 3, 1, 2))).to(
                device=device, dtype=torch.float32
            ).div_(255.0)
            deg = degrader(t)
            y = (0.299 * deg[:, 0] + 0.587 * deg[:, 1] + 0.114 * deg[:, 2]) * 255.0
            luma_vals.append(y.mean(dim=(1, 2)).cpu().numpy())
            arr_u8 = (deg * 255.0).round().clamp(0.0, 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
            out[i : i + len(arr_u8)] = arr_u8
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    mean_luma = float(np.concatenate(luma_vals).mean()) if luma_vals else 0.0
    return out, mean_luma


def _score_summary(
    g_emb: np.ndarray,
    g_norm: np.ndarray,
    p_emb: np.ndarray,
    p_norm: np.ndarray,
    is_same: np.ndarray,
) -> Dict[str, Any]:
    cos_scores = (g_emb * p_emb).sum(axis=1)
    cv10 = evaluate_10fold_verification(cos_scores, is_same)
    tar = compute_tar_at_far(cos_scores[is_same], cos_scores[~is_same])
    return {
        **cv10,
        **tar,
        "gallery_norm_mean": round(float(g_norm.mean()), 3),
        "probe_norm_genuine_mean": round(float(p_norm[is_same].mean()), 3),
        "probe_norm_impostor_mean": round(float(p_norm[~is_same].mean()), 3),
    }


def build_bin_protocol_cache(
    bin_path: Path,
    degrader: PhysicalLowLightDegrader,
    device: torch.device,
    seed_offset: int = 0,
    include_dark_gt: bool = True,
    include_dark_jitter: bool = True,
) -> Dict[str, Any]:
    """Precompute deterministic Clean, Dark-GT, and Dark-Jitter probe tensors once so all models compare identically."""
    imgs, is_same = load_insightface_bin(bin_path)
    gallery_clean = imgs[0::2]
    probe_clean = imgs[1::2]
    cache: Dict[str, Any] = {
        "name": bin_path.stem,
        "n_pairs": int(len(is_same)),
        "is_same": is_same,
        "gallery_clean": gallery_clean,
        "probe_clean": probe_clean,
    }
    if include_dark_gt:
        dark_gt, luma_gt = apply_deterministic_dark_112(probe_clean, degrader, device, seed=10000 + seed_offset)
        cache["probe_dark_gt"] = dark_gt
        cache["probe_dark_gt_mean_luma"] = round(luma_gt, 2)
    if include_dark_jitter:
        jit = _apply_jitter_batch_112(probe_clean, seed=20000 + seed_offset)
        dark_jit, luma_jit = apply_deterministic_dark_112(jit, degrader, device, seed=30000 + seed_offset)
        cache["probe_dark_jitter"] = dark_jit
        cache["probe_dark_jitter_mean_luma"] = round(luma_jit, 2)
    return cache


def evaluate_model_on_bin_cache(
    model: torch.nn.Module,
    cache: Dict[str, Any],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    is_same = cache["is_same"]
    g_s, gn_s = embed_rgb_u8_batch(model, cache["gallery_clean"], device, batch_size=batch_size, flip_avg=False)
    p_s, pn_s = embed_rgb_u8_batch(model, cache["probe_clean"], device, batch_size=batch_size, flip_avg=False)
    g_f, _ = embed_rgb_u8_batch(model, cache["gallery_clean"], device, batch_size=batch_size, flip_avg=True)
    p_f, _ = embed_rgb_u8_batch(model, cache["probe_clean"], device, batch_size=batch_size, flip_avg=True)

    out: Dict[str, Any] = {
        "n_pairs": cache["n_pairs"],
        "clean_clean_flip_avg": _score_summary(g_f, gn_s, p_f, pn_s, is_same),
        "clean_clean": _score_summary(g_s, gn_s, p_s, pn_s, is_same),
    }
    if "probe_dark_gt" in cache:
        pd_gt, pd_gt_n = embed_rgb_u8_batch(model, cache["probe_dark_gt"], device, batch_size=batch_size, flip_avg=False)
        res = _score_summary(g_s, gn_s, pd_gt, pd_gt_n, is_same)
        res["probe_mean_luma_255"] = cache["probe_dark_gt_mean_luma"]
        out["clean_to_dark_gt"] = res
    if "probe_dark_jitter" in cache:
        pd_j, pd_jn = embed_rgb_u8_batch(model, cache["probe_dark_jitter"], device, batch_size=batch_size, flip_avg=False)
        res = _score_summary(g_s, gn_s, pd_j, pd_jn, is_same)
        res["probe_mean_luma_255"] = cache["probe_dark_jitter_mean_luma"]
        out["clean_to_dark_jitter"] = res
    return out


@torch.no_grad()
def evaluate_fast_val(
    model: torch.nn.Module,
    eval_dir: Path,
    device: torch.device,
    benchmarks: Sequence[str] = ("lfw", "cfp_fp", "agedb_30"),
    bin_caches: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fast per-epoch validation returning Clean-Clean avg accuracy and Dark-Jitter TAR@FAR=1e-4."""
    degrader = PhysicalLowLightDegrader(mode="recognizer").to(device)
    if bin_caches is None:
        bin_caches = {}
        for idx, name in enumerate(benchmarks):
            p = eval_dir / f"{name}.bin"
            if p.is_file():
                bin_caches[name] = build_bin_protocol_cache(
                    p, degrader, device, seed_offset=idx * 97, include_dark_gt=False, include_dark_jitter=True
                )

    per_bench: Dict[str, Any] = {}
    clean_accs: List[float] = []
    dark_tars_1e4: List[float] = []
    dark_tars_1e3: List[float] = []

    for name, cache in bin_caches.items():
        res = evaluate_model_on_bin_cache(model, cache, device)
        per_bench[name] = {
            "clean_acc_flip": res["clean_clean_flip_avg"]["acc_mean_pct"],
            "clean_acc": res["clean_clean"]["acc_mean_pct"],
            "dark_jit_acc": res["clean_to_dark_jitter"]["acc_mean_pct"],
            "dark_jit_tar_1e4": res["clean_to_dark_jitter"]["tar_at_far"]["FAR=1e-04"]["TAR"],
            "dark_jit_tar_1e3": res["clean_to_dark_jitter"]["tar_at_far"]["FAR=1e-03"]["TAR"],
        }
        clean_accs.append(res["clean_clean_flip_avg"]["acc_mean_pct"])
        dark_tars_1e4.append(res["clean_to_dark_jitter"]["tar_at_far"]["FAR=1e-04"]["TAR"])
        dark_tars_1e3.append(res["clean_to_dark_jitter"]["tar_at_far"]["FAR=1e-03"]["TAR"])

    return {
        "benchmarks": per_bench,
        "clean_acc_flip_mean": round(float(np.mean(clean_accs)), 4) if clean_accs else 0.0,
        "dark_jit_tar_1e4_mean": round(float(np.mean(dark_tars_1e4)), 4) if dark_tars_1e4 else 0.0,
        "dark_jit_tar_1e3_mean": round(float(np.mean(dark_tars_1e3)), 4) if dark_tars_1e3 else 0.0,
    }


def _cosine_distribution_stats(scores: np.ndarray) -> Dict[str, Any]:
    bins = np.linspace(-0.4, 1.0, 29)
    hist, edges = np.histogram(scores, bins=bins)
    return {
        "count": int(len(scores)),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "p01": round(float(np.percentile(scores, 1)), 4),
        "p05": round(float(np.percentile(scores, 5)), 4),
        "p10": round(float(np.percentile(scores, 10)), 4),
        "p50": round(float(np.percentile(scores, 50)), 4),
        "p90": round(float(np.percentile(scores, 90)), 4),
        "p95": round(float(np.percentile(scores, 95)), 4),
        "p99": round(float(np.percentile(scores, 99)), 4),
        "hist_counts": [int(v) for v in hist],
        "hist_edges": [round(float(v), 3) for v in edges],
    }


def _recommend_fixed_threshold(
    clean_gen: np.ndarray,
    dark_gen: np.ndarray,
    dark_imp: np.ndarray,
) -> Dict[str, Any]:
    cands = np.linspace(0.05, 0.75, 701)
    best = None
    for th in cands:
        far = float((dark_imp >= th).mean())
        tar_c = float((clean_gen >= th).mean())
        tar_d = float((dark_gen >= th).mean())
        # Target strict FAR <= 0.1% while maximizing dark TAR
        if far <= 1e-3:
            key = (tar_d, tar_c, -far)
            if best is None or key > best[0]:
                best = (key, {"threshold": round(float(th), 4), "far_dark": round(far, 5), "tar_clean": round(tar_c, 4), "tar_dark": round(tar_d, 4)})
    if best is None:
        # Relax to <= 1% FAR if needed
        idx = int(np.argmax([0.5 * ((clean_gen >= th).mean() + (dark_gen >= th).mean()) - (dark_imp >= th).mean() for th in cands]))
        th = float(cands[idx])
        return {
            "threshold": round(th, 4),
            "far_dark": round(float((dark_imp >= th).mean()), 5),
            "tar_clean": round(float((clean_gen >= th).mean()), 4),
            "tar_dark": round(float((dark_gen >= th).mean()), 4),
        }
    return best[1]


@torch.no_grad()
def prepare_lfw_raw_closed_loop_cache(
    lfw_raw_dir: Path,
    det_ckpt: Path,
    device: torch.device,
    seed: int = 7777,
) -> Dict[str, Any]:
    """Run Phase 1 Detector on clean & physically degraded raw 250x250 LFW images to cache aligned crops."""
    pairs_csv = lfw_raw_dir / "pairs.csv"
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Missing {pairs_csv}")

    pairs: List[Tuple[str, str, bool]] = []
    unique_paths: List[str] = []
    seen_set = set()
    with open(pairs_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pa = row["image_a_path"].strip()
            pb = row["image_b_path"].strip()
            same = bool(int(row["is_same"]))
            pairs.append((pa, pb, same))
            for p in (pa, pb):
                if p not in seen_set:
                    seen_set.add(p)
                    unique_paths.append(p)

    det_model = load_detector_checkpoint("lcf_yolo11n_5pt", det_ckpt, device)
    degrader = PhysicalLowLightDegrader(mode="recognizer").to(device)

    img_to_idx = {p: i for i, p in enumerate(unique_paths)}
    n_u = len(unique_paths)
    clean_aligned = np.empty((n_u, 112, 112, 3), dtype=np.uint8)
    dark_ref5_aligned = np.empty((n_u, 112, 112, 3), dtype=np.uint8)
    dark_det5_aligned = np.empty((n_u, 112, 112, 3), dtype=np.uint8)
    dark_detected = np.zeros((n_u,), dtype=bool)
    ref_kpts_list: List[np.ndarray] = []
    det_kpts_list: List[np.ndarray] = []
    det_valid_indices: List[int] = []

    fallback_kpt = (ARCFACE_DST_112 / 112.0) * np.array([150.0, 150.0], dtype=np.float32) + np.array([50.0, 50.0], dtype=np.float32)
    fallback_box = np.array([65.0, 55.0, 185.0, 200.0], dtype=np.float32)

    clean_misses = 0
    raw_cache_dark_luma: List[float] = []

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        for i, rel in enumerate(unique_paths):
            bgr = cv2.imread(str(lfw_raw_dir / "train" / rel), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(lfw_raw_dir / "train" / rel)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # Center face selection on clean 250x250 raw LFW frame
            c_boxes, c_scores, c_kpts = predict_image_rgb(det_model, rgb, device, conf_thresh=0.15)
            if len(c_boxes) == 0:
                clean_misses += 1
                ref_box = fallback_box
                ref_kpt = fallback_kpt
            else:
                cx = (c_boxes[:, 0] + c_boxes[:, 2]) * 0.5
                cy = (c_boxes[:, 1] + c_boxes[:, 3]) * 0.5
                dist_center = np.hypot(cx - 0.5 * w, cy - 0.5 * h)
                best_i = int(np.argmin(dist_center - c_scores * 30.0))
                ref_box = c_boxes[best_i]
                ref_kpt = c_kpts[best_i]

            c_112, _ = align_face_112(rgb, ref_kpt, out_size=112)
            clean_aligned[i] = c_112

            # Physical dark degradation bounded on actual face bbox
            t_in = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device).float().div_(255.0)
            bb_norm = torch.tensor(
                [[ref_box[0] / w, ref_box[1] / h, ref_box[2] / w, ref_box[3] / h]],
                device=device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
            dark_t = degrader(t_in, bboxes_norm=bb_norm)
            dark_u8 = (dark_t[0] * 255.0).round().clamp(0.0, 255.0).byte().permute(1, 2, 0).cpu().numpy()
            x1, y1, x2, y2 = [int(max(0, min(w - 1, v))) for v in ref_box]
            face_patch = dark_u8[y1 : max(y1 + 1, y2), x1 : max(x1 + 1, x2)]
            raw_cache_dark_luma.append(
                float((0.299 * face_patch[..., 0] + 0.587 * face_patch[..., 1] + 0.114 * face_patch[..., 2]).mean())
            )

            d_ref_112, _ = align_face_112(dark_u8, ref_kpt, out_size=112)
            dark_ref5_aligned[i] = d_ref_112

            # End-to-end Phase 1 Detector inference directly on degraded 250x250 dark frame
            d_boxes, d_scores, d_kpts = predict_image_rgb(det_model, dark_u8, device, conf_thresh=0.15)
            if len(d_boxes) == 0:
                dark_detected[i] = False
                dark_det5_aligned[i] = d_ref_112
            else:
                cx = (d_boxes[:, 0] + d_boxes[:, 2]) * 0.5
                cy = (d_boxes[:, 1] + d_boxes[:, 3]) * 0.5
                ref_cx = 0.5 * (ref_box[0] + ref_box[2])
                ref_cy = 0.5 * (ref_box[1] + ref_box[3])
                best_j = int(np.argmin(np.hypot(cx - ref_cx, cy - ref_cy) - d_scores * 25.0))
                dark_detected[i] = True
                dark_det5_aligned[i] = align_face_112(dark_u8, d_kpts[best_j], out_size=112)[0]
                ref_kpts_list.append(ref_kpt)
                det_kpts_list.append(d_kpts[best_j])
                det_valid_indices.append(i)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    nme_dark_vs_ref = (
        compute_nme_5pt(np.stack(det_kpts_list, axis=0), np.stack(ref_kpts_list, axis=0))
        if det_kpts_list
        else 0.0
    )

    idx_a = np.array([img_to_idx[pa] for pa, _, _ in pairs], dtype=np.int64)
    idx_b = np.array([img_to_idx[pb] for _, pb, _ in pairs], dtype=np.int64)
    is_same = np.array([s for _, _, s in pairs], dtype=bool)
    probe_det_ok = dark_detected[idx_b]

    return {
        "n_unique": n_u,
        "n_pairs": len(pairs),
        "clean_det_rate": round(1.0 - clean_misses / max(1, n_u), 4),
        "dark_det_rate": round(float(dark_detected.mean()), 4),
        "dark_5pt_nme_pct": round(nme_dark_vs_ref * 100.0, 3),
        "dark_face_mean_luma_255": round(float(np.mean(raw_cache_dark_luma)), 2),
        "idx_a": idx_a,
        "idx_b": idx_b,
        "is_same": is_same,
        "probe_det_ok": probe_det_ok,
        "clean_aligned": clean_aligned,
        "dark_ref5_aligned": dark_ref5_aligned,
        "dark_det5_aligned": dark_det5_aligned,
    }


def evaluate_model_on_closed_loop(
    model: torch.nn.Module,
    cl_cache: Dict[str, Any],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    clean_e, clean_n = embed_rgb_u8_batch(model, cl_cache["clean_aligned"], device, batch_size=batch_size)
    dref_e, dref_n = embed_rgb_u8_batch(model, cl_cache["dark_ref5_aligned"], device, batch_size=batch_size)
    ddet_e, ddet_n = embed_rgb_u8_batch(model, cl_cache["dark_det5_aligned"], device, batch_size=batch_size)

    ia = cl_cache["idx_a"]
    ib = cl_cache["idx_b"]
    same = cl_cache["is_same"]
    det_ok = cl_cache["probe_det_ok"]

    s_clean = (clean_e[ia] * clean_e[ib]).sum(axis=1)
    s_dark_ref = (clean_e[ia] * dref_e[ib]).sum(axis=1)
    s_dark_det = (clean_e[ia] * ddet_e[ib]).sum(axis=1)

    m_clean = _score_summary(clean_e[ia], clean_n[ia], clean_e[ib], clean_n[ib], same)
    m_dref = _score_summary(clean_e[ia], clean_n[ia], dref_e[ib], dref_n[ib], same)
    m_ddet_all = _score_summary(clean_e[ia], clean_n[ia], ddet_e[ib], ddet_n[ib], same)
    m_ddet_ok = _score_summary(
        clean_e[ia[det_ok]],
        clean_n[ia[det_ok]],
        ddet_e[ib[det_ok]],
        ddet_n[ib[det_ok]],
        same[det_ok],
    )

    gap_tar_1e4 = m_dref["tar_at_far"]["FAR=1e-04"]["TAR"] - m_ddet_all["tar_at_far"]["FAR=1e-04"]["TAR"]
    gap_tar_1e3 = m_dref["tar_at_far"]["FAR=1e-03"]["TAR"] - m_ddet_all["tar_at_far"]["FAR=1e-03"]["TAR"]

    clean_gen = s_clean[same]
    dark_gen = s_dark_det[same]
    dark_imp = s_dark_det[~same]

    return {
        "clean_det_rate": cl_cache["clean_det_rate"],
        "dark_det_rate": cl_cache["dark_det_rate"],
        "dark_5pt_nme_pct": cl_cache["dark_5pt_nme_pct"],
        "dark_face_mean_luma_255": cl_cache["dark_face_mean_luma_255"],
        "clean_clean": m_clean,
        "clean_to_dark_ref5pt": m_dref,
        "clean_to_dark_phase1det5pt": m_ddet_all,
        "clean_to_dark_phase1det5pt_detected_only": m_ddet_ok,
        "tar_1e4_gap_ref_minus_det": round(float(gap_tar_1e4), 5),
        "tar_1e3_gap_ref_minus_det": round(float(gap_tar_1e3), 5),
        "cosine_distributions": {
            "clean_clean_genuine": _cosine_distribution_stats(clean_gen),
            "clean_dark_genuine": _cosine_distribution_stats(dark_gen),
            "clean_dark_impostor": _cosine_distribution_stats(dark_imp),
            "recommended_fixed_operating_point": _recommend_fixed_threshold(clean_gen, dark_gen, dark_imp),
        },
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Phase 2 Recognition Offline Acceptance Report", ""]
    models = report["models"]
    model_names = list(models.keys())
    bins = list(next(iter(models.values()))["benchmarks"].keys())

    lines.append("## 1. 5 Standard Benchmarks (Clean-Clean & Clean -> Dark)")
    lines.append("")
    lines.append("| Benchmark | Model | Clean Flip Acc (%) | Clean Single Acc (%) | Dark-GT Acc (%) | Dark-GT TAR@1e-4 | Dark-Jit Acc (%) | Dark-Jit TAR@1e-4 | Dark-Jit TAR@1e-3 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for b in bins:
        for m in model_names:
            r = models[m]["benchmarks"][b]
            cf = r["clean_clean_flip_avg"]["acc_mean_pct"]
            cs = r["clean_clean"]["acc_mean_pct"]
            dg = r.get("clean_to_dark_gt", {})
            dj = r.get("clean_to_dark_jitter", {})
            lines.append(
                f"| {b} | {m} | {cf:.2f} | {cs:.2f} | "
                f"{dg.get('acc_mean_pct', 0.0):.2f} | {dg.get('tar_at_far', {}).get('FAR=1e-04', {}).get('TAR', 0.0) * 100:.2f}% | "
                f"{dj.get('acc_mean_pct', 0.0):.2f} | {dj.get('tar_at_far', {}).get('FAR=1e-04', {}).get('TAR', 0.0) * 100:.2f}% | "
                f"{dj.get('tar_at_far', {}).get('FAR=1e-03', {}).get('TAR', 0.0) * 100:.2f}% |"
            )

    if any("closed_loop_lfw_raw" in models[m] for m in model_names):
        lines.append("")
        lines.append("## 2. End-to-End Closed-Loop: Phase 1 Detector 5-pt on Degraded LFW Raw 250x250")
        lines.append("")
        lines.append("| Model | Dark Det Rate | 5pt NME (%) | Clean-Ref5pt Dark TAR@1e-4 | Phase1Det5pt Dark TAR@1e-4 | Gap (<1.0%) | Phase1Det5pt TAR@1e-3 | Fixed Cosine Thresh |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in model_names:
            cl = models[m].get("closed_loop_lfw_raw")
            if not cl:
                continue
            tar_ref = cl["clean_to_dark_ref5pt"]["tar_at_far"]["FAR=1e-04"]["TAR"] * 100.0
            tar_det = cl["clean_to_dark_phase1det5pt"]["tar_at_far"]["FAR=1e-04"]["TAR"] * 100.0
            tar_det_1e3 = cl["clean_to_dark_phase1det5pt"]["tar_at_far"]["FAR=1e-03"]["TAR"] * 100.0
            gap = cl["tar_1e4_gap_ref_minus_det"] * 100.0
            th = cl["cosine_distributions"]["recommended_fixed_operating_point"]["threshold"]
            lines.append(
                f"| {m} | {cl['dark_det_rate']*100:.2f}% | {cl['dark_5pt_nme_pct']:.2f}% | "
                f"{tar_ref:.2f}% | {tar_det:.2f}% | {gap:.2f}% | {tar_det_1e3:.2f}% | {th:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Offline Acceptance Evaluation")
    parser.add_argument("--eval-dir", type=Path, default=Path("datasets/rec_eval"))
    parser.add_argument("--det-ckpt", type=Path, default=Path("runs/detector_yolo11n_5pt_lcf/stage2_v2/best.pt"))
    parser.add_argument("--official-ckpt", type=Path, default=None)
    parser.add_argument("--phase2-ckpt", type=Path, default=None)
    parser.add_argument("--rec-arch", default="edgeface_s_gamma_05")
    parser.add_argument("--teacher-dir", type=Path, default=None)
    parser.add_argument("--eval-teacher", action="store_true")
    parser.add_argument("--bins", nargs="+", default=list(DEFAULT_BINS))
    parser.add_argument("--skip-closed-loop", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("reports/phase2_eval_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/phase2_eval_report.md"))
    args = parser.parse_args()

    seed_everything(42)
    device = get_device()
    degrader = PhysicalLowLightDegrader(mode="recognizer").to(device)

    bin_caches: Dict[str, Dict[str, Any]] = {}
    for idx, bname in enumerate(args.bins):
        bpath = args.eval_dir / f"{bname}.bin"
        if bpath.is_file():
            print(f"[eval_phase2] Caching protocols for {bpath}...")
            bin_caches[bname] = build_bin_protocol_cache(bpath, degrader, device, seed_offset=idx * 97)

    cl_cache = None
    lfw_raw_dir = args.eval_dir / "lfw_raw"
    if not args.skip_closed_loop and lfw_raw_dir.is_dir() and args.det_ckpt.is_file():
        print(f"[eval_phase2] Running Phase-1 detector closed-loop alignment cache on {lfw_raw_dir}...")
        cl_cache = prepare_lfw_raw_closed_loop_cache(lfw_raw_dir, args.det_ckpt, device)

    results: Dict[str, Any] = {}

    def _run_one(label: str, model: torch.nn.Module) -> None:
        print(f"[eval_phase2] Evaluating {label}...")
        model.to(device).eval()
        b_res = {bname: evaluate_model_on_bin_cache(model, cache, device) for bname, cache in bin_caches.items()}
        entry: Dict[str, Any] = {"benchmarks": b_res}
        if cl_cache is not None:
            entry["closed_loop_lfw_raw"] = evaluate_model_on_closed_loop(model, cl_cache, device)
        results[label] = entry

    if args.official_ckpt and args.official_ckpt.is_file():
        m_off = build_recognizer_student(args.rec_arch)
        if isinstance(m_off, OfficialEdgeFace):
            m_off.load_official_pretrained(args.official_ckpt)
        else:
            ck = torch.load(args.official_ckpt, map_location="cpu", weights_only=False)
            m_off.load_state_dict(ck.get("ema", ck.get("model", ck)))
        _run_one("official_edgeface_s", m_off)
        del m_off
        torch.cuda.empty_cache()

    if args.phase2_ckpt and args.phase2_ckpt.is_file():
        m_p2 = build_recognizer_student(args.rec_arch)
        ck = torch.load(args.phase2_ckpt, map_location="cpu", weights_only=False)
        m_p2.load_state_dict(ck.get("ema", ck.get("model", ck)))
        _run_one("phase2_best", m_p2)
        del m_p2
        torch.cuda.empty_cache()

    if args.eval_teacher and args.teacher_dir:
        m_t = CVLfaceTeacherWrapper(model_id_or_path=str(args.teacher_dir), offline_proxy=False)
        _run_one("cvlface_teacher", m_t)
        del m_t
        torch.cuda.empty_cache()

    report = {"benchmarks": list(bin_caches.keys()), "models": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")
    print(f"[eval_phase2] Wrote {args.output_json} and {args.output_md}")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
