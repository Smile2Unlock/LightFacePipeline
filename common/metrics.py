"""Unified evaluation metrics for Detection (DarkFace score.py parity), 5-pt NME, Recognition, and ISO 30107-3 Liveness."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw - 1.0, bx + bw - 1.0)
    iy2 = min(ay + ah - 1.0, by + bh - 1.0)
    inter = max(0.0, ix2 - ix1 + 1.0) * max(0.0, iy2 - iy1 + 1.0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _voc_ap(rec: Sequence[float], prec: Sequence[float]) -> float:
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]))


def _bucket(v: float) -> str:
    if v < 24:
        return "<24"
    if v < 40:
        return "24-40"
    if v < 64:
        return "40-64"
    if v < 96:
        return "64-96"
    return ">=96"


def evaluate_darkface_rows(rows: List[Dict[str, Any]], pred_key: str = "pred") -> Dict[str, Any]:
    """Exact Python port of Smile2Unlock_v2/build/research/darkface/score.py conventions.

    Each row contains:
      - `gt`: list of `[x, y, w, h, 1]`
      - `<pred_key>`: list of `[x, y, w, h, score]`
      - optional `face_luma`, `mean_luma`
    """
    # Official-style AP50 (1,000 normalized thresholds, per-image max-IoU assignment)
    all_scores = [float(p[4]) for r in rows for p in r.get(pred_key, [])]
    if not all_scores:
        ap50_official = 0.0
    else:
        lo, hi = min(all_scores), max(all_scores)
        denom = hi - lo
        normalized = {
            ri: sorted([(float(p[4]) - lo) / denom if denom else 1.0 for p in r.get(pred_key, [])], reverse=True)
            for ri, r in enumerate(rows)
        }
        image_recall: Dict[int, List[int]] = {}
        for ri, r in enumerate(rows):
            gt = r["gt"]
            preds = sorted(r.get(pred_key, []), key=lambda p: p[4], reverse=True)
            if not gt or not preds:
                continue
            found = [False] * len(gt)
            rr: List[int] = []
            for p in preds:
                best_i = max(range(len(gt)), key=lambda i: _iou_xywh(gt[i], p))
                if _iou_xywh(gt[best_i], p) >= 0.5:
                    found[best_i] = True
                rr.append(sum(found))
            image_recall[ri] = rr

        n_gt = sum(len(r["gt"]) for r in rows)
        rec_o: List[float] = []
        prec_o: List[float] = []
        for t in range(1, 1001):
            th = 1.0 - t / 1000.0
            total_props = total_rec = 0
            for ri in range(len(rows)):
                if ri not in image_recall:
                    continue
                scores = normalized[ri]
                ix = next((i for i, s in enumerate(scores) if s < th), len(scores))
                if ix:
                    total_props += ix
                    total_rec += image_recall[ri][ix - 1]
            rec_o.append(total_rec / n_gt if n_gt else 0.0)
            prec_o.append(total_rec / total_props if total_props else 0.0)
        ap50_official = _voc_ap(rec_o, prec_o)

    # Greedy one-to-one VOC AP50
    n_gt_total = sum(len(r["gt"]) for r in rows)
    ds: List[Tuple[float, int, Sequence[float], set]] = []
    used_per_img = [set() for _ in rows]
    for ri, r in enumerate(rows):
        for p in r.get(pred_key, []):
            ds.append((float(p[4]), ri, p, used_per_img[ri]))
    ds.sort(key=lambda x: x[0], reverse=True)
    tps: List[int] = []
    fps: List[int] = []
    for _, ri, p, used in ds:
        gt = rows[ri]["gt"]
        best = -1
        best_iou = 0.5
        for gi, g in enumerate(gt):
            v = _iou_xywh(p, g)
            if gi not in used and v >= best_iou:
                best = gi
                best_iou = v
        if best >= 0:
            used.add(best)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)
    ctp = cfp = 0
    rec_g: List[float] = []
    prec_g: List[float] = []
    for tp, fp in zip(tps, fps):
        ctp += tp
        cfp += fp
        rec_g.append(ctp / n_gt_total if n_gt_total else 0.0)
        prec_g.append(ctp / (ctp + cfp))
    ap50_greedy = _voc_ap(rec_g, prec_g)

    # Summary stats
    total_gt = total_p = total_tp = zero = multi = single_ok = empty_fp = empty_imgs = 0
    face_groups = defaultdict(lambda: [0, 0])
    size_groups = defaultdict(lambda: [0, 0])
    frame_groups = defaultdict(lambda: [0, 0])

    for r in rows:
        gt = r["gt"]
        preds = sorted(r.get(pred_key, []), key=lambda x: x[4], reverse=True)
        used = set()
        tp = 0
        for p in preds:
            best = -1
            best_iou = 0.5
            for gi, g in enumerate(gt):
                v = _iou_xywh(p, g)
                if gi not in used and v >= best_iou:
                    best = gi
                    best_iou = v
            if best >= 0:
                used.add(best)
                tp += 1

        ng = len(gt)
        np_ = len(preds)
        total_gt += ng
        total_p += np_
        total_tp += tp
        zero += int(np_ == 0)
        multi += int(np_ > 1)
        single_ok += int(ng == 1 and np_ == 1 and tp == 1)
        if ng == 0:
            empty_imgs += 1
            empty_fp += np_

        if "face_luma" in r:
            for gi, luma in enumerate(r["face_luma"]):
                k = _bucket(luma)
                face_groups[k][0] += 1
                face_groups[k][1] += int(gi in used)
        for gi, g in enumerate(gt):
            k = _bucket(g[3])
            size_groups[k][0] += 1
            size_groups[k][1] += int(gi in used)
        if "mean_luma" in r:
            fk = _bucket(r["mean_luma"])
            frame_groups[fk][0] += ng
            frame_groups[fk][1] += tp

    return {
        "detections": pred_key,
        "images": len(rows),
        "gt_faces": total_gt,
        "pred_boxes": total_p,
        "matched_iou50": total_tp,
        "recall_iou50": total_tp / total_gt if total_gt else 0.0,
        "precision_iou50": total_tp / total_p if total_p else 0.0,
        "AP50_VOC_greedy": ap50_greedy,
        "AP50_official_tools_style": ap50_official,
        "zero_detection_images": zero,
        "multi_detection_images": multi,
        "exactly_one_face_images": single_ok,
        "empty_frame_fppi": empty_fp / empty_imgs if empty_imgs else 0.0,
        "face_luma_groups": {
            k: {"faces": v[0], "matched": v[1], "recall": v[1] / v[0] if v[0] else None}
            for k, v in face_groups.items()
        },
        "face_height_groups": {
            k: {"faces": v[0], "matched": v[1], "recall": v[1] / v[0] if v[0] else None}
            for k, v in size_groups.items()
        },
        "frame_luma_groups": {
            k: {"gt_faces": v[0], "matched": v[1], "recall": v[1] / v[0] if v[0] else None}
            for k, v in frame_groups.items()
        },
    }


def compute_nme_5pt(pred_kpts: np.ndarray, gt_kpts: np.ndarray) -> float:
    """Inter-ocular normalized RMSE (NME) for `[N, 5, 2]` predicted & GT landmarks."""
    pred = np.asarray(pred_kpts, dtype=np.float64)
    gt = np.asarray(gt_kpts, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (5, 2):
        raise ValueError(f"Expected [N, 5, 2] arrays, got {pred.shape} and {gt.shape}")
    iod = np.linalg.norm(gt[:, 0] - gt[:, 1], axis=1).clip(min=1.0)
    pt_err = np.linalg.norm(pred - gt, axis=2).mean(axis=1)
    return float((pt_err / iod).mean())


def compute_tar_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    far_targets: Sequence[float] = (1e-4, 1e-3, 1e-2),
) -> Dict[str, Any]:
    """Compute TAR@FAR and 1:1 cosine similarity summary statistics."""
    g = np.asarray(genuine_scores, dtype=np.float64)
    imp = np.asarray(impostor_scores, dtype=np.float64)
    y_true = np.concatenate([np.ones(len(g)), np.zeros(len(imp))])
    y_score = np.concatenate([g, imp])
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    tar_dict: Dict[str, Dict[str, float]] = {}
    for target in far_targets:
        idx = int(np.searchsorted(fpr, target, side="right")) - 1
        idx = max(0, min(idx, len(tpr) - 1))
        tar_dict[f"FAR={target:.0e}"] = {
            "TAR": float(tpr[idx]),
            "threshold": float(thresholds[idx]) if np.isfinite(thresholds[idx]) else 1.0,
        }

    return {
        "auc": auc,
        "tar_at_far": tar_dict,
        "genuine_mean": float(g.mean()) if len(g) else 0.0,
        "genuine_p05": float(np.percentile(g, 5)) if len(g) else 0.0,
        "impostor_mean": float(imp.mean()) if len(imp) else 0.0,
        "impostor_p95": float(np.percentile(imp, 95)) if len(imp) else 0.0,
    }


def compute_iso30107_metrics(
    live_probs: np.ndarray,
    is_live: np.ndarray,
    threshold: Optional[float] = None,
    apcer_targets: Sequence[float] = (0.01, 0.02),
) -> Dict[str, Any]:
    """Compute ISO/IEC 30107-3 APCER, BPCER, ACER, HTER, AUC, and operating points."""
    probs = np.asarray(live_probs, dtype=np.float64)
    labels = np.asarray(is_live, dtype=np.int64)
    live_mask = labels == 1
    spoof_mask = labels == 0

    auc = float(roc_auc_score(labels, probs)) if (live_mask.any() and spoof_mask.any()) else 0.0
    live_vals = probs[live_mask]
    spoof_vals = probs[spoof_mask]

    def _eval_at(th: float) -> Dict[str, float]:
        apcer = float((spoof_vals >= th).mean()) if len(spoof_vals) else 0.0
        bpcer = float((live_vals < th).mean()) if len(live_vals) else 0.0
        acer = 0.5 * (apcer + bpcer)
        return {"threshold": float(th), "APCER": apcer, "BPCER": bpcer, "ACER": acer, "HTER": acer}

    if threshold is None:
        # Find threshold minimizing ACER
        cand = np.unique(np.concatenate([probs, [0.5]]))
        best = min((_eval_at(float(t)) for t in cand), key=lambda d: (d["ACER"], d["APCER"]))
    else:
        best = _eval_at(threshold)

    fixed_points: Dict[str, Dict[str, float]] = {}
    for target in apcer_targets:
        if len(spoof_vals) == 0:
            th = 0.5
        else:
            # Threshold at (1 - target) quantile of spoof live_probs so APCER <= target
            th = float(np.quantile(spoof_vals, min(1.0, max(0.0, 1.0 - target))) + 1e-9)
        fixed_points[f"APCER<={target*100:.1f}%"] = _eval_at(th)

    return {
        "auc": auc,
        "operating_point": best,
        "fixed_apcer_points": fixed_points,
        "n_live": int(live_mask.sum()),
        "n_spoof": int(spoof_mask.sum()),
    }


def audit_luma_anti_shortcut(
    live_clips_rgb01: np.ndarray,
    spoof_clips_rgb01: np.ndarray,
) -> Dict[str, Any]:
    """Verify Live and Spoof training batches overlap in face luma & contrast (anti-shortcut check)."""
    def _stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # arr: [N, ..., 3, H, W] in [0, 1]
        y = (0.299 * arr[..., 0, :, :] + 0.587 * arr[..., 1, :, :] + 0.114 * arr[..., 2, :, :]) * 255.0
        axes = tuple(range(1, y.ndim))
        return y.mean(axis=axes), y.std(axis=axes)

    live_mean, live_std = _stats(live_clips_rgb01)
    spoof_mean, spoof_std = _stats(spoof_clips_rgb01)
    mean_gap = abs(float(live_mean.mean() - spoof_mean.mean()))
    std_gap = abs(float(live_std.mean() - spoof_std.mean()))

    return {
        "live_luma_mean": float(live_mean.mean()),
        "live_luma_p10_p90": [float(np.percentile(live_mean, 10)), float(np.percentile(live_mean, 90))],
        "spoof_luma_mean": float(spoof_mean.mean()),
        "spoof_luma_p10_p90": [float(np.percentile(spoof_mean, 10)), float(np.percentile(spoof_mean, 90))],
        "live_std_mean": float(live_std.mean()),
        "spoof_std_mean": float(spoof_std.mean()),
        "mean_luma_gap": mean_gap,
        "std_luma_gap": std_gap,
        "warning_shortcut_risk": bool(mean_gap > 25.0),
    }
