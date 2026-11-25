"""Phase 1 Offline Acceptance Evaluator: WIDER E/M/H, WFLW Clean & Dark NME, DarkFace 1k Val, COCO Dark FPPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from common.align import composite_tight_face_on_canvas, letterbox_rgb, unletterbox_coords
from common.degradations import PhysicalLowLightDegrader
from common.metrics import compute_nme_5pt, evaluate_darkface_rows
from common.nms import box_iou, nms
from detector.dataset import parse_darkface_label_txt
from detector.eval_darkface import evaluate_darkface
from detector.yolo11_face import LCFYOLO11Face5pt
from detector.yunet_lcf import LCFYuNetS


def load_detector_checkpoint(
    arch: str,
    ckpt_path: Path,
    device: torch.device,
    use_lcf: Optional[bool] = None,
    reg_max: int = 12,
    width_mult: float = 0.22,
) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    has_lcf_weights = any(k.startswith("lcf.") for k in state.keys())
    if use_lcf is None:
        use_lcf = has_lcf_weights
    if arch == "lcf_yolo11n_5pt":
        model = LCFYOLO11Face5pt(use_lcf=use_lcf, use_p2=True, width_mult=width_mult, reg_max=reg_max)
    elif arch == "lcf_yunet_s":
        model = LCFYuNetS(use_lcf=use_lcf, width_scale=1.5, reg_max=8)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_image_rgb(
    model: torch.nn.Module,
    rgb: np.ndarray,
    device: torch.device,
    img_size: int = 640,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run single RGB image inference -> `(boxes_xyxy [N,4], scores [N], kpts5 [N,5,2])` in original coords."""
    oh, ow = rgb.shape[:2]
    lb, scale, pad = letterbox_rgb(rgb, (img_size, img_size))
    inp = torch.from_numpy(lb.transpose(2, 0, 1)).unsqueeze(0).to(device).float().div_(255.0)
    return predict_tensor_01(model, inp, (oh, ow), scale, pad, conf_thresh=conf_thresh, iou_thresh=iou_thresh)


@torch.no_grad()
def predict_tensor_01(
    model: torch.nn.Module,
    inp_01: torch.Tensor,
    orig_hw: Tuple[int, int],
    scale: float,
    pad: Tuple[float, float],
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = model(inp_01)
    scores = out["scores"][0]
    boxes = out["boxes_xyxy"][0]
    kpts = out["kpts"][0, :, :, :2]
    mask = scores >= conf_thresh
    if not mask.any():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 5, 2), dtype=np.float32),
        )
    b_sel = boxes[mask]
    s_sel = scores[mask]
    k_sel = kpts[mask]
    if s_sel.numel() > 2000:
        top_idx = torch.topk(s_sel, k=2000).indices
        b_sel = b_sel[top_idx]
        s_sel = s_sel[top_idx]
        k_sel = k_sel[top_idx]
    keep = nms(b_sel, s_sel, iou_threshold=iou_thresh)[:300]
    b_orig, k_orig = unletterbox_coords(
        b_sel[keep].cpu().numpy(),
        k_sel[keep].cpu().numpy(),
        scale,
        pad,
        orig_hw,
    )
    assert b_orig is not None and k_orig is not None
    return b_orig, s_sel[keep].cpu().numpy(), k_orig


def _voc_ap_from_lists(recs: List[float], precs: List[float]) -> float:
    mrec = [0.0] + recs + [1.0]
    mpre = [0.0] + precs + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]))


@torch.no_grad()
def evaluate_wider_val(
    model: torch.nn.Module,
    wider_val_jsonl: Path,
    device: torch.device,
    img_size: int = 640,
    conf_thresh: float = 0.05,
    op_thresh: float = 0.35,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate WIDER FACE Val (Easy / Medium / Hard AP50 + Multi-face recall at operating threshold)."""
    rows = []
    with open(wider_val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if max_images is not None:
        rows = rows[:max_images]

    subset_dets: Dict[str, List[Tuple[float, int, np.ndarray]]] = {"all": [], "easy": [], "medium": [], "hard": []}
    subset_gts: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {"all": {}, "easy": {}, "medium": {}, "hard": {}}
    subset_n_pos: Dict[str, int] = {"all": 0, "easy": 0, "medium": 0, "hard": 0}

    multi_gt_imgs = 0
    multi_pred_ok = 0

    for img_idx, row in enumerate(rows):
        bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt_boxes = np.asarray(row["boxes"], dtype=np.float32).reshape(-1, 4)
        pred_boxes, pred_scores, _ = predict_image_rgb(model, rgb, device, img_size=img_size, conf_thresh=conf_thresh)

        # Multi-face gate recall at operating threshold
        if len(gt_boxes) >= 2:
            multi_gt_imgs += 1
            if int((pred_scores >= op_thresh).sum()) >= 2:
                multi_pred_ok += 1

        # Store predictions & per-difficulty keep/ignore flags (if available in manifest)
        for subset in ("all", "easy", "medium", "hard"):
            keep_key = f"keep_{subset}"
            if subset == "all" or keep_key not in row:
                keep_mask = np.ones((len(gt_boxes),), dtype=bool)
            else:
                keep_mask = np.asarray(row[keep_key], dtype=bool)
            subset_gts[subset][img_idx] = (gt_boxes, keep_mask)
            subset_n_pos[subset] += int(keep_mask.sum())
            for pb, ps in zip(pred_boxes, pred_scores):
                subset_dets[subset].append((float(ps), img_idx, pb))

    ap_out: Dict[str, float] = {}
    for subset in ("easy", "medium", "hard", "all"):
        dets = sorted(subset_dets[subset], key=lambda x: -x[0])
        n_pos = max(subset_n_pos[subset], 1)
        used_per_img: Dict[int, set] = {}
        tps: List[int] = []
        fps: List[int] = []
        for _, img_idx, pb in dets:
            gboxes, keep_mask = subset_gts[subset][img_idx]
            if len(gboxes) == 0:
                tps.append(0)
                fps.append(1)
                continue
            ious = box_iou(torch.from_numpy(pb[None]), torch.from_numpy(gboxes)).squeeze(0).numpy()
            best_gi = int(np.argmax(ious))
            best_iou = float(ious[best_gi])
            if best_iou >= 0.5:
                if not keep_mask[best_gi]:
                    # Standard WIDER FACE eval ignores matched invalid/hard-ignored boxes
                    continue
                used = used_per_img.setdefault(img_idx, set())
                if best_gi not in used:
                    used.add(best_gi)
                    tps.append(1)
                    fps.append(0)
                else:
                    tps.append(0)
                    fps.append(1)
            else:
                tps.append(0)
                fps.append(1)
        ctp = cfp = 0
        recs: List[float] = []
        precs: List[float] = []
        for tp, fp in zip(tps, fps):
            ctp += tp
            cfp += fp
            recs.append(ctp / n_pos)
            precs.append(ctp / max(ctp + cfp, 1))
        ap_out[f"AP50_{subset}"] = round(_voc_ap_from_lists(recs, precs), 5)

    return {
        "images": len(rows),
        **ap_out,
        "multi_face_gt_images": multi_gt_imgs,
        "multi_face_ge2_detected_rate": round(multi_pred_ok / max(multi_gt_imgs, 1), 4),
    }


@torch.no_grad()
def evaluate_wflw_5pt_nme(
    model: torch.nn.Module,
    wflw_test_jsonl: Path,
    device: torch.device,
    img_size: int = 640,
    conf_thresh: float = 0.05,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate 5-pt NME on WFLW Test (Clean vs Deterministic Physical Dark/Backlight)."""
    rows = []
    with open(wflw_test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if max_images is not None:
        rows = rows[:max_images]

    degrader = PhysicalLowLightDegrader(mode="detector").to(device)
    clean_pred_kpts: List[np.ndarray] = []
    clean_gt_kpts: List[np.ndarray] = []
    dark_pred_kpts: List[np.ndarray] = []
    dark_gt_kpts: List[np.ndarray] = []
    clean_det_hits = 0
    dark_det_hits = 0

    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        for row_idx, row in enumerate(rows):
            bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt_box = np.asarray(row["boxes"][0], dtype=np.float32)
            gt_kpt = np.asarray(row["kpts"][0], dtype=np.float32)

            if max(rgb.shape[:2]) <= 160:
                bg_rgb = None
                if row.get("bg_image_path"):
                    bg_bgr = cv2.imread(row["bg_image_path"], cv2.IMREAD_COLOR)
                    if bg_bgr is not None:
                        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
                rgb, boxes_c, kpts_c = composite_tight_face_on_canvas(
                    rgb,
                    gt_box[None],
                    gt_kpt[None],
                    bg_rgb=bg_rgb,
                    canvas_size=480,
                    target_patch_size=224,
                    center_xy=(240, 240),
                    rng=np.random.default_rng(42 + row_idx),
                )
                gt_box = boxes_c[0]
                gt_kpt = kpts_c[0]

            oh, ow = rgb.shape[:2]

            lb, scale, pad = letterbox_rgb(rgb, (img_size, img_size))
            inp_clean = torch.from_numpy(lb.transpose(2, 0, 1)).unsqueeze(0).to(device).float().div_(255.0)
            bb_norm = torch.tensor(
                [
                    [
                        (gt_box[0] * scale + pad[0]) / img_size,
                        (gt_box[1] * scale + pad[1]) / img_size,
                        (gt_box[2] * scale + pad[0]) / img_size,
                        (gt_box[3] * scale + pad[1]) / img_size,
                    ]
                ],
                device=device,
                dtype=torch.float32,
            )
            inp_dark = degrader(inp_clean, bboxes_norm=bb_norm)

            for inp_t, k_pred_list, k_gt_list, is_dark in (
                (inp_clean, clean_pred_kpts, clean_gt_kpts, False),
                (inp_dark, dark_pred_kpts, dark_gt_kpts, True),
            ):
                p_boxes, p_scores, p_kpts = predict_tensor_01(
                    model, inp_t, (oh, ow), scale, pad, conf_thresh=conf_thresh
                )
                if len(p_boxes) == 0:
                    continue
                ious = box_iou(torch.from_numpy(p_boxes), torch.from_numpy(gt_box[None])).squeeze(1).numpy()
                best_i = int(np.argmax(ious))
                if ious[best_i] >= 0.3:
                    k_pred_list.append(p_kpts[best_i])
                    k_gt_list.append(gt_kpt)
                    if is_dark:
                        dark_det_hits += 1
                    else:
                        clean_det_hits += 1
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    def _median_nme_5pt(pred_list: List[np.ndarray], gt_list: List[np.ndarray]) -> float:
        if not pred_list:
            return 1.0
        p = np.stack(pred_list).astype(np.float64)
        g = np.stack(gt_list).astype(np.float64)
        iod = np.linalg.norm(g[:, 0] - g[:, 1], axis=1).clip(min=1.0)
        pt_err = np.linalg.norm(p - g, axis=2).mean(axis=1)
        return float(np.median(pt_err / iod))

    clean_nme = compute_nme_5pt(np.stack(clean_pred_kpts), np.stack(clean_gt_kpts)) if clean_pred_kpts else 1.0
    dark_nme = compute_nme_5pt(np.stack(dark_pred_kpts), np.stack(dark_gt_kpts)) if dark_pred_kpts else 1.0
    clean_nme_med = _median_nme_5pt(clean_pred_kpts, clean_gt_kpts)
    dark_nme_med = _median_nme_5pt(dark_pred_kpts, dark_gt_kpts)
    return {
        "total_images": len(rows),
        "clean_matched": clean_det_hits,
        "clean_recall": round(clean_det_hits / max(len(rows), 1), 4),
        "clean_nme_5pt": round(clean_nme, 5),
        "clean_nme_median": round(clean_nme_med, 5),
        "dark_matched": dark_det_hits,
        "dark_recall": round(dark_det_hits / max(len(rows), 1), 4),
        "dark_nme_5pt": round(dark_nme, 5),
        "dark_nme_median": round(dark_nme_med, 5),
    }


@torch.no_grad()
def evaluate_coco_dark_fppi(
    model: torch.nn.Module,
    hard_neg_val_jsonl: Path,
    device: torch.device,
    img_size: int = 640,
    op_thresh: float = 0.35,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate False Positives Per Image (FPPI) on degraded person-free COCO frames."""
    rows = []
    with open(hard_neg_val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if max_images is not None:
        rows = rows[:max_images]

    degrader = PhysicalLowLightDegrader(mode="detector").to(device)
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    total_fps = 0
    imgs_with_fp = 0
    try:
        for row in rows:
            bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            oh, ow = rgb.shape[:2]
            lb, scale, pad = letterbox_rgb(rgb, (img_size, img_size))
            inp = torch.from_numpy(lb.transpose(2, 0, 1)).unsqueeze(0).to(device).float().div_(255.0)
            inp_dark = degrader(inp, bboxes_norm=None)
            p_boxes, _, _ = predict_tensor_01(model, inp_dark, (oh, ow), scale, pad, conf_thresh=op_thresh)
            n_fp = len(p_boxes)
            total_fps += n_fp
            if n_fp > 0:
                imgs_with_fp += 1
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    return {
        "images": len(rows),
        "op_thresh": op_thresh,
        "total_false_positives": total_fps,
        "images_with_fp": imgs_with_fp,
        "fppi": round(total_fps / max(len(rows), 1), 5),
    }


def evaluate_all_phase1(
    ckpt_path: Path,
    arch: str,
    darkface_dir: Path,
    darkface_split: Path,
    wider_val_jsonl: Path,
    wflw_test_jsonl: Path,
    hard_neg_val_jsonl: Path,
    device: torch.device,
    op_thresh: float = 0.45,
    fast_limit: Optional[int] = None,
) -> Dict[str, Any]:
    model = load_detector_checkpoint(arch, ckpt_path, device=device)
    df_res = evaluate_darkface(
        model,
        darkface_dir,
        darkface_split,
        device=device,
        conf_thresh=0.05,
        split="val",
        jsonl_out=None,
    )
    wider_res = evaluate_wider_val(model, wider_val_jsonl, device=device, op_thresh=op_thresh, max_images=fast_limit)
    wflw_res = evaluate_wflw_5pt_nme(model, wflw_test_jsonl, device=device, max_images=fast_limit)
    coco_res = evaluate_coco_dark_fppi(
        model, hard_neg_val_jsonl, device=device, op_thresh=op_thresh, max_images=fast_limit
    )
    coco_035_res = evaluate_coco_dark_fppi(
        model, hard_neg_val_jsonl, device=device, op_thresh=0.35, max_images=fast_limit
    )
    return {
        "checkpoint": str(ckpt_path),
        "arch": arch,
        "darkface_val_1000": df_res,
        "wider_val": wider_res,
        "wflw_test_5pt": wflw_res,
        "coco_dark_fppi": coco_res,
        "coco_dark_fppi_035": coco_035_res,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive Phase 1 Acceptance Evaluation")
    parser.add_argument("--arch", default="lcf_yolo11n_5pt", choices=["lcf_yolo11n_5pt", "lcf_yunet_s"])
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs or checkpoint paths")
    parser.add_argument("--darkface-dir", type=Path, default=Path("/workspace/LightFacePipeline/datasets/darkface/data"))
    parser.add_argument("--darkface-split", type=Path, default=Path("splits/darkface_split_5000_1000.json"))
    parser.add_argument("--wider-val", type=Path, default=Path("/workspace/LightFacePipeline/datasets/manifests/wider_val.jsonl"))
    parser.add_argument("--wflw-test", type=Path, default=Path("/workspace/LightFacePipeline/datasets/manifests/wflw_test_5pt.jsonl"))
    parser.add_argument("--hard-neg-val", type=Path, default=Path("/workspace/LightFacePipeline/datasets/manifests/coco_hard_neg_val.jsonl"))
    parser.add_argument("--op-thresh", type=float, default=0.45)
    parser.add_argument("--out-json", type=Path, default=Path("reports/phase1_eval_report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/phase1_eval_report.md"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports: Dict[str, Any] = {}
    for item in args.checkpoints:
        if "=" in item:
            name, path_s = item.split("=", 1)
        else:
            path_s = item
            name = Path(path_s).stem
        reports[name] = evaluate_all_phase1(
            ckpt_path=Path(path_s),
            arch=args.arch,
            darkface_dir=args.darkface_dir,
            darkface_split=args.darkface_split,
            wider_val_jsonl=args.wider_val,
            wflw_test_jsonl=args.wflw_test,
            hard_neg_val_jsonl=args.hard_neg_val,
            device=device,
            op_thresh=args.op_thresh,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Phase 1 Acceptance Evaluation Report",
        "",
        "| Variant | WIDER Easy | WIDER Med | WIDER Hard | WFLW Clean NME | WFLW Dark NME | DarkFace AP50 (Official-style) | DarkFace AP50 (Greedy) | DarkFace Recall | COCO Dark FPPI | Multi-face >=2 Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| SeetaFace `base640` (ref, 6k) | — | — | — | — | — | 0.145% | — | 0.147% | — | — |",
        "| SeetaFace `app640` (ref, 6k) | — | — | — | — | — | 0.614% | — | 0.645% | — | — |",
    ]
    for name, rep in reports.items():
        w = rep["wider_val"]
        wf = rep["wflw_test_5pt"]
        df = rep["darkface_val_1000"]
        co = rep["coco_dark_fppi"]
        lines.append(
            f"| **{name}** | {w['AP50_easy']*100:.2f}% | {w['AP50_medium']*100:.2f}% | {w['AP50_hard']*100:.2f}% | "
            f"{wf['clean_nme_5pt']*100:.2f}% | {wf['dark_nme_5pt']*100:.2f}% | "
            f"{df['AP50_official_tools_style']*100:.2f}% | {df['AP50_VOC_greedy']*100:.2f}% | "
            f"{df['recall_iou50']*100:.2f}% | {co['fppi']:.4f} | {w['multi_face_ge2_detected_rate']*100:.1f}% |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
