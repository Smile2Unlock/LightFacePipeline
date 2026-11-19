"""Evaluate Phase 1 Detector on fixed DarkFace 1,000 Val (score.py-compatible JSONL + metrics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from common.align import letterbox_rgb, unletterbox_coords
from common.metrics import evaluate_darkface_rows
from common.nms import nms
from detector.dataset import parse_darkface_label_txt
from detector.yolo11_face import LCFYOLO11Face5pt
from detector.yunet_lcf import LCFYuNetS


@torch.no_grad()
def evaluate_darkface(
    model: torch.nn.Module,
    data_dir: Path,
    split_json: Path,
    device: torch.device,
    img_size: int = 640,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.45,
    split: str = "val",
    jsonl_out: Optional[Path] = None,
) -> Dict[str, Any]:
    model.eval()
    manifest = json.loads(split_json.read_text(encoding="utf-8"))
    files: List[str] = manifest[split]
    rows: List[Dict[str, Any]] = []

    for fname in files:
        stem = Path(fname).stem
        bgr = cv2.imread(str(data_dir / "image" / fname), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        oh, ow = rgb.shape[:2]
        gt_xyxy = parse_darkface_label_txt(data_dir / "label" / f"{stem}.txt")
        gt_xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1]), 1] for b in gt_xyxy]

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mean_luma = float(gray.mean())
        face_lumas = []
        for x1, y1, x2, y2 in gt_xyxy:
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(ow, int(x2)), min(oh, int(y2))
            crop = gray[iy1:iy2, ix1:ix2]
            face_lumas.append(float(crop.mean()) if crop.size else mean_luma)

        lb, scale, pad = letterbox_rgb(rgb, (img_size, img_size))
        inp = torch.from_numpy(lb.transpose(2, 0, 1)).unsqueeze(0).to(device).float().div_(255.0)
        out = model(inp)
        scores = out["scores"][0]
        boxes = out["boxes_xyxy"][0]
        mask = scores >= conf_thresh
        preds_xywh: List[List[float]] = []
        if mask.any():
            b_sel = boxes[mask]
            s_sel = scores[mask]
            keep = nms(b_sel, s_sel, iou_threshold=iou_thresh)
            b_np = b_sel[keep].cpu().numpy()
            s_np = s_sel[keep].cpu().numpy()
            b_orig, _ = unletterbox_coords(b_np, None, scale, pad, (oh, ow))
            assert b_orig is not None
            for box, sc in zip(b_orig, s_np):
                preds_xywh.append([float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1]), float(sc)])

        rows.append(
            {
                "kind": "image",
                "image": fname,
                "gt": gt_xywh,
                "raw_native": preds_xywh,
                "mean_luma": mean_luma,
                "face_luma": face_lumas,
            }
        )

    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return evaluate_darkface_rows(rows, pred_key="raw_native")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Detector on DarkFace 1,000 Val")
    parser.add_argument("--arch", default="lcf_yolo11n_5pt", choices=["lcf_yolo11n_5pt", "lcf_yunet_s"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=Path("splits/darkface_split_5000_1000.json"))
    parser.add_argument("--jsonl-out", type=Path, default=Path("reports/darkface_val_results.jsonl"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LCFYOLO11Face5pt() if args.arch == "lcf_yolo11n_5pt" else LCFYuNetS()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model", ckpt)))
    model.to(device)

    metrics = evaluate_darkface(model, args.data_dir, args.split_json, device, jsonl_out=args.jsonl_out)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
