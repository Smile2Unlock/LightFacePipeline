#!/usr/bin/env python3
"""Build verified Phase 1 JSONL manifests for WIDER FACE, WFLW, COCO person-free negatives, and ExDark."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def build_wider_train_manifest(wider_root: Path, label_txt: Path, out_jsonl: Path) -> Dict[str, int]:
    img_root = wider_root / "WIDER_train" / "images"
    if not img_root.is_dir():
        img_root = wider_root / "train" / "images"
    if not img_root.is_dir():
        raise FileNotFoundError(f"Could not find WIDER train images under {wider_root}")

    rows: List[Dict[str, Any]] = []
    cur_img: Optional[str] = None
    cur_boxes: List[List[float]] = []
    cur_kpts: List[List[List[float]]] = []
    cur_kvalid: List[bool] = []

    def _flush() -> None:
        if cur_img is None or len(cur_boxes) == 0:
            return
        ip = img_root / cur_img
        if ip.is_file():
            rows.append(
                {
                    "image_path": str(ip),
                    "boxes": cur_boxes,
                    "kpts": cur_kpts,
                    "kpts_valid": cur_kvalid,
                }
            )

    with open(label_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                _flush()
                cur_img = s[1:].strip()
                cur_boxes = []
                cur_kpts = []
                cur_kvalid = []
                continue
            parts = [float(v) for v in s.split()]
            if len(parts) < 4:
                continue
            x, y, w, h = parts[0], parts[1], parts[2], parts[3]
            if w <= 1.0 or h <= 1.0:
                continue
            cur_boxes.append([x, y, x + w, y + h])
            if len(parts) >= 19:
                pts = [
                    [parts[4], parts[5]],
                    [parts[7], parts[8]],
                    [parts[10], parts[11]],
                    [parts[13], parts[14]],
                    [parts[16], parts[17]],
                ]
                is_valid = all(p[0] >= 0.0 and p[1] >= 0.0 for p in pts)
                cur_kpts.append(pts if is_valid else [[-1.0, -1.0]] * 5)
                cur_kvalid.append(is_valid)
            else:
                cur_kpts.append([[-1.0, -1.0]] * 5)
                cur_kvalid.append(False)
    _flush()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for r in rows:
            out_f.write(json.dumps(r) + "\n")

    n_faces = sum(len(r["boxes"]) for r in rows)
    n_valid_kpts = sum(sum(1 for kv in r["kpts_valid"] if kv) for r in rows)
    return {"images": len(rows), "faces": n_faces, "valid_kpts": n_valid_kpts}


def build_wider_val_manifest(wider_root: Path, split_dir: Path, out_jsonl: Path) -> Dict[str, int]:
    img_root = wider_root / "WIDER_val" / "images"
    if not img_root.is_dir():
        img_root = wider_root / "val" / "images"
    gt_txt = split_dir / "wider_face_val_bbx_gt.txt"
    if not img_root.is_dir() or not gt_txt.is_file():
        raise FileNotFoundError(f"Missing WIDER val images ({img_root}) or gt ({gt_txt})")

    lines = gt_txt.read_text(encoding="utf-8").splitlines()
    i = 0
    rows: List[Dict[str, Any]] = []
    easy_cnt = med_cnt = hard_cnt = 0

    while i < len(lines):
        rel = lines[i].strip()
        i += 1
        if not rel:
            continue
        n = int(lines[i].strip())
        i += 1
        boxes: List[List[float]] = []
        keep_easy: List[bool] = []
        keep_med: List[bool] = []
        keep_hard: List[bool] = []

        if n == 0:
            i += 1  # WIDER val format puts one dummy zero line when n==0
        else:
            for _ in range(n):
                vals = [int(v) for v in lines[i].strip().split()]
                i += 1
                x1, y1, w, h, blur, expr, illum, invalid, occ, pose = vals[:10]
                if w <= 0 or h <= 0:
                    continue
                boxes.append([float(x1), float(y1), float(x1 + w), float(y1 + h)])
                is_hard = invalid == 0
                is_med = is_hard and (h >= 12) and (occ <= 1) and (blur <= 1)
                is_easy = is_med and (h >= 20) and (blur == 0) and (occ == 0) and (pose == 0) and (illum == 0)
                keep_easy.append(is_easy)
                keep_med.append(is_med)
                keep_hard.append(is_hard)
                easy_cnt += int(is_easy)
                med_cnt += int(is_med)
                hard_cnt += int(is_hard)

        ip = img_root / rel
        if ip.is_file():
            rows.append(
                {
                    "image_path": str(ip),
                    "boxes": boxes,
                    "keep_easy": keep_easy,
                    "keep_medium": keep_med,
                    "keep_hard": keep_hard,
                }
            )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for r in rows:
            out_f.write(json.dumps(r) + "\n")
    return {"images": len(rows), "easy_faces": easy_cnt, "medium_faces": med_cnt, "hard_faces": hard_cnt}


def build_wflw_manifest(
    wflw_split_dir: Path,
    out_jsonl: Path,
    bg_paths: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, int]:
    csv_path = wflw_split_dir / "labels.csv"
    imgs_dir = wflw_split_dir / "imgs"
    if not csv_path.is_file() or not imgs_dir.is_dir():
        raise FileNotFoundError(f"Expected labels.csv and imgs/ inside {wflw_split_dir}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    rng = random.Random(seed)

    # Probe image resolution once
    sample_png = next(imgs_dir.glob("*.png"))
    sbgr = cv2.imread(str(sample_png), cv2.IMREAD_COLOR)
    img_h, img_w = sbgr.shape[:2]

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, parts in enumerate(reader):
            if len(parts) < 197:
                continue
            fname = Path(parts[-1].strip()).name
            ip = imgs_dir / fname
            if not ip.is_file():
                skipped += 1
                continue

            pts98 = np.array([float(v) for v in parts[:196]], dtype=np.float32).reshape(98, 2)
            # Scale normalized [0, 1] coords to pixel coords on the 112x112 patch
            pts98[:, 0] *= float(img_w)
            pts98[:, 1] *= float(img_h)

            # Verified WFLW 98->5 mapping: left_eye=96, right_eye=97, nose=54, left_mouth=76, right_mouth=82
            kpts5 = pts98[[96, 97, 54, 76, 82]]
            # Sanity check: eyes ordered left < right and mouth corners left < right
            if kpts5[0, 0] >= kpts5[1, 0] or kpts5[3, 0] >= kpts5[4, 0]:
                skipped += 1
                continue

            # Tight face bbox from 98 landmarks with slight forehead headroom
            min_xy = pts98.min(axis=0)
            max_xy = pts98.max(axis=0)
            fh_span = max(float(max_xy[1] - min_xy[1]), 16.0)
            bx1 = float(max(0.0, min_xy[0] - 0.04 * fh_span))
            by1 = float(max(0.0, min_xy[1] - 0.14 * fh_span))
            bx2 = float(min(float(img_w), max_xy[0] + 0.04 * fh_span))
            by2 = float(min(float(img_h), max_xy[1] + 0.04 * fh_span))
            if bx2 <= bx1 + 8.0 or by2 <= by1 + 8.0:
                skipped += 1
                continue

            entry: Dict[str, Any] = {
                "image_path": str(ip),
                "boxes": [[bx1, by1, bx2, by2]],
                "kpts": [kpts5.tolist()],
                "kpts_valid": [True],
            }
            if bg_paths:
                entry["bg_image_path"] = bg_paths[rng.randrange(len(bg_paths))]
            rows.append(entry)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for r in rows:
            out_f.write(json.dumps(r) + "\n")
    return {"images": len(rows), "skipped": skipped, "patch_hw": [img_h, img_w]}


def build_wflw_official_manifest(
    images_root: Path,
    coco_json: Path,
    crops_dir: Path,
    out_jsonl: Path,
    context_scale: float = 2.4,
    min_crop_size: int = 224,
) -> Dict[str, Any]:
    """Build natural-scene WFLW portrait crop manifest from official WFLW images + COCO JSON."""
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_img: Dict[int, Dict[str, Any]] = {int(im["id"]): im for im in coco.get("images", [])}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    crops_dir.mkdir(parents=True, exist_ok=True)
    indexed_rows: List[tuple[int, Dict[str, Any]]] = []
    skipped = 0
    missing_imgs = 0
    multi_face_crops = 0
    min_max_hw = 10**9

    for img_id, anns in anns_by_img.items():
        img_info = id_to_img.get(img_id)
        if img_info is None:
            skipped += len(anns)
            continue
        rel_name = str(img_info["file_name"])
        ip = images_root / rel_name
        if not ip.is_file():
            ip = images_root / "WFLW_images" / rel_name
        if not ip.is_file():
            missing_imgs += 1
            skipped += len(anns)
            continue

        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            missing_imgs += 1
            skipped += len(anns)
            continue
        img_h, img_w = bgr.shape[:2]

        # Pre-extract valid annotations on this image to reuse for primary & co-occurring faces
        valid_faces: List[tuple[int, np.ndarray, np.ndarray]] = []
        for ann in anns:
            kpts_raw = np.asarray(ann.get("keypoints", []), dtype=np.float32)
            bbox_raw = ann.get("bbox", [])
            if kpts_raw.size < 98 * 3 or len(bbox_raw) < 4:
                continue
            pts98 = kpts_raw[: 98 * 3].reshape(98, 3)[:, :2]
            pts5 = pts98[[96, 97, 54, 76, 82]].copy()
            if pts5[0, 0] >= pts5[1, 0] or pts5[3, 0] >= pts5[4, 0]:
                continue
            bx, by, bw0, bh0 = [float(v) for v in bbox_raw[:4]]
            bx1 = max(0.0, bx)
            by1 = max(0.0, by)
            bx2 = min(float(img_w), bx + bw0)
            by2 = min(float(img_h), by + bh0)
            if (bx2 - bx1) < 12.0 or (by2 - by1) < 12.0:
                continue
            valid_faces.append((int(ann["id"]), np.array([bx1, by1, bx2, by2], dtype=np.float32), pts5))

        skipped += len(anns) - len(valid_faces)
        for ann_id, pbox, _ in valid_faces:
            bw = float(pbox[2] - pbox[0])
            bh = float(pbox[3] - pbox[1])
            cx = 0.5 * float(pbox[0] + pbox[2])
            cy = 0.5 * float(pbox[1] + pbox[3])
            half = max(0.5 * context_scale * max(bw, bh), float(min_crop_size) * 0.5)

            win_w = max(1, min(img_w, int(round(2.0 * half))))
            win_h = max(1, min(img_h, int(round(2.0 * half))))
            x0 = max(0, min(img_w - win_w, int(round(cx - 0.5 * win_w))))
            y0 = max(0, min(img_h - win_h, int(round(cy - 0.5 * win_h))))
            x1 = x0 + win_w
            y1 = y0 + win_h
            cw = x1 - x0
            ch = y1 - y0
            if cw <= 8 or ch <= 8:
                skipped += 1
                continue

            crop_bgr = bgr[y0:y1, x0:x1]
            boxes_list: List[np.ndarray] = []
            kpts_list: List[np.ndarray] = []

            # Always emit the primary centered face first (index 0) for clean 1-to-1 eval indexing
            ordered_faces = [f for f in valid_faces if f[0] == ann_id] + [f for f in valid_faces if f[0] != ann_id]
            for oid, obox, okpts in ordered_faces:
                if oid != ann_id:
                    ix1 = max(float(obox[0]), float(x0))
                    iy1 = max(float(obox[1]), float(y0))
                    ix2 = min(float(obox[2]), float(x1))
                    iy2 = min(float(obox[3]), float(y1))
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area = max(float((obox[2] - obox[0]) * (obox[3] - obox[1])), 1.0)
                    if inter < 0.50 * area:
                        continue
                sbox = obox.copy()
                sbox[[0, 2]] = np.clip(sbox[[0, 2]] - float(x0), 0.0, float(cw))
                sbox[[1, 3]] = np.clip(sbox[[1, 3]] - float(y0), 0.0, float(ch))
                skpt = okpts.copy()
                skpt[:, 0] -= float(x0)
                skpt[:, 1] -= float(y0)
                if (sbox[2] - sbox[0]) < 8.0 or (sbox[3] - sbox[1]) < 8.0:
                    continue
                if oid != ann_id and (
                    np.any(skpt[:, 0] < 0.0)
                    or np.any(skpt[:, 0] > float(cw))
                    or np.any(skpt[:, 1] < 0.0)
                    or np.any(skpt[:, 1] > float(ch))
                ):
                    continue
                boxes_list.append(sbox)
                kpts_list.append(skpt)

            if not boxes_list:
                skipped += 1
                continue

            boxes_arr = np.stack(boxes_list, axis=0)
            kpts_arr = np.stack(kpts_list, axis=0)

            out_w, out_h = cw, ch
            if max(ch, cw) < min_crop_size:
                scale = float(min_crop_size) / float(max(ch, cw))
                out_w = max(1, int(round(cw * scale)))
                out_h = max(1, int(round(ch * scale)))
                crop_bgr = cv2.resize(crop_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                sx = float(out_w) / float(cw)
                sy = float(out_h) / float(ch)
                boxes_arr[:, [0, 2]] *= sx
                boxes_arr[:, [1, 3]] *= sy
                kpts_arr[..., 0] *= sx
                kpts_arr[..., 1] *= sy

            min_max_hw = min(min_max_hw, max(out_h, out_w))
            if len(boxes_arr) >= 2:
                multi_face_crops += 1

            crop_path = crops_dir / f"{ann_id:06d}.jpg"
            cv2.imwrite(str(crop_path), crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            indexed_rows.append(
                (
                    ann_id,
                    {
                        "image_path": str(crop_path),
                        "boxes": boxes_arr.tolist(),
                        "kpts": kpts_arr.tolist(),
                        "kpts_valid": [True] * len(boxes_arr),
                    },
                )
            )

    indexed_rows.sort(key=lambda pair: pair[0])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for _, r in indexed_rows:
            out_f.write(json.dumps(r) + "\n")

    return {
        "images": len(indexed_rows),
        "skipped": skipped,
        "missing_source_imgs": missing_imgs,
        "multi_face_crops": multi_face_crops,
        "min_max_hw": 0 if not indexed_rows else int(min_max_hw),
    }


def build_coco_hard_neg_manifests(
    coco_dir: Path,
    out_train: Path,
    out_val: Path,
    val_count: int = 300,
    seed: int = 42,
) -> Dict[str, int]:
    imgs = sorted(
        [str(p) for p in coco_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not imgs:
        raise FileNotFoundError(f"No COCO hard-neg images under {coco_dir}")
    rng = random.Random(seed)
    shuffled = list(imgs)
    rng.shuffle(shuffled)
    val_imgs = sorted(shuffled[: min(val_count, max(1, len(shuffled) // 5))])
    train_imgs = sorted(shuffled[len(val_imgs) :])

    for out_p, subset in ((out_train, train_imgs), (out_val, val_imgs)):
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for ip in subset:
                f.write(json.dumps({"image_path": ip, "boxes": []}) + "\n")
    return {"train_images": len(train_imgs), "val_images": len(val_imgs)}


def build_exdark_manifest(exdark_dir: Path, out_jsonl: Path) -> Dict[str, Any]:
    people_dir = next((p for p in exdark_dir.rglob("*") if p.is_dir() and p.name.lower() == "people"), None)
    subset_name = "all"
    search_root = exdark_dir
    if people_dir is not None:
        people_imgs = sorted(
            [str(p) for p in people_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if len(people_imgs) >= 300:
            imgs = people_imgs
            subset_name = "People"
            search_root = people_dir
        else:
            imgs = sorted(
                [str(p) for p in exdark_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            )
    else:
        imgs = sorted(
            [str(p) for p in exdark_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
    if not imgs:
        raise FileNotFoundError(f"No ExDark images found under {search_root}")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ip in imgs:
            f.write(json.dumps({"image_path": ip, "boxes": []}) + "\n")
    return {"images": len(imgs), "subset": subset_name}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 JSONL manifests on compute server")
    parser.add_argument("--datasets-root", type=Path, default=Path("/workspace/LightFacePipeline/datasets"))
    args = parser.parse_args()

    root = args.datasets_root
    m_dir = root / "manifests"
    m_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}

    if (root / "coco_indoor_neg").is_dir() and any((root / "coco_indoor_neg").rglob("*.jpg")):
        summary["coco_hard_neg"] = build_coco_hard_neg_manifests(
            root / "coco_indoor_neg",
            m_dir / "coco_hard_neg_train.jsonl",
            m_dir / "coco_hard_neg_val.jsonl",
        )
        val_bgs = [
            json.loads(ln)["image_path"]
            for ln in (m_dir / "coco_hard_neg_val.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    else:
        val_bgs = []

    wider_dir = root / "widerface"
    if (wider_dir / "retinaface_train_label.txt").is_file():
        summary["wider_train"] = build_wider_train_manifest(
            wider_dir, wider_dir / "retinaface_train_label.txt", m_dir / "wider_train_5pt.jsonl"
        )
    split_dir = wider_dir / "wider_face_split"
    if split_dir.is_dir():
        summary["wider_val"] = build_wider_val_manifest(wider_dir, split_dir, m_dir / "wider_val.jsonl")

    wflw_off = root / "wflw_official"
    wflw_train_json = wflw_off / "annotations" / "face_landmarks_wflw_train.json"
    wflw_test_json = wflw_off / "annotations" / "face_landmarks_wflw_test.json"
    if wflw_train_json.is_file():
        wflw_img_root = wflw_off / "WFLW_images" if (wflw_off / "WFLW_images").is_dir() else wflw_off
        summary["wflw_train"] = build_wflw_official_manifest(
            wflw_img_root,
            wflw_train_json,
            wflw_off / "crops_train",
            m_dir / "wflw_train_5pt.jsonl",
            context_scale=2.4,
            min_crop_size=224,
        )
        if wflw_test_json.is_file():
            summary["wflw_test"] = build_wflw_official_manifest(
                wflw_img_root,
                wflw_test_json,
                wflw_off / "crops_test",
                m_dir / "wflw_test_5pt.jsonl",
                context_scale=2.5,
                min_crop_size=224,
            )
    else:
        wflw_dir = root / "wflw"
        if (wflw_dir / "train_data").is_dir():
            summary["wflw_train"] = build_wflw_manifest(wflw_dir / "train_data", m_dir / "wflw_train_5pt.jsonl")
        if (wflw_dir / "test_data").is_dir():
            summary["wflw_test"] = build_wflw_manifest(
                wflw_dir / "test_data", m_dir / "wflw_test_5pt.jsonl", bg_paths=val_bgs
            )

    exdark_dir = root / "exdark"
    if exdark_dir.is_dir() and any(p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png") for p in exdark_dir.rglob("*")):
        summary["exdark_unlabeled"] = build_exdark_manifest(exdark_dir, m_dir / "exdark_unlabeled.jsonl")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
