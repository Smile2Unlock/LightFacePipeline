"""Phase 1 5-source weighted dataset loader (CPU-only decoding/geometry -> GPU degradation)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from common.align import composite_tight_face_on_canvas, letterbox_rgb


@dataclass
class DetectorSampleSpec:
    source_type: str  # "wider", "darkface_labeled", "exdark_unlabeled", "closeup_5pt", "hard_neg"
    image_path: Path
    pair_clean_path: Optional[Path]
    boxes_xyxy: np.ndarray  # [M, 4] float32 in original pixel coords
    kpts_5: np.ndarray  # [M, 5, 2] float32 in original pixel coords
    kpts_valid: np.ndarray  # [M] bool per-face landmark validity mask
    has_landmarks: bool
    has_clean_pair: bool
    apply_degrade: bool
    closeup_zoom: bool


def parse_darkface_label_txt(label_path: Path) -> np.ndarray:
    """Parse DarkFace `.txt` label: line 0 = count, subsequent lines = `x1 y1 x2 y2`."""
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return np.zeros((0, 4), dtype=np.float32)
    boxes: List[List[float]] = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 4:
            x1, y1, x2, y2 = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def load_darkface_labeled_specs(
    data_dir: Path,
    split_json: Path,
    split: str = "train",
) -> List[DetectorSampleSpec]:
    manifest = json.loads(Path(split_json).read_text(encoding="utf-8"))
    filenames: List[str] = manifest[split]
    img_dir = data_dir / "image"
    lbl_dir = data_dir / "label"
    specs: List[DetectorSampleSpec] = []
    for fname in filenames:
        stem = Path(fname).stem
        ip = img_dir / fname
        lp = lbl_dir / f"{stem}.txt"
        boxes = parse_darkface_label_txt(lp) if lp.is_file() else np.zeros((0, 4), dtype=np.float32)
        kpts = np.zeros((len(boxes), 5, 2), dtype=np.float32)
        kpts_valid = np.zeros((len(boxes),), dtype=np.bool_)
        specs.append(
            DetectorSampleSpec(
                source_type="darkface_labeled",
                image_path=ip,
                pair_clean_path=None,
                boxes_xyxy=boxes,
                kpts_5=kpts,
                kpts_valid=kpts_valid,
                has_landmarks=False,  # STRICT: DarkFace has NO landmarks
                has_clean_pair=False,
                apply_degrade=False,  # Preserve authentic nighttime sensor noise
                closeup_zoom=True,
            )
        )
    return specs


def load_jsonl_face_specs(
    jsonl_path: Path,
    source_type: str,
    has_landmarks: bool,
    has_clean_pair: bool,
    apply_degrade: bool,
    closeup_zoom: bool,
) -> List[DetectorSampleSpec]:
    """Load face JSONL manifest (`{image_path, boxes, kpts?, kpts_valid?, pair_clean_path?}`)."""
    specs: List[DetectorSampleSpec] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            boxes = np.asarray(row.get("boxes", []), dtype=np.float32).reshape(-1, 4)
            n = len(boxes)
            if has_landmarks and "kpts" in row and len(row["kpts"]) == n:
                kpts = np.asarray(row["kpts"], dtype=np.float32).reshape(-1, 5, 2)
                if "kpts_valid" in row and len(row["kpts_valid"]) == n:
                    kvalid = np.asarray(row["kpts_valid"], dtype=np.bool_)
                else:
                    kvalid = np.all(np.isfinite(kpts) & (kpts >= 0.0), axis=(1, 2))
            else:
                kpts = np.zeros((n, 5, 2), dtype=np.float32)
                kvalid = np.zeros((n,), dtype=np.bool_)
            pair_clean = Path(row["pair_clean_path"]) if row.get("pair_clean_path") else None
            specs.append(
                DetectorSampleSpec(
                    source_type=source_type,
                    image_path=Path(row["image_path"]),
                    pair_clean_path=pair_clean,
                    boxes_xyxy=boxes,
                    kpts_5=kpts,
                    kpts_valid=kvalid,
                    has_landmarks=has_landmarks and ("kpts" in row) and bool(kvalid.any()),
                    has_clean_pair=has_clean_pair,
                    apply_degrade=apply_degrade,
                    closeup_zoom=closeup_zoom,
                )
            )
    return specs


def _random_zoom_crop_around_face(
    rgb: np.ndarray,
    boxes: np.ndarray,
    kpts: np.ndarray,
    kpts_valid: np.ndarray,
    rng: random.Random,
    target_face_frac: Tuple[float, float] = (0.15, 0.65),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random zoom-in crop so close-up faces occupy 15%-65% of image height."""
    if len(boxes) == 0:
        return rgb, boxes, kpts, kpts_valid
    h, w = rgb.shape[:2]
    idx = rng.randrange(len(boxes))
    bx1, by1, bx2, by2 = boxes[idx]
    bh = max(float(by2 - by1), 8.0)
    frac = rng.uniform(*target_face_frac)
    crop_h = int(min(h, max(64.0, bh / frac)))
    crop_w = int(min(w, max(64.0, crop_h * (w / max(h, 1)))))
    cx = 0.5 * (bx1 + bx2) + rng.uniform(-0.15, 0.15) * crop_w
    cy = 0.5 * (by1 + by2) + rng.uniform(-0.15, 0.15) * crop_h
    x0 = int(max(0, min(w - crop_w, round(cx - 0.5 * crop_w))))
    y0 = int(max(0, min(h - crop_h, round(cy - 0.5 * crop_h))))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    cropped = rgb[y0:y1, x0:x1]
    new_boxes = boxes.copy()
    new_boxes[:, [0, 2]] -= x0
    new_boxes[:, [1, 3]] -= y0
    keep = (
        (new_boxes[:, 2] > 2)
        & (new_boxes[:, 3] > 2)
        & (new_boxes[:, 0] < crop_w - 2)
        & (new_boxes[:, 1] < crop_h - 2)
    )
    if not keep.any():
        return rgb, boxes, kpts, kpts_valid
    new_boxes = new_boxes[keep]
    new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, crop_w)
    new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, crop_h)

    new_kpts = kpts.copy()[keep]
    new_kpts[..., 0] -= x0
    new_kpts[..., 1] -= y0
    new_kv = kpts_valid.copy()[keep]
    # Invalidate landmarks if face center was heavily clipped
    in_bounds = (
        (new_kpts[..., 0] >= 0)
        & (new_kpts[..., 0] <= crop_w)
        & (new_kpts[..., 1] >= 0)
        & (new_kpts[..., 1] <= crop_h)
    )
    new_kv = new_kv & np.all(in_bounds, axis=1)
    return cropped, new_boxes, new_kpts, new_kv


def _random_scale_crop(
    rgb: np.ndarray,
    boxes: np.ndarray,
    kpts: np.ndarray,
    kpts_valid: np.ndarray,
    rng: random.Random,
    scale_range: Tuple[float, float] = (0.55, 1.35),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply random scale resize + random spatial crop/pad preserving boxes & landmarks."""
    h, w = rgb.shape[:2]
    sc = rng.uniform(*scale_range)
    nh = max(64, int(round(h * sc)))
    nw = max(64, int(round(w * sc)))
    scaled = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)
    b = boxes.copy()
    k = kpts.copy()
    kv = kpts_valid.copy()
    if len(b):
        b *= sc
        k *= sc
    if nh > h or nw > w:
        ch = min(h, nh)
        cw = min(w, nw)
        x0 = rng.randint(0, max(0, nw - cw))
        y0 = rng.randint(0, max(0, nh - ch))
        cropped = scaled[y0 : y0 + ch, x0 : x0 + cw]
        if len(b):
            b[:, [0, 2]] -= x0
            b[:, [1, 3]] -= y0
            keep = (b[:, 2] > 2) & (b[:, 3] > 2) & (b[:, 0] < cw - 2) & (b[:, 1] < ch - 2)
            if not keep.any():
                return rgb, boxes, kpts, kpts_valid
            b = b[keep]
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, cw)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, ch)
            k = k[keep]
            k[..., 0] -= x0
            k[..., 1] -= y0
            kv = kv[keep]
            in_bounds = (k[..., 0] >= 0) & (k[..., 0] <= cw) & (k[..., 1] >= 0) & (k[..., 1] <= ch)
            kv = kv & np.all(in_bounds, axis=1)
        return cropped, b, k, kv
    return scaled, b, k, kv


def _horizontal_flip_rgb(
    rgb: np.ndarray,
    boxes: np.ndarray,
    kpts: np.ndarray,
    kpts_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flip RGB horizontally AND swap left/right 5-pt landmark order cleanly."""
    _, w = rgb.shape[:2]
    flipped = np.ascontiguousarray(rgb[:, ::-1, :])
    b = boxes.copy()
    if len(b):
        old_x1 = b[:, 0].copy()
        old_x2 = b[:, 2].copy()
        b[:, 0] = w - old_x2
        b[:, 2] = w - old_x1
    k = kpts.copy()
    if len(k) and kpts_valid.any():
        k[..., 0] = w - k[..., 0]
        # Swap image-left/image-right eyes (0<->1) and mouth corners (3<->4)
        k[:, [0, 1]] = k[:, [1, 0]]
        k[:, [3, 4]] = k[:, [4, 3]]
    return flipped, b, k, kpts_valid


class MixedFaceDetectorDataset(Dataset):
    """5-source dataset producing CPU uint8 tensors (zero GPU/HIP calls inside DataLoader workers)."""

    SOURCE_WEIGHTS = {
        "wider": 0.40,
        "darkface_labeled": 0.20,
        "exdark_unlabeled": 0.15,
        "closeup_5pt": 0.15,
        "hard_neg": 0.10,
    }

    def __init__(
        self,
        specs: Sequence[DetectorSampleSpec],
        img_size: int = 640,
        max_faces: int = 128,
        augment: bool = True,
        seed: int = 42,
        bg_image_paths: Optional[Sequence[Path]] = None,
    ) -> None:
        self.specs = list(specs)
        self.img_size = img_size
        self.max_faces = max_faces
        self.augment = augment
        self.seed = seed
        self.bg_image_paths = list(bg_image_paths) if bg_image_paths else []

    def build_weighted_sampler(
        self,
        num_samples: Optional[int] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> WeightedRandomSampler:
        wmap = custom_weights or self.SOURCE_WEIGHTS
        counts: Dict[str, int] = {}
        for s in self.specs:
            counts[s.source_type] = counts.get(s.source_type, 0) + 1
        weights = [wmap.get(s.source_type, 0.1) / max(counts[s.source_type], 1) for s in self.specs]
        return WeightedRandomSampler(weights=weights, num_samples=num_samples or len(self.specs), replacement=True)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        spec = self.specs[idx]
        # Worker-safe per-epoch dynamic RNG when augmenting; fixed deterministic RNG when evaluating
        rng = random.Random() if self.augment else random.Random(self.seed + idx)
        bgr = cv2.imread(str(spec.image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {spec.image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        boxes = spec.boxes_xyxy.copy()
        kpts = spec.kpts_5.copy()
        kv = spec.kpts_valid.copy()

        # Composite tight 112x112 face crops (WFLW_augmented) onto full indoor/scene backgrounds
        if spec.source_type == "closeup_5pt" and max(rgb.shape[:2]) <= 160:
            bg_rgb = None
            if self.bg_image_paths:
                bg_p = self.bg_image_paths[rng.randrange(len(self.bg_image_paths))]
                bg_bgr = cv2.imread(str(bg_p), cv2.IMREAD_COLOR)
                if bg_bgr is not None:
                    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            np_rng = np.random.default_rng(rng.randrange(1 << 30))
            rgb, boxes, kpts = composite_tight_face_on_canvas(
                rgb,
                boxes,
                kpts,
                bg_rgb=bg_rgb,
                canvas_size=480,
                rng=np_rng,
            )

        if spec.pair_clean_path is not None and spec.pair_clean_path.is_file():
            cbgr = cv2.imread(str(spec.pair_clean_path), cv2.IMREAD_COLOR)
            clean_rgb = cv2.cvtColor(cbgr, cv2.COLOR_BGR2RGB) if cbgr is not None else rgb.copy()
        else:
            clean_rgb = rgb.copy()

        if self.augment:
            zoom_p = 0.30 if spec.source_type == "darkface_labeled" else 0.45
            if spec.closeup_zoom and len(boxes) > 0 and rng.random() < zoom_p:
                rgb, boxes, kpts, kv = _random_zoom_crop_around_face(rgb, boxes, kpts, kv, rng)
                clean_rgb = rgb.copy()
            elif len(boxes) > 0 and rng.random() < 0.35:
                rgb, boxes, kpts, kv = _random_scale_crop(rgb, boxes, kpts, kv, rng)
                clean_rgb = rgb.copy()

            if rng.random() < 0.5:
                rgb, boxes, kpts, kv = _horizontal_flip_rgb(rgb, boxes, kpts, kv)
                clean_rgb, _, _, _ = _horizontal_flip_rgb(
                    clean_rgb, np.zeros((0, 4), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_)
                )

        lb_rgb, scale, (pad_x, pad_y) = letterbox_rgb(rgb, (self.img_size, self.img_size))
        lb_clean, _, _ = letterbox_rgb(clean_rgb, (self.img_size, self.img_size))

        boxes_pad = np.zeros((self.max_faces, 4), dtype=np.float32)
        kpts_pad = np.zeros((self.max_faces, 5, 2), dtype=np.float32)
        valid_pad = np.zeros((self.max_faces,), dtype=np.bool_)
        kvalid_pad = np.zeros((self.max_faces,), dtype=np.bool_)

        n_keep = min(len(boxes), self.max_faces)
        if n_keep > 0:
            # Prioritize larger faces if dense crowd > max_faces
            if len(boxes) > self.max_faces:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                top_idx = np.argsort(-areas)[: self.max_faces]
                boxes = boxes[top_idx]
                kpts = kpts[top_idx]
                kv = kv[top_idx]

            b = boxes[:n_keep].copy()
            b[:, [0, 2]] = b[:, [0, 2]] * scale + pad_x
            b[:, [1, 3]] = b[:, [1, 3]] * scale + pad_y
            boxes_pad[:n_keep] = np.clip(b, 0.0, float(self.img_size))
            valid_pad[:n_keep] = True
            if spec.has_landmarks:
                k = kpts[:n_keep].copy()
                k[..., 0] = k[..., 0] * scale + pad_x
                k[..., 1] = k[..., 1] * scale + pad_y
                kpts_pad[:n_keep] = k
                kvalid_pad[:n_keep] = kv[:n_keep]

        img_u8 = torch.from_numpy(np.ascontiguousarray(lb_rgb.transpose(2, 0, 1)))
        clean_u8 = torch.from_numpy(np.ascontiguousarray(lb_clean.transpose(2, 0, 1)))
        return {
            "img_u8": img_u8,
            "clean_u8": clean_u8,
            "gt_boxes": torch.from_numpy(boxes_pad),
            "gt_kpts": torch.from_numpy(kpts_pad),
            "gt_valid": torch.from_numpy(valid_pad),
            "gt_kpts_valid": torch.from_numpy(kvalid_pad),
            "has_landmarks": torch.tensor(spec.has_landmarks, dtype=torch.bool),
            "has_clean_pair": torch.tensor(spec.has_clean_pair, dtype=torch.bool),
            "apply_degrade": torch.tensor(spec.apply_degrade, dtype=torch.bool),
            "is_unlabeled": torch.tensor(spec.source_type in ("exdark_unlabeled", "darkface_unlabeled"), dtype=torch.bool),
            "source_type": spec.source_type,
        }


class SyntheticDetectorSmokeDataset(Dataset):
    """Deterministic synthetic dataset for offline GPU/CPU smoke tests."""

    def __init__(self, n: int = 32, img_size: int = 320, max_faces: int = 8) -> None:
        self.n = n
        self.img_size = img_size
        self.max_faces = max_faces

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(idx + 100)
        img = np.full((self.img_size, self.img_size, 3), 120, dtype=np.uint8)
        cx = int(rng.integers(self.img_size // 4, 3 * self.img_size // 4))
        cy = int(rng.integers(self.img_size // 4, 3 * self.img_size // 4))
        bw = int(rng.integers(48, 96))
        bh = int(bw * 1.2)
        x1, y1 = max(4, cx - bw // 2), max(4, cy - bh // 2)
        x2, y2 = min(self.img_size - 4, x1 + bw), min(self.img_size - 4, y1 + bh)
        cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), (bw // 2, bh // 2), 0, 0, 360, (195, 165, 150), -1)

        kpts = np.array(
            [
                [x1 + 0.32 * bw, y1 + 0.38 * bh],
                [x1 + 0.68 * bw, y1 + 0.38 * bh],
                [x1 + 0.50 * bw, y1 + 0.56 * bh],
                [x1 + 0.36 * bw, y1 + 0.76 * bh],
                [x1 + 0.64 * bw, y1 + 0.76 * bh],
            ],
            dtype=np.float32,
        )
        boxes_pad = np.zeros((self.max_faces, 4), dtype=np.float32)
        kpts_pad = np.zeros((self.max_faces, 5, 2), dtype=np.float32)
        valid_pad = np.zeros((self.max_faces,), dtype=np.bool_)
        kvalid_pad = np.zeros((self.max_faces,), dtype=np.bool_)
        boxes_pad[0] = [x1, y1, x2, y2]
        kpts_pad[0] = kpts
        valid_pad[0] = True
        has_lm = idx % 3 != 0
        kvalid_pad[0] = has_lm

        u8 = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return {
            "img_u8": u8,
            "clean_u8": u8.clone(),
            "gt_boxes": torch.from_numpy(boxes_pad),
            "gt_kpts": torch.from_numpy(kpts_pad),
            "gt_valid": torch.from_numpy(valid_pad),
            "gt_kpts_valid": torch.from_numpy(kvalid_pad),
            "has_landmarks": torch.tensor(has_lm, dtype=torch.bool),
            "has_clean_pair": torch.tensor(has_lm, dtype=torch.bool),
            "apply_degrade": torch.tensor(has_lm, dtype=torch.bool),
            "is_unlabeled": torch.tensor(idx % 5 == 0, dtype=torch.bool),
            "source_type": "synthetic",
        }
