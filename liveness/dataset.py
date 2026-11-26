"""Phase 3 Liveness Datasets: strict ban on repeating static frames as Live temporal clips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from common.align import crop_dual_scale_fas
from common.contracts import FAS_CLASSES, FAS_CLIP_LEN, FAS_CROP_SIZE


class VideoClipLivenessDataset(Dataset):
    """Loads multi-frame continuous video clips `[T=8]` with Strategy C per-frame bbox jitter.

    Enforces strict anti-static-repeat redline: Live clips MUST consist of genuine distinct video frames.
    """

    def __init__(
        self,
        manifest_jsonl: Path,
        clip_len: int = FAS_CLIP_LEN,
        crop_size: int = FAS_CROP_SIZE,
        box_jitter_frac: float = 0.03,
        scale_jitter_frac: float = 0.02,
        augment: bool = True,
        hflip_prob: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.rows: List[Dict[str, Any]] = []
        with open(manifest_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    frames = row["frame_paths"]
                    if len(frames) < clip_len:
                        raise ValueError(f"Clip has only {len(frames)} < {clip_len} frames: {row.get('clip_id')}")
                    if int(row["label"]) == 0 and len(set(frames)) < 2:
                        raise ValueError(
                            f"REDLINE VIOLATION: Refusing static-image repetition as Live video clip ({row.get('clip_id')})"
                        )
                    self.rows.append(row)
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.box_jitter_frac = box_jitter_frac if augment else 0.0
        self.scale_jitter_frac = scale_jitter_frac if augment else 0.0
        self.augment = augment
        self.hflip_prob = hflip_prob if augment else 0.0
        self.seed = seed

    def build_balanced_sampler(
        self,
        num_samples: Optional[int] = None,
        balance_domains: bool = True,
    ) -> WeightedRandomSampler:
        """Balance Live (label==0) vs Spoof (label!=0) at 1:1 while balancing across source domains."""
        n_total = num_samples or len(self.rows)
        domain_cls_counts: Dict[Tuple[str, int], int] = {}
        for r in self.rows:
            src = str(r.get("source", "default"))
            is_spoof = 0 if int(r["label"]) == 0 else 1
            domain_cls_counts[(src, is_spoof)] = domain_cls_counts.get((src, is_spoof), 0) + 1

        cls_totals = {0: 0, 1: 0}
        for (_, is_spoof), cnt in domain_cls_counts.items():
            cls_totals[is_spoof] += cnt

        weights: List[float] = []
        for r in self.rows:
            src = str(r.get("source", "default"))
            is_spoof = 0 if int(r["label"]) == 0 else 1
            if balance_domains:
                # Inverse sqrt domain frequency inside Live/Spoof keeps every source well represented without extreme noise
                dc = domain_cls_counts[(src, is_spoof)]
                w = 1.0 / (max(dc, 1) ** 0.50)
            else:
                w = 0.5 / max(cls_totals[is_spoof], 1)
            weights.append(float(w))

        # Re-normalize so sum of Live weights == sum of Spoof weights == 0.5
        sum_live = sum(w for w, r in zip(weights, self.rows) if int(r["label"]) == 0)
        sum_spoof = sum(w for w, r in zip(weights, self.rows) if int(r["label"]) != 0)
        norm_weights = [
            (0.5 * w / max(sum_live, 1e-12)) if int(r["label"]) == 0 else (0.5 * w / max(sum_spoof, 1e-12))
            for w, r in zip(weights, self.rows)
        ]
        return WeightedRandomSampler(weights=norm_weights, num_samples=n_total, replacement=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        # Fresh RNG per call during training so epoch-to-epoch sampling & bbox jitter never repeat identically
        rng = np.random.default_rng() if self.augment else np.random.default_rng(self.seed + idx * 97)
        frames = row["frame_paths"]
        bboxes = row["bboxes"]

        if self.augment:
            valid_strides = [s for s in (1, 2, 3) if len(frames) >= 1 + (self.clip_len - 1) * s]
            stride = int(rng.choice(valid_strides)) if valid_strides else 1
            span = 1 + (self.clip_len - 1) * stride
            start = int(rng.integers(0, max(1, len(frames) - span + 1)))
        else:
            stride = min(2, max(1, (len(frames) - 1) // max(self.clip_len - 1, 1)))
            span = 1 + (self.clip_len - 1) * stride
            start = max(0, (len(frames) - span) // 2)
        indices = [start + i * stride for i in range(self.clip_len)]

        do_hflip = bool(self.augment and rng.random() < self.hflip_prob)

        tex_list: List[np.ndarray] = []
        ctx_list: List[np.ndarray] = []
        for fi in indices:
            bgr = cv2.imread(frames[fi], cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(frames[fi])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bb = np.asarray(bboxes[fi] if len(bboxes) > 1 else bboxes[0], dtype=np.float32)
            tex, ctx = crop_dual_scale_fas(
                rgb,
                bb,
                out_size=self.crop_size,
                box_jitter_frac=self.box_jitter_frac,
                scale_jitter_frac=self.scale_jitter_frac,
                rng=rng,
            )
            if do_hflip:
                tex = tex[:, ::-1, :]
                ctx = ctx[:, ::-1, :]
            tex_list.append(tex.transpose(2, 0, 1))
            ctx_list.append(ctx.transpose(2, 0, 1))

        tex_u8 = np.stack(tex_list, axis=0)  # [T, 3, 128, 128]
        ctx_u8 = np.stack(ctx_list, axis=0)
        label = int(row["label"])
        fine_known = bool(row.get("fine_label_known", True))

        if label == 0:
            yy, xx = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), indexing="ij")
            dome = np.clip(1.0 - (xx ** 2 + yy ** 2), 0.0, 1.0).astype(np.float32) * 0.8
            depth_target = np.broadcast_to(dome[None, None, :, :], (self.clip_len, 1, 8, 8)).copy()
        else:
            depth_target = np.zeros((self.clip_len, 1, 8, 8), dtype=np.float32)

        return {
            "tex_u8": torch.from_numpy(np.ascontiguousarray(tex_u8)),
            "ctx_u8": torch.from_numpy(np.ascontiguousarray(ctx_u8)),
            "label": torch.tensor(label, dtype=torch.long),
            "depth_target": torch.from_numpy(depth_target),
            "fine_known": torch.tensor(fine_known, dtype=torch.bool),
            "source": str(row.get("source", "unknown")),
            "attack_group": str(row.get("attack_group", FAS_CLASSES[label] if label < len(FAS_CLASSES) else "spoof")),
        }


class SyntheticLivenessSmokeDataset(Dataset):
    """Synthetic multi-frame temporal dataset for fast smoke tests (never repeats static live frames)."""

    def __init__(self, n: int = 24, clip_len: int = FAS_CLIP_LEN, size: int = FAS_CROP_SIZE) -> None:
        self.n = n
        self.clip_len = clip_len
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cls_id = idx % 3  # 0: live, 1: print, 2: replay (4: frozen synthesized online by Strategy B)
        tex_frames = []
        ctx_frames = []
        for t in range(self.clip_len):
            img = np.full((self.size, self.size, 3), 110 + 10 * cls_id, dtype=np.uint8)
            shift = t * 2 if cls_id == 0 else 0
            cv2.circle(img, (64 + shift, 64), 34, (195, 165, 150), -1)
            cv2.circle(img, (52 + shift, 54), 4, (35, 35, 35), -1)
            cv2.circle(img, (76 + shift, 54), 4, (35, 35, 35), -1)
            tex_frames.append(img.transpose(2, 0, 1))
            ctx_frames.append(img.transpose(2, 0, 1))

        if cls_id == 0:
            yy, xx = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), indexing="ij")
            dome = np.clip(1.0 - (xx ** 2 + yy ** 2), 0.0, 1.0).astype(np.float32) * 0.8
            depth_target = np.broadcast_to(dome[None, None, :, :], (self.clip_len, 1, 8, 8)).copy()
        else:
            depth_target = np.zeros((self.clip_len, 1, 8, 8), dtype=np.float32)

        return {
            "tex_u8": torch.from_numpy(np.ascontiguousarray(np.stack(tex_frames, axis=0))),
            "ctx_u8": torch.from_numpy(np.ascontiguousarray(np.stack(ctx_frames, axis=0))),
            "label": torch.tensor(cls_id, dtype=torch.long),
            "depth_target": torch.from_numpy(depth_target),
        }
