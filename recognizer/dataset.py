"""Phase 2 Twin-view dataset (clean vs 5-pt-jittered pre-aligned CPU uint8 -> GPU degradation)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.align import align_face_112
from common.contracts import ARCFACE_DST_112


class TwinViewRecognitionDataset(Dataset):
    """Loads ID face samples and outputs CPU `uint8` twin views (`clean_u8`, `dark_src_u8`).

    `clean_u8`: normal-light aligned 112x112 (with <=1px alignment noise + random H-flip).
    `dark_src_u8`: aligned with empirical Phase-1 detector 5-pt jitter (sigma=1.5~3.0 px + scale/angle),
                   ready for GPU PhysicalLowLightDegrader(mode="recognizer").
    """

    def __init__(
        self,
        manifest_jsonl: Path,
        kpt_jitter_sigma_range: Tuple[float, float] = (1.5, 3.0),
        scale_jitter: float = 0.04,
        angle_jitter_deg: float = 3.0,
        seed: int = 42,
    ) -> None:
        self.rows: List[Dict[str, Any]] = []
        with open(manifest_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.kpt_jitter_sigma_range = kpt_jitter_sigma_range
        self.scale_jitter = scale_jitter
        self.angle_jitter_deg = angle_jitter_deg
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        rng = np.random.default_rng(self.seed + idx * 131)
        bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["image_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # If image is already pre-aligned 112x112 without explicit landmarks, default to canonical ArcFace 5-pts
        if "kpts" in row and row["kpts"] is not None:
            kpts5 = np.asarray(row["kpts"], dtype=np.float32).reshape(5, 2)
        else:
            if rgb.shape[:2] != (112, 112):
                rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
            kpts5 = ARCFACE_DST_112.copy()

        # Random horizontal flip applied consistently to both views
        if rng.random() < 0.5:
            w = rgb.shape[1]
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            kpts5 = kpts5.copy()
            kpts5[:, 0] = w - kpts5[:, 0]
            kpts5[[0, 1]] = kpts5[[1, 0]]
            kpts5[[3, 4]] = kpts5[[4, 3]]

        clean_112, _ = align_face_112(rgb, kpts5, jitter_sigma_px=0.5, rng=rng)
        sigma = float(rng.uniform(*self.kpt_jitter_sigma_range))
        dark_src_112, _ = align_face_112(
            rgb,
            kpts5,
            jitter_sigma_px=sigma,
            scale_jitter=self.scale_jitter,
            angle_jitter_deg=self.angle_jitter_deg,
            rng=rng,
        )

        # Normalized 5-pt template in [0, 1] for CVLface KP-RPE teacher
        kpts_norm = (ARCFACE_DST_112 / 112.0).astype(np.float32)

        return {
            "clean_u8": torch.from_numpy(np.ascontiguousarray(clean_112.transpose(2, 0, 1))),
            "dark_src_u8": torch.from_numpy(np.ascontiguousarray(dark_src_112.transpose(2, 0, 1))),
            "kpts_norm": torch.from_numpy(kpts_norm),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class SyntheticRecognizerSmokeDataset(Dataset):
    def __init__(self, n: int = 64, num_classes: int = 16) -> None:
        self.n = n
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(idx + 500)
        cls_id = idx % self.num_classes
        base = np.full((112, 112, 3), 90 + (cls_id * 7) % 100, dtype=np.uint8)
        cv2.circle(base, (38, 52), 6, (40, 40, 40), -1)
        cv2.circle(base, (74, 52), 6, (40, 40, 40), -1)
        cv2.ellipse(base, (56, 86), (14, 6), 0, 0, 360, (120, 60, 60), -1)
        dark_src, _ = align_face_112(base, ARCFACE_DST_112, jitter_sigma_px=1.8, scale_jitter=0.03, angle_jitter_deg=2.0, rng=rng)
        return {
            "clean_u8": torch.from_numpy(np.ascontiguousarray(base.transpose(2, 0, 1))),
            "dark_src_u8": torch.from_numpy(np.ascontiguousarray(dark_src.transpose(2, 0, 1))),
            "kpts_norm": torch.from_numpy((ARCFACE_DST_112 / 112.0).astype(np.float32)),
            "label": torch.tensor(cls_id, dtype=torch.long),
        }
