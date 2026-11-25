"""Phase 2 Twin-view datasets (indexed JPEG blob & JSONL manifest) -> clean + 5-pt-jittered dark source."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.align import _perturb_landmarks, align_face_112, estimate_umeyama_similarity
from common.contracts import ARCFACE_DST_112


class IndexedBlobRecognitionDataset(Dataset):
    """High-throughput worker-safe JPEG blob reader outputting CPU `uint8` twin views.

    Expects `data_dir` containing `train_images.blob`, `offsets.npy`, `lengths.npy`, `labels.npy`.
    Uses `os.pread` with per-worker lazy file descriptors so forked/persistent DataLoader workers
    never race on shared file offsets.
    """

    def __init__(
        self,
        data_dir: Path,
        kpt_jitter_sigma_range: Tuple[float, float] = (1.2, 2.5),
        scale_jitter: float = 0.04,
        angle_jitter_deg: float = 4.0,
        jitter_prob: float = 0.90,
        clean_jitter_sigma: float = 0.0,
        augment: bool = True,
        seed: int = 42,
        teacher_embs_path: Optional[Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.blob_path = self.data_dir / "train_images.blob"
        self.offsets = np.load(self.data_dir / "offsets.npy")
        self.lengths = np.load(self.data_dir / "lengths.npy")
        self.labels = np.load(self.data_dir / "labels.npy")
        self.kpt_jitter_sigma_range = kpt_jitter_sigma_range
        self.scale_jitter = scale_jitter
        self.angle_jitter_deg = angle_jitter_deg
        self.jitter_prob = jitter_prob
        self.clean_jitter_sigma = clean_jitter_sigma
        self.augment = augment
        self.seed = seed
        self.teacher_embs_path = Path(teacher_embs_path) if teacher_embs_path else None
        self._fd: Optional[int] = None
        self._fd_pid: Optional[int] = None
        self._t_embs: Optional[np.ndarray] = None
        self._t_pid: Optional[int] = None

    def _ensure_teacher_mmap(self) -> Optional[np.ndarray]:
        if self.teacher_embs_path is None or not self.teacher_embs_path.is_file():
            return None
        pid = os.getpid()
        if self._t_embs is None or self._t_pid != pid:
            self._t_embs = np.load(str(self.teacher_embs_path), mmap_mode="r")
            self._t_pid = pid
        return self._t_embs

    def _ensure_open(self) -> int:
        pid = os.getpid()
        if self._fd is None or self._fd_pid != pid:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
            self._fd = os.open(str(self.blob_path), os.O_RDONLY)
            self._fd_pid = pid
        return self._fd

    def __del__(self) -> None:
        if getattr(self, "_fd", None) is not None:
            try:
                os.close(self._fd)  # type: ignore[arg-type]
            except OSError:
                pass
            self._fd = None

    def __len__(self) -> int:
        return int(len(self.labels))

    def _warp_prealigned_reflect(
        self,
        rgb: np.ndarray,
        jitter_sigma_px: float,
        scale_jitter: float,
        angle_jitter_deg: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        pts = _perturb_landmarks(
            ARCFACE_DST_112,
            jitter_sigma_px=jitter_sigma_px,
            scale_jitter=scale_jitter,
            angle_jitter_deg=angle_jitter_deg,
            rng=rng,
        )
        m = estimate_umeyama_similarity(pts, ARCFACE_DST_112)
        return cv2.warpAffine(
            rgb,
            m,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng() if self.augment else np.random.default_rng(self.seed + idx * 131)
        fd = self._ensure_open()
        raw = os.pread(fd, int(self.lengths[idx]), int(self.offsets[idx]))
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Corrupt JPEG at index {idx} in {self.blob_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (112, 112):
            rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Synchronized horizontal flip before splitting into clean vs dark views
        flipped = bool(self.augment and rng.random() < 0.5)
        if flipped:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])

        if self.augment and self.clean_jitter_sigma > 0.0:
            clean_112 = self._warp_prealigned_reflect(rgb, self.clean_jitter_sigma, 0.0, 0.0, rng)
        else:
            clean_112 = rgb

        if self.augment and rng.random() < self.jitter_prob:
            sigma = float(rng.uniform(*self.kpt_jitter_sigma_range))
            dark_src_112 = self._warp_prealigned_reflect(
                rgb, sigma, self.scale_jitter, self.angle_jitter_deg, rng
            )
        else:
            mild_sigma = float(rng.uniform(0.3, 0.8)) if self.augment else 0.5
            dark_src_112 = self._warp_prealigned_reflect(rgb, mild_sigma, 0.01, 1.0, rng)

        kpts_norm = (ARCFACE_DST_112 / 112.0).astype(np.float32)
        item: Dict[str, Any] = {
            "clean_u8": torch.from_numpy(np.ascontiguousarray(clean_112.transpose(2, 0, 1))),
            "dark_src_u8": torch.from_numpy(np.ascontiguousarray(dark_src_112.transpose(2, 0, 1))),
            "kpts_norm": torch.from_numpy(kpts_norm),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }
        t_embs = self._ensure_teacher_mmap()
        if t_embs is not None:
            flip_idx = 1 if flipped else 0
            item["teacher_emb"] = torch.from_numpy(np.array(t_embs[idx, flip_idx], dtype=np.float32))
        return item


class TwinViewRecognitionDataset(Dataset):
    """Loads ID face samples from JSONL manifest and outputs CPU `uint8` twin views (`clean_u8`, `dark_src_u8`).

    `clean_u8`: normal-light aligned 112x112 (with <=1px alignment noise + random H-flip).
    `dark_src_u8`: aligned with empirical Phase-1 detector 5-pt jitter (sigma=1.2~2.5 px + scale/angle),
                   ready for GPU PhysicalLowLightDegrader(mode="recognizer").
    """

    def __init__(
        self,
        manifest_jsonl: Path,
        kpt_jitter_sigma_range: Tuple[float, float] = (1.2, 2.5),
        scale_jitter: float = 0.04,
        angle_jitter_deg: float = 4.0,
        augment: bool = True,
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
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        rng = np.random.default_rng() if self.augment else np.random.default_rng(self.seed + idx * 131)
        bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["image_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if "kpts" in row and row["kpts"] is not None:
            kpts5 = np.asarray(row["kpts"], dtype=np.float32).reshape(5, 2)
        else:
            if rgb.shape[:2] != (112, 112):
                rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
            kpts5 = ARCFACE_DST_112.copy()

        # Random horizontal flip applied consistently to both views
        if self.augment and rng.random() < 0.5:
            w = rgb.shape[1]
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            kpts5 = kpts5.copy()
            kpts5[:, 0] = w - kpts5[:, 0]
            kpts5[[0, 1]] = kpts5[[1, 0]]
            kpts5[[3, 4]] = kpts5[[4, 3]]

        clean_112, _ = align_face_112(rgb, kpts5, jitter_sigma_px=0.3 if self.augment else 0.0, rng=rng)
        sigma = float(rng.uniform(*self.kpt_jitter_sigma_range))
        dark_src_112, _ = align_face_112(
            rgb,
            kpts5,
            jitter_sigma_px=sigma,
            scale_jitter=self.scale_jitter,
            angle_jitter_deg=self.angle_jitter_deg,
            rng=rng,
        )

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
