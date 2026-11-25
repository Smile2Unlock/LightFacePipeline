#!/usr/bin/env python3
"""Precompute frozen CVLface ViT-Base KP-RPE embeddings (orig + hflip) and exact identity centers."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common.contracts import ARCFACE_DST_112
from common.runtime import get_device, seed_everything
from recognizer.teacher_cvlface import CVLfaceTeacherWrapper


class _SequentialBlobDataset(Dataset):
    """Worker-safe reader returning raw `[112, 112]` orig and horizontally flipped uint8 RGB."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.blob_path = self.data_dir / "train_images.blob"
        self.offsets = np.load(self.data_dir / "offsets.npy")
        self.lengths = np.load(self.data_dir / "lengths.npy")
        self.labels = np.load(self.data_dir / "labels.npy")
        self._fd: int | None = None
        self._fd_pid: int | None = None

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        fd = self._ensure_open()
        raw = os.pread(fd, int(self.lengths[idx]), int(self.offsets[idx]))
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Corrupt JPEG at idx={idx}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (112, 112):
            rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        flip = np.ascontiguousarray(rgb[:, ::-1, :])
        return (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
            torch.from_numpy(flip.transpose(2, 0, 1)),
            int(self.labels[idx]),
        )


def precompute_teacher(
    data_dir: Path,
    teacher_dir: Path,
    batch_size: int = 128,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> Dict[str, object]:
    seed_everything(42)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    num_classes = int(meta["num_classes"])
    ds = _SequentialBlobDataset(data_dir)
    n_total = min(len(ds), max_samples) if max_samples is not None else len(ds)
    if max_samples is not None and max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, range(max_samples))  # type: ignore[assignment]

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    teacher = CVLfaceTeacherWrapper(str(teacher_dir), offline_proxy=False, use_amp=True).to(device).eval()

    kpts_orig_112 = (ARCFACE_DST_112 / 112.0).astype(np.float32)
    kpts_flip_112 = kpts_orig_112.copy()
    kpts_flip_112[:, 0] = 1.0 - kpts_flip_112[:, 0]
    kpts_flip_112[[0, 1]] = kpts_flip_112[[1, 0]]
    kpts_flip_112[[3, 4]] = kpts_flip_112[[4, 3]]
    kp_o_t = torch.from_numpy(kpts_orig_112).to(device).unsqueeze(0)
    kp_f_t = torch.from_numpy(kpts_flip_112).to(device).unsqueeze(0)

    emb_path = data_dir / "teacher_embs_fp16.npy"
    ctr_path = data_dir / "teacher_centers_fp32.npy"
    embs_mm = np.lib.format.open_memmap(str(emb_path), mode="w+", dtype=np.float16, shape=(n_total, 2, 512))
    center_sums = np.zeros((num_classes, 512), dtype=np.float64)
    class_counts = np.zeros((num_classes,), dtype=np.int64)

    t0 = time.perf_counter()
    cursor = 0
    flip_cos_sum = 0.0

    with torch.no_grad():
        for orig_u8, flip_u8, lbls in tqdm(loader, desc="Precomputing CVLface Teacher"):
            bsz = orig_u8.shape[0]
            xo = (orig_u8.to(device, non_blocking=True).float() / 255.0 - 0.5) / 0.5
            xf = (flip_u8.to(device, non_blocking=True).float() / 255.0 - 0.5) / 0.5
            zo = teacher(xo, kp_o_t.expand(bsz, -1, -1))
            zf = teacher(xf, kp_f_t.expand(bsz, -1, -1))

            flip_cos_sum += float((zo * zf).sum(dim=1).sum().item())
            pair_np = torch.stack([zo, zf], dim=1).half().cpu().numpy()
            embs_mm[cursor : cursor + bsz] = pair_np

            z_avg = (zo.double() + zf.double()) * 0.5
            z_avg_np = z_avg.cpu().numpy()
            lbl_np = lbls.numpy()
            np.add.at(center_sums, lbl_np, z_avg_np)
            np.add.at(class_counts, lbl_np, 1)
            cursor += bsz

    embs_mm.flush()
    norms = np.linalg.norm(center_sums, axis=1, keepdims=True)
    centers_fp32 = np.zeros((num_classes, 512), dtype=np.float32)
    valid = (norms[:, 0] > 1e-8)
    centers_fp32[valid] = (center_sums[valid] / norms[valid]).astype(np.float32)
    np.save(str(ctr_path), centers_fp32)

    elapsed = time.perf_counter() - t0
    stats = {
        "n_samples": cursor,
        "num_classes": num_classes,
        "classes_with_samples": int((class_counts > 0).sum()),
        "mean_orig_flip_cos": round(flip_cos_sum / max(1, cursor), 5),
        "elapsed_sec": round(elapsed, 1),
        "throughput_imgs_per_sec": round((2.0 * cursor) / max(elapsed, 1e-3), 1),
        "teacher_embs_path": str(emb_path),
        "teacher_centers_path": str(ctr_path),
    }
    (data_dir / "teacher_meta.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("[precompute] Done:", json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute frozen CVLface Teacher embeddings + class centers")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/webface4m_local"))
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=Path("datasets/pretrained/cvlface_vit_base_kprpe_webface12m"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    precompute_teacher(
        data_dir=args.data_dir,
        teacher_dir=args.teacher_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
