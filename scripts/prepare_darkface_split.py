#!/usr/bin/env python3
"""Generate deterministic 5,000 train / 1,000 val split for DarkFace labeled 6,000 images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List


DEFAULT_SOURCE = Path("/home/ation_ciger/Projects/Smile2Unlock_v2/build/research/darkface")
EXPECTED_ZIP_SHA256 = "d30abbdc7de2165b5293069a449d50764894f11622922e4769665689e9093a2e"


def _collect_from_dir(data_dir: Path) -> List[str]:
    img_dir = data_dir / "image"
    lbl_dir = data_dir / "label"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"Expected image/ and label/ inside {data_dir}")
    stems = sorted(
        (p.stem for p in img_dir.glob("*.png") if (lbl_dir / f"{p.stem}.txt").is_file()),
        key=lambda s: (0, int(s)) if s.isdigit() else (1, s),
    )
    return [f"{s}.png" for s in stems]


def _collect_from_zip(zip_path: Path) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    imgs = set()
    lbls = set()
    for name in names:
        if name.endswith(".png") and "/image/" in name:
            imgs.add(Path(name).stem)
        elif name.endswith(".txt") and "/label/" in name:
            lbls.add(Path(name).stem)
    stems = sorted(imgs & lbls, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    return [f"{s}.png" for s in stems]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed 5,000/1,000 DarkFace split manifest")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("splits/darkface_split_5000_1000.json"))
    parser.add_argument("--verify-zip-sha256", action="store_true")
    args = parser.parse_args()

    data_dir = args.source_dir / "data"
    zip_path = args.source_dir / "DarkFace_Train_2021.zip"

    if data_dir.is_dir():
        files = _collect_from_dir(data_dir)
    elif zip_path.is_file():
        files = _collect_from_zip(zip_path)
    else:
        raise FileNotFoundError(f"Neither {data_dir} nor {zip_path} exists")

    total = len(files)
    if total != args.train_count + args.val_count:
        raise ValueError(f"Expected {args.train_count + args.val_count} images, found {total}")

    rng = random.Random(args.seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    train_files = sorted(shuffled[: args.train_count], key=lambda f: int(Path(f).stem) if Path(f).stem.isdigit() else Path(f).stem)
    val_files = sorted(shuffled[args.train_count :], key=lambda f: int(Path(f).stem) if Path(f).stem.isdigit() else Path(f).stem)

    zip_sha = EXPECTED_ZIP_SHA256
    if args.verify_zip_sha256 and zip_path.is_file():
        zip_sha = _sha256_file(zip_path)
        if zip_sha != EXPECTED_ZIP_SHA256:
            raise ValueError(f"DarkFace zip SHA256 mismatch: {zip_sha}")

    canonical_payload = json.dumps({"train": train_files, "val": val_files}, separators=(",", ":"))
    split_sha = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

    out_obj: Dict[str, object] = {
        "dataset": "DarkFace_Train_2021_labeled_6000",
        "archive_sha256": zip_sha,
        "seed": args.seed,
        "total_images": total,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "split_sha256": split_sha,
        "train": train_files,
        "val": val_files,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} (train={len(train_files)}, val={len(val_files)}, split_sha256={split_sha[:12]}...)")


if __name__ == "__main__":
    main()
