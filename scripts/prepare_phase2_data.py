#!/usr/bin/env python3
"""Prepare Phase 2 local training (WebFace4M packed JPEG blob) and raw closed-loop LFW eval data."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_lfw_raw(out_dir: Path, workers: int = 8) -> Dict[str, int]:
    """Download `pairs.csv` and unique referenced 250x250 RGB images from `marcelohaps/lfw`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = Path(
        hf_hub_download(
            "marcelohaps/lfw",
            "pairs.csv",
            repo_type="dataset",
            local_dir=str(out_dir),
        )
    )
    rel_paths: Set[str] = set()
    n_pairs = 0
    with open(pairs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_pairs += 1
            rel_paths.add(row["image_a_path"].strip())
            rel_paths.add(row["image_b_path"].strip())

    # HF dataset stores images under train/images/... while pairs.csv lists images/...
    missing = [p for p in sorted(rel_paths) if not (out_dir / "train" / p).is_file()]
    print(f"[lfw_raw] pairs={n_pairs}, unique_images={len(rel_paths)}, missing={len(missing)}")

    if missing:
        def _dl_one(rel: str) -> str:
            hf_hub_download(
                "marcelohaps/lfw",
                f"train/{rel}",
                repo_type="dataset",
                local_dir=str(out_dir),
            )
            return rel

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_dl_one, rel) for rel in missing]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading lfw_raw"):
                fut.result()

    summary = {"pairs": n_pairs, "unique_images": len(rel_paths)}
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def download_webface4m_shards(shards_dir: Path, num_shards: int = 40, workers: int = 4) -> List[Path]:
    """Download pre-shuffled WebFace4M `.tar.gz` shards from `gaunernst/webface4m-wds-gz`."""
    shards_dir.mkdir(parents=True, exist_ok=True)
    filenames = [f"webface4m-{i:04d}.tar.gz" for i in range(num_shards)]
    pending = [fn for fn in filenames if not (shards_dir / fn).is_file()]
    print(f"[webface4m] target_shards={num_shards}, pending_download={len(pending)}")

    if pending:
        def _dl_shard(fn: str) -> Path:
            p = hf_hub_download(
                "gaunernst/webface4m-wds-gz",
                fn,
                repo_type="dataset",
                local_dir=str(shards_dir),
            )
            return Path(p)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_dl_shard, fn) for fn in pending]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading WebFace4M shards"):
                fut.result()

    return [shards_dir / fn for fn in filenames]


def _scan_shard_classes(shard_path: Path) -> Counter:
    counts: Counter = Counter()
    with tarfile.open(shard_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".cls"):
                continue
            raw_cls = int(tf.extractfile(member).read().decode("ascii").strip())
            counts[raw_cls] += 1
    return counts


def pack_webface4m_top_classes(
    shard_paths: List[Path],
    out_dir: Path,
    top_k_classes: int = 50000,
) -> Dict[str, object]:
    """Select `top_k_classes` densest identities across shards and pack into sequential JPEG blob."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: count identity frequencies across all shards
    total_counts: Counter = Counter()
    for sp in tqdm(shard_paths, desc="Pass 1: counting IDs"):
        total_counts.update(_scan_shard_classes(sp))

    most_common = total_counts.most_common(top_k_classes)
    selected_classes = sorted(cls_id for cls_id, _ in most_common)
    cls_remap: Dict[int, int] = {raw_id: new_id for new_id, raw_id in enumerate(selected_classes)}
    min_per_class = most_common[-1][1] if most_common else 0
    max_per_class = most_common[0][1] if most_common else 0
    expected_samples = sum(c for _, c in most_common)
    print(
        f"[pack] total_ids_seen={len(total_counts)}, selected={len(selected_classes)}, "
        f"min_per_class={min_per_class}, max_per_class={max_per_class}, expected_samples={expected_samples}"
    )

    # Pass 2: stream selected JPEG bytes sequentially into train_images.blob
    blob_path = out_dir / "train_images.blob"
    offsets: List[int] = []
    lengths: List[int] = []
    labels: List[int] = []

    cursor = 0
    with open(blob_path, "wb") as blob_f:
        for sp in tqdm(shard_paths, desc="Pass 2: packing JPEGs"):
            cls_buf: Dict[str, int] = {}
            jpg_buf: Dict[str, bytes] = {}
            with tarfile.open(sp, "r:gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    name = member.name
                    stem = ""
                    if name.endswith(".cls"):
                        stem = name[:-4]
                        raw_cls = int(tf.extractfile(member).read().decode("ascii").strip())
                        cls_buf[stem] = raw_cls
                    elif name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png"):
                        stem = name.rsplit(".", 1)[0]
                        jpg_buf[stem] = tf.extractfile(member).read()

                    if stem and stem in cls_buf and stem in jpg_buf:
                        raw_cls = cls_buf.pop(stem)
                        img_bytes = jpg_buf.pop(stem)
                        mapped = cls_remap.get(raw_cls)
                        if mapped is not None:
                            blob_f.write(img_bytes)
                            n_bytes = len(img_bytes)
                            offsets.append(cursor)
                            lengths.append(n_bytes)
                            labels.append(mapped)
                            cursor += n_bytes

    np.save(out_dir / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out_dir / "lengths.npy", np.asarray(lengths, dtype=np.int32))
    np.save(out_dir / "labels.npy", np.asarray(labels, dtype=np.int32))

    n_samples = len(labels)
    num_classes = len(selected_classes)
    meta: Dict[str, object] = {
        "num_samples": n_samples,
        "num_classes": num_classes,
        "total_ids_across_shards": len(total_counts),
        "min_per_class": int(min_per_class),
        "max_per_class": int(max_per_class),
        "mean_per_class": round(float(n_samples) / max(1, num_classes), 2),
        "shards_used": len(shard_paths),
        "blob_size_mb": round(float(cursor) / (1024 ** 2), 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("[pack] Done:", json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase 2 datasets locally")
    parser.add_argument("--lfw-raw-dir", type=Path, default=Path("datasets/rec_eval/lfw_raw"))
    parser.add_argument("--webface4m-dir", type=Path, default=Path("datasets/webface4m_local"))
    parser.add_argument("--num-shards", type=int, default=40)
    parser.add_argument("--top-k-classes", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-lfw-raw", action="store_true")
    parser.add_argument("--skip-webface4m", action="store_true")
    args = parser.parse_args()

    if not args.skip_lfw_raw:
        download_lfw_raw(args.lfw_raw_dir, workers=args.workers)
    if not args.skip_webface4m:
        shards = download_webface4m_shards(
            args.webface4m_dir / "shards",
            num_shards=args.num_shards,
            workers=min(4, args.workers),
        )
        pack_webface4m_top_classes(shards, args.webface4m_dir, top_k_classes=args.top_k_classes)


if __name__ == "__main__":
    main()
