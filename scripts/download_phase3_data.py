"""Download pretrained DINOv2-Reg Teacher + verified multi-domain real-video FAS datasets cleanly."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, snapshot_download


def _aria2_fetch(url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "--no-conf",
        "-c",
        "-x",
        "8",
        "-s",
        "8",
        "-k",
        "4M",
        "--max-tries=5",
        "--retry-wait=3",
        "--dir",
        str(out_path.parent),
        "--out",
        out_path.name,
        url,
    ]
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=False)
    return res.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0


def download_single_file(repo_id: str, filename: str, dest_dir: Path, repo_type: str = "dataset") -> Path:
    dest_path = dest_dir / filename
    if dest_path.is_file() and dest_path.stat().st_size > 1024:
        print(f"[skip] {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return dest_path

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
    print(f"[aria2] {repo_id}/{filename} -> {dest_path}", flush=True)
    if _aria2_fetch(url, dest_path):
        return dest_path

    print(f"[hf_hub fallback] {repo_id}/{filename}", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(dest_dir),
    )
    return dest_path


def download_filtered_repo(
    repo_id: str,
    dest_dir: Path,
    allow_patterns: List[str],
    max_workers: int = 4,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[snapshot] {repo_id} patterns={allow_patterns} -> {dest_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_dir),
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    return dest_dir


def main() -> None:
    root = Path("datasets")
    raw = root / "fas_raw"
    raw.mkdir(parents=True, exist_ok=True)

    # 1. Pretrained Teacher: facebook/dinov2-with-registers-base
    dino_dir = root / "pretrained" / "dinov2_reg_base"
    if not (dino_dir / "model.safetensors").is_file():
        print("[1/11] Downloading facebook/dinov2-with-registers-base ...", flush=True)
        snapshot_download(
            repo_id="facebook/dinov2-with-registers-base",
            repo_type="model",
            local_dir=str(dino_dir),
            max_workers=4,
        )
    else:
        print("[1/11] dinov2_reg_base already present.", flush=True)

    # 2. CASIA-FASD (66 MB archive)
    print("[2/11] Downloading akahana/anti-spoofing-casiafasd ...", flush=True)
    download_single_file("akahana/anti-spoofing-casiafasd", "casiafasd.tar.gz", raw / "casia_fasd")
    download_single_file("akahana/anti-spoofing-casiafasd", "train/train-00000-of-00001.parquet", raw / "casia_fasd")
    download_single_file("akahana/anti-spoofing-casiafasd", "test/train-00000-of-00001.parquet", raw / "casia_fasd")

    # 3. jonathansuru/anti_spoofing (321 MB, 332 videos + csv)
    print("[3/11] Downloading jonathansuru/anti_spoofing ...", flush=True)
    download_filtered_repo("jonathansuru/anti_spoofing", raw / "jonathansuru", ["data/*.mp4", "data/**/*.mp4", "*.csv"], max_workers=4)

    # 4. UniqueData paired Real + Replay Phone + Replay Monitors
    print("[4/11] Downloading UniqueData paired Real / Replay / Monitors ...", flush=True)
    download_single_file("UniqueData/anti-spoofing_Real", "data/data.tar.gz", raw / "uniquedata_real")
    download_single_file("UniqueData/anti-spoofing_Real", "data/anti-spoofing_Live.csv", raw / "uniquedata_real")
    download_single_file("UniqueData/anti-spoofing_replay", "data/videos.tar.gz", raw / "uniquedata_replay")
    download_single_file("UniqueData/anti-spoofing_replay", "data/anti-spoofing_replay.csv", raw / "uniquedata_replay")
    download_single_file("UniqueData/monitors-replay-attacks-dataset", "data/attacks.tar.gz", raw / "uniquedata_monitors")
    download_single_file("UniqueData/monitors-replay-attacks-dataset", "data/monitors-replay-attacks-dataset.csv", raw / "uniquedata_monitors")

    # 5. UniqueData webcam (70 videos across real/print/print_cut/outline/monitor/mask/silicone)
    print("[5/11] Downloading UniqueData/web-camera-face-liveness-detection ...", flush=True)
    download_filtered_repo("UniqueData/web-camera-face-liveness-detection", raw / "uniquedata_webcam", ["files/**/*", "*.csv"], max_workers=4)

    # 6. UniqueData mobile (40 videos across real/mask/mask_cut/outline - held-out cross-domain!)
    print("[6/11] Downloading UniqueData/on-device-face-liveness-detection ...", flush=True)
    download_filtered_repo("UniqueData/on-device-face-liveness-detection", raw / "uniquedata_mobile", ["files/**/*", "*.csv"], max_workers=4)

    # 7. UniqueData diverse Live videos (40 videos across Asian/Black/Caucasian/Hispanic)
    print("[7/11] Downloading UniqueData diverse live video datasets ...", flush=True)
    for sub, repo in [
        ("asian", "UniqueData/asian-people-liveness-detection-video-dataset"),
        ("black", "UniqueData/black-people-liveness-detection-video-dataset"),
        ("caucasian", "UniqueData/caucasian-people-liveness-detection-dataset"),
        ("hispanic", "UniqueData/hispanic-people-liveness-detection-video-dataset"),
    ]:
        download_filtered_repo(repo, raw / "uniquedata_diverse_live" / sub, ["files/**/*.mp4", "files/**/*.mov", "files/**/*.MOV", "*.csv"], max_workers=4)

    # 8. osama6/anti_spoofing_datasets video folders only (160 .mp4s)
    print("[8/11] Downloading osama6/anti_spoofing_datasets video splits ...", flush=True)
    download_filtered_repo(
        "osama6/anti_spoofing_datasets",
        raw / "osama6_videos",
        ["train/real_video/*", "train/attack/*", "test/real_video/*", "test/attack/*"],
        max_workers=4,
    )

    # 9. AxonData print + display/mobile replay + cutout + masks
    print("[9/11] Downloading AxonData/print-attack-dataset ...", flush=True)
    download_filtered_repo("AxonData/print-attack-dataset", raw / "axondata_print", ["Axon Labs Print sample/*"], max_workers=4)

    print("[10/11] Downloading AxonData/face-anti-spoofing-dataset attack subsets ...", flush=True)
    download_filtered_repo(
        "AxonData/face-anti-spoofing-dataset",
        raw / "axondata_multi",
        [
            "Replay_display_attacks/*",
            "Replay_mobile_attacks/*",
            "Cutout_attacks/*",
            "Silicone_mask/*",
            "Latex_mask/*",
        ],
        max_workers=4,
    )

    # 10. LCC_FASDCut.zip (2.65 GB)
    print("[11/11] Downloading namkuner/CelebA-Spoof-Dataset LCC_FASDCut.zip ...", flush=True)
    download_single_file("namkuner/CelebA-Spoof-Dataset", "LCC_FASDCut.zip", raw / "lcc_fasd")

    print("ALL_PHASE3_DOWNLOADS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
