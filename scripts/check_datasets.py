#!/usr/bin/env python3
"""Audit local & remote dataset availability, exact estimated download sizes (>1 GB flagged), and HF-mirror check."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


DATASET_CATALOG: List[Dict[str, Any]] = [
    {
        "phase": "Phase 1 (Detector)",
        "name": "DarkFace_Train_2021 (6k labeled + 9k unlabeled + 789 paired)",
        "approx_gb": 4.1,
        "over_1gb": True,
        "local_hint": "/home/ation_ciger/Projects/Smile2Unlock_v2/build/research/darkface/DarkFace_Train_2021.zip",
        "remote_target": "/workspace/LightFacePipeline/datasets/darkface/",
        "source": "Local Smile2Unlock_v2 archive (SHA256 verified) / Google Drive 10W3TDvEAlZfEt88hMxoEuRULr42bIV7s",
    },
    {
        "phase": "Phase 1 (Detector)",
        "name": "WIDER FACE Train + Val (with RetinaFace 5-pt labels)",
        "approx_gb": 3.5,
        "over_1gb": True,
        "local_hint": None,
        "remote_target": "/workspace/LightFacePipeline/datasets/widerface/",
        "source": "WIDER FACE official / HuggingFace mirror",
    },
    {
        "phase": "Phase 1 (Detector)",
        "name": "300W / WFLW / CelebA close-up 5-pt subset",
        "approx_gb": 1.6,
        "over_1gb": True,
        "local_hint": None,
        "remote_target": "/workspace/LightFacePipeline/datasets/closeup_5pt/",
        "source": "Academic benchmark / HF mirror",
    },
    {
        "phase": "Phase 1 (Detector)",
        "name": "COCO indoor hard negatives (~5k person-free frames)",
        "approx_gb": 0.9,
        "over_1gb": False,
        "local_hint": None,
        "remote_target": "/workspace/LightFacePipeline/datasets/coco_indoor_neg/",
        "source": "COCO 2017 val/train subset",
    },
    {
        "phase": "Phase 2 (Recognizer)",
        "name": "CASIA-WebFace (smoke, 490k imgs / 10k IDs)",
        "approx_gb": 4.0,
        "over_1gb": True,
        "local_hint": None,
        "remote_target": "/workspace/LightFacePipeline/datasets/casia_webface/",
        "source": "Academic face dataset",
    },
    {
        "phase": "Phase 2 (Recognizer)",
        "name": "WebFace4M (production, 4.2M imgs / 206k IDs)",
        "approx_gb": 22.0,
        "over_1gb": True,
        "local_hint": None,
        "remote_target": "/workspace/LightFacePipeline/datasets/webface4m/",
        "source": "InsightFace WebFace4M (research only)",
    },
    {
        "phase": "Phase 2 (Eval)",
        "name": "LFW / CFP-FP / AgeDB-30 / CPLFW / CALFW / IJB-C / TinyFace",
        "approx_gb": 5.5,
        "over_1gb": True,
        "local_hint": None,
        "remote_target": "/workspace/LightFacePipeline/datasets/rec_eval/",
        "source": "InsightFace eval packs",
    },
    {
        "phase": "Phase 3 (Liveness)",
        "name": "OULU-NPU + SiW + CASIA-FASD + Replay-Attack + MSU-MFSD",
        "approx_gb": 35.0,
        "over_1gb": True,
        "local_hint": None,
        "remote_target": "/workspace/LightFacePipeline/datasets/fas_videos/",
        "source": "Academic FAS video datasets",
    },
    {
        "phase": "Phase 3 (Liveness)",
        "name": "CelebA-Spoof (static Stage-1 spatial warmup only)",
        "approx_gb": 18.0,
        "over_1gb": True,
        "local_hint": None,
        "remote_target": "/workspace/LightFacePipeline/datasets/celeba_spoof/",
        "source": "CelebA-Spoof official release",
    },
]

TEACHER_MODELS: List[Dict[str, Any]] = [
    {
        "phase": "Phase 2 Teacher",
        "repo_id": "minchul/cvlface_adaface_vit_base_kprpe_webface12m",
        "approx_gb": 0.45,
        "over_1gb": False,
        "meta_file": "config.json",
    },
    {
        "phase": "Phase 3 Teacher",
        "repo_id": "facebook/dinov2-with-registers-base",
        "approx_gb": 0.35,
        "over_1gb": False,
        "meta_file": "config.json",
    },
]


def _probe_hf_metadata(repo_id: str, meta_file: str, hf_endpoint: str) -> Dict[str, Any]:
    url = f"{hf_endpoint.rstrip('/')}/{repo_id}/raw/main/{meta_file}"
    req = urllib.request.Request(url, headers={"User-Agent": "LightFacePipeline-Audit/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read(8192).decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            return {
                "reachable": True,
                "status": resp.status,
                "url": url,
                "keys_preview": sorted(list(parsed.keys()))[:10],
            }
    except Exception as exc:
        return {"reachable": False, "url": url, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata-only dataset & teacher connectivity audit")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_audit.json"))
    args = parser.parse_args()

    datasets_status = []
    for item in DATASET_CATALOG:
        entry = dict(item)
        remote_p = Path(item["remote_target"])
        entry["remote_exists"] = remote_p.exists() and any(remote_p.iterdir()) if remote_p.exists() else False
        local_p = Path(item["local_hint"]) if item["local_hint"] else None
        entry["local_exists"] = bool(local_p and local_p.exists())
        datasets_status.append(entry)

    teachers_status = []
    for tm in TEACHER_MODELS:
        entry = dict(tm)
        entry["metadata_probe"] = _probe_hf_metadata(tm["repo_id"], tm["meta_file"], args.hf_endpoint)
        teachers_status.append(entry)

    report = {
        "hf_endpoint": args.hf_endpoint,
        "datasets": datasets_status,
        "teachers": teachers_status,
        "total_estimated_dataset_gb": round(sum(d["approx_gb"] for d in DATASET_CATALOG), 2),
        "total_estimated_teacher_gb": round(sum(t["approx_gb"] for t in TEACHER_MODELS), 2),
        "over_1gb_items": [d["name"] for d in DATASET_CATALOG if d["over_1gb"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
