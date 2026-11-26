"""Prepare Phase 3 multi-domain real-video FAS frame cache, run Phase-1 detector bboxes, and emit disjoint clip manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tarfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from common.align import letterbox_rgb, unletterbox_coords
from detector.yolo11_face import LCFYOLO11Face5pt


VIDEO_EXTS = {".mp4", ".mov", ".3gp", ".avi", ".mkv"}
MAX_FRAMES_PER_VIDEO = 20
MIN_VALID_FRAMES = 8


def _file_sig(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(65536))
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def _extract_tar_once(tar_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extracted_ok"
    if marker.is_file():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir, filter="data")
    marker.write_text("ok\n", encoding="utf-8")


def _extract_video_frames(
    video_path: Path,
    out_dir: Path,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
    max_side: int = 640,
) -> List[Path]:
    done_flag = out_dir / ".done.json"
    if done_flag.is_file():
        saved = [out_dir / f for f in json.loads(done_flag.read_text(encoding="utf-8"))]
        if all(p.is_file() for p in saved) and len(saved) >= MIN_VALID_FRAMES:
            return saved
        if done_flag.is_file() and len(saved) < MIN_VALID_FRAMES:
            return saved

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        done_flag.write_text("[]", encoding="utf-8")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_raw: List[np.ndarray] = []
    if total >= max_frames * 2:
        idxs = np.linspace(0, total - 1, max_frames).round().astype(int).tolist()
        seen = set()
        for fi in idxs:
            if fi in seen:
                continue
            seen.add(fi)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                frames_raw.append(bgr)
    else:
        all_bgr: List[np.ndarray] = []
        while len(all_bgr) < 600:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            all_bgr.append(bgr)
        if len(all_bgr) > max_frames:
            idxs = np.linspace(0, len(all_bgr) - 1, max_frames).round().astype(int).tolist()
            frames_raw = [all_bgr[i] for i in idxs]
        else:
            frames_raw = all_bgr
    cap.release()

    names: List[str] = []
    for idx, bgr in enumerate(frames_raw):
        h, w = bgr.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        fn = f"f_{idx:04d}.jpg"
        cv2.imwrite(str(out_dir / fn), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        names.append(fn)

    done_flag.write_text(json.dumps(names), encoding="utf-8")
    return [out_dir / f for f in names]


@torch.no_grad()
def _detect_bboxes_batch(
    detector: LCFYOLO11Face5pt,
    frame_paths: List[Path],
    device: torch.device,
    score_th: float = 0.25,
    batch_size: int = 32,
) -> List[Optional[List[float]]]:
    """Run Phase-1 LCF+YOLO11n-5pt detector across frame list, returning [x1,y1,x2,y2] or None."""
    results: List[Optional[List[float]]] = []
    for i in range(0, len(frame_paths), batch_size):
        chunk = frame_paths[i : i + batch_size]
        imgs_t = []
        metas = []
        valid_mask = []
        for p in chunk:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                valid_mask.append(False)
                metas.append(((0, 0), 1.0, (0.0, 0.0)))
                continue
            oh, ow = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lb, scale, pad = letterbox_rgb(rgb, (640, 640))
            imgs_t.append(torch.from_numpy(lb.transpose(2, 0, 1)).float().div_(255.0))
            metas.append(((oh, ow), scale, pad))
            valid_mask.append(True)

        if imgs_t:
            x = torch.stack(imgs_t, dim=0).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                out = detector(x)
            boxes = out["boxes_xyxy"].float().cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
        else:
            boxes, scores = np.empty((0, 0, 4)), np.empty((0, 0))

        ptr = 0
        for ok, (orig_hw, scale, pad) in zip(valid_mask, metas):
            if not ok:
                results.append(None)
                continue
            sc_row = scores[ptr]
            top_i = int(np.argmax(sc_row))
            top_s = float(sc_row[top_i])
            if top_s < score_th:
                results.append(None)
            else:
                b_orig, _ = unletterbox_coords(boxes[ptr, top_i : top_i + 1], None, scale, pad, orig_hw)
                x1, y1, x2, y2 = [float(v) for v in b_orig[0]]
                if (x2 - x1) < 12 or (y2 - y1) < 12:
                    results.append(None)
                else:
                    results.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
            ptr += 1
    return results


def _repair_track_bboxes(
    frame_paths: List[Path],
    raw_boxes: List[Optional[List[float]]],
    cropped_fallback: bool = False,
) -> Tuple[List[Path], List[List[float]]]:
    """Filter/interpolate missing boxes within a video frame sequence."""
    n = len(frame_paths)
    if cropped_fallback:
        out_p, out_b = [], []
        for p, b in zip(frame_paths, raw_boxes):
            if b is None:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                h, w = im.shape[:2]
                b = [round(0.14 * w, 2), round(0.14 * h, 2), round(0.86 * w, 2), round(0.86 * h, 2)]
            out_p.append(p)
            out_b.append(b)
        return out_p, out_b

    # Interpolate gaps <= 2 frames; drop missing frames otherwise
    repaired: List[Optional[List[float]]] = list(raw_boxes)
    valid_idxs = [i for i, b in enumerate(raw_boxes) if b is not None]
    if len(valid_idxs) < MIN_VALID_FRAMES:
        return [], []
    for k in range(len(valid_idxs) - 1):
        i0, i1 = valid_idxs[k], valid_idxs[k + 1]
        gap = i1 - i0 - 1
        if 1 <= gap <= 2:
            b0 = np.asarray(raw_boxes[i0], dtype=np.float32)
            b1 = np.asarray(raw_boxes[i1], dtype=np.float32)
            for g in range(1, gap + 1):
                alpha = float(g) / float(gap + 1)
                bi = ((1.0 - alpha) * b0 + alpha * b1).tolist()
                repaired[i0 + g] = [round(float(v), 2) for v in bi]

    out_p, out_b = [], []
    for p, b in zip(frame_paths, repaired):
        if b is not None:
            out_p.append(p)
            out_b.append(b)
    return out_p, out_b


def _extract_lcc_sequences(zip_path: Path, out_root: Path) -> List[Dict[str, Any]]:
    seqs: Dict[Tuple[str, str, str], List[Tuple[int, str]]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        for n in zf.namelist():
            if not n.endswith((".png", ".jpg")):
                continue
            parts = n.split("/")
            if len(parts) < 4:
                continue
            split = parts[1].replace("LCC_FASDCut_", "")
            if split == "evaluation":
                # LCC_FASDCut_evaluation filenames are randomized without video IDs -> skip cleanly!
                continue
            cls = parts[2]
            stem = parts[3].rsplit(".", 1)[0]
            if "_" not in stem:
                continue
            vid, fstr = stem.rsplit("_", 1)
            fnum = int(fstr) if fstr.isdigit() else 0
            seqs[(split, cls, vid)].append((fnum, n))

        records: List[Dict[str, Any]] = []
        for (raw_split, cls, vid), items in sorted(seqs.items()):
            if len(items) < MIN_VALID_FRAMES:
                continue
            items.sort(key=lambda x: x[0])
            # Extract frames to datasets/fas_frames/lcc_fasd/<raw_split>/<cls>/<vid>/
            seq_dir = out_root / "lcc_fasd" / raw_split / cls / vid
            seq_dir.mkdir(parents=True, exist_ok=True)
            fpaths: List[Path] = []
            for idx, (_, member) in enumerate(items[:MAX_FRAMES_PER_VIDEO]):
                dst = seq_dir / f"f_{idx:04d}.jpg"
                if not dst.is_file():
                    buf = np.frombuffer(zf.read(member), dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                fpaths.append(dst)
            if len(fpaths) < MIN_VALID_FRAMES:
                continue

            # Subject key: YOUTUBE_id70_s0 -> YOUTUBE_id70
            subj_m = re.match(r"^(.*?_id\d+)", vid)
            subj_id = f"lcc_{subj_m.group(1)}" if subj_m else f"lcc_{vid}"
            if raw_split == "training":
                split_dst = "train"
            else:
                # Subject-disjoint 50/50 val/test from development set
                h_val = int(hashlib.md5(subj_id.encode()).hexdigest()[:8], 16) % 2
                split_dst = "val" if h_val == 0 else "test"

            records.append(
                {
                    "clip_id": f"lcc_{raw_split}_{cls}_{vid}",
                    "source": "lcc_fasd",
                    "split": split_dst,
                    "label": 0 if cls == "real" else 2,
                    "fine_label_known": True if cls == "real" else False,
                    "attack_group": "live" if cls == "real" else "generic_spoof",
                    "frame_paths_p": fpaths,
                    "cropped_fallback": True,
                }
            )
    return records


def _extract_casia_sequences(tar_path: Path, out_root: Path) -> List[Dict[str, Any]]:
    seqs: Dict[Tuple[str, str], List[Tuple[int, str, str]]] = defaultdict(list)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile() and "/color/" in m.name and m.name.endswith(".jpg")}
        for mname in members:
            base = mname.rsplit("/", 1)[-1]
            if ".avi_" not in base:
                continue
            vid, rest = base.split(".avi_", 1)
            fnum_str = rest.split("_", 1)[0]
            fnum = int(fnum_str) if fnum_str.isdigit() else 0
            raw_split = "test" if "test_img" in mname else "train"
            is_real = "_real." in base
            # CASIA protocol: _1, _2, _HR_1 = real; _3, _4, _HR_2 = wrapped print (1); _5, _6, _HR_3 = cut print (1); _7, _8, _HR_4 = tablet replay (2)
            if is_real:
                lbl, grp = 0, "live"
            elif vid.endswith(("_7", "_8", "_HR_4")):
                lbl, grp = 2, "replay"
            else:
                lbl, grp = 1, "print"
            seqs[(raw_split, vid)].append((fnum, mname, f"{lbl}:{grp}"))

        records: List[Dict[str, Any]] = []
        for (raw_split, vid), items in sorted(seqs.items()):
            if len(items) < MIN_VALID_FRAMES:
                continue
            items.sort(key=lambda x: x[0])
            lbl_str, grp = items[0][2].split(":", 1)
            lbl = int(lbl_str)
            seq_dir = out_root / "casia_fasd" / raw_split / vid
            seq_dir.mkdir(parents=True, exist_ok=True)
            fpaths: List[Path] = []
            for idx, (_, mname, _) in enumerate(items[:MAX_FRAMES_PER_VIDEO]):
                dst = seq_dir / f"f_{idx:04d}.jpg"
                if not dst.is_file():
                    raw_bytes = tf.extractfile(members[mname]).read()
                    bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                fpaths.append(dst)
            if len(fpaths) < MIN_VALID_FRAMES:
                continue
            records.append(
                {
                    "clip_id": f"casia_{raw_split}_{vid}",
                    "source": "casia_fasd_zeroshot",
                    "split": "test",  # 100% Held-out Zero-Shot Cross-Domain Benchmark!
                    "label": lbl,
                    "fine_label_known": True,
                    "attack_group": grp,
                    "frame_paths_p": fpaths,
                    "cropped_fallback": False,
                }
            )
    return records


def _stable_split_ratio(key: str, val_frac: float = 0.15, test_frac: float = 0.15) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def _extract_video_job(job: Tuple[int, str, str, str]) -> Tuple[int, List[str]]:
    cv2.setNumThreads(1)
    idx, video_path, out_dir, _ = job
    paths = _extract_video_frames(Path(video_path), Path(out_dir))
    return idx, [str(p) for p in paths]


def main() -> None:
    raw = Path("datasets/fas_raw")
    frames_root = Path("datasets/fas_frames")
    frames_root.mkdir(parents=True, exist_ok=True)

    # Extract tarballs once
    _extract_tar_once(raw / "uniquedata_real" / "data" / "data.tar.gz", raw / "uniquedata_real" / "extracted")
    _extract_tar_once(raw / "uniquedata_replay" / "data" / "videos.tar.gz", raw / "uniquedata_replay" / "extracted")
    _extract_tar_once(raw / "uniquedata_monitors" / "data" / "attacks.tar.gz", raw / "uniquedata_monitors" / "extracted")

    candidates: List[Dict[str, Any]] = []
    video_jobs: List[Tuple[int, str, str, str]] = []
    video_meta: List[Dict[str, Any]] = []
    seen_sigs = set()

    def _add_video(
        clip_id: str,
        source: str,
        split: str,
        label: int,
        fine_known: bool,
        attack_group: str,
        video_path: Path,
    ) -> None:
        sig = _file_sig(video_path)
        if sig in seen_sigs:
            return
        seen_sigs.add(sig)
        idx = len(video_jobs)
        video_jobs.append((idx, str(video_path), str(frames_root / source / clip_id), clip_id))
        video_meta.append(
            {
                "clip_id": clip_id,
                "source": source,
                "split": split,
                "label": label,
                "fine_label_known": fine_known,
                "attack_group": attack_group,
                "cropped_fallback": False,
            }
        )

    print("[1/4] Extracting frames from video datasets...", flush=True)

    # 1) jonathansuru (official train/val/test)
    for split in ("train", "val", "test"):
        for cls_dir, lbl, fk, grp in [("real", 0, True, "live"), ("attack", 2, False, "generic_spoof")]:
            for vp in sorted((raw / "jonathansuru" / "data" / split / cls_dir).glob("*")):
                if vp.suffix.lower() in VIDEO_EXTS:
                    _add_video(f"js_{split}_{cls_dir}_{vp.stem}", "jonathansuru", split, lbl, fk, grp, vp)

    # 2) osama6_videos (official test -> test; train -> 82% train / 18% val)
    for cls_dir, lbl, fk, grp in [("real_video", 0, True, "live"), ("attack", 2, False, "generic_spoof")]:
        for vp in sorted((raw / "osama6_videos" / "test" / cls_dir).glob("*")):
            if vp.suffix.lower() in VIDEO_EXTS:
                _add_video(f"os6_test_{cls_dir}_{vp.stem}", "osama6", "test", lbl, fk, grp, vp)
        for vp in sorted((raw / "osama6_videos" / "train" / cls_dir).glob("*")):
            if vp.suffix.lower() in VIDEO_EXTS:
                sp = _stable_split_ratio(f"os6_{vp.stem}", val_frac=0.18, test_frac=0.0)
                _add_video(f"os6_tr_{cls_dir}_{vp.stem}", "osama6", sp, lbl, fk, grp, vp)

    # 3) uniquedata_paired (subject-linked ud_real + ud_replay via anti-spoofing_replay.csv + ud_monitors)
    replay_to_subject: Dict[str, str] = {}
    rep_csv = raw / "uniquedata_replay" / "data" / "anti-spoofing_replay.csv"
    if rep_csv.is_file():
        with open(rep_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                link_dir = str(row.get("link", "")).split("/")[0]
                subj = str(row.get("live_video_id", "")).strip()
                if link_dir and subj:
                    replay_to_subject[link_dir] = subj

    for vp in sorted((raw / "uniquedata_real" / "extracted").rglob("*")):
        if vp.suffix.lower() in VIDEO_EXTS:
            subj = vp.parent.name
            sp = _stable_split_ratio(f"ud_subj_{subj}", val_frac=0.20, test_frac=0.20)
            _add_video(f"ud_real_{subj}", "uniquedata_paired", sp, 0, True, "live", vp)

    for vp in sorted((raw / "uniquedata_replay" / "extracted").rglob("*")):
        if vp.suffix.lower() in VIDEO_EXTS:
            link_dir = vp.parent.name
            subj = replay_to_subject.get(link_dir, link_dir)
            sp = _stable_split_ratio(f"ud_subj_{subj}", val_frac=0.20, test_frac=0.20)
            _add_video(f"ud_rep_{link_dir}", "uniquedata_paired", sp, 2, True, "replay", vp)

    for vp in sorted((raw / "uniquedata_monitors" / "extracted").rglob("*")):
        if vp.suffix.lower() in VIDEO_EXTS:
            sp = _stable_split_ratio(f"ud_mon_{vp.stem}", val_frac=0.20, test_frac=0.20)
            _add_video(f"ud_mon_{vp.stem}", "uniquedata_paired", sp, 2, True, "replay", vp)

    # 4) uniquedata_webcam (stratified 60/20/20 per folder)
    webcam_map = {
        "real": (0, "live"),
        "print": (1, "print"),
        "print_cut": (1, "print"),
        "outline": (1, "print"),
        "monitor": (2, "replay"),
        "mask": (3, "mask_cutout"),
        "silicone": (3, "mask_cutout"),
    }
    for sub, (lbl, grp) in webcam_map.items():
        vids = sorted([p for p in (raw / "uniquedata_webcam" / "files" / sub).glob("*") if p.suffix.lower() in VIDEO_EXTS])
        for i, vp in enumerate(vids):
            sp = "test" if i % 5 == 0 else ("val" if i % 5 == 1 else "train")
            _add_video(f"ud_wc_{sub}_{vp.stem}", "uniquedata_webcam", sp, lbl, True, grp, vp)

    # 5) uniquedata_mobile (100% Held-out Zero-Shot Cross-Domain Benchmark!)
    mobile_map = {
        "real": (0, "live"),
        "mask": (3, "mask_cutout"),
        "mask_cut": (3, "mask_cutout"),
        "outline": (1, "print"),
    }
    for sub, (lbl, grp) in mobile_map.items():
        for vp in sorted((raw / "uniquedata_mobile" / "files" / sub).glob("*")):
            if vp.suffix.lower() in VIDEO_EXTS:
                _add_video(f"ud_mob_{sub}_{vp.stem}", "uniquedata_mobile_zeroshot", "test", lbl, True, grp, vp)

    # 6) uniquedata_diverse_live
    for eth in ("asian", "black", "caucasian", "hispanic"):
        vids = sorted([p for p in (raw / "uniquedata_diverse_live" / eth / "files").rglob("*") if p.suffix.lower() in VIDEO_EXTS])
        for i, vp in enumerate(vids):
            sp = "test" if i % 7 == 0 else ("val" if i % 7 == 1 else "train")
            _add_video(f"ud_div_{eth}_{vp.parent.name}", "uniquedata_diverse_live", sp, 0, True, "live", vp)

    # 7) axondata print + multi
    axon_groups: List[Tuple[str, Path, int, str]] = [
        ("print", raw / "axondata_print" / "Axon Labs Print sample", 1, "print"),
        ("cutout", raw / "axondata_multi" / "Cutout_attacks", 1, "print"),
        ("disp", raw / "axondata_multi" / "Replay_display_attacks", 2, "replay"),
        ("mob", raw / "axondata_multi" / "Replay_mobile_attacks", 2, "replay"),
        ("silicone", raw / "axondata_multi" / "Silicone_mask", 3, "mask_cutout"),
        ("latex", raw / "axondata_multi" / "Latex_mask", 3, "mask_cutout"),
    ]
    for tag, folder, lbl, grp in axon_groups:
        vids = sorted([p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS])
        for i, vp in enumerate(vids):
            sp = "test" if i % 6 == 0 else ("val" if i % 6 == 1 else "train")
            _add_video(f"axon_{tag}_{vp.parent.name}_{vp.stem}", "axondata", sp, lbl, True, grp, vp)

    print(f"[1/4] Parallel extracting frames for {len(video_jobs)} videos (workers=6)...", flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_extract_video_job, j) for j in video_jobs]
        done_cnt = 0
        for fut in as_completed(futs):
            idx, f_strs = fut.result()
            done_cnt += 1
            if done_cnt % 100 == 0 or done_cnt == len(video_jobs):
                print(f"  extracted {done_cnt}/{len(video_jobs)} videos", flush=True)
            if len(f_strs) >= MIN_VALID_FRAMES:
                meta = dict(video_meta[idx])
                meta["frame_paths_p"] = [Path(p) for p in f_strs]
                candidates.append(meta)

    print(f"[2/4] Extracting LCC_FASD & CASIA-FASD ordered frame sequences...", flush=True)
    candidates.extend(_extract_lcc_sequences(raw / "lcc_fasd" / "LCC_FASDCut.zip", frames_root))
    candidates.extend(_extract_casia_sequences(raw / "casia_fasd" / "casiafasd.tar.gz", frames_root))
    print(f"Total candidate video sequences before Phase-1 detection: {len(candidates)}", flush=True)

    # 8) Run Phase-1 LCFYOLO11Face5pt detector across all frames
    print("[3/4] Running locked Phase-1 detector on GPU to locate per-frame face bboxes...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = LCFYOLO11Face5pt().to(device).eval()
    ck = torch.load("runs/detector_yolo11n_5pt_lcf/stage2_v2/best.pt", map_location="cpu", weights_only=False)
    detector.load_state_dict(ck.get("ema", ck.get("model", ck)), strict=True)

    all_frame_paths: List[Path] = []
    offsets: List[Tuple[int, int]] = []
    for rec in candidates:
        s = len(all_frame_paths)
        all_frame_paths.extend(rec["frame_paths_p"])
        offsets.append((s, len(all_frame_paths)))

    print(f"Total extracted frames to run detector on: {len(all_frame_paths)}", flush=True)
    all_raw_boxes = _detect_bboxes_batch(detector, all_frame_paths, device=device, score_th=0.25, batch_size=32)

    print("[4/4] Building & saving manifests...", flush=True)
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    dropped = 0
    for rec, (s, e) in zip(candidates, offsets):
        raw_boxes = all_raw_boxes[s:e]
        valid_f, valid_b = _repair_track_bboxes(
            rec["frame_paths_p"], raw_boxes, cropped_fallback=rec["cropped_fallback"]
        )
        if len(valid_f) < MIN_VALID_FRAMES:
            dropped += 1
            continue
        f_strs = [str(p) for p in valid_f]
        if int(rec["label"]) == 0 and len(set(f_strs)) < 2:
            dropped += 1
            continue
        by_split[rec["split"]].append(
            {
                "clip_id": rec["clip_id"],
                "source": rec["source"],
                "label": int(rec["label"]),
                "fine_label_known": bool(rec["fine_label_known"]),
                "attack_group": rec["attack_group"],
                "frame_paths": f_strs,
                "bboxes": valid_b,
            }
        )

    man_dir = Path("datasets/manifests")
    man_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"dropped_short_or_nodet": dropped, "splits": {}}
    for sp, rows in by_split.items():
        out_p = man_dir / f"fas_video_clips_{sp}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        src_cnt = defaultdict(lambda: {"live": 0, "spoof": 0})
        cls_cnt = defaultdict(int)
        for r in rows:
            src_cnt[r["source"]]["live" if r["label"] == 0 else "spoof"] += 1
            cls_cnt[r["attack_group"]] += 1
        summary["splits"][sp] = {
            "total": len(rows),
            "live": sum(1 for r in rows if r["label"] == 0),
            "spoof": sum(1 for r in rows if r["label"] != 0),
            "by_attack_group": dict(cls_cnt),
            "by_source": {k: dict(v) for k, v in sorted(src_cnt.items())},
        }

    rep_p = Path("reports/phase3_data_summary.json")
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
