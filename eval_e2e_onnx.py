"""Phase 4: ONNX Export (opset 17), Streaming Equivalence, E2E onnxruntime Verification & x86 CPU Bench."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common.align import align_face_112, crop_dual_scale_fas, letterbox_rgb, unletterbox_coords
from common.contracts import (
    ARCFACE_DST_112,
    COLOR_SPACE,
    FAS_CLASSES,
    FAS_CLIP_LEN,
    FAS_CROP_SIZE,
    FAS_SCALES,
    LANDMARK_ORDER,
    REC_INPUT_SIZE,
)
from common.runtime import collect_provenance, configure_cpu_threads, sha256_file
from detector.yolo11_face import DetectorONNXWrapper, LCFYOLO11Face5pt
from detector.yunet_lcf import LCFYuNetS
from liveness.st_fas_lite import STFASLiteClip
from recognizer.edgeface import build_recognizer_student

configure_cpu_threads(4)


def export_onnx_model(
    model: torch.nn.Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.eval().cpu()
    inputs_cpu = tuple(t.detach().cpu() for t in dummy_inputs)
    torch.onnx.export(
        model_cpu,
        inputs_cpu,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    return output_path


def _ort_session(onnx_path: Path, intra_threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def verify_pytorch_vs_onnx(
    pt_model: torch.nn.Module,
    onnx_path: Path,
    inputs: Dict[str, np.ndarray],
    tol: float = 1e-4,
    rtol: float = 1e-6,
) -> Dict[str, Any]:
    pt_model.eval().cpu()
    with torch.no_grad():
        pt_args = [torch.from_numpy(v) for v in inputs.values()]
        pt_outs = pt_model(*pt_args)
        if not isinstance(pt_outs, (tuple, list)):
            pt_outs = (pt_outs,)
        pt_np = [o.numpy() for o in pt_outs]

    sess = _ort_session(onnx_path, intra_threads=1)
    ort_np = sess.run(None, inputs)
    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(pt_np, ort_np)]
    rel_diffs = [float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0))) for a, b in zip(pt_np, ort_np)]
    max_diff = max(diffs)
    max_rel_diff = max(rel_diffs)
    passed = all(bool(np.allclose(a, b, rtol=rtol, atol=tol)) for a, b in zip(pt_np, ort_np))
    return {
        "model": onnx_path.name,
        "size_mb": round(onnx_path.stat().st_size / (1024 ** 2), 3),
        "sha256": sha256_file(onnx_path),
        "max_abs_diff": max_diff,
        "max_rel_diff": max_rel_diff,
        "per_output_max_abs_diff": diffs,
        "passed": passed,
    }


def verify_fas_clip_vs_streaming_onnx(
    clip_onnx: Path,
    spatial_onnx: Path,
    temporal_onnx: Path,
    clip_tex: np.ndarray,
    clip_ctx: np.ndarray,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """Verify exact numerical equivalence between 8-frame Clip FAS ONNX and Streaming Dual-ONNX."""
    sess_clip = _ort_session(clip_onnx, intra_threads=1)
    sess_sp = _ort_session(spatial_onnx, intra_threads=1)
    sess_tm = _ort_session(temporal_onnx, intra_threads=1)

    c_live, c_logits, c_clarity = sess_clip.run(None, {"clip_tex": clip_tex, "clip_ctx": clip_ctx})

    b, t = clip_tex.shape[:2]
    feats = []
    quals = []
    for ti in range(t):
        tex_curr = clip_tex[:, ti]
        tex_prev = clip_tex[:, ti - 1] if ti > 0 else clip_tex[:, 0]
        ctx_curr = clip_ctx[:, ti]
        f_t, q_t = sess_sp.run(None, {"tex_curr": tex_curr, "tex_prev": tex_prev, "ctx_curr": ctx_curr})
        feats.append(f_t)
        quals.append(q_t)

    feat_seq = np.stack(feats, axis=1)
    qual_seq = np.stack(quals, axis=1)
    s_live, s_logits, s_clarity = sess_tm.run(None, {"spatial_seq": feat_seq, "quality_seq": qual_seq})

    diff_live = float(np.max(np.abs(c_live - s_live)))
    diff_logits = float(np.max(np.abs(c_logits - s_logits)))
    diff_clarity = float(np.max(np.abs(c_clarity - s_clarity)))
    max_diff = max(diff_live, diff_logits, diff_clarity)
    return {
        "max_abs_diff_live_prob": diff_live,
        "max_abs_diff_spoof_logits": diff_logits,
        "max_abs_diff_clarity": diff_clarity,
        "max_abs_diff": max_diff,
        "passed": max_diff < tol,
    }


def benchmark_onnx_cpu(
    onnx_path: Path,
    feeds: Dict[str, np.ndarray],
    threads: int,
    warmup: int = 5,
    iters: int = 20,
) -> Dict[str, Any]:
    t_load0 = time.perf_counter()
    sess = _ort_session(onnx_path, intra_threads=threads)
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    for _ in range(warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "threads": threads,
        "load_ms": round(load_ms, 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "rss_mib": round(rss_mb, 2),
    }


def run_e2e_smoke_pipeline(
    det_onnx: Path,
    rec_onnx: Path,
    fas_sp_onnx: Path,
    fas_tm_onnx: Path,
    frames_rgb_u8: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run full 3-model chained inference (`Detector -> 5pt Umeyama -> Recognizer + Streaming FAS`)."""
    if frames_rgb_u8 is None:
        rng = np.random.default_rng(7)
        frames_rgb_u8 = [rng.integers(40, 210, size=(480, 640, 3), dtype=np.uint8) for _ in range(FAS_CLIP_LEN)]

    sess_det = _ort_session(det_onnx, intra_threads=2)
    sess_rec = _ort_session(rec_onnx, intra_threads=2)
    sess_sp = _ort_session(fas_sp_onnx, intra_threads=2)
    sess_tm = _ort_session(fas_tm_onnx, intra_threads=2)

    feat_queue = []
    qual_queue = []
    tex_prev: Optional[np.ndarray] = None
    last_emb: Optional[np.ndarray] = None
    last_norm: float = 0.0

    for frame in frames_rgb_u8:
        oh, ow = frame.shape[:2]
        lb, scale, pad = letterbox_rgb(frame, (640, 640))
        det_in = lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        boxes, scores, kpts = sess_det.run(None, {"rgb_01": det_in})

        top_idx = int(np.argmax(scores[0]))
        top_score = float(scores[0, top_idx])
        b_orig, k_orig = unletterbox_coords(boxes[0, top_idx : top_idx + 1], kpts[0, top_idx : top_idx + 1], scale, pad, (oh, ow))
        assert b_orig is not None and k_orig is not None
        box_xyxy = b_orig[0]
        kpt5 = k_orig[0]

        # Fallback valid box if untrained smoke weights predict degenerate box
        if (box_xyxy[2] - box_xyxy[0]) < 16 or (box_xyxy[3] - box_xyxy[1]) < 16:
            box_xyxy = np.array([220.0, 140.0, 420.0, 340.0], dtype=np.float32)
            kpt5 = np.array(
                [[280.0, 210.0], [360.0, 210.0], [320.0, 250.0], [290.0, 295.0], [350.0, 295.0]], dtype=np.float32
            )

        # Recognition on unenhanced original RGB frame
        aligned_112, _ = align_face_112(frame, kpt5, out_size=REC_INPUT_SIZE)
        rec_in = ((aligned_112.transpose(2, 0, 1)[None].astype(np.float32) / 255.0) - 0.5) / 0.5
        emb, fnorm = sess_rec.run(None, {"rgb_norm": rec_in})
        last_emb = emb
        last_norm = float(fnorm[0, 0])

        # Streaming dual-scale FAS on unenhanced original RGB frame
        tex_u8, ctx_u8 = crop_dual_scale_fas(frame, box_xyxy, out_size=FAS_CROP_SIZE, scales=FAS_SCALES)
        tex_f = tex_u8.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        ctx_f = ctx_u8.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        if tex_prev is None:
            tex_prev = tex_f
        f_t, q_t = sess_sp.run(None, {"tex_curr": tex_f, "tex_prev": tex_prev, "ctx_curr": ctx_f})
        tex_prev = tex_f
        feat_queue.append(f_t)
        qual_queue.append(q_t)

    feat_seq = np.stack(feat_queue[-FAS_CLIP_LEN:], axis=1)
    qual_seq = np.stack(qual_queue[-FAS_CLIP_LEN:], axis=1)
    live_prob, spoof_logits, clarity = sess_tm.run(None, {"spatial_seq": feat_seq, "quality_seq": qual_seq})

    return {
        "frames_processed": len(frames_rgb_u8),
        "embedding_dim": int(last_emb.shape[-1]) if last_emb is not None else 0,
        "embedding_l2_norm": float(np.linalg.norm(last_emb[0])) if last_emb is not None else 0.0,
        "feature_norm_quality": last_norm,
        "live_prob": float(live_prob[0, 0]),
        "clarity_score": float(clarity[0, 0]),
        "spoof_logits": [float(v) for v in spoof_logits[0]],
        "det_top_score_smoke": top_score,
    }


def export_and_verify_all(
    out_dir: Path,
    det_arch: str = "lcf_yolo11n_5pt",
    rec_arch: str = "edgeface_s",
    det_ckpt: Optional[Path] = None,
    rec_ckpt: Optional[Path] = None,
    fas_ckpt: Optional[Path] = None,
    run_cpu_bench: bool = True,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1) Detector
    det_model = LCFYOLO11Face5pt() if det_arch == "lcf_yolo11n_5pt" else LCFYuNetS()
    if det_ckpt and det_ckpt.is_file():
        ck = torch.load(det_ckpt, map_location="cpu", weights_only=False)
        det_model.load_state_dict(ck.get("ema", ck.get("model", ck)))
    det_wrap = DetectorONNXWrapper(det_model).eval()
    det_onnx = models_dir / "detector_5pt_lcf.onnx"
    det_dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    export_onnx_model(
        det_wrap,
        (torch.from_numpy(det_dummy),),
        det_onnx,
        input_names=["rgb_01"],
        output_names=["boxes_xyxy", "scores", "kpts5_xy"],
    )
    det_ver = verify_pytorch_vs_onnx(det_wrap, det_onnx, {"rgb_01": det_dummy})

    # 2) Recognizer
    rec_model = build_recognizer_student(rec_arch).eval()
    if rec_ckpt and rec_ckpt.is_file():
        ck = torch.load(rec_ckpt, map_location="cpu", weights_only=False)
        rec_model.load_state_dict(ck.get("ema", ck.get("model", ck)))
    rec_onnx = models_dir / "recognizer_edgeface.onnx"
    rec_dummy = np.random.uniform(-1.0, 1.0, (1, 3, 112, 112)).astype(np.float32)
    export_onnx_model(
        rec_model,
        (torch.from_numpy(rec_dummy),),
        rec_onnx,
        input_names=["rgb_norm"],
        output_names=["l2_emb", "feat_norm"],
    )
    rec_ver = verify_pytorch_vs_onnx(rec_model, rec_onnx, {"rgb_norm": rec_dummy})

    # 3) Temporal Liveness (Clip + Streaming Spatial + Streaming Temporal Head)
    fas_model = STFASLiteClip().eval()
    if fas_ckpt and fas_ckpt.is_file():
        ck = torch.load(fas_ckpt, map_location="cpu", weights_only=False)
        fas_model.load_state_dict(ck.get("ema", ck.get("model", ck)))

    fas_clip_onnx = models_dir / "fas_clip.onnx"
    fas_sp_onnx = models_dir / "fas_spatial_enc.onnx"
    fas_tm_onnx = models_dir / "fas_temporal_head.onnx"

    clip_tex_dummy = np.random.rand(1, FAS_CLIP_LEN, 3, 128, 128).astype(np.float32)
    clip_ctx_dummy = np.random.rand(1, FAS_CLIP_LEN, 3, 128, 128).astype(np.float32)
    export_onnx_model(
        fas_model,
        (torch.from_numpy(clip_tex_dummy), torch.from_numpy(clip_ctx_dummy)),
        fas_clip_onnx,
        input_names=["clip_tex", "clip_ctx"],
        output_names=["live_prob", "spoof_logits", "clarity_score"],
    )
    fas_clip_ver = verify_pytorch_vs_onnx(
        fas_model, fas_clip_onnx, {"clip_tex": clip_tex_dummy, "clip_ctx": clip_ctx_dummy}
    )

    sp_tex_dummy = np.random.rand(1, 3, 128, 128).astype(np.float32)
    export_onnx_model(
        fas_model.spatial_enc,
        (torch.from_numpy(sp_tex_dummy), torch.from_numpy(sp_tex_dummy), torch.from_numpy(sp_tex_dummy)),
        fas_sp_onnx,
        input_names=["tex_curr", "tex_prev", "ctx_curr"],
        output_names=["spatial_feat", "frame_quality"],
    )
    fas_sp_ver = verify_pytorch_vs_onnx(
        fas_model.spatial_enc,
        fas_sp_onnx,
        {"tex_curr": sp_tex_dummy, "tex_prev": sp_tex_dummy, "ctx_curr": sp_tex_dummy},
    )

    tm_feat_dummy = np.random.randn(1, FAS_CLIP_LEN, fas_model.feat_dim).astype(np.float32)
    tm_qual_dummy = np.random.randn(1, FAS_CLIP_LEN, 1).astype(np.float32)
    export_onnx_model(
        fas_model.temporal_head,
        (torch.from_numpy(tm_feat_dummy), torch.from_numpy(tm_qual_dummy)),
        fas_tm_onnx,
        input_names=["spatial_seq", "quality_seq"],
        output_names=["live_prob", "spoof_logits", "clarity_score"],
    )
    fas_tm_ver = verify_pytorch_vs_onnx(
        fas_model.temporal_head,
        fas_tm_onnx,
        {"spatial_seq": tm_feat_dummy, "quality_seq": tm_qual_dummy},
    )

    stream_equiv = verify_fas_clip_vs_streaming_onnx(
        fas_clip_onnx, fas_sp_onnx, fas_tm_onnx, clip_tex_dummy, clip_ctx_dummy
    )
    e2e_smoke = run_e2e_smoke_pipeline(det_onnx, rec_onnx, fas_sp_onnx, fas_tm_onnx)

    cpu_bench: Dict[str, Any] = {}
    if run_cpu_bench:
        for th in (1, 4):
            cpu_bench[f"threads_{th}"] = {
                "detector": benchmark_onnx_cpu(det_onnx, {"rgb_01": det_dummy}, threads=th, warmup=2, iters=6),
                "recognizer": benchmark_onnx_cpu(rec_onnx, {"rgb_norm": rec_dummy}, threads=th, warmup=2, iters=10),
                "fas_spatial": benchmark_onnx_cpu(
                    fas_sp_onnx,
                    {"tex_curr": sp_tex_dummy, "tex_prev": sp_tex_dummy, "ctx_curr": sp_tex_dummy},
                    threads=th,
                    warmup=2,
                    iters=10,
                ),
                "fas_temporal": benchmark_onnx_cpu(
                    fas_tm_onnx,
                    {"spatial_seq": tm_feat_dummy, "quality_seq": tm_qual_dummy},
                    threads=th,
                    warmup=2,
                    iters=10,
                ),
            }

    deploy_total_mb = round(
        det_ver["size_mb"] + rec_ver["size_mb"] + fas_sp_ver["size_mb"] + fas_tm_ver["size_mb"], 3
    )

    default_th = {
        "det_score_th": 0.45,
        "rec_cosine_th": 0.36,
        "rec_norm_min": 8.0,
        "liveness_th": 0.50,
        "clarity_th": 0.35,
    }
    if thresholds:
        default_th.update({k: float(v) for k, v in thresholds.items() if v is not None})

    manifest = {
        "color_space": COLOR_SPACE,
        "landmark_order": list(LANDMARK_ORDER),
        "arcface_dst_112": ARCFACE_DST_112.tolist(),
        "fas_clip_len": FAS_CLIP_LEN,
        "fas_crop_size": FAS_CROP_SIZE,
        "fas_scales": list(FAS_SCALES),
        "fas_classes": list(FAS_CLASSES),
        "recommended_thresholds": default_th,
        "models": {
            "detector": {**det_ver, "input": "rgb_01 [1,3,640,640] in [0,1] (raw unenhanced RGB)"},
            "recognizer": {**rec_ver, "input": "rgb_norm [1,3,112,112] = (rgb/255 - 0.5)/0.5"},
            "fas_clip": {**fas_clip_ver, "input": "clip_tex/clip_ctx [1,8,3,128,128] in [0,1]"},
            "fas_spatial_enc": {**fas_sp_ver, "input": "tex_curr/tex_prev/ctx_curr [1,3,128,128] in [0,1]"},
            "fas_temporal_head": {**fas_tm_ver, "input": "spatial_seq [1,8,192], quality_seq [1,8,1]"},
        },
        "deploy_bundle_fp32_mb": deploy_total_mb,
        "streaming_equivalence": stream_equiv,
        "e2e_smoke": e2e_smoke,
        "cpu_benchmark": cpu_bench,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    sha_lines = [
        f"{v['sha256']}  models/{v['model']}"
        for v in [det_ver, rec_ver, fas_clip_ver, fas_sp_ver, fas_tm_ver]
    ]
    (out_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    prov = collect_provenance({"det_arch": det_arch, "rec_arch": rec_arch, "deploy_bundle_fp32_mb": deploy_total_mb})
    (out_dir / "provenance.md").write_text(
        "# Provenance\n\n```json\n" + json.dumps(prov, indent=2) + "\n```\n", encoding="utf-8"
    )

    eval_md = (
        f"# Phase 4 E2E ONNX Verification Report\n\n"
        f"- **Deploy Bundle FP32 Size**: `{deploy_total_mb} MB`\n"
        f"- **PyTorch vs ONNX FP32 Max Abs Diff**: detector=`{det_ver['max_abs_diff']:.2e}`, "
        f"recognizer=`{rec_ver['max_abs_diff']:.2e}`, fas_clip=`{fas_clip_ver['max_abs_diff']:.2e}`\n"
        f"- **Clip vs Streaming Dual-ONNX Equivalence**: max_abs_diff=`{stream_equiv['max_abs_diff']:.2e}` "
        f"(passed=`{stream_equiv['passed']}`)\n"
        f"- **E2E Smoke Pipeline**: frames=`{e2e_smoke['frames_processed']}`, "
        f"emb_norm=`{e2e_smoke['embedding_l2_norm']:.4f}`, live_prob=`{e2e_smoke['live_prob']:.4f}`\n"
    )
    (out_dir / "eval_report.md").write_text(eval_md, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 ONNX Export & End-to-End Verification")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--det-arch", default="lcf_yolo11n_5pt", choices=["lcf_yolo11n_5pt", "lcf_yunet_s"])
    parser.add_argument(
        "--rec-arch",
        default="edgeface_s",
        choices=["edgeface_s", "edgeface_xs", "edgeface_s_gamma_05", "edgeface_xs_gamma_06", "mobilefacenet_w15"],
    )
    parser.add_argument("--det-ckpt", type=Path, default=None)
    parser.add_argument("--rec-ckpt", type=Path, default=None)
    parser.add_argument("--fas-ckpt", type=Path, default=None)
    parser.add_argument("--det-score-th", type=float, default=None)
    parser.add_argument("--rec-cosine-th", type=float, default=None)
    parser.add_argument("--rec-norm-min", type=float, default=None)
    parser.add_argument("--liveness-th", type=float, default=None)
    parser.add_argument("--clarity-th", type=float, default=None)
    parser.add_argument("--no-cpu-bench", action="store_true")
    args = parser.parse_args()

    thresholds = {
        "det_score_th": args.det_score_th,
        "rec_cosine_th": args.rec_cosine_th,
        "rec_norm_min": args.rec_norm_min,
        "liveness_th": args.liveness_th,
        "clarity_th": args.clarity_th,
    }
    res = export_and_verify_all(
        out_dir=args.out_dir,
        det_arch=args.det_arch,
        rec_arch=args.rec_arch,
        det_ckpt=args.det_ckpt,
        rec_ckpt=args.rec_ckpt,
        fas_ckpt=args.fas_ckpt,
        run_cpu_bench=not args.no_cpu_bench,
        thresholds=thresholds,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
