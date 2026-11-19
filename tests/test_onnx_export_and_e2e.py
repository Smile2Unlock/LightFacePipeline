"""Tests for Phase 4 ONNX export (opset 17), <1e-4 numerical alignment, streaming equivalence & E2E pipeline."""

from __future__ import annotations

from pathlib import Path

from eval_e2e_onnx import export_and_verify_all


def test_onnx_export_streaming_equivalence_and_e2e(tmp_path: Path) -> None:
    manifest = export_and_verify_all(
        out_dir=tmp_path,
        det_arch="lcf_yolo11n_5pt",
        rec_arch="edgeface_xs",
        run_cpu_bench=True,
    )

    # All models export & match PyTorch within < 1e-4
    for name, info in manifest["models"].items():
        assert info["passed"], f"ONNX mismatch for {name}: {info['max_abs_diff']}"
        assert info["max_abs_diff"] < 1e-4

    # Compact deploy bundle check (LCF+YOLO11n + EdgeFace-XS + ST-FAS-Lite <= 20 MB)
    assert manifest["models"]["detector"]["size_mb"] <= 7.5
    assert manifest["models"]["recognizer"]["size_mb"] <= 8.5
    assert (manifest["models"]["fas_spatial_enc"]["size_mb"] + manifest["models"]["fas_temporal_head"]["size_mb"]) <= 6.0
    assert manifest["deploy_bundle_fp32_mb"] <= 20.0

    # Clip FAS ONNX vs Streaming Dual-ONNX equivalence < 1e-4
    seq = manifest["streaming_equivalence"]
    assert seq["passed"], f"Streaming equivalence failed: {seq['max_abs_diff']}"
    assert seq["max_abs_diff"] < 1e-4

    # End-to-end 3-model onnxruntime smoke pipeline
    e2e = manifest["e2e_smoke"]
    assert e2e["frames_processed"] == 8
    assert e2e["embedding_dim"] == 512
    assert abs(e2e["embedding_l2_norm"] - 1.0) < 1e-4
    assert 0.0 <= e2e["live_prob"] <= 1.0
    assert 0.0 <= e2e["clarity_score"] <= 1.0

    # Artifacts created
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "SHA256SUMS").is_file()
    assert (tmp_path / "eval_report.md").is_file()
    assert (tmp_path / "provenance.md").is_file()
