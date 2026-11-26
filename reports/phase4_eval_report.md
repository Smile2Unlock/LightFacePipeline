# Phase 4 E2E ONNX Verification Report

- **Deploy Bundle FP32 Size**: `26.374 MB`
- **PyTorch vs ONNX FP32 Max Abs Diff**: detector=`1.83e-04`, recognizer=`1.64e-06`, fas_clip=`1.31e-06`
- **Clip vs Streaming Dual-ONNX Equivalence**: max_abs_diff=`0.00e+00` (passed=`True`)
- **E2E Smoke Pipeline**: frames=`8`, emb_norm=`1.0000`, live_prob=`0.4792`
