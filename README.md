# LightFacePipeline

Low-light full-chain face authentication training pipeline (**Detection + 5-pt Alignment + Cross-Illumination Recognition + Native Temporal Liveness**) targeting compact ONNX deployment (`~15–25 MB` FP32) and trained on a 64GB ROCm/DTK HCU (`gfx936`).

> **Scope**: Offline training, distillation, ONNX export, and end-to-end verification only. See [docs/low_light_face_pipeline_training_plan.md](docs/low_light_face_pipeline_training_plan.md) for the complete architecture specification.

## Architecture Overview

```text
Raw RGB Frame Stream (t-7 ... t)
  ├──▶ [Phase 1] LCF + YOLO11n-face-5pt (~5-6 MB FP32; fallback LCF+YuNet-S <1 MB)
  │       └─ outputs: bbox(x1,y1,x2,y2), det_score, 5-pts (RGB image coords)
  ├──▶ (5-pt Umeyama 112x112 align) ──▶ [Phase 2] EdgeFace-S / XS (~7-14 MB FP32)
  │                                        └─ outputs: 512-d L2 embedding + ||z|| quality
  └──▶ (1.5x & 2.7x 128x128 dual-scale crops) ──▶ [Phase 3] ST-FAS-Lite (~3-5 MB FP32)
                                                     └─ outputs: live_prob, spoof_logits[5], clarity
```

## ROCm / DTK Environment & Storage Layout

Verified on the target Hygon BW HCU (`gfx936:sramecc+:xnack-`, 65,520 MiB VRAM, PyTorch `2.9.0+das.opt1.dtk2604`, HIP `6.3.26093`):
- **Recommended precision**: `fp16` + `GradScaler` (2.5× faster than `bf16` on Conv/DW-SepConv; AdaFace/Cosine math stays FP32).
- **Recommended SDPA backend**: `flash_attention` (4.2× faster than `math` fallback).
- **CPU Threading inside cgroup containers**: The host reports 128 physical cores while the container cgroup quota is 15 vCPUs (`cpu.max=1500000 100000`). Always keep `OMP_NUM_THREADS=4` (configured automatically via `common.runtime.configure_cpu_threads` & `tests/conftest.py`) and `num_workers: 4` to prevent 128-thread OpenMP spinlock contention during CPU/ONNX operations.
- **Remote storage**:
  - Scratch / active repo / datasets / runs: `/workspace/LightFacePipeline/` (`~836 GiB` free, `rw`).
  - Persistent backups (checkpoints / ONNX artifacts / reports / splits): `/public/home/ation_ciger/LightFacePipeline/` (`rw`, 50 GB quota).

## Quickstart

```bash
# 1. Phase 0 ROCm/DTK stress test (pure PyTorch, run first on compute node)
python3 env_check_rocm.py --output reports/rocm_stress_report.json

# 2. Generate deterministic DarkFace 5,000 / 1,000 split
python3 scripts/prepare_darkface_split.py --source-dir /path/to/Smile2Unlock_v2/build/research/darkface

# 3. Run full unit + GPU + ONNX equivalence verification suite
pytest -v tests/

# 4. Smoke training checks per phase
python3 -m detector.train --smoke
python3 -m recognizer.train --smoke
python3 -m liveness.train --smoke

# 5. Phase 4 ONNX export, streaming equivalence (<1e-4), and CPU latency benchmark
python3 eval_e2e_onnx.py --out-dir artifacts
```
