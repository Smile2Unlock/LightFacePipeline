# Phase 1 Acceptance Evaluation Report

| Variant | WIDER Easy | WIDER Med | WIDER Hard | WFLW Clean NME | WFLW Dark NME | DarkFace AP50 (Official-style) | DarkFace AP50 (Greedy) | DarkFace Recall | COCO Dark FPPI | Multi-face >=2 Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SeetaFace `base640` (ref, 6k) | — | — | — | — | — | 0.145% | — | 0.147% | — | — |
| SeetaFace `app640` (ref, 6k) | — | — | — | — | — | 0.614% | — | 0.645% | — | — |
| **clean_baseline_v2** | 96.25% | 88.44% | 57.25% | 6.83% | 7.97% | 8.47% | 8.48% | 14.27% | 0.0733 | 85.3% |
| **stage1_v2** | 96.20% | 87.86% | 56.95% | 6.56% | 6.78% | 36.87% | 36.89% | 47.19% | 0.0233 | 84.5% |
| **stage2_v2** | 96.00% | 87.57% | 56.26% | 6.43% | 6.72% | 37.13% | 37.16% | 47.42% | 0.0200 | 80.8% |
