# Phase 2 Recognition Offline Acceptance Report

## 1. 5 Standard Benchmarks (Clean-Clean & Clean -> Dark)

| Benchmark | Model | Clean Flip Acc (%) | Clean Single Acc (%) | Dark-GT Acc (%) | Dark-GT TAR@1e-4 | Dark-Jit Acc (%) | Dark-Jit TAR@1e-4 | Dark-Jit TAR@1e-3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lfw | official_edgeface_s | 99.77 | 99.80 | 96.42 | 83.87% | 96.32 | 83.40% | 88.17% |
| lfw | phase2_best | 99.80 | 99.78 | 99.37 | 97.43% | 99.40 | 98.10% | 98.87% |
| lfw | cvlface_teacher | 99.83 | 99.83 | 98.40 | 95.07% | 98.08 | 94.53% | 95.60% |
| cfp_fp | official_edgeface_s | 97.99 | 98.04 | 92.47 | 41.23% | 91.01 | 64.60% | 68.54% |
| cfp_fp | phase2_best | 98.04 | 98.09 | 96.10 | 74.46% | 95.76 | 74.17% | 81.11% |
| cfp_fp | cvlface_teacher | 99.07 | 98.97 | 94.73 | 85.71% | 93.67 | 79.46% | 84.51% |
| agedb_30 | official_edgeface_s | 96.97 | 96.85 | 92.27 | 56.10% | 91.02 | 59.73% | 66.13% |
| agedb_30 | phase2_best | 96.93 | 96.82 | 95.38 | 67.70% | 94.90 | 63.13% | 76.70% |
| agedb_30 | cvlface_teacher | 97.90 | 97.90 | 94.93 | 80.53% | 94.45 | 78.20% | 80.00% |
| cplfw | official_edgeface_s | 92.50 | 92.47 | 87.25 | 3.30% | 87.63 | 16.03% | 52.73% |
| cplfw | phase2_best | 92.42 | 92.47 | 90.53 | 59.30% | 91.20 | 56.10% | 65.37% |
| cplfw | cvlface_teacher | 94.37 | 94.20 | 89.25 | 43.10% | 89.90 | 54.40% | 72.90% |
| calfw | official_edgeface_s | 95.45 | 95.60 | 92.52 | 72.43% | 92.52 | 69.10% | 75.17% |
| calfw | phase2_best | 95.45 | 95.50 | 94.93 | 84.03% | 95.23 | 81.23% | 85.83% |
| calfw | cvlface_teacher | 95.85 | 95.93 | 94.53 | 71.83% | 94.77 | 71.80% | 88.07% |

## 2. End-to-End Closed-Loop: Phase 1 Detector 5-pt on Degraded LFW Raw 250x250

| Model | Dark Det Rate | 5pt NME (%) | Clean-Ref5pt Dark TAR@1e-4 | Phase1Det5pt Dark TAR@1e-4 | Gap (<1.0%) | Phase1Det5pt TAR@1e-3 | Fixed Cosine Thresh |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_edgeface_s | 100.00% | 1.72% | 98.93% | 98.70% | 0.23% | 99.40% | 0.3150 |
| phase2_best | 100.00% | 1.72% | 99.07% | 99.03% | 0.03% | 99.47% | 0.3200 |
| cvlface_teacher | 100.00% | 1.72% | 99.63% | 99.63% | 0.00% | 99.63% | 0.1900 |
