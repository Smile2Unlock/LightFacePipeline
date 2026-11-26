# Phase 3 Low-Light Temporal Liveness Evaluation Report (ISO/IEC 30107-3)

- **Checkpoint**: `runs/liveness_st_fas_lite_local/student_v2/best.pt`
- **Val-Calibrated Thresholds**: `th_min_acer = 0.7623`, `th_apcer<=2% = 0.8730`, `th_apcer<=1% = 0.8919`
- **Overall Test (All 9 Domains)**: `AUC = 0.8462`, `ACER@th_acer = 26.94%`, `BPCER@th_apcer2 = 71.04%`, `APCER@th_apcer2 = 0.94%`
- **Real-FPS Test (Excl. 1-fps CASIA-2012)**: `AUC = 0.9153`, `ACER@th_acer = 18.99%`
- **Luma Anti-Shortcut Audit**: `live_mean=76.80`, `spoof_mean=77.10`, `gap=0.30`, `warning=False`
- **Bbox ±6% Jitter Sensitivity (Low-Light)**: `clean ACER=29.01%` vs `jittered ACER=30.41%` (`Δ=+0.0140`)

## Stratified Lighting Matrix (Val-Calibrated `th_min_acer`)

| Condition | AUC | APCER (%) | BPCER (%) | ACER (%) |
|---|---:|---:|---:|---:|
| normal | 0.8461 | 1.33 | 51.08 | 26.20 |
| backlight | 0.7360 | 21.28 | 43.17 | 32.22 |
| low_light | 0.8002 | 4.79 | 53.24 | 29.01 |
| screen_glow | 0.8093 | 7.98 | 41.73 | 24.85 |
| dim_screen | 0.9998 | 0.00 | 53.24 | 26.62 |
| frozen_rigid | 0.9139 | 0.00 | 52.16 | 26.08 |
