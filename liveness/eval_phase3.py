"""Phase 3 Full ISO/IEC 30107-3 Stratified Evaluation Report Generator."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.degradations import LivenessPhysicalAugmentor
from common.metrics import audit_luma_anti_shortcut, compute_iso30107_metrics
from common.runtime import get_amp_autocast, get_device, get_sdpa_context
from liveness.dataset import VideoClipLivenessDataset
from liveness.st_fas_lite import STFASLiteClip


@torch.no_grad()
def _collect_predictions(
    model: STFASLiteClip,
    loader: DataLoader,
    augmentor: LivenessPhysicalAugmentor,
    lighting_mode: str,
    device: torch.device,
    bbox_jitter_std: float = 0.0,
) -> Dict[str, Any]:
    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    is_live: List[int] = []
    sources: List[str] = []
    attack_groups: List[str] = []
    clarities: List[float] = []
    live_luma: List[np.ndarray] = []
    spoof_luma: List[np.ndarray] = []

    for bi, batch in enumerate(loader):
        tex = batch["tex_u8"].to(device, non_blocking=True).float().div_(255.0)
        ctx = batch["ctx_u8"].to(device, non_blocking=True).float().div_(255.0)
        lbl = batch["label"].numpy()
        src = list(batch["source"])
        grp = list(batch["attack_group"])

        if bbox_jitter_std > 0.0:
            grid_base = torch.nn.functional.affine_grid(
                torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device).repeat(tex.shape[0] * tex.shape[1], 1, 1),
                size=(tex.shape[0] * tex.shape[1], 3, tex.shape[-2], tex.shape[-1]),
                align_corners=False,
            )
            g = torch.Generator(device=device).manual_seed(777 + bi)
            delta = torch.randn((tex.shape[0] * tex.shape[1], 1, 1, 2), generator=g, device=device) * bbox_jitter_std
            grid = grid_base + delta
            bt = tex.shape[0] * tex.shape[1]
            tex = torch.nn.functional.grid_sample(tex.view(bt, 3, 128, 128), grid, mode="bilinear", padding_mode="border", align_corners=False).view_as(tex)
            ctx = torch.nn.functional.grid_sample(ctx.view(bt, 3, 128, 128), grid_base + delta * (1.5 / 2.7), mode="bilinear", padding_mode="border", align_corners=False).view_as(ctx)

        if lighting_mode == "frozen_rigid":
            # Keep only live samples and freeze them into synthetic hand-shake spoofs
            live_mask = lbl == 0
            if not live_mask.any():
                continue
            lm_idx = torch.from_numpy(np.where(live_mask)[0]).to(device)
            tex, ctx = augmentor.apply_eval_lighting(tex[lm_idx], ctx[lm_idx], "frozen_rigid", seed=9000 + bi)
            lbl = np.full(len(lm_idx), 4, dtype=np.int64)
            src = [src[i] for i in np.where(live_mask)[0]]
            grp = ["frozen_rigid"] * len(lm_idx)
        elif lighting_mode == "dim_screen":
            # Apply dim screen replay moire/bezel only to spoof samples (paired against low_light live in main)
            spoof_mask = lbl != 0
            if not spoof_mask.any():
                continue
            sm_idx = torch.from_numpy(np.where(spoof_mask)[0]).to(device)
            tex, ctx = augmentor.apply_eval_lighting(tex[sm_idx], ctx[sm_idx], "dim_screen", seed=8000 + bi)
            lbl = lbl[spoof_mask]
            src = [src[i] for i in np.where(spoof_mask)[0]]
            grp = ["dim_screen"] * len(sm_idx)
        else:
            tex, ctx = augmentor.apply_eval_lighting(tex, ctx, lighting_mode, seed=3000 + bi * 13)

        with get_sdpa_context("math"), get_amp_autocast(device, "fp16"):
            lp, _, clarity = model(tex, ctx)

        probs.extend(lp.squeeze(-1).float().cpu().numpy().tolist())
        clarities.extend(clarity.squeeze(-1).float().cpu().numpy().tolist())
        labels.extend(lbl.tolist())
        is_live.extend((lbl == 0).astype(np.int64).tolist())
        sources.extend(src)
        attack_groups.extend(grp)

        if len(live_luma) < 10:
            arr = tex.detach().cpu().numpy()
            lm_arr = lbl == 0
            sm_arr = lbl != 0
            if lm_arr.any():
                live_luma.append(arr[lm_arr])
            if sm_arr.any():
                spoof_luma.append(arr[sm_arr])

    return {
        "probs": np.asarray(probs, dtype=np.float64),
        "is_live": np.asarray(is_live, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "sources": sources,
        "attack_groups": attack_groups,
        "clarities": np.asarray(clarities, dtype=np.float64),
        "live_luma": np.concatenate(live_luma, axis=0) if live_luma else np.empty((0, 8, 3, 128, 128)),
        "spoof_luma": np.concatenate(spoof_luma, axis=0) if spoof_luma else np.empty((0, 8, 3, 128, 128)),
    }


def _eval_with_th(probs: np.ndarray, is_live: np.ndarray, th: float) -> Dict[str, Any]:
    return compute_iso30107_metrics(probs, is_live, threshold=th)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 ISO 30107-3 Full Evaluation")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=Path("datasets/manifests/fas_video_clips_val.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("datasets/manifests/fas_video_clips_test.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/phase3_eval_report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/phase3_eval_report.md"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    model = STFASLiteClip().to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("ema", ck.get("model", ck)), strict=True)
    augmentor = LivenessPhysicalAugmentor().to(device)

    val_ds = VideoClipLivenessDataset(args.val_manifest, augment=False)
    test_ds = VideoClipLivenessDataset(args.test_manifest, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 1) Calibrate unified thresholds ONLY on Val across all lighting regimes + frozen negatives
    val_all_p, val_all_y = [], []
    for cond in ("normal", "backlight", "low_light", "screen_glow", "frozen_rigid"):
        res = _collect_predictions(model, val_loader, augmentor, cond, device)
        val_all_p.append(res["probs"])
        val_all_y.append(res["is_live"])
    val_cal = compute_iso30107_metrics(np.concatenate(val_all_p), np.concatenate(val_all_y))
    th_acer = float(val_cal["operating_point"]["threshold"])
    th_apcer2 = float(val_cal["fixed_apcer_points"]["APCER<=2.0%"]["threshold"])
    th_apcer1 = float(val_cal["fixed_apcer_points"]["APCER<=1.0%"]["threshold"])

    # 2) Evaluate on Test across 4 lighting conditions + dim_screen + frozen_rigid
    test_by_cond: Dict[str, Any] = {}
    all_t_p, all_t_y, all_t_src = [], [], []
    live_luma_list, spoof_luma_list = [], []
    for cond in ("normal", "backlight", "low_light", "screen_glow"):
        res = _collect_predictions(model, test_loader, augmentor, cond, device)
        m_acer = _eval_with_th(res["probs"], res["is_live"], th_acer)
        m_a2 = _eval_with_th(res["probs"], res["is_live"], th_apcer2)
        m_free = compute_iso30107_metrics(res["probs"], res["is_live"])
        test_by_cond[cond] = {
            "res": res,
            "summary": {
                "auc": m_free["auc"],
                "n_live": m_free["n_live"],
                "n_spoof": m_free["n_spoof"],
                "at_val_th_acer": m_acer["operating_point"],
                "at_val_th_apcer2": m_a2["operating_point"],
                "clarity_mean": float(res["clarities"].mean()),
            },
        }
        all_t_p.append(res["probs"])
        all_t_y.append(res["is_live"])
        all_t_src.extend(res["sources"])
        if len(res["live_luma"]) > 0:
            live_luma_list.append(res["live_luma"])
        if len(res["spoof_luma"]) > 0:
            spoof_luma_list.append(res["spoof_luma"])

    low_res = test_by_cond["low_light"]["res"]
    normal_res = test_by_cond["normal"]["res"]
    live_low_p = low_res["probs"][low_res["is_live"] == 1]

    # Dim-screen replay Spoofs paired against dark Live faces
    ds_res = _collect_predictions(model, test_loader, augmentor, "dim_screen", device)
    ds_p = np.concatenate([live_low_p, ds_res["probs"]])
    ds_y = np.concatenate([np.ones(len(live_low_p), dtype=np.int64), ds_res["is_live"]])
    dim_screen_summary = {
        "auc": compute_iso30107_metrics(ds_p, ds_y)["auc"],
        "n_live": int(len(live_low_p)),
        "n_spoof": int(len(ds_res["probs"])),
        "at_val_th_acer": _eval_with_th(ds_p, ds_y, th_acer)["operating_point"],
        "at_val_th_apcer2": _eval_with_th(ds_p, ds_y, th_apcer2)["operating_point"],
        "clarity_mean": float(ds_res["clarities"].mean()),
    }

    # Frozen-Rigid vs Live (combine frozen_rigid spoofs with normal+low_light live)
    fr_res = _collect_predictions(model, test_loader, augmentor, "frozen_rigid", device)
    live_mix_p = np.concatenate([
        normal_res["probs"][normal_res["is_live"] == 1],
        live_low_p,
    ])
    fr_p = np.concatenate([live_mix_p, fr_res["probs"]])
    fr_y = np.concatenate([np.ones(len(live_mix_p), dtype=np.int64), fr_res["is_live"]])
    frozen_summary = {
        "auc": compute_iso30107_metrics(fr_p, fr_y)["auc"],
        "n_live": int(len(live_mix_p)),
        "n_spoof": int(len(fr_res["probs"])),
        "at_val_th_acer": _eval_with_th(fr_p, fr_y, th_acer)["operating_point"],
        "at_val_th_apcer2": _eval_with_th(fr_p, fr_y, th_apcer2)["operating_point"],
    }

    # 3) Attack-Group breakdown under Normal + Low-Light
    attack_breakdown: Dict[str, Any] = {}
    for cond_name in ("normal", "low_light"):
        r = test_by_cond[cond_name]["res"]
        live_p = r["probs"][r["is_live"] == 1]
        groups = sorted({g for g, il in zip(r["attack_groups"], r["is_live"]) if il == 0})
        attack_breakdown[cond_name] = {}
        for g in groups:
            sp_p = r["probs"][np.asarray([gg == g and il == 0 for gg, il in zip(r["attack_groups"], r["is_live"])])]
            if len(sp_p) == 0:
                continue
            gp = np.concatenate([live_p, sp_p])
            gy = np.concatenate([np.ones(len(live_p), dtype=np.int64), np.zeros(len(sp_p), dtype=np.int64)])
            attack_breakdown[cond_name][g] = {
                "n_spoof": int(len(sp_p)),
                "auc": compute_iso30107_metrics(gp, gy)["auc"],
                "apcer_at_val_th": float((sp_p < 0.0).mean() if False else (sp_p >= th_acer).mean()),
                "apcer_at_th_apcer2": float((sp_p >= th_apcer2).mean()),
            }

    # 4) Per-Domain / Cross-Domain Breakdown
    domain_breakdown: Dict[str, Any] = {}
    for cond_name in ("normal", "low_light"):
        r = test_by_cond[cond_name]["res"]
        srcs = sorted(set(r["sources"]))
        domain_breakdown[cond_name] = {}
        for s in srcs:
            mask = np.asarray([ss == s for ss in r["sources"]])
            p_s, y_s = r["probs"][mask], r["is_live"][mask]
            if (y_s == 1).any() and (y_s == 0).any():
                m_s = compute_iso30107_metrics(p_s, y_s, threshold=th_acer)
                domain_breakdown[cond_name][s] = {
                    "n": int(len(y_s)),
                    "auc": m_s["auc"],
                    "ACER_at_val_th": m_s["operating_point"]["ACER"],
                    "APCER": m_s["operating_point"]["APCER"],
                    "BPCER": m_s["operating_point"]["BPCER"],
                }
            else:
                domain_breakdown[cond_name][s] = {
                    "n": int(len(y_s)),
                    "live_count": int((y_s == 1).sum()),
                    "spoof_count": int((y_s == 0).sum()),
                    "mean_live_prob": float(p_s.mean()),
                }

    # 5) Phase-1 detector bbox jitter sensitivity on low_light
    jit_res = _collect_predictions(model, test_loader, augmentor, "low_light", device, bbox_jitter_std=0.06)
    jit_acer = _eval_with_th(jit_res["probs"], jit_res["is_live"], th_acer)["operating_point"]["ACER"]
    clean_ll_acer = test_by_cond["low_light"]["summary"]["at_val_th_acer"]["ACER"]

    # 6) Anti-shortcut Luma Audit & Overall Test Metric
    p_overall = np.concatenate(all_t_p + [ds_res["probs"], fr_res["probs"]])
    y_overall = np.concatenate(all_t_y + [ds_res["is_live"], fr_res["is_live"]])
    s_overall = np.asarray(all_t_src + ds_res["sources"] + fr_res["sources"])
    overall_m = compute_iso30107_metrics(p_overall, y_overall, threshold=th_acer)
    overall_a2 = compute_iso30107_metrics(p_overall, y_overall, threshold=th_apcer2)

    real_fps_mask = s_overall != "casia_fasd_zeroshot"
    rf_m = compute_iso30107_metrics(p_overall[real_fps_mask], y_overall[real_fps_mask], threshold=th_acer)
    luma_audit = audit_luma_anti_shortcut(np.concatenate(live_luma_list, axis=0), np.concatenate(spoof_luma_list, axis=0))

    report = {
        "ckpt": str(args.ckpt),
        "val_calibrated_thresholds": {
            "th_min_acer": th_acer,
            "th_apcer_2pct": th_apcer2,
            "th_apcer_1pct": th_apcer1,
            "val_all_auc": val_cal["auc"],
            "val_all_acer": val_cal["operating_point"]["ACER"],
        },
        "overall_test": {
            "auc": overall_m["auc"],
            "at_val_th_acer": overall_m["operating_point"],
            "at_val_th_apcer2": overall_a2["operating_point"],
            "n_live": overall_m["n_live"],
            "n_spoof": overall_m["n_spoof"],
        },
        "overall_test_real_fps": {
            "auc": rf_m["auc"],
            "at_val_th_acer": rf_m["operating_point"],
            "n_live": rf_m["n_live"],
            "n_spoof": rf_m["n_spoof"],
        },
        "lighting_matrix": {k: v["summary"] for k, v in test_by_cond.items()}
        | {"dim_screen": dim_screen_summary, "frozen_rigid": frozen_summary},
        "attack_breakdown": attack_breakdown,
        "domain_breakdown": domain_breakdown,
        "bbox_jitter_sensitivity": {
            "clean_low_light_acer": clean_ll_acer,
            "jitter_6pct_low_light_acer": jit_acer,
            "delta_acer": float(jit_acer - clean_ll_acer),
        },
        "luma_anti_shortcut_audit": luma_audit,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 3 Low-Light Temporal Liveness Evaluation Report (ISO/IEC 30107-3)",
        "",
        f"- **Checkpoint**: `{args.ckpt}`",
        f"- **Val-Calibrated Thresholds**: `th_min_acer = {th_acer:.4f}`, `th_apcer<=2% = {th_apcer2:.4f}`, `th_apcer<=1% = {th_apcer1:.4f}`",
        f"- **Overall Test (All 9 Domains)**: `AUC = {overall_m['auc']:.4f}`, `ACER@th_acer = {overall_m['operating_point']['ACER']*100:.2f}%`, "
        f"`BPCER@th_apcer2 = {overall_a2['operating_point']['BPCER']*100:.2f}%`, `APCER@th_apcer2 = {overall_a2['operating_point']['APCER']*100:.2f}%`",
        f"- **Real-FPS Test (Excl. 1-fps CASIA-2012)**: `AUC = {rf_m['auc']:.4f}`, `ACER@th_acer = {rf_m['operating_point']['ACER']*100:.2f}%`",
        f"- **Luma Anti-Shortcut Audit**: `live_mean={luma_audit['live_luma_mean']:.2f}`, `spoof_mean={luma_audit['spoof_luma_mean']:.2f}`, "
        f"`gap={luma_audit['mean_luma_gap']:.2f}`, `warning={luma_audit['warning_shortcut_risk']}`",
        f"- **Bbox ±6% Jitter Sensitivity (Low-Light)**: `clean ACER={clean_ll_acer*100:.2f}%` vs `jittered ACER={jit_acer*100:.2f}%` (`Δ={jit_acer-clean_ll_acer:+.4f}`)",
        "",
        "## Stratified Lighting Matrix (Val-Calibrated `th_min_acer`)",
        "",
        "| Condition | AUC | APCER (%) | BPCER (%) | ACER (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cond, s in report["lighting_matrix"].items():
        op = s["at_val_th_acer"]
        lines.append(f"| {cond} | {s['auc']:.4f} | {op['APCER']*100:.2f} | {op['BPCER']*100:.2f} | {op['ACER']*100:.2f} |")

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"val_calibrated_thresholds": report["val_calibrated_thresholds"], "overall_test": report["overall_test"]}, indent=2))


if __name__ == "__main__":
    main()
