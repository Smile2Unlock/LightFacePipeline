"""Phase 1 Detector training: Stage 1 (Supervised + HLA distill) & Stage 2 (EMA Mean-Teacher)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from common.degradations import PhysicalLowLightDegrader
from common.nms import nms
from common.runtime import (
    ModelEMA,
    get_amp_autocast,
    get_device,
    get_grad_scaler,
    get_sdpa_context,
    save_checkpoint,
    seed_everything,
)
from detector.dataset import (
    MixedFaceDetectorDataset,
    SyntheticDetectorSmokeDataset,
    load_darkface_labeled_specs,
    load_jsonl_face_specs,
)
from detector.losses import DetectorLoss, HLAFeatureDistillLoss
from detector.yolo11_face import LCFYOLO11Face5pt, build_detector_teacher
from detector.yunet_lcf import LCFYuNetS


def build_detector_student(cfg: Dict[str, Any]) -> nn.Module:
    arch = cfg["model"]["arch"]
    if arch == "lcf_yolo11n_5pt":
        return LCFYOLO11Face5pt(
            use_lcf=cfg["model"].get("use_lcf", True),
            use_p2=cfg["model"].get("use_p2", True),
            width_mult=cfg["model"].get("width_mult", 0.28),
            reg_max=cfg["model"].get("reg_max", 16),
        )
    if arch == "lcf_yunet_s":
        return LCFYuNetS(
            use_lcf=cfg["model"].get("use_lcf", True),
            width_scale=cfg["model"].get("width_scale", 1.5),
            reg_max=cfg["model"].get("reg_max", 8),
        )
    raise ValueError(f"Unknown detector arch: {arch}")


@torch.no_grad()
def generate_mean_teacher_pseudo_labels(
    ema_model: nn.Module,
    dark_imgs: torch.Tensor,
    conf_thresh: float = 0.80,
    iou_thresh: float = 0.45,
    max_faces: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate geometrically consistent pseudo-labels from EMA teacher on gently lifted dark images."""
    ema_model.eval()
    # Gentle linear brightness lift for EMA teacher view (no aggressive distortion)
    mean_luma = dark_imgs.mean(dim=(1, 2, 3), keepdim=True).clamp(min=0.02)
    lifted = (dark_imgs * (0.35 / mean_luma).clamp(1.0, 4.5)).clamp(0.0, 1.0)
    out = ema_model(lifted)
    b = dark_imgs.shape[0]
    h, w = dark_imgs.shape[-2:]
    p_boxes = torch.zeros((b, max_faces, 4), device=dark_imgs.device, dtype=torch.float32)
    p_kpts = torch.zeros((b, max_faces, 5, 2), device=dark_imgs.device, dtype=torch.float32)
    p_valid = torch.zeros((b, max_faces), device=dark_imgs.device, dtype=torch.bool)

    for i in range(b):
        scores = out["scores"][i]
        keep_mask = scores >= conf_thresh
        if not keep_mask.any():
            continue
        boxes_i = out["boxes_xyxy"][i][keep_mask]
        scores_i = scores[keep_mask]
        kpts_i = out["kpts"][i][keep_mask, :, :2]
        keep_idx = nms(boxes_i, scores_i, iou_threshold=iou_thresh)[:max_faces]
        for j, idx in enumerate(keep_idx):
            bx = boxes_i[idx]
            kp = kpts_i[idx]
            # Geometric self-consistency: all 5 keypoints lie inside expanded box (1.1x)
            bw = (bx[2] - bx[0]).clamp(min=1.0)
            bh = (bx[3] - bx[1]).clamp(min=1.0)
            in_x = (kp[:, 0] >= bx[0] - 0.1 * bw) & (kp[:, 0] <= bx[2] + 0.1 * bw)
            in_y = (kp[:, 1] >= bx[1] - 0.1 * bh) & (kp[:, 1] <= bx[3] + 0.1 * bh)
            eye_sep_ok = torch.norm(kp[0] - kp[1]) > 0.12 * bw
            if bool((in_x & in_y).all().item() and eye_sep_ok.item()):
                p_boxes[i, j] = torch.tensor(
                    [bx[0].clamp(0, w), bx[1].clamp(0, h), bx[2].clamp(0, w), bx[3].clamp(0, h)],
                    device=dark_imgs.device,
                )
                p_kpts[i, j] = kp
                p_valid[i, j] = True
    return p_boxes, p_kpts, p_valid


def train_one_epoch_detector(
    student: nn.Module,
    teacher: Optional[nn.Module],
    ema: ModelEMA,
    loader: DataLoader,
    det_loss_fn: DetectorLoss,
    hla_loss_fn: Optional[HLAFeatureDistillLoss],
    degrader: PhysicalLowLightDegrader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "fp16",
    sdpa_backend: str = "flash_attention",
    hla_weight: float = 0.5,
    stage2_semi: bool = False,
    pseudo_conf_thresh: float = 0.80,
    multiscales: Tuple[int, ...] = (320, 480, 640),
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    student.train()
    if teacher is not None:
        teacher.eval()
    scaler = get_grad_scaler(device, precision)

    stats = {"loss": 0.0, "det_loss": 0.0, "hla_loss": 0.0, "semi_loss": 0.0}
    steps = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        img_clean = batch["clean_u8"].to(device, non_blocking=True).float().div_(255.0)
        img_raw = batch["img_u8"].to(device, non_blocking=True).float().div_(255.0)
        gt_boxes = batch["gt_boxes"].to(device, non_blocking=True)
        gt_kpts = batch["gt_kpts"].to(device, non_blocking=True)
        gt_valid = batch["gt_valid"].to(device, non_blocking=True)
        has_lm = batch["has_landmarks"].to(device, non_blocking=True)
        apply_deg = batch["apply_degrade"].to(device, non_blocking=True)
        is_unlabeled = batch["is_unlabeled"].to(device, non_blocking=True)

        # Multi-scale resize on GPU cleanly aligned to 32x
        target_sz = multiscales[step % len(multiscales)]
        cur_h, cur_w = img_raw.shape[-2:]
        if target_sz != cur_h or target_sz != cur_w:
            scale_f = float(target_sz) / float(cur_h)
            img_raw = F.interpolate(img_raw, size=(target_sz, target_sz), mode="bilinear", align_corners=False)
            img_clean = F.interpolate(img_clean, size=(target_sz, target_sz), mode="bilinear", align_corners=False)
            gt_boxes = gt_boxes * scale_f
            gt_kpts = gt_kpts * scale_f

        # Vectorized GPU physical low-light degradation for clean sources (never double-darken DarkFace)
        img_student = img_raw.clone()
        deg_idx = (apply_deg & (torch.rand(img_raw.shape[0], device=device) < 0.5)).nonzero(as_tuple=False).squeeze(1)
        if deg_idx.numel() > 0:
            # Pass normalized first GT face bbox so backlight field dims the face & boosts window background
            bb0 = gt_boxes[deg_idx, 0] / float(target_sz)
            img_student[deg_idx] = degrader(img_raw[deg_idx], bboxes_norm=bb0)

        optimizer.zero_grad(set_to_none=True)
        with get_sdpa_context(sdpa_backend), get_amp_autocast(device, precision):
            out_s = student(img_student)

            sup_mask = ~is_unlabeled
            det_loss = img_student.new_zeros((), dtype=torch.float32)
            if sup_mask.any():
                sup_out = {
                    k: (v[sup_mask] if isinstance(v, torch.Tensor) else [f[sup_mask] for f in v])
                    for k, v in out_s.items()
                }
                d_res = det_loss_fn(
                    sup_out,
                    (target_sz, target_sz),
                    gt_boxes[sup_mask],
                    gt_valid[sup_mask],
                    gt_kpts[sup_mask],
                    has_lm[sup_mask],
                )
                det_loss = d_res["loss"]

            hla_loss = img_student.new_zeros((), dtype=torch.float32)
            if teacher is not None and hla_loss_fn is not None:
                with torch.no_grad():
                    out_t = teacher(img_clean)
                hla_loss = hla_loss_fn(out_s["fpn_features"], out_t["fpn_features"]) * hla_weight

            semi_loss = img_student.new_zeros((), dtype=torch.float32)
            if stage2_semi and is_unlabeled.any():
                p_boxes, p_kpts, p_valid = generate_mean_teacher_pseudo_labels(
                    ema.ema, img_student[is_unlabeled], conf_thresh=pseudo_conf_thresh
                )
                if p_valid.any():
                    semi_out = {
                        k: (v[is_unlabeled] if isinstance(v, torch.Tensor) else [f[is_unlabeled] for f in v])
                        for k, v in out_s.items()
                    }
                    p_has_lm = torch.ones((p_boxes.shape[0],), device=device, dtype=torch.bool)
                    semi_loss = det_loss_fn(
                        semi_out, (target_sz, target_sz), p_boxes, p_valid, p_kpts, p_has_lm
                    )["loss"] * 0.5

            loss = det_loss + hla_loss + semi_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(student)

        stats["loss"] += float(loss.detach().item())
        stats["det_loss"] += float(det_loss.detach().item())
        stats["hla_loss"] += float(hla_loss.detach().item())
        stats["semi_loss"] += float(semi_loss.detach().item())
        steps += 1

    for k in stats:
        stats[k] /= max(steps, 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Detector Training")
    parser.add_argument("--config", type=Path, default=Path("configs/detector_yolo11n_5pt_lcf.yaml"))
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))
    device = get_device()

    student = build_detector_student(cfg).to(device)
    teacher = build_detector_teacher(use_p2=cfg["model"].get("use_p2", True)).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    ema = ModelEMA(student, decay=cfg["train"].get("ema_decay", 0.9999))

    det_loss_fn = DetectorLoss(strides=student.head.strides, reg_max=student.head.reg_max).to(device)
    hla_loss_fn = HLAFeatureDistillLoss(student.fpn_channels, teacher.fpn_channels).to(device)
    degrader = PhysicalLowLightDegrader(mode="detector").to(device)

    lcf_params = list(student.lcf.parameters()) if hasattr(student, "lcf") else []
    lcf_ids = {id(p) for p in lcf_params}
    head_params = [p for p in student.head.parameters() if id(p) not in lcf_ids]
    head_ids = lcf_ids | {id(p) for p in head_params}
    backbone_params = [p for p in student.parameters() if id(p) not in head_ids]

    lr_bb = cfg["train"].get("lr_backbone", 2e-4)
    lr_hd = cfg["train"].get("lr_head_lcf", 1e-3)
    if args.stage == 2:
        lr_bb = cfg["train"].get("stage2_lr", 5e-5)
        lr_hd = cfg["train"].get("stage2_lr", 5e-5)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_bb},
            {"params": lcf_params + head_params + list(hla_loss_fn.parameters()), "lr": lr_hd},
        ],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    if args.smoke:
        ds = SyntheticDetectorSmokeDataset(n=16, img_size=320)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        res = train_one_epoch_detector(
            student=student,
            teacher=teacher,
            ema=ema,
            loader=loader,
            det_loss_fn=det_loss_fn,
            hla_loss_fn=hla_loss_fn,
            degrader=degrader,
            optimizer=optimizer,
            device=device,
            precision=cfg["train"].get("precision", "fp16"),
            sdpa_backend=cfg["train"].get("sdpa_backend", "flash_attention"),
            stage2_semi=(args.stage == 2),
            multiscales=(320,),
            max_steps=2,
        )
        print("Smoke detector train metrics:", res)
        return

    # Assemble real 5-source dataset from configured manifests
    specs = []
    data_cfg = cfg["data"]
    if data_cfg.get("darkface_dir") and data_cfg.get("darkface_split"):
        specs.extend(load_darkface_labeled_specs(Path(data_cfg["darkface_dir"]), Path(data_cfg["darkface_split"]), "train"))
    for key, src_name, has_lm, deg, zoom in [
        ("wider_train_jsonl", "wider", True, True, True),
        ("closeup_5pt_jsonl", "closeup_5pt", True, True, False),
        ("darkface_paired_jsonl", "darkface_paired", False, False, False),
        ("darkface_unlabeled_jsonl", "darkface_unlabeled", False, False, False),
        ("hard_neg_jsonl", "hard_neg", False, True, False),
    ]:
        p = data_cfg.get(key)
        if p and Path(p).is_file():
            specs.extend(load_jsonl_face_specs(Path(p), src_name, has_lm, deg, zoom))
    if not specs:
        raise RuntimeError("No training datasets found in config paths. Run check_datasets.py or use --smoke.")

    ds = MixedFaceDetectorDataset(specs, img_size=max(cfg["train"].get("multiscales", [640])))
    sampler = ds.build_weighted_sampler()
    loader = DataLoader(
        ds,
        batch_size=cfg["train"].get("batch_size", 128),
        sampler=sampler,
        num_workers=cfg["train"].get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )

    epochs = cfg["train"].get("stage1_epochs", 50) if args.stage == 1 else cfg["train"].get("stage2_epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    run_dir = Path(cfg["train"].get("run_dir", "runs/detector"))
    backup_dir = Path(cfg["train"]["backup_dir"]) if cfg["train"].get("backup_dir") else None

    for epoch in range(1, epochs + 1):
        stats = train_one_epoch_detector(
            student=student,
            teacher=teacher,
            ema=ema,
            loader=loader,
            det_loss_fn=det_loss_fn,
            hla_loss_fn=hla_loss_fn,
            degrader=degrader,
            optimizer=optimizer,
            device=device,
            precision=cfg["train"].get("precision", "fp16"),
            sdpa_backend=cfg["train"].get("sdpa_backend", "flash_attention"),
            stage2_semi=(args.stage == 2),
            multiscales=tuple(cfg["train"].get("multiscales", [320, 480, 640])),
        )
        scheduler.step()
        print(f"[Stage {args.stage}][Epoch {epoch}/{epochs}] {stats}")
        if epoch % 5 == 0 or epoch == epochs:
            save_checkpoint(
                run_dir / f"detector_s{args.stage}_epoch{epoch}.pt",
                student,
                optimizer=optimizer,
                ema=ema,
                epoch=epoch,
                metrics=stats,
                backup_dir=backup_dir,
            )


if __name__ == "__main__":
    main()
