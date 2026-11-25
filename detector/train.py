"""Phase 1 Detector training: teacher, clean_baseline, stage1 (Low-Light + HLA), stage2 (Mean-Teacher)."""

from __future__ import annotations

import argparse
import json
import math
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
from detector.eval_darkface import evaluate_darkface
from detector.eval_phase1 import evaluate_wflw_5pt_nme, evaluate_wider_val
from detector.losses import DetectorLoss, HLAFeatureDistillLoss
from detector.yolo11_face import LCFYOLO11Face5pt, build_detector_teacher
from detector.yunet_lcf import LCFYuNetS


def build_detector_student(cfg: Dict[str, Any], force_lcf: Optional[bool] = None) -> nn.Module:
    arch = cfg["model"]["arch"]
    use_lcf = cfg["model"].get("use_lcf", True) if force_lcf is None else force_lcf
    if arch == "lcf_yolo11n_5pt":
        return LCFYOLO11Face5pt(
            use_lcf=use_lcf,
            use_p2=cfg["model"].get("use_p2", True),
            width_mult=cfg["model"].get("width_mult", 0.22),
            reg_max=cfg["model"].get("reg_max", 12),
        )
    if arch == "lcf_yunet_s":
        return LCFYuNetS(
            use_lcf=use_lcf,
            width_scale=cfg["model"].get("width_scale", 1.5),
            reg_max=cfg["model"].get("reg_max", 8),
        )
    raise ValueError(f"Unknown detector arch: {arch}")


def freeze_batchnorm(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


@torch.no_grad()
def generate_mean_teacher_pseudo_labels(
    ema_model: nn.Module,
    dark_imgs: torch.Tensor,
    conf_thresh: float = 0.75,
    iou_thresh: float = 0.45,
    max_faces: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate geometrically consistent pseudo-labels from EMA teacher on gently lifted dark images."""
    ema_model.eval()
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


def _pick_random_valid_bboxes_norm(
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    indices: torch.Tensor,
    img_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split indices into face-containing (with random valid GT bbox) and 0-face hard-neg indices."""
    face_idx: List[int] = []
    face_bbs: List[torch.Tensor] = []
    neg_idx: List[int] = []
    for idx in indices.tolist():
        valid_pos = gt_valid[idx].nonzero(as_tuple=False).squeeze(1)
        if valid_pos.numel() > 0:
            pick = valid_pos[torch.randint(0, valid_pos.numel(), (1,)).item()]
            face_idx.append(idx)
            face_bbs.append((gt_boxes[idx, pick] / float(img_size)).clamp(0.0, 1.0))
        else:
            neg_idx.append(idx)
    device = gt_boxes.device
    f_t = torch.tensor(face_idx, dtype=torch.long, device=device)
    n_t = torch.tensor(neg_idx, dtype=torch.long, device=device)
    bb_t = torch.stack(face_bbs, dim=0) if face_bbs else torch.zeros((0, 4), device=device)
    return f_t, bb_t, n_t


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
    pseudo_conf_thresh: float = 0.75,
    multiscales: Tuple[int, ...] = (320, 480, 640),
    degrade_prob: float = 0.5,
    max_steps: Optional[int] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_accum: int = 1,
) -> Dict[str, float]:
    student.train()
    if stage2_semi:
        freeze_batchnorm(student)
    if teacher is not None:
        teacher.eval()
    scaler = scaler or get_grad_scaler(device, precision)
    grad_accum = max(1, int(grad_accum))

    def _optimizer_step() -> None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(student)

    stats = {"loss": 0.0, "det_loss": 0.0, "hla_loss": 0.0, "semi_loss": 0.0}
    steps = 0
    accum_counter = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        img_clean = batch["clean_u8"].to(device, non_blocking=True).float().div_(255.0)
        img_raw = batch["img_u8"].to(device, non_blocking=True).float().div_(255.0)
        gt_boxes = batch["gt_boxes"].to(device, non_blocking=True)
        gt_kpts = batch["gt_kpts"].to(device, non_blocking=True)
        gt_valid = batch["gt_valid"].to(device, non_blocking=True)
        gt_kpts_valid = batch["gt_kpts_valid"].to(device, non_blocking=True)
        has_lm = batch["has_landmarks"].to(device, non_blocking=True)
        has_clean_pair = batch["has_clean_pair"].to(device, non_blocking=True)
        apply_deg = batch["apply_degrade"].to(device, non_blocking=True)
        is_unlabeled = batch["is_unlabeled"].to(device, non_blocking=True)

        target_sz = multiscales[(step // grad_accum) % len(multiscales)]
        cur_h = img_raw.shape[-2]
        if target_sz != cur_h:
            scale_f = float(target_sz) / float(cur_h)
            img_raw = F.interpolate(img_raw, size=(target_sz, target_sz), mode="bilinear", align_corners=False)
            img_clean = F.interpolate(img_clean, size=(target_sz, target_sz), mode="bilinear", align_corners=False)
            gt_boxes = gt_boxes * scale_f
            gt_kpts = gt_kpts * scale_f

        img_student = img_raw.clone()
        deg_mask = apply_deg & (torch.rand(img_raw.shape[0], device=device) < degrade_prob)
        deg_idx = deg_mask.nonzero(as_tuple=False).squeeze(1)
        if deg_idx.numel() > 0:
            f_idx, bb_norm, n_idx = _pick_random_valid_bboxes_norm(gt_boxes, gt_valid, deg_idx, target_sz)
            if f_idx.numel() > 0:
                img_student[f_idx] = degrader(img_raw[f_idx], bboxes_norm=bb_norm)
            if n_idx.numel() > 0:
                img_student[n_idx] = degrader(img_raw[n_idx], bboxes_norm=None)

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
                    gt_kpts_valid[sup_mask],
                )
                det_loss = d_res["loss"]

            # HLA distillation STRICTLY gated to samples with true normal-light paired teacher views
            hla_loss = img_student.new_zeros((), dtype=torch.float32)
            if teacher is not None and hla_loss_fn is not None and has_clean_pair.any():
                pair_mask = has_clean_pair
                with torch.no_grad():
                    out_t = teacher(img_clean[pair_mask])
                s_feats_pair = [f[pair_mask] for f in out_s["fpn_features"]]
                hla_loss = hla_loss_fn(s_feats_pair, out_t["fpn_features"]) * hla_weight

            semi_loss = img_student.new_zeros((), dtype=torch.float32)
            if stage2_semi and is_unlabeled.any():
                p_boxes, p_kpts, p_valid = generate_mean_teacher_pseudo_labels(
                    ema.ema, img_student[is_unlabeled], conf_thresh=pseudo_conf_thresh
                )
                has_pseudo = p_valid.any(dim=1)
                if has_pseudo.any():
                    unlab_idx = is_unlabeled.nonzero(as_tuple=False).squeeze(1)[has_pseudo]
                    semi_out = {
                        k: (v[unlab_idx] if isinstance(v, torch.Tensor) else [f[unlab_idx] for f in v])
                        for k, v in out_s.items()
                    }
                    pb_sel = p_boxes[has_pseudo]
                    pv_sel = p_valid[has_pseudo]
                    pk_sel = p_kpts[has_pseudo]
                    p_has_lm = torch.ones((pb_sel.shape[0],), device=device, dtype=torch.bool)
                    semi_loss = (
                        det_loss_fn(semi_out, (target_sz, target_sz), pb_sel, pv_sel, pk_sel, p_has_lm)["loss"]
                        * 0.5
                    )

            loss = det_loss + hla_loss + semi_loss

        scaler.scale(loss / float(grad_accum)).backward()
        accum_counter += 1
        if accum_counter >= grad_accum:
            _optimizer_step()
            accum_counter = 0

        stats["loss"] += float(loss.detach().item())
        stats["det_loss"] += float(det_loss.detach().item())
        stats["hla_loss"] += float(hla_loss.detach().item())
        stats["semi_loss"] += float(semi_loss.detach().item())
        steps += 1

    if accum_counter > 0:
        _optimizer_step()

    for k in stats:
        stats[k] /= max(steps, 1)
    return stats


def _warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 3,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(warmup_epochs, 1))
        progress = float(epoch_idx - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Detector Training")
    parser.add_argument("--config", type=Path, default=Path("configs/detector_yolo11n_5pt_lcf.yaml"))
    parser.add_argument(
        "--mode",
        default="stage1",
        choices=["teacher", "clean_baseline", "stage1", "stage2"],
        help="Training phase mode",
    )
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--teacher-ckpt", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.mode == "teacher":
        student = build_detector_teacher(use_p2=cfg["model"].get("use_p2", True)).to(device)
    elif args.mode == "clean_baseline":
        student = build_detector_student(cfg, force_lcf=False).to(device)
    else:
        student = build_detector_student(cfg, force_lcf=True).to(device)

    if args.init_ckpt is not None:
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        state = ck.get("ema", ck.get("model", ck))
        # strict=False cleanly lets zero-init LCF stay identity when loading clean_baseline weights!
        missing, unexpected = student.load_state_dict(state, strict=False)
        print(f"Loaded init checkpoint {args.init_ckpt}: missing={missing}, unexpected={unexpected}")

    teacher = None
    hla_loss_fn = None
    if args.mode == "stage1" and not args.smoke:
        if args.teacher_ckpt is None or not args.teacher_ckpt.is_file():
            raise FileNotFoundError("Stage 1 requires a trained --teacher-ckpt for HLA distillation")
        teacher = build_detector_teacher(use_p2=cfg["model"].get("use_p2", True)).to(device)
        t_ck = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        teacher.load_state_dict(t_ck.get("ema", t_ck.get("model", t_ck)))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        hla_loss_fn = HLAFeatureDistillLoss(student.fpn_channels, teacher.fpn_channels).to(device)
    elif args.smoke and args.mode == "stage1":
        teacher = build_detector_teacher(use_p2=cfg["model"].get("use_p2", True)).to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        hla_loss_fn = HLAFeatureDistillLoss(student.fpn_channels, teacher.fpn_channels).to(device)

    ema = ModelEMA(
        student,
        decay=cfg["train"].get("ema_decay", 0.9995),
        warmup_tau=(0.0 if args.mode == "stage2" else 500.0),
    )
    det_loss_fn = DetectorLoss(strides=student.head.strides, reg_max=student.head.reg_max).to(device)
    degrader = PhysicalLowLightDegrader(mode="detector").to(device)

    lcf_params = list(student.lcf.parameters()) if hasattr(student, "lcf") and isinstance(student.lcf, nn.Module) else []
    lcf_ids = {id(p) for p in lcf_params}
    head_params = [p for p in student.head.parameters() if id(p) not in lcf_ids]
    head_ids = lcf_ids | {id(p) for p in head_params}
    backbone_params = [p for p in student.parameters() if id(p) not in head_ids]

    if args.mode in ("teacher", "clean_baseline"):
        lr_bb = 1e-3
        lr_hd = 1e-3
    elif args.mode == "stage2":
        lr_bb = cfg["train"].get("stage2_lr", 5e-5)
        lr_hd = cfg["train"].get("stage2_lr", 5e-5)
    else:
        lr_bb = cfg["train"].get("lr_backbone", 2e-4)
        lr_hd = cfg["train"].get("lr_head_lcf", 1e-3)

    extra_params = list(hla_loss_fn.parameters()) if hla_loss_fn is not None else []
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_bb},
            {"params": lcf_params + head_params + extra_params, "lr": lr_hd},
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
            stage2_semi=(args.mode == "stage2"),
            multiscales=(320,),
            max_steps=2,
        )
        print("Smoke detector train metrics:", res)
        return

    data_cfg = cfg["data"]
    specs = []
    custom_weights: Dict[str, float] = {}

    if args.mode in ("teacher", "clean_baseline"):
        # Normal-light WIDER Train + WFLW Train (no darkening)
        specs.extend(load_jsonl_face_specs(Path(data_cfg["wider_train_jsonl"]), "wider", True, False, False, True))
        specs.extend(load_jsonl_face_specs(Path(data_cfg["wflw_train_jsonl"]), "closeup_5pt", True, False, False, False))
        custom_weights = (
            {"wider": 0.70, "closeup_5pt": 0.30}
            if args.mode == "clean_baseline"
            else {"wider": 0.75, "closeup_5pt": 0.25}
        )
        degrade_prob = 0.0
    else:
        # Low-light 5-source weighted mix
        specs.extend(load_jsonl_face_specs(Path(data_cfg["wider_train_jsonl"]), "wider", True, True, True, True))
        specs.extend(load_darkface_labeled_specs(Path(data_cfg["darkface_dir"]), Path(data_cfg["darkface_split"]), "train"))
        specs.extend(load_jsonl_face_specs(Path(data_cfg["wflw_train_jsonl"]), "closeup_5pt", True, True, True, False))
        specs.extend(load_jsonl_face_specs(Path(data_cfg["hard_neg_train_jsonl"]), "hard_neg", False, False, True, False))
        if args.mode == "stage2":
            specs.extend(load_jsonl_face_specs(Path(data_cfg["exdark_unlabeled_jsonl"]), "exdark_unlabeled", False, False, False, False))
            custom_weights = {"wider": 0.35, "darkface_labeled": 0.20, "exdark_unlabeled": 0.15, "closeup_5pt": 0.15, "hard_neg": 0.15}
        else:
            custom_weights = {"wider": 0.40, "darkface_labeled": 0.25, "closeup_5pt": 0.20, "hard_neg": 0.15}
        degrade_prob = 0.65

    bg_paths: List[Path] = []
    if data_cfg.get("hard_neg_train_jsonl") and Path(data_cfg["hard_neg_train_jsonl"]).is_file():
        bg_paths = [
            Path(json.loads(ln)["image_path"])
            for ln in Path(data_cfg["hard_neg_train_jsonl"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    img_max = max(cfg["train"].get("multiscales", [640]))
    ds = MixedFaceDetectorDataset(
        specs,
        img_size=img_max,
        max_faces=128,
        augment=True,
        seed=cfg.get("seed", 42),
        bg_image_paths=bg_paths,
    )
    batch_size = args.batch_size or cfg["train"].get("batch_size", 64)
    grad_accum = max(1, int(args.grad_accum or cfg["train"].get("grad_accum", 1)))
    num_workers = int(cfg["train"].get("num_workers", 4))
    # Fixed clean epoch budget (~16k samples/epoch for clean/stage1; ~12k for stage2)
    samples_per_epoch = 12000 if args.mode == "stage2" else 16000
    sampler = ds.build_weighted_sampler(num_samples=samples_per_epoch, custom_weights=custom_weights)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    default_epochs = {"teacher": 35, "clean_baseline": 45, "stage1": cfg["train"].get("stage1_epochs", 50), "stage2": cfg["train"].get("stage2_epochs", 20)}[args.mode]
    epochs = args.epochs or default_epochs
    scheduler = _warmup_cosine_scheduler(optimizer, total_epochs=epochs, warmup_epochs=min(3, epochs))

    run_base = Path(cfg["train"].get("run_dir", "runs/detector"))
    run_dir = run_base / (args.run_name or args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(cfg["train"]["backup_dir"]) / (args.run_name or args.mode) if cfg["train"].get("backup_dir") else None
    history_path = run_dir / "history.jsonl"

    wflw_test_path = Path(data_cfg["wflw_test_jsonl"]) if data_cfg.get("wflw_test_jsonl") else None
    wider_val_path = Path(data_cfg["wider_val_jsonl"]) if data_cfg.get("wider_val_jsonl") else None
    df_dir = Path(data_cfg["darkface_dir"]) if data_cfg.get("darkface_dir") else None
    df_split = Path(data_cfg["darkface_split"]) if data_cfg.get("darkface_split") else None
    scaler = get_grad_scaler(device, cfg["train"].get("precision", "fp16"))
    best_score = -1e9

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
            hla_weight=cfg["train"].get("hla_weight", 0.5),
            stage2_semi=(args.mode == "stage2"),
            pseudo_conf_thresh=cfg["train"].get("pseudo_conf_thresh", 0.75),
            multiscales=tuple(cfg["train"].get("multiscales", [320, 480, 640])),
            degrade_prob=degrade_prob,
            scaler=scaler,
            grad_accum=grad_accum,
        )
        scheduler.step()
        stats["epoch"] = epoch
        stats["lr_head"] = round(optimizer.param_groups[-1]["lr"], 7)

        if epoch % args.eval_every == 0 or epoch == epochs:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            clean_rec, clean_nme = 0.0, 1.0
            dark_rec, dark_nme = 0.0, 1.0
            if wflw_test_path and wflw_test_path.is_file():
                eval_res = evaluate_wflw_5pt_nme(ema.ema, wflw_test_path, device=device, max_images=300)
                clean_nme = eval_res["clean_nme_5pt"]
                dark_nme = eval_res["dark_nme_5pt"]
                clean_rec = eval_res["clean_recall"]
                dark_rec = eval_res["dark_recall"]
                stats["wflw_clean_nme"] = clean_nme
                stats["wflw_dark_nme"] = dark_nme
                stats["wflw_clean_med"] = eval_res.get("clean_nme_median", clean_nme)
                stats["wflw_dark_med"] = eval_res.get("dark_nme_median", dark_nme)
                stats["wflw_clean_rec"] = clean_rec
                stats["wflw_dark_rec"] = dark_rec

            if wider_val_path and wider_val_path.is_file():
                w_res = evaluate_wider_val(ema.ema, wider_val_path, device=device, max_images=300)
                stats["wider_easy"] = w_res["AP50_easy"]
                stats["wider_med"] = w_res["AP50_medium"]
                stats["wider_hard"] = w_res["AP50_hard"]

            if args.mode in ("stage1", "stage2") and df_dir and df_dir.is_dir() and df_split and df_split.is_file():
                df_res = evaluate_darkface(ema.ema, df_dir, df_split, device=device, conf_thresh=0.05, split="val")
                stats["darkface_ap50"] = df_res["AP50_official_tools_style"]
                stats["darkface_greedy"] = df_res["AP50_VOC_greedy"]
                stats["darkface_rec"] = df_res["recall_iou50"]

            if args.mode in ("teacher", "clean_baseline"):
                score = 0.5 * stats.get("wider_med", 0.0) + 0.5 * clean_rec - 3.0 * clean_nme
            else:
                score = (
                    0.5 * stats.get("darkface_ap50", 0.0)
                    + 0.25 * stats.get("wider_med", 0.0)
                    + 0.25 * dark_rec
                    - 3.0 * dark_nme
                )
            if score > best_score:
                best_score = score
                save_checkpoint(run_dir / "best.pt", student, optimizer=optimizer, ema=ema, epoch=epoch, metrics=stats, backup_dir=backup_dir)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        with open(history_path, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(stats) + "\n")
        print(f"[{args.mode}][Epoch {epoch}/{epochs}] {stats}", flush=True)

        if epoch % 10 == 0 or epoch == epochs:
            save_checkpoint(run_dir / f"epoch_{epoch}.pt", student, optimizer=optimizer, ema=ema, epoch=epoch, metrics=stats, backup_dir=backup_dir)


if __name__ == "__main__":
    main()
