"""Phase 2 Recognizer Training: CVLface KP-RPE Teacher + AdaFace + Rank-Compatible AdaDistill -> EdgeFace-S/XS."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from common.degradations import PhysicalLowLightDegrader
from common.runtime import (
    ModelEMA,
    get_amp_autocast,
    get_device,
    get_grad_scaler,
    get_sdpa_context,
    save_checkpoint,
    seed_everything,
)
from recognizer.dataset import (
    IndexedBlobRecognitionDataset,
    SyntheticRecognizerSmokeDataset,
    TwinViewRecognitionDataset,
)
from recognizer.edgeface import OfficialEdgeFace, build_recognizer_student
from recognizer.eval_phase2 import build_bin_protocol_cache, evaluate_fast_val
from recognizer.losses import AdaDistillLoss
from recognizer.teacher_cvlface import CVLfaceTeacherWrapper


@torch.no_grad()
def align_student_head_procrustes(
    student: nn.Module,
    dataset: Any,
    device: torch.device,
    teacher: Optional[nn.Module] = None,
    n_samples: int = 4096,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Step-0 orthogonal Procrustes rotation aligning OfficialEdgeFace's head to Teacher's hypersphere."""
    if not (isinstance(student, OfficialEdgeFace) and hasattr(student.backbone, "model")):
        return {"applied": False, "reason": "no_unnormalized_final_linear"}

    fc = getattr(student.backbone.model.head, "fc", None)
    final_lin = getattr(fc, "linear2", fc)
    if not isinstance(final_lin, nn.Linear):
        return {"applied": False, "reason": "final_fc_not_linear"}

    total_n = len(dataset)
    if total_n == 0:
        return {"applied": False, "reason": "empty_dataset"}

    take_n = min(int(n_samples), total_n)
    indices = np.linspace(0, total_n - 1, take_n, dtype=np.int64).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)

    was_training = student.training
    student.eval()
    zs_list: List[torch.Tensor] = []
    norms_list: List[torch.Tensor] = []
    zt_list: List[torch.Tensor] = []
    first_x: Optional[torch.Tensor] = None

    for batch in loader:
        clean_01 = batch["clean_u8"].to(device, non_blocking=True).float().div_(255.0)
        x_clean = (clean_01 - 0.5) / 0.5
        if first_x is None:
            first_x = x_clean[: min(256, x_clean.shape[0])]
        z_s, norm_s = student(x_clean)
        if "teacher_emb" in batch:
            z_t = F.normalize(batch["teacher_emb"].to(device, non_blocking=True).float(), dim=1)
        elif teacher is not None:
            kpts_norm = batch["kpts_norm"].to(device, non_blocking=True)
            z_t = teacher(x_clean, kpts_norm)
        else:
            return {"applied": False, "reason": "missing_teacher_or_precomputed_emb"}
        zs_list.append(z_s.float())
        norms_list.append(norm_s.squeeze(-1).float())
        zt_list.append(z_t.float())

    z_all = torch.cat(zs_list, dim=0)
    norms_all = torch.cat(norms_list, dim=0)
    zt_all = torch.cat(zt_list, dim=0)

    cos_before = float((z_all * zt_all).sum(dim=1).mean().item())
    assert first_x is not None

    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
    prev_tf32_cu = torch.backends.cudnn.allow_tf32 if device.type == "cuda" else None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        z_sub_before, _ = student(first_x)
        pair_before = z_sub_before @ z_sub_before.t()

        # Exact orthogonal Procrustes in FP64: minimize || Z_s @ R - Z_t ||_F
        m_mat = z_all.double().t() @ zt_all.double()
        u_mat, _, vt_mat = torch.linalg.svd(m_mat, full_matrices=False)
        rot = (u_mat @ vt_mat).float()

        rot_t = rot.t().to(device=final_lin.weight.device, dtype=final_lin.weight.dtype)
        final_lin.weight.data.copy_(rot_t @ final_lin.weight.data)
        if final_lin.bias is not None:
            final_lin.bias.data.copy_(rot_t @ final_lin.bias.data)

        z_sub_after, _ = student(first_x)
        pair_after = z_sub_after @ z_sub_after.t()
        max_pair_cos_diff = float((pair_after - pair_before).abs().max().item())
    finally:
        if device.type == "cuda" and prev_tf32_mm is not None and prev_tf32_cu is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
            torch.backends.cudnn.allow_tf32 = prev_tf32_cu

    z_rot_all = F.normalize(z_all @ rot.to(z_all.device), dim=1)
    cos_after = float((z_rot_all * zt_all).sum(dim=1).mean().item())

    if was_training:
        student.train()

    return {
        "applied": True,
        "n_samples": int(z_all.shape[0]),
        "cos_teacher_before": round(cos_before, 5),
        "cos_teacher_after": round(cos_after, 5),
        "max_pair_cos_diff": max_pair_cos_diff,
        "norm_mean": round(float(norms_all.mean().item()), 4),
        "norm_std": round(float(norms_all.std(unbiased=False).item()), 4),
    }


@torch.no_grad()
def calibrate_adaface_norm_stats(
    student: nn.Module,
    dataset: Any,
    device: torch.device,
    n_samples: int = 4096,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Measure student pre-normalization feature norm mean/std on clean face samples."""
    total_n = len(dataset)
    if total_n == 0:
        return {"norm_mean": 20.0, "norm_std": 10.0}
    take_n = min(int(n_samples), total_n)
    indices = np.linspace(0, total_n - 1, take_n, dtype=np.int64).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    was_training = student.training
    student.eval()
    norms: List[torch.Tensor] = []
    for batch in loader:
        clean_01 = batch["clean_u8"].to(device, non_blocking=True).float().div_(255.0)
        _, norm_s = student((clean_01 - 0.5) / 0.5)
        norms.append(norm_s.squeeze(-1).float())
    if was_training:
        student.train()
    all_n = torch.cat(norms, dim=0)
    return {
        "norm_mean": round(float(all_n.mean().item()), 4),
        "norm_std": round(float(all_n.std(unbiased=False).item()), 4),
    }


def set_recognizer_warmup_trainable(student: nn.Module, freeze_early: bool) -> None:
    """Toggle early backbone stages trainability (`freeze_early=True` trains late stages + head)."""
    if (
        isinstance(student, OfficialEdgeFace)
        and hasattr(student.backbone, "model")
        and hasattr(student.backbone.model, "stages")
    ):
        core = student.backbone.model
        for p in student.parameters():
            p.requires_grad_(not freeze_early)
        if freeze_early:
            for m in (core.stages[2], core.stages[3], core.head):
                for p in m.parameters():
                    p.requires_grad_(True)
        return

    if not hasattr(student, "stages"):
        for p in student.parameters():
            p.requires_grad_(True)
        return
    for p in student.parameters():
        p.requires_grad_(not freeze_early)
    if freeze_early:
        for m in (
            student.downsample_layers[2],
            student.downsample_layers[3],
            student.stages[2],
            student.stages[3],
            student.head_norm,
            student.head_fc,
            student.head_bn,
        ):
            for p in m.parameters():
                p.requires_grad_(True)


def _split_optimizer_param_groups(
    student: nn.Module,
    criterion: nn.Module,
    lr_backbone: float,
    lr_head: float,
    lr_early: Optional[float] = None,
    lr_student_head: Optional[float] = None,
) -> List[Dict[str, Any]]:
    early_lr = float(lr_early if lr_early is not None else lr_backbone * 0.5)
    head_s_lr = float(lr_student_head if lr_student_head is not None else lr_backbone * 1.2)

    early_params: List[torch.nn.Parameter] = []
    late_params: List[torch.nn.Parameter] = []
    student_head_params: List[torch.nn.Parameter] = []
    criterion_params: List[torch.nn.Parameter] = list(criterion.parameters())

    for name, param in student.named_parameters():
        if name.startswith("backbone.model.stem.") or name.startswith("backbone.model.stages.0.") or name.startswith("backbone.model.stages.1."):
            early_params.append(param)
        elif name.startswith("downsample_layers.0.") or name.startswith("downsample_layers.1.") or name.startswith("stages.0.") or name.startswith("stages.1."):
            early_params.append(param)
        elif name.startswith("backbone.model.head.") or name.startswith("head_"):
            student_head_params.append(param)
        else:
            late_params.append(param)

    groups: List[Dict[str, Any]] = []
    if early_params:
        groups.append({"params": early_params, "lr": early_lr, "name": "early"})
    if late_params:
        groups.append({"params": late_params, "lr": lr_backbone, "name": "late"})
    if student_head_params:
        groups.append({"params": student_head_params, "lr": head_s_lr, "name": "student_head"})
    if criterion_params:
        groups.append({"params": criterion_params, "lr": lr_head, "name": "criterion"})
    return groups


def train_one_epoch_recognizer(
    student: nn.Module,
    teacher: Optional[nn.Module],
    ema: ModelEMA,
    loader: DataLoader,
    criterion: AdaDistillLoss,
    degrader: PhysicalLowLightDegrader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "fp16",
    sdpa_backend: str = "flash_attention",
    distill_warmup_scale: float = 1.0,
    max_steps: Optional[int] = None,
    grad_accum: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    dark_apply_prob: float = 0.90,
    ref_student: Optional[nn.Module] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    global_opt_step: int = 0,
    distill_warmup_steps: int = 0,
) -> Dict[str, Any]:
    student.train()
    if teacher is not None:
        teacher.eval()
    if ref_student is not None:
        ref_student.eval()
    scaler = scaler if scaler is not None else get_grad_scaler(device, precision)
    grad_accum = max(1, int(grad_accum))

    all_trainable = list(student.parameters()) + list(criterion.parameters())
    cur_opt_step = int(global_opt_step)

    def _optimizer_step() -> None:
        nonlocal cur_opt_step
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(all_trainable, max_norm=5.0)
        old_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
        optimizer.zero_grad(set_to_none=True)
        if new_scale >= old_scale:
            if scheduler is not None:
                scheduler.step()
            cur_opt_step += 1
        ema.update(student)

    stat_keys = (
        "loss",
        "ada_clean",
        "ada_dark",
        "adadistill",
        "teacher_cos",
        "student_cos",
        "clean_kd",
        "rel_kd",
        "ref_clean",
        "dark_ref",
    )
    stats: Dict[str, float] = {k: 0.0 for k in stat_keys}
    steps = 0
    accum_counter = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        clean_01 = batch["clean_u8"].to(device, non_blocking=True).float().div_(255.0)
        dark_src_01 = batch["dark_src_u8"].to(device, non_blocking=True).float().div_(255.0)
        kpts_norm = batch["kpts_norm"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        degraded_01 = degrader(dark_src_01)
        if dark_apply_prob < 1.0:
            apply_mask = torch.rand((clean_01.shape[0], 1, 1, 1), device=device) < dark_apply_prob
            dark_01 = torch.where(apply_mask, degraded_01, dark_src_01)
        else:
            dark_01 = degraded_01

        x_clean = (clean_01 - 0.5) / 0.5
        x_dark = (dark_01 - 0.5) / 0.5

        with get_sdpa_context(sdpa_backend), get_amp_autocast(device, precision):
            if "teacher_emb" in batch:
                z_teacher = F.normalize(batch["teacher_emb"].to(device, non_blocking=True).float(), dim=1)
            else:
                assert teacher is not None
                with torch.no_grad():
                    z_teacher = teacher(x_clean, kpts_norm)

            if ref_student is not None:
                with torch.no_grad():
                    z_ref_clean, _ = ref_student(x_clean)
            else:
                z_ref_clean = None

            z_clean, norm_clean = student(x_clean)
            z_dark, norm_dark = student(x_dark)

        step_warmup = (
            min(1.0, float(cur_opt_step + 1) / float(distill_warmup_steps))
            if distill_warmup_steps > 0
            else distill_warmup_scale
        )
        loss_dict = criterion(
            z_clean,
            norm_clean,
            z_dark,
            norm_dark,
            z_teacher,
            labels,
            distill_warmup_scale=step_warmup,
            z_ref_clean=z_ref_clean,
        )
        loss = loss_dict["loss"] / float(grad_accum)
        scaler.scale(loss).backward()
        accum_counter += 1
        if accum_counter >= grad_accum:
            _optimizer_step()
            accum_counter = 0

        for k in stat_keys:
            stats[k] += float(loss_dict[k].detach().item())
        steps += 1

    if accum_counter > 0:
        _optimizer_step()

    out: Dict[str, Any] = {k: v / max(steps, 1) for k, v in stats.items()}
    out["opt_steps_total"] = cur_opt_step
    out["lr_late"] = float(optimizer.param_groups[min(1, len(optimizer.param_groups) - 1)]["lr"])
    return out


def _build_train_dataset(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    kpt_sigma = tuple(data_cfg.get("kpt_jitter_sigma_range", [1.2, 2.5]))
    scale_jit = float(data_cfg.get("scale_jitter", 0.04))
    angle_jit = float(data_cfg.get("angle_jitter_deg", 4.0))
    manifest_or_dir = Path(data_cfg["train_manifest"])
    teacher_embs_p = data_cfg.get("teacher_embs_path")

    if data_cfg.get("dataset_type") == "blob" or manifest_or_dir.is_dir():
        return IndexedBlobRecognitionDataset(
            manifest_or_dir,
            kpt_jitter_sigma_range=(float(kpt_sigma[0]), float(kpt_sigma[1])),
            scale_jitter=scale_jit,
            angle_jitter_deg=angle_jit,
            jitter_prob=float(data_cfg.get("jitter_prob", 0.90)),
            clean_jitter_sigma=float(data_cfg.get("clean_jitter_sigma", 0.0)),
            augment=True,
            seed=int(cfg.get("seed", 42)),
            teacher_embs_path=Path(teacher_embs_p) if teacher_embs_p else None,
        )
    return TwinViewRecognitionDataset(
        manifest_or_dir,
        kpt_jitter_sigma_range=(float(kpt_sigma[0]), float(kpt_sigma[1])),
        scale_jitter=scale_jit,
        angle_jitter_deg=angle_jit,
        augment=True,
        seed=int(cfg.get("seed", 42)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Recognizer Training")
    parser.add_argument("--config", type=Path, default=Path("configs/recognizer_edgeface_s.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    student = build_recognizer_student(cfg["model"]["arch"], emb_dim=cfg["model"].get("emb_dim", 512))
    init_ckpt = args.init_ckpt or (Path(cfg["model"]["init_ckpt"]) if cfg["model"].get("init_ckpt") else None)
    has_init = bool(init_ckpt is not None and init_ckpt.is_file() and not args.smoke)
    if has_init and init_ckpt is not None:
        print(f"[train] Initializing student from {init_ckpt}")
        if isinstance(student, OfficialEdgeFace):
            student.load_official_pretrained(init_ckpt)
        else:
            ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            student.load_state_dict(ck.get("ema", ck.get("model", ck)))
    student = student.to(device)

    teacher_embs_p = cfg["data"].get("teacher_embs_path")
    has_precomputed_teacher = bool((not args.smoke) and teacher_embs_p and Path(teacher_embs_p).is_file())
    if has_precomputed_teacher:
        print(f"[train] Using precomputed teacher embeddings: {teacher_embs_p}")
        teacher = None
    else:
        teacher = CVLfaceTeacherWrapper(
            model_id_or_path=cfg["teacher"].get("model_id"),
            hf_endpoint=cfg["teacher"].get("hf_endpoint", "https://hf-mirror.com"),
            offline_proxy=args.smoke or cfg["teacher"].get("offline_proxy", False),
        ).to(device)

    num_classes = 16 if args.smoke else int(cfg["data"]["num_classes"])
    loss_cfg = cfg["loss"]
    criterion = AdaDistillLoss(
        num_classes=num_classes,
        emb_dim=cfg["model"].get("emb_dim", 512),
        partial_fc_rate=1.0 if args.smoke else cfg["train"].get("partial_fc_rate", 0.2),
        lambda1_dark_adaface=loss_cfg.get("lambda1", 1.0),
        lambda2_adadistill=loss_cfg.get("lambda2", 1.5),
        lambda3_teacher_cos=loss_cfg.get("lambda3", 3.0),
        lambda4_student_cos=loss_cfg.get("lambda4", 0.5),
        lambda_clean_kd=loss_cfg.get("lambda_clean_kd", 0.0),
        lambda_ada_clean=loss_cfg.get("lambda_ada_clean", 1.0),
        lambda_ref_clean=loss_cfg.get("lambda_ref_clean", 0.0),
        lambda_dark_ref=loss_cfg.get("lambda_dark_ref", 0.0),
        lambda_rel_kd=loss_cfg.get("lambda_rel_kd", 0.0),
        use_kd_projector=bool(loss_cfg.get("use_kd_projector", False)),
        adadistill_mode=str(loss_cfg.get("adadistill_mode", "adaface")),
        kd_temp=float(loss_cfg.get("kd_temp", 16.0)),
    ).to(device)
    degrader = PhysicalLowLightDegrader(mode="recognizer").to(device)

    lr_early = cfg["train"].get("lr_early")
    lr_backbone = float(cfg["train"].get("lr_backbone", 1.5e-4))
    lr_student_head = cfg["train"].get("lr_student_head")
    lr_head = float(cfg["train"].get("lr_head", 5e-4))

    if args.smoke:
        ema = ModelEMA(student, decay=cfg["train"].get("ema_decay", 0.9995))
        param_groups = _split_optimizer_param_groups(
            student, criterion, lr_backbone=lr_backbone, lr_head=lr_head, lr_early=lr_early, lr_student_head=lr_student_head
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["train"].get("weight_decay", 5e-4))
        scaler = get_grad_scaler(device, cfg["train"].get("precision", "fp16"))
        ds = SyntheticRecognizerSmokeDataset(n=32, num_classes=num_classes)
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
        res = train_one_epoch_recognizer(
            student=student,
            teacher=teacher,
            ema=ema,
            loader=loader,
            criterion=criterion,
            degrader=degrader,
            optimizer=optimizer,
            device=device,
            precision=cfg["train"].get("precision", "fp16"),
            sdpa_backend=cfg["train"].get("sdpa_backend", "flash_attention"),
            distill_warmup_scale=0.5,
            max_steps=2,
            scaler=scaler,
        )
        print("Smoke recognizer train metrics:", res)
        return

    ds = _build_train_dataset(cfg)
    use_ref_student = bool(cfg["train"].get("use_ref_student", False) and has_init)
    ref_student: Optional[nn.Module] = None

    teacher_centers_p = cfg["data"].get("teacher_centers_path")
    if teacher_centers_p and Path(teacher_centers_p).is_file():
        tc_np = np.load(str(teacher_centers_p))
        criterion.load_precomputed_teacher_centers(torch.from_numpy(tc_np), init_partial_fc=(not use_ref_student))
        print(
            f"[train] Loaded precomputed teacher centers from {teacher_centers_p}: shape={tc_np.shape}, "
            f"init_partial_fc={not use_ref_student}"
        )

    init_step_info: Dict[str, Any] = {}
    if use_ref_student:
        ref_student = copy.deepcopy(student).eval().to(device)
        for p in ref_student.parameters():
            p.requires_grad_(False)
        cal = calibrate_adaface_norm_stats(student, ds, device, n_samples=4096)
        criterion.adaface.batch_mean.fill_(float(cal["norm_mean"]))
        criterion.adaface.batch_std.fill_(max(0.20, float(cal["norm_std"])))
        init_step_info = {"mode": "ref_student_clean_anchor", **cal}
        print(f"[train] Frozen ref_student anchor enabled; AdaFace norm calibration={cal}")
    else:
        procrustes_info = align_student_head_procrustes(student, ds, device, teacher=teacher, n_samples=4096)
        print(f"[train] Step-0 Procrustes: {json.dumps(procrustes_info)}")
        if procrustes_info.get("applied"):
            criterion.adaface.batch_mean.fill_(float(procrustes_info["norm_mean"]))
            criterion.adaface.batch_std.fill_(max(0.20, float(procrustes_info["norm_std"])))
        init_step_info = {"mode": "procrustes", **procrustes_info}

    ema = ModelEMA(
        student,
        decay=float(cfg["train"].get("ema_decay", 0.9995)),
        warmup_tau=float(cfg["train"].get("ema_warmup_tau", 500.0)),
    )
    param_groups = _split_optimizer_param_groups(
        student,
        criterion,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        lr_early=lr_early,
        lr_student_head=lr_student_head,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["train"].get("weight_decay", 5e-4))
    scaler = get_grad_scaler(device, cfg["train"].get("precision", "fp16"))

    num_workers = int(cfg["train"].get("num_workers", 4))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size_pairs", 256)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    epochs = int(args.epochs if args.epochs is not None else cfg["train"].get("epochs", 6))
    grad_accum = int(cfg["train"].get("grad_accum", 2))
    steps_per_ep = len(loader) if args.max_steps_per_epoch is None else min(len(loader), args.max_steps_per_epoch)
    updates_per_ep = max(1, math.ceil(steps_per_ep / max(1, grad_accum)))
    total_opt_steps = max(1, epochs * updates_per_ep)
    warmup_opt_steps = int(cfg["train"].get("warmup_opt_steps", 300))

    def _lr_lambda(step_idx: int) -> float:
        if warmup_opt_steps > 0 and step_idx < warmup_opt_steps:
            return 0.05 + 0.95 * (float(step_idx + 1) / float(warmup_opt_steps))
        prog = float(step_idx - warmup_opt_steps) / float(max(1, total_opt_steps - warmup_opt_steps))
        prog = min(1.0, max(0.0, prog))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    run_dir = Path(cfg["train"].get("run_dir", "runs/recognizer"))
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(cfg["train"]["backup_dir"]) if cfg["train"].get("backup_dir") else None
    history_path = run_dir / "history.jsonl"

    with open(history_path, "a", encoding="utf-8") as hf:
        hf.write(json.dumps({"step0_init": init_step_info}) + "\n")

    eval_dir = args.eval_dir or (Path(cfg["data"]["eval_dir"]) if cfg["data"].get("eval_dir") else None)
    val_caches = None
    baseline_clean_acc = None
    if eval_dir is not None and eval_dir.is_dir():
        val_caches = {}
        for idx, bname in enumerate(("lfw", "cfp_fp", "agedb_30")):
            bp = eval_dir / f"{bname}.bin"
            if bp.is_file():
                val_caches[bname] = build_bin_protocol_cache(
                    bp, degrader, device, seed_offset=idx * 97, include_dark_gt=False, include_dark_jitter=True
                )
        if val_caches:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            base_val = evaluate_fast_val(ema.ema, eval_dir, device, bin_caches=val_caches)
            baseline_clean_acc = base_val["clean_acc_flip_mean"]
            print(f"[Epoch 0 Baseline] {json.dumps(base_val)}")
            with open(history_path, "a", encoding="utf-8") as hf:
                hf.write(json.dumps({"epoch": 0, "eval": base_val}) + "\n")

    best_score = -1.0
    max_clean_drop = float(cfg["train"].get("max_clean_drop_pct", 0.30))
    warmup_freeze_early = bool(cfg["train"].get("warmup_freeze_early", not use_ref_student))
    distill_warmup_steps = int(cfg["train"].get("distill_warmup_steps", warmup_opt_steps))
    global_opt_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        set_recognizer_warmup_trainable(student, freeze_early=(warmup_freeze_early and epoch == 1))
        stats = train_one_epoch_recognizer(
            student=student,
            teacher=teacher,
            ema=ema,
            loader=loader,
            criterion=criterion,
            degrader=degrader,
            optimizer=optimizer,
            device=device,
            precision=cfg["train"].get("precision", "fp16"),
            sdpa_backend=cfg["train"].get("sdpa_backend", "flash_attention"),
            distill_warmup_scale=min(1.0, float(epoch) / 3.0),
            max_steps=args.max_steps_per_epoch,
            grad_accum=grad_accum,
            scaler=scaler,
            dark_apply_prob=float(cfg["train"].get("dark_apply_prob", 0.90)),
            ref_student=ref_student,
            scheduler=scheduler,
            global_opt_step=global_opt_step,
            distill_warmup_steps=distill_warmup_steps,
        )
        global_opt_step = int(stats["opt_steps_total"])
        elapsed = time.perf_counter() - t0
        stats["epoch_sec"] = round(elapsed, 1)
        if device.type == "cuda":
            stats["max_vram_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2)

        eval_res = None
        if val_caches and (epoch % args.eval_every == 0 or epoch == epochs):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            eval_res = evaluate_fast_val(ema.ema, eval_dir, device, bin_caches=val_caches)
            if device.type == "cuda":
                torch.cuda.empty_cache()

            dark_score = eval_res["dark_jit_tar_1e4_mean"]
            clean_ok = (baseline_clean_acc is None) or (eval_res["clean_acc_flip_mean"] >= baseline_clean_acc - max_clean_drop)
            if clean_ok and dark_score > best_score:
                best_score = dark_score
                save_checkpoint(
                    run_dir / "best.pt",
                    student,
                    optimizer=None,
                    ema=ema,
                    epoch=epoch,
                    metrics={"train": stats, "eval": eval_res},
                    backup_dir=backup_dir,
                )

        record = {"epoch": epoch, "train": stats, "eval": eval_res}
        print(f"[Epoch {epoch}/{epochs}] {json.dumps(record)}")
        with open(history_path, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(record) + "\n")

        save_checkpoint(
            run_dir / "last.pt",
            student,
            optimizer=optimizer,
            ema=ema,
            epoch=epoch,
            metrics=record,
            backup_dir=backup_dir,
        )


if __name__ == "__main__":
    main()
