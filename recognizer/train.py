"""Phase 2 Recognizer Training: CVLface KP-RPE Teacher + AdaFace + AdaDistill -> EdgeFace-S/XS."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

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


def set_recognizer_warmup_trainable(student: nn.Module, freeze_early: bool) -> None:
    """For Epoch 1 warmup, train only last two stages + head; afterwards unfreeze full network."""
    # Official Idiap EdgeFace (`timm` EdgeNeXt backbone)
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

    # Custom pure-PyTorch EdgeFaceS/XS
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
) -> List[Dict[str, Any]]:
    early_params: List[torch.nn.Parameter] = []
    late_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = list(criterion.parameters())

    for name, param in student.named_parameters():
        if name.startswith("backbone.model.stem.") or name.startswith("backbone.model.stages.0.") or name.startswith("backbone.model.stages.1."):
            early_params.append(param)
        elif name.startswith("downsample_layers.0.") or name.startswith("downsample_layers.1.") or name.startswith("stages.0.") or name.startswith("stages.1."):
            early_params.append(param)
        elif name.startswith("backbone.model.head.") or name.startswith("head_"):
            head_params.append(param)
        else:
            late_params.append(param)

    groups: List[Dict[str, Any]] = []
    if early_params:
        groups.append({"params": early_params, "lr": lr_backbone * 0.5})
    if late_params:
        groups.append({"params": late_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    return groups


def train_one_epoch_recognizer(
    student: nn.Module,
    teacher: nn.Module,
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
) -> Dict[str, float]:
    student.train()
    teacher.eval()
    scaler = scaler if scaler is not None else get_grad_scaler(device, precision)
    grad_accum = max(1, int(grad_accum))

    all_trainable = list(student.parameters()) + list(criterion.parameters())

    def _optimizer_step() -> None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(all_trainable, max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(student)

    stats = {
        "loss": 0.0,
        "ada_clean": 0.0,
        "ada_dark": 0.0,
        "adadistill": 0.0,
        "teacher_cos": 0.0,
        "student_cos": 0.0,
        "clean_kd": 0.0,
    }
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

        # GPU physical low-light degradation (90% degraded + 10% normal-light jitter bridge)
        degraded_01 = degrader(dark_src_01)
        if dark_apply_prob < 1.0:
            apply_mask = (torch.rand((clean_01.shape[0], 1, 1, 1), device=device) < dark_apply_prob)
            dark_01 = torch.where(apply_mask, degraded_01, dark_src_01)
        else:
            dark_01 = degraded_01

        x_clean = (clean_01 - 0.5) / 0.5
        x_dark = (dark_01 - 0.5) / 0.5

        with get_sdpa_context(sdpa_backend), get_amp_autocast(device, precision):
            with torch.no_grad():
                z_teacher = teacher(x_clean, kpts_norm)
            z_clean, norm_clean = student(x_clean)
            z_dark, norm_dark = student(x_dark)

        loss_dict = criterion(
            z_clean,
            norm_clean,
            z_dark,
            norm_dark,
            z_teacher,
            labels,
            distill_warmup_scale=distill_warmup_scale,
        )
        loss = loss_dict["loss"] / float(grad_accum)
        scaler.scale(loss).backward()
        accum_counter += 1
        if accum_counter >= grad_accum:
            _optimizer_step()
            accum_counter = 0

        for k in stats:
            stats[k] += float(loss_dict[k].detach().item())
        steps += 1

    if accum_counter > 0:
        _optimizer_step()

    for k in stats:
        stats[k] /= max(steps, 1)
    return stats


def _build_train_dataset(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    kpt_sigma = tuple(data_cfg.get("kpt_jitter_sigma_range", [1.2, 2.5]))
    scale_jit = float(data_cfg.get("scale_jitter", 0.04))
    angle_jit = float(data_cfg.get("angle_jitter_deg", 4.0))
    manifest_or_dir = Path(data_cfg["train_manifest"])

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
    if init_ckpt is not None and init_ckpt.is_file() and not args.smoke:
        print(f"[train] Initializing student from {init_ckpt}")
        if isinstance(student, OfficialEdgeFace):
            student.load_official_pretrained(init_ckpt)
        else:
            ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            student.load_state_dict(ck.get("ema", ck.get("model", ck)))
    student = student.to(device)

    teacher = CVLfaceTeacherWrapper(
        model_id_or_path=cfg["teacher"].get("model_id"),
        hf_endpoint=cfg["teacher"].get("hf_endpoint", "https://hf-mirror.com"),
        offline_proxy=args.smoke or cfg["teacher"].get("offline_proxy", False),
    ).to(device)
    ema = ModelEMA(student, decay=cfg["train"].get("ema_decay", 0.9995))

    num_classes = 16 if args.smoke else int(cfg["data"]["num_classes"])
    criterion = AdaDistillLoss(
        num_classes=num_classes,
        emb_dim=cfg["model"].get("emb_dim", 512),
        partial_fc_rate=1.0 if args.smoke else cfg["train"].get("partial_fc_rate", 0.2),
        lambda1_dark_adaface=cfg["loss"].get("lambda1", 1.0),
        lambda2_adadistill=cfg["loss"].get("lambda2", 1.5),
        lambda3_teacher_cos=cfg["loss"].get("lambda3", 3.0),
        lambda4_student_cos=cfg["loss"].get("lambda4", 0.5),
        lambda_clean_kd=cfg["loss"].get("lambda_clean_kd", 0.0),
    ).to(device)
    degrader = PhysicalLowLightDegrader(mode="recognizer").to(device)

    lr_backbone = float(cfg["train"].get("lr_backbone", 1.5e-4))
    lr_head = float(cfg["train"].get("lr_head", 5e-4))
    param_groups = _split_optimizer_param_groups(student, criterion, lr_backbone=lr_backbone, lr_head=lr_head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["train"].get("weight_decay", 5e-4))
    scaler = get_grad_scaler(device, cfg["train"].get("precision", "fp16"))

    if args.smoke:
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
    num_workers = int(cfg["train"].get("num_workers", 4))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size_pairs", 128)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    epochs = int(args.epochs if args.epochs is not None else cfg["train"].get("epochs", 8))
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    run_dir = Path(cfg["train"].get("run_dir", "runs/recognizer"))
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(cfg["train"]["backup_dir"]) if cfg["train"].get("backup_dir") else None
    history_path = run_dir / "history.jsonl"

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

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        set_recognizer_warmup_trainable(student, freeze_early=(epoch == 1))
        warmup_scale = min(1.0, float(epoch) / 3.0)
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
            distill_warmup_scale=warmup_scale,
            max_steps=args.max_steps_per_epoch,
            grad_accum=grad_accum,
            scaler=scaler,
            dark_apply_prob=float(cfg["train"].get("dark_apply_prob", 0.90)),
        )
        scheduler.step()
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
