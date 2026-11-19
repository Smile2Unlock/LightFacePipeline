"""Phase 2 Recognizer Training: CVLface KP-RPE Teacher + AdaFace + AdaDistill -> EdgeFace-S/XS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

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
from recognizer.dataset import SyntheticRecognizerSmokeDataset, TwinViewRecognitionDataset
from recognizer.edgeface import build_recognizer_student
from recognizer.losses import AdaDistillLoss
from recognizer.teacher_cvlface import CVLfaceTeacherWrapper


def set_recognizer_warmup_trainable(student: nn.Module, freeze_early: bool) -> None:
    """For Epoch 1 warmup, train only last two stages + head; afterwards unfreeze full network."""
    if not hasattr(student, "stages"):
        for p in student.parameters():
            p.requires_grad_(True)
        return
    for p in student.parameters():
        p.requires_grad_(not freeze_early)
    if freeze_early:
        for m in (student.downsample_layers[2], student.downsample_layers[3], student.stages[2], student.stages[3], student.head_norm, student.head_fc, student.head_bn):
            for p in m.parameters():
                p.requires_grad_(True)


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
) -> Dict[str, float]:
    student.train()
    teacher.eval()
    scaler = get_grad_scaler(device, precision)

    stats = {"loss": 0.0, "ada_clean": 0.0, "ada_dark": 0.0, "adadistill": 0.0, "teacher_cos": 0.0}
    steps = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        clean_01 = batch["clean_u8"].to(device, non_blocking=True).float().div_(255.0)
        dark_src_01 = batch["dark_src_u8"].to(device, non_blocking=True).float().div_(255.0)
        kpts_norm = batch["kpts_norm"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # GPU physical low-light degradation with strict luma >= 25/255 safety floor
        dark_01 = degrader(dark_src_01)

        # Normalize to [-1, 1] per unified contract
        x_clean = (clean_01 - 0.5) / 0.5
        x_dark = (dark_01 - 0.5) / 0.5

        optimizer.zero_grad(set_to_none=True)
        with get_sdpa_context(sdpa_backend), get_amp_autocast(device, precision):
            with torch.no_grad():
                z_teacher = teacher(x_clean, kpts_norm)
            z_clean, norm_clean = student(x_clean)
            z_dark, norm_dark = student(x_dark)

        # AdaFace / Cosine / AdaDistill strictly in FP32 outside autocast
        loss_dict = criterion(
            z_clean,
            norm_clean,
            z_dark,
            norm_dark,
            z_teacher,
            labels,
            distill_warmup_scale=distill_warmup_scale,
        )
        loss = loss_dict["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(student)

        for k in stats:
            stats[k] += float(loss_dict[k].detach().item())
        steps += 1

    for k in stats:
        stats[k] /= max(steps, 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Recognizer Training")
    parser.add_argument("--config", type=Path, default=Path("configs/recognizer_edgeface_s.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))
    device = get_device()

    student = build_recognizer_student(cfg["model"]["arch"], emb_dim=cfg["model"].get("emb_dim", 512)).to(device)
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
    ).to(device)
    degrader = PhysicalLowLightDegrader(mode="recognizer").to(device)

    backbone_params = [p for n, p in student.named_parameters() if not n.startswith("head_")]
    head_params = [p for n, p in student.named_parameters() if n.startswith("head_")] + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["train"].get("lr_backbone", 2e-4)},
            {"params": head_params, "lr": cfg["train"].get("lr_head", 5e-4)},
        ],
        weight_decay=cfg["train"].get("weight_decay", 5e-4),
    )

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
        )
        print("Smoke recognizer train metrics:", res)
        return

    ds = TwinViewRecognitionDataset(Path(cfg["data"]["train_manifest"]))
    loader = DataLoader(
        ds,
        batch_size=cfg["train"].get("batch_size_pairs", 512),
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )
    epochs = cfg["train"].get("epochs", 16)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    run_dir = Path(cfg["train"].get("run_dir", "runs/recognizer"))
    backup_dir = Path(cfg["train"]["backup_dir"]) if cfg["train"].get("backup_dir") else None

    for epoch in range(1, epochs + 1):
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
        )
        scheduler.step()
        print(f"[Epoch {epoch}/{epochs}] {stats}")
        save_checkpoint(
            run_dir / f"recognizer_epoch{epoch}.pt",
            student,
            optimizer=optimizer,
            ema=ema,
            epoch=epoch,
            metrics=stats,
            backup_dir=backup_dir,
        )


if __name__ == "__main__":
    main()
