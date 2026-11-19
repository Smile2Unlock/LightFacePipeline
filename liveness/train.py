"""Phase 3 Temporal Liveness Training: DINOv2-Reg Teacher -> ST-FAS-Lite + Strategy A/B + Anti-Shortcut Audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from common.degradations import LivenessPhysicalAugmentor
from common.metrics import audit_luma_anti_shortcut
from common.runtime import (
    ModelEMA,
    get_amp_autocast,
    get_device,
    get_grad_scaler,
    get_sdpa_context,
    save_checkpoint,
    seed_everything,
)
from liveness.dataset import SyntheticLivenessSmokeDataset, VideoClipLivenessDataset
from liveness.losses import FASCombinedLoss
from liveness.st_fas_lite import STFASLiteClip
from liveness.teacher_dinov2 import DualStreamDINOv2Teacher


def _apply_differentiated_batch_augmentation(
    tex: torch.Tensor,
    ctx: torch.Tensor,
    labels: torch.Tensor,
    depth_target: torch.Tensor,
    augmentor: LivenessPhysicalAugmentor,
    frozen_neg_prob: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Strategy A differentiated physical low-light + Strategy B frozen-rigid hard negatives."""
    tex_out = tex.clone()
    ctx_out = ctx.clone()
    labels_out = labels.clone()
    depth_out = depth_target.clone()

    live_idx = (labels == 0).nonzero(as_tuple=False).squeeze(1)
    print_idx = (labels == 1).nonzero(as_tuple=False).squeeze(1)
    replay_idx = (labels == 2).nonzero(as_tuple=False).squeeze(1)

    if live_idx.numel() > 0:
        tex_out[live_idx] = augmentor.augment_live(tex[live_idx])
        ctx_out[live_idx] = augmentor.augment_live(ctx[live_idx])

        # Strategy B: convert a subset of low-light live clips into rigid-2D/frozen hard negatives (label=4)
        n_convert = int(round(live_idx.numel() * frozen_neg_prob))
        if n_convert > 0:
            conv_idx = live_idx[:n_convert]
            tex_out[conv_idx] = augmentor.synthesize_frozen_rigid_negative(tex_out[conv_idx])
            ctx_out[conv_idx] = augmentor.synthesize_frozen_rigid_negative(ctx_out[conv_idx])
            labels_out[conv_idx] = 4
            depth_out[conv_idx] = 0.0

    if print_idx.numel() > 0:
        tex_out[print_idx] = augmentor.augment_print(tex[print_idx])
        ctx_out[print_idx] = augmentor.augment_print(ctx[print_idx])

    if replay_idx.numel() > 0:
        tex_out[replay_idx] = augmentor.augment_replay(tex[replay_idx], dark_dim_ratio=0.30)
        ctx_out[replay_idx] = augmentor.augment_replay(ctx[replay_idx], dark_dim_ratio=0.30)

    return tex_out, ctx_out, labels_out, depth_out


def train_one_epoch_liveness(
    student: STFASLiteClip,
    teacher: Optional[DualStreamDINOv2Teacher],
    ema: ModelEMA,
    loader: DataLoader,
    criterion: FASCombinedLoss,
    augmentor: LivenessPhysicalAugmentor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "fp16",
    sdpa_backend: str = "flash_attention",
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    student.train()
    if teacher is not None:
        teacher.eval()
    scaler = get_grad_scaler(device, precision)

    stats = {"loss": 0.0, "ce_loss": 0.0, "depth_loss": 0.0, "kd_loss": 0.0, "patch_kd_loss": 0.0}
    live_luma_samples = []
    spoof_luma_samples = []
    steps = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        tex = batch["tex_u8"].to(device, non_blocking=True).float().div_(255.0)
        ctx = batch["ctx_u8"].to(device, non_blocking=True).float().div_(255.0)
        labels = batch["label"].to(device, non_blocking=True)
        depth_target = batch["depth_target"].to(device, non_blocking=True)

        tex, ctx, labels, depth_target = _apply_differentiated_batch_augmentation(
            tex, ctx, labels, depth_target, augmentor
        )
        clarity_target = (tex.mean(dim=(2, 3, 4)).mean(dim=1) > 0.12).float()

        if step < 4:
            lm = labels == 0
            sm = labels != 0
            if lm.any():
                live_luma_samples.append(tex[lm].detach().cpu().numpy())
            if sm.any():
                spoof_luma_samples.append(tex[sm].detach().cpu().numpy())

        optimizer.zero_grad(set_to_none=True)
        with get_sdpa_context(sdpa_backend), get_amp_autocast(device, precision):
            stu_out = student.forward_train(tex, ctx)
            teacher_out = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(tex, ctx)

        loss_dict = criterion(
            stu_out,
            labels,
            depth_target=depth_target,
            clarity_target=clarity_target,
            teacher_out=teacher_out,
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

    if live_luma_samples and spoof_luma_samples:
        stats["luma_audit"] = audit_luma_anti_shortcut(
            np.concatenate(live_luma_samples, axis=0),
            np.concatenate(spoof_luma_samples, axis=0),
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Temporal Liveness Training")
    parser.add_argument("--config", type=Path, default=Path("configs/liveness_st_fas_lite.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))
    device = get_device()

    student = STFASLiteClip(
        feat_dim=cfg["model"].get("feat_dim", 192),
        clip_len=cfg["model"].get("clip_len", 8),
        num_classes=cfg["model"].get("num_classes", 5),
    ).to(device)
    teacher = DualStreamDINOv2Teacher(
        model_id_or_path=cfg["teacher"].get("model_id"),
        hf_endpoint=cfg["teacher"].get("hf_endpoint", "https://hf-mirror.com"),
        lora_r=cfg["teacher"].get("lora_r", 16),
        offline_config_only=args.smoke or cfg["teacher"].get("offline_config_only", False),
    ).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    ema = ModelEMA(student, decay=cfg["train"].get("ema_decay", 0.9995))
    criterion = FASCombinedLoss().to(device)
    augmentor = LivenessPhysicalAugmentor().to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["train"].get("lr", 3e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    if args.smoke:
        ds = SyntheticLivenessSmokeDataset(n=16)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        res = train_one_epoch_liveness(
            student=student,
            teacher=teacher,
            ema=ema,
            loader=loader,
            criterion=criterion,
            augmentor=augmentor,
            optimizer=optimizer,
            device=device,
            precision=cfg["train"].get("precision", "fp16"),
            sdpa_backend=cfg["train"].get("sdpa_backend", "flash_attention"),
            max_steps=2,
        )
        print("Smoke liveness train metrics:", res)
        return

    ds = VideoClipLivenessDataset(Path(cfg["data"]["train_manifest"]))
    sampler = ds.build_balanced_sampler()
    loader = DataLoader(
        ds,
        batch_size=cfg["train"].get("batch_size_clips", 64),
        sampler=sampler,
        num_workers=cfg["train"].get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )
    epochs = cfg["train"].get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    run_dir = Path(cfg["train"].get("run_dir", "runs/liveness"))
    backup_dir = Path(cfg["train"]["backup_dir"]) if cfg["train"].get("backup_dir") else None

    for epoch in range(1, epochs + 1):
        stats = train_one_epoch_liveness(
            student=student,
            teacher=teacher,
            ema=ema,
            loader=loader,
            criterion=criterion,
            augmentor=augmentor,
            optimizer=optimizer,
            device=device,
            precision=cfg["train"].get("precision", "fp16"),
            sdpa_backend=cfg["train"].get("sdpa_backend", "flash_attention"),
        )
        scheduler.step()
        print(f"[Epoch {epoch}/{epochs}] {stats}")
        if epoch % 5 == 0 or epoch == epochs:
            save_checkpoint(
                run_dir / f"liveness_epoch{epoch}.pt",
                student,
                optimizer=optimizer,
                ema=ema,
                epoch=epoch,
                metrics=stats,
                backup_dir=backup_dir,
            )


if __name__ == "__main__":
    main()
