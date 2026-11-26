"""Phase 3 Temporal Liveness Training: Stage 1 DINOv2-Reg Teacher -> Stage 2 ST-FAS-Lite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from common.degradations import LivenessPhysicalAugmentor, compute_rgb_luma_255
from common.metrics import audit_luma_anti_shortcut, compute_iso30107_metrics
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
from liveness.losses import FASCombinedLoss, SSDGSingleSideDomainLoss, focal_fine_or_binary_loss
from liveness.st_fas_lite import STFASLiteClip
from liveness.teacher_dinov2 import DualStreamDINOv2Teacher


@torch.no_grad()
def compute_clarity_target(tex: torch.Tensor) -> torch.Tensor:
    """Continuous clarity target in [0.05, 0.98] combining face luminance & Laplacian sharpness."""
    b, t, _, h, w = tex.shape
    luma_mean = compute_rgb_luma_255(tex).mean(dim=1)  # [B]
    luma_score = ((luma_mean - 12.0) / 55.0).clamp(0.0, 1.0)

    gray = (0.299 * tex[:, :, 0] + 0.587 * tex[:, :, 1] + 0.114 * tex[:, :, 2]).reshape(b * t, 1, h, w)
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=tex.device, dtype=tex.dtype).view(
        1, 1, 3, 3
    )
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), k)
    lap_var = lap.var(dim=(-2, -1)).view(b, t).mean(dim=1)
    sharp_score = (lap_var / 0.006).clamp(0.0, 1.0)
    return torch.sqrt((luma_score + 1e-4) * (0.25 + 0.75 * sharp_score)).clamp(0.05, 0.98)


def _apply_differentiated_batch_augmentation(
    tex: torch.Tensor,
    ctx: torch.Tensor,
    labels: torch.Tensor,
    depth_target: Optional[torch.Tensor],
    augmentor: LivenessPhysicalAugmentor,
    fine_known: Optional[torch.Tensor] = None,
    p_normal: float = 0.30,
    frozen_neg_prob: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Synchronized Strategy A (Normal + Differentiated Low-Light) & Strategy B (Frozen-Rigid) on (tex, ctx)."""
    return augmentor.augment_pair(
        tex,
        ctx,
        labels,
        depth_target=depth_target,
        fine_known=fine_known,
        p_normal=p_normal,
        p_frozen=frozen_neg_prob,
    )


def train_one_epoch_teacher(
    teacher: DualStreamDINOv2Teacher,
    loader: DataLoader,
    augmentor: LivenessPhysicalAugmentor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "fp16",
    sdpa_backend: str = "math",
    grad_accum: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[Any] = None,
    p_normal: float = 0.30,
    p_frozen: float = 0.20,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    teacher.train()
    scaler = scaler or get_grad_scaler(device, precision)
    ssdg_fn = SSDGSingleSideDomainLoss(margin=0.5).to(device)
    trainable_params = [p for p in teacher.parameters() if p.requires_grad]

    stats = {"loss": 0.0, "ce_loss": 0.0, "ssdg_loss": 0.0}
    live_luma_samples: List[np.ndarray] = []
    spoof_luma_samples: List[np.ndarray] = []
    steps = 0
    accum_counter = 0
    total_batches = min(len(loader), max_steps) if max_steps is not None else len(loader)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        tex = batch["tex_u8"].to(device, non_blocking=True).float().div_(255.0)
        ctx = batch["ctx_u8"].to(device, non_blocking=True).float().div_(255.0)
        labels = batch["label"].to(device, non_blocking=True)
        fk = batch["fine_known"].to(device, non_blocking=True) if "fine_known" in batch else None

        tex, ctx, labels, _, fk = _apply_differentiated_batch_augmentation(
            tex, ctx, labels, None, augmentor, fine_known=fk, p_normal=p_normal, frozen_neg_prob=p_frozen
        )

        if len(live_luma_samples) < 8:
            lm = labels == 0
            sm = labels != 0
            if lm.any():
                live_luma_samples.append(tex[lm].detach().cpu().numpy())
            if sm.any():
                spoof_luma_samples.append(tex[sm].detach().cpu().numpy())

        with get_sdpa_context(sdpa_backend):
            with get_amp_autocast(device, precision):
                out = teacher(tex, ctx)
                l_ce = focal_fine_or_binary_loss(out["logits"], labels, fine_known=fk, gamma=2.0)
                l_ssdg = ssdg_fn(out["pooled_feat"], labels)
                loss = l_ce + 0.2 * l_ssdg

            scaled_loss = loss / max(grad_accum, 1)
            scaler.scale(scaled_loss).backward()
        accum_counter += 1

        if accum_counter == grad_accum or (step + 1) == total_batches:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            accum_counter = 0

        stats["loss"] += float(loss.detach().item())
        stats["ce_loss"] += float(l_ce.detach().item())
        stats["ssdg_loss"] += float(l_ssdg.detach().item())
        steps += 1

    for k in stats:
        stats[k] /= max(steps, 1)

    if live_luma_samples and spoof_luma_samples:
        stats["luma_audit"] = audit_luma_anti_shortcut(
            np.concatenate(live_luma_samples, axis=0),
            np.concatenate(spoof_luma_samples, axis=0),
        )
    return stats


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
    grad_accum: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[Any] = None,
    p_normal: float = 0.30,
    p_frozen: float = 0.20,
) -> Dict[str, Any]:
    student.train()
    if teacher is not None:
        teacher.eval()
    scaler = scaler or get_grad_scaler(device, precision)

    stats = {"loss": 0.0, "ce_loss": 0.0, "depth_loss": 0.0, "clarity_loss": 0.0, "kd_loss": 0.0, "patch_kd_loss": 0.0}
    live_luma_samples: List[np.ndarray] = []
    spoof_luma_samples: List[np.ndarray] = []
    steps = 0
    accum_counter = 0
    total_batches = min(len(loader), max_steps) if max_steps is not None else len(loader)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        tex = batch["tex_u8"].to(device, non_blocking=True).float().div_(255.0)
        ctx = batch["ctx_u8"].to(device, non_blocking=True).float().div_(255.0)
        labels = batch["label"].to(device, non_blocking=True)
        depth_target = batch["depth_target"].to(device, non_blocking=True)
        fk = batch["fine_known"].to(device, non_blocking=True) if "fine_known" in batch else None

        tex, ctx, labels, depth_target, fk = _apply_differentiated_batch_augmentation(
            tex, ctx, labels, depth_target, augmentor, fine_known=fk, p_normal=p_normal, frozen_neg_prob=p_frozen
        )
        clarity_target = compute_clarity_target(tex)

        if len(live_luma_samples) < 8:
            lm = labels == 0
            sm = labels != 0
            if lm.any():
                live_luma_samples.append(tex[lm].detach().cpu().numpy())
            if sm.any():
                spoof_luma_samples.append(tex[sm].detach().cpu().numpy())

        with get_sdpa_context(sdpa_backend):
            with get_amp_autocast(device, precision):
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
                fine_known=fk,
            )
            loss = loss_dict["loss"]
            scaler.scale(loss / max(grad_accum, 1)).backward()
        accum_counter += 1

        if accum_counter == grad_accum or (step + 1) == total_batches:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(student)
            if scheduler is not None:
                scheduler.step()
            accum_counter = 0

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


@torch.no_grad()
def evaluate_liveness_validation(
    model: nn.Module,
    is_teacher: bool,
    val_loader: DataLoader,
    augmentor: LivenessPhysicalAugmentor,
    device: torch.device,
    precision: str = "fp16",
    sdpa_backend: str = "math",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Multi-condition ISO/IEC 30107-3 validation across Normal, Low-Light, Backlight & Frozen-Rigid."""
    model.eval()
    probs_by_cond: Dict[str, List[np.ndarray]] = {
        "normal": [],
        "low_light": [],
        "backlight": [],
        "frozen_vs_live": [],
    }
    labels_by_cond: Dict[str, List[np.ndarray]] = {k: [] for k in probs_by_cond}
    live_luma: List[np.ndarray] = []
    spoof_luma: List[np.ndarray] = []

    def _predict_live_prob(t_in: torch.Tensor, c_in: torch.Tensor) -> np.ndarray:
        with get_sdpa_context(sdpa_backend), get_amp_autocast(device, precision):
            if is_teacher:
                logits = model(t_in, c_in)["logits"]
                lp = torch.softmax(logits.float(), dim=-1)[:, 0]
            else:
                lp, _, _ = model(t_in, c_in)
                lp = lp.squeeze(-1)
        return lp.float().cpu().numpy()

    for bi, batch in enumerate(val_loader):
        if max_batches is not None and bi >= max_batches:
            break
        tex = batch["tex_u8"].to(device, non_blocking=True).float().div_(255.0)
        ctx = batch["ctx_u8"].to(device, non_blocking=True).float().div_(255.0)
        is_live = (batch["label"].numpy() == 0).astype(np.int64)

        for cond in ("normal", "low_light", "backlight"):
            t_c, c_c = augmentor.apply_eval_lighting(tex, ctx, cond, seed=1000 + bi * 7)
            p_c = _predict_live_prob(t_c, c_c)
            probs_by_cond[cond].append(p_c)
            labels_by_cond[cond].append(is_live)
            if cond in ("normal", "low_light") and len(live_luma) < 12:
                lm = is_live == 1
                sm = is_live == 0
                if lm.any():
                    live_luma.append(t_c[lm].detach().cpu().numpy())
                if sm.any():
                    spoof_luma.append(t_c[sm].detach().cpu().numpy())

        live_idx = (batch["label"] == 0).nonzero(as_tuple=False).squeeze(1)
        if live_idx.numel() > 0:
            t_fr, c_fr = augmentor.apply_eval_lighting(tex[live_idx], ctx[live_idx], "frozen_rigid", seed=5000 + bi)
            p_fr = _predict_live_prob(t_fr, c_fr)
            probs_by_cond["frozen_vs_live"].append(p_fr)
            labels_by_cond["frozen_vs_live"].append(np.zeros(live_idx.numel(), dtype=np.int64))
            # Add corresponding low-light live samples into frozen_vs_live comparison
            probs_by_cond["frozen_vs_live"].append(probs_by_cond["low_light"][-1][live_idx.cpu().numpy()])
            labels_by_cond["frozen_vs_live"].append(np.ones(live_idx.numel(), dtype=np.int64))

    out: Dict[str, Any] = {}
    all_p: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for cond in ("normal", "low_light", "backlight", "frozen_vs_live"):
        if not probs_by_cond[cond]:
            continue
        p = np.concatenate(probs_by_cond[cond])
        y = np.concatenate(labels_by_cond[cond])
        m = compute_iso30107_metrics(p, y)
        out[cond] = {
            "auc": m["auc"],
            "ACER": m["operating_point"]["ACER"],
            "threshold": m["operating_point"]["threshold"],
            "BPCER_at_APCER_2pct": m["fixed_apcer_points"]["APCER<=2.0%"]["BPCER"],
            "BPCER_at_APCER_1pct": m["fixed_apcer_points"]["APCER<=1.0%"]["BPCER"],
        }
        all_p.append(p)
        all_y.append(y)

    if all_p:
        p_all = np.concatenate(all_p)
        y_all = np.concatenate(all_y)
        m_all = compute_iso30107_metrics(p_all, y_all)
        out["all"] = {
            "auc": m_all["auc"],
            "ACER": m_all["operating_point"]["ACER"],
            "threshold": m_all["operating_point"]["threshold"],
            "APCER": m_all["operating_point"]["APCER"],
            "BPCER": m_all["operating_point"]["BPCER"],
            "BPCER_at_APCER_2pct": m_all["fixed_apcer_points"]["APCER<=2.0%"]["BPCER"],
            "BPCER_at_APCER_1pct": m_all["fixed_apcer_points"]["APCER<=1.0%"]["BPCER"],
            "n_live": m_all["n_live"],
            "n_spoof": m_all["n_spoof"],
        }

    if live_luma and spoof_luma:
        out["luma_audit"] = audit_luma_anti_shortcut(
            np.concatenate(live_luma, axis=0),
            np.concatenate(spoof_luma, axis=0),
        )
    return out


def _make_warmup_cosine(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 20):
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.02, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Temporal Liveness Training")
    parser.add_argument("--config", type=Path, default=Path("configs/liveness_st_fas_lite_local.yaml"))
    parser.add_argument("--stage", choices=["teacher", "student"], default="student")
    parser.add_argument("--teacher-ckpt", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_cfg = cfg.get("train", {})
    precision = str(train_cfg.get("precision", "fp16"))
    sdpa_backend = str(train_cfg.get("sdpa_backend", "math"))
    p_normal = float(train_cfg.get("p_normal", 0.30))
    p_frozen = float(train_cfg.get("p_frozen", 0.20))
    augmentor = LivenessPhysicalAugmentor().to(device)

    teacher_cfg = cfg.get("teacher", {})
    teacher = DualStreamDINOv2Teacher(
        model_id_or_path=teacher_cfg.get("model_id"),
        hf_endpoint=teacher_cfg.get("hf_endpoint", "https://hf-mirror.com"),
        lora_r=int(teacher_cfg.get("lora_r", 16)),
        offline_config_only=args.smoke or bool(teacher_cfg.get("offline_config_only", False)),
        clip_len=int(cfg["model"].get("clip_len", 8)),
        num_classes=int(cfg["model"].get("num_classes", 5)),
        enable_grad_checkpointing=bool(teacher_cfg.get("enable_grad_checkpointing", True)) and (args.stage == "teacher") and not args.smoke,
    ).to(device)

    if args.smoke:
        student = STFASLiteClip(
            feat_dim=cfg["model"].get("feat_dim", 192),
            clip_len=cfg["model"].get("clip_len", 8),
            num_classes=cfg["model"].get("num_classes", 5),
        ).to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        ema = ModelEMA(student, decay=0.995)
        criterion = FASCombinedLoss().to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
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
            precision=precision,
            sdpa_backend=sdpa_backend,
            max_steps=2,
        )
        print("Smoke liveness train metrics:", res)
        return

    train_ds = VideoClipLivenessDataset(Path(cfg["data"]["train_manifest"]), augment=True)
    val_path = cfg["data"].get("val_manifest")
    val_ds = VideoClipLivenessDataset(Path(val_path), augment=False) if val_path and Path(val_path).is_file() else None

    if args.stage == "teacher":
        st_cfg = cfg.get("teacher_stage", train_cfg)
        bs = int(args.batch_size or st_cfg.get("batch_size_clips", 4))
        ga = int(args.grad_accum or st_cfg.get("grad_accum", 8))
        epochs = int(args.epochs or st_cfg.get("epochs", 12))
        n_samples = int(args.samples_per_epoch or st_cfg.get("samples_per_epoch", len(train_ds)))
        val_max_batches = args.val_max_batches if args.val_max_batches is not None else st_cfg.get("val_max_batches", None)
        run_dir = Path(args.run_dir or Path(train_cfg.get("run_dir", "runs/liveness")) / "teacher")
        run_dir.mkdir(parents=True, exist_ok=True)

        lora_params, hf_params, head_params = [], [], []
        for name, p in teacher.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("hf_branch."):
                hf_params.append(p)
            elif name.startswith("vit."):
                lora_params.append(p)
            else:
                head_params.append(p)

        base_lr = float(st_cfg.get("lr", 3e-4))
        hf_lr = float(st_cfg.get("hf_lr", 1e-4))
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params + head_params, "lr": base_lr},
                {"params": hf_params, "lr": hf_lr},
            ],
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        sampler = train_ds.build_balanced_sampler(num_samples=n_samples, balance_domains=True)
        loader = DataLoader(
            train_ds,
            batch_size=bs,
            sampler=sampler,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=True,
            drop_last=True,
        )
        val_loader = (
            DataLoader(val_ds, batch_size=max(bs * 2, 8), shuffle=False, num_workers=2, pin_memory=True)
            if val_ds is not None
            else None
        )
        opt_steps_per_epoch = max(1, math.ceil(min(len(loader), args.max_steps or len(loader)) / max(ga, 1)))
        scheduler = _make_warmup_cosine(optimizer, total_steps=opt_steps_per_epoch * epochs, warmup_steps=min(30, opt_steps_per_epoch))
        scaler = get_grad_scaler(device, precision)
        best_acer = 1.0
        history_path = run_dir / "history.jsonl"

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch_teacher(
                teacher=teacher,
                loader=loader,
                augmentor=augmentor,
                optimizer=optimizer,
                device=device,
                precision=precision,
                sdpa_backend=sdpa_backend,
                grad_accum=ga,
                scaler=scaler,
                scheduler=scheduler,
                p_normal=p_normal,
                p_frozen=p_frozen,
                max_steps=args.max_steps,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_m: Dict[str, Any] = {}
            if val_loader is not None:
                val_m = evaluate_liveness_validation(
                    teacher, True, val_loader, augmentor, device, precision=precision, sdpa_backend=sdpa_backend, max_batches=val_max_batches
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            rec = {"epoch": epoch, "train": tr, "val": val_m}
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[Teacher Epoch {epoch}/{epochs}] loss={tr['loss']:.4f} val_all={val_m.get('all', {})}", flush=True)

            save_checkpoint(run_dir / "last.pt", teacher, optimizer=optimizer, epoch=epoch, metrics=rec)
            cur_acer = float(val_m.get("all", {}).get("ACER", tr["loss"]))
            if cur_acer < best_acer:
                best_acer = cur_acer
                save_checkpoint(run_dir / "best.pt", teacher, optimizer=optimizer, epoch=epoch, metrics=rec)
        return

    # Stage 2: Student Distillation
    st_cfg = cfg.get("student_stage", train_cfg)
    bs = int(args.batch_size or st_cfg.get("batch_size_clips", 8))
    ga = int(args.grad_accum or st_cfg.get("grad_accum", 4))
    epochs = int(args.epochs or st_cfg.get("epochs", 20))
    n_samples = int(args.samples_per_epoch or st_cfg.get("samples_per_epoch", len(train_ds)))
    val_max_batches = args.val_max_batches if args.val_max_batches is not None else st_cfg.get("val_max_batches", None)
    run_dir = Path(args.run_dir or Path(train_cfg.get("run_dir", "runs/liveness")) / "student")
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher_ckpt = args.teacher_ckpt or (Path(train_cfg.get("run_dir", "runs/liveness")) / "teacher" / "best.pt")
    if teacher_ckpt.is_file():
        ck = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        teacher.load_state_dict(ck.get("model", ck), strict=True)
        print(f"Loaded trained Teacher checkpoint from {teacher_ckpt} (epoch={ck.get('epoch')})", flush=True)
    else:
        raise FileNotFoundError(f"Trained teacher checkpoint not found: {teacher_ckpt}")

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    student = STFASLiteClip(
        feat_dim=cfg["model"].get("feat_dim", 192),
        clip_len=cfg["model"].get("clip_len", 8),
        num_classes=cfg["model"].get("num_classes", 5),
    ).to(device)
    ema = ModelEMA(student, decay=float(st_cfg.get("ema_decay", 0.995)), warmup_tau=150.0)
    criterion = FASCombinedLoss().to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(st_cfg.get("lr", 1.5e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    sampler = train_ds.build_balanced_sampler(num_samples=n_samples, balance_domains=True)
    loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=max(bs * 2, 16), shuffle=False, num_workers=2, pin_memory=True)
        if val_ds is not None
        else None
    )
    opt_steps_per_epoch = max(1, math.ceil(min(len(loader), args.max_steps or len(loader)) / max(ga, 1)))
    scheduler = _make_warmup_cosine(optimizer, total_steps=opt_steps_per_epoch * epochs, warmup_steps=min(40, opt_steps_per_epoch))
    scaler = get_grad_scaler(device, precision)
    best_acer = 1.0
    history_path = run_dir / "history.jsonl"

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch_liveness(
            student=student,
            teacher=teacher,
            ema=ema,
            loader=loader,
            criterion=criterion,
            augmentor=augmentor,
            optimizer=optimizer,
            device=device,
            precision=precision,
            sdpa_backend=sdpa_backend,
            grad_accum=ga,
            scaler=scaler,
            scheduler=scheduler,
            p_normal=p_normal,
            p_frozen=p_frozen,
            max_steps=args.max_steps,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_raw: Dict[str, Any] = {}
        val_ema: Dict[str, Any] = {}
        if val_loader is not None:
            val_raw = evaluate_liveness_validation(
                student, False, val_loader, augmentor, device, precision=precision, sdpa_backend=sdpa_backend, max_batches=val_max_batches
            )
            val_ema = evaluate_liveness_validation(
                ema.ema, False, val_loader, augmentor, device, precision=precision, sdpa_backend=sdpa_backend, max_batches=val_max_batches
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rec = {"epoch": epoch, "train": tr, "val_raw": val_raw, "val_ema": val_ema}
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"[Student Epoch {epoch}/{epochs}] loss={tr['loss']:.4f} kd={tr['kd_loss']:.4f} "
            f"val_ema_all={val_ema.get('all', {})}",
            flush=True,
        )
        save_checkpoint(run_dir / "last.pt", student, optimizer=optimizer, ema=ema, epoch=epoch, metrics=rec)
        cur_acer = float(val_ema.get("all", {}).get("ACER", val_raw.get("all", {}).get("ACER", tr["loss"])))
        if cur_acer < best_acer:
            best_acer = cur_acer
            save_checkpoint(run_dir / "best.pt", student, optimizer=optimizer, ema=ema, epoch=epoch, metrics=rec)


if __name__ == "__main__":
    main()
