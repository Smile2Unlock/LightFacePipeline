"""Phase 2 Losses: AdaFace Quality-Adaptive Margin, Partial FC, and AdaDistill Dynamic Class Centers."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialFC(nn.Module):
    """Memory-efficient Partial FC class-center store supporting 50K–200K+ IDs on single GPU."""

    def __init__(self, num_classes: int, emb_dim: int = 512, sample_rate: float = 0.2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.sample_rate = sample_rate
        self.centers = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.01)

    def sample_centers_with_index(
        self, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample positive + random negative centers; returns `(weight [M, D], mapped [B], index [M])`."""
        device = self.centers.device
        pos_unique = torch.unique(labels)  # sorted ascending
        target_m = max(int(round(self.num_classes * self.sample_rate)), pos_unique.numel())
        target_m = min(target_m, self.num_classes)

        if target_m >= self.num_classes:
            all_idx = torch.arange(self.num_classes, device=device, dtype=torch.long)
            return self.centers, labels, all_idx

        num_neg = target_m - pos_unique.numel()
        if num_neg > 0:
            perm = torch.randperm(self.num_classes, device=device)
            is_pos = torch.isin(perm, pos_unique)
            neg = perm[~is_pos][:num_neg]
            index = torch.cat([pos_unique, neg], dim=0)
        else:
            index = pos_unique

        mapped = torch.searchsorted(pos_unique, labels)
        return self.centers[index], mapped, index

    def sample_centers(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample positive + random negative centers; returns `(sampled_weight [M, D], mapped_labels [B])`."""
        w_sub, mapped, _ = self.sample_centers_with_index(labels)
        return w_sub, mapped


class AdaFaceMarginLoss(nn.Module):
    """AdaFace quality-adaptive angular/additive margin loss (strictly FP32 cosine stability)."""

    def __init__(
        self,
        m: float = 0.4,
        h: float = 0.333,
        s: float = 64.0,
        t_alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(10.0))

    def forward(
        self,
        l2_emb: torch.Tensor,
        feat_norm: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        valid_class_mask: Optional[torch.Tensor] = None,
        update_stats: bool = True,
    ) -> torch.Tensor:
        z = F.normalize(l2_emb.float(), dim=1)
        w = F.normalize(weight.float(), dim=1)
        norms = feat_norm.detach().float().clamp(0.001, 100.0)

        if update_stats:
            with torch.no_grad():
                mean = norms.mean()
                std = norms.std(unbiased=False).clamp(min=1e-4)
                self.batch_mean.mul_(1.0 - self.t_alpha).add_(mean, alpha=self.t_alpha)
                self.batch_std.mul_(1.0 - self.t_alpha).add_(std, alpha=self.t_alpha)

        margin_scaler = ((norms - self.batch_mean) / (self.batch_std + 1e-4)) * self.h
        margin_scaler = margin_scaler.clamp(-1.0, 1.0)

        cosine = torch.matmul(z, w.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        idx = torch.arange(z.shape[0], device=z.device)
        cos_y = cosine[idx, labels].unsqueeze(1)
        theta_y = torch.acos(cos_y)

        g_angle = -self.m * margin_scaler
        g_add = self.m + self.m * margin_scaler

        cos_theta_m = torch.cos((theta_y + g_angle).clamp(0.0, math.pi)) - g_add
        logits = cosine.clone()
        logits[idx, labels] = cos_theta_m.squeeze(1)
        scaled = logits * self.s
        if valid_class_mask is not None:
            invalid = (~valid_class_mask).unsqueeze(0).expand_as(scaled).clone()
            invalid[idx, labels] = False
            scaled = scaled.masked_fill(invalid, -1e4)
        return F.cross_entropy(scaled, labels)


class AdaDistillLoss(nn.Module):
    """Combined Phase 2 AdaFace + AdaDistill dynamic teacher class-center distillation."""

    def __init__(
        self,
        num_classes: int,
        emb_dim: int = 512,
        partial_fc_rate: float = 0.2,
        lambda1_dark_adaface: float = 1.0,
        lambda2_adadistill: float = 1.5,
        lambda3_teacher_cos: float = 3.0,
        lambda4_student_cos: float = 0.5,
        lambda_clean_kd: float = 0.0,
        center_momentum: float = 0.2,
        init_partial_fc_from_teacher: bool = True,
    ) -> None:
        super().__init__()
        self.partial_fc = PartialFC(num_classes, emb_dim=emb_dim, sample_rate=partial_fc_rate)
        self.adaface = AdaFaceMarginLoss()
        self.lambda1 = lambda1_dark_adaface
        self.lambda2 = lambda2_adadistill
        self.lambda3 = lambda3_teacher_cos
        self.lambda4 = lambda4_student_cos
        self.lambda_clean_kd = lambda_clean_kd
        self.center_momentum = center_momentum
        self.init_partial_fc_from_teacher = init_partial_fc_from_teacher

        self.register_buffer("teacher_centers", F.normalize(torch.randn(num_classes, emb_dim), dim=1))
        self.register_buffer("teacher_seen", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def load_precomputed_teacher_centers(self, centers: torch.Tensor) -> None:
        """Initialize exact teacher centers & PartialFC centers before Epoch 1."""
        normed = F.normalize(centers.detach().to(self.teacher_centers.device, dtype=self.teacher_centers.dtype), p=2, dim=1)
        valid = torch.norm(centers.detach(), p=2, dim=1) > 1e-6
        self.teacher_centers.copy_(normed)
        self.teacher_seen.copy_(valid.to(self.teacher_seen.device))
        if self.init_partial_fc_from_teacher:
            self.partial_fc.centers.data[valid.to(self.partial_fc.centers.device)] = (
                normed[valid].to(self.partial_fc.centers.dtype) * 0.25
            )

    @torch.no_grad()
    def update_teacher_centers(self, z_teacher_clean: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pure-GPU vectorized EMA update of teacher identity centers without CPU sync."""
        zt = F.normalize(z_teacher_clean.float(), dim=1)
        unique_cls, inv = torch.unique(labels, return_inverse=True)
        n_uniq = unique_cls.numel()
        sums = torch.zeros((n_uniq, zt.size(1)), device=zt.device, dtype=zt.dtype)
        sums.index_add_(0, inv, zt)
        counts = torch.bincount(inv, minlength=n_uniq).to(zt.dtype).unsqueeze(1).clamp(min=1.0)
        mean_z = F.normalize(sums / counts, p=2, dim=1)

        seen_mask = self.teacher_seen[unique_cls]
        old = self.teacher_centers[unique_cls]
        blended = F.normalize((1.0 - self.center_momentum) * old + self.center_momentum * mean_z, p=2, dim=1)
        self.teacher_centers[unique_cls] = torch.where(seen_mask.unsqueeze(1), blended, mean_z)

        if self.init_partial_fc_from_teacher:
            unseen_mask = ~seen_mask
            if unseen_mask.any():
                unseen_cls = unique_cls[unseen_mask]
                # Match initial PartialFC vector norm magnitude (~0.25) while adopting teacher direction
                self.partial_fc.centers.data[unseen_cls] = (
                    mean_z[unseen_mask].to(self.partial_fc.centers.dtype) * 0.25
                )

        self.teacher_seen[unique_cls] = True
        return self.teacher_centers[labels]

    def forward(
        self,
        z_clean: torch.Tensor,
        norm_clean: torch.Tensor,
        z_dark: torch.Tensor,
        norm_dark: torch.Tensor,
        z_teacher_clean: torch.Tensor,
        labels: torch.Tensor,
        distill_warmup_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        c_teacher = self.update_teacher_centers(z_teacher_clean, labels).detach()
        w_sub, mapped_labels, sampled_idx = self.partial_fc.sample_centers_with_index(labels)

        l_ada_clean = self.adaface(z_clean, norm_clean, w_sub, mapped_labels, update_stats=True)
        l_ada_dark = self.adaface(z_dark, norm_dark, w_sub, mapped_labels, update_stats=True)

        # AdaDistill dynamic teacher-center AdaFace CE (mask unseen negative centers in early epochs)
        c_sub = self.teacher_centers[sampled_idx].detach()
        seen_sub = self.teacher_seen[sampled_idx]
        l_adadistill = self.adaface(
            z_dark,
            norm_dark,
            c_sub,
            mapped_labels,
            valid_class_mask=seen_sub,
            update_stats=False,
        )

        zd = F.normalize(z_dark.float(), dim=1)
        zc = F.normalize(z_clean.float(), dim=1)
        zt = F.normalize(z_teacher_clean.detach().float(), dim=1)

        l_t_cos = (1.0 - (zd * zt).sum(dim=1)).mean()
        l_s_cos = (1.0 - (zd * zc.detach()).sum(dim=1)).mean()
        l_clean_kd = (1.0 - (zc * zt).sum(dim=1)).mean()

        w2 = self.lambda2 * distill_warmup_scale
        w3 = self.lambda3 * distill_warmup_scale
        total = (
            l_ada_clean
            + self.lambda1 * l_ada_dark
            + w2 * l_adadistill
            + w3 * l_t_cos
            + self.lambda4 * l_s_cos
            + self.lambda_clean_kd * l_clean_kd
        )
        return {
            "loss": total,
            "ada_clean": l_ada_clean,
            "ada_dark": l_ada_dark,
            "adadistill": l_adadistill,
            "teacher_cos": l_t_cos,
            "student_cos": l_s_cos,
            "clean_kd": l_clean_kd,
        }
