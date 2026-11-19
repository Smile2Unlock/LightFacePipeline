"""Phase 2 Losses: AdaFace Quality-Adaptive Margin, Partial FC, and AdaDistill Dynamic Class Centers."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialFC(nn.Module):
    """Memory-efficient Partial FC class-center store supporting 200K+ WebFace4M IDs on single 64GB GPU."""

    def __init__(self, num_classes: int, emb_dim: int = 512, sample_rate: float = 0.2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.sample_rate = sample_rate
        self.centers = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.01)

    def sample_centers(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample positive + random negative centers; returns `(sampled_weight [M, D], mapped_labels [B])`."""
        device = self.centers.device
        pos_unique = torch.unique(labels)
        target_m = max(int(round(self.num_classes * self.sample_rate)), pos_unique.numel())
        target_m = min(target_m, self.num_classes)

        if target_m >= self.num_classes:
            return self.centers, labels

        # Sample negative centers not in pos_unique
        num_neg = target_m - pos_unique.numel()
        if num_neg > 0:
            perm = torch.randperm(self.num_classes, device=device)
            is_pos = torch.isin(perm, pos_unique)
            neg = perm[~is_pos][:num_neg]
            index = torch.cat([pos_unique, neg], dim=0)
        else:
            index = pos_unique

        # Map original labels -> [0, len(index)-1]
        mapped = torch.searchsorted(pos_unique, labels)
        return self.centers[index], mapped


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
    ) -> torch.Tensor:
        # Strict FP32 math for ROCm FP16 safety
        z = F.normalize(l2_emb.float(), dim=1)
        w = F.normalize(weight.float(), dim=1)
        norms = feat_norm.detach().float().clamp(0.001, 100.0)

        with torch.no_grad():
            mean = norms.mean()
            std = norms.std(unbiased=False).clamp(min=1e-4)
            self.batch_mean.mul_(1.0 - self.t_alpha).add_(mean, alpha=self.t_alpha)
            self.batch_std.mul_(1.0 - self.t_alpha).add_(std, alpha=self.t_alpha)

        margin_scaler = ((norms - self.batch_mean) / (self.batch_std + 1e-4)) * self.h
        margin_scaler = margin_scaler.clamp(-1.0, 1.0)

        cosine = torch.matmul(z, w.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # Positive class adaptive g_angle & g_add
        idx = torch.arange(z.shape[0], device=z.device)
        cos_y = cosine[idx, labels].unsqueeze(1)
        sin_y = torch.sqrt((1.0 - cos_y * cos_y).clamp(min=1e-6))
        theta_y = torch.acos(cos_y)

        g_angle = -self.m * margin_scaler
        g_add = self.m + self.m * margin_scaler

        cos_theta_m = torch.cos((theta_y + g_angle).clamp(0.0, math.pi)) - g_add
        logits = cosine.clone()
        logits[idx, labels] = cos_theta_m.squeeze(1)
        return F.cross_entropy(logits * self.s, labels)


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
        center_momentum: float = 0.2,
    ) -> None:
        super().__init__()
        self.partial_fc = PartialFC(num_classes, emb_dim=emb_dim, sample_rate=partial_fc_rate)
        self.adaface = AdaFaceMarginLoss()
        self.lambda1 = lambda1_dark_adaface
        self.lambda2 = lambda2_adadistill
        self.lambda3 = lambda3_teacher_cos
        self.lambda4 = lambda4_student_cos
        self.center_momentum = center_momentum

        # Exponentially updated dynamic teacher class centers on unit hypersphere (stored in buffer)
        self.register_buffer("teacher_centers", F.normalize(torch.randn(num_classes, emb_dim), dim=1))
        self.register_buffer("teacher_seen", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update_teacher_centers(self, z_teacher_clean: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        zt = F.normalize(z_teacher_clean.float(), dim=1)
        for cls_id in torch.unique(labels):
            mask = labels == cls_id
            mean_vec = F.normalize(zt[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            if not bool(self.teacher_seen[cls_id].item()):
                self.teacher_centers[cls_id].copy_(mean_vec)
                self.teacher_seen[cls_id] = True
            else:
                updated = (1.0 - self.center_momentum) * self.teacher_centers[cls_id] + self.center_momentum * mean_vec
                self.teacher_centers[cls_id].copy_(F.normalize(updated, dim=0))
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
        w_sub, mapped_labels = self.partial_fc.sample_centers(labels)

        l_ada_clean = self.adaface(z_clean, norm_clean, w_sub, mapped_labels)
        l_ada_dark = self.adaface(z_dark, norm_dark, w_sub, mapped_labels)

        c_teacher = self.update_teacher_centers(z_teacher_clean, labels).detach()
        zd = F.normalize(z_dark.float(), dim=1)
        zc = F.normalize(z_clean.float(), dim=1)
        zt = F.normalize(z_teacher_clean.detach().float(), dim=1)

        l_adadistill = (1.0 - (zd * c_teacher).sum(dim=1)).mean()
        l_t_cos = (1.0 - (zd * zt).sum(dim=1)).mean()
        l_s_cos = (1.0 - (zd * zc.detach()).sum(dim=1)).mean()

        w2 = self.lambda2 * distill_warmup_scale
        w3 = self.lambda3 * distill_warmup_scale
        total = l_ada_clean + self.lambda1 * l_ada_dark + w2 * l_adadistill + w3 * l_t_cos + self.lambda4 * l_s_cos
        return {
            "loss": total,
            "ada_clean": l_ada_clean,
            "ada_dark": l_ada_dark,
            "adadistill": l_adadistill,
            "teacher_cos": l_t_cos,
            "student_cos": l_s_cos,
        }
