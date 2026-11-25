"""Phase 2 Losses: AdaFace Quality-Adaptive Margin, Partial FC, and Rank-Compatible AdaDistill."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialFC(nn.Module):
    """Memory-efficient Partial FC class-center store supporting 40K–200K+ IDs on single GPU."""

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


def _group_mean_unit(z: torch.Tensor, unique_cls: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    sums = torch.zeros((unique_cls.numel(), z.size(1)), device=z.device, dtype=z.dtype)
    sums.index_add_(0, inv, z)
    counts = torch.bincount(inv, minlength=unique_cls.numel()).to(z.dtype).unsqueeze(1).clamp(min=1.0)
    return F.normalize(sums / counts, p=2, dim=1)


class AdaDistillLoss(nn.Module):
    """Combined Phase 2 AdaFace + AdaDistill (AdaFace CE or Prototype KL) + Clean Reference Anchor."""

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
        lambda_ada_clean: float = 1.0,
        lambda_ref_clean: float = 0.0,
        lambda_dark_ref: float = 0.0,
        lambda_rel_kd: float = 0.0,
        use_kd_projector: bool = False,
        adadistill_mode: str = "adaface",
        kd_temp: float = 16.0,
        center_momentum: float = 0.2,
        init_partial_fc_from_teacher: bool = True,
    ) -> None:
        super().__init__()
        self.partial_fc = PartialFC(num_classes, emb_dim=emb_dim, sample_rate=partial_fc_rate)
        self.adaface = AdaFaceMarginLoss()
        self.lambda_ada_clean = lambda_ada_clean
        self.lambda1 = lambda1_dark_adaface
        self.lambda2 = lambda2_adadistill
        self.lambda3 = lambda3_teacher_cos
        self.lambda4 = lambda4_student_cos
        self.lambda_clean_kd = lambda_clean_kd
        self.lambda_ref_clean = lambda_ref_clean
        self.lambda_dark_ref = lambda_dark_ref
        self.lambda_rel_kd = lambda_rel_kd
        self.adadistill_mode = adadistill_mode
        self.kd_temp = kd_temp
        self.center_momentum = center_momentum
        self.init_partial_fc_from_teacher = init_partial_fc_from_teacher

        if use_kd_projector:
            self.kd_proj: Optional[nn.Linear] = nn.Linear(emb_dim, emb_dim, bias=False)
            nn.init.eye_(self.kd_proj.weight)
        else:
            self.kd_proj = None

        self.register_buffer("teacher_centers", F.normalize(torch.randn(num_classes, emb_dim), dim=1))
        self.register_buffer("teacher_seen", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("partial_fc_seen", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def load_precomputed_teacher_centers(self, centers: torch.Tensor, init_partial_fc: bool = True) -> None:
        """Initialize exact teacher centers (and optionally PartialFC centers) before Epoch 1."""
        normed = F.normalize(centers.detach().to(self.teacher_centers.device, dtype=self.teacher_centers.dtype), p=2, dim=1)
        valid = torch.norm(centers.detach(), p=2, dim=1) > 1e-6
        self.teacher_centers.copy_(normed)
        self.teacher_seen.copy_(valid.to(self.teacher_seen.device))
        if init_partial_fc and self.init_partial_fc_from_teacher:
            v_pfc = valid.to(self.partial_fc.centers.device)
            self.partial_fc.centers.data[v_pfc] = normed[v_pfc].to(self.partial_fc.centers.dtype) * 0.25
            self.partial_fc_seen.copy_(valid.to(self.partial_fc_seen.device))

    @torch.no_grad()
    def update_teacher_centers(
        self,
        z_teacher_clean: torch.Tensor,
        labels: torch.Tensor,
        z_ref_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pure-GPU vectorized EMA update of teacher (and PartialFC) centers without CPU sync."""
        zt = F.normalize(z_teacher_clean.float(), dim=1)
        unique_cls, inv = torch.unique(labels, return_inverse=True)
        mean_z = _group_mean_unit(zt, unique_cls, inv)

        seen_mask = self.teacher_seen[unique_cls]
        old = self.teacher_centers[unique_cls]
        blended = F.normalize((1.0 - self.center_momentum) * old + self.center_momentum * mean_z, p=2, dim=1)
        self.teacher_centers[unique_cls] = torch.where(seen_mask.unsqueeze(1), blended, mean_z)
        self.teacher_seen[unique_cls] = True

        if self.init_partial_fc_from_teacher:
            pfc_seen = self.partial_fc_seen[unique_cls]
            unseen_pfc = ~pfc_seen
            if unseen_pfc.any():
                if z_ref_clean is not None:
                    zr = F.normalize(z_ref_clean.detach().float(), dim=1)
                    init_vecs = _group_mean_unit(zr, unique_cls, inv)[unseen_pfc]
                else:
                    init_vecs = mean_z[unseen_pfc]
                unseen_cls = unique_cls[unseen_pfc]
                self.partial_fc.centers.data[unseen_cls] = init_vecs.to(self.partial_fc.centers.dtype) * 0.25
            self.partial_fc_seen[unique_cls] = True

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
        z_ref_clean: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self.update_teacher_centers(z_teacher_clean, labels, z_ref_clean=z_ref_clean)
        w_sub, mapped_labels, sampled_idx = self.partial_fc.sample_centers_with_index(labels)

        l_ada_clean = self.adaface(z_clean, norm_clean, w_sub, mapped_labels, update_stats=True)
        l_ada_dark = self.adaface(z_dark, norm_dark, w_sub, mapped_labels, update_stats=True)

        zd = F.normalize(z_dark.float(), dim=1)
        zc = F.normalize(z_clean.float(), dim=1)
        zt = F.normalize(z_teacher_clean.detach().float(), dim=1)

        c_sub = self.teacher_centers[sampled_idx].detach()
        seen_sub = self.teacher_seen[sampled_idx]

        if self.adadistill_mode == "kl":
            t_logits = torch.matmul(zt, F.normalize(c_sub.float(), dim=1).t()) * self.kd_temp
            s_logits = torch.matmul(zd, F.normalize(w_sub.float(), dim=1).t()) * self.kd_temp
            invalid = (~seen_sub).unsqueeze(0).expand_as(t_logits).clone()
            idx_rows = torch.arange(zd.shape[0], device=zd.device)
            invalid[idx_rows, mapped_labels] = False
            t_logits = t_logits.masked_fill(invalid, -1e4)
            s_logits = s_logits.masked_fill(invalid, -1e4)
            l_adadistill = F.kl_div(
                F.log_softmax(s_logits, dim=1),
                F.softmax(t_logits, dim=1),
                reduction="batchmean",
            )
        else:
            l_adadistill = self.adaface(
                z_dark,
                norm_dark,
                c_sub,
                mapped_labels,
                valid_class_mask=seen_sub,
                update_stats=False,
            )

        if self.kd_proj is not None:
            zd_p = F.normalize(self.kd_proj(zd).float(), dim=1)
            zc_p = F.normalize(self.kd_proj(zc).float(), dim=1)
        else:
            zd_p = zd
            zc_p = zc

        l_t_cos = (1.0 - (zd_p * zt).sum(dim=1)).mean()
        l_clean_kd = (1.0 - (zc_p * zt).sum(dim=1)).mean()
        l_s_cos = (1.0 - (zd * zc.detach()).sum(dim=1)).mean()

        if self.lambda_rel_kd > 0.0:
            s_t = torch.matmul(zt, zt.t())
            l_rel_kd = (
                F.mse_loss(torch.matmul(zc.detach(), zd.t()), s_t)
                + 0.5 * F.mse_loss(torch.matmul(zd, zd.t()), s_t)
                + 0.5 * F.mse_loss(torch.matmul(zc, zc.t()), s_t)
            )
        else:
            l_rel_kd = zd.new_zeros(())

        if z_ref_clean is not None:
            zr = F.normalize(z_ref_clean.detach().float(), dim=1)
            l_ref_clean = (1.0 - (zc * zr).sum(dim=1)).mean()
            l_dark_ref = (1.0 - (zd * zr).sum(dim=1)).mean()
        else:
            l_ref_clean = zd.new_zeros(())
            l_dark_ref = zd.new_zeros(())

        w2 = self.lambda2 * distill_warmup_scale
        w3 = self.lambda3 * distill_warmup_scale
        total = (
            self.lambda_ada_clean * l_ada_clean
            + self.lambda1 * l_ada_dark
            + w2 * l_adadistill
            + w3 * l_t_cos
            + self.lambda4 * l_s_cos
            + self.lambda_clean_kd * l_clean_kd
            + self.lambda_rel_kd * l_rel_kd
            + self.lambda_ref_clean * l_ref_clean
            + self.lambda_dark_ref * l_dark_ref
        )
        return {
            "loss": total,
            "ada_clean": l_ada_clean,
            "ada_dark": l_ada_dark,
            "adadistill": l_adadistill,
            "teacher_cos": l_t_cos,
            "student_cos": l_s_cos,
            "clean_kd": l_clean_kd,
            "rel_kd": l_rel_kd,
            "ref_clean": l_ref_clean,
            "dark_ref": l_dark_ref,
        }
