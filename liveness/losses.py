"""Phase 3 Losses: Focal-CE, Pseudo-3D Depth, Clarity, SSDG Contrastive, KD-Logits, and Patch-Affinity-KD."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits.float(), targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class SSDGSingleSideDomainLoss(nn.Module):
    """Single-Side Domain Generalization: compact Live embeddings, separate Live vs Spoof clusters."""

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(feats.float(), dim=1)
        live_mask = labels == 0
        spoof_mask = labels != 0
        if not live_mask.any():
            return z.new_zeros(())
        live_center = F.normalize(z[live_mask].mean(dim=0, keepdim=True), dim=1)
        live_compact = (1.0 - (z[live_mask] * live_center).sum(dim=1)).mean()
        if not spoof_mask.any():
            return live_compact
        spoof_sim = (z[spoof_mask] * live_center).sum(dim=1)
        push = F.relu(spoof_sim - (1.0 - self.margin)).mean()
        return live_compact + push


class PatchAffinityKDLoss(nn.Module):
    """Align Student 1.5x spatial patch affinity matrix to DINOv2-with-Registers pure patch affinity."""

    def forward(self, student_tex_fmap: torch.Tensor, teacher_affinity: torch.Tensor) -> torch.Tensor:
        # student_tex_fmap: [B, T, C, h, w], teacher_affinity: [B, T, P_t, P_t]
        b, t, c, h, w = student_tex_fmap.shape
        p_t = teacher_affinity.shape[-1]
        grid_t = int(round(p_t ** 0.5))

        fmap_flat = student_tex_fmap.reshape(b * t, c, h, w).float()
        if grid_t * grid_t == p_t and (h, w) != (grid_t, grid_t):
            fmap_flat = F.interpolate(fmap_flat, size=(grid_t, grid_t), mode="bilinear", align_corners=False)

        s_tok = F.normalize(fmap_flat.flatten(2).transpose(1, 2), dim=-1)  # [B*T, P, C]
        s_aff = torch.bmm(s_tok, s_tok.transpose(1, 2))
        t_aff = teacher_affinity.detach().float().reshape(b * t, s_aff.shape[1], s_aff.shape[2])
        return F.smooth_l1_loss(s_aff, t_aff)


class FASCombinedLoss(nn.Module):
    """Composite Phase 3 loss matching Section 七.4 equation."""

    def __init__(
        self,
        w_depth: float = 0.5,
        w_clarity: float = 0.3,
        w_kd_logits: float = 1.0,
        w_patch_kd: float = 1.0,
        w_ssdg: float = 0.2,
        kd_temp: float = 2.0,
    ) -> None:
        super().__init__()
        self.focal = FocalCrossEntropyLoss(gamma=2.0)
        self.ssdg = SSDGSingleSideDomainLoss()
        self.patch_kd = PatchAffinityKDLoss()
        self.w_depth = w_depth
        self.w_clarity = w_clarity
        self.w_kd_logits = w_kd_logits
        self.w_patch_kd = w_patch_kd
        self.w_ssdg = w_ssdg
        self.kd_temp = kd_temp

    def forward(
        self,
        stu_out: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        depth_target: Optional[torch.Tensor] = None,
        clarity_target: Optional[torch.Tensor] = None,
        teacher_out: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        l_ce = self.focal(stu_out["spoof_logits"], labels)
        l_ssdg = self.ssdg(stu_out["pooled_feat"], labels)

        # Pseudo-3D depth map supervision: planar spoofs (1,2,4) strictly pushed to 0 plane
        l_depth = l_ce.new_zeros(())
        if depth_target is not None:
            l_depth = F.l1_loss(stu_out["depth_pred"].float(), depth_target.float())

        l_clarity = l_ce.new_zeros(())
        if clarity_target is not None:
            l_clarity = F.mse_loss(stu_out["clarity_score"].float().squeeze(-1), clarity_target.float())

        l_kd = l_ce.new_zeros(())
        l_patch = l_ce.new_zeros(())
        if teacher_out is not None:
            s_log = F.log_softmax(stu_out["spoof_logits"].float() / self.kd_temp, dim=-1)
            t_prob = F.softmax(teacher_out["logits"].detach().float() / self.kd_temp, dim=-1)
            l_kd = F.kl_div(s_log, t_prob, reduction="batchmean") * (self.kd_temp ** 2)
            l_patch = self.patch_kd(stu_out["tex_fmap"], teacher_out["patch_affinity"])

        total = (
            l_ce
            + self.w_depth * l_depth
            + self.w_clarity * l_clarity
            + self.w_kd_logits * l_kd
            + self.w_patch_kd * l_patch
            + self.w_ssdg * l_ssdg
        )
        return {
            "loss": total,
            "ce_loss": l_ce,
            "depth_loss": l_depth,
            "clarity_loss": l_clarity,
            "kd_loss": l_kd,
            "patch_kd_loss": l_patch,
            "ssdg_loss": l_ssdg,
        }
