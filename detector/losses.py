"""Phase 1 Detector losses: TAL assigner, DFL + EIoU, masked 5-pt WingLoss, and HLA feature distillation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.nms import box_eiou_loss, box_iou


class WingLoss(nn.Module):
    """WingLoss for robust 5-point landmark regression."""

    def __init__(self, w: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs()
        small = diff < self.w
        loss = torch.where(small, self.w * torch.log1p(diff / self.epsilon), diff - self.c)
        return loss


class HLAFeatureDistillLoss(nn.Module):
    """HLA-style cross-illumination FPN distillation (Channel-Wise Divergence + Cosine) on P3/P4/P5."""

    def __init__(self, student_channels: List[int], teacher_channels: List[int], temp: float = 4.0) -> None:
        super().__init__()
        self.temp = temp
        self.adapters = nn.ModuleList(
            nn.Conv2d(sc, tc, kernel_size=1, bias=False) if sc != tc else nn.Identity()
            for sc, tc in zip(student_channels, teacher_channels)
        )

    def forward(self, student_feats: List[torch.Tensor], teacher_feats: List[torch.Tensor]) -> torch.Tensor:
        total = student_feats[0].new_zeros((), dtype=torch.float32)
        for adapter, sf, tf in zip(self.adapters, student_feats, teacher_feats):
            sf_proj = adapter(sf).float()
            tf_f = tf.detach().float()
            if sf_proj.shape[-2:] != tf_f.shape[-2:]:
                sf_proj = F.interpolate(sf_proj, size=tf_f.shape[-2:], mode="bilinear", align_corners=False)

            b, c, h, w = sf_proj.shape
            # Channel-wise KL divergence (CWD)
            s_prob = F.log_softmax(sf_proj.view(b, c, h * w) / self.temp, dim=-1)
            t_prob = F.softmax(tf_f.view(b, c, h * w) / self.temp, dim=-1)
            cwd = F.kl_div(s_prob, t_prob, reduction="batchmean") * (self.temp ** 2) / max(c, 1)

            # Spatial-channel cosine alignment
            cos = 1.0 - F.cosine_similarity(sf_proj, tf_f, dim=1).mean()
            total = total + cwd + 0.5 * cos
        return total / max(len(self.adapters), 1)


def _generate_anchor_points(img_hw: Tuple[int, int], strides: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ih, iw = img_hw
    pts: List[torch.Tensor] = []
    stride_vec: List[torch.Tensor] = []
    for s in strides:
        fh, fw = ih // s, iw // s
        sy = (torch.arange(fh, device=device, dtype=torch.float32) + 0.5) * float(s)
        sx = (torch.arange(fw, device=device, dtype=torch.float32) + 0.5) * float(s)
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        pts.append(torch.stack((gx, gy), dim=-1).reshape(-1, 2))
        stride_vec.append(torch.full((fh * fw,), float(s), device=device, dtype=torch.float32))
    return torch.cat(pts, dim=0), torch.cat(stride_vec, dim=0)


class DetectorLoss(nn.Module):
    """Combined TAL-style Anchor-Free Detector Loss with strict `has_landmarks` mask."""

    def __init__(
        self,
        strides: List[int],
        reg_max: int = 16,
        topk: int = 10,
        cls_weight: float = 1.0,
        box_weight: float = 5.0,
        dfl_weight: float = 1.5,
        kpt_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.strides = strides
        self.reg_max = reg_max
        self.topk = topk
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        self.kpt_weight = kpt_weight
        self.wing = WingLoss(w=10.0, epsilon=2.0)

    @torch.no_grad()
    def _assign_tal(
        self,
        anchor_pts: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Task-Aligned Assigner per image. Returns (fg_mask [N], target_box [N,4], matched_gt_idx [N], target_score [N])."""
        n_anc = anchor_pts.shape[0]
        valid_idx = gt_valid.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return (
                torch.zeros(n_anc, dtype=torch.bool, device=anchor_pts.device),
                torch.zeros(n_anc, dtype=torch.long, device=anchor_pts.device),
                torch.zeros(n_anc, dtype=torch.float32, device=anchor_pts.device),
            )

        gboxes = gt_boxes[valid_idx]  # [M, 4]
        # Inside GT box check
        ax = anchor_pts[:, 0:1]
        ay = anchor_pts[:, 1:2]
        in_box = (ax >= gboxes[None, :, 0]) & (ax <= gboxes[None, :, 2]) & (ay >= gboxes[None, :, 1]) & (ay <= gboxes[None, :, 3])
        ious = box_iou(pred_boxes.float(), gboxes.float())  # [N, M]
        align_metric = (pred_scores.float().clamp(1e-6, 1.0)[:, None] ** 0.5) * (ious.clamp(min=0.0) ** 6.0)
        align_metric = align_metric * in_box.float()

        # Select top-k candidates per GT
        k = min(self.topk, n_anc)
        _, topk_idx = align_metric.topk(k, dim=0)
        cand_mask = torch.zeros_like(in_box)
        cand_mask.scatter_(0, topk_idx, True)
        cand_mask = cand_mask & in_box

        # Resolve multi-GT conflicts by highest IoU
        masked_iou = ious * cand_mask.float()
        best_iou, best_m = masked_iou.max(dim=1)
        fg_mask = cand_mask.any(dim=1)
        matched_gt = valid_idx[best_m]
        target_scores = torch.where(fg_mask, best_iou.clamp(0.0, 1.0), torch.zeros_like(best_iou))
        return fg_mask, matched_gt, target_scores

    def forward(
        self,
        outputs: Dict[str, object],
        img_hw: Tuple[int, int],
        gt_boxes: torch.Tensor,
        gt_valid: torch.Tensor,
        gt_kpts: Optional[torch.Tensor] = None,
        has_landmarks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute supervised detection + masked 5-pt landmark losses.

        Args:
            gt_boxes: `[B, M, 4]` xyxy in pixels.
            gt_valid: `[B, M]` bool mask for valid GT faces.
            gt_kpts: `[B, M, 5, 2]` pixel coordinates (ignored whenever `has_landmarks[b] == 0`).
            has_landmarks: `[B]` bool/float mask (1 for WIDER/300W, 0 for DarkFace).
        """
        pred_boxes = outputs["boxes_xyxy"]
        pred_scores = outputs["scores"]
        pred_kpts = outputs["kpts"]
        cls_logits = outputs["cls_logits"]
        dfl_dist = outputs["dfl_dist"]

        b, n_anc, _ = pred_boxes.shape
        device = pred_boxes.device
        anchor_pts, anchor_strides = _generate_anchor_points(img_hw, self.strides, device)
        if anchor_pts.shape[0] != n_anc:
            raise ValueError(f"Anchor count mismatch: {anchor_pts.shape[0]} vs {n_anc}")

        total_cls = pred_boxes.new_zeros((), dtype=torch.float32)
        total_box = pred_boxes.new_zeros((), dtype=torch.float32)
        total_dfl = pred_boxes.new_zeros((), dtype=torch.float32)
        total_kpt = pred_boxes.new_zeros((), dtype=torch.float32)
        fg_count = 0
        kpt_fg_count = 0

        for i in range(b):
            fg_mask, matched_gt, target_score = self._assign_tal(
                anchor_pts, pred_boxes[i].detach(), pred_scores[i].detach(), gt_boxes[i], gt_valid[i]
            )
            # Classification BCE with soft TAL alignment target
            cls_loss_i = F.binary_cross_entropy_with_logits(
                cls_logits[i].float(), target_score.float(), reduction="sum"
            )
            total_cls = total_cls + cls_loss_i

            n_fg = int(fg_mask.sum().item())
            if n_fg == 0:
                continue
            fg_count += n_fg

            t_boxes = gt_boxes[i][matched_gt[fg_mask]].float()
            p_boxes = pred_boxes[i][fg_mask].float()
            weights = target_score[fg_mask].float().clamp(min=0.1)

            eiou = box_eiou_loss(p_boxes, t_boxes)
            total_box = total_box + (eiou * weights).sum()

            # DFL target in grid units
            ap = anchor_pts[fg_mask]
            st = anchor_strides[fg_mask].unsqueeze(-1)
            tl = (ap[:, 0:1] - t_boxes[:, 0:1]) / st
            tt = (ap[:, 1:2] - t_boxes[:, 1:2]) / st
            tr = (t_boxes[:, 2:3] - ap[:, 0:1]) / st
            tb = (t_boxes[:, 3:4] - ap[:, 1:2]) / st
            target_ltrb = torch.cat((tl, tt, tr, tb), dim=-1).clamp(0.0, float(self.reg_max - 1) - 0.01)

            tl_int = target_ltrb.long()
            tr_int = (tl_int + 1).clamp(max=self.reg_max - 1)
            wl = tr_int.float() - target_ltrb
            wr = 1.0 - wl

            p_dfl = dfl_dist[i][fg_mask].reshape(-1, self.reg_max)
            dfl_loss = (
                F.cross_entropy(p_dfl, tl_int.reshape(-1), reduction="none") * wl.reshape(-1)
                + F.cross_entropy(p_dfl, tr_int.reshape(-1), reduction="none") * wr.reshape(-1)
            ).view(n_fg, 4).mean(dim=-1)
            total_dfl = total_dfl + (dfl_loss * weights).sum()

            # Strictly gated 5-pt WingLoss! DarkFace (has_landmarks == 0) receives ZERO gradient!
            if gt_kpts is not None and has_landmarks is not None and bool(has_landmarks[i].item()):
                t_kpts = gt_kpts[i][matched_gt[fg_mask]].float()  # [n_fg, 5, 2]
                p_kpts_xy = pred_kpts[i][fg_mask, :, :2].float()
                # Normalize pixel error by box diagonal/stride to stabilize scale
                scale_norm = (t_boxes[:, 2] - t_boxes[:, 0]).clamp(min=8.0).view(n_fg, 1, 1) / 32.0
                wing_err = self.wing(p_kpts_xy / scale_norm, t_kpts / scale_norm).mean(dim=(1, 2))
                total_kpt = total_kpt + (wing_err * weights).sum()
                kpt_fg_count += n_fg

        denom = max(float(fg_count), 1.0)
        l_cls = (total_cls / denom) * self.cls_weight
        l_box = (total_box / denom) * self.box_weight
        l_dfl = (total_dfl / denom) * self.dfl_weight
        l_kpt = (total_kpt / max(float(kpt_fg_count), 1.0)) * self.kpt_weight if kpt_fg_count > 0 else pred_boxes.new_zeros(())

        return {
            "loss": l_cls + l_box + l_dfl + l_kpt,
            "cls_loss": l_cls,
            "box_loss": l_box,
            "dfl_loss": l_dfl,
            "kpt_loss": l_kpt,
        }
