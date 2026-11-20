"""Phase 1 Detector losses: fast candidate-filtered TAL assigner, DFL + EIoU, masked 5-pt WingLoss, and HLA distillation."""

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
        return torch.where(small, self.w * torch.log1p(diff / self.epsilon), diff - self.c)


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
            s_prob = F.log_softmax(sf_proj.view(b, c, h * w) / self.temp, dim=-1)
            t_prob = F.softmax(tf_f.view(b, c, h * w) / self.temp, dim=-1)
            cwd = F.kl_div(s_prob, t_prob, reduction="batchmean") * (self.temp ** 2) / max(c, 1)
            cos = 1.0 - F.cosine_similarity(sf_proj, tf_f, dim=1).mean()
            total = total + cwd + 0.5 * cos
        return total / max(len(self.adapters), 1)


def _generate_anchor_points(
    img_hw: Tuple[int, int], strides: List[int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """Combined TAL-style Anchor-Free Detector Loss with per-face landmark validity masking."""

    def __init__(
        self,
        strides: List[int],
        reg_max: int = 12,
        topk: int = 10,
        cls_weight: float = 0.5,
        box_weight: float = 7.5,
        dfl_weight: float = 1.5,
        kpt_weight: float = 1.2,
    ) -> None:
        super().__init__()
        self.strides = strides
        self.reg_max = reg_max
        self.topk = topk
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        self.kpt_weight = kpt_weight
        self.wing = WingLoss(w=6.0, epsilon=1.0)
        self._anchor_cache: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_anchors(self, img_hw: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (img_hw[0], img_hw[1], str(device))
        cached = self._anchor_cache.get(key)
        if cached is None:
            cached = _generate_anchor_points(img_hw, self.strides, device)
            self._anchor_cache[key] = cached
        return cached

    @torch.no_grad()
    def _assign_tal(
        self,
        anchor_pts: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Candidate-filtered Task-Aligned Assigner with tiny-face nearest-anchor fallback & soft target norm."""
        n_anc = anchor_pts.shape[0]
        valid_idx = gt_valid.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return (
                torch.zeros(n_anc, dtype=torch.bool, device=anchor_pts.device),
                torch.zeros(n_anc, dtype=torch.long, device=anchor_pts.device),
                torch.zeros(n_anc, dtype=torch.float32, device=anchor_pts.device),
            )

        gboxes = gt_boxes[valid_idx]  # [M, 4]
        ax = anchor_pts[:, 0:1]
        ay = anchor_pts[:, 1:2]
        in_box = (
            (ax >= gboxes[None, :, 0])
            & (ax <= gboxes[None, :, 2])
            & (ay >= gboxes[None, :, 1])
            & (ay <= gboxes[None, :, 3])
        )

        # Tiny-face fallback: if a tiny DarkFace box (<4px) falls between grid centers, assign nearest anchor
        empty_gt = in_box.sum(dim=0) == 0
        if empty_gt.any():
            gcx = 0.5 * (gboxes[:, 0] + gboxes[:, 2])
            gcy = 0.5 * (gboxes[:, 1] + gboxes[:, 3])
            dist2 = (ax - gcx[None, :]) ** 2 + (ay - gcy[None, :]) ** 2
            nearest = dist2.argmin(dim=0)
            empty_idx = empty_gt.nonzero(as_tuple=False).squeeze(1)
            in_box[nearest[empty_idx], empty_idx] = True

        cand_rows = in_box.any(dim=1)
        cand_idx = cand_rows.nonzero(as_tuple=False).squeeze(1)
        if cand_idx.numel() == 0:
            return (
                torch.zeros(n_anc, dtype=torch.bool, device=anchor_pts.device),
                torch.zeros(n_anc, dtype=torch.long, device=anchor_pts.device),
                torch.zeros(n_anc, dtype=torch.float32, device=anchor_pts.device),
            )

        c_in_box = in_box[cand_idx]  # [K, M]
        c_ious = box_iou(pred_boxes[cand_idx].float(), gboxes.float())  # [K, M]
        c_scores = pred_scores[cand_idx].float().clamp(1e-6, 1.0)[:, None]  # [K, 1]
        c_align = (c_scores ** 0.5) * (c_ious.clamp(min=1e-3) ** 6.0) * c_in_box.float()

        k = min(self.topk, cand_idx.numel())
        _, topk_c = c_align.topk(k, dim=0)
        c_cand_mask = torch.zeros_like(c_in_box)
        c_cand_mask.scatter_(0, topk_c, True)
        c_cand_mask = c_cand_mask & c_in_box

        c_masked_iou_pos = c_ious * c_cand_mask.float()
        c_masked_iou_tie = torch.where(c_cand_mask, c_ious, torch.full_like(c_ious, -1.0))
        best_iou_c, best_m_c = c_masked_iou_tie.max(dim=1)
        c_fg = c_cand_mask.any(dim=1)

        # YOLO11 TAL soft-target normalization: prevents score^0.5 * iou^6 from vanishing at epoch 0
        c_pos_align = c_align * c_cand_mask.float()
        max_align_per_gt = c_pos_align.amax(dim=0, keepdim=True).clamp(min=1e-20)
        max_iou_per_gt = c_masked_iou_pos.amax(dim=0, keepdim=True)
        norm_align = c_pos_align * (max_iou_per_gt / max_align_per_gt)
        c_target_scores = torch.where(c_fg, norm_align.amax(dim=1).clamp(0.0, 1.0), torch.zeros_like(best_iou_c))

        fg_mask = torch.zeros(n_anc, dtype=torch.bool, device=anchor_pts.device)
        matched_gt = torch.zeros(n_anc, dtype=torch.long, device=anchor_pts.device)
        target_scores = torch.zeros(n_anc, dtype=torch.float32, device=anchor_pts.device)
        fg_anchors = cand_idx[c_fg]
        fg_mask[fg_anchors] = True
        matched_gt[fg_anchors] = valid_idx[best_m_c[c_fg]]
        target_scores[fg_anchors] = c_target_scores[c_fg]
        return fg_mask, matched_gt, target_scores

    def forward(
        self,
        outputs: Dict[str, object],
        img_hw: Tuple[int, int],
        gt_boxes: torch.Tensor,
        gt_valid: torch.Tensor,
        gt_kpts: Optional[torch.Tensor] = None,
        has_landmarks: Optional[torch.Tensor] = None,
        gt_kpts_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute supervised detection + strictly masked 5-pt landmark losses (batched, minimal GPU sync)."""
        pred_boxes = outputs["boxes_xyxy"]
        pred_scores = outputs["scores"]
        pred_kpts = outputs["kpts"]
        cls_logits = outputs["cls_logits"]
        dfl_dist = outputs["dfl_dist"]

        b, n_anc, _ = pred_boxes.shape
        device = pred_boxes.device
        anchor_pts, anchor_strides = self._get_anchors(img_hw, device)
        if anchor_pts.shape[0] != n_anc:
            raise ValueError(f"Anchor count mismatch: {anchor_pts.shape[0]} vs {n_anc}")

        fg_masks = torch.zeros((b, n_anc), dtype=torch.bool, device=device)
        matched_gts = torch.zeros((b, n_anc), dtype=torch.long, device=device)
        target_scores = torch.zeros((b, n_anc), dtype=torch.float32, device=device)

        valid_counts = gt_valid.sum(dim=1).tolist()
        with torch.no_grad():
            for i in range(b):
                if valid_counts[i] == 0:
                    continue
                fg_i, mgt_i, tsc_i = self._assign_tal(
                    anchor_pts, pred_boxes[i].detach(), pred_scores[i].detach(), gt_boxes[i], gt_valid[i]
                )
                fg_masks[i] = fg_i
                matched_gts[i] = mgt_i
                target_scores[i] = tsc_i

        total_cls = F.binary_cross_entropy_with_logits(cls_logits.float(), target_scores, reduction="sum")
        target_score_sum = target_scores.sum()

        total_box = pred_boxes.new_zeros((), dtype=torch.float32)
        total_dfl = pred_boxes.new_zeros((), dtype=torch.float32)
        total_kpt = pred_boxes.new_zeros((), dtype=torch.float32)
        kpt_weight_sum = pred_boxes.new_zeros((), dtype=torch.float32)
        has_kpt_loss = False

        b_idx, a_idx = fg_masks.nonzero(as_tuple=True)
        if b_idx.numel() > 0:
            m_idx = matched_gts[b_idx, a_idx]
            t_boxes = gt_boxes[b_idx, m_idx].float()
            p_boxes = pred_boxes[b_idx, a_idx].float()
            weights = target_scores[b_idx, a_idx].float().clamp(min=0.05)

            eiou = box_eiou_loss(p_boxes, t_boxes)
            total_box = (eiou * weights).sum()

            ap = anchor_pts[a_idx]
            st = anchor_strides[a_idx].unsqueeze(-1)
            tl = (ap[:, 0:1] - t_boxes[:, 0:1]) / st
            tt = (ap[:, 1:2] - t_boxes[:, 1:2]) / st
            tr = (t_boxes[:, 2:3] - ap[:, 0:1]) / st
            tb = (t_boxes[:, 3:4] - ap[:, 1:2]) / st
            target_ltrb = torch.cat((tl, tt, tr, tb), dim=-1).clamp(0.0, float(self.reg_max - 1) - 0.01)

            tl_int = target_ltrb.long()
            tr_int = (tl_int + 1).clamp(max=self.reg_max - 1)
            wl = tr_int.float() - target_ltrb
            wr = 1.0 - wl

            p_dfl = dfl_dist[b_idx, a_idx].float().reshape(-1, self.reg_max)
            dfl_loss = (
                F.cross_entropy(p_dfl, tl_int.reshape(-1), reduction="none") * wl.reshape(-1)
                + F.cross_entropy(p_dfl, tr_int.reshape(-1), reduction="none") * wr.reshape(-1)
            ).view(-1, 4).mean(dim=-1)
            total_dfl = (dfl_loss * weights).sum()

            # Strictly gated 5-pt WingLoss: BOTH image-level `has_landmarks[b_idx]` AND per-face `gt_kpts_valid`
            if gt_kpts is not None and has_landmarks is not None:
                k_mask = has_landmarks[b_idx]
                if gt_kpts_valid is not None:
                    k_mask = k_mask & gt_kpts_valid[b_idx, m_idx]
                if k_mask.any():
                    has_kpt_loss = True
                    bk = b_idx[k_mask]
                    ak = a_idx[k_mask]
                    mk = m_idx[k_mask]
                    t_kpts = gt_kpts[bk, mk].float()
                    p_kpts_xy = pred_kpts[bk, ak, :, :2].float()
                    kb = t_boxes[k_mask]
                    kw = weights[k_mask]
                    # Scale-invariant area-root norm: sub-pixel 1-3% NME sits inside WingLoss log cusp
                    scale_norm = torch.sqrt(
                        (kb[:, 2] - kb[:, 0]).clamp(min=4.0) * (kb[:, 3] - kb[:, 1]).clamp(min=4.0)
                    ).view(-1, 1, 1) / 32.0
                    wing_err = self.wing(p_kpts_xy / scale_norm, t_kpts / scale_norm).mean(dim=(1, 2))
                    total_kpt = (wing_err * kw).sum()
                    kpt_weight_sum = kw.sum()

        denom = target_score_sum.clamp(min=1.0)
        l_cls = (total_cls / denom) * self.cls_weight
        l_box = (total_box / denom) * self.box_weight
        l_dfl = (total_dfl / denom) * self.dfl_weight
        l_kpt = (
            (total_kpt / kpt_weight_sum.clamp(min=1.0)) * self.kpt_weight
            if has_kpt_loss
            else pred_boxes.new_zeros(())
        )

        return {
            "loss": l_cls + l_box + l_dfl + l_kpt,
            "cls_loss": l_cls,
            "box_loss": l_box,
            "dfl_loss": l_dfl,
            "kpt_loss": l_kpt,
        }
