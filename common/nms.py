"""Pure-PyTorch bounding box IoU / CIoU / EIoU and NMS (no torchvision dependency)."""

from __future__ import annotations

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between [N, 4] and [M, 4] xyxy boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def box_eiou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise EIoU loss for aligned [N, 4] xyxy predicted and target boxes."""
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = target.unbind(-1)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)

    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = pw * ph + gw * gh - inter + eps
    iou = inter / union

    cx1 = torch.minimum(px1, gx1)
    cy1 = torch.minimum(py1, gy1)
    cx2 = torch.maximum(px2, gx2)
    cy2 = torch.maximum(py2, gy2)
    cw = (cx2 - cx1).clamp(min=eps)
    ch = (cy2 - cy1).clamp(min=eps)
    c2 = cw * cw + ch * ch + eps

    pcx = 0.5 * (px1 + px2)
    pcy = 0.5 * (py1 + py2)
    gcx = 0.5 * (gx1 + gx2)
    gcy = 0.5 * (gy1 + gy2)

    rho2_center = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    rho2_w = (pw - gw) ** 2
    rho2_h = (ph - gh) ** 2

    eiou = iou - (rho2_center / c2 + rho2_w / (cw * cw + eps) + rho2_h / (ch * ch + eps))
    return 1.0 - eiou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """Pure-PyTorch greedy NMS returning kept indices sorted by descending score."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-7)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float = 0.45,
) -> torch.Tensor:
    """Category/batch-aware pure-PyTorch NMS via coordinate offset trick."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    max_coord = boxes.max()
    offsets = idxs.to(boxes.dtype) * (max_coord + 1.0)
    return nms(boxes + offsets[:, None], scores, iou_threshold=iou_threshold)
