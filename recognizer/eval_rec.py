"""Phase 2 Evaluation: Clean-Gallery -> Dark-Probe & GT-5pt vs Detector-5pt TAR@FAR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from common.metrics import compute_tar_at_far


@torch.no_grad()
def evaluate_clean_to_dark_pairs(
    model: torch.nn.Module,
    gallery_imgs_112: torch.Tensor,
    probe_imgs_112: torch.Tensor,
    is_same: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Compute Clean-Gallery -> Dark-Probe 1:1 cosine similarity & TAR@FAR."""
    model.eval()

    def _embed(imgs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        embs: List[np.ndarray] = []
        norms: List[np.ndarray] = []
        for i in range(0, len(imgs), batch_size):
            batch = ((imgs[i : i + batch_size].to(device).float() - 0.5) / 0.5)
            z, n = model(batch)
            embs.append(z.cpu().numpy())
            norms.append(n.squeeze(-1).cpu().numpy())
        return np.concatenate(embs, axis=0), np.concatenate(norms, axis=0)

    g_emb, g_norm = _embed(gallery_imgs_112)
    p_emb, p_norm = _embed(probe_imgs_112)
    cos_scores = (g_emb * p_emb).sum(axis=1)
    same_mask = np.asarray(is_same, dtype=bool)
    res = compute_tar_at_far(cos_scores[same_mask], cos_scores[~same_mask])
    res["probe_norm_genuine_mean"] = float(p_norm[same_mask].mean()) if same_mask.any() else 0.0
    res["gallery_norm_mean"] = float(g_norm.mean()) if len(g_norm) else 0.0
    return res
