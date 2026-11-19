"""Unified contracts across Detection+5pt, Recognition, and Temporal Liveness."""

from __future__ import annotations

import numpy as np

COLOR_SPACE = "RGB"

LANDMARK_ORDER = (
    "left_eye_center",
    "right_eye_center",
    "nose_tip",
    "left_mouth_corner",
    "right_mouth_corner",
)

# Standard ArcFace / CVLface 112x112 similarity alignment target coordinates (pixels)
ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

REC_INPUT_SIZE = 112
REC_NORM_MEAN = 0.5
REC_NORM_STD = 0.5
REC_EMBEDDING_DIM = 512

FAS_CLIP_LEN = 8
FAS_CROP_SIZE = 128
FAS_SCALES = (1.5, 2.7)  # (texture scale, context scale)
FAS_CLASSES = ("live", "print", "replay", "mask_cutout", "frozen_rigid")

# Module-specific minimum safe face luma thresholds in [0, 255]
DET_MIN_FACE_LUMA_255 = 10.0
REC_TRIGGER_LUMA_FLOOR_255 = 22.0
REC_SAFE_LUMA_FLOOR_255 = 25.0
REC_SAFE_LUMA_CEIL_255 = 220.0
