"""Semi-transparent instance masks inside each bounding box."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(image: np.ndarray, bbox, mask_roi: np.ndarray | None, color: tuple[int, int, int], alpha: float = 0.45) -> None:
    if mask_roi is None or mask_roi.size == 0:
        return
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    mask = mask_roi
    if mask.shape[:2] != roi.shape[:2]:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    selected = mask > 127
    if not np.any(selected):
        return
    blended = roi.astype(np.float32)
    paint = np.array(color, dtype=np.float32)
    blended[selected] = blended[selected] * (1.0 - alpha) + paint * alpha
    image[y1:y2, x1:x2] = blended.astype(np.uint8)
