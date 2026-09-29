"""Stable overlay colors for track ids."""

from __future__ import annotations

import cv2
import numpy as np


def id_color(track_id: int) -> tuple[int, int, int]:
    hue = int((int(track_id) * 47) % 180)
    pixel = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
