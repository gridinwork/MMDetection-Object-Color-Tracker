"""Text blocks drawn above each object."""

from __future__ import annotations

import cv2
import numpy as np


def draw_label_block(image: np.ndarray, anchor_x: int, anchor_y: int, lines: list[str], color: tuple[int, int, int]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes) + 10
    line_h = max(size[1] for size in sizes) + 6
    height = line_h * len(lines) + 4
    x1 = max(0, anchor_x)
    y2 = anchor_y
    y1 = y2 - height
    if y1 < 0:
        y1 = min(image.shape[0] - height, anchor_y)
        y2 = y1 + height
    x2 = min(image.shape[1] - 1, x1 + width)
    y1 = max(0, int(y1))
    y2 = min(image.shape[0] - 1, int(y2))
    x1 = int(x1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    for index, line in enumerate(lines):
        baseline = y1 + (index + 1) * line_h - 4
        cv2.putText(image, line, (x1 + 4, baseline), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_palette_row(image: np.ndarray, origin_x: int, origin_y: int, palette: list[dict]) -> None:
    x = origin_x
    y = origin_y
    for item in palette[:3]:
        bgr = item.get("bgr") or (200, 200, 200)
        cv2.rectangle(image, (x, y), (x + 12, y + 12), bgr, -1)
        cv2.rectangle(image, (x, y), (x + 12, y + 12), (255, 255, 255), 1)
        text = f"{item['name']} {int(round(item['ratio'] * 100))}%"
        cv2.putText(image, text, (x + 16, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
