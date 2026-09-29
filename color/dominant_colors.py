"""K-means palette on object pixels."""

from __future__ import annotations

import cv2
import numpy as np

from color.color_classifier import ColorClassifier


def _subsample(pixels: np.ndarray, limit: int = 1500) -> np.ndarray:
    if len(pixels) <= limit:
        return pixels
    step = max(1, len(pixels) // limit)
    return pixels[::step][:limit]


def dominant_palette(pixels_bgr: np.ndarray, classifier: ColorClassifier, k: int = 3) -> list[dict]:
    pixels = np.asarray(pixels_bgr, dtype=np.uint8).reshape(-1, 3)
    if len(pixels) == 0:
        return []
    pixels = _subsample(pixels)
    clusters = int(min(k, len(pixels)))
    if clusters == 1:
        center = pixels.mean(axis=0)
        bgr = np.clip(np.round(center), 0, 255).astype(np.uint8)
        return [_entry(bgr, 1.0, classifier)]
    data = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _compact, labels, centers = cv2.kmeans(
        data, clusters, None, criteria, 2, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=clusters).astype(np.float32)
    total = float(counts.sum()) or 1.0
    order = np.argsort(-counts)
    merged: dict[str, dict] = {}
    sequence: list[str] = []
    for index in order:
        bgr = np.clip(np.round(centers[index]), 0, 255).astype(np.uint8)
        ratio = float(counts[index] / total)
        item = _entry(bgr, ratio, classifier)
        key = item["name"]
        if key in merged:
            merged[key]["ratio"] += item["ratio"]
            continue
        merged[key] = item
        sequence.append(key)
    palette = [merged[name] for name in sequence]
    palette.sort(key=lambda item: item["ratio"], reverse=True)
    return palette[:k]


def _entry(bgr: np.ndarray, ratio: float, classifier: ColorClassifier) -> dict:
    blue, green, red = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return {
        "name": classifier.classify_bgr_pixel(bgr),
        "rgb": (red, green, blue),
        "bgr": (blue, green, red),
        "ratio": ratio,
    }
