"""Sample object pixels and estimate dominant color."""

from __future__ import annotations

import cv2
import numpy as np

from color.color_classifier import ColorClassifier
from color.dominant_colors import dominant_palette


def sample_object_pixels(frame_bgr: np.ndarray, bbox, mask_roi: np.ndarray | None) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if mask_roi is not None and mask_roi.size > 0:
        mask = mask_roi
        if mask.shape[:2] != crop.shape[:2]:
            mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask > 127)
        if len(xs) >= 30:
            pixels = crop[ys, xs]
            return _limit(pixels)
    return _inner_pixels(crop)


def _inner_pixels(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    margin_x = max(1, int(width * 0.2))
    margin_y = max(1, int(height * 0.2))
    if width <= margin_x * 2 or height <= margin_y * 2:
        inner = crop
    else:
        inner = crop[margin_y : height - margin_y, margin_x : width - margin_x]
    pixels = inner.reshape(-1, 3)
    return _limit(pixels)


def _limit(pixels: np.ndarray, limit: int = 2500) -> np.ndarray:
    if len(pixels) <= limit:
        return pixels
    step = max(1, len(pixels) // limit)
    return pixels[::step][:limit]


def measure_color(pixels_bgr: np.ndarray, classifier: ColorClassifier, with_palette: bool) -> dict | None:
    if pixels_bgr is None or len(pixels_bgr) < 15:
        return None
    mean_bgr = pixels_bgr.mean(axis=0)
    median_bgr = np.median(pixels_bgr, axis=0)
    mean_pixel = np.uint8([[np.clip(np.round(mean_bgr), 0, 255).astype(np.uint8)]])
    median_pixel = np.uint8([[np.clip(np.round(median_bgr), 0, 255).astype(np.uint8)]])
    mean_hsv = cv2.cvtColor(mean_pixel, cv2.COLOR_BGR2HSV)[0, 0]
    median_lab = cv2.cvtColor(median_pixel, cv2.COLOR_BGR2LAB)[0, 0]
    name = classifier.classify_lab_hsv(median_lab, mean_hsv)
    blue, green, red = [int(v) for v in median_pixel[0, 0]]
    palette = dominant_palette(pixels_bgr, classifier, 3) if with_palette else []
    return {
        "name": name,
        "rgb": (red, green, blue),
        "hsv": (int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])),
        "lab": (int(median_lab[0]), int(median_lab[1]), int(median_lab[2])),
        "palette": palette,
    }


class ColorAnalyzer:
    """Updates track colors on a stride so clustering is not done every frame."""

    def __init__(self) -> None:
        self.classifier = ColorClassifier()
        self._cache: dict[int, dict] = {}
        self._spatial: list[dict] = []

    def reset(self) -> None:
        self._cache.clear()
        self._spatial.clear()

    def apply(self, tracks, frame_bgr, frame_index: int, every: int, use_mask: bool, palette: bool, tracking: bool) -> None:
        every = max(1, int(every))
        used_ids = set()
        refreshed_spatial = []
        for track in tracks:
            info = None
            if tracking:
                cached = self._cache.get(track.track_id)
                due = cached is None or (frame_index - int(cached.get("frame", -999))) >= every
                if due:
                    info = self._compute(track, frame_bgr, use_mask, palette)
                    if info is None and cached is not None:
                        info = cached
                    elif info is not None:
                        info = dict(info)
                        info["frame"] = frame_index
                        self._cache[track.track_id] = info
                else:
                    info = cached
                used_ids.add(track.track_id)
            else:
                cached = self._match_spatial(track)
                due = cached is None or (frame_index - int(cached.get("frame", -999))) >= every
                if due:
                    info = self._compute(track, frame_bgr, use_mask, palette)
                    if info is not None:
                        info = dict(info)
                        info["frame"] = frame_index
                        info["bbox"] = np.asarray(track.bbox, dtype=np.float32).copy()
                        info["class_name"] = track.class_name
                        refreshed_spatial.append(info)
                else:
                    info = cached
                    refreshed_spatial.append(cached)
            self._write(track, info)
        if tracking:
            stale = [key for key in self._cache if key not in used_ids]
            for key in stale:
                self._cache.pop(key, None)
        else:
            self._spatial = refreshed_spatial[-80:]

    def _compute(self, track, frame_bgr, use_mask: bool, palette: bool):
        mask = track.mask_roi if use_mask else None
        pixels = sample_object_pixels(frame_bgr, track.bbox, mask)
        return measure_color(pixels, self.classifier, palette)

    def _match_spatial(self, track) -> dict | None:
        best = None
        best_iou = 0.35
        box = np.asarray(track.bbox, dtype=np.float32)
        for item in self._spatial:
            if item.get("class_name") != track.class_name:
                continue
            score = _iou(box, item["bbox"])
            if score > best_iou:
                best_iou = score
                best = item
        return best

    @staticmethod
    def _write(track, info) -> None:
        if not info:
            track.color_name = ""
            track.rgb = None
            track.hsv = None
            track.lab = None
            track.palette = []
            return
        track.color_name = info.get("name") or ""
        track.rgb = info.get("rgb")
        track.hsv = info.get("hsv")
        track.lab = info.get("lab")
        track.palette = list(info.get("palette") or [])


def _iou(a, b) -> float:
    tlx = max(float(a[0]), float(b[0]))
    tly = max(float(a[1]), float(b[1]))
    brx = min(float(a[2]), float(b[2]))
    bry = min(float(a[3]), float(b[3]))
    inter = max(0.0, brx - tlx) * max(0.0, bry - tly)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union
