"""Map measured LAB/HSV color to a readable name."""

from __future__ import annotations

import json
from functools import lru_cache

import cv2
import numpy as np

from utils.paths import project_path


@lru_cache(maxsize=1)
def _config() -> dict:
    path = project_path("color", "color_ranges.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _bgr_to_lab_pixel(rgb: tuple[int, int, int]) -> np.ndarray:
    red, green, blue = rgb
    pixel = np.uint8([[[blue, green, red]]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return lab[0, 0].astype(np.float32)


@lru_cache(maxsize=1)
def _prototypes() -> list[tuple[str, np.ndarray]]:
    items = []
    for entry in _config()["categories"]:
        items.append((entry["name"], _bgr_to_lab_pixel(tuple(entry["rgb"]))))
    return items


class ColorClassifier:
    def classify_lab_hsv(self, lab: np.ndarray, hsv: np.ndarray) -> str:
        neutral = _config()["neutral"]
        _hue, saturation, value = (float(hsv[0]), float(hsv[1]), float(hsv[2]))
        if value <= float(neutral["black_v_max"]):
            return "BLACK"
        if saturation <= float(neutral["white_s_max"]) and value >= float(neutral["white_v_min"]):
            return "WHITE"
        if saturation <= float(neutral["gray_s_max"]):
            return "GRAY"
        sample = np.asarray(lab, dtype=np.float32)
        best_name = "GRAY"
        best_dist = 1e9
        for name, proto in _prototypes():
            dist = float(np.linalg.norm(sample - proto))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def classify_bgr_pixel(self, bgr: np.ndarray) -> str:
        pixel = np.uint8([[bgr]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
        lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
        return self.classify_lab_hsv(lab, hsv)
