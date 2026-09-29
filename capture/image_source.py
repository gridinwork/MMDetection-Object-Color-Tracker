"""Still image loader."""

from __future__ import annotations

from pathlib import Path

import cv2


class ImageSource:
    @staticmethod
    def load(path: str):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")
        return image

    @staticmethod
    def supported(path: str) -> bool:
        return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
