"""Video file playback with seek."""

from __future__ import annotations

import cv2


class VideoFileCapture:
    def __init__(self) -> None:
        self.cap = None
        self.path = ""

    def open(self, path: str) -> None:
        self.close()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {path}")
        self.cap = cap
        self.path = path

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    @property
    def fps(self) -> float:
        if self.cap is None:
            return 25.0
        value = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        return value if value > 1 else 25.0

    @property
    def frame_count(self) -> int:
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def position(self) -> int:
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

    def seek_ratio(self, ratio: float) -> None:
        if self.cap is None:
            return
        total = self.frame_count
        if total <= 0:
            return
        frame = int(max(0.0, min(0.999, ratio)) * total)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
