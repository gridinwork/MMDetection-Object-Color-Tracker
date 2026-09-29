"""USB camera open, scan and read."""

from __future__ import annotations

import cv2


class CameraCapture:
    def __init__(self) -> None:
        self.cap = None
        self.index = 0

    @staticmethod
    def scan(max_index: int = 6) -> list[int]:
        found = []
        for index in range(max_index):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            opened = cap.isOpened()
            ok = False
            if opened:
                ok, _frame = cap.read()
            cap.release()
            if opened and ok:
                found.append(index)
        return found

    def open(self, index: int, width: int, height: int) -> None:
        self.close()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Camera {index} is not available")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.index = index

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
