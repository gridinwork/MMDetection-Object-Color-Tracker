"""Track identity, motion and a short trajectory."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


class Trajectory:
    def __init__(self, maxlen: int = 30) -> None:
        self.maxlen = maxlen
        self.points: deque[tuple[float, float]] = deque(maxlen=maxlen)

    def add(self, x: float, y: float) -> None:
        self.points.append((float(x), float(y)))

    def resize(self, maxlen: int) -> None:
        maxlen = max(2, int(maxlen))
        if maxlen == self.maxlen:
            return
        self.maxlen = maxlen
        self.points = deque(self.points, maxlen=maxlen)


@dataclass
class Detection:
    class_id: int
    class_name: str
    score: float
    bbox: np.ndarray
    mask_roi: np.ndarray | None = None


@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    score: float
    bbox: np.ndarray
    mask_roi: np.ndarray | None = None
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    motion: str = "STATIC"
    trajectory: Trajectory = field(default_factory=Trajectory)
    cx: float = 0.0
    cy: float = 0.0
    color_name: str = ""
    rgb: tuple | None = None
    hsv: tuple | None = None
    lab: tuple | None = None
    palette: list = field(default_factory=list)
    tracked: bool = True

    def center(self) -> tuple[float, float]:
        return float(self.cx), float(self.cy)
