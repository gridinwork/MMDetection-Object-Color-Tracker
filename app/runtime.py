"""Shared runtime options read by the inference thread."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    source: str = "webcam"
    camera_index: int = 0
    resolution: str = "1280x720"
    device: str = "auto"
    model: str = "auto"
    optimization: str = "balanced"
    confidence: float = 0.40
    processing_scale: float = 0.75
    detection: bool = True
    segmentation: bool = False
    tracking: bool = True
    color: bool = True
    trails: bool = False
    labels: bool = True
    show_confidence: bool = True
    show_id: bool = True
    debug: bool = False
    detailed_color: bool = False
    palette: bool = False
    class_filter_all: bool = True
    enabled_classes: set = field(default_factory=set)
    lost_frames_timeout: int = 20
    trail_length: int = 30
    color_every: int = 5
    max_objects: int = 40
    mirror: bool = True
    locked_id: int | None = None


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cfg = RuntimeConfig()
        self.capture_fps = 0.0
        self.display_fps = 0.0
        self.dropped = 0

    def snapshot(self) -> RuntimeConfig:
        with self._lock:
            return copy.deepcopy(self.cfg)

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self.cfg, key, value)

    def set_rates(self, capture_fps: float | None = None, display_fps: float | None = None, dropped: int | None = None) -> None:
        with self._lock:
            if capture_fps is not None:
                self.capture_fps = float(capture_fps)
            if display_fps is not None:
                self.display_fps = float(display_fps)
            if dropped is not None:
                self.dropped = int(dropped)

    def rates(self) -> dict:
        with self._lock:
            return {
                "capture_fps": self.capture_fps,
                "display_fps": self.display_fps,
                "dropped": self.dropped,
            }
