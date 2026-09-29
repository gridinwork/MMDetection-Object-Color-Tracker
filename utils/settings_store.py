"""Persist GUI settings in config/settings.json."""

from __future__ import annotations

import json
from copy import deepcopy

from utils.paths import project_path

DEFAULT_SETTINGS = {
    "source": "webcam",
    "camera_index": 0,
    "resolution": "1280x720",
    "device": "auto",
    "model": "auto",
    "optimization": "balanced",
    "confidence": 0.40,
    "processing_scale": 0.75,
    "modes": {
        "detection": True,
        "segmentation": False,
        "tracking": True,
        "color": True,
        "trails": False,
        "labels": True,
        "confidence": True,
        "object_id": True,
        "debug": False,
        "detailed_color": False,
        "palette": False,
    },
    "class_filter_all": True,
    "enabled_classes": [],
    "lost_frames_timeout": 20,
    "trail_length": 30,
    "color_every": 5,
    "max_objects": 40,
    "mirror": True,
    "window": {"x": 60, "y": 40, "w": 1480, "h": 900},
    "video_path": "",
    "image_path": "",
}


def load_settings() -> dict:
    path = project_path("config", "settings.json")
    settings = deepcopy(DEFAULT_SETTINGS)
    if not path.is_file():
        return settings
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings
    if not isinstance(stored, dict):
        return settings
    settings.update({key: stored[key] for key in stored if key in settings and key != "modes"})
    if isinstance(stored.get("modes"), dict):
        settings["modes"].update(stored["modes"])
    if isinstance(stored.get("window"), dict):
        settings["window"].update(stored["window"])
    return settings


def save_settings(settings: dict) -> None:
    path = project_path("config", "settings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
