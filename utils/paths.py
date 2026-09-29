"""Project paths and directory setup."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIR_NAMES = (
    "models",
    "checkpoints",
    "screenshots",
    "recordings",
    "objects",
    "datasets",
    "training",
    "custom_models",
    "custom_models/configs",
    "logs",
    "assets",
)


def ensure_dirs() -> None:
    for name in DIR_NAMES:
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
