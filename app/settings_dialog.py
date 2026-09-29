"""Tracker limits saved with the rest of the settings."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QSpinBox


class SettingsDialog(QDialog):
    def __init__(
        self,
        lost_frames: int,
        trail_length: int,
        color_every: int,
        max_objects: int,
        mirror: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(360)
        self.lost_frames = QSpinBox()
        self.lost_frames.setRange(1, 120)
        self.lost_frames.setValue(int(lost_frames))
        self.trail_length = QSpinBox()
        self.trail_length.setRange(5, 120)
        self.trail_length.setValue(int(trail_length))
        self.color_every = QSpinBox()
        self.color_every.setRange(1, 30)
        self.color_every.setValue(int(color_every))
        self.max_objects = QSpinBox()
        self.max_objects.setRange(1, 100)
        self.max_objects.setValue(int(max_objects))
        self.mirror = QCheckBox("Mirror left to right")
        self.mirror.setChecked(bool(mirror))
        form = QFormLayout(self)
        form.addRow("lost_frames_timeout", self.lost_frames)
        form.addRow("Trail length", self.trail_length)
        form.addRow("Color update, frames", self.color_every)
        form.addRow("Max objects", self.max_objects)
        form.addRow(self.mirror)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict:
        return {
            "lost_frames_timeout": int(self.lost_frames.value()),
            "trail_length": int(self.trail_length.value()),
            "color_every": int(self.color_every.value()),
            "max_objects": int(self.max_objects.value()),
            "mirror": bool(self.mirror.isChecked()),
        }
