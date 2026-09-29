"""Details for the object selected or locked in the live view."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class ObjectInfoPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("OBJECT INFO")
        self._labels: dict[str, QLabel] = {}
        form = QFormLayout()
        for key, title in (
            ("id", "ID"),
            ("class", "Class"),
            ("score", "Confidence"),
            ("box", "Bounding Box"),
            ("color", "Dominant Color"),
            ("rgb", "RGB"),
            ("hsv", "HSV"),
            ("lab", "LAB"),
            ("palette", "Palette"),
            ("tracked", "Tracked"),
            ("age", "Age"),
            ("motion", "Motion"),
            ("velocity", "Velocity"),
        ):
            label = QLabel("—")
            label.setWordWrap(True)
            self._labels[key] = label
            form.addRow(title, label)
        self.lock_button = QPushButton("LOCK OBJECT")
        self.lock_button.setObjectName("accent")
        self.save_object_button = QPushButton("SAVE OBJECT")
        self.save_mask_button = QPushButton("SAVE MASKED OBJECT")
        buttons = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self.lock_button)
        buttons.addLayout(row)
        buttons.addWidget(self.save_object_button)
        buttons.addWidget(self.save_mask_button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.show_track(None)

    def show_track(self, track: dict | None, locked: bool = False) -> None:
        self.lock_button.setText("UNLOCK" if locked else "LOCK OBJECT")
        enabled = track is not None
        self.lock_button.setEnabled(enabled or locked)
        self.save_object_button.setEnabled(enabled)
        self.save_mask_button.setEnabled(enabled and bool(track and track.get("mask_roi") is not None))
        if track is None:
            for label in self._labels.values():
                label.setText("—")
            return
        x, y, w, h = track["bbox"]
        self._labels["id"].setText(str(track["id"]))
        self._labels["class"].setText(str(track["class_name"]))
        self._labels["score"].setText(f"{track['score']:.2f}")
        self._labels["box"].setText(f"x {x:.0f}   y {y:.0f}\nwidth {w:.0f}   height {h:.0f}")
        self._labels["color"].setText(track.get("color") or "—")
        self._labels["rgb"].setText(_fmt_tuple(track.get("rgb")))
        self._labels["hsv"].setText(_fmt_tuple(track.get("hsv")))
        self._labels["lab"].setText(_fmt_tuple(track.get("lab")))
        palette = track.get("palette") or []
        if palette:
            self._labels["palette"].setText(
                "\n".join(f"{item['name']} {int(round(item['ratio'] * 100))}%" for item in palette)
            )
        else:
            self._labels["palette"].setText("—")
        self._labels["tracked"].setText("YES" if track.get("tracked") else "NO")
        self._labels["age"].setText(f"{track.get('age', 0)} frames")
        self._labels["motion"].setText(track.get("motion") or "—")
        self._labels["velocity"].setText(
            f"vx {track.get('velocity_x', 0):.1f}   vy {track.get('velocity_y', 0):.1f}"
        )


def _fmt_tuple(value) -> str:
    if not value:
        return "—"
    return ", ".join(str(int(part)) for part in value)
