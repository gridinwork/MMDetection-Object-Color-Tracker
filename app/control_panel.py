"""Detection modes, class filter and debug readout."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QWidget):
    modes_changed = Signal()
    confidence_changed = Signal(float)
    preset_clicked = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        modes = QGroupBox("DETECTION MODES")
        modes_layout = QVBoxLayout(modes)
        self.boxes: dict[str, QCheckBox] = {}
        for key, title in (
            ("detection", "Object Detection"),
            ("segmentation", "Instance Segmentation"),
            ("tracking", "Object Tracking"),
            ("color", "Color Analysis"),
            ("trails", "Motion Trails"),
            ("labels", "Labels"),
            ("confidence", "Confidence"),
            ("object_id", "Object ID"),
            ("debug", "Debug"),
            ("detailed_color", "Detailed Color Info"),
            ("palette", "Show Color Palette"),
        ):
            box = QCheckBox(title)
            box.toggled.connect(lambda _checked=False: self.modes_changed.emit())
            self.boxes[key] = box
            modes_layout.addWidget(box)
        root.addWidget(modes)

        presets = QGroupBox("PRESETS")
        preset_row = QHBoxLayout(presets)
        for name in ("DETECTION", "SEGMENTATION", "TRACKING", "COLOR", "ALL"):
            button = QPushButton(name)
            button.clicked.connect(lambda _checked=False, value=name: self.apply_preset(value))
            preset_row.addWidget(button)
        root.addWidget(presets)

        confidence_box = QGroupBox("DETECTION CONFIDENCE")
        confidence_layout = QVBoxLayout(confidence_box)
        self.confidence_label = QLabel("0.40")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(40)
        self.confidence_slider.valueChanged.connect(self._on_confidence)
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        root.addWidget(confidence_box)

        detected = QGroupBox("DETECTED CLASSES")
        detected_layout = QVBoxLayout(detected)
        self.detected_label = QLabel("—")
        self.detected_label.setWordWrap(True)
        detected_layout.addWidget(self.detected_label)
        root.addWidget(detected)

        classes = QGroupBox("CLASS FILTER")
        classes_layout = QVBoxLayout(classes)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search Class")
        self.search.textChanged.connect(self._filter_classes)
        classes_layout.addWidget(self.search)
        row = QHBoxLayout()
        all_button = QPushButton("ALL")
        none_button = QPushButton("NONE")
        all_button.clicked.connect(self.select_all_classes)
        none_button.clicked.connect(self.select_no_classes)
        row.addWidget(all_button)
        row.addWidget(none_button)
        classes_layout.addLayout(row)
        self.class_host = QWidget()
        self.class_layout = QVBoxLayout(self.class_host)
        self.class_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.class_host)
        scroll.setMinimumHeight(180)
        classes_layout.addWidget(scroll)
        root.addWidget(classes)
        self._class_boxes: list[QCheckBox] = []

        debug = QGroupBox("DEBUG")
        debug_layout = QVBoxLayout(debug)
        self.debug_label = QLabel("Debug is off")
        self.debug_label.setWordWrap(True)
        self.debug_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        debug_layout.addWidget(self.debug_label)
        root.addWidget(debug)
        root.addStretch(1)

    def set_modes(self, modes: dict) -> None:
        for key, box in self.boxes.items():
            if key in modes:
                box.blockSignals(True)
                box.setChecked(bool(modes[key]))
                box.blockSignals(False)

    def mode_values(self) -> dict:
        return {key: box.isChecked() for key, box in self.boxes.items()}

    def set_confidence(self, value: float) -> None:
        self.confidence_slider.blockSignals(True)
        self.confidence_slider.setValue(int(round(value * 100)))
        self.confidence_slider.blockSignals(False)
        self.confidence_label.setText(f"{value:.2f}")

    def _on_confidence(self, raw: int) -> None:
        value = raw / 100.0
        self.confidence_label.setText(f"{value:.2f}")
        self.confidence_changed.emit(value)

    def apply_preset(self, name: str) -> None:
        presets = {
            "DETECTION": {"detection": True, "labels": True, "confidence": True},
            "SEGMENTATION": {"detection": True, "segmentation": True, "labels": True, "confidence": True},
            "TRACKING": {
                "detection": True,
                "tracking": True,
                "labels": True,
                "confidence": True,
                "object_id": True,
            },
            "COLOR": {"detection": True, "color": True, "labels": True, "confidence": True},
            "ALL": {
                "detection": True,
                "segmentation": True,
                "tracking": True,
                "color": True,
                "labels": True,
                "confidence": True,
                "object_id": True,
            },
        }
        chosen = presets[name]
        keys = (
            "detection",
            "segmentation",
            "tracking",
            "color",
            "trails",
            "labels",
            "confidence",
            "object_id",
            "debug",
            "detailed_color",
            "palette",
        )
        for key in keys:
            box = self.boxes[key]
            box.blockSignals(True)
            box.setChecked(bool(chosen.get(key, False)))
            box.blockSignals(False)
        self.preset_clicked.emit(name)
        self.modes_changed.emit()

    def set_class_names(self, names: list[str], enabled: set[str] | None) -> None:
        while self.class_layout.count():
            item = self.class_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._class_boxes = []
        allow = set(enabled) if enabled is not None else set(names)
        for name in names:
            box = QCheckBox(name)
            box.setChecked(name in allow)
            box.toggled.connect(lambda _checked=False: self.modes_changed.emit())
            self.class_layout.addWidget(box)
            self._class_boxes.append(box)
        self._filter_classes(self.search.text())

    def class_filter_state(self) -> tuple[bool, set[str]]:
        if not self._class_boxes:
            return True, set()
        enabled = {box.text() for box in self._class_boxes if box.isChecked()}
        all_on = len(enabled) == len(self._class_boxes)
        return all_on, enabled

    def select_all_classes(self) -> None:
        for box in self._class_boxes:
            box.blockSignals(True)
            box.setChecked(True)
            box.blockSignals(False)
        self.modes_changed.emit()

    def select_no_classes(self) -> None:
        for box in self._class_boxes:
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
        self.modes_changed.emit()

    def _filter_classes(self, text: str) -> None:
        query = text.strip().lower()
        for box in self._class_boxes:
            box.setVisible(query in box.text().lower())

    def set_detected(self, counts: dict) -> None:
        if not counts:
            self.detected_label.setText("—")
            return
        lines = [f"{name}  ×{count}" for name, count in sorted(counts.items())]
        self.detected_label.setText("\n".join(lines))

    def set_debug(self, text: str) -> None:
        self.debug_label.setText(text or "Debug is off")
