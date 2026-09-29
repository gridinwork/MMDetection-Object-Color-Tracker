"""Main window: source bar, live view and the right-hand controls."""

from __future__ import annotations

import math
import re
import time
from datetime import datetime

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.control_panel import ControlPanel
from app.model_manager_dialog import ModelManagerDialog
from app.object_info_panel import ObjectInfoPanel
from app.runtime import SharedState
from app.settings_dialog import SettingsDialog
from app.video_widget import VideoWidget
from app.workers import CaptureWorker, FpsMeter, InferenceWorker, LatestFrameSlot
from capture.image_source import ImageSource
from inference.model_registry import ModelRegistry
from utils.logger import get_logger
from utils.paths import project_path
from utils.settings_store import load_settings, save_settings

logger = get_logger("vision_studio.ui")

OPTIMIZATION = {
    "max_fps": (0.5, 8),
    "balanced": (0.75, 5),
    "max_quality": (1.0, 2),
}
RECORD_DELAY = 3.0
RECORD_SECONDS = 15.0
RECORD_FPS = 15


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MMDetection Object & Color Vision Studio")
        self.resize(1480, 900)
        self.shared = SharedState()
        self.settings = load_settings()
        self.registry = ModelRegistry()
        self.slot = LatestFrameSlot()
        self.capture = CaptureWorker(self.slot, self.shared)
        self.inference = InferenceWorker(self.slot, self.shared)
        self._packet = None
        self._tracks: list = []
        self._still = None
        self._image_path = self.settings.get("image_path") or ""
        self._video_path = self.settings.get("video_path") or ""
        self._selected_id = None
        self._locked_id = None
        self._slider_held = False
        self._class_names: list[str] = []
        self._last_popup = ""
        self._display_fps = FpsMeter()
        self._live = False
        self._record_phase = "idle"
        self._record_timer = QTimer(self)
        self._record_timer.setInterval(50)
        self._record_timer.timeout.connect(self._record_tick)
        self._record_writer = None
        self._record_path = None
        self._record_size = (0, 0)
        self._record_frames = 0
        self._record_t0 = 0.0
        self._record_write_at = 0.0
        self._record_end_at = 0.0
        self._record_next = 0.0
        self._countdown_left = 0
        self._build()
        self._connect()
        self._restore()
        self.capture.start()
        self.inference.start()
        self.capture.enqueue("scan")
        self.statusBar().showMessage("Scanning cameras...")

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addLayout(self._build_toolbar())
        body = QHBoxLayout()
        self.video = VideoWidget()
        body.addWidget(self.video, 1)
        self.controls = ControlPanel()
        self.info = ObjectInfoPanel()
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(self.controls)
        side_layout.addWidget(self.info)
        tools = QHBoxLayout()
        self.model_button = QPushButton("MODEL MANAGER")
        self.settings_button = QPushButton("SETTINGS")
        tools.addWidget(self.model_button)
        tools.addWidget(self.settings_button)
        side_layout.addLayout(tools)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setFixedWidth(390)
        body.addWidget(scroll)
        root.addLayout(body, 1)
        status = QStatusBar()
        self.setStatusBar(status)
        self.fps_label = QLabel("CAMERA 0 | INFERENCE 0 | DISPLAY 0 | DROPPED 0")
        self.model_label = QLabel("Model: —")
        self.device_label = QLabel("Device: —")
        status.addWidget(self.fps_label, 1)
        status.addPermanentWidget(self.model_label)
        status.addPermanentWidget(self.device_label)

    def _build_toolbar(self) -> QGridLayout:
        grid = QGridLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItem("Webcam", "webcam")
        self.source_combo.addItem("Video File", "video")
        self.source_combo.addItem("Image", "image")
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Camera 0", 0)
        self.resolution_combo = QComboBox()
        for value in ("640x480", "1280x720", "1920x1080"):
            self.resolution_combo.addItem(value)
        self.device_combo = QComboBox()
        self.device_combo.addItem("AUTO", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.model_combo = QComboBox()
        self.model_combo.addItem("Auto", "auto")
        self.model_combo.addItem("Fast", "fast")
        self.model_combo.addItem("Balanced", "balanced")
        self.model_combo.addItem("Accurate", "accurate")
        for spec in self.registry.all():
            self.model_combo.addItem(spec.display_name, spec.model_id)
        self.opt_combo = QComboBox()
        self.opt_combo.addItem("MAX FPS", "max_fps")
        self.opt_combo.addItem("BALANCED", "balanced")
        self.opt_combo.addItem("MAX QUALITY", "max_quality")
        self.scale_combo = QComboBox()
        self.scale_combo.addItem("100%", 1.0)
        self.scale_combo.addItem("75%", 0.75)
        self.scale_combo.addItem("50%", 0.5)
        self.start_button = QPushButton("START")
        self.start_button.setObjectName("start")
        self.record_button = QPushButton(f"RECORD {int(RECORD_SECONDS)}s")
        self.record_button.setObjectName("record")
        self.record_clock = QLabel(_clock_hms(RECORD_SECONDS))
        self.record_clock.setObjectName("recordTimer")
        self.record_clock.setAlignment(Qt.AlignCenter)
        self.stop_button = QPushButton("STOP")
        self.stop_button.setObjectName("stop")
        self.pause_button = QPushButton("PAUSE")
        self.pause_button.setCheckable(True)
        self.shot_button = QPushButton("SAVE SCREENSHOT")
        self.clean_button = QPushButton("SAVE CLEAN FRAME")
        self.browse_button = QPushButton("BROWSE")
        self.refresh_button = QPushButton("REFRESH CAMERAS")
        self.path_label = QLabel("No file selected")
        self.path_label.setObjectName("muted")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        self.time_label = QLabel("00:00 / 00:00")
        self.note_label = QLabel("")
        self.note_label.setObjectName("muted")

        grid.addWidget(QLabel("SOURCE"), 0, 0)
        grid.addWidget(self.source_combo, 0, 1)
        grid.addWidget(QLabel("Camera"), 0, 2)
        grid.addWidget(self.camera_combo, 0, 3)
        grid.addWidget(QLabel("Resolution"), 0, 4)
        grid.addWidget(self.resolution_combo, 0, 5)
        grid.addWidget(QLabel("Device"), 0, 6)
        grid.addWidget(self.device_combo, 0, 7)
        grid.addWidget(QLabel("Model"), 1, 0)
        grid.addWidget(self.model_combo, 1, 1, 1, 2)
        grid.addWidget(QLabel("Optimization"), 1, 3)
        grid.addWidget(self.opt_combo, 1, 4)
        grid.addWidget(QLabel("Processing Scale"), 1, 5)
        grid.addWidget(self.scale_combo, 1, 6)
        buttons = QHBoxLayout()
        for widget in (
            self.start_button,
            self.record_button,
            self.record_clock,
            self.stop_button,
            self.pause_button,
            self.shot_button,
            self.clean_button,
            self.browse_button,
            self.refresh_button,
        ):
            buttons.addWidget(widget)
        grid.addLayout(buttons, 2, 0, 1, 8)
        grid.addWidget(self.path_label, 3, 0, 1, 5)
        grid.addWidget(self.slider, 3, 5, 1, 2)
        grid.addWidget(self.time_label, 3, 7)
        grid.addWidget(self.note_label, 4, 0, 1, 8)
        self._camera_widgets = (self.camera_combo, self.resolution_combo, self.refresh_button)
        return grid

    def _connect(self) -> None:
        self.start_button.clicked.connect(self.start_source)
        self.record_button.clicked.connect(self._toggle_record)
        self.stop_button.clicked.connect(self.stop_source)
        self.pause_button.toggled.connect(self._pause)
        self.shot_button.clicked.connect(self.save_screenshot)
        self.clean_button.clicked.connect(self.save_clean_frame)
        self.browse_button.clicked.connect(self._browse)
        self.refresh_button.clicked.connect(lambda: self.capture.enqueue("scan"))
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.opt_combo.currentIndexChanged.connect(self._optimization_changed)
        self.scale_combo.currentIndexChanged.connect(self._live_option_changed)
        self.model_combo.currentIndexChanged.connect(self._live_option_changed)
        self.device_combo.currentIndexChanged.connect(self._live_option_changed)
        self.resolution_combo.currentIndexChanged.connect(self._live_option_changed)
        self.camera_combo.currentIndexChanged.connect(self._live_option_changed)
        self.controls.modes_changed.connect(self._live_option_changed)
        self.controls.confidence_changed.connect(self._confidence_changed)
        self.controls.confidence_slider.sliderReleased.connect(self._maybe_rerun_still)
        self.video.clicked.connect(self._on_click)
        self.info.lock_button.clicked.connect(self._toggle_lock)
        self.info.save_object_button.clicked.connect(lambda: self.save_object(False))
        self.info.save_mask_button.clicked.connect(lambda: self.save_object(True))
        self.model_button.clicked.connect(self._open_models)
        self.settings_button.clicked.connect(self._open_settings)
        self.slider.sliderPressed.connect(self._slider_pressed)
        self.slider.sliderReleased.connect(self._slider_released)
        self.capture.cameras_found.connect(self._on_cameras)
        self.capture.source_failed.connect(self._on_error)
        self.capture.video_position.connect(self._on_video_position)
        self.capture.video_ended.connect(self._on_video_ended)
        self.inference.packet_ready.connect(self._on_packet)
        self.inference.error.connect(self._on_error)
        self.inference.classes_ready.connect(self._on_classes)
        self.inference.devices_ready.connect(self._on_devices)
        self.inference.status.connect(self._on_status)

    def _restore(self) -> None:
        settings = self.settings
        self._select(self.source_combo, settings.get("source"))
        self._select(self.resolution_combo, settings.get("resolution"), by_text=True)
        self._select(self.device_combo, settings.get("device"))
        self._select(self.model_combo, settings.get("model"))
        self._select(self.opt_combo, settings.get("optimization"))
        self._select(self.scale_combo, settings.get("processing_scale"))
        self.controls.set_modes(settings.get("modes") or {})
        self.controls.set_confidence(float(settings.get("confidence", 0.4)))
        window = settings.get("window") or {}
        self.setGeometry(int(window.get("x", 60)), int(window.get("y", 40)), int(window.get("w", 1480)), int(window.get("h", 900)))
        if self._video_path:
            self.path_label.setText(self._video_path)
        if settings.get("source") == "image" and self._image_path:
            self.path_label.setText(self._image_path)
        self._apply_source_widgets()
        self.shared.update(
            lost_frames_timeout=int(settings.get("lost_frames_timeout", 20)),
            trail_length=int(settings.get("trail_length", 30)),
            color_every=int(settings.get("color_every", 5)),
            max_objects=int(settings.get("max_objects", 40)),
            mirror=bool(settings.get("mirror", True)),
        )
        self._sync_config()

    def _select(self, combo: QComboBox, value, by_text: bool = False) -> None:
        combo.blockSignals(True)
        if by_text:
            index = combo.findText(str(value))
        else:
            index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _sync_config(self) -> None:
        modes = self.controls.mode_values()
        all_classes, enabled = self.controls.class_filter_state()
        if not self._class_names:
            all_classes = bool(self.settings.get("class_filter_all", True))
            enabled = set(self.settings.get("enabled_classes") or [])
        self.shared.update(
            source=self.source_combo.currentData(),
            camera_index=int(self.camera_combo.currentData() or 0),
            resolution=self.resolution_combo.currentText(),
            device=self.device_combo.currentData() or "auto",
            model=self.model_combo.currentData() or "auto",
            optimization=self.opt_combo.currentData() or "balanced",
            confidence=self.controls.confidence_slider.value() / 100.0,
            processing_scale=float(self.scale_combo.currentData() or 1.0),
            detection=modes["detection"],
            segmentation=modes["segmentation"],
            tracking=modes["tracking"],
            color=modes["color"],
            trails=modes["trails"],
            labels=modes["labels"],
            show_confidence=modes["confidence"],
            show_id=modes["object_id"],
            debug=modes["debug"],
            detailed_color=modes["detailed_color"],
            palette=modes["palette"],
            class_filter_all=all_classes,
            enabled_classes=enabled,
            locked_id=self._locked_id,
        )

    def _live_option_changed(self, *_args) -> None:
        self._sync_config()
        self._maybe_rerun_still()

    def _confidence_changed(self, value: float) -> None:
        self.shared.update(confidence=float(value))

    def _optimization_changed(self, *_args) -> None:
        key = self.opt_combo.currentData()
        scale, every = OPTIMIZATION.get(key, (0.75, 5))
        self.scale_combo.blockSignals(True)
        index = self.scale_combo.findData(scale)
        if index >= 0:
            self.scale_combo.setCurrentIndex(index)
        self.scale_combo.blockSignals(False)
        self.shared.update(color_every=every)
        self._sync_config()
        self._maybe_rerun_still()

    def _source_changed(self, *_args) -> None:
        self.stop_source()
        self._apply_source_widgets()
        kind = self.source_combo.currentData()
        if kind == "video" and self._video_path:
            self.path_label.setText(self._video_path)
        elif kind == "image" and self._image_path:
            self.path_label.setText(self._image_path)
        else:
            self.path_label.setText("No file selected")
        self._sync_config()

    def _apply_source_widgets(self) -> None:
        kind = self.source_combo.currentData()
        webcam = kind == "webcam"
        video = kind == "video"
        for widget in (self.camera_combo, self.resolution_combo, self.refresh_button):
            widget.setVisible(webcam)
        self.slider.setEnabled(video)
        self.pause_button.setEnabled(kind != "image")
        self.browse_button.setVisible(kind in ("video", "image"))

    def start_source(self, *_args) -> None:
        self._sync_config()
        kind = self.source_combo.currentData()
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(False)
        self.pause_button.blockSignals(False)
        if kind == "webcam":
            width, height = _parse_resolution(self.resolution_combo.currentText())
            index = int(self.camera_combo.currentData() or 0)
            self.capture.enqueue("start_camera", index=index, width=width, height=height)
            self._live = True
            self.statusBar().showMessage(f"Starting camera {index}")
        elif kind == "video":
            if not self._video_path:
                self._browse()
            if not self._video_path:
                return
            self.capture.enqueue("start_video", path=self._video_path)
            self._live = True
            self.statusBar().showMessage("Playing video")
        else:
            if self._still is None and self._image_path:
                self.open_image(self._image_path)
                return
            if self._still is None:
                self._browse()
                return
            self.inference.submit_image(self._still, True)
            self.statusBar().showMessage("Analyzing image")

    def stop_source(self, *_args) -> None:
        self.capture.enqueue("stop")
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(False)
        self.pause_button.blockSignals(False)
        self._live = False
        if self._record_phase != "idle":
            self._finish_recording(cancelled=True)
        else:
            self.statusBar().showMessage("Stopped")

    def _pause(self, checked: bool) -> None:
        self.capture.enqueue("pause", paused=checked)

    def _browse(self, *_args) -> None:
        kind = self.source_combo.currentData()
        if kind == "video":
            path, _selected = QFileDialog.getOpenFileName(
                self, "Video File", self._video_path, "Video (*.mp4 *.avi *.mov *.mkv)"
            )
            if path:
                self._video_path = path
                self.path_label.setText(path)
                self.start_source()
        elif kind == "image":
            path, _selected = QFileDialog.getOpenFileName(
                self, "Image", self._image_path, "Image (*.jpg *.jpeg *.png *.webp)"
            )
            if path:
                self.open_image(path)

    def open_image(self, path: str) -> None:
        try:
            image = ImageSource.load(path)
        except RuntimeError as exc:
            self._on_error(str(exc))
            return
        self.stop_source()
        self._image_path = path
        self._still = image
        self.path_label.setText(path)
        self._select(self.source_combo, "image")
        self._apply_source_widgets()
        self._sync_config()
        self.inference.submit_image(image, True)

    def _maybe_rerun_still(self, *_args) -> None:
        if self.source_combo.currentData() == "image" and self._still is not None:
            self.inference.submit_image(self._still, True)

    def _on_cameras(self, found: list) -> None:
        preferred = int(self.settings.get("camera_index", 0))
        current = self.camera_combo.currentData()
        if current is not None:
            preferred = int(current)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        indexes = list(found) if found else [0]
        for index in indexes:
            self.camera_combo.addItem(f"Camera {index}", int(index))
        index = self.camera_combo.findData(preferred)
        self.camera_combo.setCurrentIndex(index if index >= 0 else 0)
        self.camera_combo.blockSignals(False)
        if found:
            self.statusBar().showMessage("Cameras: " + ", ".join(f"Camera {i}" for i in found))
        else:
            self.statusBar().showMessage("No camera responded. Camera 0 is still listed.")

    def _on_devices(self, devices: list) -> None:
        current = self.device_combo.currentData() or self.settings.get("device") or "auto"
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("AUTO", "auto")
        for device in devices:
            self.device_combo.addItem(
                f"CUDA:{device['index']}  {device['name']}",
                f"cuda:{device['index']}",
            )
        self.device_combo.addItem("CPU", "cpu")
        index = self.device_combo.findData(current)
        self.device_combo.setCurrentIndex(index if index >= 0 else 0)
        self.device_combo.blockSignals(False)
        self._sync_config()

    def _on_classes(self, names: list) -> None:
        if list(names) == self._class_names:
            return
        self._class_names = list(names)
        if self.settings.get("class_filter_all", True):
            enabled = set(names)
        else:
            enabled = set(self.settings.get("enabled_classes") or [])
        self.controls.set_class_names(self._class_names, enabled)
        self._sync_config()

    def _on_packet(self, *_args) -> None:
        packet = self.inference.latest_packet()
        if packet is None:
            return
        self._packet = packet
        self._tracks = list(packet.tracks)
        self.video.set_bgr(self._preview_frame(packet.display_bgr))
        self.controls.set_detected(packet.detected_counts)
        display_fps = self._display_fps.tick()
        self.shared.set_rates(display_fps=display_fps)
        self.fps_label.setText(
            f"CAMERA {packet.capture_fps:.1f} | INFERENCE {packet.inference_fps:.1f} | "
            f"DISPLAY {display_fps:.1f} | DROPPED {packet.dropped}"
        )
        self.model_label.setText(f"Model: {packet.active_model or '—'}")
        self.device_label.setText(packet.device_text or "Device: —")
        self.note_label.setText(packet.model_note)
        if self.controls.boxes["debug"].isChecked():
            timing = packet.timings
            self.controls.set_debug(
                "\n".join(
                    [
                        f"FRAME {timing.get('frame', 0)}",
                        f"CAPTURE FPS {packet.capture_fps:.1f}",
                        f"INFERENCE FPS {packet.inference_fps:.1f}",
                        f"DISPLAY FPS {display_fps:.1f}",
                        f"PREPROCESS MS {timing.get('preprocess_ms', 0):.1f}",
                        f"MODEL MS {timing.get('model_ms', 0):.1f}",
                        f"TRACKER MS {timing.get('tracker_ms', 0):.1f}",
                        f"COLOR MS {timing.get('color_ms', 0):.1f}",
                        f"RENDER MS {timing.get('render_ms', 0):.1f}",
                        f"TOTAL MS {timing.get('total_ms', 0):.1f}",
                        f"QUEUE SIZE {packet.queue_size}",
                        f"DROPPED FRAMES {packet.dropped}",
                        f"GPU {packet.device_text}",
                        f"ACTIVE MODEL {packet.active_model}",
                    ]
                )
            )
        else:
            self.controls.set_debug("Debug is off")
        self._follow_selection()
        video = packet.video_meta or {}
        if video and not self._slider_held:
            total = max(1, int(video.get("total") or 1))
            index = int(video.get("index") or 0)
            self.slider.blockSignals(True)
            self.slider.setValue(int(index / total * 1000))
            self.slider.blockSignals(False)
            self.time_label.setText(f"{_clock(video.get('seconds', 0))} / {_clock(video.get('duration', 0))}")

    def _follow_selection(self) -> None:
        target = self._locked_id if self._locked_id is not None else self._selected_id
        match = None
        if target is not None:
            match = next((track for track in self._tracks if track["id"] == target), None)
        if self._locked_id is not None:
            self.info.show_track(match, locked=True)
        else:
            self.info.show_track(match, locked=False)

    def _on_click(self, x: float, y: float) -> None:
        if self._locked_id is not None:
            return
        if x < 0:
            self._selected_id = None
            self.info.show_track(None)
            return
        best = None
        best_area = 1e18
        for track in self._tracks:
            x1, y1, x2, y2 = track["xyxy"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(1.0, (x2 - x1) * (y2 - y1))
                if area < best_area:
                    best_area = area
                    best = track
        if best is None:
            self._selected_id = None
            self.info.show_track(None)
            return
        self._selected_id = best["id"]
        self.info.show_track(best, locked=False)

    def _toggle_lock(self, *_args) -> None:
        if self._locked_id is not None:
            logger.info("unlock object %s", self._locked_id)
            self._locked_id = None
            self.shared.update(locked_id=None)
            self._follow_selection()
            return
        if self._selected_id is None:
            return
        self._locked_id = self._selected_id
        self.shared.update(locked_id=self._locked_id)
        logger.info("lock object %s", self._locked_id)
        self._follow_selection()

    def _current_track(self) -> dict | None:
        target = self._locked_id if self._locked_id is not None else self._selected_id
        if target is None:
            return None
        return next((track for track in self._tracks if track["id"] == target), None)

    def save_screenshot(self, *_args) -> None:
        self._save_frame(clean=False)

    def save_clean_frame(self, *_args) -> None:
        self._save_frame(clean=True)

    def _save_frame(self, clean: bool) -> None:
        if self._packet is None:
            self.statusBar().showMessage("No frame to save")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = project_path("screenshots")
        folder.mkdir(parents=True, exist_ok=True)
        if clean:
            path = folder / f"mmdet_clean_{stamp}.jpg"
            image = self._packet.clean_bgr
        else:
            path = folder / f"mmdet_{stamp}.jpg"
            image = self._packet.display_bgr
        if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            self._on_error(f"Could not save {path.name}")
            return
        self.statusBar().showMessage(f"Saved {path}")

    def save_object(self, masked: bool) -> None:
        track = self._current_track()
        if track is None or self._packet is None:
            self.statusBar().showMessage("Select an object first")
            return
        image = self._packet.clean_bgr
        x1, y1, x2, y2 = [int(round(value)) for value in track["xyxy"]]
        height, width = image.shape[:2]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9]+", "_", track["class_name"]).strip("_") or "object"
        folder = project_path("objects")
        folder.mkdir(parents=True, exist_ok=True)
        if masked:
            mask = track.get("mask_roi")
            if mask is None:
                self.statusBar().showMessage("This object has no mask")
                return
            if mask.shape[:2] != crop.shape[:2]:
                mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = mask
            path = folder / f"{safe}_ID{track['id']}_{stamp}_mask.png"
            ok = cv2.imwrite(str(path), bgra)
        else:
            path = folder / f"{safe}_ID{track['id']}_{stamp}.png"
            ok = cv2.imwrite(str(path), crop)
        if not ok:
            self._on_error(f"Could not save {path.name}")
            return
        self.statusBar().showMessage(f"Saved {path}")

    def _open_models(self, *_args) -> None:
        dialog = ModelManagerDialog(self.registry, self)
        dialog.exec()
        self.registry.reload_custom()
        self._rebuild_model_combo()

    def _rebuild_model_combo(self) -> None:
        current = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Auto", "auto")
        self.model_combo.addItem("Fast", "fast")
        self.model_combo.addItem("Balanced", "balanced")
        self.model_combo.addItem("Accurate", "accurate")
        for spec in self.registry.all():
            self.model_combo.addItem(spec.display_name, spec.model_id)
        index = self.model_combo.findData(current)
        self.model_combo.setCurrentIndex(index if index >= 0 else 0)
        self.model_combo.blockSignals(False)

    def _open_settings(self, *_args) -> None:
        cfg = self.shared.snapshot()
        dialog = SettingsDialog(
            cfg.lost_frames_timeout,
            cfg.trail_length,
            cfg.color_every,
            cfg.max_objects,
            cfg.mirror,
            self,
        )
        if dialog.exec() != SettingsDialog.Accepted:
            return
        self.shared.update(**dialog.values())
        self._maybe_rerun_still()

    def _slider_pressed(self, *_args) -> None:
        self._slider_held = True

    def _slider_released(self, *_args) -> None:
        self._slider_held = False
        self.capture.enqueue("seek", ratio=self.slider.value() / 1000.0)

    def _on_video_position(self, index: int, total: int, fps: float) -> None:
        if self._slider_held or total <= 0:
            return
        fps = fps if fps > 1 else 25.0
        self.slider.blockSignals(True)
        self.slider.setValue(int(index / total * 1000))
        self.slider.blockSignals(False)
        self.time_label.setText(f"{_clock(index / fps)} / {_clock(total / fps)}")

    def _on_video_ended(self, *_args) -> None:
        self._live = False
        if self._record_phase != "idle":
            self._finish_recording(cancelled=True)
        self.statusBar().showMessage("Video ended")
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(False)
        self.pause_button.blockSignals(False)

    def _on_status(self, text: str) -> None:
        if text:
            self.model_label.setText(f"Model: {text}")

    def _on_error(self, message: str) -> None:
        if not message or message == self._last_popup:
            return
        self._last_popup = message
        logger.error(message)
        QMessageBox.warning(self, "MMDetection Object & Color Vision Studio", message)

    def _collect_settings(self) -> dict:
        modes = self.controls.mode_values()
        all_classes, enabled = self.controls.class_filter_state()
        if not self._class_names:
            all_classes = bool(self.settings.get("class_filter_all", True))
            enabled = list(self.settings.get("enabled_classes") or [])
        else:
            enabled = sorted(enabled)
        geometry = self.geometry()
        cfg = self.shared.snapshot()
        return {
            "source": self.source_combo.currentData(),
            "camera_index": int(self.camera_combo.currentData() or 0),
            "resolution": self.resolution_combo.currentText(),
            "device": self.device_combo.currentData() or "auto",
            "model": self.model_combo.currentData() or "auto",
            "optimization": self.opt_combo.currentData() or "balanced",
            "confidence": self.controls.confidence_slider.value() / 100.0,
            "processing_scale": float(self.scale_combo.currentData() or 1.0),
            "modes": modes,
            "class_filter_all": all_classes,
            "enabled_classes": enabled,
            "lost_frames_timeout": cfg.lost_frames_timeout,
            "trail_length": cfg.trail_length,
            "color_every": cfg.color_every,
            "max_objects": cfg.max_objects,
            "mirror": bool(cfg.mirror),
            "window": {"x": geometry.x(), "y": geometry.y(), "w": geometry.width(), "h": geometry.height()},
            "video_path": self._video_path,
            "image_path": self._image_path,
        }

    def _save_settings(self) -> None:
        try:
            save_settings(self._collect_settings())
        except OSError as exc:
            logger.error("settings save failed: %s", exc)

    def _toggle_record(self, *_args) -> None:
        if self._record_phase != "idle":
            self._finish_recording(cancelled=True)
            return
        if self.source_combo.currentData() == "image":
            self.statusBar().showMessage("Recording needs Webcam or Video File. Press START first.")
            return
        if not self._live:
            self.start_source()
        now = time.perf_counter()
        self._record_phase = "countdown"
        self._record_t0 = now
        self._record_write_at = now + RECORD_DELAY
        self._record_end_at = self._record_write_at + RECORD_SECONDS
        self._record_next = self._record_write_at
        self._record_frames = 0
        self._countdown_left = int(RECORD_DELAY)
        self._show_record_clock()
        self._record_timer.start()
        self.statusBar().showMessage(f"Step away. Recording starts in {int(RECORD_DELAY)} seconds.")
        logger.info("recording countdown %.0fs then %.0fs", RECORD_DELAY, RECORD_SECONDS)

    def _record_tick(self) -> None:
        now = time.perf_counter()
        if self._record_phase == "countdown":
            left = max(0.0, self._record_write_at - now)
            self._countdown_left = int(math.ceil(left)) if left > 0 else 0
            self._show_record_clock()
            self.statusBar().showMessage(
                f"Recording starts in {_clock_hms(left)}. Clip length {_clock_hms(RECORD_SECONDS)}."
            )
            self._refresh_record_preview()
            if now < self._record_write_at:
                return
            if self._packet is None:
                if now > self._record_write_at + 8:
                    self.statusBar().showMessage("No camera frame yet. Recording cancelled.")
                    self._reset_record_ui()
                return
            if not self._open_writer():
                self._reset_record_ui()
                return
            self._record_phase = "recording"
            self._record_next = now
            logger.info("recording started %s", self._record_path)
        if self._record_phase != "recording":
            return
        target_frames = int(RECORD_FPS * RECORD_SECONDS)
        if now >= self._record_end_at or self._record_frames >= target_frames:
            self._finish_recording(cancelled=False)
            return
        while self._record_next <= now and self._record_frames < target_frames:
            if not self._write_record_frame():
                break
            self._record_next += 1.0 / RECORD_FPS
            self._record_frames += 1
        left = max(0.0, self._record_end_at - now)
        self._show_record_clock()
        self.statusBar().showMessage(f"Recording {_clock_hms(left)} / {_clock_hms(RECORD_SECONDS)}")
        self._refresh_record_preview()

    def _refresh_record_preview(self) -> None:
        if self._packet is None or self._record_phase not in ("countdown", "recording"):
            return
        self.video.set_bgr(self._preview_frame(self._packet.display_bgr))

    def _open_writer(self) -> bool:
        frame = self._packet.display_bgr if self._packet is not None else None
        if frame is None:
            return False
        height, width = frame.shape[:2]
        width -= width % 2
        height -= height % 2
        if width < 2 or height < 2:
            return False
        folder = project_path("recordings")
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = folder / f"mmdet_rec_{stamp}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(RECORD_FPS),
            (width, height),
        )
        if not writer.isOpened():
            self.statusBar().showMessage("Could not open the video file for recording.")
            return False
        self._record_writer = writer
        self._record_path = path
        self._record_size = (width, height)
        return True

    def _write_record_frame(self) -> bool:
        if self._record_writer is None or self._packet is None:
            return False
        frame = self._packet.display_bgr
        width, height = self._record_size
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        stamped = _stamp_banner(frame, self._record_banner_text())
        self._record_writer.write(stamped)
        return True

    def _show_record_clock(self) -> None:
        if self._record_phase == "countdown":
            left = max(0.0, self._record_write_at - time.perf_counter())
            self.record_button.setText(f"CANCEL {_clock_hms(left)}")
            self.record_clock.setText(f"IN {_clock_hms(left)}")
            return
        if self._record_phase == "recording":
            left = max(0.0, self._record_end_at - time.perf_counter())
            self.record_button.setText(f"REC {_clock_hms(left)}")
            self.record_clock.setText(f"{_clock_hms(left)} / {_clock_hms(RECORD_SECONDS)}")

    def _record_banner_text(self) -> str:
        if self._record_phase == "countdown":
            left = max(0.0, self._record_write_at - time.perf_counter())
            return f"REC IN {_clock_hms(left)}"
        left = max(0.0, self._record_end_at - time.perf_counter())
        return f"{_clock_hms(left)} / {_clock_hms(RECORD_SECONDS)}"

    def _preview_frame(self, frame):
        if self._record_phase in ("countdown", "recording"):
            return _stamp_banner(frame, self._record_banner_text())
        return frame

    def _finish_recording(self, cancelled: bool) -> None:
        path = self._record_path
        frames = self._record_frames
        writer = self._record_writer
        self._record_writer = None
        self._record_path = None
        self._reset_record_ui()
        if writer is not None:
            writer.release()
        if path is not None and frames > 0 and path.is_file():
            logger.info("recording saved %s frames=%s cancelled=%s", path, frames, cancelled)
            self.statusBar().showMessage(f"Saved {path}")
            return
        if path is not None and path.is_file() and frames == 0:
            path.unlink()
        if cancelled:
            self.statusBar().showMessage("Recording cancelled.")

    def _reset_record_ui(self) -> None:
        self._record_timer.stop()
        self._record_phase = "idle"
        self._countdown_left = 0
        self.record_button.setText(f"RECORD {int(RECORD_SECONDS)}s")
        self.record_clock.setText(_clock_hms(RECORD_SECONDS))

    def closeEvent(self, event) -> None:
        if self._record_phase != "idle":
            self._finish_recording(cancelled=True)
        self._save_settings()
        self.capture.shutdown()
        self.inference.shutdown()
        self.capture.wait(4000)
        self.inference.wait(20000)
        logger.info("shutdown")
        event.accept()


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        width, height = text.lower().split("x")
        return int(width), int(height)
    except ValueError:
        return 1280, 720


def _stamp_banner(frame, text: str):
    image = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.05
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    left, top = 12, 12
    right = min(image.shape[1] - 8, left + text_width + 28)
    bottom = min(image.shape[0] - 8, top + text_height + baseline + 22)
    cv2.rectangle(image, (left, top), (right, bottom), (40, 30, 150), -1)
    cv2.putText(image, text, (left + 14, top + text_height + 8), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def _clock_hms(seconds) -> str:
    tenths = max(0, int(round(float(seconds or 0) * 10)))
    minutes, remainder = divmod(tenths, 600)
    whole, tenth = divmod(remainder, 10)
    return f"{minutes:02d}:{whole:02d}.{tenth}"


def _clock(seconds) -> str:
    total = max(0, int(float(seconds or 0)))
    return f"{total // 60:02d}:{total % 60:02d}"
