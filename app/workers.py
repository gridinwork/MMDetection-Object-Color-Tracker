"""Capture and inference threads. The GUI only receives the latest result."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

import cv2
from PySide6.QtCore import QThread, Signal

from capture.source_manager import SourceManager
from utils.logger import get_logger

logger = get_logger("vision_studio.workers")


class LatestFrameSlot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self._meta: dict = {}
        self._seq = 0
        self._taken = 0
        self._dropped = 0

    def push(self, frame, meta: dict | None = None) -> None:
        with self._lock:
            if self._frame is not None and self._seq != self._taken:
                self._dropped += 1
            self._frame = frame
            self._meta = dict(meta or {})
            self._seq += 1

    def take(self):
        with self._lock:
            if self._frame is None or self._seq == self._taken:
                return None
            self._taken = self._seq
            depth = 0
            return self._frame, self._meta, self._dropped, depth

    def depth(self) -> int:
        with self._lock:
            if self._frame is None or self._seq == self._taken:
                return 0
            return 1

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped


class FpsMeter:
    def __init__(self) -> None:
        self.times: deque[float] = deque(maxlen=60)

    def fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        if span <= 0:
            return 0.0
        return (len(self.times) - 1) / span

    def tick(self) -> float:
        self.times.append(time.perf_counter())
        return self.fps()


class CaptureWorker(QThread):
    cameras_found = Signal(list)
    source_failed = Signal(str)
    source_started = Signal(str)
    video_position = Signal(int, int, float)
    video_ended = Signal()

    def __init__(self, slot: LatestFrameSlot, shared) -> None:
        super().__init__()
        self.slot = slot
        self.shared = shared
        self._commands: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.sources = SourceManager()
        self._paused = False
        self._fps = FpsMeter()

    def enqueue(self, command: str, **payload) -> None:
        self._commands.put((command, payload))

    def shutdown(self) -> None:
        self._stop.set()
        self.enqueue("stop")

    def run(self) -> None:
        while not self._stop.is_set():
            self._drain()
            if self.sources.kind == "" or self._paused and self.sources.kind == "video":
                self.msleep(20)
                continue
            ok, frame = self.sources.read()
            self._drain()
            if self.sources.kind == "":
                continue
            if not ok or frame is None:
                if self.sources.kind == "video":
                    self.video_ended.emit()
                    self.sources.close()
                elif self.sources.kind == "webcam":
                    self.source_failed.emit("Camera frame grab failed")
                    self.sources.close()
                continue
            if self._paused and self.sources.kind == "webcam":
                self.msleep(10)
                continue
            meta = {"reset_tracker": False}
            if self.sources.kind == "video":
                cap = self.sources.video
                fps = cap.fps
                total = max(1, cap.frame_count)
                index = cap.position
                meta["video"] = {
                    "index": index,
                    "total": total,
                    "fps": fps,
                    "seconds": index / fps,
                    "duration": total / fps,
                }
                self.video_position.emit(index, total, fps)
                self.slot.push(frame.copy(), meta)
                self._pace(fps)
            else:
                self.slot.push(frame.copy(), meta)
            capture_fps = self._fps.tick()
            self.shared.set_rates(capture_fps=capture_fps, dropped=self.slot.dropped)

    def _pace(self, fps: float) -> None:
        delay = 1.0 / max(1.0, fps)
        time.sleep(max(0.0, delay * 0.85))

    def _drain(self) -> None:
        while True:
            try:
                command, payload = self._commands.get_nowait()
            except queue.Empty:
                return
            self._handle(command, payload)

    def _handle(self, command: str, payload: dict) -> None:
        try:
            if command == "scan":
                found = self.sources.scan_cameras()
                self.cameras_found.emit(found)
            elif command == "start_camera":
                self.sources.open_camera(int(payload["index"]), int(payload["width"]), int(payload["height"]))
                self._paused = False
                self._push_reset_marker()
                logger.info("camera start index=%s %sx%s", payload["index"], payload["width"], payload["height"])
                self.source_started.emit("webcam")
            elif command == "start_video":
                self.sources.open_video(payload["path"])
                self._paused = False
                self._push_reset_marker()
                logger.info("video start %s", payload["path"])
                self.source_started.emit("video")
            elif command == "stop":
                kind = self.sources.kind
                self.sources.close()
                self._paused = False
                if kind == "webcam":
                    logger.info("camera stop")
            elif command == "pause":
                self._paused = bool(payload.get("paused"))
            elif command == "seek":
                if self.sources.kind == "video":
                    self.sources.video.seek_ratio(float(payload["ratio"]))
                    self._push_reset_marker()
        except Exception as exc:
            logger.exception("capture command failed")
            self.source_failed.emit(str(exc))
            self.sources.close()

    def _push_reset_marker(self) -> None:
        ok, frame = self.sources.read()
        if ok and frame is not None:
            meta = {"reset_tracker": True}
            if self.sources.kind == "video":
                cap = self.sources.video
                fps = cap.fps
                total = max(1, cap.frame_count)
                index = cap.position
                meta["video"] = {
                    "index": index,
                    "total": total,
                    "fps": fps,
                    "seconds": index / fps,
                    "duration": total / fps,
                }
            self.slot.push(frame.copy(), meta)


class InferenceWorker(QThread):
    packet_ready = Signal()
    status = Signal(str)
    error = Signal(str)
    classes_ready = Signal(list)
    devices_ready = Signal(list)

    def __init__(self, slot: LatestFrameSlot, shared) -> None:
        super().__init__()
        self.slot = slot
        self.shared = shared
        self._stop = threading.Event()
        self._results_lock = threading.Lock()
        self.packet = None
        self._fps = FpsMeter()
        self._last_error = ""
        self._manager = None
        self._announced_classes: tuple = ()
        self._applied_mirror: bool | None = None

    def submit_image(self, frame, reset: bool) -> None:
        self.slot.push(frame.copy(), {"reset_tracker": reset, "kind": "image"})

    def shutdown(self) -> None:
        self._stop.set()

    def run(self) -> None:
        from inference.inference_manager import InferenceManager
        from inference.mmdet_backend import ModelNotInstalled
        from utils.gpu_info import cuda_devices, log_runtime_versions

        self._manager = InferenceManager()
        try:
            log_runtime_versions(logger)
        except Exception as exc:
            self.error.emit(f"Runtime import failed: {exc}")
            logger.exception("runtime import failed")
            return
        self.devices_ready.emit(cuda_devices())
        self.status.emit("Runtime ready")
        while not self._stop.is_set():
            item = self.slot.take()
            if item is None:
                self.msleep(5)
                continue
            frame, meta, dropped, _depth = item
            if meta.get("reset_tracker"):
                self._manager.reset_tracking()
                logger.info("tracking reset")
            cfg = self.shared.snapshot()
            if cfg.mirror and frame is not None:
                frame = cv2.flip(frame, 1)
            if self._applied_mirror is not None and cfg.mirror != self._applied_mirror:
                self._manager.reset_tracking()
            self._applied_mirror = bool(cfg.mirror)
            rates = self.shared.rates()
            inference_fps = self._fps.fps()
            stats = {
                "capture_fps": rates["capture_fps"] if meta.get("kind") != "image" else 0.0,
                "display_fps": rates["display_fps"],
                "inference_fps": inference_fps,
                "dropped": dropped,
                "queue_size": self.slot.depth(),
                "video_meta": meta.get("video") or {},
                "device_text": "",
            }
            try:
                packet = self._manager.process(frame, cfg, stats)
                self._last_error = ""
            except ModelNotInstalled as exc:
                message = str(exc)
                if message != self._last_error:
                    self._last_error = message
                    logger.error("inference error %s", message)
                    self.error.emit(message)
                from inference.inference_manager import FramePacket

                packet = FramePacket(display_bgr=frame, clean_bgr=frame.copy(), active_model=message)
                self.msleep(400)
            except Exception as exc:
                logger.exception("inference error")
                message = str(exc)
                if message != self._last_error:
                    self._last_error = message
                    self.error.emit(message)
                from inference.inference_manager import FramePacket

                packet = FramePacket(display_bgr=frame, clean_bgr=frame.copy(), active_model="error")
            inference_fps = self._fps.tick()
            packet.inference_fps = inference_fps
            self.shared.set_rates(dropped=dropped)
            names = tuple(self._manager.backend.class_names)
            if names and names != self._announced_classes:
                self._announced_classes = names
                self.classes_ready.emit(list(names))
            with self._results_lock:
                self.packet = packet
            self.status.emit(packet.active_model or "idle")
            self.packet_ready.emit()
        self._manager.close()
        logger.info("inference stopped")

    def latest_packet(self):
        with self._results_lock:
            return self.packet
