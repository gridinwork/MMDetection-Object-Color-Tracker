"""One-frame pipeline: scale, detect, track, color, draw."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from color.color_analyzer import ColorAnalyzer
from inference.mmdet_backend import MMDetBackend, ModelNotInstalled
from inference.mode_manager import resolve_model
from inference.model_registry import ModelRegistry
from tracking.object_tracker import ObjectTracker
from utils.gpu_info import describe_device, resolve_device
from utils.logger import get_logger
from visualization.renderer import OverlayRenderer

logger = get_logger("vision_studio.pipeline")


@dataclass
class FramePacket:
    display_bgr: np.ndarray
    clean_bgr: np.ndarray
    tracks: list = field(default_factory=list)
    detected_counts: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    active_model: str = ""
    model_note: str = ""
    device_text: str = ""
    capture_fps: float = 0.0
    inference_fps: float = 0.0
    display_fps: float = 0.0
    dropped: int = 0
    queue_size: int = 0
    video_meta: dict = field(default_factory=dict)


class InferenceManager:
    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.backend = MMDetBackend()
        self.tracker = ObjectTracker()
        self.colors = ColorAnalyzer()
        self.renderer = OverlayRenderer()
        self.loaded_id = ""
        self.loaded_device = ""
        self.frame_index = 0
        self._mode_signature = None

    def close(self) -> None:
        logger.info("model unloaded %s", self.loaded_id or "-")
        self.backend.close()
        self.loaded_id = ""
        self.loaded_device = ""

    def reset_tracking(self) -> None:
        self.tracker.reset()
        self.colors.reset()

    def ensure_model(self, selection: str, segmentation: bool, optimization: str, device_choice: str) -> tuple[str, str]:
        device = resolve_device(device_choice)
        model_id, note = resolve_model(selection, segmentation, optimization, device, self.registry)
        if model_id == self.loaded_id and device == self.loaded_device and self.backend.inferencer is not None:
            return model_id, note
        spec = self.registry.get(model_id)
        if self.loaded_id:
            logger.info("model unloaded %s", self.loaded_id)
        self.backend.load(spec, device)
        self.loaded_id = model_id
        self.loaded_device = device
        logger.info("selected model %s device %s", spec.display_name, device)
        return model_id, note

    def process(self, frame_bgr: np.ndarray, cfg, stats: dict) -> FramePacket:
        started = time.perf_counter()
        self.frame_index += 1
        frame_bgr = np.ascontiguousarray(frame_bgr)
        clean = frame_bgr.copy()
        run_detector = bool(cfg.detection or cfg.segmentation)
        model_note = ""
        active_name = self.loaded_id
        preprocess_ms = 0.0
        model_ms = 0.0
        tracker_ms = 0.0
        color_ms = 0.0
        tracks = []
        counts: dict[str, int] = {}

        if run_detector:
            model_id, model_note = self.ensure_model(cfg.model, cfg.segmentation, cfg.optimization, cfg.device)
            active_name = self.registry.get(model_id).display_name
            self._log_mode(cfg, model_id)
            t0 = time.perf_counter()
            scale = float(cfg.processing_scale)
            if scale < 0.99:
                scaled_w = max(32, int(frame_bgr.shape[1] * scale))
                scaled_h = max(32, int(frame_bgr.shape[0] * scale))
                small = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            else:
                small = frame_bgr
                scale = 1.0
            preprocess_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            detections = self.backend.infer(small, float(cfg.confidence), int(cfg.max_objects))
            model_ms = (time.perf_counter() - t1) * 1000.0
            if scale != 1.0:
                inv = 1.0 / scale
                for det in detections:
                    det.bbox = det.bbox.astype(np.float32) * inv
                    if det.mask_roi is not None:
                        out_w = max(1, int(round(det.bbox[2] - det.bbox[0])))
                        out_h = max(1, int(round(det.bbox[3] - det.bbox[1])))
                        det.mask_roi = cv2.resize(det.mask_roi, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            enabled = cfg.enabled_classes if not cfg.class_filter_all else None
            if enabled is not None:
                detections = [det for det in detections if det.class_name in enabled]
            for det in detections:
                if det.score >= float(cfg.confidence):
                    counts[det.class_name] = counts.get(det.class_name, 0) + 1
            t2 = time.perf_counter()
            tracks = self.tracker.update(
                detections,
                high_thr=float(cfg.confidence),
                lost_frames_timeout=int(cfg.lost_frames_timeout),
                trail_length=int(cfg.trail_length),
                enabled=bool(cfg.tracking),
            )
            if enabled is not None:
                tracks = [track for track in tracks if track.class_name in enabled]
            tracker_ms = (time.perf_counter() - t2) * 1000.0
            t3 = time.perf_counter()
            if cfg.color and tracks:
                self.colors.apply(
                    tracks,
                    clean,
                    self.frame_index,
                    int(cfg.color_every),
                    use_mask=bool(cfg.segmentation),
                    palette=bool(cfg.palette),
                    tracking=bool(cfg.tracking),
                )
            color_ms = (time.perf_counter() - t3) * 1000.0
        else:
            self.tracker.reset()

        device_text = ""
        if self.frame_index % 20 == 0 or not stats.get("device_text"):
            try:
                device_text = describe_device(self.loaded_device or resolve_device(cfg.device))
            except Exception:
                device_text = self.loaded_device or cfg.device
            stats["device_text"] = device_text
        else:
            device_text = stats.get("device_text") or ""

        render_started = time.perf_counter()
        options = {
            "segmentation": bool(cfg.segmentation),
            "tracking": bool(cfg.tracking),
            "color": bool(cfg.color),
            "trails": bool(cfg.trails),
            "labels": bool(cfg.labels),
            "show_confidence": bool(cfg.show_confidence),
            "show_id": bool(cfg.show_id),
            "detailed_color": bool(cfg.detailed_color),
            "palette": bool(cfg.palette),
            "locked_id": cfg.locked_id,
            "debug_lines": [],
        }
        display = self.renderer.render(clean, tracks, options)
        render_ms = (time.perf_counter() - render_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        timings = {
            "frame": self.frame_index,
            "preprocess_ms": preprocess_ms,
            "model_ms": model_ms,
            "tracker_ms": tracker_ms,
            "color_ms": color_ms,
            "render_ms": render_ms,
            "total_ms": total_ms,
        }
        if cfg.debug:
            debug_lines = [
                f"FRAME {self.frame_index}",
                f"CAPTURE FPS {stats.get('capture_fps', 0):.1f}",
                f"INFERENCE FPS {stats.get('inference_fps', 0):.1f}",
                f"DISPLAY FPS {stats.get('display_fps', 0):.1f}",
                f"PREPROCESS MS {preprocess_ms:.1f}",
                f"MODEL MS {model_ms:.1f}",
                f"TRACKER MS {tracker_ms:.1f}",
                f"COLOR MS {color_ms:.1f}",
                f"RENDER MS {render_ms:.1f}",
                f"TOTAL MS {total_ms:.1f}",
                f"QUEUE SIZE {stats.get('queue_size', 0)}",
                f"DROPPED FRAMES {stats.get('dropped', 0)}",
                f"GPU {device_text}",
                f"ACTIVE MODEL {active_name}",
            ]
            display = self.renderer.render(clean, tracks, {**options, "debug_lines": debug_lines})
            timings["render_ms"] = (time.perf_counter() - render_started) * 1000.0
            timings["total_ms"] = (time.perf_counter() - started) * 1000.0
        return FramePacket(
            display_bgr=display,
            clean_bgr=clean,
            tracks=[_public_track(track) for track in tracks],
            detected_counts=counts,
            timings=timings,
            active_model=active_name,
            model_note=model_note,
            device_text=device_text,
            capture_fps=float(stats.get("capture_fps", 0)),
            inference_fps=float(stats.get("inference_fps", 0)),
            display_fps=float(stats.get("display_fps", 0)),
            dropped=int(stats.get("dropped", 0)),
            queue_size=int(stats.get("queue_size", 0)),
            video_meta=dict(stats.get("video_meta") or {}),
        )

    def _log_mode(self, cfg, model_id: str) -> None:
        signature = (
            model_id,
            bool(cfg.detection),
            bool(cfg.segmentation),
            bool(cfg.tracking),
            bool(cfg.color),
            cfg.device,
        )
        if signature != self._mode_signature:
            self._mode_signature = signature
            logger.info(
                "mode change detection=%s segmentation=%s tracking=%s color=%s model=%s device=%s",
                cfg.detection,
                cfg.segmentation,
                cfg.tracking,
                cfg.color,
                model_id,
                cfg.device,
            )


def _public_track(track) -> dict:
    x1, y1, x2, y2 = [float(v) for v in track.bbox]
    mask = None if track.mask_roi is None else track.mask_roi.copy()
    return {
        "id": int(track.track_id),
        "class_name": track.class_name,
        "score": float(track.score),
        "xyxy": (x1, y1, x2, y2),
        "bbox": (x1, y1, x2 - x1, y2 - y1),
        "color": track.color_name,
        "rgb": track.rgb,
        "hsv": track.hsv,
        "lab": track.lab,
        "palette": list(track.palette or []),
        "tracked": bool(track.tracked),
        "age": int(track.age),
        "motion": track.motion,
        "velocity_x": float(track.velocity_x),
        "velocity_y": float(track.velocity_y),
        "mask_roi": mask,
    }
