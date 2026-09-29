"""Draw boxes, masks, ids, colors and debug timing on a frame."""

from __future__ import annotations

import cv2
import numpy as np

from visualization.colors import id_color
from visualization.labels import draw_label_block, draw_palette_row
from visualization.masks import overlay_mask


class OverlayRenderer:
    def render(self, frame_bgr: np.ndarray, tracks, options: dict) -> np.ndarray:
        canvas = frame_bgr.copy()
        locked_id = options.get("locked_id")
        locked_track = None
        for track in tracks:
            color = id_color(track.track_id)
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
            if options.get("segmentation"):
                overlay_mask(canvas, track.bbox, track.mask_roi, color, 0.45)
            if options.get("trails") and options.get("tracking"):
                points = list(track.trajectory.points)
                if len(points) >= 2:
                    poly = np.array([[int(px), int(py)] for px, py in points], dtype=np.int32)
                    cv2.polylines(canvas, [poly], False, color, 2, cv2.LINE_AA)
            thickness = 3 if locked_id is not None and track.track_id == locked_id else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            if locked_id is not None and track.track_id == locked_id:
                locked_track = track
                self._corners(canvas, x1, y1, x2, y2, (255, 255, 255))
            lines = self._lines(track, options)
            draw_label_block(canvas, x1, y1 - 2, lines, color)
            if options.get("palette") and track.palette:
                draw_palette_row(canvas, x1, min(canvas.shape[0] - 16, y2 + 4), track.palette)
        if locked_track is not None:
            banner = f"TARGET: {locked_track.class_name.upper()} #{locked_track.track_id}"
            cv2.rectangle(canvas, (8, 8), (8 + 18 * len(banner), 36), (20, 20, 20), -1)
            cv2.putText(canvas, banner, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        debug_lines = options.get("debug_lines") or []
        if debug_lines:
            self._debug(canvas, debug_lines)
        return canvas

    @staticmethod
    def _lines(track, options: dict) -> list[str]:
        parts = []
        if options.get("show_id") and options.get("tracking"):
            parts.append(f"#{track.track_id}")
        if options.get("labels"):
            parts.append(track.class_name.upper())
        if options.get("show_confidence"):
            parts.append(f"{track.score * 100:.0f}%")
        lines = []
        if parts:
            lines.append(" | ".join(parts))
        if options.get("color") and track.color_name:
            lines.append(f"COLOR: {track.color_name}")
            if options.get("detailed_color") and track.rgb:
                red, green, blue = track.rgb
                lines.append(f"RGB: {red}, {green}, {blue}")
        return lines

    @staticmethod
    def _corners(image, x1, y1, x2, y2, color) -> None:
        length = max(12, min(28, (x2 - x1) // 5, (y2 - y1) // 5))
        pairs = (
            ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
            ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
            ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
            ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
        )
        for corner, horizontal, vertical in pairs:
            cv2.line(image, corner, horizontal, color, 2, cv2.LINE_AA)
            cv2.line(image, corner, vertical, color, 2, cv2.LINE_AA)

    @staticmethod
    def _debug(image, lines: list[str]) -> None:
        y = 52
        for line in lines:
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            y += 16
