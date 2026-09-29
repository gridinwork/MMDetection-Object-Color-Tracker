"""Live view rendered inside Qt, with click-to-select."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget


class VideoWidget(QWidget):
    clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._img_w = 1
        self._img_h = 1
        self.setMinimumSize(640, 360)
        self.setMouseTracking(False)

    def set_bgr(self, frame_bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        height, width, _channels = rgb.shape
        image = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image.copy())
        self._img_w = width
        self._img_h = height
        self.update()

    def clear_view(self) -> None:
        self._pixmap = None
        self.update()

    def _target_rect(self) -> QRectF:
        if self._pixmap is None or self._img_w <= 0 or self._img_h <= 0:
            return QRectF()
        scale = min(self.width() / self._img_w, self.height() / self._img_h)
        draw_w = self._img_w * scale
        draw_h = self._img_h * scale
        x = (self.width() - draw_w) / 2
        y = (self.height() - draw_h) / 2
        return QRectF(x, y, draw_w, draw_h)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0e0e12"))
        if self._pixmap is not None:
            painter.drawPixmap(self._target_rect().toRect(), self._pixmap)
        else:
            painter.setPen(QColor("#9aa0a6"))
            painter.drawText(self.rect(), Qt.AlignCenter, "LIVE VIEW")

    def mousePressEvent(self, event) -> None:
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        rect = self._target_rect()
        pos = event.position()
        if not rect.contains(pos):
            self.clicked.emit(-1.0, -1.0)
            return
        rel_x = (pos.x() - rect.x()) / max(1.0, rect.width())
        rel_y = (pos.y() - rect.y()) / max(1.0, rect.height())
        self.clicked.emit(rel_x * self._img_w, rel_y * self._img_h)
