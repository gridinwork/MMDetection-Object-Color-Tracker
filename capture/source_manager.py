"""Select webcam, video file or still image."""

from __future__ import annotations

from capture.camera_capture import CameraCapture
from capture.image_source import ImageSource
from capture.video_capture import VideoFileCapture


class SourceManager:
    def __init__(self) -> None:
        self.camera = CameraCapture()
        self.video = VideoFileCapture()
        self.images = ImageSource()
        self.kind = ""

    def scan_cameras(self) -> list[int]:
        return CameraCapture.scan()

    def open_camera(self, index: int, width: int, height: int) -> None:
        self.close()
        self.camera.open(index, width, height)
        self.kind = "webcam"

    def open_video(self, path: str) -> None:
        self.close()
        self.video.open(path)
        self.kind = "video"

    def load_image(self, path: str):
        self.close()
        self.kind = "image"
        return self.images.load(path)

    def read(self):
        if self.kind == "webcam":
            return self.camera.read()
        if self.kind == "video":
            return self.video.read()
        return False, None

    def close(self) -> None:
        self.camera.close()
        self.video.close()
        self.kind = ""
