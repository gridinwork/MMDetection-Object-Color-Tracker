"""Exercise detection, masks, video, webcam, screenshots and the GUI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime import RuntimeConfig
from inference.inference_manager import InferenceManager
from utils.paths import project_path


def _cfg(**overrides) -> RuntimeConfig:
    cfg = RuntimeConfig(
        detection=True,
        segmentation=False,
        tracking=True,
        color=True,
        labels=True,
        show_confidence=True,
        show_id=True,
        confidence=0.4,
        processing_scale=0.75,
        model="fast",
        device="auto",
        optimization="balanced",
        color_every=1,
        max_objects=20,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def check_image(manager: InferenceManager, image) -> dict:
    packet = manager.process(image, _cfg(), {"capture_fps": 0, "display_fps": 0, "dropped": 0, "queue_size": 0})
    print("detection tracks", len(packet.tracks), "model", packet.active_model, "ms", round(packet.timings["total_ms"], 1))
    if not packet.tracks:
        raise RuntimeError("image pipeline returned no tracks")
    colored = [track for track in packet.tracks if track.get("color")]
    print("colored", len(colored), "example", packet.tracks[0]["class_name"], packet.tracks[0].get("color"))
    if not colored:
        raise RuntimeError("color analysis produced no names")
    return packet


def check_video_and_tracking(manager: InferenceManager, image) -> None:
    import cv2

    path = project_path("assets", "motion_test.mp4")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (image.shape[1], image.shape[0]))
    frames = []
    for step in range(12):
        frame = image.copy()
        x = 30 + step * 20
        cv2.rectangle(frame, (x, 40), (x + 80, 140), (40, 40, 220), -1)
        writer.write(frame)
        frames.append(frame)
    writer.release()

    from capture.video_capture import VideoFileCapture

    video = VideoFileCapture()
    video.open(str(path))
    fps = video.fps
    video.seek_ratio(0.4)
    position = video.position
    ok, frame = video.read()
    video.close()
    if not ok or frame is None:
        raise RuntimeError("video seek/read failed")
    print("video opened", fps, "seek position", position, frame.shape)

    manager.reset_tracking()
    seen = []
    model_ms = []
    for frame in frames:
        packet = manager.process(frame, _cfg(color=False, segmentation=False), {})
        ids = {track["id"] for track in packet.tracks if track["score"] >= 0.55}
        seen.append(ids)
        model_ms.append(packet.timings["model_ms"])
    persistent = seen[0] & seen[-1]
    print("video tracking persistent ids", sorted(persistent), "model ms", round(sum(model_ms) / len(model_ms), 1))
    if not persistent:
        raise RuntimeError(f"no track id survived the clip: {seen[0]} -> {seen[-1]}")


def check_segmentation(manager: InferenceManager, image) -> dict:
    manager.reset_tracking()
    packet = manager.process(
        image,
        _cfg(segmentation=True, model="fast", color=True, palette=True),
        {},
    )
    print("segmentation model", packet.active_model, "tracks", len(packet.tracks))
    if "Ins" not in packet.active_model:
        raise RuntimeError(f"expected an instance model, got {packet.active_model}")
    masked = [track for track in packet.tracks if track.get("mask_roi") is not None]
    print("masks", len(masked))
    if not masked:
        raise RuntimeError("segmentation returned no masks")
    return packet


def check_webcam() -> None:
    from capture.camera_capture import CameraCapture

    found = CameraCapture.scan(4)
    print("cameras", found)
    if not found:
        print("webcam skipped: no camera returned a frame")
        return
    camera = CameraCapture()
    camera.open(found[0], 1280, 720)
    ok, frame = camera.read()
    camera.close()
    if not ok or frame is None:
        raise RuntimeError("webcam opened but grab failed")
    print("webcam frame", frame.shape)


def check_saves(packet, clean_shape_source) -> None:
    import cv2

    from app.main_window import _parse_resolution

    assert _parse_resolution("1280x720") == (1280, 720)
    shot = project_path("screenshots", "verify_overlay.jpg")
    clean = project_path("screenshots", "verify_clean.jpg")
    if not cv2.imwrite(str(shot), packet.display_bgr):
        raise RuntimeError("screenshot write failed")
    if not cv2.imwrite(str(clean), packet.clean_bgr):
        raise RuntimeError("clean frame write failed")
    track = next(track for track in packet.tracks if track.get("mask_roi") is not None)
    x1, y1, x2, y2 = [int(round(value)) for value in track["xyxy"]]
    height, width = packet.clean_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = packet.clean_bgr[y1:y2, x1:x2]
    mask = track["mask_roi"]
    if mask.shape[:2] != crop.shape[:2]:
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    crop_path = project_path("objects", "verify_crop.png")
    mask_path = project_path("objects", "verify_mask.png")
    if not cv2.imwrite(str(crop_path), crop):
        raise RuntimeError("crop write failed")
    if not cv2.imwrite(str(mask_path), bgra):
        raise RuntimeError("masked crop write failed")
    print("saved", shot.name, clean.name, crop_path.name, mask_path.name)
    _ = clean_shape_source


def check_gui(image_path: Path) -> None:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    state = {"done": False, "error": ""}

    def start() -> None:
        window.open_image(str(image_path))

    def finish() -> None:
        if state["done"]:
            return
        state["error"] = "GUI timed out before a detection arrived"
        window.close()
        app.quit()

    def poll() -> None:
        if state.get("seg_requested") or not window._tracks:
            return
        window.save_screenshot()
        window.save_clean_frame()
        window._selected_id = window._tracks[0]["id"]
        window.save_object(False)
        window.controls.boxes["segmentation"].setChecked(True)
        state["seg_requested"] = time.perf_counter()

    def poll_mask() -> None:
        if state.get("done"):
            return
        if not state.get("seg_requested"):
            return
        masked = [track for track in window._tracks if track.get("mask_roi") is not None]
        model = window.model_label.text()
        if masked and "Ins" in model:
            window._selected_id = masked[0]["id"]
            window.save_object(True)
            state["done"] = True
            print("gui tracks", len(window._tracks), "model", model)
            window.close()
            app.quit()

    QTimer.singleShot(300, start)
    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(400)
    mask_timer = QTimer()
    mask_timer.timeout.connect(poll_mask)
    mask_timer.start(400)
    QTimer.singleShot(90000, finish)
    app.exec()
    if not state["done"]:
        raise RuntimeError(state["error"] or "GUI closed before verification finished")
    print("gui ok")


def main() -> int:
    import argparse

    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui-only", action="store_true")
    args = parser.parse_args()
    image_path = project_path("assets", "demo.jpg")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("demo.jpg is missing; run scripts/smoke_test.py first")
    if not args.gui_only:
        check_webcam()
        manager = InferenceManager()
        try:
            packet = check_image(manager, image)
            check_video_and_tracking(manager, image)
            seg_packet = check_segmentation(manager, image)
            check_saves(seg_packet, image)
        finally:
            manager.close()
        _ = packet
    check_gui(image_path)
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("VERIFY FAILED:", exc)
        raise
