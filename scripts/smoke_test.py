"""Smoke test: tracker, colors, DetInferencer and one CUDA image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import ensure_dirs, project_path


def check_tracker() -> None:
    import numpy as np

    from tracking.object_tracker import ObjectTracker
    from tracking.track_state import Detection

    tracker = ObjectTracker(lost_frames_timeout=20)
    first_id = None
    for step in range(8):
        x = 20 + step * 12
        det = Detection(1, "cup", 0.9, np.array([x, 30, x + 40, 80], dtype=np.float32))
        visible = tracker.update([det], high_thr=0.4, lost_frames_timeout=20, trail_length=30, enabled=True)
        if len(visible) != 1:
            raise RuntimeError(f"tracker lost the cup at step {step}")
        if first_id is None:
            first_id = visible[0].track_id
        elif visible[0].track_id != first_id:
            raise RuntimeError("track id changed while the cup moved")
    for _step in range(5):
        visible = tracker.update([], high_thr=0.4, lost_frames_timeout=20, trail_length=30, enabled=True)
        if visible:
            raise RuntimeError("empty frame should hide the track")
    det = Detection(1, "cup", 0.88, np.array([20 + 8 * 12, 30, 20 + 8 * 12 + 40, 80], dtype=np.float32))
    visible = tracker.update([det], high_thr=0.4, lost_frames_timeout=20, trail_length=30, enabled=True)
    if not visible or visible[0].track_id != first_id:
        raise RuntimeError("track id was not recovered after a short gap")
    print("tracker ok", first_id)


def check_colors() -> None:
    import numpy as np

    from color.color_classifier import ColorClassifier

    classifier = ColorClassifier()
    samples = {
        (30, 30, 210): "RED",
        (20, 20, 20): "BLACK",
        (245, 245, 245): "WHITE",
        (210, 60, 30): "BLUE",
        (40, 220, 240): "YELLOW",
        (40, 170, 40): "GREEN",
    }
    for bgr, expected in samples.items():
        name = classifier.classify_bgr_pixel(np.array(bgr, dtype=np.uint8))
        print(f"color {bgr} -> {name}")
        if name != expected:
            raise RuntimeError(f"expected {expected} for {bgr}, got {name}")
    print("colors ok")


def check_imports() -> None:
    import io
    from contextlib import redirect_stdout

    import cv2
    import mmcv
    import mmengine
    import mmdet
    import torch
    from mmdet.apis import DetInferencer
    from PySide6.QtWidgets import QApplication

    print("python", sys.version.split()[0])
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
    print("mmcv", mmcv.__version__)
    print("mmengine", mmengine.__version__)
    print("mmdet", mmdet.__version__)
    print("opencv", cv2.__version__)
    print("DetInferencer", DetInferencer.__name__)
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"gpu cuda:{index} {props.name} {props.total_memory / 1024**2:.0f} MB")
    else:
        print("CUDA inference skipped: torch.cuda.is_available() is false")
    _app = QApplication.instance() or QApplication([])
    with redirect_stdout(io.StringIO()):
        names = DetInferencer.list_models("mmdet")
    required = [
        "rtmdet_tiny_8xb32-300e_coco",
        "rtmdet_s_8xb32-300e_coco",
        "rtmdet_l_8xb32-300e_coco",
        "rtmdet-ins_tiny_8xb32-300e_coco",
        "rtmdet-ins_s_8xb32-300e_coco",
        "rtmdet-ins_m_8xb32-300e_coco",
        "mask-rcnn_r50_fpn_1x_coco",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        mask_like = [name for name in names if "mask" in name.lower() or "rcnn" in name.lower()]
        raise RuntimeError(f"Model names missing from this MMDetection build: {missing}. Mask-like: {mask_like[:20]}")
    print("model names ok")


def download_defaults() -> None:
    from inference.model_registry import ModelRegistry
    from utils.downloader import download_file

    registry = ModelRegistry()
    for model_id in ("rtmdet_tiny", "rtmdet_ins_tiny"):
        spec = registry.get(model_id)
        if spec.installed:
            print("already installed", spec.display_name)
            continue
        print("downloading", spec.display_name)
        download_file(spec.checkpoint_url, spec.checkpoint_path, progress=_print_progress)
        print("\nsaved", spec.checkpoint_path)


def _print_progress(done: int, total: int) -> None:
    if total <= 0:
        return
    percent = done * 100 // total
    print(f"\r{percent}%", end="", flush=True)


def ensure_demo() -> Path:
    path = project_path("assets", "demo.jpg")
    if path.is_file() and path.stat().st_size > 1000:
        return path
    import urllib.request

    url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def run_inference() -> None:
    import cv2
    import torch
    from mmdet.apis import DetInferencer

    from utils.gpu_info import resolve_device

    weights = project_path("checkpoints", "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth")
    if not weights.is_file():
        raise RuntimeError("Default detection checkpoint is missing")
    image_path = ensure_demo()
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("demo image is unreadable")
    device = resolve_device("auto")
    print("inference device", device)
    inferencer = DetInferencer(
        model="rtmdet_tiny_8xb32-300e_coco",
        weights=str(weights),
        device=device,
        show_progress=False,
    )
    result = inferencer(
        image,
        return_vis=False,
        show=False,
        no_save_vis=True,
        draw_pred=False,
        pred_score_thr=0.4,
        return_datasamples=True,
        no_save_pred=True,
        out_dir="",
    )
    sample = result["predictions"][0]
    scores = sample.pred_instances.scores.detach().cpu()
    labels = sample.pred_instances.labels.detach().cpu()
    keep = scores >= 0.4
    classes = inferencer.model.dataset_meta["classes"]
    print("detections", int(keep.sum()))
    shown = 0
    for score, label in zip(scores[keep], labels[keep]):
        print(f"  {classes[int(label)]} {float(score):.2f}")
        shown += 1
        if shown >= 8:
            break
    if int(keep.sum()) < 1:
        raise RuntimeError("demo image produced no detections above 0.40")
    if device.startswith("cuda"):
        print("cuda inference ok", torch.cuda.get_device_name(int(device.split(":")[1])))
    else:
        print("cpu inference ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-defaults", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    check_tracker()
    check_colors()
    check_imports()
    if args.download_defaults:
        download_defaults()
    run_inference()
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("SMOKE FAILED:", exc)
        raise
