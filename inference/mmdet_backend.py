"""DetInferencer wrapper for the MMDetection 3.3 API."""

from __future__ import annotations

import cv2
import numpy as np

from inference.model_registry import ModelSpec
from tracking.track_state import Detection
from utils.logger import get_logger

logger = get_logger("vision_studio.mmdet")

LOW_SCORE = 0.10


class ModelNotInstalled(RuntimeError):
    """Raised when the selected checkpoint is not on disk."""


class MMDetBackend:
    def __init__(self) -> None:
        self.inferencer = None
        self.spec: ModelSpec | None = None
        self.device = "cpu"
        self.class_names: tuple[str, ...] = ()

    def close(self) -> None:
        self.inferencer = None
        self.spec = None
        self.class_names = ()
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.info("model cache clear skipped: %s", exc)

    def load(self, spec: ModelSpec, device: str) -> list[str]:
        if not spec.installed:
            raise ModelNotInstalled(
                f"{spec.display_name} is not installed. Open Model Manager and download it."
            )
        from mmdet.apis import DetInferencer

        logger.info("model loaded start %s on %s", spec.model_id, device)
        self.close()
        weights = str(spec.checkpoint_path)
        model_ref = spec.mmdet_name
        if model_ref.endswith(".py"):
            from pathlib import Path

            from utils.paths import project_path

            path = Path(model_ref)
            if not path.is_file():
                path = project_path(model_ref)
            model_ref = str(path)
        import torch

        torch.backends.cudnn.benchmark = True
        self.inferencer = DetInferencer(
            model=model_ref,
            weights=weights,
            device=device,
            show_progress=False,
        )
        self.spec = spec
        self.device = device
        meta = getattr(self.inferencer.model, "dataset_meta", {}) or {}
        classes = meta.get("classes") or ()
        self.class_names = tuple(str(name) for name in classes)
        self.set_score_threshold(LOW_SCORE)
        logger.info("model loaded %s classes=%s", spec.display_name, len(self.class_names))
        return list(self.class_names)

    def set_score_threshold(self, threshold: float) -> None:
        if self.inferencer is None:
            return
        model = self.inferencer.model
        modules = [model, getattr(model, "bbox_head", None)]
        for module in modules:
            if module is None:
                continue
            cfg = getattr(module, "test_cfg", None)
            _write_score(cfg, float(threshold))

    def infer(self, image_bgr: np.ndarray, user_score: float, max_objects: int) -> list[Detection]:
        if self.inferencer is None:
            return []
        self.set_score_threshold(LOW_SCORE)
        output = self.inferencer(
            image_bgr,
            batch_size=1,
            return_vis=False,
            show=False,
            no_save_vis=True,
            draw_pred=False,
            pred_score_thr=float(user_score),
            return_datasamples=True,
            print_result=False,
            no_save_pred=True,
            out_dir="",
        )
        predictions = output.get("predictions") or []
        if not predictions:
            return []
        return self._to_detections(predictions[0], image_bgr.shape, user_score, max_objects)

    def _to_detections(self, sample, shape, user_score: float, max_objects: int) -> list[Detection]:
        instances = sample.pred_instances
        if instances is None or len(instances) == 0:
            return []
        bboxes = _to_numpy(instances.bboxes)
        scores = _to_numpy(instances.scores)
        labels = _to_numpy(instances.labels).astype(int)
        if bboxes.size == 0:
            return []
        keep = np.where(scores >= LOW_SCORE)[0]
        if len(keep) == 0:
            return []
        order = keep[np.argsort(-scores[keep])]
        cap = max(int(max_objects), 20) + 20
        order = order[:cap]
        masks = _fit_masks(_slice_masks(instances, order), shape)
        detections = []
        height, width = shape[:2]
        for local_index, index in enumerate(order):
            score = float(scores[index])
            label = int(labels[index])
            name = self.class_names[label] if 0 <= label < len(self.class_names) else str(label)
            box = bboxes[index].astype(np.float32).copy()
            box[0] = np.clip(box[0], 0, width - 1)
            box[2] = np.clip(box[2], 0, width - 1)
            box[1] = np.clip(box[1], 0, height - 1)
            box[3] = np.clip(box[3], 0, height - 1)
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            mask_roi = None
            if masks is not None:
                mask_roi = _crop_mask(masks[local_index], box)
            detections.append(
                Detection(
                    class_id=label,
                    class_name=name,
                    score=score,
                    bbox=box,
                    mask_roi=mask_roi,
                )
            )
        high = [det for det in detections if det.score >= user_score]
        low = [det for det in detections if det.score < user_score]
        high = high[: max(1, int(max_objects))]
        low = low[:20]
        return high + low


def _write_score(cfg, threshold: float) -> None:
    if cfg is None:
        return
    try:
        keys = list(cfg.keys())
    except Exception:
        return
    if "score_thr" in keys:
        cfg["score_thr"] = threshold
    for key in ("rcnn", "rpn"):
        if key in keys:
            _write_score(cfg[key], threshold)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _slice_masks(instances, order: np.ndarray) -> np.ndarray | None:
    raw = instances.get("masks") if hasattr(instances, "get") else getattr(instances, "masks", None)
    if raw is None:
        return None
    try:
        if hasattr(raw, "detach"):
            selected = raw[order].detach().cpu().numpy()
            return selected
        if hasattr(raw, "to_ndarray"):
            array = raw.to_ndarray()
            return np.asarray(array)[order]
        array = np.asarray(raw)
        return array[order]
    except Exception as exc:
        logger.warning("mask extraction failed: %s", exc)
        return None


def _fit_masks(masks: np.ndarray | None, shape) -> np.ndarray | None:
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        return None
    image_h, image_w = shape[:2]
    if masks.shape[1] == image_h and masks.shape[2] == image_w:
        return masks
    fitted = []
    for mask in masks:
        image = mask.astype(np.uint8)
        if image.size and int(image.max()) <= 1:
            image = image * 255
        fitted.append(cv2.resize(image, (image_w, image_h), interpolation=cv2.INTER_NEAREST))
    if not fitted:
        return None
    return np.stack(fitted)


def _crop_mask(mask: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    if mask.ndim == 3:
        mask = mask[0]
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.dtype != np.uint8:
        crop = (crop > 0).astype(np.uint8) * 255
    elif crop.max() <= 1:
        crop = crop * 255
    out_w = max(1, int(round(bbox[2] - bbox[0])))
    out_h = max(1, int(round(bbox[3] - bbox[1])))
    if crop.shape[1] != out_w or crop.shape[0] != out_h:
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return crop
