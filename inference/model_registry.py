"""MMDetection models that exist in the current 3.3 model zoo."""

from __future__ import annotations

import json
from dataclasses import dataclass

from utils.downloader import checkpoint_looks_valid
from utils.logger import get_logger
from utils.paths import project_path

logger = get_logger("vision_studio.models")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    task: str
    profile: str
    mmdet_name: str
    checkpoint_url: str
    checkpoint_file: str
    input_size: int
    supports_masks: bool
    description: str

    @property
    def config_path(self) -> str:
        return self.mmdet_name

    @property
    def checkpoint_path(self):
        return project_path("checkpoints", self.checkpoint_file)

    @property
    def installed(self) -> bool:
        return checkpoint_looks_valid(self.checkpoint_path)


# Names, configs and weight URLs come from configs/rtmdet/metafile.yml on main (mmdet 3.3.0).
BUILTIN_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="rtmdet_tiny",
        display_name="RTMDet Fast",
        task="detection",
        profile="fast",
        mmdet_name="rtmdet_tiny_8xb32-300e_coco",
        checkpoint_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
        checkpoint_file="rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
        input_size=640,
        supports_masks=False,
        description="RTMDet-Tiny, COCO box AP 40.9. Default realtime detection model.",
    ),
    ModelSpec(
        model_id="rtmdet_s",
        display_name="RTMDet Balanced",
        task="detection",
        profile="balanced",
        mmdet_name="rtmdet_s_8xb32-300e_coco",
        checkpoint_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
        checkpoint_file="rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
        input_size=640,
        supports_masks=False,
        description="RTMDet-S, COCO box AP 44.5. Balanced speed and quality.",
    ),
    ModelSpec(
        model_id="rtmdet_l",
        display_name="RTMDet Accurate",
        task="detection",
        profile="accurate",
        mmdet_name="rtmdet_l_8xb32-300e_coco",
        checkpoint_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
        checkpoint_file="rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
        input_size=640,
        supports_masks=False,
        description="RTMDet-L, COCO box AP 51.3. Heavier detection model.",
    ),
    ModelSpec(
        model_id="rtmdet_ins_tiny",
        display_name="RTMDet-Ins Fast",
        task="instance_segmentation",
        profile="fast",
        mmdet_name="rtmdet-ins_tiny_8xb32-300e_coco",
        checkpoint_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth",
        checkpoint_file="rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth",
        input_size=640,
        supports_masks=True,
        description="RTMDet-Ins-Tiny. Realtime boxes and instance masks, mask AP 35.4.",
    ),
    ModelSpec(
        model_id="rtmdet_ins_s",
        display_name="RTMDet-Ins Balanced",
        task="instance_segmentation",
        profile="balanced",
        mmdet_name="rtmdet-ins_s_8xb32-300e_coco",
        checkpoint_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth",
        checkpoint_file="rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth",
        input_size=640,
        supports_masks=True,
        description="RTMDet-Ins-S. Instance segmentation, mask AP 38.7.",
    ),
    ModelSpec(
        model_id="rtmdet_ins_m",
        display_name="RTMDet-Ins Accurate",
        task="instance_segmentation",
        profile="accurate",
        mmdet_name="rtmdet-ins_m_8xb32-300e_coco",
        checkpoint_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth",
        checkpoint_file="rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth",
        input_size=640,
        supports_masks=True,
        description="RTMDet-Ins-M. Heavier instance segmentation, mask AP 42.1.",
    ),
    ModelSpec(
        model_id="mask_rcnn_r50",
        display_name="Mask R-CNN R50",
        task="instance_segmentation",
        profile="accurate",
        mmdet_name="mask-rcnn_r50_fpn_1x_coco",
        checkpoint_url="https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
        checkpoint_file="mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
        input_size=800,
        supports_masks=True,
        description="Mask R-CNN ResNet-50 FPN. Accurate masks, slower than RTMDet-Ins.",
    ),
)

DETECTION_BY_PROFILE = {
    "fast": "rtmdet_tiny",
    "balanced": "rtmdet_s",
    "accurate": "rtmdet_l",
}
SEGMENTATION_BY_PROFILE = {
    "fast": "rtmdet_ins_tiny",
    "balanced": "rtmdet_ins_s",
    "accurate": "rtmdet_ins_m",
}


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {spec.model_id: spec for spec in BUILTIN_MODELS}
        self._load_custom()

    def _load_custom(self) -> None:
        folder = project_path("custom_models")
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skip custom model %s: %s", path.name, exc)
                continue
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                try:
                    spec = ModelSpec(
                        model_id=str(item["model_id"]),
                        display_name=str(item.get("display_name") or item["model_id"]),
                        task=str(item.get("task") or "detection"),
                        profile=str(item.get("profile") or "accurate"),
                        mmdet_name=str(item["mmdet_name"]),
                        checkpoint_url=str(item.get("checkpoint_url") or ""),
                        checkpoint_file=str(item["checkpoint_file"]),
                        input_size=int(item.get("input_size") or 640),
                        supports_masks=bool(item.get("supports_masks")),
                        description=str(item.get("description") or "Custom model"),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Invalid custom model entry in %s: %s", path.name, exc)
                    continue
                self._models[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._models:
            raise KeyError(model_id)
        return self._models[model_id]

    def all(self) -> list[ModelSpec]:
        return list(self._models.values())

    def reload_custom(self) -> None:
        self._models = {spec.model_id: spec for spec in BUILTIN_MODELS}
        self._load_custom()
