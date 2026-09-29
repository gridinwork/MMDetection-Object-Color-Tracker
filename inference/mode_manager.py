"""Pick one model so detection and segmentation are not both inferred."""

from __future__ import annotations

from inference.model_registry import DETECTION_BY_PROFILE, SEGMENTATION_BY_PROFILE, ModelRegistry

PROFILE_IDS = ("fast", "balanced", "accurate")
_LIGHTER = {
    "fast": ("fast",),
    "balanced": ("balanced", "fast"),
    "accurate": ("accurate", "balanced", "fast"),
}


def _pick_installed(table: dict[str, str], profile: str, registry: ModelRegistry) -> tuple[str, str]:
    requested = table[profile]
    for name in _LIGHTER[profile]:
        model_id = table[name]
        spec = registry.get(model_id)
        if not spec.installed:
            continue
        if model_id == requested:
            return model_id, ""
        requested_name = registry.get(requested).display_name
        return model_id, f"{requested_name} is not installed, using {spec.display_name}"
    return requested, ""


def resolve_model(selection: str, segmentation: bool, optimization: str, device: str, registry: ModelRegistry) -> tuple[str, str]:
    """Return (model_id, note). The note is empty when the selection is used as-is."""
    profile = {
        "max_fps": "fast",
        "balanced": "balanced",
        "max_quality": "accurate",
    }.get(optimization, "balanced")
    if device == "cpu" and selection == "auto":
        profile = "fast"

    if selection == "auto" or selection in PROFILE_IDS:
        chosen_profile = selection if selection in PROFILE_IDS else profile
        table = SEGMENTATION_BY_PROFILE if segmentation else DETECTION_BY_PROFILE
        return _pick_installed(table, chosen_profile, registry)

    spec = registry.get(selection)
    if segmentation and spec.supports_masks:
        return spec.model_id, ""
    if not segmentation and not spec.supports_masks:
        return spec.model_id, ""
    if segmentation and not spec.supports_masks:
        model_id, note = _pick_installed(SEGMENTATION_BY_PROFILE, spec.profile, registry)
        if not note:
            note = f"Segmentation uses {registry.get(model_id).display_name}"
        return model_id, note
    model_id, note = _pick_installed(DETECTION_BY_PROFILE, spec.profile, registry)
    if not note:
        note = f"Detection uses {registry.get(model_id).display_name}"
    return model_id, note
