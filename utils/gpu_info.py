"""CUDA device selection and memory readout."""

from __future__ import annotations


def cuda_devices() -> list[dict]:
    import torch

    devices = []
    if not torch.cuda.is_available():
        return devices
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_mb": props.total_memory / (1024 * 1024),
            }
        )
    return devices


def best_cuda_index() -> int | None:
    devices = cuda_devices()
    if not devices:
        return None
    return max(devices, key=lambda item: item["total_mb"])["index"]


def resolve_device(choice: str) -> str:
    import torch

    if choice == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if choice == "auto":
        index = best_cuda_index()
        return "cpu" if index is None else f"cuda:{index}"
    if choice.startswith("cuda:"):
        try:
            index = int(choice.split(":", 1)[1])
        except ValueError:
            index = best_cuda_index() or 0
        if index < torch.cuda.device_count():
            return f"cuda:{index}"
    index = best_cuda_index()
    return "cpu" if index is None else f"cuda:{index}"


def describe_device(device: str) -> str:
    import torch

    if device == "cpu" or not torch.cuda.is_available():
        return "CPU"
    index = int(device.split(":")[1]) if ":" in device else 0
    if index >= torch.cuda.device_count():
        return device
    props = torch.cuda.get_device_properties(index)
    total = props.total_memory / (1024 * 1024)
    used = torch.cuda.memory_allocated(index) / (1024 * 1024)
    return f"{props.name} | VRAM {used:.0f}/{total:.0f} MB"


def log_runtime_versions(logger) -> None:
    import sys

    import cv2
    import mmcv
    import mmengine
    import mmdet
    import torch

    logger.info("Python %s", sys.version.replace("\n", " "))
    logger.info("Torch %s", torch.__version__)
    logger.info("CUDA available %s | torch CUDA %s", torch.cuda.is_available(), torch.version.cuda)
    logger.info("MMCV %s", mmcv.__version__)
    logger.info("MMEngine %s", mmengine.__version__)
    logger.info("MMDetection %s", mmdet.__version__)
    logger.info("OpenCV %s", cv2.__version__)
    for device in cuda_devices():
        logger.info(
            "GPU cuda:%s %s %.0f MB",
            device["index"],
            device["name"],
            device["total_mb"],
        )
