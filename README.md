# MMDetection Object & Color Vision Studio

![Main application interface](docs/images/interface.png)

A local Windows computer-vision desktop application built on top of **OpenMMLab MMDetection 3.3.0**, **MMCV**, **MMEngine**, **RTMDet / RTMDet-Ins**, OpenCV and PySide6.

The application processes webcam streams, video files and still images locally. It combines object detection, instance segmentation, persistent object tracking, motion analysis and pixel-based color recognition in one desktop interface.

## Project background

The goal was to turn the MMDetection model/inference ecosystem into a practical real-time desktop vision application instead of using command-line demos and separate scripts.

This project was developed as a desktop application around the algorithms, model zoo and inference infrastructure provided by [OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection). MMDetection itself is not vendored into this repository; it is installed as an external dependency. The application layer, GUI, source management, local object tracker, color-analysis pipeline, visualization, model manager and workflow integration were developed specifically for this project.

## Main capabilities

- Webcam, video-file and image input
- Object detection
- Instance segmentation
- Persistent object IDs between frames
- ByteTrack-style two-stage association implemented locally with SciPy
- Motion state estimation: static / left / right / up / down
- Motion trails
- Pixel-based object color analysis
- RGB / HSV / LAB measurements
- Dominant-color palette extraction
- Per-class filtering
- Detection-confidence control
- Object locking and inspection
- Save annotated screenshot
- Save clean frame
- Save cropped object
- Save masked object with transparent background
- NVIDIA CUDA / CPU execution
- Multiple GPU selection
- Runtime FPS and VRAM status
- Download / verify / remove model checkpoints
- Support for custom MMDetection configurations and weights

## Demonstration

### Object detection and color recognition

![Detection and color recognition](docs/images/demo_01.png)

The application detects multiple people and objects and estimates visible colors directly from image pixels rather than deriving colors from class names.

![Multi-object tracking and color analysis](docs/images/demo_02.png)

### Instance segmentation

![Instance segmentation example 1](docs/images/demo_03.png)

When an instance-segmentation model is active, the program can use object masks for more accurate color measurement and render semi-transparent masks over detected instances.

![Instance segmentation example 2](docs/images/demo_04.png)

### Tracking, masks and motion trails

![Tracking and segmentation example](docs/images/demo_05.png)

Objects can retain persistent IDs across frames. Their center movement is used to estimate motion direction, and recent positions can be rendered as motion trails.

## Detection profiles

The application maps speed/quality profiles to models available in MMDetection 3.3.0.

| Profile | Detection model | Typical purpose |
| --- | --- | --- |
| Fast | RTMDet-Tiny | Maximum FPS |
| Balanced | RTMDet-S | General-purpose balance |
| Accurate | RTMDet-L | Higher detection accuracy |

Instance-segmentation profiles use the corresponding RTMDet-Ins models. Mask R-CNN R50 is also available through the model registry as a heavier segmentation option.

## Optimization modes

The **Optimization** setting controls processing scale and color-analysis frequency:

- **MAX FPS** — 50% processing scale, less frequent color calculation
- **BALANCED** — 75% processing scale
- **MAX QUALITY** — 100% processing scale, more frequent color calculation

Bounding boxes and masks are transformed back into full-frame coordinates before display.

## Object tracking

Tracking is performed locally without requiring MMTracking.

The tracker uses a ByteTrack-style strategy:

1. High-confidence detections are matched first.
2. Lower-confidence detections are then used to recover existing tracks.
3. Temporarily lost objects remain available for a configurable number of frames.
4. Center-point history is used for motion direction and trajectory rendering.

This allows labels such as `CUP #2` to remain attached to the same physical object while it moves through the scene.

## Color analysis

Color information is calculated from the actual image region belonging to each object.

If an instance mask is available, only masked pixels are analyzed. Otherwise the analyzer uses the central region of the bounding box to reduce background contamination.

The color subsystem calculates:

- mean RGB;
- HSV;
- LAB;
- named color classification;
- dominant-color palette.

Reference color ranges are stored in:

```text
color/color_ranges.json
```

## Supported sources

### Webcam

Select **Webcam**, camera index, resolution, device and model, then press **START**.

### Video

Supported formats include MP4, AVI, MOV and MKV. Video playback includes seeking, pause and stop controls. If inference is slower than the video stream, old frames are dropped so processing stays close to real time.

### Image

Supported still-image formats include JPG, JPEG, PNG and WEBP. Detection is run immediately after loading and can be recomputed when modes change.

## Model Manager

The integrated Model Manager can:

- display installed/not-installed model status;
- download model weights;
- verify checkpoints with PyTorch;
- delete local checkpoints;
- switch between detection and segmentation models.

Downloaded checkpoints are stored locally in `checkpoints/` and are intentionally excluded from this public repository because they are large third-party model files.

## Custom models

The architecture can register custom MMDetection models without changing the main application code.

See:

[custom_models/HOW_TO_ADD_CUSTOM_MODEL.md](custom_models/HOW_TO_ADD_CUSTOM_MODEL.md)

A custom entry can point to:

- a local MMDetection config;
- a checkpoint;
- task type;
- profile;
- class definitions;
- mask support.

## GPU and CPU

Device options include:

- **AUTO** — selects a CUDA GPU with the most suitable available memory, otherwise CPU;
- explicit CUDA device selection;
- CPU mode.

The status bar reports runtime model/device information and, where available, GPU memory usage.

## Installation

Recommended environment:

- Windows 10/11
- Python 3.11
- NVIDIA GPU optional but recommended

Run:

```text
install.bat
```

The installer creates a local virtual environment and installs the matching stack used by this project:

- PyTorch 2.1.2
- CUDA 12.1 build when NVIDIA CUDA is available
- MMCV 2.1.0
- MMEngine
- MMDetection 3.3.0
- PySide6
- OpenCV
- SciPy
- Pillow

It also downloads the lightweight starter checkpoints and runs a smoke test.

Then launch with:

```text
start.bat
```

## Repository structure

```text
app/              PySide6 desktop interface
capture/          camera, image and video sources
color/            RGB/HSV/LAB and dominant-color analysis
config/           persistent application settings
custom_models/    custom-model registration documentation
inference/        MMDetection inference and model registry
scripts/          installation/runtime verification utilities
tracking/         persistent object tracking and trajectories
utils/            downloader, GPU, settings and path helpers
visualization/    labels, masks and renderer
checkpoints/      downloaded model weights (not committed)
objects/          saved object crops (not committed)
screenshots/      generated screenshots (not committed)
```

## Privacy / local processing

After Python packages and selected model weights have been downloaded, normal camera/video/image processing can run locally. Frames do not need to be uploaded to a cloud service by this application.

## License and upstream attribution

This repository's original application code is released under the **Apache License 2.0**.

The project depends on the OpenMMLab ecosystem rather than redistributing its source code:

- [MMDetection 3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0) — Apache License 2.0
- [MMCV 2.1.0](https://github.com/open-mmlab/mmcv/tree/v2.1.0) — Apache License 2.0
- [MMEngine](https://github.com/open-mmlab/mmengine) — Apache License 2.0
- RTMDet / RTMDet-Ins / Mask R-CNN checkpoints are downloaded separately and are not committed to this repository.
- COCO-trained models can also be subject to the terms of the datasets and assets used to train them.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution and third-party notes.

## Responsible use

Computer-vision systems can produce incorrect detections, classifications, masks, tracking IDs or color estimates. Validate results for your own environment before using them in safety-critical, security-critical or automated decision-making systems.
