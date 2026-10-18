# Gesture CV Engine

Real-time hand gesture recognition engine written in modern C++ (C++17) utilizing **OpenCV** and **ONNX Runtime**. Features an optimized CPU inference pipeline and low-overhead HUD rendering.

![Gesture Recognition Demo](assets/demo.gif)

---

## Features

- **Real-Time CPU Inference:** Sustained **30 FPS** (~31 ms end-to-end latency) on CPU via ONNX Runtime with multi-threading and full graph optimization (`ORT_ENABLE_ALL`).
- **Hand Presence Gating:** Variance-based standard deviation filter (`isHandPresent`) prevents false positives on uniform backgrounds before passing frames to inference.
- **Temporal Smoothing:** Majority-voting sliding window (`TemporalSmoother`) eliminates flicker between gesture states.
- **Zero-Heap HUD:** Reticle and telemetry rendered strictly via bounded sub-mat ROIs without full-frame heap allocations.
- **Aspect-Ratio Safe:** Dynamic letterboxing onto a fixed 854x480 (16:9) canvas to preserve camera proportions.

---

## Machine Learning & Pipeline

- **Architecture:** Lightweight custom CNN (3x Conv2D + BatchNorm + Dropout + Dense) trained from scratch in **PyTorch**.
- **Domain Gap Adaptation:** Trained on a [Synthetic Hand Gestures Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers/data) with heavy data augmentations (rotations, shifts, lighting variations, and motion blur) to generalize to real webcam conditions.
- **Hand Orientation & Invariance:** Trained on the **back of the hand** (dorsal view). Gestures are hand-agnostic — both left and right hands map to the same classes.
- **Input Pipeline:** Central 240x240 ROI cropped from camera feed -> processed via `cv::dnn::blobFromImage` (resized to `128x128`, scaled to `[0.0, 1.0]`, BGR-to-RGB) -> planar `[1, 3, 128, 128]` NCHW tensor.
- **Direct Probability Output:** Model incorporates an export-time `Softmax` head, returning class confidence scores without manual post-processing.
- **Recognized Classes:** `1_one`, `2_two`, `3_three`, `4_four`, `5_five`.

---

## Tech Stack

- **C++17** / **CMake** (>= 3.20)
- **OpenCV 4.x** (`core`, `imgproc`, `highgui`, `dnn`)
- **ONNX Runtime 1.18.0**
- **PyTorch** (Model training, augmentations, and ONNX export)

---

## Quick Start

### 1. Prerequisites

Install system compiler, CMake, OpenCV development packages, and archive utilities:

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev wget tar
```

---

### 2. Download ONNX Runtime

Download and extract the required pre-built ONNX Runtime 1.18.0 binaries into `third_party/`:

```bash
mkdir -p third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz -C third_party/
rm onnxruntime-linux-x64-1.18.0.tgz
```

---

### 3. Build

Configure the build directory with Release optimizations and compile using all available CPU threads:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

### 4. Run

Run the binary directly from the repository root so it can locate `models/gesture_model.onnx` and `models/labels.txt`:

```bash
# Launch with default camera (/dev/video0)
./build/gesture_engine

# Or pass a specific camera index (e.g. /dev/video1)
./build/gesture_engine 1
```

*(Press `q` or `ESC` in the viewport to exit)*

---

### 5. WSL2 Camera Forwarding (Optional)

If running inside WSL2 on Windows 11, forward your physical webcam using `usbipd`:

```powershell
# Windows PowerShell (Administrator):
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

---

## Controls & Telemetry

- **Target Reticle:** Place your hand within the central bounding box on the screen.
- **Hand Orientation:** Show the **back of your hand** (dorsal side) facing the camera, not the palm.
- **Hand Invariance:** Works with **either left or right hand** (both map to the same target classes).
- **Tracking States:**
  - `SCANNING` (Amber): No hand detected or background is too uniform (variance check active).
  - `LOCKED` (Cyan): Hand detected, displaying recognized gesture (`ONE` to `FIVE`) and prediction confidence.