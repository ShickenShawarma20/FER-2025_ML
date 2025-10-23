# FER-2025_ML

Minimal real-time facial emotion recognition demo powered by OpenCV and ONNXRuntime.

## Quick start

Install the required packages:

```bash
pip install -r requirements.txt
```

(Optional) prefetch the face detector and emotion classification models:

```bash
python scripts/download_models.py
```

Run the real-time demo (defaults to webcam `0`):

```bash
python -m fer2025.app.realtime_demo --camera 0
```

Use `--save-metrics` to store per-frame JSON analytics and `--classes 8` to enable the full FER+ label set.
