# RetinaFace Detection & Verification

Face detection, 5-point landmark localization, and 1-to-many identity verification built with RetinaFace, DeepFace (ArcFace), OpenCV, and Streamlit.

- **Live Demo:** https://imdatascientistsachin-ratina-face.hf.space
- **Hugging Face Space:** https://huggingface.co/spaces/ImdataScientistSachin/Ratina_face

## Features

- **Face Detection & Landmark Localization:** Detects bounding boxes and 5 facial landmarks (`right_eye`, `left_eye`, `nose`, `mouth_right`, `mouth_left`) via `retina-face` (ResNet50 + FPN).
- **Face Alignment & Extraction:** Extracts aligned face crops using detected landmarks.
- **Identity Verification:** 1-to-many verification via `deepface` with ArcFace embeddings (configurable model).
- **Optional Telemetry:** Optional MLflow latency and confidence logging (`use_mlflow=True` on `FaceDetector`).
- **Interfaces:** Interactive Streamlit web app (`app.py`) and CLI (`main.py`).

## Project Structure

```text
RetinaFace-Detection/
├── .github/workflows/
│   ├── ci.yml               # Ruff lint + pytest CI
│   └── hf_sync.yml          # Sync app to Hugging Face Space
├── src/
│   └── detector.py          # FaceDetector wrapper
├── tests/
│   └── test_detector.py     # Unit tests
├── app.py                   # Streamlit web UI
├── main.py                  # CLI entry point
├── conftest.py              # Pytest sys.path setup
├── Dockerfile               # Production container (port 7860)
├── requirements.txt         # Runtime dependencies
└── requirements-dev.txt     # Lint and test dependencies
```

## Quick Start

### Local Setup

```bash
git clone https://github.com/ImdataScientistSachin/RetinaFace-Detection.git
cd RetinaFace-Detection
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the Streamlit UI:

```bash
streamlit run app.py
```

Run via CLI:

```bash
python main.py detect --image path/to/image.jpg --output_dir output
python main.py verify path/to/ref.jpg path/to/test.jpg --model ArcFace
```

### Docker

```bash
docker build -t retinaface-detection .
docker run -p 7860:7860 retinaface-detection
```

### Lint & Test

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest tests/ -v
```
