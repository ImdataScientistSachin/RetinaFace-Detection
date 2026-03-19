---
title: RetinaFace Pro
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# RetinaFace Pro — Face Detection, Landmark Localisation & Identity Verification.

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/Backend-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview
**RetinaFace Pro** is an industry-grade computer vision pipeline designed for dense face detection, facial landmark localisation, and identity verification. This project refactors experimental research into a production-ready suite powered by the **RetinaFace** architecture and **DeepFace** (ArcFace).

### The Model simultaneously predicts:
- **Face Detection:** Highly accurate bounding boxes.
- **5-Point Landmarks:** High-precision localisation for eyes, nose, and mouth corners.
- **Identity Verification:** One-to-Many identity matching with ArcFace embeddings.
- **Confidence Scores:** Reliable quality measurements for every detection.

## 🌟 Key Features
- **Multi-Face Analysis:** Detect and extract portraits from group shots in a single pass.
- **1-to-Many Identity Check:** Compare a reference persona against multiple test images with detailed distance metrics.
- **Dual Interface:** 
    - **Modern Web Dashboard:** Interactive UI for real-time visualization and feedback.
    - **Powerful CLI Tool:** High-throughput batch processing for folders and automated scripts.
- **Production Standard:** Fully Dockerized and pre-configured for free deployment on **Hugging Face Spaces**.

## 🧬 Tech Stack
- **Python 3.10+**
- **RetinaFace** — SOTA detection backbone.
- **DeepFace (ArcFace)** — identity verification model.
- **Streamlit** — interactive web application framework.
- **OpenCV & TensorFlow** — image processing and inference engine.

## 🏗️ Project Structure
```text
Ratina_Face/
├── src/                # Core logic (FaceDetector wrapper)
├── examples/           # Sample images & testing data
├── tests/              # Automated verification suite (pytest)
├── .github/workflows/  # CI/CD pipelines (GitHub Actions)
├── app.py              # Streamlit Web Application
├── main.py             # Command-Line Tool
├── Dockerfile          # Production container setup
└── requirements.txt    # Managed dependencies
```

## 🚀 Quick Start

### 1. Local Installation
```bash
git clone https://github.com/ImdataScientistSachin/Ratina-Face
cd Ratina-Face

# Install dependencies via the safe module syntax
python -m pip install -r requirements.txt
```

### 2. Launch Web Dashboard
> [!TIP]
> On Windows systems, always use the `python -m` prefix to avoid path-space errors.
```bash
python -m streamlit run app.py
```

### 3. Command Line Usage
```bash
# Detect and process a folder of images
python main.py detect --input_dir examples/ --output_dir results

# Verify identity match between two images
python main.py verify examples/user1.jpg examples/user2.jpg
```

## 🐳 Deployment & CI/CD
Deploy to **Hugging Face Spaces** or any cloud provider using our optimized Docker setup:
```bash
docker build -t retinaface-pro .
docker run -p 7860:7860 retinaface-pro
```
*Continuous integration is handled automatically via GitHub Actions (Linting & Pytest).*

## � License
MIT License — free to use, modify, and distribute.

## 👤 Author
**Sachin Paunikar** — [LinkedIn](https://www.linkedin.com/in/sachin-paunikar-datascientists) | [GitHub](https://github.com/ImdataScientistSachin)
