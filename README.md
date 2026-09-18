# Explainable Age Estimation with Masking

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics%20YOLO26-111F68?style=for-the-badge&logo=yolo&logoColor=white)](https://www.ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Click](https://img.shields.io/badge/Click-CLI-black?style=for-the-badge&logo=gnu-bash&logoColor=white)](https://click.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![ROCm](https://img.shields.io/badge/AMD%20ROCm-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)

---

## What this project does and how far it came

This project investigates which regions of a human face carry the most information for **age estimation**, using a masking-based ablation study. A YOLO26n-cls backbone was repurposed from its original 1000-class ImageNet classifier into a single-value age regression model, then trained on a combined UTKFace + Lagenda dataset (~109,000 images).

To determine feature importance, nine models were trained under an identical, locked configuration — one on the unmasked (base) dataset, and eight more, each on a version of the dataset with a specific facial region masked out (eyes, eyes + surrounding area, nose, lips, chin, a central "bulls-eye" region, upper face, and lower face). Comparing each masked model's error against the base model's error reveals how much age-relevant signal was removed along with that region.

As an additional exploratory step, three fusion models (Decision Tree, Random Forest, and Bayesian Ridge) were trained to combine all nine individual models' predictions into a single, improved estimate.

**Current state of the project:**
- Training pipeline fully built, tuned, and locked to a final configuration (batch size 64, hybrid class-weighted L1 loss, Softplus-constrained regression head)
- All 9 models (base + 8 masking conditions) trained and evaluated on a held-out test set
- Fusion models trained and evaluated, with bootstrap confidence intervals confirming their improvement over the base model is statistically real, not noise
- A command-line interface (CLI) built with Click, covering training, batch inference, single-image testing, evaluation, and fusion-model inference
- A CPU-only Docker container built and verified, covering all commands except training (which requires GPU access, see [Docker setup](#docker-setup))

---

## Project structure

Age-Estimation/
├── src/
│ ├── cli.py # Click-based CLI: train, infer, evaluate, model-test, model-fusion
│ ├── dataset.py # AgeDataset, filename parsing, blocklist, class weighting
│ ├── model.py # build_age_model() — YOLO26n-cls with regression head
│ ├── train.py # training-step functions (plain and weighted)
│ ├── evaluate.py # validate()
│ ├── logging_utils.py # CSV logging helpers
│ ├── exec.py # plain L1 baseline training runner
│ ├── exec_class_weighted.py # per-age weighted training runner
│ ├── exec_class_weighted_hybrid.py # final locked training runner (hybrid weighting)
│ ├── build_fusion_data.py # builds per-image prediction table across all 9 models
│ ├── train_fusion_model.py # trains and bootstrap-tests the 3 fusion models
│ ├── test_fusion_per_image.py # per-image fusion model results
│ └── masking/
│ ├── masking.py # applies a masking condition across a dataset
│ ├── mediapipe_init.py # MediaPipe face/pose landmarker setup
│ ├── body_part.py # facial region landmark index definitions
│ └── failurelog.py # logs images where face detection failed
├── tests/
│ ├── test_model.py # test-set evaluation with per-image CSV output
│ └── predict_single_image.py # single-image inference sanity check
├── data/
│ ├── splits/ # train / val / test, unmasked
│ └── masked/ # one folder per masking condition, each with train/val/test
├── checkpoints/ # trained model weights (.pt) and fusion models (.joblib)
├── logs/ , CLILogs/ # CSV logs from training, testing, and CLI runs
├── mediapipe/ # MediaPipe .task model files
├── Dockerfile
├── requirements-docker.txt
└── .dockerignore

---

## Setup requirements

- **Python 3.12**
- **GPU (for training only):** NVIDIA (CUDA) or AMD (ROCm 7.2+); training runs on CPU as a fallback but is significantly slower
- **WSL2 + Ubuntu 24.04** if on Windows with an AMD GPU (see project notes for ROCm/WSL setup)

**Libraries needed** (install into a virtual environment):
```bash
pip install torch torchvision  # matched to your GPU backend — see PyTorch's install selector
pip install ultralytics mediapipe click pandas scikit-learn joblib pillow numpy
```

For AMD/ROCm specifically, install the ROCm-matched PyTorch wheels from `repo.radeon.com` rather than the default PyPI build — see project setup notes for exact versions used (torch 2.9.1+rocm7.2.0, torchvision 0.24.0+rocm7.2.0).

---

## Docker setup

A CPU-only Docker image is provided, covering `infer`, `evaluate`, `model-test`, and `model-fusion`. **Training is not containerized**, since it requires GPU access, and configuring ROCm GPU passthrough inside Docker under WSL2 was judged too high-risk within the project's timeline — training should be run directly on a machine with a properly configured GPU environment instead.

**Build the image:**
```bash
docker build -t age-estimation-cli .
```

**Run a command**, mounting your local data/checkpoints so the container can access them:
```bash
docker run --rm \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/CLILogs:/app/CLILogs \
  age-estimation-cli evaluate --checkpoint checkpoints/base_model.pt --test-dir data/splits/test
```

---

## How to use the CLI

All commands are run via `python3 src/cli.py <command> [options]` (or the equivalent `docker run` form above).

**`train`** — train a model from scratch or resume from a checkpoint:
```bash
python3 src/cli.py train \
  --train-dir data/splits/train --val-dir data/splits/val \
  --batch-size 64 --epochs 30 --lr 0.001 \
  --blocklist logs/no_face_detection.csv
```

**`infer`** — batch inference over a folder of images, writing predictions to a CSV:
```bash
python3 src/cli.py infer --checkpoint checkpoints/base_model.pt --input-dir data/splits/test
```

**`evaluate`** — full test-set evaluation, reporting MAE and logging per-image results:
```bash
python3 src/cli.py evaluate --checkpoint checkpoints/base_model.pt --test-dir data/splits/test --blocklist logs/no_face_detection.csv
```

**`model-test`** — quick single-image prediction from any trained model, by name:
```bash
python3 src/cli.py model-test --model eyes_masked --image path/to/photo.jpg
```

**`model-fusion`** — runs a single image through all 9 models (masking it live for each condition) and combines the results via a chosen fusion method:
```bash
python3 src/cli.py model-fusion --image path/to/photo.jpg --method bayesian
```

---

## How we did the training

**Model:** YOLO26n-cls backbone (ImageNet-pretrained), with the original 1000-class head replaced by `Linear(1280 → 1) + Softplus`, converting it into an age regression model. Softplus was chosen over ReLU to avoid dead gradients near zero and guarantee non-negative age predictions.

**Data:** UTKFace + Lagenda-derived images, combined and split 70/20/10 (train/val/test). ~10% of images were excluded via a blocklist where MediaPipe failed to detect a face, applied consistently across all splits and all masking conditions.

**Hyperparameters:** determined through systematic, one-variable-at-a-time testing — batch size (32/64/128), learning rate (0.0001–0.01), and loss function (L1 vs. MSE) — landing on batch size 64, learning rate 0.001–0.003, 30 epochs, and L1 loss.

**Class imbalance:** the dataset is heavily skewed toward ages 20–40. A hybrid class-weighted loss was implemented — inverse-square-root frequency weighting per exact age below a threshold, pooled into buckets above it, capped to prevent instability — which measurably reduced error on underrepresented older ages at a small cost to overall accuracy.

**Masking experiments:** each of the 8 masked-condition models was trained under the identical locked configuration, differing only in the input dataset, and evaluated on the matching masked test set — isolating the effect of the masked region on model accuracy.

**Fusion models:** trained on validation-set predictions (to avoid using test data at any stage), evaluated on the held-out test set, with a bootstrap analysis confirming the resulting improvement over the base model is statistically significant.

Full methodology, results, and limitations are documented in the accompanying project report.

---

## Authors

**Iman Jahanpanah** — [imanjahanpanah121@gmail.com](mailto:imanjahanpanah121@gmail.com)

**Omid Tavassoli** — [portfolio.omidtavassoli.dev](https://portfolio.omidtavassoli.dev)