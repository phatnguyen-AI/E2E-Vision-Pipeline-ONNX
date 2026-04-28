# E2E Vision Pipeline ONNX

End-to-end dermatological image classification system. Covers the full ML lifecycle from raw data ingestion through model training, ONNX export, REST API serving, and a production-ready web frontend -- all containerized and deployable on commodity hardware.

**Live demo:** [e2e-vision-pipeline-onnx.vercel.app](https://e2-e-vision-pipeline-onnx.vercel.app/)

https://github.com/user-attachments/assets/404fa155-dccc-4df5-95a1-dd1e0801e463

---

## Problem Statement

Skin lesion classification is a high-stakes, class-imbalanced problem. The ISIC 2019 dataset contains eight diagnostic categories with severe long-tail distribution (e.g., Nevus dominates at >60% of samples while Dermatofibroma accounts for <1%). A naive model will over-predict majority classes and fail on the clinically critical minority ones (Melanoma, SCC).

This project addresses that challenge end-to-end: from data balancing and augmentation strategy through optimized inference, demonstrating production-oriented thinking at every stage.

---

## Architecture Overview

```
Raw Data (Kaggle ISIC 2019)
    |
    v
[Data Pipeline] -- download, stratified split (70/15/15), class-aware augmentation
    |
    v
[Training Pipeline] -- ResNet50 (ImageNet V2 pretrained), two-phase fine-tuning
    |
    v
[Model Export] -- PyTorch -> ONNX (opset 12, dynamic batch, constant folding)
    |
    v
[Serving Layer] -- FastAPI + ONNX Runtime (CPU), Docker container
    |
    v
[Frontend] -- Vanilla HTML/CSS/JS, drag-and-drop image upload, responsive layout
```

---

## Key Technical Decisions

### Class Imbalance Strategy

The dataset exhibits extreme imbalance across 8 classes. Rather than applying uniform oversampling, the preprocessing pipeline uses a **tiered augmentation strategy** with class-specific multipliers:

| Class | Samples (approx.) | Augmentation Factor | Transform Tier |
|-------|-------------------|---------------------|----------------|
| NV    | ~12,000           | 0x (none)           | --             |
| MEL   | ~4,500            | 0.2x                | Light          |
| BCC   | ~3,300            | 0.5x                | Light          |
| BKL   | ~2,600            | 1.0x                | Medium         |
| AK    | ~800              | 4.0x                | Medium         |
| SCC   | ~600              | 6.0x                | Heavy          |
| VASC  | ~250              | 13.0x               | Heavy          |
| DF    | ~230              | 14.0x               | Heavy          |

Transform tiers escalate from simple flips (Light) through rotation + color jitter (Medium) to affine transforms + aggressive color perturbation (Heavy). This prevents the model from memorizing augmented copies of rare classes while still providing meaningful training signal.

Inverse-frequency class weights are additionally applied at the loss function level.

### Two-Phase Training

Training is split into two phases to maximize transfer learning effectiveness:

**Phase 1 (52 epochs) -- Feature Adaptation**
- Backbone frozen; only the custom classification head trains
- Loss: Cross-Entropy with class weights + label smoothing (0.1)
- At epoch 8, `layer3` and `layer4` are unfrozen with discriminative learning rates (backbone: 1e-5, head: 1e-4)
- BatchNorm layers in unfrozen blocks remain in eval mode to preserve pretrained statistics

**Phase 2 (50 epochs) -- Full Fine-Tuning**
- Loads best checkpoint from Phase 1
- Switches loss to Focal Loss (gamma=2.0) with class weights to further address hard examples in minority classes
- All parameters trainable; AdamW with weight decay 1e-3

### Model Architecture

- Base: ResNet50 (ImageNet1K_V2 weights)
- Custom head: `Dropout(0.4) -> Linear(2048, 1024) -> LeakyReLU -> Dropout(0.4) -> Linear(1024, 8)`
- The double-dropout design is intentional: the first layer regularizes the high-dimensional backbone features; the second prevents co-adaptation in the compressed representation

### ONNX Export

The trained PyTorch model is exported to ONNX format with:
- Opset version 12
- Dynamic batch axis for flexible serving
- Constant folding enabled for graph optimization
- Numerical equivalence validated (max absolute error < 1e-4 against PyTorch output)

This reduces the inference dependency from PyTorch (~2GB) to ONNX Runtime (~50MB), which is critical for deploying on resource-constrained environments.

---

## Evaluation Results

Evaluated on the held-out test split (15% of data, no augmentation applied):

| Metric             | Score  |
|--------------------|--------|
| Accuracy           | 0.8181 |
| Weighted F1-Score  | 0.8181 |
| Weighted Precision | 0.8220 |
| Weighted Recall    | 0.8181 |

---

## Supported Diagnostic Classes

| Code | Diagnosis                         | Clinical Significance       |
|------|-----------------------------------|-----------------------------|
| NV   | Melanocytic Nevus                 | Benign                      |
| MEL  | Melanoma                          | Malignant, high mortality   |
| BCC  | Basal Cell Carcinoma              | Malignant, most common      |
| BKL  | Benign Keratosis-like Lesion      | Benign                      |
| AK   | Actinic Keratosis                 | Pre-malignant               |
| SCC  | Squamous Cell Carcinoma           | Malignant                   |
| VASC | Vascular Lesion                   | Benign, vascular origin     |
| DF   | Dermatofibroma                    | Benign                      |

---

## Project Structure

```
E2E-Vision-Pipeline-ONNX/
|-- convert/
|   +-- convert_onnx.py              # PyTorch-to-ONNX export with numerical validation
|-- model/
|   |-- best_model.pt                # Best PyTorch checkpoint (~98MB)
|   +-- resnet50_final.onnx          # Exported ONNX model (~98MB)
|-- notebooks/
|   |-- 01_eda.ipynb                  # Exploratory data analysis
|   +-- 02_model_research.ipynb       # Architecture experimentation
|-- src/
|   |-- api/
|   |   |-- api.py                    # FastAPI application (predict endpoint, CORS, static mount)
|   |   |-- Dockerfile                # Production container definition
|   |   +-- requirements-backend.txt  # Minimal runtime dependencies
|   |-- data/
|   |   |-- dataset.py                # ISICDataset (PyTorch Dataset, multi-format image loading)
|   |   |-- downloader.py             # Kaggle dataset downloader via kagglehub
|   |   +-- preprocess.py             # Stratified split, tiered augmentation pipeline
|   |-- models/
|   |   |-- loss.py                   # Focal Loss implementation with class weight support
|   |   +-- model.py                  # ResNet50 transfer learning setup, selective unfreezing
|   |-- test/
|   |   |-- evaluate.py               # ONNX model evaluation on test set
|   |   +-- onnx_demo.py              # Batch inference CLI with per-class accuracy report
|   +-- ui/
|       |-- index.html                # Single-page application
|       |-- style.css                 # Responsive layout, medical-themed design
|       |-- script.js                 # Drag-and-drop upload, API integration
|       +-- nginx.conf                # Production static file serving
|-- requirements.txt                  # Full development dependencies
+-- README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- 8GB RAM minimum
- CPU with AVX instruction support (for ONNX Runtime)

### Setup

```bash
git clone https://github.com/phatnguyen-AI/E2E-Vision-Pipeline-ONNX.git
cd E2E-Vision-Pipeline-ONNX

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run API Server

```bash
uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker

```bash
docker build -f src/api/Dockerfile -t skin-classifier .
docker run -p 8000:8000 skin-classifier
```

---

## API Reference

### POST /predict

Accepts a dermatological image and returns the predicted diagnosis.

**Request:**
```
Content-Type: multipart/form-data
Body: file=<image file>
```

**Response:**
```json
{
  "class_code": "MEL",
  "disease": "U hac to (Melanoma)",
  "confidence": 0.923456
}
```

---

## Technology Stack

| Layer          | Technology                                      |
|----------------|-------------------------------------------------|
| Training       | PyTorch 2.0, torchvision, scikit-learn          |
| Augmentation   | torchvision.transforms (custom SquarePad)       |
| Loss Function  | Cross-Entropy (Phase 1), Focal Loss (Phase 2)   |
| Model Export   | torch.onnx, ONNX Runtime                        |
| Serving        | FastAPI, Uvicorn                                 |
| Frontend       | HTML5, CSS3, JavaScript (vanilla)                |
| Infrastructure | Docker, Nginx                                    |
| Monitoring     | TensorBoard                                      |

---

## License

MIT License. See [LICENSE](LICENSE) for full terms.

This software is developed for research and educational purposes. It is not intended as a substitute for professional medical diagnosis.

---

## Author

Phat Nguyen -- [tanphat6406@gmail.com](mailto:tanphat6406@gmail.com) -- [LinkedIn](https://linkedin.com/in/phat-nguyen-a264722b7)
