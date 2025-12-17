# End-to-End Image Classification Pipeline

## Project Overview

This repository implements an end-to-end image classification pipeline built for demonstrable, production-minded AI engineering. It contains data collection and preprocessing, model development and training with PyTorch, ONNX conversion for optimized inference, evaluation scripts, and a simple Streamlit UI for quick demos.

This project showcases skills typically expected from an AI Engineer: data engineering, model design and training, reproducibility, model optimization and exporting, inference deployment, and clear documentation for handoff.

## Live Demo & Video
- **Live Web UI:** [https://image-classification-bjbd.onrender.com](https://image-classification-bjbd.onrender.com) (Completed Web UI for recruiters to test)

### Video Walkthrough
[<video controls width="100%">
  <source src="images/demo_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>](https://github.com/user-attachments/assets/1b0c08d2-3170-4a79-837a-b906de459a51)


## Key Highlights / Skills Demonstrated
- **Data Pipeline:** Data ingestion, cleaning, augmentation and dataset splitting implemented in `src/data`.
- **Modeling:** Custom training loop and model definitions using PyTorch (`src/models`). Transfer learning with `resnet50` backbone and experiment variations.
- **Training & Evaluation:** Training scripts with configurable hyperparameters in `train/train.py` and evaluation utilities in `src/test`.
- **Model Optimization:** Conversion to ONNX (`convert/convert_onnx.py`) and validation of exported model (`model/resnet50_final.onnx`).
- **Inference & Deployment:** Example inference scripts in `src/test/onnx_demo.py` and a demo UI in `src/ui/streamlit.py`.
- **Reproducibility:** Requirements files (`requirements.txt`, `requirements-backend.txt`, `requirements-frontend.txt`) and deterministic training seeds in code.
- **Experiment Tracking:** Clear folder structure for `data/processed`, model artifacts in `model/`, and notebooks for EDA and experimentation (`notebooks/`).

## Repository Structure (high level)
- **`src/`**: Core codebase
	- `src/data/`: dataset, downloader, preprocessing utilities
	- `src/models/`: model definitions and loss implementations
	- `src/api/`: lightweight API example for serving model (`api.py`)
	- `src/test/`: evaluation scripts and ONNX demo
	- `src/ui/`: Streamlit app for quick demo
- **`train/`**: training entrypoint and training-related utilities
- **`convert/`**: scripts to export models to ONNX
- **`data/`**: raw and processed datasets (structured for reproducible experiments)
- **`model/`**: trained artifacts (e.g., `best_model.pt`, `resnet50_final.onnx`)
- **`notebooks/`**: EDA and model research notebooks

## Quick Start — Reproduce Locally

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Prepare data:
- Place raw images into `data/raw/` following the expected folder/label layout, or run the downloader in `src/data/downloader.py` if available.

3. Preprocess data:

```bash
python -m src.data.preprocess
```

4. Train a model (example):

```bash
python train/train.py --config configs/train_config.yaml
```

5. Evaluate and export:

```bash
python src/test/evaluate.py --weights model/best_model.pt
python convert/convert_onnx.py --weights model/best_model.pt --output model/resnet50_final.onnx
```

6. Run ONNX demo or Streamlit UI:

```bash
python src/test/onnx_demo.py --onnx model/resnet50_final.onnx
streamlit run src/ui/streamlit.py
```

## Reproducibility & Experiments
- Seeded experiments and clear `data/processed_*` splits support reproducible results.
- Use the notebooks in `notebooks/` for exploratory analysis and to justify preprocessing/model choices.

## Design Decisions (talking points for interviews)
- Transfer learning with `resnet50` backbone: trade-off between performance and inference cost.
- Data augmentation strategies used to reduce overfitting and improve generalization.
- Loss function choices and class imbalance handling in `src/models/loss.py`.
- Exporting to ONNX for compatibility with inference runtimes and easier deployment to cloud edge devices.
- Simple API + Streamlit demo to show end-to-end flow from training to serving.

## How This Shows My AI Engineering Skills
- **End-to-end ownership:** From data ingestion to model export and demo UI.
- **Production awareness:** ONNX export, lightweight API, and structured artifacts for deployment.
- **Tooling & best practices:** Virtual environments, requirements files, modular code structure.
- **Experimentation mindset:** Notebooks, clear train/eval scripts, and artifact versioning.
- **Collaboration-ready:** Clean code layout and documented run commands for easy handoff.

## Files To Inspect for Specific Skills
- **Data engineering:** `src/data/dataset.py`, `src/data/preprocess.py`
- **Modeling & training:** `src/models/model.py`, `train/train.py`
- **Export & inference:** `convert/convert_onnx.py`, `src/test/onnx_demo.py`
- **Demo & serving:** `src/ui/streamlit.py`, `src/api/api.py`

## Next Steps / Improvements (optional talking points)
- Add experiment tracking (e.g., MLflow or Weights & Biases) for hyperparameter search and artifact lineage.
- Build CI pipeline to run linting, tests, and optionally smoke-test the ONNX model.
- Containerize the inference API with a small Dockerfile and provide deployment manifests.

---

## Case Study: Handling Edge Cases & Hard Samples

Below is a real-world example from the `images/` directory illustrating an image with significant noise, irregular lighting, and contrast variations—challenges frequently encountered in production environments.

![](images/hard_sample.png)

**Strategy for handling such anomalies:**
- **Preprocessing:** Implemented brightness balancing, color normalization, and smart cropping in `src/data/preprocess.py`.
- **Data Augmentation:** Utilized random variations (rotation, brightness jitter, contrast) to enhance model robustness against input variability.
- **Robust Loss & Sampling:** Tuned sampling strategies and loss functions to address class imbalance and effectively handle outliers (`src/models/loss.py`).
- **Inference Safeguards:** Incorporated output filtering and confidence thresholding in `src/test/onnx_demo.py` to prevent over-confident predictions on low-quality inputs.

If you'd like, I can also:
- Translate this README to English for LinkedIn/GitHub audiences.
- Add a short `README_EN.md` with bullet-point highlights for recruiters.
- Create a one-page PDF summary you can attach to applications.

File updated: `README.md` — contains a concise, interview-focused presentation of the project and the AI engineering skills it demonstrates.
