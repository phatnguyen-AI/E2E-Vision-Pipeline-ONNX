from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware # <--- 1. Import thêm cái này
import io
import os
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image
import onnxruntime as ort

app = FastAPI(title="Image Disease Classification (ONNX)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)
# ---------------------------------------------------------


# Model path (relative to repo root)
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model" / "resnet50_final.onnx"
# Static UI directory (professional medical-themed frontend)
STATIC_DIR = BASE_DIR / "src" / "ui" / "static"

# Mount static files (served at /ui)
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


# Classes used during training (same order as the model output)
TARGET_CLASSES = [
    "AK",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "SCC",
    "VASC",
]

# Map class codes to Vietnamese disease names
VIETNAMESE_LABELS: Dict[str, str] = {
    "NV": "Nốt ruồi (Nevus)",
    "MEL": "U hắc tố (Melanoma)",
    "BCC": "Ung thư biểu mô tế bào đáy (Basal cell carcinoma)",
    "BKL": "Tăng sừng lành tính (Benign keratosis-like)",
    "AK": "Dày sừng quang hóa (Actinic keratosis)",
    "SCC": "Ung thư biểu mô tế bào vảy (Squamous cell carcinoma)",
    "VASC": "Tổn thương mạch máu (Vascular lesion)",
    "DF": "U sợi da (Dermatofibroma)",
}


def load_onnx_session(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at: {model_path}")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess


# Load model at startup
try:
    session = load_onnx_session(MODEL_PATH)
    input_name = session.get_inputs()[0].name
except Exception as e:
    session = None
    input_name = None


def preprocess_image(file_bytes: bytes, img_size: int = 224) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((img_size, img_size))
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Nhận file ảnh, trả về tên bệnh (tiếng Việt) và hệ số confident."""
    if session is None:
        raise HTTPException(status_code=500, detail="ONNX model not loaded on server")

    content = await file.read()
    try:
        input_tensor = preprocess_image(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể xử lý ảnh: {e}")

    try:
        outputs = session.run(None, {input_name: input_tensor})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi chạy mô hình: {e}")

    logits = outputs[0]
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]

    probs = softmax(logits)
    top_idx = int(np.argmax(probs))
    class_code = TARGET_CLASSES[top_idx] if top_idx < len(TARGET_CLASSES) else str(top_idx)
    vietnamese_name = VIETNAMESE_LABELS.get(class_code, class_code)
    confidence = float(probs[top_idx])

    return JSONResponse({
        "class_code": class_code,
        "disease": vietnamese_name,
        "confidence": round(confidence, 6),
    })


@app.get("/")
def root():
    if STATIC_DIR.exists():
        return RedirectResponse(url="/ui/")
    return {"message": "FastAPI ONNX inference — POST an image to /predict"}


if __name__ == "__main__":
    import uvicorn
    # uvicorn src.api.api:app --reload
    uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)