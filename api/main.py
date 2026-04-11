import base64
import io
import json
import os
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from utils.preprocessing import (
    DISEASE_NAMES,
    LABEL_NAMES,
    find_last_conv_layer,
    load_trained_model,
    make_gradcam_heatmap,
    prepare_inference_image,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_model.keras")
MODEL_META_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

app = FastAPI(title="Skin Cancer Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model: tf.keras.Model | None = None
_last_conv_layer: str | None = None
_active_model_path: str | None = None
FIXED_TTA_RUNS = 8


def _extract_quality_metrics(meta: dict[str, Any]) -> dict[str, float]:
    metrics = meta.get("metrics") or {}
    if not isinstance(metrics, dict):
        return {}

    quality = {}
    if "accuracy" in metrics:
        quality["accuracy"] = float(metrics["accuracy"])
    if "top2_acc" in metrics:
        quality["top2_acc"] = float(metrics["top2_acc"])
    if "loss" in metrics:
        quality["loss"] = float(metrics["loss"])
    return quality


def _resolve_model_path() -> str | None:
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH

    models_dir = os.path.join(PROJECT_ROOT, "models")
    if not os.path.isdir(models_dir):
        return None

    candidates = [
        os.path.join(models_dir, name)
        for name in os.listdir(models_dir)
        if name.endswith(".best.keras")
    ]
    if not candidates:
        return None

    # Prefer historically stronger backbones first, then newest checkpoint in that tier.
    backbone_priority = {"efficientnet": 3, "resnet": 2, "mobilenet": 1}

    def _score(path: str) -> tuple[int, float]:
        name = os.path.basename(path).lower()
        priority = 0
        for key, value in backbone_priority.items():
            if key in name:
                priority = value
                break
        return priority, os.path.getmtime(path)

    return max(candidates, key=_score)


def _load_latest_model() -> bool:
    global _model
    global _last_conv_layer
    global _active_model_path

    resolved_path = _resolve_model_path()
    if resolved_path is None:
        return False

    _model = load_trained_model(resolved_path)
    _active_model_path = resolved_path
    _last_conv_layer = find_last_conv_layer(_model)
    return True


@app.on_event("startup")
def load_model_on_startup() -> None:
    _load_latest_model()


def _get_model() -> tf.keras.Model:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not found. Train the model first.")
    return _model


def _overlay_gradcam(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> str:
    image_np = image_np.astype(np.float32)
    if image_np.max() > 1:
        image_np = image_np / 255.0

    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    merged = np.clip((1 - alpha) * image_np + alpha * heatmap_color, 0, 1)

    encoded_ok, encoded = cv2.imencode(".png", np.uint8(merged * 255))
    if not encoded_ok:
        raise HTTPException(status_code=500, detail="Failed to encode Grad-CAM image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _load_model_meta() -> dict[str, Any]:
    if not os.path.isfile(MODEL_META_PATH):
        return {}
    with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _tta_augment(image_batch: np.ndarray) -> np.ndarray:
    tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    tensor = tf.image.random_flip_left_right(tensor)
    tensor = tf.image.random_flip_up_down(tensor)
    tensor = tf.image.random_brightness(tensor, max_delta=0.06)
    tensor = tf.image.random_contrast(tensor, lower=0.9, upper=1.1)
    return tf.clip_by_value(tensor, 0.0, 255.0).numpy()


def _apply_confidence_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if np.isclose(temperature, 1.0):
        return probs

    clipped = np.clip(probs.astype(np.float64), 1e-8, 1.0)
    sharpened = np.power(clipped, 1.0 / temperature)
    normalized = sharpened / np.sum(sharpened)
    return normalized.astype(np.float32)


@app.get("/health")
def health() -> dict[str, Any]:
    meta = _load_model_meta()
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": _active_model_path or MODEL_PATH,
        "model_metrics": _extract_quality_metrics(meta),
    }


@app.get("/classes")
def classes() -> dict[str, Any]:
    return {
        "labels": LABEL_NAMES,
        "disease_names": DISEASE_NAMES,
        "metadata": _load_model_meta(),
    }


@app.post("/reload-model")
def reload_model() -> dict[str, Any]:
    loaded = _load_latest_model()
    return {
        "reloaded": loaded,
        "model_loaded": _model is not None,
        "model_path": _active_model_path or MODEL_PATH,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(default=True),
    confidence_temperature: float = Query(default=0.78, gt=0.0, le=2.0),
    include_probabilities: bool = Query(default=False),
) -> dict[str, Any]:
    model = _get_model()
    tta_runs = FIXED_TTA_RUNS

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    model_input = prepare_inference_image(image)

    if tta_runs == 1:
        probs = model.predict(model_input, verbose=0)[0]
    else:
        pred_list = []
        for _ in range(tta_runs):
            augmented = _tta_augment(model_input)
            pred_list.append(model.predict(augmented, verbose=0)[0])
        probs = np.mean(np.stack(pred_list, axis=0), axis=0)

    probs = _apply_confidence_temperature(probs, temperature=confidence_temperature)

    top_idx = int(np.argmax(probs))
    top_label = LABEL_NAMES[top_idx]

    response = {
        "predicted_label": top_label,
        "predicted_disease": DISEASE_NAMES[top_label],
        "confidence": float(probs[top_idx]),
        "tta_runs": int(tta_runs),
        "confidence_temperature": float(confidence_temperature),
    }

    if include_probabilities:
        response["probabilities"] = {
            LABEL_NAMES[i]: {
                "probability": float(probs[i]),
                "disease": DISEASE_NAMES[LABEL_NAMES[i]],
            }
            for i in range(len(probs))
        }

    if explain:
        try:
            last_conv = _last_conv_layer or find_last_conv_layer(model)
            heatmap = make_gradcam_heatmap(model_input, model, last_conv)
            response["gradcam_base64"] = _overlay_gradcam(model_input[0], heatmap)
        except Exception as exc:
            response["gradcam_error"] = str(exc)

    return response
