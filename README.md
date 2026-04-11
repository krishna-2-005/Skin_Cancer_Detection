# Skin Cancer Detection (HAM10000)

End-to-end skin lesion classification project using TensorFlow transfer learning, FastAPI inference, and a React frontend with Grad-CAM explainability.

This repository is intended for education and research workflows.

## Clinical Disclaimer

This software is not a medical device and must not be used for clinical diagnosis or treatment decisions. Model predictions are probabilistic and can be wrong. Final diagnosis must always be made by a qualified dermatologist.

## 1. Project Overview

This project provides a full machine learning lifecycle for skin lesion analysis:

1. Train deep learning models on HAM10000.
2. Evaluate and compare saved checkpoints.
3. Serve the best model through an API.
4. Run a browser-based UI for image upload and prediction.
5. Provide Grad-CAM visual explanations.

### Objective

Build a practical, reproducible, leakage-aware research pipeline for 7-class lesion classification that can be demonstrated end-to-end.

### Expected Outcome

1. Trained checkpoint files in `models/`.
2. Evaluation output (accuracy, per-class metrics, confusion matrix, curves).
3. FastAPI endpoint for inference.
4. React interface for usability and visualization.

## 2. Class Labels

| Label | Disease Name |
|---|---|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

## 3. Repository Structure

```text
skin-cancer-detection/
  api/
    main.py
  app/
    streamlit_app.py
  dataset/
    HAM10000_metadata.csv
    HAM10000_images_part_1/
    HAM10000_images_part_2/
  frontend/
    index.html
    src/
      App.jsx
      styles.css
  models/
    *.best.keras
    *.final.keras
  notebooks/
    training.ipynb
  utils/
    preprocessing.py
  evaluate_checkpoint.py
  train.py
  requirements.txt
```

## 4. Tools and Technologies

### Core ML

1. TensorFlow / Keras for transfer learning and inference.
2. NumPy / Pandas for data manipulation.
3. scikit-learn for evaluation/reporting support.

### Image and Explainability

1. Pillow for image loading.
2. OpenCV for Grad-CAM overlay processing.

### Serving and UI

1. FastAPI + Uvicorn for inference API.
2. React 18 + Vite 5 + Axios for frontend.

### Dataset

1. HAM10000 metadata and image folders.

## 5. Environment Setup

### Requirements

1. Python 3.10 or 3.11 recommended.
2. Node.js 18+ recommended.
3. Windows, macOS, or Linux.

### Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend setup

```bash
cd frontend
npm install
```

## 6. Dataset Setup

Required paths:

1. `dataset/HAM10000_metadata.csv`
2. `dataset/HAM10000_images_part_1/*.jpg`
3. `dataset/HAM10000_images_part_2/*.jpg`

Training and evaluation scripts read metadata, map `dx` labels to class IDs, and resolve image files from both image folders.

## 7. Complete Training Workflow (Start to End)

1. Define task as 7-class lesion classification.
2. Load metadata and resolve image paths.
3. Validate records and class mapping.
4. Split data:
   1. Default: grouped split by `lesion_id` (lower leakage risk).
   2. Optional: image-stratified split.
5. Build preprocessing and tf.data pipelines.
6. Choose backbone (`efficientnetb0`, `mobilenetv2`, `resnet50`).
7. Train phase 1 (frozen backbone, head training).
8. Train phase 2 (fine-tune selected backbone layers).
9. Apply class weights and optional oversampling.
10. Evaluate test performance and class-wise metrics.
11. Save artifacts and checkpoints in `models/`.
12. Re-evaluate any checkpoint with `evaluate_checkpoint.py`.

### Common training command

```bash
python train.py --backbone efficientnetb0 --epochs-frozen 18 --epochs-finetune 24 --mixed-precision
```

### Other examples

```bash
python train.py --backbone efficientnetb0 --loss-type focal --focal-gamma 2.0 --focal-alpha 0.25
python train.py --backbone mobilenetv2 --epochs-frozen 10 --epochs-finetune 12
```

### Useful `train.py` arguments

| Argument | Description | Default |
|---|---|---|
| --backbone | efficientnetb0, mobilenetv2, resnet50 | efficientnetb0 |
| --epochs-frozen | epochs for phase 1 | 18 |
| --epochs-finetune | epochs for phase 2 | 24 |
| --batch-size | batch size | 32 |
| --split-strategy | grouped or image-stratified | grouped |
| --finetune-layers | trainable tail layers in phase 2 | 120 |
| --loss-type | crossentropy or focal | crossentropy |
| --lr-frozen | phase 1 learning rate | 7e-4 |
| --lr-finetune | phase 2 learning rate | 1e-5 |
| --mixed-precision | enable mixed_float16 | false |
| --oversample-minority | oversample minority classes | false |
| --oversample-target | oversampling target: median or max | median |
| --seed | reproducibility seed | 42 |

## 8. Model Order Used in This Project

Observed training/checkpoint chronology in `models/`:

1. `efficientnetb0_20260312_202959.best.keras`
2. `efficientnetb0_20260312_210821.best.keras`
3. `mobilenetv2_20260312_221130.best.keras`
4. `efficientnetb0_20260401_093605.best.keras`
5. `efficientnetb0_20260401_094357.best.keras` (latest active checkpoint)

## 9. Evaluation Workflow

Evaluate latest `.best.keras`:

```bash
python evaluate_checkpoint.py
```

Evaluate a specific checkpoint:

```bash
python evaluate_checkpoint.py --model-path models/efficientnetb0_YYYYMMDD_HHMMSS.best.keras
```

Evaluate with image-stratified split:

```bash
python evaluate_checkpoint.py --split-strategy image-stratified
```

## 10. Final Results Snapshot (April 2026)

### Overall evaluated model accuracy

Using grouped split on checkpoint `models/efficientnetb0_20260401_094357.best.keras`:

1. Accuracy: `0.7236` (72.36%).
2. Test samples: `1523`.

### Sample inference outputs from UI

| Sample Image | Predicted Disease | Label | Single-image Confidence |
|---|---|---|---|
| ISIC_0029295.jpg | Melanocytic Nevi | nv | 96.6% |
| ISIC_0029105.jpg | Melanocytic Nevi | nv | 99.3% |

Note: single-image confidence is not the same as dataset-level accuracy.

## 11. API Service

### Start API

Default:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

If port 8000 is occupied locally:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

### Model loading behavior

On startup or reload:

1. Use `models/cnn_model.keras` if present.
2. Else select best available `.best.keras`.
3. Family preference: efficientnet, then resnet, then mobilenet.
4. Within family, latest file mtime wins.

### Endpoints

1. `GET /health` for status/model path.
2. `GET /classes` for label and disease mapping.
3. `POST /reload-model` to reload latest resolvable model.
4. `POST /predict` for image inference.

### `/predict` parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| explain | bool | true | Include Grad-CAM result |
| confidence_temperature | float | 0.78 | Probability sharpening/smoothing |
| include_probabilities | bool | false | Return full class distribution |

Backend currently enforces fixed TTA runs (`8`) internally.

Supported input content types:

1. `image/jpeg`
2. `image/jpg`
3. `image/png`
4. `image/webp`

### Example prediction request (PowerShell)

```powershell
curl.exe -X POST "http://127.0.0.1:8001/predict?explain=true&include_probabilities=true" `
  -F "file=@dataset/HAM10000_images_part_1/ISIC_0027419.jpg"
```

### Typical response fields

1. `predicted_label`
2. `predicted_disease`
3. `confidence`
4. `tta_runs`
5. `confidence_temperature`
6. `probabilities` (optional)
7. `gradcam_base64` (optional)
8. `gradcam_error` (optional)

## 12. Frontend

### Run

```bash
cd frontend
npm run dev
```

### Build

```bash
npm run build
npm run preview
```

### API base URL

Set before starting frontend when using non-default API port:

```bash
VITE_API_BASE=http://127.0.0.1:8001
```

### Current UI behavior

1. Shows API/model status, model path, and prediction results.
2. Shows upload preview, prediction confidence, class breakdown (optional), and Grad-CAM.
3. Confidence boost control is hidden in UI and fixed in request logic.
4. TTA control is hidden in UI and fixed to `8`.

## 13. Architecture and Data Flow

### Architecture

1. Frontend layer: React/Vite UI (`frontend/src/App.jsx`).
2. API layer: FastAPI inference service (`api/main.py`).
3. Model layer: TensorFlow model checkpoints in `models/`.
4. Data layer: file-based dataset and artifact storage.

### Data flow

1. Input image uploaded in frontend.
2. Frontend sends multipart request to API.
3. API validates file and preprocesses image tensor.
4. Model runs prediction with fixed TTA.
5. API post-processes probabilities and optional Grad-CAM.
6. JSON response returned to frontend.
7. Frontend renders prediction and visual explanation.

## 14. Troubleshooting

### API returns 404 on expected routes

1. Another project may already use port 8000.
2. Start this API on 8001 and set `VITE_API_BASE` accordingly.
3. Verify `http://127.0.0.1:8001/health` responds.

### Model not loading

1. Ensure at least one model exists:
   1. `models/cnn_model.keras`
   2. `models/*.best.keras`
2. Call `POST /reload-model` after adding checkpoints.

### Frontend cannot connect

1. Confirm API host/port.
2. Confirm `VITE_API_BASE` matches API URL.

### Grad-CAM missing

1. Prediction can still succeed if Grad-CAM fails.
2. Check `gradcam_error` in API response.

## 15. Security and Deployment Notes

1. CORS is currently open (`allow_origins=["*"]`). Restrict for production.
2. Add authentication, validation, and rate limiting for public deployment.
3. Add request logging and monitoring before production use.

## 16. Streamlit (Optional)

Path: `app/streamlit_app.py`

Compatibility note:

1. Streamlit app may expect `.h5` by default.
2. Training currently outputs `.keras` checkpoints.

## 17. Suggested End-to-End Run Order

1. Place HAM10000 files in `dataset/`.
2. Train model with grouped split.
3. Evaluate checkpoint and confirm metrics.
4. Start API and verify `/health`.
5. Start frontend and run predictions.
6. Reload latest checkpoint when retraining.

## Acknowledgments

1. HAM10000 dataset creators and contributors.
2. TensorFlow, FastAPI, and React open-source communities.

## License and Usage

Use this project for education, experimentation, and research. Ensure compliance with HAM10000 licensing and attribution requirements before redistribution or commercial use.
