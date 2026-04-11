import os
import sys
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import (
    IMAGE_SIZE, LABEL_NAMES, DISEASE_NAMES, make_gradcam_heatmap,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered",
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_model.h5")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalise, and add batch dimension."""
    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def overlay_gradcam(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    superimposed = heatmap_color * alpha + img
    return np.clip(superimposed, 0, 1)


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🔬 Skin Cancer Detection")
st.markdown(
    "Upload a skin lesion image and the AI model will predict the type of skin "
    "cancer along with a confidence score and a **Grad‑CAM** heatmap showing "
    "which region the model focused on."
)

uploaded_file = st.file_uploader(
    "Upload a skin lesion image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if not os.path.isfile(MODEL_PATH):
        st.error(
            f"Model file not found at `{MODEL_PATH}`. "
            "Please train the model first by running `notebooks/training.ipynb`."
        )
    else:
        model = load_model()
        img_array = preprocess_image(image)

        with st.spinner("Classifying..."):
            preds = model.predict(img_array, verbose=0)[0]

        top_idx = int(np.argmax(preds))
        top_label = LABEL_NAMES[top_idx]
        confidence = float(preds[top_idx]) * 100

        st.success(f"**Prediction:** {DISEASE_NAMES[top_label]}")
        st.metric(label="Confidence", value=f"{confidence:.1f}%")

        # ── Show all class probabilities ─────────────────────────────────
        st.subheader("Class Probabilities")
        prob_data = {
            DISEASE_NAMES[LABEL_NAMES[i]]: float(preds[i]) * 100
            for i in range(len(preds))
        }
        st.bar_chart(prob_data)

        # ── Grad-CAM ─────────────────────────────────────────────────────
        st.subheader("Grad-CAM Explainability")
        try:
            last_conv_layer = None
            for layer in model.layers[::-1]:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break

            if last_conv_layer:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
                original_img = img_array[0]
                superimposed = overlay_gradcam(original_img, heatmap)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption="Input Image", use_container_width=True)
                with col2:
                    st.image(superimposed, caption="Grad-CAM Overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate Grad-CAM: {e}")
else:
    st.info("Please upload a skin lesion image to get started.")

st.markdown("---")
st.caption(
    "Built with TensorFlow & Streamlit · HAM10000 Dataset · "
    "For educational / research purposes only — not a medical device."
)
