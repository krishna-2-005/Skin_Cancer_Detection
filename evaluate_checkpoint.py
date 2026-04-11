import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from utils.preprocessing import IMAGE_SIZE, build_tf_dataset, load_metadata, resolve_image_paths, split_metadata_grouped, split_metadata_image_stratified
from utils.preprocessing import load_trained_model


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

parser = argparse.ArgumentParser(description="Evaluate a HAM10000 checkpoint on the test split")
parser.add_argument("--model-path", default=None, help="Path to a .keras checkpoint; defaults to latest .best.keras")
parser.add_argument(
    "--split-strategy",
    choices=["grouped", "image-stratified"],
    default="grouped",
    help="Dataset split strategy used for evaluation",
)
args = parser.parse_args()

if args.model_path:
    MODEL_PATH = os.path.abspath(args.model_path)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
else:
    best_candidates = [
        os.path.join(MODELS_DIR, name)
        for name in os.listdir(MODELS_DIR)
        if name.endswith(".best.keras")
    ]
    if not best_candidates:
        raise FileNotFoundError("No .best.keras checkpoint found in models directory")
    MODEL_PATH = max(best_candidates, key=os.path.getmtime)

print(f"Evaluating checkpoint: {MODEL_PATH}")
print(f"Split strategy: {args.split_strategy}")

image_dirs = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]
metadata_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

meta = load_metadata(metadata_path)
meta = resolve_image_paths(meta, image_dirs)
if args.split_strategy == "image-stratified":
    _, _, test_df = split_metadata_image_stratified(meta, seed=42)
else:
    _, _, test_df = split_metadata_grouped(meta, seed=42)

y_test = test_df["label"].to_numpy(dtype=np.int32)
test_ds = build_tf_dataset(test_df, image_size=IMAGE_SIZE, batch_size=32)

model = load_trained_model(MODEL_PATH)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
)
metrics = model.evaluate(test_ds, verbose=0)
print("metrics_names:", model.metrics_names)
print("metrics:", metrics)

probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(probs, axis=1)
print(classification_report(y_test, y_pred, digits=4))
