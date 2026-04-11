"""Train a robust HAM10000 classifier with transfer learning and evaluation artifacts."""

import argparse
import json
import os
import random
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from utils.preprocessing import (
    BACKBONES,
    BATCH_SIZE,
    DISEASE_NAMES,
    IMAGE_SIZE,
    LABEL_NAMES,
    build_tf_dataset,
    build_transfer_model,
    compute_balanced_class_weights,
    load_metadata,
    make_sparse_categorical_focal_loss,
    resolve_image_paths,
    split_metadata_image_stratified,
    split_metadata_grouped,
    unfreeze_last_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HAM10000 skin cancer detector")
    parser.add_argument("--backbone", type=str, default="efficientnetb0", choices=list(BACKBONES.keys()))
    parser.add_argument("--epochs-frozen", type=int, default=18)
    parser.add_argument("--epochs-finetune", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune-layers", type=int, default=120)
    parser.add_argument("--split-strategy", choices=["grouped", "image-stratified"], default="grouped")
    parser.add_argument("--loss-type", choices=["crossentropy", "focal"], default="crossentropy")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--lr-frozen", type=float, default=7e-4)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--oversample-minority", action="store_true")
    parser.add_argument(
        "--oversample-target",
        choices=["median", "max"],
        default="median",
        help="Target class size for oversampling minority classes in training split",
    )
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_history_plot(history_all: dict[str, list[float]], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_all["accuracy"], label="Train")
    axes[0].plot(history_all["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_all["loss"], label="Train")
    axes[1].plot(history_all["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=160)
    plt.close(fig)


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def compile_for_training(
    model: tf.keras.Model,
    lr: float,
    loss_type: str,
    label_smoothing: float,
    focal_gamma: float,
    focal_alpha: float,
) -> None:
    if loss_type == "focal":
        loss_fn = make_sparse_categorical_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
    else:
        if label_smoothing > 0:
            print(
                "[warning] label_smoothing is ignored for sparse categorical crossentropy. "
                "Use focal loss or convert labels to one-hot for smoothing."
            )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
    )


def oversample_minority_classes(
    train_df: pd.DataFrame,
    seed: int,
    target: str = "median",
) -> pd.DataFrame:
    class_counts = train_df["label"].value_counts().sort_index()
    if class_counts.empty:
        return train_df

    target_size = int(class_counts.max() if target == "max" else class_counts.median())
    target_size = max(target_size, int(class_counts.min()))

    rebalanced_parts = []
    for _, group in train_df.groupby("label"):
        n = len(group)
        if n < target_size:
            sampled = group.sample(n=target_size, replace=True, random_state=seed)
            rebalanced_parts.append(sampled)
        else:
            rebalanced_parts.append(group)

    return pd.concat(rebalanced_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "dataset")
    image_dirs = [
        os.path.join(data_dir, "HAM10000_images_part_1"),
        os.path.join(data_dir, "HAM10000_images_part_2"),
    ]
    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"TensorFlow {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Backbone: {args.backbone}")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Loss type: {args.loss_type}")

    print("\n[1/6] Loading metadata and image paths")
    meta = load_metadata(metadata_path)
    meta = resolve_image_paths(meta, image_dirs)
    print(f"Usable samples: {len(meta)}")
    print(meta["dx"].value_counts().to_string())

    print("\n[2/6] Train/val/test split")
    if args.split_strategy == "image-stratified":
        train_df, val_df, test_df = split_metadata_image_stratified(meta, seed=args.seed)
    else:
        train_df, val_df, test_df = split_metadata_grouped(meta, seed=args.seed)

    if args.oversample_minority:
        before_counts = train_df["dx"].value_counts().to_dict()
        train_df = oversample_minority_classes(train_df, seed=args.seed, target=args.oversample_target)
        after_counts = train_df["dx"].value_counts().to_dict()
        print("Oversampling enabled")
        print("Class counts before:", before_counts)
        print("Class counts after:", after_counts)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    print("\n[3/6] Building tf.data datasets")
    train_ds = build_tf_dataset(
        train_df,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        seed=args.seed,
    )
    val_ds = build_tf_dataset(val_df, image_size=IMAGE_SIZE, batch_size=args.batch_size)
    test_ds = build_tf_dataset(test_df, image_size=IMAGE_SIZE, batch_size=args.batch_size)

    class_weights = compute_balanced_class_weights(y_train)
    print("Class weights:")
    for k, v in class_weights.items():
        print(f"  {LABEL_NAMES[k]}: {v:.3f}")

    print("\n[4/6] Building model")
    model, base_model = build_transfer_model(backbone=args.backbone, image_size=IMAGE_SIZE, lr=args.lr_frozen)
    compile_for_training(
        model,
        lr=args.lr_frozen,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )
    print(model.summary())

    run_name = f"{args.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_path = os.path.join(models_dir, f"{run_name}.best.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ]

    print("\n[5/6] Training phase 1 (frozen backbone)")
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_frozen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("Training phase 2 (fine-tuning)")
    unfreeze_last_layers(base_model, trainable_layers=args.finetune_layers)
    compile_for_training(
        model,
        lr=args.lr_finetune,
        loss_type=args.loss_type,
        label_smoothing=max(0.0, args.label_smoothing * 0.5),
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[6/6] Evaluating and exporting artifacts")
    test_metrics = model.evaluate(test_ds, verbose=0)
    metric_names = model.metrics_names
    metric_map = {name: float(val) for name, val in zip(metric_names, test_metrics)}
    print("Test metrics:", metric_map)

    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    target_names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4,
        output_dict=True,
    )

    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=target_names, yticklabels=target_names)
    plt.title("HAM10000 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    history_all = merge_histories(history_frozen, history_finetune)
    save_history_plot(history_all, models_dir)

    model_path = os.path.join(models_dir, "cnn_model.keras")
    run_model_path = os.path.join(models_dir, f"{run_name}.final.keras")
    model.save(model_path)
    model.save(run_model_path)

    metadata = {
        "model_path": model_path,
        "backbone": args.backbone,
        "image_size": list(IMAGE_SIZE),
        "split_strategy": args.split_strategy,
        "loss_type": args.loss_type,
        "label_names": LABEL_NAMES,
        "disease_names": DISEASE_NAMES,
        "metrics": metric_map,
        "classification_report": report,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(models_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Run model saved to: {run_model_path}")
    print(f"Best checkpoint: {checkpoint_path}")
    print("Artifacts: confusion_matrix.png, training_curves.png, model_metadata.json")


if __name__ == "__main__":
    main()
