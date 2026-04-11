import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


IMAGE_SIZE = (224, 224)
NUM_CLASSES = 7
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

DISEASE_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}


@dataclass
class BackboneSpec:
    name: str
    model_cls: object
    preprocess_input: object


BACKBONES = {
    "mobilenetv2": BackboneSpec(
        name="MobileNetV2",
        model_cls=tf.keras.applications.MobileNetV2,
        preprocess_input=tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "efficientnetb0": BackboneSpec(
        name="EfficientNetB0",
        model_cls=tf.keras.applications.EfficientNetB0,
        preprocess_input=tf.keras.applications.efficientnet.preprocess_input,
    ),
    "resnet50": BackboneSpec(
        name="ResNet50",
        model_cls=tf.keras.applications.ResNet50,
        preprocess_input=tf.keras.applications.resnet50.preprocess_input,
    ),
}


def infer_backbone_from_model_path(model_path: str) -> str:
    lower = os.path.basename(model_path).lower()
    for key in BACKBONES:
        if key in lower:
            return key
    return "efficientnetb0"


def get_backbone_custom_objects(backbone: str) -> dict[str, object]:
    backbone_key = backbone.lower()
    if backbone_key not in BACKBONES:
        return {}
    return {"preprocess_input": BACKBONES[backbone_key].preprocess_input}


def load_trained_model(model_path: str, backbone: str | None = None) -> tf.keras.Model:
    """Load a saved keras model and resolve preprocess_input for Lambda deserialization."""
    selected_backbone = backbone or infer_backbone_from_model_path(model_path)
    custom_objects = get_backbone_custom_objects(selected_backbone)
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load HAM10000 metadata and map textual labels to integer ids."""
    df = pd.read_csv(metadata_path)
    df = df[df["dx"].isin(LABEL_MAP)].copy()
    df["label"] = df["dx"].map(LABEL_MAP)
    return df


def resolve_image_paths(df: pd.DataFrame, image_dirs: list[str]) -> pd.DataFrame:
    """Attach image file paths and remove rows that do not have an image on disk."""
    image_lookup = {}
    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            continue
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(".jpg"):
                image_lookup[os.path.splitext(fname)[0]] = os.path.join(image_dir, fname)

    resolved = df.copy()
    resolved["image_path"] = resolved["image_id"].map(image_lookup)
    resolved = resolved[resolved["image_path"].notna()].copy()
    return resolved


def split_metadata_grouped(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by lesion_id to reduce patient/lesion leakage while preserving class ratios.
    This is done per class by splitting unique lesions into train/val/test buckets.
    """
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be in (0,1) and sum < 1")

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for class_name in LABEL_MAP:
        class_df = df[df["dx"] == class_name].copy()
        lesion_ids = class_df["lesion_id"].dropna().unique()
        rng.shuffle(lesion_ids)

        n_total = len(lesion_ids)
        if n_total == 0:
            continue

        n_test = max(1, int(round(n_total * test_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_test = min(n_test, n_total - 1)
        n_val = min(n_val, n_total - n_test - 1) if n_total > 2 else 0

        test_lesions = set(lesion_ids[:n_test])
        val_lesions = set(lesion_ids[n_test : n_test + n_val])
        train_lesions = set(lesion_ids[n_test + n_val :])

        if not train_lesions:
            train_lesions = set(lesion_ids[n_test:])
            val_lesions = set()

        test_parts.append(class_df[class_df["lesion_id"].isin(test_lesions)])
        val_parts.append(class_df[class_df["lesion_id"].isin(val_lesions)])
        train_parts.append(class_df[class_df["lesion_id"].isin(train_lesions)])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def split_metadata_image_stratified(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split image rows with stratification by class (higher headline accuracy, more leakage risk)."""
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be in (0,1) and sum < 1")

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["label"],
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: tuple[int, int]):
    bytestr = tf.io.read_file(path)
    image = tf.image.decode_jpeg(bytestr, channels=3)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    return image, label


def get_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        ],
        name="augment",
    )


def build_tf_dataset(
    dataframe: pd.DataFrame,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    paths = dataframe["image_path"].astype(str).values
    labels = dataframe["label"].astype(np.int32).values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(lambda p, y: _decode_and_resize(p, y, image_size), num_parallel_calls=AUTOTUNE)

    if augment:
        aug = get_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def compute_balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}


def make_sparse_categorical_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)

        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        loss = -alpha * focal_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    return focal_loss


def build_transfer_model(
    backbone: str = "efficientnetb0",
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.35,
    lr: float = 1e-3,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build and compile a transfer-learning classifier and return (model, base_model)."""
    backbone_key = backbone.lower()
    if backbone_key not in BACKBONES:
        valid = ", ".join(BACKBONES)
        raise ValueError(f"Unknown backbone '{backbone}'. Valid options: {valid}")

    spec = BACKBONES[backbone_key]
    base_model = spec.model_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    x = tf.keras.layers.Lambda(spec.preprocess_input, name="preprocess")(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"ham10000_{spec.name.lower()}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
    )
    return model, base_model


def unfreeze_last_layers(base_model: tf.keras.Model, trainable_layers: int = 30) -> None:
    base_model.trainable = True
    if trainable_layers <= 0:
        return
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False


def prepare_inference_image(image: Image.Image, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    resized = image.convert("RGB").resize(image_size)
    arr = np.array(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            # For nested backbone models (e.g., EfficientNet functional block),
            # use the top-level layer name so tensors stay connected in the outer graph.
            for inner in reversed(layer.layers):
                if isinstance(inner, tf.keras.layers.Conv2D):
                    return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
) -> np.ndarray:
    conv_output_tensor = None
    try:
        conv_output_tensor = model.get_layer(last_conv_layer_name).output
    except ValueError:
        for layer in model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    conv_output_tensor = layer.get_layer(last_conv_layer_name).output
                    break
                except ValueError:
                    continue

    if conv_output_tensor is None:
        raise ValueError(f"Could not locate conv layer '{last_conv_layer_name}' in model")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[conv_output_tensor, model.output],
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            top_class = tf.argmax(predictions[0])
            loss = predictions[:, top_class]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        # Fallback for Keras graph issues with nested loaded models.
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor, training=False)
            top_class = tf.argmax(predictions[0])
            loss = predictions[:, top_class]

        input_grads = tape.gradient(loss, img_tensor)[0]
        heatmap = tf.reduce_mean(tf.math.abs(input_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
