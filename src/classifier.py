"""
CNN Clothing Category Classification Model & Training Pipeline.
Module 6: Clothing Category Classification.

Provides transfer learning architecture (MobileNetV2), training routine with callbacks,
evaluation metrics, confusion matrix generation, and the ClothingClassifier inference engine.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows if supported
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.config import (
    TARGET_IMG_SIZE,
    NUM_CLASSES,
    CLOTHING_CATEGORIES,
    CATEGORY_TO_IDX,
    IDX_TO_CATEGORY,
    CLASSIFIER_MODEL_PATH,
    CONFUSION_MATRIX_PNG,
    BATCH_SIZE,
)
from src.preprocessing import FashionImagePreprocessor, create_tf_dataset


def build_category_classifier(
    backbone: str = "mobilenet_v2",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Constructs a transfer learning CNN classifier with MobileNetV2 backbone
    and custom classification head for clothing category prediction.
    """
    if backbone.lower() == "mobilenet_v2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone.lower() == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose 'mobilenet_v2' or 'resnet50'.")

    # Freeze base model weights for initial transfer learning stage
    base_model.trainable = not freeze_backbone

    # Build model using functional API
    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout_35")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="category_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{backbone}_clothing_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )

    return model


def train_category_classifier(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 3,
    checkpoint_path: Union[str, Path] = CLASSIFIER_MODEL_PATH,
    early_stopping_patience: int = 3,
) -> tf.keras.callbacks.History:
    """
    Trains the clothing classifier with ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau callbacks.
    """
    chk_p = Path(checkpoint_path)
    chk_p.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(chk_p),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n[*] Commencing CNN Classifier Training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history


def evaluate_classifier(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    test_df: pd.DataFrame,
    output_png: Union[str, Path] = CONFUSION_MATRIX_PNG,
) -> Dict[str, Any]:
    """
    Evaluates classifier on test dataset, generating Accuracy, Top-3 Accuracy,
    Classification Report, and saving Confusion Matrix heatmap.
    """
    eval_results = model.evaluate(test_ds, verbose=1)
    loss, accuracy, top3_acc = eval_results[0], eval_results[1], eval_results[2]

    # Predict test items to compute confusion matrix
    print("[*] Generating test set predictions for confusion matrix...")
    y_true_indices = test_df["category_idx"].values[:len(test_ds) * BATCH_SIZE]
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    # Slice y_true to match actual predictions length
    actual_len = len(y_pred_indices)
    y_true_slice = y_true_indices[:actual_len]

    # Compute confusion matrix
    cm = confusion_matrix(y_true_slice, y_pred_indices, labels=list(range(NUM_CLASSES)))

    # Plot and save heatmap
    out_p = Path(output_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLOTHING_CATEGORIES,
        yticklabels=CLOTHING_CATEGORIES,
    )
    plt.title("Clothing Category Classifier - Test Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=11)
    plt.ylabel("True Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_p, dpi=200)
    plt.close()
    print(f"[*] Saved confusion matrix plot to: {out_p}")

    report = classification_report(
        y_true_slice,
        y_pred_indices,
        target_names=CLOTHING_CATEGORIES,
        zero_division=0,
        output_dict=True,
    )

    return {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "test_top3_accuracy": float(top3_acc),
        "classification_report": report,
        "confusion_matrix_path": str(out_p),
    }


class ClothingClassifier:
    """
    Inference interface for predicting clothing categories from input images.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        path = Path(model_path) if model_path else CLASSIFIER_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Trained classifier model not found at: {path}. Run train_classifier.py first.")
        self.model = tf.keras.models.load_model(str(path))
        self.preprocessor = FashionImagePreprocessor(
            target_size=TARGET_IMG_SIZE,
            normalization_mode="scale_0_1",
            use_letterbox=True,
        )

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Classifies an input fashion image.

        Returns:
            Dictionary with 'top_category', 'confidence', and list of 'top_k' predictions.
        """
        # Preprocess image to (224, 224, 3) float32
        img_arr = self.preprocessor.preprocess(image_input, mode="scale_0_1")
        # Expand batch dimension (1, 224, 224, 3)
        batch = np.expand_dims(img_arr, axis=0)

        probs = self.model.predict(batch, verbose=0)[0]
        top_indices = np.argsort(-probs)[:top_k]

        top_k_list = []
        for idx in top_indices:
            top_k_list.append({
                "category": IDX_TO_CATEGORY.get(int(idx), "Unknown"),
                "probability": round(float(probs[idx]), 4),
            })

        return {
            "top_category": top_k_list[0]["category"],
            "confidence": top_k_list[0]["probability"],
            "top_k": top_k_list,
        }
