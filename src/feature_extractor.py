"""
Deep CNN Fashion Feature Extraction Engine.
Module 7: Fashion Feature Extraction.

Extracts dense semantic visual embeddings (1280-dim) and convolutional feature maps
from fashion garments using transfer learning backbones (MobileNetV2 / ResNet50).
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

import numpy as np
from PIL import Image, ImageFile
import tensorflow as tf

from src.config import (
    TARGET_IMG_SIZE,
    FEATURE_BACKBONE,
    EMBEDDING_DIM,
    CLASSIFIER_MODEL_PATH,
    BATCH_SIZE,
)
from src.preprocessing import FashionImagePreprocessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FashionFeatureExtractor:
    """
    Extracts deep visual representation vectors (embeddings) and spatial feature maps
    from fashion product images for outfit similarity matching and recommendation.
    """

    def __init__(
        self,
        backbone: str = FEATURE_BACKBONE,
        weights: str = "imagenet",
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        use_letterbox: bool = True,
    ):
        """
        Initialize the CNN Feature Extractor.

        Args:
            backbone: 'mobilenet_v2' (1280-dim) or 'resnet50' (2048-dim).
            weights: 'imagenet' or path to a trained .keras model file.
            input_shape: (H, W, Channels) input dimension.
            use_letterbox: Preserves garment aspect ratio with letterboxing.
        """
        self.backbone_name = backbone.lower()
        self.input_shape = input_shape
        self.use_letterbox = use_letterbox

        # Step 1: Build base feature extraction network
        self.model, self.embedding_dim = self._build_feature_model(weights)

        # Step 2: Initialize preprocessor
        self.preprocessor = FashionImagePreprocessor(
            target_size=(input_shape[0], input_shape[1]),
            normalization_mode="scale_0_1",
            use_letterbox=use_letterbox,
        )

    def _build_feature_model(self, weights: str) -> Tuple[tf.keras.Model, int]:
        """Loads backbone and attaches GlobalAveragePooling2D output layer."""
        if self.backbone_name == "mobilenet_v2":
            if weights != "imagenet" and os.path.exists(weights):
                # Extract backbone from fine-tuned classifier
                full_model = tf.keras.models.load_model(weights)
                # Find the global_avg_pool layer output
                try:
                    gap_layer = full_model.get_layer("global_avg_pool")
                    feature_model = tf.keras.Model(
                        inputs=full_model.input,
                        outputs=gap_layer.output,
                        name="mobilenetv2_feature_extractor",
                    )
                    return feature_model, 1280
                except ValueError:
                    pass

            base = tf.keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet",
                pooling="avg",
            )
            return base, 1280

        elif self.backbone_name == "resnet50":
            base = tf.keras.applications.ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet",
                pooling="avg",
            )
            return base, 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

    def extract_features(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Extracts a 1D visual representation vector from a single fashion garment.

        Args:
            image_input: Image path, PIL Image, or numpy array.
            l2_normalize: If True, projects embedding onto unit hypersphere (L2-norm = 1.0).

        Returns:
            np.ndarray of shape (embedding_dim,) float32.
        """
        # Preprocess to (224, 224, 3) float32
        img_arr = self.preprocessor.preprocess(image_input, mode="scale_0_1")
        batch = np.expand_dims(img_arr, axis=0)

        # Forward pass
        feature_vec = self.model.predict(batch, verbose=0)[0].astype(np.float32)

        if l2_normalize:
            norm = np.linalg.norm(feature_vec)
            if norm > 1e-12:
                feature_vec = feature_vec / norm

        return feature_vec

    def extract_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = BATCH_SIZE,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Extracts features for a batch/collection of fashion garments.

        Returns:
            np.ndarray of shape (num_images, embedding_dim) float32.
        """
        all_embeddings = []

        for i in range(0, len(image_inputs), batch_size):
            chunk = image_inputs[i : i + batch_size]
            batch_arrs = []
            for item in chunk:
                arr = self.preprocessor.preprocess(item, mode="scale_0_1")
                batch_arrs.append(arr)

            batch_tensor = np.array(batch_arrs, dtype=np.float32)
            preds = self.model.predict(batch_tensor, verbose=0).astype(np.float32)

            if l2_normalize:
                norms = np.linalg.norm(preds, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                preds = preds / norms

            all_embeddings.append(preds)

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.empty((0, self.embedding_dim), dtype=np.float32)

    def extract_intermediate_feature_maps(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        layer_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extracts 2D spatial feature activation maps from an intermediate convolutional layer
        to visualize what patterns/textures the network is detecting.

        Returns:
            np.ndarray of shape (H_feat, W_feat, num_filters).
        """
        img_arr = self.preprocessor.preprocess(image_input, mode="scale_0_1")
        batch = np.expand_dims(img_arr, axis=0)

        # Identify intermediate layer
        target_layer = None
        if layer_name:
            target_layer = self.model.get_layer(layer_name)
        else:
            # Pick a representative deep convolutional layer
            for layer in reversed(self.model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    target_layer = layer
                    break

        if target_layer is None:
            # Default to penultimate layer before pooling
            target_layer = self.model.layers[-2]

        sub_model = tf.keras.Model(inputs=self.model.input, outputs=target_layer.output)
        feature_maps = sub_model.predict(batch, verbose=0)[0]
        return feature_maps

    @staticmethod
    def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes Cosine Similarity between two feature vectors.
        If both are L2-normalized, this is the dot product.
        """
        dot = float(np.dot(vec1, vec2))
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 1e-12 and norm2 > 1e-12:
            return float(dot / (norm1 * norm2))
        return 0.0
