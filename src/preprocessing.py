"""
Fashion Image Preprocessing and Batch Pipeline Module.
Module 4: Fashion Image Preprocessing.

Provides aspect-ratio preserving resizing, multi-mode normalization,
data augmentation, stratified dataset splitting, and high-performance tf.data streaming.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

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
import pandas as pd
from PIL import Image, ImageOps, ImageFile
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.config import (
    TARGET_IMG_SIZE,
    DEFAULT_NORMALIZATION,
    NORMALIZATION_MODES,
    SPLITS_CSV,
    CLEANED_METADATA_CSV,
    CLOTHING_CATEGORIES,
    CATEGORY_TO_IDX,
    NUM_CLASSES,
    BATCH_SIZE,
    RANDOM_STATE,
)

# Allow truncated image files to load safely without throwing exceptions
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ImageNet statistics for standardization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FashionImagePreprocessor:
    """
    Standard image preprocessor for fashion garments.
    Supports letterboxing (aspect-ratio padding), direct interpolation,
    and multiple normalization protocols.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = TARGET_IMG_SIZE,
        normalization_mode: Literal["scale_0_1", "tf_minus1_to_1", "imagenet"] = DEFAULT_NORMALIZATION,
        use_letterbox: bool = True,
        pad_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.target_size = target_size
        self.normalization_mode = normalization_mode
        self.use_letterbox = use_letterbox
        self.pad_color = pad_color

    def load_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Safely loads an image from file path or converts existing array/Pillow object to 3-channel RGB.
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image not found at path: {path_str}")
            with Image.open(path_str) as img:
                return img.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                # Grayscale to RGB
                return Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[-1] == 4:
                # RGBA to RGB
                return Image.fromarray(image_input).convert("RGB")
            return Image.fromarray(image_input.astype(np.uint8))
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def resize_and_pad(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        pad_color: Optional[Tuple[int, int, int]] = None,
    ) -> Image.Image:
        """
        Resizes image while preserving original aspect ratio, centering it on a padded canvas.
        Essential for fashion apparel so garments aren't artificially stretched.
        """
        target_w, target_h = target_size or self.target_size
        color = pad_color or self.pad_color

        orig_w, orig_h = image.size
        if orig_w == 0 or orig_h == 0:
            return Image.new("RGB", (target_w, target_h), color)

        # Determine scaling factor
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        # High-quality resize
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Create padded canvas and paste centered
        canvas = Image.new("RGB", (target_w, target_h), color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))

        return canvas

    def direct_resize(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Standard direct resize without padding."""
        target_w, target_h = target_size or self.target_size
        return image.resize((target_w, target_h), Image.Resampling.BILINEAR)

    def normalize(
        self,
        img_array: np.ndarray,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Scales/standardizes pixel values to target numerical distribution.

        Modes:
            - 'scale_0_1': Scales [0, 255] to [0.0, 1.0].
            - 'tf_minus1_to_1': Scales [0, 255] to [-1.0, 1.0] (Inception/MobileNet).
            - 'imagenet': Normalizes with ImageNet dataset mean and std.
        """
        mode = mode or self.normalization_mode
        arr = img_array.astype(np.float32)

        if mode == "scale_0_1":
            return arr / 255.0
        elif mode == "tf_minus1_to_1":
            return (arr / 127.5) - 1.0
        elif mode == "imagenet":
            scaled = arr / 255.0
            return (scaled - IMAGENET_MEAN) / IMAGENET_STD
        else:
            raise ValueError(f"Unknown normalization mode '{mode}'. Choose from {NORMALIZATION_MODES}")

    def denormalize(
        self,
        norm_array: np.ndarray,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Inverts normalization to restore uint8 [0, 255] RGB for visual display."""
        mode = mode or self.normalization_mode
        arr = norm_array.copy().astype(np.float32)

        if mode == "scale_0_1":
            out = arr * 255.0
        elif mode == "tf_minus1_to_1":
            out = (arr + 1.0) * 127.5
        elif mode == "imagenet":
            out = (arr * IMAGENET_STD) + IMAGENET_MEAN
            out = out * 255.0
        else:
            out = arr

        return np.clip(out, 0, 255).astype(np.uint8)

    def preprocess(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        mode: Optional[str] = None,
        use_letterbox: Optional[bool] = None,
    ) -> np.ndarray:
        """
        End-to-end preprocessing:
        1. Loads & ensures 3-channel RGB.
        2. Resizes with aspect-ratio letterboxing or direct scaling.
        3. Normalizes pixel intensities.
        Returns: np.ndarray of shape (H, W, 3) float32.
        """
        target = target_size or self.target_size
        norm_mode = mode or self.normalization_mode
        letterbox = self.use_letterbox if use_letterbox is None else use_letterbox

        img = self.load_image(image_input)

        if letterbox:
            img = self.resize_and_pad(img, target)
        else:
            img = self.direct_resize(img, target)

        arr = np.array(img, dtype=np.float32)
        return self.normalize(arr, norm_mode)


def get_data_augmentation_pipeline(
    rotation_factor: float = 0.05,
    zoom_factor: float = 0.08,
    flip_mode: str = "horizontal",
) -> tf.keras.Sequential:
    """
    Creates a Keras Sequential layer pipeline for on-the-fly training augmentation.
    Helps CNN models generalize across slight garment rotations, zooms, and flips.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(flip_mode),
            tf.keras.layers.RandomRotation(rotation_factor),
            tf.keras.layers.RandomZoom((-zoom_factor, zoom_factor)),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="fashion_augmentation",
    )


def create_stratified_splits(
    df: Optional[pd.DataFrame] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = RANDOM_STATE,
    output_path: Optional[Union[str, Path]] = SPLITS_CSV,
) -> pd.DataFrame:
    """
    Splits the fashion dataset catalog into stratified Train, Validation, and Test sets
    preserving category class proportions across each split.
    """
    if df is None:
        csv_path = Path(metadata_path) if metadata_path else CLEANED_METADATA_CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"Clean metadata CSV not found at: {csv_path}")
        df = pd.read_csv(csv_path)

    # Ensure canonical_category column exists
    if "canonical_category" not in df.columns:
        raise ValueError("DataFrame must contain 'canonical_category' column for stratified splitting.")

    # Remove any categories with fewer than 3 samples (stratified split requirement)
    cat_counts = df["canonical_category"].value_counts()
    valid_cats = cat_counts[cat_counts >= 3].index
    df_valid = df[df["canonical_category"].isin(valid_cats)].copy()

    # Step 1: Split off Test set
    test_size = test_ratio / (train_ratio + val_ratio + test_ratio)
    train_val_df, test_df = train_test_split(
        df_valid,
        test_size=test_size,
        stratify=df_valid["canonical_category"],
        random_state=random_state,
    )

    # Step 2: Split Train and Validation
    val_relative_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["canonical_category"],
        random_state=random_state,
    )

    # Mark split labels
    df_valid.loc[train_df.index, "split"] = "train"
    df_valid.loc[val_df.index, "split"] = "val"
    df_valid.loc[test_df.index, "split"] = "test"

    # Add numeric category index for neural network targets
    df_valid["category_idx"] = df_valid["canonical_category"].map(CATEGORY_TO_IDX).fillna(-1).astype(int)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_valid.to_csv(out_p, index=False)
        print(f"[Stratified Splits] Saved {len(df_valid)} split records to: {out_p}")
        print(f"  * Train set: {len(train_df):,} items ({len(train_df)/len(df_valid)*100:.1f}%)")
        print(f"  * Val set  : {len(val_df):,} items ({len(val_df)/len(df_valid)*100:.1f}%)")
        print(f"  * Test set : {len(test_df):,} items ({len(test_df)/len(df_valid)*100:.1f}%)")

    return df_valid


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    is_training: bool = True,
    target_size: Tuple[int, int] = TARGET_IMG_SIZE,
    mode: str = DEFAULT_NORMALIZATION,
    augment: bool = False,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """
    Constructs a high-performance tf.data streaming pipeline from DataFrame image paths.
    Processes images asynchronously with parallel disk I/O, scaling, and prefetching.
    Guarantees no RAM explosion with large catalogs.
    """
    if "image_path" not in df.columns or "category_idx" not in df.columns:
        raise ValueError("DataFrame must contain 'image_path' and 'category_idx' columns.")

    image_paths = df["image_path"].values.astype(str)
    category_indices = df["category_idx"].values.astype(np.int32)
    one_hot_labels = tf.keras.utils.to_categorical(category_indices, num_classes=NUM_CLASSES)

    target_h, target_w = target_size

    def _parse_and_preprocess(img_path, label):
        # Read file from disk
        raw = tf.io.read_file(img_path)
        # Decode JPEG to uint8 tensor (3 channels)
        img = tf.io.decode_jpeg(raw, channels=3)
        # Resize using bilinear interpolation
        img = tf.image.resize(img, [target_h, target_w])

        # Normalize
        if mode == "scale_0_1":
            img = img / 255.0
        elif mode == "tf_minus1_to_1":
            img = (img / 127.5) - 1.0
        elif mode == "imagenet":
            img = (img / 255.0 - tf.constant(IMAGENET_MEAN)) / tf.constant(IMAGENET_STD)

        return img, label

    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, one_hot_labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(df), shuffle_buffer), reshuffle_each_iteration=True)

    # Parallel mapping with AUTOTUNE
    dataset = dataset.map(_parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Optional training data augmentation
    if is_training and augment:
        aug_layer = get_data_augmentation_pipeline()
        dataset = dataset.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Batching and prefetching
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
