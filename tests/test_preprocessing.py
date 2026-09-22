"""
Unit tests for Module 4: Fashion Image Preprocessing.
"""
import unittest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing import (
    FashionImagePreprocessor,
    create_stratified_splits,
    create_tf_dataset,
    get_data_augmentation_pipeline,
)
from src.config import NUM_CLASSES, CLOTHING_CATEGORIES


class TestFashionPreprocessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.preprocessor = FashionImagePreprocessor(target_size=(224, 224))

        # Create synthetic images of different aspect ratios
        self.img_square = Image.new("RGB", (300, 300), color=(100, 150, 200))
        self.img_tall = Image.new("RGB", (100, 400), color=(200, 50, 50))
        self.img_wide = Image.new("RGB", (500, 150), color=(50, 200, 50))
        self.img_rgba = Image.new("RGBA", (200, 200), color=(50, 50, 50, 128))
        self.img_gray = Image.new("L", (150, 150), color=120)

        # Save sample images for file path testing
        self.tall_path = Path(self.test_dir) / "tall.jpg"
        self.img_tall.save(self.tall_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_letterbox_padding_preserves_shape(self):
        # Tall image (e.g. dress or trousers)
        padded_tall = self.preprocessor.resize_and_pad(self.img_tall, target_size=(224, 224))
        self.assertEqual(padded_tall.size, (224, 224))

        # Wide image (e.g. shoes or belt)
        padded_wide = self.preprocessor.resize_and_pad(self.img_wide, target_size=(224, 224))
        self.assertEqual(padded_wide.size, (224, 224))

    def test_channel_conversions(self):
        # RGBA to RGB
        rgb_from_rgba = self.preprocessor.load_image(self.img_rgba)
        self.assertEqual(rgb_from_rgba.mode, "RGB")

        # Grayscale to RGB
        rgb_from_gray = self.preprocessor.load_image(self.img_gray)
        self.assertEqual(rgb_from_gray.mode, "RGB")

    def test_normalization_modes(self):
        raw_arr = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

        # Scale [0, 1]
        norm_0_1 = self.preprocessor.normalize(raw_arr, mode="scale_0_1")
        self.assertAlmostEqual(norm_0_1.min(), 0.0, delta=0.01)
        self.assertAlmostEqual(norm_0_1.max(), 1.0, delta=0.01)

        # Scale [-1, 1]
        norm_m1_1 = self.preprocessor.normalize(raw_arr, mode="tf_minus1_to_1")
        self.assertAlmostEqual(norm_m1_1.min(), -1.0, delta=0.05)
        self.assertAlmostEqual(norm_m1_1.max(), 1.0, delta=0.05)

        # Denormalize
        restored = self.preprocessor.denormalize(norm_0_1, mode="scale_0_1")
        self.assertEqual(restored.dtype, np.uint8)
        self.assertEqual(restored.shape, (224, 224, 3))

    def test_preprocess_pipeline(self):
        out = self.preprocessor.preprocess(self.tall_path, target_size=(224, 224), mode="scale_0_1")
        self.assertEqual(out.shape, (224, 224, 3))
        self.assertEqual(out.dtype, np.float32)

    def test_stratified_splits(self):
        # Create mock DataFrame with multiple categories
        rows = []
        for cat in CLOTHING_CATEGORIES[:4]:
            for i in range(20):
                rows.append({"id": f"{cat}_{i}", "canonical_category": cat, "image_path": str(self.tall_path)})
        df = pd.DataFrame(rows)

        split_df = create_stratified_splits(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, output_path=None)
        self.assertIn("split", split_df.columns)
        self.assertIn("category_idx", split_df.columns)

        train_count = (split_df["split"] == "train").sum()
        val_count = (split_df["split"] == "val").sum()
        test_count = (split_df["split"] == "test").sum()

        self.assertEqual(len(split_df), len(df))
        self.assertAlmostEqual(train_count / len(df), 0.8, delta=0.05)
        self.assertAlmostEqual(val_count / len(df), 0.1, delta=0.05)
        self.assertAlmostEqual(test_count / len(df), 0.1, delta=0.05)

    def test_tf_dataset_batch_generator(self):
        # Build mock dataset
        rows = []
        for i in range(16):
            rows.append({
                "id": str(i),
                "image_path": str(self.tall_path),
                "canonical_category": "T-Shirt",
                "category_idx": 0,
            })
        df = pd.DataFrame(rows)

        ds = create_tf_dataset(df, batch_size=4, is_training=False, target_size=(224, 224))
        for batch_images, batch_labels in ds.take(1):
            self.assertEqual(batch_images.shape, (4, 224, 224, 3))
            self.assertEqual(batch_labels.shape, (4, NUM_CLASSES))
            break


if __name__ == "__main__":
    unittest.main()
