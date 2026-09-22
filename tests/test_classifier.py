"""
Unit tests for Module 6: Clothing Category Classifier architecture and forward pass.
"""
import unittest
import numpy as np
import tensorflow as tf

from src.classifier import build_category_classifier
from src.config import NUM_CLASSES, TARGET_IMG_SIZE


class TestCategoryClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build lightweight model for testing
        cls.model = build_category_classifier(
            backbone="mobilenet_v2",
            input_shape=(TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1], 3),
            num_classes=NUM_CLASSES,
            freeze_backbone=True,
        )

    def test_model_input_output_shapes(self):
        input_shape = self.model.input_shape
        output_shape = self.model.output_shape

        self.assertEqual(input_shape, (None, TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1], 3))
        self.assertEqual(output_shape, (None, NUM_CLASSES))

    def test_forward_pass_probabilities(self):
        # Create synthetic input batch of 2 images
        dummy_batch = np.random.uniform(0.0, 1.0, size=(2, TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1], 3)).astype(np.float32)
        preds = self.model.predict(dummy_batch, verbose=0)

        self.assertEqual(preds.shape, (2, NUM_CLASSES))
        # Each row should sum to 1.0 (softmax)
        for row in preds:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, delta=0.01)

    def test_backbone_frozen(self):
        # MobileNetV2 base model should be frozen during Stage 1
        base_layer = self.model.get_layer("mobilenetv2_1.00_224")
        self.assertFalse(base_layer.trainable)


if __name__ == "__main__":
    unittest.main()
