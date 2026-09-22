"""
Unit tests for ClothingImageAnalyzer (Module 14 / Core AI Pipeline).
"""
import unittest
import numpy as np
from PIL import Image

from src.analyzer import ClothingImageAnalyzer


class TestClothingImageAnalyzer(unittest.TestCase):
    """Tests for image feature extraction, category inference, and color extraction."""

    def setUp(self):
        self.analyzer = ClothingImageAnalyzer()

    def test_analyze_synthetic_image(self):
        # Create a synthetic 100x100 blue garment image
        img = Image.new("RGB", (100, 100), color=(20, 50, 180))
        result = self.analyzer.analyze_image(img)

        self.assertIn("category", result)
        self.assertIn("part", result)
        self.assertIn("dominant_color", result)
        self.assertIn("confidence_score", result)
        self.assertIn("visual_features", result)
        self.assertIn("embedding_vector", result)

        self.assertEqual(len(result["embedding_vector"]), 1280)
        self.assertGreaterEqual(result["confidence_score"], 0.70)
        self.assertIn(result["dominant_color"], ["Blue", "Navy Blue"])

    def test_analyze_with_known_metadata(self):
        img = Image.new("RGB", (60, 60), color=(255, 255, 255))
        meta = {
            "id": "999",
            "canonical_category": "Jeans",
            "outfit_part": "bottom",
            "baseColour": "Blue",
            "usage": "Casual",
            "productDisplayName": "Men Classic Slim Fit Jeans",
        }
        result = self.analyzer.analyze_image(img, known_metadata=meta)
        self.assertEqual(result["category"], "Jeans")
        self.assertEqual(result["part"], "bottom")
        self.assertEqual(result["dominant_color"], "Blue")
        self.assertEqual(result["confidence_score"], 0.98)


if __name__ == "__main__":
    unittest.main()
