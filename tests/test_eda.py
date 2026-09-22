"""
Unit tests for Module 5: Fashion Image Exploratory Analysis and Color Analytics.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

from src.eda_analysis import ColorExtractor


class TestColorAnalytics(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_rgb_to_hsv_conversion(self):
        # Red: Hue 0
        h, s, v = ColorExtractor.rgb_to_hsv((255, 0, 0))
        self.assertAlmostEqual(h, 0.0, delta=1.0)
        self.assertAlmostEqual(s, 1.0, delta=0.01)
        self.assertAlmostEqual(v, 1.0, delta=0.01)

        # Green: Hue 120
        h, s, v = ColorExtractor.rgb_to_hsv((0, 255, 0))
        self.assertAlmostEqual(h, 120.0, delta=1.0)

        # Blue: Hue 240
        h, s, v = ColorExtractor.rgb_to_hsv((0, 0, 255))
        self.assertAlmostEqual(h, 240.0, delta=1.0)

    def test_rgb_to_hex(self):
        self.assertEqual(ColorExtractor.rgb_to_hex((255, 255, 255)), "#FFFFFF")
        self.assertEqual(ColorExtractor.rgb_to_hex((0, 0, 0)), "#000000")
        self.assertEqual(ColorExtractor.rgb_to_hex((255, 0, 0)), "#FF0000")

    def test_neutral_color_detection(self):
        # True neutrals
        self.assertTrue(ColorExtractor.is_neutral_color((15, 15, 15)))      # Black
        self.assertTrue(ColorExtractor.is_neutral_color((245, 245, 245)))  # White
        self.assertTrue(ColorExtractor.is_neutral_color((128, 128, 128)))  # Gray
        self.assertTrue(ColorExtractor.is_neutral_color((220, 205, 175)))  # Beige

        # Non-neutrals
        self.assertFalse(ColorExtractor.is_neutral_color((240, 30, 30)))   # Vivid Red
        self.assertFalse(ColorExtractor.is_neutral_color((30, 220, 30)))   # Vivid Green
        self.assertFalse(ColorExtractor.is_neutral_color((240, 220, 20)))  # Vivid Yellow

    def test_color_harmony_rules(self):
        # Neutral match: Black t-shirt with vivid red pants
        match_neutral = ColorExtractor.compute_color_harmony_type((20, 20, 20), (220, 30, 30))
        self.assertEqual(match_neutral, "neutral_match")

        # Monochromatic / Analogous: Navy blue (0, 30, 100) and Sky blue (50, 150, 250)
        match_analogous = ColorExtractor.compute_color_harmony_type((0, 100, 220), (30, 130, 230))
        self.assertIn(match_analogous, ["monochromatic_or_analogous", "neutral_match"])

        # Complementary: Blue (Hue ~240) and Orange (Hue ~30)
        match_comp = ColorExtractor.compute_color_harmony_type((20, 100, 240), (240, 120, 20))
        self.assertEqual(match_comp, "complementary")

    def test_dominant_color_extraction_with_background_filter(self):
        # Create synthetic image: predominantly red garment centered on white studio background
        img = Image.new("RGB", (100, 100), color=(250, 250, 250))  # White background
        pixels = np.array(img)
        pixels[25:75, 25:75] = [210, 30, 40]  # Red garment square in center
        synthetic_img = Image.fromarray(pixels)

        palettes = ColorExtractor.extract_dominant_colors(synthetic_img, k=2, filter_background=True)
        self.assertGreaterEqual(len(palettes), 1)

        # Primary cluster should be the red garment
        primary = palettes[0]
        self.assertGreater(primary["rgb"][0], 180)  # High red channel
        self.assertLess(primary["rgb"][1], 60)      # Low green
        self.assertLess(primary["rgb"][2], 60)      # Low blue


if __name__ == "__main__":
    unittest.main()
