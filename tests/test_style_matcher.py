"""
Unit tests for Module 9: Clothing Similarity & Style Matching Engine.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.embeddings import EmbeddingManager
from src.style_matcher import StyleMatcher


class TestStyleMatcher(unittest.TestCase):
    """Tests for fashion color harmony, occasion matching, and composite outfit compatibility."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.emb_path = Path(self.temp_dir) / "test_embeddings.npy"
        self.idx_path = Path(self.temp_dir) / "test_index.csv"

        # Create 4 synthetic items
        # Item 101: Men Casual White T-Shirt (top)
        # Item 102: Men Casual Blue Denim Jeans (bottom)
        # Item 103: Men Formal Black Trousers (bottom)
        # Item 104: Women Casual Floral Skirt (bottom)
        np.random.seed(42)
        raw_vecs = np.random.randn(4, 1280).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        vectors = raw_vecs / norms

        self.df = pd.DataFrame({
            "id": ["101", "102", "103", "104"],
            "productDisplayName": [
                "White Graphic T-Shirt",
                "Blue Denim Slim Jeans",
                "Black Tailored Trousers",
                "Floral Summer Skirt",
            ],
            "canonical_category": ["T-Shirt", "Jeans", "Trousers", "Skirt"],
            "outfit_part": ["top", "bottom", "bottom", "bottom"],
            "baseColour": ["White", "Blue", "Black", "Pink"],
            "gender": ["Men", "Men", "Men", "Women"],
            "usage": ["Casual", "Casual", "Formal", "Casual"],
            "image_path": [f"/data/images/{i}.jpg" for i in ["101", "102", "103", "104"]],
        })

        np.save(str(self.emb_path), vectors)
        self.df.to_csv(str(self.idx_path), index=False)

        self.manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        self.matcher = StyleMatcher(embedding_manager=self.manager)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_color_compatibility_neutrals(self):
        # Dual neutrals (Black & White)
        score, harmony = StyleMatcher.compute_color_compatibility("Black", "White")
        self.assertGreaterEqual(score, 0.95)
        self.assertEqual(harmony, "dual_neutral_classic")

        # Neutral + Accent (White + Blue)
        score_accent, _ = StyleMatcher.compute_color_compatibility("White", "Blue")
        self.assertGreaterEqual(score_accent, 0.90)

    def test_color_compatibility_complementary(self):
        score, harmony = StyleMatcher.compute_color_compatibility("Blue", "Orange")
        self.assertGreaterEqual(score, 0.85)
        self.assertEqual(harmony, "complementary_contrast")

    def test_usage_compatibility(self):
        score_same, _ = StyleMatcher.compute_usage_compatibility("Casual", "Casual")
        self.assertEqual(score_same, 1.0)

        score_clash, _ = StyleMatcher.compute_usage_compatibility("Formal", "Sports")
        self.assertLessEqual(score_clash, 0.3)

    def test_gender_compatibility(self):
        self.assertTrue(StyleMatcher.is_gender_compatible("Men", "Men"))
        self.assertTrue(StyleMatcher.is_gender_compatible("Men", "Unisex"))
        self.assertFalse(StyleMatcher.is_gender_compatible("Men", "Women"))

    def test_score_pairing_breakdown(self):
        item_a = self.df.iloc[0].to_dict()  # Men Casual White T-Shirt
        item_b = self.df.iloc[1].to_dict()  # Men Casual Blue Denim Jeans

        res = self.matcher.score_pairing(item_a, item_b, visual_similarity=0.75)
        self.assertIn("composite_score", res)
        self.assertIn("color_harmony", res)
        self.assertTrue(res["gender_compatible"])
        self.assertGreater(res["composite_score"], 0.70)

    def test_find_compatible_bottoms_for_top(self):
        matches = self.matcher.find_compatible_garments(
            query_item_id="101",
            target_part="bottom",
            top_k=2,
            enforce_gender=True,
        )
        self.assertTrue(len(matches) > 0)
        # Verify matched items are bottoms and gender-aligned
        for m in matches:
            self.assertEqual(m["outfit_part"], "bottom")
            self.assertEqual(m["gender"], "Men")
            self.assertIn("composite_score", m)


if __name__ == "__main__":
    unittest.main()
