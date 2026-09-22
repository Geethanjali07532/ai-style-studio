"""
Unit tests for Module 10: Outfit Recommendation Engine.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.embeddings import EmbeddingManager
from src.style_matcher import StyleMatcher
from src.recommender import OutfitRecommender


class TestOutfitRecommender(unittest.TestCase):
    """Tests for multi-item outfit generation, beam search, and cohesion scoring."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.emb_path = Path(self.temp_dir) / "test_embeddings.npy"
        self.idx_path = Path(self.temp_dir) / "test_index.csv"

        # Synthetic catalog containing:
        # 1. Men White Shirt (top)
        # 2. Men Blue Jeans (bottom)
        # 3. Men Black Shoes (shoes)
        # 4. Men Silver Watch (accessory)
        # 5. Women Floral Dress (top / Dress)
        # 6. Women Red Heels (shoes)
        # 7. Women Leather Handbag (accessory)
        # 8. Men White Sneakers (shoes)
        np.random.seed(42)
        raw_vecs = np.random.randn(8, 1280).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        vectors = raw_vecs / norms

        self.df = pd.DataFrame({
            "id": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "productDisplayName": [
                "Men Classic White Shirt",
                "Men Slim Blue Jeans",
                "Men Black Formal Shoes",
                "Men Silver Analog Watch",
                "Women Red Floral Dress",
                "Women Red High Heels",
                "Women Black Handbag",
                "Men White Canvas Sneakers",
            ],
            "canonical_category": ["Shirt", "Jeans", "Shoes", "Accessories", "Dress", "Shoes", "Accessories", "Sneakers"],
            "outfit_part": ["top", "bottom", "shoes", "accessory", "top", "shoes", "accessory", "shoes"],
            "baseColour": ["White", "Blue", "Black", "Silver", "Red", "Red", "Black", "White"],
            "gender": ["Men", "Men", "Men", "Men", "Women", "Women", "Women", "Men"],
            "usage": ["Casual", "Casual", "Casual", "Casual", "Party", "Party", "Party", "Casual"],
            "image_path": [f"/data/images/{i}.jpg" for i in range(1, 9)],
        })

        np.save(str(self.emb_path), vectors)
        self.df.to_csv(str(self.idx_path), index=False)

        self.manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        self.recommender = OutfitRecommender(embedding_manager=self.manager)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_score_outfit_metrics(self):
        items = [self.df.iloc[0].to_dict(), self.df.iloc[1].to_dict(), self.df.iloc[2].to_dict()]
        result = self.recommender.score_outfit(items)

        self.assertIn("cohesion_score", result)
        self.assertIn("color_palette", result)
        self.assertEqual(result["num_items"], 3)
        self.assertEqual(len(result["pairwise_breakdown"]), 3)  # 3 choose 2 = 3 pairs
        self.assertGreater(result["cohesion_score"], 0.40)

    def test_generate_outfit_from_seed_top(self):
        outfits = self.recommender.generate_outfit_from_seed(
            seed_item_id="1",
            include_accessory=True,
            top_k_outfits=1,
        )
        self.assertEqual(len(outfits), 1)
        outfit = outfits[0]
        self.assertIn("cohesion_score", outfit)
        self.assertGreaterEqual(outfit["cohesion_score"], 0.50)

        # Check parts included: should have top, bottom, shoes, accessory
        parts = [item["outfit_part"] for item in outfit["items"]]
        self.assertIn("top", parts)
        self.assertIn("bottom", parts)
        self.assertIn("shoes", parts)
        self.assertIn("accessory", parts)

        # Ensure all items match seed gender
        for item in outfit["items"]:
            self.assertIn(item["gender"], ["Men", "Unisex"])

    def test_generate_outfit_from_seed_dress(self):
        # Dress should not include a 'bottom'
        outfits = self.recommender.generate_outfit_from_seed(
            seed_item_id="5",
            include_accessory=True,
            top_k_outfits=1,
        )
        self.assertEqual(len(outfits), 1)
        outfit = outfits[0]
        parts = [item["outfit_part"] for item in outfit["items"]]
        self.assertIn("top", parts)  # Dress
        self.assertNotIn("bottom", parts)
        self.assertIn("shoes", parts)

    def test_generate_outfit_by_occasion(self):
        outfits = self.recommender.generate_outfit_by_occasion(
            occasion="Party",
            gender="Women",
            include_accessory=True,
            top_k_outfits=1,
        )
        self.assertEqual(len(outfits), 1)
        outfit = outfits[0]
        # Seed should be the Party Dress
        self.assertEqual(outfit["items"][0]["usage"], "Party")
        self.assertEqual(outfit["items"][0]["gender"], "Women")

    def test_substitute_item_in_outfit(self):
        # Create an outfit: White Shirt (1) + Blue Jeans (2) + Black Shoes (3)
        outfit_items = [self.df.iloc[0].to_dict(), self.df.iloc[1].to_dict(), self.df.iloc[2].to_dict()]
        
        # Replace the shoes (id 3)
        alternatives = self.recommender.substitute_item_in_outfit(
            outfit_items=outfit_items,
            replace_item_id="3",
            top_k_alternatives=2,
        )
        # Should return alternative shoes
        self.assertTrue(len(alternatives) > 0)
        for alt in alternatives:
            self.assertIn("new_outfit_cohesion", alt)
            self.assertNotEqual(alt["id"], "3")


if __name__ == "__main__":
    unittest.main()
