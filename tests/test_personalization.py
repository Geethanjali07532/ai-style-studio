"""
Unit tests for Module 11: User Preference & Personalization Engine.
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
from src.personalization import (
    UserProfile,
    PersonalizedRecommender,
    create_preset_profile,
)


class TestPersonalization(unittest.TestCase):
    """Tests for user style profiles, item affinity calculation, and personalized outfit re-ranking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.emb_path = Path(self.temp_dir) / "test_embeddings.npy"
        self.idx_path = Path(self.temp_dir) / "test_index.csv"

        # Create 6 synthetic items:
        # Item 1: Men Black T-Shirt (top, Casual)
        # Item 2: Men Blue Jeans (bottom, Casual)
        # Item 3: Men White Sneakers (shoes, Casual)
        # Item 4: Men Pink Floral Shirt (top, Casual)
        # Item 5: Men Yellow Chinos (bottom, Casual)
        # Item 6: Men Red Shoes (shoes, Casual)
        np.random.seed(42)
        raw_vecs = np.random.randn(6, 1280).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        vectors = raw_vecs / norms

        self.df = pd.DataFrame({
            "id": ["1", "2", "3", "4", "5", "6"],
            "productDisplayName": [
                "Men Black Crew Neck T-Shirt",
                "Men Classic Blue Jeans",
                "Men White Canvas Sneakers",
                "Men Pink Floral Casual Shirt",
                "Men Mustard Yellow Chinos",
                "Men Red High Top Shoes",
            ],
            "canonical_category": ["T-Shirt", "Jeans", "Sneakers", "Shirt", "Trousers", "Shoes"],
            "outfit_part": ["top", "bottom", "shoes", "top", "bottom", "shoes"],
            "baseColour": ["Black", "Blue", "White", "Pink", "Yellow", "Red"],
            "gender": ["Men", "Men", "Men", "Men", "Men", "Men"],
            "usage": ["Casual", "Casual", "Casual", "Casual", "Casual", "Casual"],
            "image_path": [f"/data/images/{i}.jpg" for i in range(1, 7)],
        })

        np.save(str(self.emb_path), vectors)
        self.df.to_csv(str(self.idx_path), index=False)

        self.manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        self.outfit_recommender = OutfitRecommender(embedding_manager=self.manager)
        self.pers_recommender = PersonalizedRecommender(
            outfit_recommender=self.outfit_recommender,
            personalization_weight=0.40,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_user_profile_creation_and_serialization(self):
        profile = UserProfile(
            user_id="user_123",
            gender="Men",
            archetype="Minimalist",
            favorite_colors=["Black", "White"],
            disliked_colors=["Pink"],
        )
        profile.add_interaction("1", "like")
        profile.add_interaction("4", "dislike")

        self.assertIn("1", profile.liked_item_ids)
        self.assertIn("4", profile.disliked_item_ids)

        p_dict = profile.to_dict()
        restored = UserProfile.from_dict(p_dict)
        self.assertEqual(restored.user_id, "user_123")
        self.assertEqual(restored.favorite_colors, ["Black", "White"])
        self.assertIn("1", restored.liked_item_ids)

    def test_create_preset_profile(self):
        minimalist = create_preset_profile("Minimalist", "user_min", "Men")
        self.assertEqual(minimalist.archetype, "Minimalist")
        self.assertIn("Black", minimalist.favorite_colors)
        self.assertIn("Pink", minimalist.disliked_colors)

        vibrant = create_preset_profile("Vibrant Eclectic", "user_vib", "Women")
        self.assertEqual(vibrant.gender, "Women")
        self.assertIn("Red", vibrant.favorite_colors)

    def test_compute_item_affinity(self):
        profile = UserProfile(
            user_id="user_test",
            favorite_colors=["Black", "White"],
            disliked_colors=["Pink"],
            preferred_categories=["T-Shirt"],
            disliked_categories=["Skirt"],
        )

        item_black_tshirt = self.df.iloc[0].to_dict()  # Black T-Shirt
        item_pink_shirt = self.df.iloc[3].to_dict()   # Pink Shirt

        aff_black = self.pers_recommender.compute_item_affinity(item_black_tshirt, profile)
        aff_pink = self.pers_recommender.compute_item_affinity(item_pink_shirt, profile)

        # Black T-shirt has favorite color and preferred category
        self.assertGreater(aff_black, 0.70)
        # Pink Shirt has disliked color
        self.assertLess(aff_pink, 0.40)
        self.assertGreater(aff_black, aff_pink)

    def test_score_personalized_outfit(self):
        profile = UserProfile(
            user_id="user_test",
            favorite_colors=["Black", "Blue", "White"],
            disliked_colors=["Pink", "Yellow"],
        )

        # Outfit A: Black T-shirt (1) + Blue Jeans (2) + White Sneakers (3) (Matches preferences)
        outfit_a = {
            "cohesion_score": 0.80,
            "items": [self.df.iloc[0].to_dict(), self.df.iloc[1].to_dict(), self.df.iloc[2].to_dict()],
        }
        # Outfit B: Pink Shirt (4) + Yellow Chinos (5) + Red Shoes (6) (Contains disliked colors)
        outfit_b = {
            "cohesion_score": 0.80,
            "items": [self.df.iloc[3].to_dict(), self.df.iloc[4].to_dict(), self.df.iloc[5].to_dict()],
        }

        res_a = self.pers_recommender.score_personalized_outfit(outfit_a, profile)
        res_b = self.pers_recommender.score_personalized_outfit(outfit_b, profile)

        self.assertIn("personalized_score", res_a)
        self.assertIn("user_affinity", res_a)
        self.assertFalse(res_a["has_disliked_element"])
        self.assertTrue(res_b["has_disliked_element"])
        self.assertGreater(res_a["personalized_score"], res_b["personalized_score"])

    def test_recommend_feed(self):
        profile = UserProfile(
            user_id="user_test",
            gender="Men",
            favorite_colors=["Black", "White"],
            disliked_colors=["Pink"],
        )

        feed = self.pers_recommender.recommend_feed(profile, top_k=4)
        self.assertEqual(len(feed), 4)
        # Highest affinity item should be at the top
        self.assertGreaterEqual(feed[0]["user_affinity"], feed[-1]["user_affinity"])


if __name__ == "__main__":
    unittest.main()
