"""
Unit tests for Module 13: Recommendation Optimization & Threshold Tuning.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.embeddings import EmbeddingManager
from src.style_matcher import StyleMatcher
from src.optimizer import ThresholdTuner, OptimizedOutfitRecommender


class TestOptimizer(unittest.TestCase):
    """Tests for threshold grid search, LRU pair caching, and fast outfit generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.emb_path = Path(self.temp_dir) / "test_embeddings.npy"
        self.idx_path = Path(self.temp_dir) / "test_index.csv"

        np.random.seed(42)
        raw_vecs = np.random.randn(6, 1280).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        vectors = raw_vecs / norms

        self.df = pd.DataFrame({
            "id": ["1", "2", "3", "4", "5", "6"],
            "productDisplayName": [
                "Men White Crew T-Shirt",
                "Men Blue Slim Jeans",
                "Men Black Formal Shoes",
                "Men Silver Analog Watch",
                "Men Grey Sweatshirt",
                "Men Black Chino Trousers",
            ],
            "canonical_category": ["T-Shirt", "Jeans", "Shoes", "Accessories", "T-Shirt", "Trousers"],
            "outfit_part": ["top", "bottom", "shoes", "accessory", "top", "bottom"],
            "baseColour": ["White", "Blue", "Black", "Silver", "Grey", "Black"],
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
        self.matcher = StyleMatcher(self.manager)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_threshold_tuner(self):
        tuner = ThresholdTuner(style_matcher=self.matcher)
        res = tuner.tune_compatibility_threshold(num_eval_pairs=4, threshold_steps=5)

        self.assertGreaterEqual(res.optimal_threshold, 0.50)
        self.assertLessEqual(res.optimal_threshold, 0.90)
        self.assertGreaterEqual(res.max_f1_score, 0.0)
        self.assertEqual(len(res.grid_curve), 5)

    def test_lru_cache_and_hit_tracking(self):
        opt = OptimizedOutfitRecommender(style_matcher=self.matcher, cache_capacity=10)

        item_a = self.df.iloc[0].to_dict()
        item_b = self.df.iloc[1].to_dict()

        # First call: miss
        res1 = opt.cached_score_pairing(item_a, item_b)
        self.assertEqual(opt.cache_misses, 1)
        self.assertEqual(opt.cache_hits, 0)

        # Second call (same pair, reversed order): hit!
        res2 = opt.cached_score_pairing(item_b, item_a)
        self.assertEqual(opt.cache_hits, 1)
        self.assertEqual(res1["composite_score"], res2["composite_score"])

        stats = opt.get_cache_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_ratio_pct"], 50.0)

    def test_fast_outfit_generation_latency(self):
        opt = OptimizedOutfitRecommender(style_matcher=self.matcher)
        outfits, lat_ms = opt.generate_outfit_fast(seed_item_id="1", top_k=1)

        self.assertEqual(len(outfits), 1)
        self.assertGreater(lat_ms, 0.0)
        self.assertIn("cohesion_score", outfits[0])


if __name__ == "__main__":
    unittest.main()
