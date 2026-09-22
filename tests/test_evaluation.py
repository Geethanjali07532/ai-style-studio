"""
Unit tests for Module 12: Recommendation Model Evaluation.
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
from src.evaluation import RecommendationEvaluator


class TestRecommendationEvaluator(unittest.TestCase):
    """Tests for mathematical precision, recall, MRR, hit rate, and diversity metrics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.emb_path = Path(self.temp_dir) / "test_embeddings.npy"
        self.idx_path = Path(self.temp_dir) / "test_index.csv"

        # Synthetic catalog with 4 items
        # 1: Unit vector [1, 0, 0, ...]
        # 2: Unit vector [0, 1, 0, ...] (Orthogonal to 1, distance = 1.0)
        # 3: Unit vector [1, 0, 0, ...] (Identical to 1, distance = 0.0)
        # 4: Unit vector [0, 0, 1, ...]
        vectors = np.zeros((4, 1280), dtype=np.float32)
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        vectors[2, 0] = 1.0
        vectors[3, 2] = 1.0

        self.df = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "productDisplayName": ["Item 1", "Item 2", "Item 3", "Item 4"],
            "canonical_category": ["Shirt", "Shirt", "Shirt", "Jeans"],
            "outfit_part": ["top", "top", "top", "bottom"],
            "baseColour": ["White", "Black", "White", "Blue"],
            "gender": ["Men", "Men", "Men", "Men"],
            "usage": ["Casual", "Casual", "Casual", "Casual"],
            "image_path": [f"/data/images/{i}.jpg" for i in range(1, 5)],
        })

        np.save(str(self.emb_path), vectors)
        self.df.to_csv(str(self.idx_path), index=False)

        self.manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        self.evaluator = RecommendationEvaluator(embedding_manager=self.manager)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_precision_at_k(self):
        recommended = ["1", "2", "3", "4", "5"]
        relevant = {"1", "3", "9"}

        p_at_1 = RecommendationEvaluator.compute_precision_at_k(recommended, relevant, k=1)
        self.assertEqual(p_at_1, 1.0)

        p_at_3 = RecommendationEvaluator.compute_precision_at_k(recommended, relevant, k=3)
        self.assertAlmostEqual(p_at_3, 2.0 / 3.0, places=3)

        p_at_5 = RecommendationEvaluator.compute_precision_at_k(recommended, relevant, k=5)
        self.assertEqual(p_at_5, 2.0 / 5.0)

        # Empty / edge cases
        self.assertEqual(RecommendationEvaluator.compute_precision_at_k([], relevant, k=5), 0.0)
        self.assertEqual(RecommendationEvaluator.compute_precision_at_k(recommended, relevant, k=0), 0.0)

    def test_recall_at_k(self):
        recommended = ["1", "2", "3", "4", "5"]
        relevant = {"1", "3", "9"}  # Total 3 relevant items

        r_at_1 = RecommendationEvaluator.compute_recall_at_k(recommended, relevant, k=1)
        self.assertAlmostEqual(r_at_1, 1.0 / 3.0, places=3)

        r_at_5 = RecommendationEvaluator.compute_recall_at_k(recommended, relevant, k=5)
        self.assertAlmostEqual(r_at_5, 2.0 / 3.0, places=3)

        # Empty relevant
        self.assertEqual(RecommendationEvaluator.compute_recall_at_k(recommended, set(), k=5), 0.0)

    def test_hit_rate_at_k(self):
        recommended = ["1", "2", "3", "4"]
        relevant = {"3"}

        self.assertEqual(RecommendationEvaluator.compute_hit_rate_at_k(recommended, relevant, k=2), 0.0)
        self.assertEqual(RecommendationEvaluator.compute_hit_rate_at_k(recommended, relevant, k=3), 1.0)
        self.assertEqual(RecommendationEvaluator.compute_hit_rate_at_k(recommended, {"99"}, k=4), 0.0)

    def test_mrr(self):
        # First relevant item at rank 1 -> 1.0
        self.assertEqual(RecommendationEvaluator.compute_mrr(["1", "2", "3"], {"1"}), 1.0)

        # First relevant item at rank 2 -> 0.5
        self.assertEqual(RecommendationEvaluator.compute_mrr(["1", "2", "3"], {"2"}), 0.5)

        # First relevant item at rank 3 -> 1/3
        self.assertAlmostEqual(RecommendationEvaluator.compute_mrr(["1", "2", "3"], {"3"}), 1.0 / 3.0, places=3)

        # No match -> 0.0
        self.assertEqual(RecommendationEvaluator.compute_mrr(["1", "2", "3"], {"99"}), 0.0)

    def test_intra_list_diversity(self):
        # Items 1 and 3 are identical (distance = 0.0)
        ild_identical = self.evaluator.compute_intra_list_diversity(["1", "3"])
        self.assertAlmostEqual(ild_identical, 0.0, places=3)

        # Items 1 and 2 are orthogonal (distance = 1.0)
        ild_orthogonal = self.evaluator.compute_intra_list_diversity(["1", "2"])
        self.assertAlmostEqual(ild_orthogonal, 1.0, places=3)

    def test_catalog_coverage(self):
        coverage_res = self.evaluator.evaluate_catalog_coverage(num_queries=4, top_k=2)
        self.assertIn("catalog_coverage_percentage", coverage_res)
        self.assertGreaterEqual(coverage_res["catalog_coverage_percentage"], 0.0)
        self.assertLessEqual(coverage_res["catalog_coverage_percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()
