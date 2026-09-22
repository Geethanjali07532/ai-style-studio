"""
Unit tests for Module 8: Image Embedding Generation & Vector Search Engine.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.embeddings import EmbeddingManager


class TestEmbeddingManager(unittest.TestCase):
    """Tests for EmbeddingManager loading, indexing, and vector search."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.emb_path = Path(self.temp_dir) / "test_embeddings.npy"
        self.idx_path = Path(self.temp_dir) / "test_index.csv"

        # Create synthetic 1280-dim normalized embeddings for 5 items
        np.random.seed(42)
        raw_vecs = np.random.randn(5, 1280).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        self.vectors = raw_vecs / norms

        self.df = pd.DataFrame({
            "id": ["1001", "1002", "1003", "1004", "1005"],
            "productDisplayName": [
                "Navy Blue Casual Shirt",
                "Black Slim Fit Jeans",
                "Blue Denim Jeans",
                "Floral Summer Dress",
                "Classic Running Shoes",
            ],
            "canonical_category": ["Shirt", "Jeans", "Jeans", "Dress", "Sneakers"],
            "outfit_part": ["top", "bottom", "bottom", "top", "shoes"],
            "gender": ["Men", "Men", "Women", "Women", "Unisex"],
            "usage": ["Casual", "Casual", "Casual", "Ethnic", "Sports"],
            "image_path": [
                f"/data/images/{i}.jpg" for i in ["1001", "1002", "1003", "1004", "1005"]
            ],
        })

        np.save(str(self.emb_path), self.vectors)
        self.df.to_csv(str(self.idx_path), index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_and_shape(self):
        manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        self.assertTrue(manager.is_loaded())
        self.assertEqual(len(manager), 5)
        self.assertEqual(manager.embeddings.shape, (5, 1280))
        self.assertEqual(len(manager.index_df), 5)

    def test_get_embedding(self):
        manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        vec = manager.get_embedding("1002")
        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape, (1280,))
        np.testing.assert_allclose(vec, self.vectors[1], atol=1e-6)

        # Missing item
        self.assertIsNone(manager.get_embedding("999999"))

    def test_search_by_item_id_self_excluded(self):
        manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        results = manager.search_by_item_id("1001", top_k=3, exclude_self=True)
        self.assertEqual(len(results), 3)
        retrieved_ids = [r["id"] for r in results]
        self.assertNotIn("1001", retrieved_ids)

    def test_category_and_gender_filtering(self):
        manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        # Search bottoms only
        results = manager.search_by_item_id(
            "1001",
            top_k=5,
            outfit_part="bottom",
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["outfit_part"], "bottom")
            self.assertEqual(r["canonical_category"], "Jeans")

        # Search Women's garments only
        women_results = manager.search_by_item_id(
            "1001",
            top_k=5,
            gender="Women",
        )
        # 1003 (Women), 1004 (Women), 1005 (Unisex)
        for r in women_results:
            self.assertIn(r["gender"], ["Women", "Unisex"])

    def test_search_by_vector_exact_match(self):
        manager = EmbeddingManager(
            embeddings_path=self.emb_path,
            index_path=self.idx_path,
            auto_load=True,
        )
        # Searching with exact vector for 1004
        vec_1004 = self.vectors[3]
        results = manager.search_by_vector(vec_1004, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1004")
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
