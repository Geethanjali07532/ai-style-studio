"""
Unit tests for Module 7: Fashion Feature Extraction.
"""
import unittest
import numpy as np
from PIL import Image

from src.feature_extractor import FashionFeatureExtractor
from src.config import EMBEDDING_DIM


class TestFeatureExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = FashionFeatureExtractor(backbone="mobilenet_v2", weights="imagenet")
        # Create synthetic test images
        cls.test_img1 = Image.new("RGB", (224, 224), color=(100, 150, 200))
        cls.test_img2 = Image.new("RGB", (128, 256), color=(200, 50, 80))

    def test_extracted_vector_dimension_and_l2_norm(self):
        vec = self.extractor.extract_features(self.test_img1, l2_normalize=True)
        self.assertEqual(vec.shape, (EMBEDDING_DIM,))
        self.assertEqual(vec.dtype, np.float32)

        # L2 norm must be 1.0
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(float(norm), 1.0, delta=0.002)

    def test_batch_extraction_shape_and_norms(self):
        batch_inputs = [self.test_img1, self.test_img2, self.test_img1]
        embeddings = self.extractor.extract_batch(batch_inputs, batch_size=2, l2_normalize=True)

        self.assertEqual(embeddings.shape, (3, EMBEDDING_DIM))
        self.assertEqual(embeddings.dtype, np.float32)

        for row in embeddings:
            row_norm = np.linalg.norm(row)
            self.assertAlmostEqual(float(row_norm), 1.0, delta=0.002)

    def test_cosine_similarity(self):
        vec1 = self.extractor.extract_features(self.test_img1, l2_normalize=True)
        vec2 = self.extractor.extract_features(self.test_img2, l2_normalize=True)

        # Similarity with itself must be ~1.0
        self_sim = FashionFeatureExtractor.compute_similarity(vec1, vec1)
        self.assertAlmostEqual(self_sim, 1.0, delta=0.001)

        # Cross similarity should be between -1.0 and 1.0
        cross_sim = FashionFeatureExtractor.compute_similarity(vec1, vec2)
        self.assertGreaterEqual(cross_sim, -1.0)
        self.assertLessEqual(cross_sim, 1.0)

    def test_intermediate_feature_map_extraction(self):
        fmap = self.extractor.extract_intermediate_feature_maps(self.test_img1)
        # Should be a 3D tensor: (H_feat, W_feat, Channels)
        self.assertEqual(len(fmap.shape), 3)
        self.assertGreater(fmap.shape[0], 0)
        self.assertGreater(fmap.shape[1], 0)
        self.assertGreater(fmap.shape[2], 0)


if __name__ == "__main__":
    unittest.main()
