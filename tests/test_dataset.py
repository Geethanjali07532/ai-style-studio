"""
Unit tests for FashionDataset and data ingestion logic.
"""
import unittest
import tempfile
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image

from src.dataset import FashionDataset
from src.config import CLOTHING_CATEGORIES, CATEGORY_TO_OUTFIT_PART


class TestFashionDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for tests
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = Path(self.test_dir) / "images"
        self.images_dir.mkdir(parents=True)
        self.styles_csv = Path(self.test_dir) / "styles.csv"

        # Create mock images and metadata
        mock_data = [
            {"id": "101", "articleType": "Tshirts", "gender": "Men", "usage": "Casual", "baseColour": "Blue", "productDisplayName": "Blue T-shirt"},
            {"id": "102", "articleType": "Jeans", "gender": "Men", "usage": "Casual", "baseColour": "Black", "productDisplayName": "Black Denim Jeans"},
            {"id": "103", "articleType": "Casual Shoes", "gender": "Men", "usage": "Casual", "baseColour": "White", "productDisplayName": "White Sneakers"},
            {"id": "104", "articleType": "Dresses", "gender": "Women", "usage": "Formal", "baseColour": "Red", "productDisplayName": "Red Evening Dress"},
        ]
        pd.DataFrame(mock_data).to_csv(self.styles_csv, index=False)

        for item in mock_data:
            img = Image.new("RGB", (64, 64), color=(100, 150, 200))
            img.save(self.images_dir / f"{item['id']}.jpg")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_and_mapping(self):
        dataset = FashionDataset(metadata_path=self.styles_csv, images_dir=self.images_dir)
        self.assertEqual(len(dataset), 4)

        # Check category mappings
        item_101 = dataset.get_item("101")
        self.assertIsNotNone(item_101)
        self.assertEqual(item_101["canonical_category"], "T-Shirt")
        self.assertEqual(item_101["outfit_part"], "top")

        item_102 = dataset.get_item("102")
        self.assertEqual(item_102["canonical_category"], "Jeans")
        self.assertEqual(item_102["outfit_part"], "bottom")

        item_103 = dataset.get_item("103")
        self.assertEqual(item_103["canonical_category"], "Shoes")
        self.assertEqual(item_103["outfit_part"], "shoes")

    def test_image_loading(self):
        dataset = FashionDataset(metadata_path=self.styles_csv, images_dir=self.images_dir)
        img = dataset.load_image("101", target_size=(224, 224))
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (224, 224))

    def test_sample_outfit(self):
        dataset = FashionDataset(metadata_path=self.styles_csv, images_dir=self.images_dir)
        outfit = dataset.sample_random_outfit(gender="Men")
        self.assertIn("top", outfit)
        self.assertIn("bottom", outfit)
        self.assertIn("shoes", outfit)
        self.assertIsNotNone(outfit["top"])
        self.assertEqual(outfit["top"]["id"], "101")
        self.assertEqual(outfit["bottom"]["id"], "102")
        self.assertEqual(outfit["shoes"]["id"], "103")


if __name__ == "__main__":
    unittest.main()
