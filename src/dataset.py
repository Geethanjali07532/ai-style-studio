"""
Fashion Dataset Loader and Ingestion Module.
Handles loading, cleaning, category mapping, and retrieval of fashion items.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image


from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_STYLES_CSV,
    RAW_IMAGES_DIR,
    CLEANED_METADATA_CSV,
    DEFAULT_IMAGE_SIZE,
    CLOTHING_CATEGORIES,
    CATEGORY_TO_OUTFIT_PART,
    KAGGLE_ARTICLE_MAPPING,
)


class FashionDataset:
    """
    Catalog and dataset manager for fashion items and outfit recommendation.
    Supports Kaggle Fashion Product Images dataset and custom catalog exports.
    """

    def __init__(
        self,
        metadata_path: Optional[str | Path] = None,
        images_dir: Optional[str | Path] = None,
        auto_clean: bool = True,
        filter_valid_images: bool = True,
    ):
        """
        Initialize the FashionDataset catalog.

        Args:
            metadata_path: Path to styles.csv or metadata_clean.csv. If None, auto-discovers.
            images_dir: Path to directory containing product images. If None, auto-discovers.
            auto_clean: Whether to automatically map categories and validate fields.
            filter_valid_images: If True, drops items whose image files are missing on disk.
        """
        self.metadata_path = self._resolve_metadata_path(metadata_path)
        self.images_dir = self._resolve_images_dir(images_dir)
        self.df: pd.DataFrame = pd.DataFrame()

        if self.metadata_path and self.metadata_path.exists():
            self._load_and_prepare_data(auto_clean=auto_clean, filter_valid_images=filter_valid_images)

    def _resolve_metadata_path(self, user_path: Optional[str | Path]) -> Optional[Path]:
        """Locates the metadata CSV file across potential locations."""
        if user_path:
            p = Path(user_path)
            if p.exists():
                return p

        # Check clean metadata first (processed)
        if CLEANED_METADATA_CSV.exists():
            return CLEANED_METADATA_CSV

        # Check standard raw styles CSV
        if RAW_STYLES_CSV.exists():
            return RAW_STYLES_CSV

        # Recursive search in raw directory for styles.csv or any metadata csv
        if RAW_DATA_DIR.exists():
            for match in RAW_DATA_DIR.rglob("styles.csv"):
                return match
            for match in RAW_DATA_DIR.rglob("*metadata*.csv"):
                return match

        return None

    def _resolve_images_dir(self, user_path: Optional[str | Path]) -> Optional[Path]:
        """Locates the image folder across potential locations."""
        if user_path:
            p = Path(user_path)
            if p.exists():
                return p

        if RAW_IMAGES_DIR.exists():
            return RAW_IMAGES_DIR

        # Search for folder named 'images' under raw directory
        if RAW_DATA_DIR.exists():
            for match in RAW_DATA_DIR.rglob("images"):
                if match.is_dir():
                    return match

        return None

    def _load_and_prepare_data(self, auto_clean: bool, filter_valid_images: bool):
        """Loads the CSV file and maps columns to unified taxonomy."""
        try:
            # on_bad_lines='skip' is essential for Kaggle styles.csv which has slight formatting quirks
            df = pd.read_csv(self.metadata_path, on_bad_lines="skip")
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata CSV at {self.metadata_path}: {e}")

        # Standardize column names (lowercase & stripped)
        df.columns = [c.strip() for c in df.columns]

        # Ensure ID column is present and formatted as string for filename matching
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        elif "item_id" in df.columns:
            df["id"] = df["item_id"].astype(str)
        else:
            df["id"] = df.index.astype(str)

        # Resolve image file paths
        if self.images_dir and self.images_dir.exists():
            df["image_path"] = df["id"].apply(lambda item_id: str(self.images_dir / f"{item_id}.jpg"))
            if filter_valid_images:
                df["image_exists"] = df["image_path"].apply(os.path.exists)
                initial_len = len(df)
                df = df[df["image_exists"]].copy()
                df.drop(columns=["image_exists"], inplace=True)
                missing_count = initial_len - len(df)
                if missing_count > 0:
                    print(f"[FashionDataset] Filtered {missing_count} items with missing image files.")
        else:
            df["image_path"] = ""

        if auto_clean:
            df = self._map_categories_and_taxonomy(df)

        self.df = df

    def _map_categories_and_taxonomy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes categories and attaches outfit parts."""
        # Detect article type column
        article_col = None
        for cand in ["articleType", "article_type", "category", "subCategory"]:
            if cand in df.columns:
                article_col = cand
                break

        if article_col:
            # Map using KAGGLE_ARTICLE_MAPPING with fallback to original value
            df["canonical_category"] = df[article_col].map(KAGGLE_ARTICLE_MAPPING)
            # Fill unmapped with direct match if in CLOTHING_CATEGORIES or 'Accessories'
            unmapped_mask = df["canonical_category"].isna()
            df.loc[unmapped_mask, "canonical_category"] = df.loc[unmapped_mask, article_col].apply(
                lambda val: val if val in CLOTHING_CATEGORIES else "Accessories"
            )
        else:
            df["canonical_category"] = "Accessories"

        # Map to high-level outfit part (top, bottom, shoes, outerwear, accessory)
        df["outfit_part"] = df["canonical_category"].map(CATEGORY_TO_OUTFIT_PART).fillna("accessory")

        # Standardize gender, usage, baseColour if available
        if "gender" not in df.columns:
            df["gender"] = "Unisex"
        if "usage" not in df.columns:
            df["usage"] = "Casual"
        if "baseColour" in df.columns and "base_colour" not in df.columns:
            df["base_colour"] = df["baseColour"]
        elif "base_colour" not in df.columns:
            df["base_colour"] = "Unknown"

        return df

    def __len__(self) -> int:
        return len(self.df)

    def is_loaded(self) -> bool:
        """Returns True if the dataset metadata is loaded and non-empty."""
        return not self.df.empty

    def get_item(self, item_id: str | int) -> Optional[Dict[str, Any]]:
        """Fetch metadata dictionary for a specific item ID."""
        item_id_str = str(item_id)
        match = self.df[self.df["id"] == item_id_str]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def load_image(
        self,
        item_id: str | int,
        target_size: Optional[Tuple[int, int]] = DEFAULT_IMAGE_SIZE,
        as_numpy: bool = False,
    ) -> Optional[Image.Image | np.ndarray]:
        """
        Loads an item's image from disk, converts to RGB, and optionally resizes.

        Args:
            item_id: Clothing item ID.
            target_size: (width, height) tuple to resize image to. If None, original size.
            as_numpy: If True, returns numpy array [0, 255] uint8.
        """
        item = self.get_item(item_id)
        if not item or not item.get("image_path"):
            return None

        img_path = item["image_path"]
        if not os.path.exists(img_path):
            return None

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if target_size:
                    img = img.resize(target_size, Image.Resampling.BILINEAR)
                if as_numpy:
                    return np.array(img, dtype=np.uint8)
                return img
        except Exception as e:
            print(f"[FashionDataset] Error opening image {img_path}: {e}")
            return None

    def get_items_by_category(self, category: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Returns items matching a canonical category."""
        subset = self.df[self.df["canonical_category"] == category]
        if limit is not None:
            return subset.head(limit)
        return subset

    def get_items_by_outfit_part(self, part: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Returns items matching an outfit part ('top', 'bottom', 'shoes', etc.)."""
        subset = self.df[self.df["outfit_part"] == part]
        if limit is not None:
            return subset.head(limit)
        return subset

    def sample_random_outfit(
        self,
        gender: Optional[str] = None,
        usage: Optional[str] = None,
        include_accessory: bool = True,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Samples a coordinated candidate outfit from available items.
        Returns a dictionary with keys: 'top', 'bottom', 'shoes', and optionally 'accessory'.
        """
        pool = self.df.copy()
        if gender and "gender" in pool.columns and gender != "All":
            pool = pool[pool["gender"].isin([gender, "Unisex"])]
        if usage and "usage" in pool.columns and usage != "All":
            usage_matches = pool[pool["usage"].str.lower() == usage.lower()]
            if not usage_matches.empty:
                pool = usage_matches

        outfit: Dict[str, Optional[Dict[str, Any]]] = {}
        for part in ["top", "bottom", "shoes"]:
            candidates = pool[pool["outfit_part"] == part]
            if not candidates.empty:
                outfit[part] = candidates.sample(1, random_state=None).iloc[0].to_dict()
            else:
                outfit[part] = None

        if include_accessory:
            acc_candidates = pool[pool["outfit_part"] == "accessory"]
            if not acc_candidates.empty:
                outfit["accessory"] = acc_candidates.sample(1, random_state=None).iloc[0].to_dict()
            else:
                outfit["accessory"] = None

        return outfit

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates summary statistics of the catalog."""
        if self.df.empty:
            return {
                "total_items": 0,
                "loaded": False,
                "message": "Dataset is currently empty. Place Kaggle dataset in data/raw/.",
            }

        stats = {
            "total_items": len(self.df),
            "loaded": True,
            "images_found": sum(self.df["image_path"].apply(os.path.exists)) if "image_path" in self.df.columns else 0,
            "categories": self.df["canonical_category"].value_counts().to_dict(),
            "outfit_parts": self.df["outfit_part"].value_counts().to_dict(),
        }

        if "gender" in self.df.columns:
            stats["gender_distribution"] = self.df["gender"].value_counts().head(5).to_dict()
        if "usage" in self.df.columns:
            stats["usage_distribution"] = self.df["usage"].value_counts().head(5).to_dict()
        if "base_colour" in self.df.columns:
            stats["top_colours"] = self.df["base_colour"].value_counts().head(8).to_dict()

        return stats

    def save_cleaned_metadata(self, output_path: Optional[str | Path] = None) -> Path:
        """Saves the cleaned and validated dataframe to CSV."""
        out = Path(output_path) if output_path else CLEANED_METADATA_CSV
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(out, index=False)
        print(f"[FashionDataset] Saved {len(self.df)} cleaned catalog items to: {out}")
        return out
