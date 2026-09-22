"""
Fashion Image Exploratory Data Analysis & Color Analytics Engine.
Module 5: Fashion Image Exploratory Analysis.

Provides K-Means dominant color extraction, HSV color space conversion,
fashion color harmony matching, multi-dimensional demographic crosstabs,
and automated EDA visualization generation.
"""
from __future__ import annotations
import argparse
import colorsys
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows if supported
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import seaborn as sns
from sklearn.cluster import KMeans

from src.config import (
    CLEANED_METADATA_CSV,
    EDA_FIGURES_DIR,
    EDA_SUMMARY_JSON,
    CLOTHING_CATEGORIES,
    OUTFIT_PARTS,
    RANDOM_STATE,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ColorExtractor:
    """
    Extracts dominant color palettes and analyzes color harmony for fashion garments.
    """

    @staticmethod
    def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Converts RGB [0, 255] to HSV:
        Hue in degrees [0, 360), Saturation in [0, 1], Value in [0, 1].
        """
        r, g, b = [c / 255.0 for c in rgb[:3]]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return round(h * 360.0, 1), round(s, 3), round(v, 3)

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Converts RGB tuple to uppercase hex code."""
        return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    @staticmethod
    def is_neutral_color(rgb: Tuple[int, int, int]) -> bool:
        """
        Checks if a color is considered a fashion neutral:
        Black, White, Gray, Beige, Charcoal, Navy, Off-white, Khaki.
        Neutrals can pair effortlessly with any accent color in outfit matching.
        """
        h, s, v = ColorExtractor.rgb_to_hsv(rgb)
        # Black or very dark
        if v < 0.18:
            return True
        # White or very light neutral
        if v > 0.88 and s < 0.15:
            return True
        # Gray (low saturation)
        if s < 0.18:
            return True
        # Beige / Khaki (Hue around yellow-orange [30-55], moderate saturation, high value)
        if 25 <= h <= 55 and s < 0.35 and v > 0.65:
            return True
        # Deep Navy Blue (Hue around 210-240, low brightness)
        if 210 <= h <= 245 and v < 0.35:
            return True

        return False

    @staticmethod
    def compute_color_harmony_type(color1_rgb: Tuple[int, int, int], color2_rgb: Tuple[int, int, int]) -> str:
        """
        Determines the fashion color harmony relationship between two garments:
        - 'neutral_match': One or both items are neutrals (versatile baseline).
        - 'monochromatic_or_analogous': Hues within 35 degrees (cohesive tonal look).
        - 'complementary': Hues roughly opposite on the color wheel (150-210 deg).
        - 'contrasting': Striking accent pairing.
        """
        if ColorExtractor.is_neutral_color(color1_rgb) or ColorExtractor.is_neutral_color(color2_rgb):
            return "neutral_match"

        h1, s1, v1 = ColorExtractor.rgb_to_hsv(color1_rgb)
        h2, s2, v2 = ColorExtractor.rgb_to_hsv(color2_rgb)

        hue_diff = abs(h1 - h2)
        hue_distance = min(hue_diff, 360.0 - hue_diff)

        if hue_distance <= 35.0:
            return "monochromatic_or_analogous"
        elif 145.0 <= hue_distance <= 215.0:
            return "complementary"
        else:
            return "contrasting"

    @classmethod
    def extract_dominant_colors(
        cls,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        k: int = 3,
        filter_background: bool = True,
        sample_size: int = 4000,
    ) -> List[Dict[str, Any]]:
        """
        Extracts the top K dominant colors from garment pixels using K-Means clustering.
        Automatically filters out near-white studio catalog backgrounds.
        """
        if isinstance(image_input, (str, Path)):
            with Image.open(str(image_input)) as img:
                img_rgb = img.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img_rgb = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img_rgb = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input: {type(image_input)}")

        # Downsample image for rapid K-Means clustering
        img_thumb = img_rgb.resize((64, 64), Image.Resampling.BILINEAR)
        pixels = np.array(img_thumb).reshape(-1, 3).astype(np.float32)

        if filter_background:
            # White or near-white background filter: R, G, B > 225 and close to grayscale
            diff_rg = np.abs(pixels[:, 0] - pixels[:, 1])
            diff_gb = np.abs(pixels[:, 1] - pixels[:, 2])
            is_white_bg = (pixels[:, 0] > 225) & (pixels[:, 1] > 225) & (pixels[:, 2] > 225) & (diff_rg < 18) & (diff_gb < 18)
            garment_pixels = pixels[~is_white_bg]

            # If background filtering leaves too few pixels, keep all
            if len(garment_pixels) > 100:
                pixels = garment_pixels

        # Subsample if pixel count is large
        if len(pixels) > sample_size:
            idx = np.random.choice(len(pixels), sample_size, replace=False)
            pixels = pixels[idx]

        actual_k = min(k, len(pixels))
        if actual_k <= 0:
            return [{"rgb": (128, 128, 128), "hex": "#808080", "percentage": 1.0, "hsv": (0, 0, 0.5), "is_neutral": True}]

        kmeans = KMeans(n_clusters=actual_k, random_state=RANDOM_STATE, n_init=4)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_

        # Calculate cluster percentages
        counts = np.bincount(labels, minlength=actual_k)
        total = len(labels)

        # Sort by frequency descending
        sorted_indices = np.argsort(-counts)

        results = []
        for idx in sorted_indices:
            rgb_tuple = tuple(int(round(c)) for c in centers[idx])
            pct = round(float(counts[idx] / total), 3)
            hsv_val = cls.rgb_to_hsv(rgb_tuple)
            results.append({
                "rgb": rgb_tuple,
                "hex": cls.rgb_to_hex(rgb_tuple),
                "percentage": pct,
                "hsv": hsv_val,
                "is_neutral": cls.is_neutral_color(rgb_tuple),
            })

        return results


class DatasetAnalyzer:
    """
    Computes statistical distributions, cross-tabulations, and visual analytics
    across the entire fashion catalog.
    """

    def __init__(self, metadata_path: Optional[Union[str, Path]] = None):
        csv_path = Path(metadata_path) if metadata_path else CLEANED_METADATA_CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"Clean metadata CSV not found at: {csv_path}")
        self.df = pd.read_csv(csv_path)

    def compute_all_metrics(self, color_sample_size: int = 150) -> Dict[str, Any]:
        """
        Runs comprehensive analysis: category distributions, gender breakdown,
        usage patterns, and sample dominant color clustering.
        """
        df = self.df
        total_items = len(df)

        metrics: Dict[str, Any] = {
            "total_items": total_items,
            "category_counts": df["canonical_category"].value_counts().to_dict(),
            "outfit_part_counts": df["outfit_part"].value_counts().to_dict(),
        }

        # Demographics
        if "gender" in df.columns:
            metrics["gender_counts"] = df["gender"].value_counts().to_dict()
            metrics["gender_by_category"] = (
                pd.crosstab(df["gender"], df["canonical_category"]).to_dict()
            )

        # Usage / Context
        if "usage" in df.columns:
            metrics["usage_counts"] = df["usage"].value_counts().head(8).to_dict()
            metrics["usage_by_category"] = (
                pd.crosstab(df["usage"], df["canonical_category"]).to_dict()
            )

        # Season
        if "season" in df.columns:
            metrics["season_counts"] = df["season"].value_counts().to_dict()

        # Dominant Colors from Metadata
        if "base_colour" in df.columns:
            metrics["top_metadata_colours"] = df["base_colour"].value_counts().head(12).to_dict()

        # Sample pixel-based dominant color extraction
        print(f"[*] Extracting pixel dominant colors on sample of {color_sample_size} garments...")
        valid_sample = df[df["image_path"].apply(os.path.exists)].sample(
            min(color_sample_size, len(df)), random_state=RANDOM_STATE
        )

        extracted_palettes = []
        neutral_count = 0
        total_extracted = 0

        for _, row in valid_sample.iterrows():
            try:
                palette = ColorExtractor.extract_dominant_colors(row["image_path"], k=3)
                if palette:
                    primary = palette[0]
                    extracted_palettes.append({
                        "id": str(row.get("id")),
                        "category": str(row.get("canonical_category")),
                        "primary_hex": primary["hex"],
                        "primary_rgb": primary["rgb"],
                        "is_neutral": primary["is_neutral"],
                    })
                    if primary["is_neutral"]:
                        neutral_count += 1
                    total_extracted += 1
            except Exception:
                continue

        metrics["pixel_color_sample"] = {
            "sampled_count": total_extracted,
            "neutral_ratio": round(neutral_count / max(1, total_extracted), 3),
            "sample_palettes": extracted_palettes[:20],
        }

        return metrics

    def generate_figures(self, output_dir: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Generates publication-quality EDA figures and saves them to reports/eda_figures/.
        """
        out_dir = Path(output_dir) if output_dir else EDA_FIGURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_plots: List[Path] = []

        sns.set_theme(style="whitegrid", palette="muted")
        df = self.df

        # Figure 1: Category Distribution Bar Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        cat_counts = df["canonical_category"].value_counts()
        sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="viridis", ax=ax)
        ax.set_title("Catalog Garments by Canonical Category", fontsize=13, fontweight="bold")
        ax.set_xlabel("Item Count", fontsize=11)
        for i, v in enumerate(cat_counts.values):
            ax.text(v + 150, i, f"{v:,}", va="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        p1 = out_dir / "category_distribution.png"
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        saved_plots.append(p1)

        # Figure 2: Gender vs Category Heatmap
        if "gender" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            top_genders = df["gender"].value_counts().head(4).index
            ct = pd.crosstab(df[df["gender"].isin(top_genders)]["gender"], df["canonical_category"])
            sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax)
            ax.set_title("Gender Breakdown Across Clothing Categories", fontsize=13, fontweight="bold")
            plt.tight_layout()
            p2 = out_dir / "gender_by_category_heatmap.png"
            fig.savefig(p2, dpi=200)
            plt.close(fig)
            saved_plots.append(p2)

        # Figure 3: Usage / Occasion Breakdown
        if "usage" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            usage_counts = df["usage"].value_counts().head(6)
            sns.barplot(x=usage_counts.index, y=usage_counts.values, palette="mako", ax=ax)
            ax.set_title("Style & Usage Occasion Distribution", fontsize=13, fontweight="bold")
            ax.set_ylabel("Item Count")
            plt.xticks(rotation=20)
            plt.tight_layout()
            p3 = out_dir / "usage_distribution.png"
            fig.savefig(p3, dpi=200)
            plt.close(fig)
            saved_plots.append(p3)

        # Figure 4: Dominant Color Swatches Grid
        if "base_colour" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            color_counts = df["base_colour"].value_counts().head(10)
            sns.barplot(x=color_counts.values, y=color_counts.index, palette="rocket", ax=ax)
            ax.set_title("Top 10 Garment Colors (Metadata)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Item Count")
            plt.tight_layout()
            p4 = out_dir / "dominant_colors_palette.png"
            fig.savefig(p4, dpi=200)
            plt.close(fig)
            saved_plots.append(p4)

        return saved_plots

    def save_summary_json(self, metrics: Dict[str, Any], output_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves metrics dictionary to JSON."""
        out_p = Path(output_path) if output_path else EDA_SUMMARY_JSON
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[*] Saved EDA summary to: {out_p}")
        return out_p


def main():
    parser = argparse.ArgumentParser(description="Module 5: Fashion Image Exploratory Analysis")
    parser.add_argument("--sample-size", type=int, default=150, help="Number of images to sample for pixel color clustering")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  AI Outfit Recommendation - Module 5: Fashion Image Exploratory Analysis")
    print("=" * 75 + "\n")

    analyzer = DatasetAnalyzer()
    metrics = analyzer.compute_all_metrics(color_sample_size=args.sample_size)
    analyzer.save_summary_json(metrics)
    plots = analyzer.generate_figures()

    print(f"\n[OK] Generated {len(plots)} publication figures in: {EDA_FIGURES_DIR}")
    for p in plots:
        print(f"  * {p.name}")

    print("\n[SUCCESS] Module 5 Exploratory Data Analysis Complete!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
