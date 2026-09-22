"""
Clothing Similarity and Style Matching Engine.
Module 9: Clothing Similarity & Style Matching.

Integrates deep CNN visual embeddings, fashion color harmony theory,
demographic filtering, and occasion/usage alignment into a composite
clothing compatibility and outfit matching system.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.config import (
    CATEGORY_TO_OUTFIT_PART,
    CLOTHING_CATEGORIES,
    OUTFIT_PARTS,
)
from src.embeddings import EmbeddingManager


# --- Fashion Theory Constants ---

# Neutral colors serve as versatile base anchors compatible with almost any color
NEUTRAL_COLORS = {
    "black",
    "white",
    "grey",
    "gray",
    "navy blue",
    "navy",
    "beige",
    "cream",
    "khaki",
    "brown",
    "tan",
    "silver",
    "charcoal",
    "off white",
}

# Warm and Cool Color groupings for harmonious palette matching
WARM_COLORS = {"red", "maroon", "burgundy", "orange", "rust", "yellow", "mustard", "gold", "peach", "coral", "pink"}
COOL_COLORS = {"blue", "teal", "turquoise", "cyan", "green", "olive", "sea green", "purple", "lavender", "violet"}

# Complementary / high-contrast pairings widely celebrated in styling
COMPLEMENTARY_PAIRS = {
    ("blue", "orange"), ("orange", "blue"),
    ("navy blue", "tan"), ("tan", "navy blue"),
    ("navy", "tan"), ("tan", "navy"),
    ("navy blue", "mustard"), ("mustard", "navy blue"),
    ("olive", "maroon"), ("maroon", "olive"),
    ("pink", "grey"), ("grey", "pink"),
    ("yellow", "grey"), ("grey", "yellow"),
}

# Occasion / Usage compatibility lookup matrix
USAGE_COMPATIBILITY_SCORES = {
    ("casual", "casual"): 1.0,
    ("casual", "smart casual"): 0.90,
    ("smart casual", "casual"): 0.90,
    ("formal", "formal"): 1.0,
    ("formal", "party"): 0.85,
    ("party", "formal"): 0.85,
    ("party", "party"): 1.0,
    ("sports", "sports"): 1.0,
    ("casual", "sports"): 0.70,
    ("sports", "casual"): 0.70,
    ("ethnic", "ethnic"): 1.0,
    ("ethnic", "casual"): 0.60,
    ("formal", "sports"): 0.20,
    ("sports", "formal"): 0.20,
    ("ethnic", "sports"): 0.15,
    ("sports", "ethnic"): 0.15,
}


class StyleMatcher:
    """
    Computes visual similarity, color harmony, occasion alignment,
    and composite outfit compatibility between fashion garments.
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        weight_visual: float = 0.45,
        weight_color: float = 0.35,
        weight_usage: float = 0.20,
    ):
        """
        Args:
            embedding_manager: Pre-loaded EmbeddingManager instance.
            weight_visual: Importance of deep visual embedding cosine similarity.
            weight_color: Importance of color harmony.
            weight_usage: Importance of occasion/usage alignment.
        """
        self.manager = embedding_manager if embedding_manager else EmbeddingManager()
        self.w_visual = weight_visual
        self.w_color = weight_color
        self.w_usage = weight_usage

    @staticmethod
    def compute_color_compatibility(color1: Optional[str], color2: Optional[str]) -> Tuple[float, str]:
        """
        Evaluates color harmony between two garment base colors.

        Returns:
            Tuple of (score [0.0 to 1.0], harmony_category_str).
        """
        if not color1 or not color2 or pd.isna(color1) or pd.isna(color2):
            return 0.70, "unknown_neutral_fallback"

        c1 = str(color1).strip().lower()
        c2 = str(color2).strip().lower()

        # Identical color (Monochromatic matching)
        if c1 == c2:
            return 0.92, "monochromatic"

        # Neutral Anchor Rule (Black/White/Navy/Grey pair with everything)
        if c1 in NEUTRAL_COLORS or c2 in NEUTRAL_COLORS:
            # Dual neutrals (e.g. Black + White, Navy + Khaki) are exceptionally versatile
            if c1 in NEUTRAL_COLORS and c2 in NEUTRAL_COLORS:
                return 0.98, "dual_neutral_classic"
            return 0.90, "neutral_accent_match"

        # Established complementary pairings
        if (c1, c2) in COMPLEMENTARY_PAIRS or (c2, c1) in COMPLEMENTARY_PAIRS:
            return 0.88, "complementary_contrast"

        # Warm-Warm or Cool-Cool palette harmony
        if (c1 in WARM_COLORS and c2 in WARM_COLORS) or (c1 in COOL_COLORS and c2 in COOL_COLORS):
            return 0.82, "analogous_temperature_match"

        # Warm with Cool contrast
        return 0.65, "contrasting_statement"

    @staticmethod
    def compute_usage_compatibility(usage1: Optional[str], usage2: Optional[str]) -> Tuple[float, str]:
        """
        Evaluates occasion and lifestyle compatibility between two garments.
        """
        if not usage1 or not usage2 or pd.isna(usage1) or pd.isna(usage2):
            return 0.80, "unspecified_compatible"

        u1 = str(usage1).strip().lower()
        u2 = str(usage2).strip().lower()

        pair = (u1, u2)
        if pair in USAGE_COMPATIBILITY_SCORES:
            score = USAGE_COMPATIBILITY_SCORES[pair]
            return score, f"{u1}_and_{u2}"

        # Default fallback
        if u1 == u2:
            return 1.0, f"exact_{u1}"
        return 0.65, "moderate_alignment"

    @staticmethod
    def is_gender_compatible(gender1: Optional[str], gender2: Optional[str]) -> bool:
        """Checks whether two items share demographic alignment."""
        if not gender1 or not gender2:
            return True
        g1, g2 = str(gender1).capitalize(), str(gender2).capitalize()
        if g1 in ("Unisex", "All") or g2 in ("Unisex", "All"):
            return True
        return g1 == g2

    def score_pairing(
        self,
        item_a: Union[Dict[str, Any], pd.Series],
        item_b: Union[Dict[str, Any], pd.Series],
        visual_similarity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates a comprehensive compatibility breakdown between two garments.
        """
        # 1. Visual Similarity (from precomputed embedding cosine or lookup)
        if visual_similarity is None:
            vec_a = item_a.get("embedding_vector") if item_a.get("embedding_vector") is not None else item_a.get("embedding_vec")
            if vec_a is None:
                id_a = str(item_a.get("id", ""))
                vec_a = self.manager.get_embedding(id_a) if id_a else None

            vec_b = item_b.get("embedding_vector") if item_b.get("embedding_vector") is not None else item_b.get("embedding_vec")
            if vec_b is None:
                id_b = str(item_b.get("id", ""))
                vec_b = self.manager.get_embedding(id_b) if id_b else None

            if vec_a is not None and vec_b is not None:
                # Dot product of L2 normalized vectors
                visual_similarity = float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))
            else:
                visual_similarity = 0.50

        # Rescale visual similarity from [-1, 1] or [0, 1] to positive weight
        v_score = max(0.0, float(visual_similarity))

        # 2. Color Harmony
        color_a = item_a.get("baseColour", item_a.get("dominant_color"))
        color_b = item_b.get("baseColour", item_b.get("dominant_color"))
        c_score, harmony_type = self.compute_color_compatibility(color_a, color_b)

        # 3. Occasion / Usage Alignment
        usage_a = item_a.get("usage", item_a.get("occasion"))
        usage_b = item_b.get("usage", item_b.get("occasion"))
        u_score, usage_rel = self.compute_usage_compatibility(usage_a, usage_b)

        # 4. Gender compatibility multiplier
        gender_a = item_a.get("gender")
        gender_b = item_b.get("gender")
        gender_ok = self.is_gender_compatible(gender_a, gender_b)
        gender_penalty = 1.0 if gender_ok else 0.40

        # Composite weighted score
        composite_score = (
            self.w_visual * v_score +
            self.w_color * c_score +
            self.w_usage * u_score
        ) * gender_penalty

        return {
            "composite_score": round(float(composite_score), 4),
            "visual_similarity": round(float(v_score), 4),
            "color_score": round(float(c_score), 4),
            "color_harmony": harmony_type,
            "usage_score": round(float(u_score), 4),
            "usage_relation": usage_rel,
            "gender_compatible": gender_ok,
        }

    def find_compatible_garments(
        self,
        query_item_id: Union[str, int, Dict[str, Any]],
        target_part: Optional[str] = None,
        target_category: Optional[str] = None,
        top_k: int = 6,
        candidate_pool_size: int = 50,
        enforce_gender: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves and ranks the top-K most compatible complementary garments
        for a given query item (either catalog ID or custom item dict).
        """
        if isinstance(query_item_id, dict):
            query_dict = query_item_id
            str_id = str(query_dict.get("id", "custom_uploaded_item"))
            query_vec = query_dict.get("embedding_vector")
            if query_vec is None:
                query_vec = query_dict.get("embedding_vec")
            if query_vec is None and "id" in query_dict:
                query_vec = self.manager.get_embedding(str(query_dict["id"]))
            if query_vec is None:
                raise ValueError("Custom query item dictionary must contain an 'embedding_vector'.")
        else:
            str_id = str(query_item_id)
            query_vec = self.manager.get_embedding(str_id)
            if query_vec is None:
                raise KeyError(f"Item ID {str_id} not found in visual embedding index.")

            query_row = self.manager.index_df[self.manager.index_df["id"] == str_id]
            if len(query_row) == 0:
                raise KeyError(f"Item ID {str_id} not found in index table.")
            query_dict = query_row.iloc[0].to_dict()

        query_gender = query_dict.get("gender") if enforce_gender else None

        # 1. Fetch initial candidate pool via fast vector similarity
        candidates = self.manager.search_by_vector(
            query_vec=query_vec,
            top_k=candidate_pool_size,
            category=target_category,
            outfit_part=target_part,
            gender=query_gender,
            exclude_ids=[str_id],
        )

        if not candidates:
            return []

        # 2. Re-rank candidate pool using composite compatibility scoring
        scored_candidates = []
        for cand in candidates:
            sim = cand.get("similarity_score", 0.5)
            breakdown = self.score_pairing(query_dict, cand, visual_similarity=sim)
            merged = {**cand, **breakdown}
            scored_candidates.append(merged)

        # 3. Sort by composite compatibility descending
        scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        return scored_candidates[:top_k]

    def find_visually_similar_garments(
        self,
        query_item_id: Union[str, int],
        top_k: int = 6,
        same_category_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves alternative or visually near-identical items within the same category.
        """
        str_id = str(query_item_id)
        query_row = self.manager.index_df[self.manager.index_df["id"] == str_id]
        if len(query_row) == 0:
            raise KeyError(f"Item ID {str_id} not found in index table.")
        query_dict = query_row.iloc[0].to_dict()

        cat = query_dict.get("canonical_category") if same_category_only else None
        return self.manager.search_by_item_id(
            item_id=str_id,
            top_k=top_k,
            category=cat,
            exclude_self=True,
        )
