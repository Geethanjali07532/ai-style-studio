"""
User Preference and Personalization Engine.
Module 11: User Preference & Personalization.

Models individual user style profiles, demographic constraints,
color affinities, interaction history, and personalized outfit re-ranking.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
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

from src.embeddings import EmbeddingManager
from src.style_matcher import StyleMatcher
from src.recommender import OutfitRecommender


@dataclass
class UserProfile:
    """
    Represents an individual user's fashion style preferences, demographic identity,
    favorite/disliked colors and categories, and interaction history.
    """
    user_id: str
    gender: str = "Men"  # "Men", "Women", "Unisex"
    archetype: str = "Casual"  # "Minimalist", "Smart Casual", "Streetwear", "Vibrant"
    preferred_occasions: List[str] = field(default_factory=lambda: ["Casual"])
    favorite_colors: List[str] = field(default_factory=list)
    disliked_colors: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)
    disliked_categories: List[str] = field(default_factory=list)
    liked_item_ids: List[str] = field(default_factory=list)
    disliked_item_ids: List[str] = field(default_factory=list)

    def add_interaction(self, item_id: Union[str, int], interaction_type: str = "like"):
        """Records a user interaction (like, save, dislike)."""
        str_id = str(item_id)
        if interaction_type.lower() in ("like", "save", "buy"):
            if str_id not in self.liked_item_ids:
                self.liked_item_ids.append(str_id)
            if str_id in self.disliked_item_ids:
                self.disliked_item_ids.remove(str_id)
        elif interaction_type.lower() in ("dislike", "hide"):
            if str_id not in self.disliked_item_ids:
                self.disliked_item_ids.append(str_id)
            if str_id in self.liked_item_ids:
                self.liked_item_ids.remove(str_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "gender": self.gender,
            "archetype": self.archetype,
            "preferred_occasions": self.preferred_occasions,
            "favorite_colors": self.favorite_colors,
            "disliked_colors": self.disliked_colors,
            "preferred_categories": self.preferred_categories,
            "disliked_categories": self.disliked_categories,
            "liked_item_ids": self.liked_item_ids,
            "disliked_item_ids": self.disliked_item_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserProfile:
        return cls(**data)


# Standard Fashion Style Archetype Presets
STYLE_ARCHETYPES = {
    "Minimalist": {
        "preferred_occasions": ["Casual", "Smart Casual"],
        "favorite_colors": ["Black", "White", "Grey", "Navy Blue"],
        "disliked_colors": ["Pink", "Yellow", "Orange", "Mustard"],
        "preferred_categories": ["T-Shirt", "Shirt", "Jeans", "Trousers", "Sneakers"],
        "disliked_categories": [],
    },
    "Smart Casual": {
        "preferred_occasions": ["Smart Casual", "Formal", "Casual"],
        "favorite_colors": ["Navy Blue", "White", "Beige", "Blue", "Brown", "Grey"],
        "disliked_colors": ["Fluorescent Green", "Orange"],
        "preferred_categories": ["Shirt", "Trousers", "Jacket", "Shoes"],
        "disliked_categories": [],
    },
    "Vibrant Eclectic": {
        "preferred_occasions": ["Casual", "Party"],
        "favorite_colors": ["Red", "Blue", "Yellow", "Pink", "Orange", "Teal"],
        "disliked_colors": [],
        "preferred_categories": ["T-Shirt", "Dress", "Skirt", "Sneakers", "Accessories"],
        "disliked_categories": [],
    },
    "Athletic Streetwear": {
        "preferred_occasions": ["Sports", "Casual"],
        "favorite_colors": ["Black", "Grey", "Red", "Blue", "White"],
        "disliked_colors": ["Beige", "Cream"],
        "preferred_categories": ["T-Shirt", "Trousers", "Sneakers", "Accessories"],
        "disliked_categories": ["Formal Shoes", "Skirt"],
    },
}


def create_preset_profile(archetype: str, user_id: str, gender: str = "Men") -> UserProfile:
    """Factory helper to generate pre-configured persona profiles."""
    if archetype not in STYLE_ARCHETYPES:
        raise ValueError(f"Unknown archetype: {archetype}. Choose from {list(STYLE_ARCHETYPES.keys())}")

    preset = STYLE_ARCHETYPES[archetype]
    return UserProfile(
        user_id=user_id,
        gender=gender,
        archetype=archetype,
        preferred_occasions=preset["preferred_occasions"].copy(),
        favorite_colors=preset["favorite_colors"].copy(),
        disliked_colors=preset["disliked_colors"].copy(),
        preferred_categories=preset["preferred_categories"].copy(),
        disliked_categories=preset["disliked_categories"].copy(),
    )


class PersonalizedRecommender:
    """
    Personalized outfit and clothing recommendation system.
    Adjusts objective aesthetic cohesion with subjective user style affinities.
    """

    def __init__(
        self,
        outfit_recommender: Optional[OutfitRecommender] = None,
        personalization_weight: float = 0.35,
    ):
        """
        Args:
            outfit_recommender: Underlying OutfitRecommender instance.
            personalization_weight: Alpha weight for personalization (0.0 = purely objective, 1.0 = purely user preference).
        """
        self.recommender = outfit_recommender if outfit_recommender else OutfitRecommender()
        self.manager = self.recommender.manager
        self.matcher = self.recommender.matcher
        self.alpha = personalization_weight

    def compute_item_affinity(self, item: Dict[str, Any], profile: UserProfile) -> float:
        """
        Computes a user affinity score for a single clothing item in [0.0, 1.0].
        Considers color preferences, category affinities, and interaction history.
        """
        # Base neutral affinity
        affinity = 0.50

        # 1. Color Affinity
        item_color = str(item.get("baseColour", "")).strip().lower()
        fav_colors = [c.strip().lower() for c in profile.favorite_colors]
        disliked_colors = [c.strip().lower() for c in profile.disliked_colors]

        if item_color in fav_colors:
            affinity += 0.20
        elif item_color in disliked_colors:
            affinity -= 0.30

        # 2. Category Affinity
        item_cat = item.get("canonical_category", "")
        if item_cat in profile.preferred_categories:
            affinity += 0.15
        elif item_cat in profile.disliked_categories:
            affinity -= 0.40

        # 3. Occasion / Usage Affinity
        item_usage = str(item.get("usage", "")).strip().lower()
        user_usages = [u.strip().lower() for u in profile.preferred_occasions]
        if item_usage in user_usages:
            affinity += 0.10

        # 4. Interaction History Affinity (Visual similarity to previously liked garments)
        if profile.liked_item_ids:
            liked_sims = []
            item_id = str(item.get("id", ""))
            vec = item.get("embedding_vector") if item.get("embedding_vector") is not None else self.manager.get_embedding(item_id)
            if vec is not None:
                for liked_id in profile.liked_item_ids[-10:]:  # Last 10 likes
                    liked_vec = self.manager.get_embedding(liked_id)
                    if liked_vec is not None:
                        sim = float(np.dot(vec, liked_vec))
                        liked_sims.append(sim)
            if liked_sims:
                avg_liked_sim = np.mean(liked_sims)
                # Boost if visually similar to liked items
                affinity += 0.15 * max(0.0, avg_liked_sim)

        # 5. Severe dislike penalty
        if str(item.get("id", "")) in profile.disliked_item_ids:
            affinity = 0.05

        return float(np.clip(affinity, 0.0, 1.0))

    def score_personalized_outfit(
        self,
        outfit: Dict[str, Any],
        profile: UserProfile,
    ) -> Dict[str, Any]:
        """
        Re-scores a candidate outfit by fusing global aesthetic cohesion with user personal affinity.

        Formula:
            Final Score = (1 - alpha) * CohesionScore + alpha * MeanUserAffinity
        """
        base_cohesion = outfit.get("cohesion_score", 0.5)
        items = outfit.get("items", [])

        if not items:
            return {**outfit, "personalized_score": base_cohesion, "user_affinity": 0.5}

        # Compute affinities for each item in the outfit
        item_affinities = [self.compute_item_affinity(item, profile) for item in items]
        mean_affinity = float(np.mean(item_affinities))

        # Check if any item contains an explicitly disliked category or color
        has_disliked_category = any(
            item.get("canonical_category") in profile.disliked_categories for item in items
        )
        has_disliked_color = any(
            str(item.get("baseColour", "")).lower() in [c.lower() for c in profile.disliked_colors]
            for item in items
        )

        penalty = 0.60 if (has_disliked_category or has_disliked_color) else 1.0

        # Fused personalized score
        personalized_score = ((1.0 - self.alpha) * base_cohesion + self.alpha * mean_affinity) * penalty

        result = outfit.copy()
        result["base_cohesion_score"] = base_cohesion
        result["user_affinity"] = round(mean_affinity, 4)
        result["personalized_score"] = round(float(personalized_score), 4)
        result["has_disliked_element"] = has_disliked_category or has_disliked_color
        return result

    def recommend_personalized_outfits(
        self,
        profile: UserProfile,
        seed_item_id: Optional[Union[str, int, Dict[str, Any]]] = None,
        occasion: Optional[str] = None,
        candidate_pool: int = 6,
        top_k: int = 3,
        include_accessory: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generates and re-ranks complete outfits tailored to a user profile.

        Args:
            profile: Target UserProfile.
            seed_item_id: Optional starting garment ID or custom item dict.
            occasion: Optional occasion prompt (e.g. 'Casual', 'College', 'Office', 'Party', 'Formal').
            candidate_pool: Number of preliminary candidate outfits to score.
            top_k: Number of final personalized outfits to return.
            include_accessory: Whether to include accessories in the outfits.

        Returns:
            List of personalized outfits sorted by personalized_score descending.
        """
        target_occasion = occasion if occasion else (profile.preferred_occasions[0] if profile.preferred_occasions else "Casual")

        if seed_item_id is not None:
            # Generate from seed
            candidates = self.recommender.generate_outfit_from_seed(
                seed_item_id=seed_item_id,
                include_accessory=include_accessory,
                top_k_outfits=candidate_pool,
            )
        else:
            # Generate by occasion and demographic
            candidates = self.recommender.generate_outfit_by_occasion(
                occasion=target_occasion,
                gender=profile.gender,
                include_accessory=include_accessory,
                top_k_outfits=candidate_pool,
            )

        if not candidates:
            return []

        # Re-score each candidate with user personalization
        personalized_outfits = [
            self.score_personalized_outfit(outfit, profile) for outfit in candidates
        ]

        # Deduplicate outfits if any share the exact same item signatures
        unique_pers_outfits = []
        seen_pers = set()
        for out in personalized_outfits:
            sig = tuple(sorted(str(it.get("id")) for it in out.get("items", [])))
            if sig not in seen_pers:
                seen_pers.add(sig)
                unique_pers_outfits.append(out)

        # Sort descending by personalized score
        unique_pers_outfits.sort(key=lambda x: x["personalized_score"], reverse=True)
        return unique_pers_outfits[:top_k]

    def recommend_feed(
        self,
        profile: UserProfile,
        target_category: Optional[str] = None,
        target_part: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generates a personalized browsing feed of individual garments tailored to the user.
        """
        df = self.manager.index_df

        # Filter demographic
        mask = df["gender"].isin([profile.gender, "Unisex"])
        if target_category:
            mask &= (df["canonical_category"] == target_category)
        if target_part:
            mask &= (df["outfit_part"] == target_part)

        candidates_df = df[mask]
        if len(candidates_df) == 0:
            candidates_df = df

        # Sample candidate items and score affinities
        sample_size = min(len(candidates_df), 100)
        sampled_rows = candidates_df.sample(n=sample_size, random_state=42)

        scored_feed = []
        for _, row in sampled_rows.iterrows():
            item_dict = row.to_dict()
            affinity = self.compute_item_affinity(item_dict, profile)
            scored_feed.append({
                **item_dict,
                "user_affinity": round(affinity, 4),
            })

        # Rank by affinity
        scored_feed.sort(key=lambda x: x["user_affinity"], reverse=True)
        return scored_feed[:top_k]
