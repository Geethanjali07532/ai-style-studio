"""
AI Outfit Recommendation Engine.
Module 10: Outfit Recommendation Engine.

Generates complete, stylistically harmonious multi-item outfits
(Top + Bottom + Shoes + Accessory/Outerwear, or Dress + Shoes + Accessory)
using candidate pruning, beam search, and global outfit cohesion scoring.
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
from src.style_matcher import StyleMatcher


class OutfitRecommender:
    """
    Multi-item outfit generation engine.
    Assembles complete, wearable fashion looks from seed garments or occasion prompts.
    """

    def __init__(
        self,
        style_matcher: Optional[StyleMatcher] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        if style_matcher is not None:
            self.matcher = style_matcher
            self.manager = style_matcher.manager
        elif embedding_manager is not None:
            self.manager = embedding_manager
            self.matcher = StyleMatcher(embedding_manager=embedding_manager)
        else:
            self.manager = EmbeddingManager()
            self.matcher = StyleMatcher(embedding_manager=self.manager)

    def score_outfit(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes the global outfit cohesion score across all pairwise combinations of items.

        Returns:
            Dict containing 'cohesion_score', 'pairwise_scores', and 'color_palette'.
        """
        if len(items) < 2:
            return {
                "cohesion_score": 1.0,
                "pairwise_breakdown": [],
                "color_palette": [item.get("baseColour", "Unknown") for item in items],
            }

        pairwise_scores = []
        total_score = 0.0
        n_pairs = 0

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item_a = items[i]
                item_b = items[j]
                res = self.matcher.score_pairing(item_a, item_b)
                pairwise_scores.append({
                    "item_a_id": item_a.get("id"),
                    "item_a_part": item_a.get("outfit_part"),
                    "item_b_id": item_b.get("id"),
                    "item_b_part": item_b.get("outfit_part"),
                    "score": res["composite_score"],
                    "color_harmony": res["color_harmony"],
                    "visual_sim": res["visual_similarity"],
                })
                total_score += res["composite_score"]
                n_pairs += 1

        avg_cohesion = total_score / max(1, n_pairs)
        palette = [str(item.get("baseColour", "")) for item in items if item.get("baseColour")]

        return {
            "cohesion_score": round(float(avg_cohesion), 4),
            "num_items": len(items),
            "pairwise_breakdown": pairwise_scores,
            "color_palette": palette,
        }

    def generate_outfit_from_seed(
        self,
        seed_item_id: Union[str, int],
        include_accessory: bool = True,
        include_outerwear: bool = False,
        beam_width: int = 4,
        top_k_outfits: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Builds complete outfits centered around a given seed garment.

        Args:
            seed_item_id: Catalog ID of the starting garment.
            include_accessory: Whether to add a matching accessory/bag/watch.
            include_outerwear: Whether to add a layer (jacket/blazer).
            beam_width: Candidate search beam size at each outfit slot.
            top_k_outfits: Number of best complete outfits to return.

        Returns:
            List of complete outfit dicts, ranked by global cohesion score.
        """
        if isinstance(seed_item_id, dict):
            seed_item = seed_item_id
            str_id = str(seed_item.get("id", "uploaded_seed"))
        else:
            str_id = str(seed_item_id)
            seed_row = self.manager.index_df[self.manager.index_df["id"] == str_id]
            if len(seed_row) == 0:
                raise KeyError(f"Seed garment ID {str_id} not found in catalog index.")
            seed_item = seed_row.iloc[0].to_dict()

        canonical_cat = seed_item.get("canonical_category", seed_item.get("category", ""))
        seed_part = seed_item.get("outfit_part", seed_item.get("part", ""))
        gender = seed_item.get("gender")

        # Determine target template slots
        is_dress = str(canonical_cat).lower() == "dress"

        if is_dress:
            # Dress serves as complete body garment (No bottom needed)
            slots_to_fill = ["shoes"]
            if include_accessory:
                slots_to_fill.append("accessory")
            if include_outerwear:
                slots_to_fill.append("outerwear")
        else:
            # Standard 2-piece / 3-piece / 4-piece outfit
            if seed_part == "top":
                slots_to_fill = ["bottom", "shoes"]
            elif seed_part == "bottom":
                slots_to_fill = ["top", "shoes"]
            elif seed_part == "shoes":
                slots_to_fill = ["top", "bottom"]
            elif seed_part == "outerwear":
                slots_to_fill = ["top", "bottom", "shoes"]
            else:  # accessory
                slots_to_fill = ["top", "bottom", "shoes"]

            if include_accessory and seed_part != "accessory" and "accessory" not in slots_to_fill:
                slots_to_fill.append("accessory")
            if include_outerwear and seed_part != "outerwear" and "outerwear" not in slots_to_fill:
                slots_to_fill.append("outerwear")

        # Beam search outfit assembly
        # Each beam is a list of items: [seed_item]
        current_beams = [[seed_item]]

        for slot_part in slots_to_fill:
            next_beams = []
            for beam in current_beams:
                # Find candidates for this slot compatible with the seed
                anchor = beam[0]
                candidates = self.matcher.find_compatible_garments(
                    query_item_id=anchor,
                    target_part=slot_part,
                    top_k=beam_width,
                    enforce_gender=True,
                )

                if not candidates:
                    # Fallback
                    anchor_vec = anchor.get("embedding_vector", anchor.get("embedding_vec"))
                    if anchor_vec is not None:
                        candidates = self.manager.search_by_vector(
                            anchor_vec,
                            top_k=beam_width,
                            outfit_part=slot_part,
                        )
                    else:
                        candidates = self.manager.search_by_item_id(
                            anchor.get("id", str_id),
                            top_k=beam_width,
                            outfit_part=slot_part,
                        )

                for cand in candidates:
                    # Prevent duplicate IDs
                    if any(cand["id"] == existing["id"] for existing in beam):
                        continue
                    extended_beam = beam + [cand]
                    next_beams.append(extended_beam)

            if next_beams:
                # Score each beam and retain top-N beams
                scored_beams = []
                for b in next_beams:
                    sc = self.score_outfit(b)["cohesion_score"]
                    scored_beams.append((sc, b))
                scored_beams.sort(key=lambda x: x[0], reverse=True)
                # Keep top beam_width * 2 candidates
                current_beams = [b for _, b in scored_beams[: beam_width * 3]]
            else:
                # Could not fill slot, continue with current beam
                pass

        # Final ranking of complete outfits with deduplication
        final_outfits = []
        seen_outfits = set()
        for beam in current_beams:
            sig = tuple(sorted(str(it.get("id")) for it in beam))
            if sig in seen_outfits:
                continue
            seen_outfits.add(sig)
            assessment = self.score_outfit(beam)
            final_outfits.append({
                "outfit_id": f"outfit_{seed_item['id']}_{len(final_outfits)+1}",
                "cohesion_score": assessment["cohesion_score"],
                "num_items": len(beam),
                "items": beam,
                "color_palette": assessment["color_palette"],
                "pairwise_breakdown": assessment["pairwise_breakdown"],
            })

        # Sort descending by cohesion score
        final_outfits.sort(key=lambda x: x["cohesion_score"], reverse=True)
        return final_outfits[:top_k_outfits]

    def generate_outfit_by_occasion(
        self,
        occasion: str = "Casual",
        gender: str = "Men",
        include_accessory: bool = True,
        top_k_outfits: int = 3,
        seed_top_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates complete outfits curated for a specific occasion and demographic.

        Args:
            occasion: 'Casual', 'Formal', 'Sports', 'Party', 'Ethnic'.
            gender: 'Men', 'Women', 'Unisex'.
            include_accessory: Whether to accessorize the look.
            top_k_outfits: Number of diverse outfits to return.
            seed_top_category: Optional specific top ('Shirt', 'T-Shirt', 'Dress').

        Returns:
            List of complete recommended outfits.
        """
        # Map user-facing occasions to dataset usages & preferred categories
        occ_lower = occasion.lower() if occasion else "casual"
        OCCASION_MAPPING = {
            "college": (["casual", "sports"], ["T-Shirt", "Jeans", "Shirt"]),
            "office": (["formal", "smart casual", "casual"], ["Shirt", "Trousers"]),
            "party": (["party", "casual", "ethnic"], ["Dress", "Shirt", "T-Shirt"]),
            "formal": (["formal", "smart casual"], ["Shirt", "Trousers"]),
            "casual": (["casual"], ["T-Shirt", "Shirt", "Dress"]),
            "sports": (["sports", "casual"], ["T-Shirt", "Track Pants", "Shorts"]),
            "ethnic": (["ethnic", "party"], ["Kurtas", "Ethnic"]),
        }

        target_usages, preferred_cats = OCCASION_MAPPING.get(occ_lower, ([occ_lower], []))

        # Filter candidate tops/dresses matching occasion & gender
        df = self.manager.index_df
        mask = (df["outfit_part"].isin(["top", "dress"])) & (df["gender"].isin([gender, "Unisex"]))
        if target_usages:
            mask &= (df["usage"].str.lower().isin(target_usages))
        if seed_top_category:
            mask &= (df["canonical_category"] == seed_top_category)
        elif preferred_cats:
            cat_mask = mask & (df["canonical_category"].isin(preferred_cats))
            if len(df[cat_mask]) > 0:
                mask = cat_mask

        candidate_seeds = df[mask]
        if len(candidate_seeds) == 0:
            # Relax occasion if no exact match
            mask_relaxed = (df["outfit_part"].isin(["top", "dress"])) & (df["gender"].isin([gender, "Unisex"]))
            candidate_seeds = df[mask_relaxed]

        if len(candidate_seeds) == 0:
            return []

        # Sample diverse seed tops across different colors
        sample_seeds = candidate_seeds.drop_duplicates(subset=["baseColour"]).head(top_k_outfits * 2)
        if len(sample_seeds) < top_k_outfits:
            sample_seeds = candidate_seeds.head(top_k_outfits * 2)

        all_candidate_outfits = []
        for _, seed_row in sample_seeds.iterrows():
            outfits = self.generate_outfit_from_seed(
                seed_item_id=seed_row["id"],
                include_accessory=include_accessory,
                top_k_outfits=1,
            )
            all_candidate_outfits.extend(outfits)

        # Sort and deduplicate all generated outfits
        all_candidate_outfits.sort(key=lambda x: x["cohesion_score"], reverse=True)
        unique_occ_outfits = []
        seen_occ = set()
        for out in all_candidate_outfits:
            sig = tuple(sorted(str(it.get("id")) for it in out.get("items", [])))
            if sig not in seen_occ:
                seen_occ.add(sig)
                unique_occ_outfits.append(out)
        return unique_occ_outfits[:top_k_outfits]

    def substitute_item_in_outfit(
        self,
        outfit_items: List[Dict[str, Any]],
        replace_item_id: Union[str, int],
        top_k_alternatives: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Allows user to swap out one item from an outfit (e.g. swap jeans for chinos)
        while preserving maximal outfit harmony.

        Returns:
            List of candidate replacement items with updated outfit cohesion scores.
        """
        str_id = str(replace_item_id)
        target_item = None
        remaining_items = []
        for item in outfit_items:
            if str(item.get("id")) == str_id:
                target_item = item
            else:
                remaining_items.append(item)

        if target_item is None:
            raise KeyError(f"Item ID {str_id} not found in current outfit.")

        # Find candidates of same part
        target_part = target_item.get("outfit_part")
        target_cat = target_item.get("canonical_category")
        gender = target_item.get("gender")

        # Query candidates of the same outfit part (e.g. Sneakers or Shoes for footwear)
        cand_items = self.manager.search_by_item_id(
            item_id=str_id,
            top_k=top_k_alternatives * 3,
            category=None,
            outfit_part=target_part,
            gender=gender,
            exclude_self=True,
        )

        scored_replacements = []
        for cand in cand_items:
            hypothetical_outfit = remaining_items + [cand]
            score_data = self.score_outfit(hypothetical_outfit)
            cand_augmented = {
                **cand,
                "new_outfit_cohesion": score_data["cohesion_score"],
                "color_harmony_with_seed": self.matcher.compute_color_compatibility(
                    remaining_items[0].get("baseColour"), cand.get("baseColour")
                )[1],
            }
            scored_replacements.append(cand_augmented)

        scored_replacements.sort(key=lambda x: x["new_outfit_cohesion"], reverse=True)
        return scored_replacements[:top_k_alternatives]
