"""
Recommendation System Optimization Engine.
Module 13: Recommendation Optimization.

Provides threshold tuning (Precision vs Recall calibration),
multi-tier caching (LRU + categorical pre-indexing), and sub-millisecond latency profiling.
"""
from __future__ import annotations
import functools
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

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
    REPORTS_DIR,
    RANDOM_STATE,
)
from src.embeddings import EmbeddingManager
from src.style_matcher import StyleMatcher
from src.recommender import OutfitRecommender


@dataclass
class ThresholdTuningResult:
    """Stores grid search results for compatibility threshold calibration."""
    optimal_threshold: float
    max_f1_score: float
    precision_at_optimal: float
    recall_at_optimal: float
    grid_curve: List[Dict[str, float]]


class ThresholdTuner:
    """
    Calibrates recommendation acceptance thresholds to balance precision vs recall.
    """

    def __init__(self, style_matcher: Optional[StyleMatcher] = None):
        self.matcher = style_matcher if style_matcher else StyleMatcher()
        self.manager = self.matcher.manager

    def tune_compatibility_threshold(
        self,
        num_eval_pairs: int = 200,
        threshold_steps: int = 15,
        min_threshold: float = 0.50,
        max_threshold: float = 0.90,
        seed: int = RANDOM_STATE,
    ) -> ThresholdTuningResult:
        """
        Executes a grid search across candidate thresholds [min_threshold, max_threshold]
        to discover the optimal cutoff that balances precision and recall.
        """
        df = self.manager.index_df
        tops = df[df["outfit_part"] == "top"]
        bottoms = df[df["outfit_part"] == "bottom"]

        sample_n = min(num_eval_pairs, len(tops), len(bottoms))
        tops_sample = tops.sample(n=sample_n, random_state=seed).to_dict("records")
        bottoms_sample = bottoms.sample(n=sample_n, random_state=seed + 1).to_dict("records")

        # Ground truth definition:
        # A pair is defined as ground-truth compatible if they share gender alignment,
        # have compatible occasions, and have non-clashing colors.
        pair_data = []
        for t, b in zip(tops_sample, bottoms_sample):
            res = self.matcher.score_pairing(t, b)
            score = res["composite_score"]

            # Ground truth rule:
            is_gender_ok = StyleMatcher.is_gender_compatible(t.get("gender"), b.get("gender"))
            usage_score, _ = StyleMatcher.compute_usage_compatibility(t.get("usage"), b.get("usage"))
            color_score, _ = StyleMatcher.compute_color_compatibility(t.get("baseColour"), b.get("baseColour"))

            is_true_match = (is_gender_ok and usage_score >= 0.70 and color_score >= 0.75)
            pair_data.append((score, is_true_match))

        thresholds = np.linspace(min_threshold, max_threshold, threshold_steps)
        curve = []
        best_f1 = -1.0
        optimal_tau = min_threshold
        opt_p, opt_r = 0.0, 0.0

        for tau in thresholds:
            tp = sum(1 for s, true in pair_data if s >= tau and true)
            fp = sum(1 for s, true in pair_data if s >= tau and not true)
            fn = sum(1 for s, true in pair_data if s < tau and true)
            tn = sum(1 for s, true in pair_data if s < tau and not true)

            prec = tp / max(1, (tp + fp))
            rec = tp / max(1, (tp + fn))
            f1 = (2 * prec * rec) / max(1e-6, (prec + rec))
            rejection_rate = (fn + tn) / max(1, len(pair_data))

            entry = {
                "threshold": round(float(tau), 3),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "rejection_rate": round(float(rejection_rate), 4),
            }
            curve.append(entry)

            if f1 > best_f1:
                best_f1 = f1
                optimal_tau = float(tau)
                opt_p = prec
                opt_r = rec

        return ThresholdTuningResult(
            optimal_threshold=round(optimal_tau, 3),
            max_f1_score=round(best_f1, 4),
            precision_at_optimal=round(opt_p, 4),
            recall_at_optimal=round(opt_r, 4),
            grid_curve=curve,
        )


class OptimizedOutfitRecommender(OutfitRecommender):
    """
    High-throughput optimized recommendation engine featuring:
    1. Pre-partitioned categorical indices for zero-copy lookups.
    2. In-memory LRU cache for pairwise compatibility scores.
    3. Latency instrumentation measuring microsecond-level execution.
    """

    def __init__(
        self,
        style_matcher: Optional[StyleMatcher] = None,
        cache_capacity: int = 10000,
        compatibility_threshold: float = 0.72,
    ):
        super().__init__(style_matcher=style_matcher)
        self.cache_capacity = cache_capacity
        self.threshold = compatibility_threshold
        self.pair_cache: OrderedDict[Tuple[str, str], Dict[str, Any]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

        # Build categorical index partitions: (outfit_part, gender) -> DataFrame rows
        self._build_index_partitions()

    def _build_index_partitions(self):
        """Pre-partitions catalog dataframe into fast index buckets."""
        df = self.manager.index_df
        self.partitions: Dict[Tuple[str, str], pd.DataFrame] = {}

        parts = df["outfit_part"].unique()
        genders = df["gender"].unique()

        for part in parts:
            for gender in genders:
                sub_df = df[(df["outfit_part"] == part) & (df["gender"] == gender)]
                if len(sub_df) > 0:
                    self.partitions[(part, gender)] = sub_df

        print(f"[OptimizedRecommender] Pre-computed {len(self.partitions)} categorical index partitions.")

    def cached_score_pairing(
        self,
        item_a: Dict[str, Any],
        item_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculates or retrieves cached pairwise compatibility score.
        Eliminates duplicate dot-products during multi-beam search.
        """
        id_a, id_b = str(item_a.get("id")), str(item_b.get("id"))
        key = (id_a, id_b) if id_a < id_b else (id_b, id_a)

        if key in self.pair_cache:
            self.cache_hits += 1
            # Move to end (LRU)
            self.pair_cache.move_to_end(key)
            return self.pair_cache[key]

        self.cache_misses += 1
        res = self.matcher.score_pairing(item_a, item_b)

        if len(self.pair_cache) >= self.cache_capacity:
            self.pair_cache.popitem(last=False)  # Evict oldest
        self.pair_cache[key] = res
        return res

    def get_cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        hit_ratio = (self.cache_hits / max(1, total)) * 100.0
        return {
            "cache_size": len(self.pair_cache),
            "capacity": self.cache_capacity,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_ratio_pct": round(hit_ratio, 2),
        }

    def generate_outfit_fast(
        self,
        seed_item_id: Union[str, int],
        top_k: int = 3,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes fast outfit recommendation with latency tracking in milliseconds.
        """
        t0 = time.perf_counter()
        outfits = self.generate_outfit_from_seed(
            seed_item_id=seed_item_id,
            include_accessory=True,
            top_k_outfits=top_k,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return outfits, round(latency_ms, 2)
