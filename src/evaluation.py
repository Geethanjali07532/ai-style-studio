"""
Recommendation System Evaluation Engine.
Module 12: Recommendation Model Evaluation.

Computes standard Information Retrieval and Recommender System metrics:
Precision@K, Recall@K, Hit Rate@K, Mean Reciprocal Rank (MRR),
Intra-List Diversity (ILD), and Catalog Coverage.
"""
from __future__ import annotations
import json
import os
import sys
import time
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
    EMBEDDINGS_DIR,
    RANDOM_STATE,
)
from src.embeddings import EmbeddingManager
from src.style_matcher import StyleMatcher
from src.recommender import OutfitRecommender


class RecommendationEvaluator:
    """
    Evaluates visual similarity matching and outfit recommendation models
    using ranking, accuracy, diversity, and coverage metrics.
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        style_matcher: Optional[StyleMatcher] = None,
        outfit_recommender: Optional[OutfitRecommender] = None,
    ):
        self.manager = embedding_manager if embedding_manager else EmbeddingManager()
        self.matcher = style_matcher if style_matcher else StyleMatcher(self.manager)
        self.recommender = (
            outfit_recommender
            if outfit_recommender
            else OutfitRecommender(style_matcher=self.matcher)
        )

    # --- Core Mathematical Metrics ---

    @staticmethod
    def compute_precision_at_k(
        recommended_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """
        Precision@K: Fraction of top-K recommendations that are relevant.
        """
        if k <= 0:
            return 0.0
        top_k = [str(i) for i in recommended_ids[:k]]
        if not top_k:
            return 0.0
        hits = sum(1 for item in top_k if item in relevant_ids)
        return float(hits / k)

    @staticmethod
    def compute_recall_at_k(
        recommended_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """
        Recall@K: Fraction of all relevant items captured in top-K.
        """
        if not relevant_ids or k <= 0:
            return 0.0
        top_k = [str(i) for i in recommended_ids[:k]]
        hits = sum(1 for item in top_k if item in relevant_ids)
        return float(hits / len(relevant_ids))

    @staticmethod
    def compute_hit_rate_at_k(
        recommended_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """
        Hit Rate@K: 1.0 if at least one relevant item appears in top-K, else 0.0.
        """
        top_k = [str(i) for i in recommended_ids[:k]]
        return 1.0 if any(item in relevant_ids for item in top_k) else 0.0

    @staticmethod
    def compute_mrr(
        recommended_ids: List[str],
        relevant_ids: Set[str],
    ) -> float:
        """
        Mean Reciprocal Rank (MRR): 1 / rank of the first relevant item.
        """
        for rank, item in enumerate(recommended_ids, start=1):
            if str(item) in relevant_ids:
                return float(1.0 / rank)
        return 0.0

    def compute_intra_list_diversity(self, item_ids: List[str]) -> float:
        """
        Intra-List Diversity (ILD): Average cosine distance (1 - CosSim)
        across all pairs of items within a recommendation list.
        Higher ILD prevents monotonous/repetitive recommendations.
        """
        valid_vecs = []
        for i_id in item_ids:
            v = self.manager.get_embedding(str(i_id))
            if v is not None:
                valid_vecs.append(v)

        if len(valid_vecs) < 2:
            return 0.0

        n = len(valid_vecs)
        total_dist = 0.0
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(valid_vecs[i], valid_vecs[j]))
                dist = 1.0 - max(0.0, sim)
                total_dist += dist
                pair_count += 1

        return float(total_dist / max(1, pair_count))

    # --- System-Level Evaluation Benchmarks ---

    def evaluate_similarity_retrieval(
        self,
        num_queries: int = 100,
        k_values: Tuple[int, ...] = (1, 3, 5, 10),
        seed: int = RANDOM_STATE,
    ) -> Dict[str, Any]:
        """
        Evaluates visual similarity retrieval:
        Items sharing the exact same canonical category, demographic gender,
        and usage/occasion are defined as ground-truth stylistic matches.
        """
        df = self.manager.index_df
        sample_n = min(num_queries, len(df))
        query_samples = df.sample(n=sample_n, random_state=seed)

        metrics = {
            f"Precision@{k}": [] for k in k_values
        }
        for k in k_values:
            metrics[f"Recall@{k}"] = []
            metrics[f"HitRate@{k}"] = []
        mrr_scores = []
        ild_scores = []

        t0 = time.perf_counter()

        for _, q_row in query_samples.iterrows():
            q_id = str(q_row["id"])
            q_cat = q_row.get("canonical_category")
            q_gender = q_row.get("gender")
            q_usage = q_row.get("usage")

            # Ground truth relevant items: same category + gender + usage
            rel_mask = (
                (df["id"] != q_id) &
                (df["canonical_category"] == q_cat) &
                (df["gender"] == q_gender) &
                (df["usage"] == q_usage)
            )
            relevant_ids = set(df[rel_mask]["id"].astype(str).tolist())
            if not relevant_ids:
                # Relax usage if no strict match
                rel_mask = (df["id"] != q_id) & (df["canonical_category"] == q_cat) & (df["gender"] == q_gender)
                relevant_ids = set(df[rel_mask]["id"].astype(str).tolist())

            if not relevant_ids:
                continue

            # Query top-max(k) candidates
            max_k = max(k_values)
            matches = self.manager.search_by_item_id(
                item_id=q_id,
                top_k=max_k,
                exclude_self=True,
            )
            retrieved_ids = [str(m["id"]) for m in matches]

            for k in k_values:
                metrics[f"Precision@{k}"].append(
                    self.compute_precision_at_k(retrieved_ids, relevant_ids, k)
                )
                metrics[f"Recall@{k}"].append(
                    self.compute_recall_at_k(retrieved_ids, relevant_ids, k)
                )
                metrics[f"HitRate@{k}"].append(
                    self.compute_hit_rate_at_k(retrieved_ids, relevant_ids, k)
                )

            mrr_scores.append(self.compute_mrr(retrieved_ids, relevant_ids))
            ild_scores.append(self.compute_intra_list_diversity(retrieved_ids[:5]))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_query_latency_ms = elapsed_ms / max(1, sample_n)

        results = {
            "num_queries_evaluated": len(mrr_scores),
            "avg_query_latency_ms": round(avg_query_latency_ms, 3),
            "MRR": round(float(np.mean(mrr_scores)), 4),
            "IntraListDiversity@5": round(float(np.mean(ild_scores)), 4),
        }
        for k in k_values:
            results[f"Precision@{k}"] = round(float(np.mean(metrics[f"Precision@{k}"])), 4)
            results[f"Recall@{k}"] = round(float(np.mean(metrics[f"Recall@{k}"])), 4)
            results[f"HitRate@{k}"] = round(float(np.mean(metrics[f"HitRate@{k}"])), 4)

        return results

    def evaluate_outfit_recommendations(
        self,
        num_queries: int = 50,
        seed: int = RANDOM_STATE,
    ) -> Dict[str, Any]:
        """
        Evaluates multi-item outfit generation:
        Measures mean outfit cohesion, gender consistency rate, and color diversity.
        """
        df = self.manager.index_df
        tops = df[df["outfit_part"] == "top"]
        sample_n = min(num_queries, len(tops))
        seed_tops = tops.sample(n=sample_n, random_state=seed)

        cohesion_scores = []
        gender_consistency_count = 0
        palette_sizes = []
        generation_latencies_ms = []

        for _, s_row in seed_tops.iterrows():
            seed_id = str(s_row["id"])
            t0 = time.perf_counter()

            outfits = self.recommender.generate_outfit_from_seed(
                seed_item_id=seed_id,
                include_accessory=True,
                top_k_outfits=1,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            generation_latencies_ms.append(lat_ms)

            if outfits:
                outfit = outfits[0]
                cohesion_scores.append(outfit["cohesion_score"])
                palette_sizes.append(len(set(outfit["color_palette"])))

                # Check gender consistency
                seed_gender = s_row.get("gender")
                items = outfit["items"]
                is_consistent = all(
                    it.get("gender") in (seed_gender, "Unisex", "All") for it in items
                )
                if is_consistent:
                    gender_consistency_count += 1

        total_evaluated = len(cohesion_scores)
        gender_consistency_rate = (
            (gender_consistency_count / total_evaluated) if total_evaluated > 0 else 0.0
        )

        return {
            "num_outfits_evaluated": total_evaluated,
            "mean_outfit_cohesion": round(float(np.mean(cohesion_scores)), 4),
            "median_outfit_cohesion": round(float(np.median(cohesion_scores)), 4),
            "gender_consistency_rate": round(float(gender_consistency_rate), 4),
            "avg_unique_colors_per_outfit": round(float(np.mean(palette_sizes)), 2),
            "avg_generation_latency_ms": round(float(np.mean(generation_latencies_ms)), 2),
        }

    def evaluate_catalog_coverage(
        self,
        num_queries: int = 100,
        top_k: int = 5,
        seed: int = RANDOM_STATE,
    ) -> Dict[str, Any]:
        """
        Calculates Catalog Coverage: % of unique catalog garments recommended.
        High coverage confirms the recommender does not suffer from popularity bias.
        """
        df = self.manager.index_df
        total_catalog_items = len(df)
        sample_n = min(num_queries, len(df))
        query_samples = df.sample(n=sample_n, random_state=seed)

        recommended_item_ids: Set[str] = set()

        for _, q_row in query_samples.iterrows():
            matches = self.manager.search_by_item_id(
                item_id=str(q_row["id"]),
                top_k=top_k,
                exclude_self=True,
            )
            for m in matches:
                recommended_item_ids.add(str(m["id"]))

        coverage_pct = (len(recommended_item_ids) / max(1, total_catalog_items)) * 100.0

        return {
            "catalog_total_items": total_catalog_items,
            "queries_executed": sample_n,
            "top_k_per_query": top_k,
            "unique_items_recommended": len(recommended_item_ids),
            "catalog_coverage_percentage": round(float(coverage_pct), 2),
        }

    def run_comprehensive_benchmark(
        self,
        num_sim_queries: int = 100,
        num_outfit_queries: int = 50,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full evaluation suite and serializes the report.
        """
        print("=" * 70)
        print("      RECOMMENDATION SYSTEM BENCHMARK & EVALUATION (MODULE 12)")
        print("=" * 70)

        # 1. Similarity Retrieval Evaluation
        print(f"[*] Benchmarking Similarity Retrieval ({num_sim_queries} queries)...")
        sim_metrics = self.evaluate_similarity_retrieval(num_queries=num_sim_queries)
        print(f"    - MRR                : {sim_metrics['MRR']}")
        print(f"    - Precision@5        : {sim_metrics['Precision@5']}")
        print(f"    - Recall@5           : {sim_metrics['Recall@5']}")
        print(f"    - HitRate@5          : {sim_metrics['HitRate@5'] * 100:.1f}%")
        print(f"    - Intra-List Div@5   : {sim_metrics['IntraListDiversity@5']}")
        print(f"    - Avg Latency        : {sim_metrics['avg_query_latency_ms']:.2f} ms")

        # 2. Outfit Assembly Evaluation
        print(f"\n[*] Benchmarking Outfit Cohesion ({num_outfit_queries} queries)...")
        outfit_metrics = self.evaluate_outfit_recommendations(num_queries=num_outfit_queries)
        print(f"    - Mean Outfit Cohesion: {outfit_metrics['mean_outfit_cohesion']}")
        print(f"    - Gender Consistency : {outfit_metrics['gender_consistency_rate'] * 100:.1f}%")
        print(f"    - Unique Colors/Look : {outfit_metrics['avg_unique_colors_per_outfit']}")
        print(f"    - Avg Outfit Latency : {outfit_metrics['avg_generation_latency_ms']:.2f} ms")

        # 3. Catalog Coverage
        print(f"\n[*] Benchmarking Catalog Coverage ({num_sim_queries} queries)...")
        coverage_metrics = self.evaluate_catalog_coverage(num_queries=num_sim_queries, top_k=5)
        print(f"    - Unique Items Recom : {coverage_metrics['unique_items_recommended']:,}")
        print(f"    - Catalog Coverage   : {coverage_metrics['catalog_coverage_percentage']:.2f}%")

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "catalog_size": len(self.manager),
            "similarity_retrieval": sim_metrics,
            "outfit_recommendation": outfit_metrics,
            "catalog_coverage": coverage_metrics,
        }

        # Save report JSON
        out_path = Path(output_file) if output_file else REPORTS_DIR / "evaluation_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n[+] Benchmark report saved to: {out_path}")
        print("=" * 70)
        return report
