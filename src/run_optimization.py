"""
CLI Runner for Recommendation System Optimization & Threshold Tuning.
Module 13: Recommendation Optimization.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR
from src.optimizer import ThresholdTuner, OptimizedOutfitRecommender


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run threshold tuning and latency optimization benchmark."
    )
    parser.add_argument(
        "--eval-pairs",
        type=int,
        default=200,
        help="Number of garment pairs for threshold grid search (default: 200).",
    )
    parser.add_argument(
        "--benchmark-queries",
        type=int,
        default=25,
        help="Number of queries for latency benchmarking (default: 25).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(REPORTS_DIR / "optimization_results.json"),
        help="Path to save optimization results JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("      RECOMMENDATION OPTIMIZATION & TUNING PIPELINE (MODULE 13)")
    print("=" * 70)

    # 1. Threshold Tuning
    print(f"[*] Running Threshold Grid Search ({args.eval_pairs} evaluation pairs)...")
    tuner = ThresholdTuner()
    tuning_res = tuner.tune_compatibility_threshold(num_eval_pairs=args.eval_pairs)

    print(f"\n[+] Optimal Compatibility Cutoff Threshold: tau* = {tuning_res.optimal_threshold:.3f}")
    print(f"    - Maximum F1-Score : {tuning_res.max_f1_score:.4f}")
    print(f"    - Precision at tau*: {tuning_res.precision_at_optimal:.4f}")
    print(f"    - Recall at tau*   : {tuning_res.recall_at_optimal:.4f}")

    # 2. Optimized Recommender Latency Profiling
    print(f"\n[*] Profiling Optimized Recommender ({args.benchmark_queries} queries)...")
    opt_engine = OptimizedOutfitRecommender(compatibility_threshold=tuning_res.optimal_threshold)

    df = opt_engine.manager.index_df
    seed_tops = df[df["outfit_part"] == "top"].sample(n=args.benchmark_queries, random_state=42)

    latencies = []
    for _, row in seed_tops.iterrows():
        _, lat_ms = opt_engine.generate_outfit_fast(row["id"], top_k=2)
        latencies.append(lat_ms)

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))

    cache_stats = opt_engine.get_cache_stats()

    print(f"\n[+] Latency Benchmarks across {args.benchmark_queries} complete outfit generations:")
    print(f"    - Mean Latency : {avg_lat:.2f} ms")
    print(f"    - p95 Latency  : {p95_lat:.2f} ms")
    print(f"    - Min / Max    : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"    - Throughput   : {1000.0 / max(0.1, avg_lat):.1f} outfits/sec")
    print(f"    - Cache Status : {cache_stats['cache_size']} entries cached (Hit Ratio: {cache_stats['hit_ratio_pct']}%)")

    # Save results JSON
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold_tuning": {
            "optimal_threshold": tuning_res.optimal_threshold,
            "max_f1_score": tuning_res.max_f1_score,
            "precision_at_optimal": tuning_res.precision_at_optimal,
            "recall_at_optimal": tuning_res.recall_at_optimal,
            "grid_curve": tuning_res.grid_curve,
        },
        "latency_optimization": {
            "mean_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "throughput_outfits_per_sec": round(1000.0 / max(0.1, avg_lat), 1),
            "cache_stats": cache_stats,
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Optimization results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    import numpy as np
    main()
