"""
CLI Runner for Recommendation System Evaluation.
Module 12: Recommendation Model Evaluation.
"""
from __future__ import annotations
import argparse
import sys
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
from src.evaluation import RecommendationEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation benchmark for the Outfit Recommendation System."
    )
    parser.add_argument(
        "--sim-queries",
        type=int,
        default=50,
        help="Number of test queries for similarity retrieval evaluation (default: 50).",
    )
    parser.add_argument(
        "--outfit-queries",
        type=int,
        default=25,
        help="Number of seed items for outfit assembly evaluation (default: 25).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(REPORTS_DIR / "evaluation_results.json"),
        help="Path to save evaluation JSON results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = RecommendationEvaluator()
    evaluator.run_comprehensive_benchmark(
        num_sim_queries=args.sim_queries,
        num_outfit_queries=args.outfit_queries,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
