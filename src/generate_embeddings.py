"""
Visual Feature Embedding Generation Pipeline.
Module 8: Image Embedding Generation.

Batches fashion product catalog through the deep CNN Feature Extractor
(MobileNetV2, 1280-dim) and serializes the resulting L2-normalized embedding matrix (.npy)
and corresponding metadata index (.csv) for sub-millisecond retrieval.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Force unbuffered stdout for immediate live logging
sys.stdout.reconfigure(line_buffering=True)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_METADATA_CSV,
    EMBEDDINGS_DIR,
    EMBEDDINGS_NPY,
    EMBEDDINGS_INDEX_CSV,
    CLASSIFIER_MODEL_PATH,
    FEATURE_BACKBONE,
    BATCH_SIZE,
    RANDOM_STATE,
)
from src.feature_extractor import FashionFeatureExtractor
from src.embeddings import EmbeddingManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and serialize visual embeddings for the fashion catalog."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2500,
        help="Maximum number of garments to embed (default: 2500 for fast high-quality catalog).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for CNN inference (default: 64).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=FEATURE_BACKBONE,
        choices=["mobilenet_v2", "resnet50"],
        help="Backbone architecture to use.",
    )
    parser.add_argument(
        "--use-finetuned",
        action="store_true",
        help="Use fine-tuned weights from models/category_classifier.keras if available.",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        default=str(EMBEDDINGS_NPY),
        help="Path for saving numpy embedding array.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(EMBEDDINGS_INDEX_CSV),
        help="Path for saving embedding index CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducible catalog sampling when --limit is set.",
    )
    return parser.parse_args()


def generate_catalog_embeddings(
    limit: Optional[int] = 2500,
    batch_size: int = 64,
    backbone: str = FEATURE_BACKBONE,
    use_finetuned: bool = False,
    output_npy: Optional[str] = None,
    output_csv: Optional[str] = None,
    seed: int = RANDOM_STATE,
) -> EmbeddingManager:
    """
    Orchestrates the catalog embedding generation process.
    """
    npy_path = Path(output_npy) if output_npy else EMBEDDINGS_NPY
    csv_path = Path(output_csv) if output_csv else EMBEDDINGS_INDEX_CSV
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("      FASHION EMBEDDING GENERATION PIPELINE (MODULE 8)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load cleaned metadata
    if not CLEANED_METADATA_CSV.exists():
        raise FileNotFoundError(f"Cleaned metadata not found at {CLEANED_METADATA_CSV}")

    df = pd.read_csv(CLEANED_METADATA_CSV)
    print(f"[*] Loaded metadata catalog: {len(df):,} items.", flush=True)

    # 2. Sample if limit is specified (balanced across canonical categories)
    if limit is not None and limit < len(df):
        cats = df["canonical_category"].dropna().unique()
        per_cat = max(1, limit // len(cats))
        samples = []
        for cat in cats:
            sub = df[df["canonical_category"] == cat]
            n_take = min(len(sub), per_cat)
            samples.append(sub.sample(n=n_take, random_state=seed))
        sampled = pd.concat(samples, ignore_index=True)
        if len(sampled) < limit:
            remaining = df[~df["id"].isin(sampled["id"])]
            n_remain = min(len(remaining), limit - len(sampled))
            extra = remaining.sample(n=n_remain, random_state=seed)
            valid_df = pd.concat([sampled, extra], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        else:
            valid_df = sampled.sample(n=limit, random_state=seed).reset_index(drop=True)
        print(f"[*] Sampled {len(valid_df):,} balanced garments across {valid_df['canonical_category'].nunique()} categories.", flush=True)
    else:
        valid_df = df.copy()

    # 3. Quick verify image paths exist on disk for sampled subset
    exists_mask = [Path(p).exists() for p in valid_df["image_path"]]
    valid_df = valid_df[exists_mask].reset_index(drop=True)
    print(f"[*] Verified image files exist on disk: {len(valid_df):,} items.", flush=True)

    # 4. Initialize Feature Extractor
    weights = "imagenet"
    if use_finetuned and CLASSIFIER_MODEL_PATH.exists():
        weights = str(CLASSIFIER_MODEL_PATH)
        print(f"[*] Using fine-tuned classifier weights: {CLASSIFIER_MODEL_PATH}", flush=True)
    else:
        print(f"[*] Using ImageNet pretrained backbone weights.", flush=True)

    extractor = FashionFeatureExtractor(
        backbone=backbone,
        weights=weights,
        use_letterbox=True,
    )
    print(f"[*] Feature Extractor initialized: backbone={backbone}, dim={extractor.embedding_dim}", flush=True)

    # 5. Batch embedding generation with progress tracking
    image_paths = valid_df["image_path"].tolist()
    total_images = len(image_paths)
    all_vectors = []

    print(f"[*] Generating embeddings for {total_images:,} garments (batch size: {batch_size})...", flush=True)
    start_time = time.time()

    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_vecs = extractor.extract_batch(batch_paths, batch_size=len(batch_paths), l2_normalize=True)
        all_vectors.append(batch_vecs)

        done = min(i + batch_size, total_images)
        elapsed = time.time() - start_time
        speed = done / max(elapsed, 0.001)
        eta_sec = (total_images - done) / max(speed, 0.001)

        if done % (batch_size * 2) == 0 or done == total_images:
            pct = (done / total_images) * 100.0
            print(f"    [{done:,}/{total_images:,}] ({pct:5.1f}%) - Speed: {speed:5.1f} img/s - ETA: {eta_sec:4.0f}s", flush=True)

    embeddings_matrix = np.vstack(all_vectors).astype(np.float32)
    total_time = time.time() - start_time
    avg_speed = total_images / max(total_time, 0.001)

    print(f"\n[+] Embedding generation completed in {total_time:.2f}s ({avg_speed:.1f} images/sec).", flush=True)
    print(f"    Matrix shape: {embeddings_matrix.shape}, dtype: {embeddings_matrix.dtype}", flush=True)
    mem_mb = embeddings_matrix.nbytes / (1024 * 1024)
    print(f"    In-memory size: {mem_mb:.2f} MB", flush=True)

    # 6. Build index DataFrame with key metadata attributes
    cols_to_keep = [
        "id",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "year",
        "usage",
        "productDisplayName",
        "canonical_category",
        "outfit_part",
        "image_path",
    ]
    cols_to_keep = [c for c in cols_to_keep if c in valid_df.columns]
    index_df = valid_df[cols_to_keep].copy()

    # 7. Save using EmbeddingManager
    manager = EmbeddingManager(embeddings_path=npy_path, index_path=csv_path, auto_load=False)
    manager.save(embeddings_matrix, index_df, output_dir=npy_path.parent)

    # 8. Sanity check search
    sample_id = index_df.iloc[0]["id"]
    sample_row = index_df.iloc[0]
    sample_cat = sample_row.get("canonical_category", "Unknown")
    print(f"\n[*] Running sanity check similarity search for Item #{sample_id} ({sample_cat})...", flush=True)

    t_search_start = time.perf_counter()
    matches = manager.search_by_item_id(sample_id, top_k=5)
    t_search_ms = (time.perf_counter() - t_search_start) * 1000.0

    print(f"[+] Retrieved top 5 nearest neighbors in {t_search_ms:.3f} ms:", flush=True)
    for rank, m in enumerate(matches, start=1):
        name = str(m.get("productDisplayName", "N/A"))[:35]
        cat = m.get("canonical_category", "N/A")
        sim = m.get("similarity_score", 0.0)
        print(f"    {rank}. ID: {m['id']} | Sim: {sim:.4f} | Cat: {cat} | {name}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("[SUCCESS] Module 8 Visual Embeddings successfully created and indexed!", flush=True)
    print("=" * 70, flush=True)
    return manager


if __name__ == "__main__":
    args = parse_args()
    generate_catalog_embeddings(
        limit=args.limit,
        batch_size=args.batch_size,
        backbone=args.backbone,
        use_finetuned=args.use_finetuned,
        output_npy=args.output_npy,
        output_csv=args.output_csv,
        seed=args.seed,
    )
