"""
Visual Embedding Management and Vector Search Engine.
Module 8: Image Embedding Generation.

Handles serialization, indexing, sub-millisecond similarity retrieval,
and metadata filtering over 1,280-dimensional visual feature vectors.
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

# Ensure UTF-8 output encoding on Windows if supported
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.config import (
    EMBEDDINGS_NPY,
    EMBEDDINGS_INDEX_CSV,
    EMBEDDING_DIM,
)


class EmbeddingManager:
    """
    Manages in-memory visual embedding matrices and performs sub-millisecond
    vector similarity searches with categorical and demographic filters.
    """

    def __init__(
        self,
        embeddings_path: Optional[Union[str, Path]] = None,
        index_path: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ):
        self.embeddings_path = Path(embeddings_path) if embeddings_path else EMBEDDINGS_NPY
        self.index_path = Path(index_path) if index_path else EMBEDDINGS_INDEX_CSV

        self.embeddings: np.ndarray = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self.index_df: pd.DataFrame = pd.DataFrame()
        self.id_to_idx: Dict[str, int] = {}

        if auto_load and self.embeddings_path.exists() and self.index_path.exists():
            self.load()

    def load(self, embeddings_path: Optional[Union[str, Path]] = None, index_path: Optional[Union[str, Path]] = None):
        """Loads numpy embedding matrix and index CSV table into memory."""
        e_path = Path(embeddings_path) if embeddings_path else self.embeddings_path
        i_path = Path(index_path) if index_path else self.index_path

        if not e_path.exists() or not i_path.exists():
            raise FileNotFoundError(f"Embedding files not found. Expected: {e_path} and {i_path}")

        self.embeddings = np.load(str(e_path)).astype(np.float32)
        self.index_df = pd.read_csv(str(i_path))
        self.index_df["id"] = self.index_df["id"].astype(str)

        # Ensure image paths resolve portably across Windows, Linux, and Docker
        def _resolve_portable_path(raw_path: Any, item_id: str) -> str:
            p = Path(str(raw_path))
            if p.exists():
                return str(p)
            cand1 = PROJECT_ROOT / "data" / "raw" / "images" / f"{item_id}.jpg"
            if cand1.exists():
                return str(cand1)
            cand2 = PROJECT_ROOT / "data" / "images" / f"{item_id}.jpg"
            if cand2.exists():
                return str(cand2)
            return str(p)

        self.index_df["image_path"] = [
            _resolve_portable_path(r.get("image_path", ""), str(r["id"]))
            for _, r in self.index_df.iterrows()
        ]

        if len(self.embeddings) != len(self.index_df):
            raise ValueError(
                f"Embedding count ({len(self.embeddings)}) does not match index rows ({len(self.index_df)})."
            )

        # Build rapid O(1) ID-to-row lookup
        self.id_to_idx = {str(item_id): idx for idx, item_id in enumerate(self.index_df["id"])}
        print(f"[EmbeddingManager] Loaded {len(self.embeddings):,} visual embeddings ({self.embeddings.shape[1]}-dim).")

    def is_loaded(self) -> bool:
        """Returns True if the embedding index is populated and ready for search."""
        return len(self.embeddings) > 0 and len(self.index_df) > 0

    def __len__(self) -> int:
        return len(self.embeddings)

    def get_embedding(self, item_id: Union[str, int]) -> Optional[np.ndarray]:
        """Fetches the 1280-dimensional feature vector for an item ID."""
        str_id = str(item_id)
        if str_id not in self.id_to_idx:
            return None
        return self.embeddings[self.id_to_idx[str_id]]

    def search_by_vector(
        self,
        query_vec: Optional[np.ndarray] = None,
        top_k: int = 10,
        category: Optional[str] = None,
        outfit_part: Optional[str] = None,
        gender: Optional[str] = None,
        usage: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
        query_vector: Optional[np.ndarray] = None,
        exclude_id: Optional[Union[str, int]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Executes a vectorized cosine similarity query across the indexed embeddings.
        Supports filtering by clothing category, outfit part, gender, and occasion.

        Returns:
            List of matching item dictionaries ordered by similarity score descending.
        """
        if not self.is_loaded():
            raise RuntimeError("EmbeddingManager has no loaded embeddings. Run generate_embeddings.py first.")

        target_vec = query_vec if query_vec is not None else query_vector
        if target_vec is None:
            raise ValueError("Must provide either query_vec or query_vector.")

        if exclude_id is not None:
            if exclude_ids is None:
                exclude_ids = [str(exclude_id)]
            else:
                exclude_ids.append(str(exclude_id))

        # Ensure query is 1D and L2-normalized
        q = np.asarray(target_vec).flatten().astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        # Vectorized dot product (Cosine Similarity) across entire matrix in <5ms
        scores = np.dot(self.embeddings, q)

        # Apply metadata filtering masks
        mask = np.ones(len(self.index_df), dtype=bool)

        if category and "canonical_category" in self.index_df.columns:
            mask &= (self.index_df["canonical_category"] == category).values

        if outfit_part and "outfit_part" in self.index_df.columns:
            mask &= (self.index_df["outfit_part"] == outfit_part).values

        if gender and "gender" in self.index_df.columns and gender != "All":
            mask &= self.index_df["gender"].isin([gender, "Unisex"]).values

        if usage and "usage" in self.index_df.columns and usage != "All":
            mask &= (self.index_df["usage"].str.lower() == usage.lower()).values

        if exclude_ids:
            exclude_set = set(str(i) for i in exclude_ids)
            mask &= ~self.index_df["id"].isin(exclude_set).values

        candidate_indices = np.where(mask)[0]
        if len(candidate_indices) == 0:
            return []

        candidate_scores = scores[candidate_indices]

        # Rank top K
        actual_k = min(top_k, len(candidate_scores))
        top_sub_indices = np.argsort(-candidate_scores)[:actual_k]
        top_row_indices = candidate_indices[top_sub_indices]

        results = []
        for row_idx in top_row_indices:
            row_dict = self.index_df.iloc[row_idx].to_dict()
            row_dict["similarity_score"] = round(float(scores[row_idx]), 4)
            results.append(row_dict)

        return results

    def search_by_item_id(
        self,
        item_id: Union[str, int],
        top_k: int = 10,
        exclude_self: bool = True,
        category: Optional[str] = None,
        outfit_part: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds the visually most similar items to a specified catalog item ID.
        """
        vec = self.get_embedding(item_id)
        if vec is None:
            return []

        exclude = [str(item_id)] if exclude_self else None
        return self.search_by_vector(
            query_vec=vec,
            top_k=top_k,
            category=category,
            outfit_part=outfit_part,
            gender=gender,
            exclude_ids=exclude,
        )

    def save(
        self,
        embeddings_matrix: np.ndarray,
        index_df: pd.DataFrame,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Saves embedding array (.npy) and index dataframe (.csv)."""
        out_dir = Path(output_dir) if output_dir else self.embeddings_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        npy_path = out_dir / self.embeddings_path.name
        csv_path = out_dir / self.index_path.name

        np.save(str(npy_path), embeddings_matrix.astype(np.float32))
        index_df.to_csv(str(csv_path), index=False)

        print(f"[EmbeddingManager] Saved {len(embeddings_matrix):,} vectors to: {npy_path}")
        print(f"[EmbeddingManager] Saved index metadata to: {csv_path}")

        # Update instance state
        self.embeddings = embeddings_matrix.astype(np.float32)
        self.index_df = index_df
        self.id_to_idx = {str(item_id): idx for idx, item_id in enumerate(index_df["id"])}
