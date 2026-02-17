"""
Vector Embedding Store using FAISS.

Stores FinBERT CLS embeddings in a FAISS index for fast
similarity search. Enables finding historically similar news
headlines and their associated market outcomes.
"""

import os
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import faiss
import yaml

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """FAISS-based vector store for news headline embeddings."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        emb_config = self.config.get("embeddings", {})
        self.embedding_dim = emb_config.get("embedding_dim", 768)
        self.index_path = emb_config.get(
            "faiss_index_path", "data/embeddings/news_index.faiss"
        )
        self.metadata_path = emb_config.get(
            "metadata_path", "data/embeddings/metadata.pkl"
        )
        self.top_k = emb_config.get("top_k_similar", 5)

        self.index = None
        self.metadata = []

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ):
        """
        Build a FAISS index from embeddings with associated metadata.

        Uses IndexFlatIP (inner product / cosine similarity on normalized vectors)
        for exact nearest-neighbor search.

        Args:
            embeddings: numpy array of shape (n, embedding_dim).
            metadata: List of dicts with keys like
                [headline, date, ticker, sentiment_label, sentiment_score].
                Must be same length as embeddings.
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                f"must have the same length"
            )

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings.astype(np.float32))

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

        logger.info(
            f"Built FAISS index: {self.index.ntotal} vectors, "
            f"dim={self.embedding_dim}"
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ):
        """
        Add new embeddings to an existing index.

        Args:
            embeddings: numpy array of shape (n, embedding_dim).
            metadata: Corresponding metadata dicts.
        """
        if self.index is None:
            self.build_index(embeddings, metadata)
            return

        faiss.normalize_L2(embeddings.astype(np.float32))
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

        logger.info(
            f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}"
        )

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: Optional[int] = None,
    ) -> list[list[dict]]:
        """
        Find the most similar headlines for each query embedding.

        Args:
            query_embeddings: numpy array of shape (n_queries, embedding_dim).
            top_k: Number of results per query. Defaults to config value.

        Returns:
            List of lists of result dicts, each containing:
                - All metadata fields (headline, date, ticker, etc.)
                - similarity_score: cosine similarity (0-1)
                - rank: position in results (1-indexed)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Empty index. Build or load an index first.")
            return [[] for _ in range(len(query_embeddings))]

        k = min(top_k or self.top_k, self.index.ntotal)

        query = query_embeddings.astype(np.float32).copy()
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, k)

        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for rank, (score, idx) in enumerate(
                zip(query_scores, query_indices), 1
            ):
                if idx == -1:
                    continue
                result = {**self.metadata[idx]}
                result["similarity_score"] = float(score)
                result["rank"] = rank
                results.append(result)
            all_results.append(results)

        return all_results

    def search_text(
        self,
        texts: list[str],
        finbert_model,
        top_k: Optional[int] = None,
    ) -> list[list[dict]]:
        """
        Search by text — generates embeddings on the fly.

        Args:
            texts: List of headline strings to search for.
            finbert_model: FinBERTSentiment instance for embedding extraction.
            top_k: Number of results per query.

        Returns:
            List of lists of similar headline results.
        """
        embeddings = finbert_model.extract_embeddings(texts)
        return self.search(embeddings, top_k)

    def get_similarity_features(
        self,
        query_embeddings: np.ndarray,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get aggregated similarity features for LSTM input.

        For each query, returns the mean similarity score and mean
        sentiment score of the top-K most similar historical headlines.

        Args:
            query_embeddings: numpy array of shape (n, embedding_dim).
            top_k: Number of neighbors to consider.

        Returns:
            numpy array of shape (n, 2) with columns:
                [mean_similarity, mean_historical_sentiment]
        """
        results = self.search(query_embeddings, top_k)
        features = []

        for query_results in results:
            if not query_results:
                features.append([0.0, 0.0])
                continue

            sim_scores = [r["similarity_score"] for r in query_results]
            sent_scores = [
                r.get("sentiment_score", 0.0) for r in query_results
            ]

            features.append([
                float(np.mean(sim_scores)),
                float(np.mean(sent_scores)),
            ])

        return np.array(features)

    def save(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Save the FAISS index and metadata to disk."""
        idx_path = index_path or self.index_path
        meta_path = metadata_path or self.metadata_path

        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, idx_path)
            logger.info(f"Saved FAISS index to {idx_path}")

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
            logger.info(f"Saved metadata ({len(self.metadata)} records) to {meta_path}")

    def load(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Load a previously saved FAISS index and metadata."""
        idx_path = index_path or self.index_path
        meta_path = metadata_path or self.metadata_path

        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"FAISS index not found at {idx_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        self.index = faiss.read_index(idx_path)
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors from {idx_path}")

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
            logger.info(f"Loaded metadata: {len(self.metadata)} records from {meta_path}")

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self.index else 0
