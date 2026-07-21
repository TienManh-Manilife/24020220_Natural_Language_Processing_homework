from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_directory, normalize_word


class EmbeddingIndex:
    """Vectorized nearest-neighbor index using cosine similarity."""

    def __init__(self, embeddings: dict[str, np.ndarray]) -> None:
        if not embeddings:
            raise ValueError("Embedding dictionary is empty.")

        self.words = list(embeddings.keys())
        self.word_to_index = {word: index for index, word in enumerate(self.words)}
        matrix = np.vstack([embeddings[word] for word in self.words]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.zero_rows = norms[:, 0] == 0.0
        norms[self.zero_rows] = 1.0
        self.matrix = matrix / norms

    def nearest_words(self, word: str, k: int = 10) -> list[tuple[str, float]]:
        query = normalize_word(word)
        if query not in self.word_to_index:
            raise KeyError(f"Word not found in embeddings: {query}")
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        query_index = self.word_to_index[query]
        if self.zero_rows[query_index]:
            raise ValueError(f"Query word has a zero vector: {query}")

        scores = self.matrix @ self.matrix[query_index]
        scores[query_index] = -np.inf
        scores[self.zero_rows] = -np.inf

        available = max(0, len(self.words) - 1 - int(self.zero_rows.sum()))
        k = min(k, available)
        if k == 0:
            return []

        indices = np.argpartition(scores, -k)[-k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [(self.words[index], float(scores[index])) for index in indices]


def run_nearest_word_search(
    embeddings: dict[str, np.ndarray],
    query_words: list[str],
    k: int,
    results_dir: str | Path,
) -> pd.DataFrame:
    output_dir = ensure_directory(results_dir)
    index = EmbeddingIndex(embeddings)
    rows: list[dict[str, object]] = []

    for query in query_words:
        try:
            neighbors = index.nearest_words(query, k)
            for rank, (neighbor, score) in enumerate(neighbors, start=1):
                rows.append({
                    "Query": normalize_word(query),
                    "Rank": rank,
                    "Neighbor": neighbor,
                    "Cosine": score,
                    "Error": "",
                })
        except (KeyError, ValueError) as error:
            rows.append({
                "Query": normalize_word(query),
                "Rank": None,
                "Neighbor": None,
                "Cosine": None,
                "Error": str(error),
            })

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_dir / "nearest_words.csv", index=False, encoding="utf-8-sig")
    return dataframe
