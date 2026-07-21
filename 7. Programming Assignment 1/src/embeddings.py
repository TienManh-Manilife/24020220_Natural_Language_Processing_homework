from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import normalize_word


def load_embeddings(
    file_path: str | Path,
    expected_dimension: int | None = 150,
    max_words: int | None = None,
) -> dict[str, np.ndarray]:
    """Load text-format Word2Vec embeddings into a dictionary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    embeddings: dict[str, np.ndarray] = {}
    dimension = expected_dimension

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, raw_line in enumerate(file, start=1):
            parts = raw_line.strip().split()
            if not parts:
                continue

            # Optional header: vocabulary_size dimension
            if (
                line_number == 1
                and len(parts) == 2
                and parts[0].isdigit()
                and parts[1].isdigit()
            ):
                header_dimension = int(parts[1])
                if dimension is None:
                    dimension = header_dimension
                elif dimension != header_dimension:
                    raise ValueError(
                        f"Header dimension {header_dimension} does not match "
                        f"expected dimension {dimension}."
                    )
                continue

            if len(parts) < 2:
                continue

            word = normalize_word(parts[0])
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                print(f"Skipping malformed line {line_number}.")
                continue

            if dimension is None:
                dimension = int(vector.size)

            if vector.size != dimension:
                print(
                    f"Skipping {word!r} at line {line_number}: "
                    f"expected {dimension}, got {vector.size}."
                )
                continue

            embeddings[word] = vector
            if max_words is not None and len(embeddings) >= max_words:
                break

    if not embeddings:
        raise ValueError(f"No valid embeddings were loaded from {path}.")

    return embeddings
