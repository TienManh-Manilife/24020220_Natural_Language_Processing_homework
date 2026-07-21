from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any


def normalize_word(word: str) -> str:
    """Normalize Vietnamese Unicode while preserving underscores."""
    return unicodedata.normalize("NFC", word.strip())


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def unordered_pair_key(word1: str, word2: str) -> tuple[str, str]:
    """Create an order-independent key for a word pair."""
    return tuple(sorted((normalize_word(word1), normalize_word(word2))))
