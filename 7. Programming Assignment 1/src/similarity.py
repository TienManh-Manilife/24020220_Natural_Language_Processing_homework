from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .utils import ensure_directory, normalize_word, save_json
from .embeddings import load_embeddings


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate raw cosine similarity in the range [-1, 1]."""
    norm1 = float(np.linalg.norm(vector1))
    norm2 = float(np.linalg.norm(vector2))
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(vector1, vector2) / (norm1 * norm2))


def _standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(column).strip().lower(): column for column in dataframe.columns}
    rename_map: dict[object, str] = {}

    for target, candidates in {
        "Word1": ["word1", "w1"],
        "Word2": ["word2", "w2"],
        "POS": ["pos"],
        "Sim1": ["sim1"],
        "Sim2": ["sim2"],
        "STD": ["std"],
    }.items():
        for candidate in candidates:
            if candidate in lookup:
                rename_map[lookup[candidate]] = target
                break

    dataframe = dataframe.rename(columns=rename_map)
    required = {"Word1", "Word2", "Sim1", "Sim2"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"ViSim file is missing columns: {sorted(missing)}")
    if "POS" not in dataframe.columns:
        dataframe["POS"] = "UNKNOWN"
    return dataframe


def load_visim(file_path: str | Path) -> pd.DataFrame:
    """Read a tab- or whitespace-separated ViSim-400 file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ViSim file not found: {path}")

    last_error: Exception | None = None
    for options in ({"sep": "\t"}, {"sep": r"\s+", "engine": "python"}):
        try:
            return _standardize_columns(pd.read_csv(path, **options))
        except Exception as error:
            last_error = error
    raise ValueError(f"Could not parse ViSim file: {last_error}")


def evaluate_word_pairs(
    dataframe: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, object]] = []
    oov_rows: list[dict[str, object]] = []

    for row in dataframe.itertuples(index=False):
        word1 = normalize_word(str(row.Word1))
        word2 = normalize_word(str(row.Word2))
        missing = [word for word in (word1, word2) if word not in embeddings]

        if missing:
            oov_rows.append({
                "Word1": word1,
                "Word2": word2,
                "Missing": ",".join(missing),
            })
            continue

        results.append({
            "Word1": word1,
            "Word2": word2,
            "POS": str(row.POS),
            "Sim1": float(row.Sim1),
            "Sim2": float(row.Sim2),
            "Cosine": cosine_similarity(embeddings[word1], embeddings[word2]),
        })

    return pd.DataFrame(results), pd.DataFrame(oov_rows)


def correlation_report(results: pd.DataFrame) -> dict[str, float | int]:
    if results.empty:
        raise ValueError("No valid ViSim pairs are available for evaluation.")

    report: dict[str, float | int] = {"valid_pairs": int(len(results))}
    for column in ("Sim1", "Sim2"):
        pearson = pearsonr(results["Cosine"], results[column])
        spearman = spearmanr(results["Cosine"], results[column])
        key = column.lower()
        report[f"pearson_{key}"] = float(pearson.statistic)
        report[f"pearson_{key}_pvalue"] = float(pearson.pvalue)
        report[f"spearman_{key}"] = float(spearman.statistic)
        report[f"spearman_{key}_pvalue"] = float(spearman.pvalue)
    return report


def run_similarity_experiment(
    embeddings: dict[str, np.ndarray],
    visim_path: str | Path,
    results_dir: str | Path,
) -> dict[str, float | int]:
    output_dir = ensure_directory(results_dir)
    visim = load_visim(visim_path)
    predictions, oov = evaluate_word_pairs(visim, embeddings)
    report = correlation_report(predictions)
    report.update({
        "total_pairs": int(len(visim)),
        "oov_pairs": int(len(oov)),
        "coverage": float(len(predictions) / len(visim)),
    })

    predictions.to_csv(output_dir / "visim_predictions.csv", index=False, encoding="utf-8-sig")
    oov.to_csv(output_dir / "visim_oov.csv", index=False, encoding="utf-8-sig")
    save_json(report, output_dir / "visim_metrics.json")
    return report
