from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .similarity import cosine_similarity
from .utils import ensure_directory, normalize_word, save_json, unordered_pair_key

LABELS = ["ANT", "SYN"]


def load_labeled_pair_file(
    file_path: str | Path,
    label: str,
) -> list[tuple[str, str, str]]:
    """Read a synonym/antonym file whose label is determined by the file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Training pair file not found: {path}")

    label = label.strip().upper()
    if label not in LABELS:
        raise ValueError(f"Unsupported label: {label}")

    samples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, raw_line in enumerate(file, start=1):
            parts = raw_line.strip().split()
            if len(parts) < 2:
                continue
            word1 = normalize_word(parts[0])
            word2 = normalize_word(parts[1])
            if (
                line_number == 1
                and word1.lower() in {"word1", "w1"}
                and word2.lower() in {"word2", "w2"}
            ):
                continue
            samples.append((word1, word2, label))

    if not samples:
        raise ValueError(f"No pairs were loaded from {path}.")
    return samples


def deduplicate_samples(
    samples: Iterable[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Remove duplicate/reversed pairs and discard conflicting labels."""
    labels_by_pair: dict[tuple[str, str], set[str]] = {}
    examples: dict[tuple[str, str], tuple[str, str, str]] = {}

    for word1, word2, label in samples:
        key = unordered_pair_key(word1, word2)
        labels_by_pair.setdefault(key, set()).add(label)
        examples[key] = (word1, word2, label)

    return [
        examples[key]
        for key, labels in labels_by_pair.items()
        if len(labels) == 1
    ]


def create_pair_features(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """Return |v1-v2|, v1*v2 and cosine as symmetric pair features."""
    return np.concatenate([
        np.abs(vector1 - vector2),
        vector1 * vector2,
        np.asarray([cosine_similarity(vector1, vector2)], dtype=np.float32),
    ]).astype(np.float32)


def vectorize_samples(
    samples: Iterable[tuple[str, str, str]],
    embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    oov_rows: list[dict[str, str]] = []

    for word1, word2, label in samples:
        missing = [word for word in (word1, word2) if word not in embeddings]
        if missing:
            oov_rows.append({
                "Word1": word1,
                "Word2": word2,
                "Relation": label,
                "Missing": ",".join(missing),
            })
            continue
        features.append(create_pair_features(embeddings[word1], embeddings[word2]))
        labels.append(label)

    if not features:
        raise ValueError("No training samples could be vectorized.")
    return np.vstack(features), np.asarray(labels), pd.DataFrame(oov_rows)


def _standardize_vicon_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(column).strip().lower(): column for column in dataframe.columns}
    rename_map: dict[object, str] = {}
    for target, candidates in {
        "Word1": ["word1", "w1"],
        "Word2": ["word2", "w2"],
        "Relation": ["relation", "label", "class"],
    }.items():
        for candidate in candidates:
            if candidate in lookup:
                rename_map[lookup[candidate]] = target
                break

    dataframe = dataframe.rename(columns=rename_map)
    required = {"Word1", "Word2", "Relation"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"ViCon file is missing columns: {sorted(missing)}")
    return dataframe


def load_vicon_file(file_path: str | Path, pos_group: str) -> pd.DataFrame:
    """Load a tab- or whitespace-separated ViCon file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ViCon file not found: {path}")

    last_error: Exception | None = None
    for options in ({"sep": "\t"}, {"sep": r"\s+", "engine": "python"}):
        try:
            dataframe = _standardize_vicon_columns(pd.read_csv(path, **options))
            dataframe["Word1"] = dataframe["Word1"].map(lambda x: normalize_word(str(x)))
            dataframe["Word2"] = dataframe["Word2"].map(lambda x: normalize_word(str(x)))
            dataframe["Relation"] = dataframe["Relation"].map(lambda x: str(x).strip().upper())
            dataframe["POS_GROUP"] = pos_group.upper()
            return dataframe
        except Exception as error:
            last_error = error
    raise ValueError(f"Could not parse ViCon file: {last_error}")


def vectorize_vicon(
    dataframe: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    valid_rows: list[dict[str, object]] = []
    oov_rows: list[dict[str, object]] = []

    for row in dataframe.itertuples(index=False):
        word1 = normalize_word(str(row.Word1))
        word2 = normalize_word(str(row.Word2))
        relation = str(row.Relation).strip().upper()
        pos_group = str(row.POS_GROUP).strip().upper()

        if relation not in LABELS:
            raise ValueError(f"Unsupported relation {relation!r} for ({word1}, {word2}).")

        missing = [word for word in (word1, word2) if word not in embeddings]
        if missing:
            oov_rows.append({
                "Word1": word1,
                "Word2": word2,
                "Relation": relation,
                "POS_GROUP": pos_group,
                "Missing": ",".join(missing),
            })
            continue

        features.append(create_pair_features(embeddings[word1], embeddings[word2]))
        labels.append(relation)
        valid_rows.append({
            "Word1": word1,
            "Word2": word2,
            "Relation": relation,
            "POS_GROUP": pos_group,
        })

    if not features:
        raise ValueError("No ViCon samples could be vectorized.")
    return np.vstack(features), np.asarray(labels), pd.DataFrame(valid_rows), pd.DataFrame(oov_rows)


def build_model(model_name: str = "logreg", random_state: int = 42) -> Pipeline:
    """Build a Logistic Regression or MLP classification pipeline."""
    model_name = model_name.strip().lower()
    if model_name == "logreg":
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_name == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=random_state,
        )
    else:
        raise ValueError("model_name must be 'logreg' or 'mlp'.")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def calculate_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_precision": float(precision_score(true_labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true_labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_labels, predictions, average="weighted", zero_division=0)),
    }

    for label in LABELS:
        binary_true = (true_labels == label).astype(int)
        binary_pred = (predictions == label).astype(int)
        key = label.lower()
        metrics[f"{key}_precision"] = float(precision_score(binary_true, binary_pred, zero_division=0))
        metrics[f"{key}_recall"] = float(recall_score(binary_true, binary_pred, zero_division=0))
        metrics[f"{key}_f1"] = float(f1_score(binary_true, binary_pred, zero_division=0))
    return metrics


def save_confusion_matrix(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    matrix = confusion_matrix(true_labels, predictions, labels=LABELS)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix)
    axis.set_xticks(range(len(LABELS)), labels=LABELS)
    axis.set_yticks(range(len(LABELS)), labels=LABELS)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_classification_report(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str | Path,
) -> None:
    text = classification_report(
        true_labels,
        predictions,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )
    Path(output_path).write_text(text, encoding="utf-8")


def run_classification_experiment(
    embeddings: dict[str, np.ndarray],
    synonym_path: str | Path,
    antonym_path: str | Path,
    vicon_paths: dict[str, str | Path],
    results_dir: str | Path,
    model_name: str = "logreg",
    validation_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    output_dir = ensure_directory(results_dir)

    samples = load_labeled_pair_file(synonym_path, "SYN") + load_labeled_pair_file(antonym_path, "ANT")
    samples = deduplicate_samples(samples)
    X, y, train_oov = vectorize_samples(samples, embeddings)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=validation_size,
        stratify=y,
        random_state=random_state,
    )

    validation_model = build_model(model_name, random_state)
    validation_model.fit(X_train, y_train)
    validation_predictions = validation_model.predict(X_validation)
    validation_metrics = calculate_metrics(y_validation, validation_predictions)

    write_classification_report(
        y_validation,
        validation_predictions,
        output_dir / "validation_classification_report.txt",
    )
    save_json(validation_metrics, output_dir / "validation_metrics.json")
    save_confusion_matrix(
        y_validation,
        validation_predictions,
        output_dir / "validation_confusion_matrix.png",
        "Validation confusion matrix",
    )

    vicon = pd.concat(
        [load_vicon_file(path, pos) for pos, path in vicon_paths.items()],
        ignore_index=True,
    )
    X_test, y_test, test_rows, test_oov = vectorize_vicon(vicon, embeddings)

    final_model = build_model(model_name, random_state)
    final_model.fit(X, y)
    test_predictions = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    predictions_output = test_rows.copy()
    predictions_output["Prediction"] = test_predictions
    predictions_output["Correct"] = predictions_output["Relation"] == predictions_output["Prediction"]
    predictions_output.to_csv(output_dir / "vicon_predictions.csv", index=False, encoding="utf-8-sig")
    train_oov.to_csv(output_dir / "training_oov.csv", index=False, encoding="utf-8-sig")
    test_oov.to_csv(output_dir / "vicon_oov.csv", index=False, encoding="utf-8-sig")

    write_classification_report(y_test, test_predictions, output_dir / "test_classification_report.txt")
    save_json(test_metrics, output_dir / "test_metrics.json")
    save_confusion_matrix(
        y_test,
        test_predictions,
        output_dir / "test_confusion_matrix.png",
        "ViCon test confusion matrix",
    )

    per_pos_metrics: dict[str, dict[str, float]] = {}
    for pos in sorted(predictions_output["POS_GROUP"].unique()):
        subset = predictions_output[predictions_output["POS_GROUP"] == pos]
        per_pos_metrics[pos] = calculate_metrics(
            subset["Relation"].to_numpy(),
            subset["Prediction"].to_numpy(),
        )
    save_json(per_pos_metrics, output_dir / "test_metrics_by_pos.json")

    summary: dict[str, object] = {
        "model": model_name,
        "embedding_vocabulary_size": int(len(embeddings)),
        "training_samples_after_deduplication": int(len(samples)),
        "training_samples_used": int(len(y)),
        "training_oov_pairs": int(len(train_oov)),
        "training_label_distribution": dict(Counter(y)),
        "validation_samples": int(len(y_validation)),
        "validation_metrics": validation_metrics,
        "test_samples_used": int(len(y_test)),
        "test_oov_pairs": int(len(test_oov)),
        "test_label_distribution": dict(Counter(y_test)),
        "test_metrics": test_metrics,
        "test_metrics_by_pos": per_pos_metrics,
    }
    save_json(summary, output_dir / "classification_summary.json")
    return summary
