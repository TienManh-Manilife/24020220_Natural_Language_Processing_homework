from __future__ import annotations

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from similarity import cosine_similarity
from utils import load_embeddings, normalize_word


LABELS = ["ANT", "SYN"]


# Đọc file chứa các cặp đồng nghĩa hoặc trái nghĩa dùng để huấn luyện
def load_labeled_pair_file(file_path: str | Path, label: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file huấn luyện: {path}")

    label = label.strip().upper()

    if label not in LABELS:
        raise ValueError(f"Nhãn không hợp lệ: {label}")

    samples: list[tuple[str, str, str]] = []

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, raw_line in enumerate(file, start=1):
            parts = raw_line.strip().split()

            if len(parts) < 2:
                continue

            word1 = normalize_word(parts[0])
            word2 = normalize_word(parts[1])

            # Bỏ qua dòng tiêu đề nếu có
            if line_number == 1 and word1.lower() in {"word1", "w1"} and word2.lower() in {"word2", "w2"}:
                continue

            samples.append((word1, word2, label))

    if not samples:
        raise ValueError(f"Không đọc được cặp từ nào từ file: {path}")

    return samples


# Tạo khóa không phụ thuộc thứ tự của cặp từ
def create_pair_key(word1: str, word2: str):
    return tuple(sorted((word1, word2)))


# Loại các cặp trùng nhau, cặp đảo và cặp có nhãn mâu thuẫn
def remove_duplicate_samples(samples: Iterable[tuple[str, str, str]]):
    labels_by_pair: dict[tuple[str, str], set[str]] = {}
    sample_by_pair: dict[tuple[str, str], tuple[str, str, str]] = {}

    for word1, word2, label in samples:
        key = create_pair_key(word1, word2)

        if key not in labels_by_pair:
            labels_by_pair[key] = set()

        labels_by_pair[key].add(label)
        sample_by_pair[key] = (word1, word2, label)

    cleaned_samples: list[tuple[str, str, str]] = []

    for key, labels in labels_by_pair.items():
        # Chỉ giữ cặp có đúng một nhãn
        if len(labels) == 1:
            cleaned_samples.append(sample_by_pair[key])

    return cleaned_samples


# Tạo đặc trưng cho một cặp vector từ
def create_pair_features(vector1: np.ndarray, vector2: np.ndarray):
    difference = np.abs(vector1 - vector2)
    product = vector1 * vector2
    cosine = np.asarray([cosine_similarity(vector1, vector2)], dtype=np.float32)

    return np.concatenate([
        difference,
        product,
        cosine,
    ]).astype(np.float32)


# Chuyển dữ liệu huấn luyện thành ma trận đặc trưng
def vectorize_training_samples(samples: Iterable[tuple[str, str, str]], embeddings: dict[str, np.ndarray]):
    features: list[np.ndarray] = []
    labels: list[str] = []
    valid_rows: list[dict[str, str]] = []
    oov_rows: list[dict[str, str]] = []

    for word1, word2, label in samples:
        missing = []

        if word1 not in embeddings:
            missing.append(word1)

        if word2 not in embeddings:
            missing.append(word2)

        if missing:
            oov_rows.append({"Word1": word1, "Word2": word2, "Relation": label, "Missing": ",".join(missing)})
            continue

        pair_features = create_pair_features(embeddings[word1], embeddings[word2],
        )

        features.append(pair_features)
        labels.append(label)

        valid_rows.append({"Word1": word1, "Word2": word2, "Relation": label})

    if not features:
        raise ValueError("Không có cặp từ huấn luyện hợp lệ.")

    return (np.vstack(features), np.asarray(labels), pd.DataFrame(valid_rows), pd.DataFrame(oov_rows))


# Đọc một file ViCon-400
def load_vicon_file(file_path: str | Path, pos_group: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ViCon: {path}")

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, raw_line in enumerate(file, start=1):
            parts = raw_line.strip().split()

            if len(parts) < 3:
                continue

            word1 = normalize_word(parts[0])
            word2 = normalize_word(parts[1])
            relation = parts[2].strip().upper()

            # Bỏ qua dòng tiêu đề
            if line_number == 1 and word1.lower() in {"word1", "w1"} and word2.lower() in {"word2", "w2"}:
                continue

            if relation not in LABELS:
                continue

            rows.append({
                "Word1": word1,
                "Word2": word2,
                "Relation": relation,
                "POS_GROUP": pos_group.upper(),
            })

    if not rows:
        raise ValueError(f"Không đọc được dữ liệu ViCon từ file: {path}")

    return pd.DataFrame(rows)


# Đọc toàn bộ các file ViCon-400
def load_vicon_dataset(vicon_paths: dict[str, str | Path]):
    dataframes: list[pd.DataFrame] = []

    for pos_group, file_path in vicon_paths.items():
        dataframe = load_vicon_file(file_path, pos_group)
        dataframes.append(dataframe)

    if not dataframes:
        raise ValueError("Không có file ViCon nào được cung cấp.")

    return pd.concat(dataframes, ignore_index=True)


# Chuyển ViCon thành dữ liệu kiểm thử
def vectorize_vicon(dataframe: pd.DataFrame, embeddings: dict[str, np.ndarray]):
    features: list[np.ndarray] = []
    labels: list[str] = []
    valid_rows: list[dict[str, str]] = []
    oov_rows: list[dict[str, str]] = []

    for row in dataframe.itertuples(index=False):
        word1 = normalize_word(str(row.Word1))
        word2 = normalize_word(str(row.Word2))
        relation = str(row.Relation).strip().upper()
        pos_group = str(row.POS_GROUP).strip().upper()

        missing = []

        if word1 not in embeddings:
            missing.append(word1)

        if word2 not in embeddings:
            missing.append(word2)

        if missing:
            oov_rows.append({
                "Word1": word1,
                "Word2": word2,
                "Relation": relation,
                "POS_GROUP": pos_group,
                "Missing": ",".join(missing),
            })
            continue

        pair_features = create_pair_features(embeddings[word1], embeddings[word2])

        features.append(pair_features)
        labels.append(relation)

        valid_rows.append({
            "Word1": word1,
            "Word2": word2,
            "Relation": relation,
            "POS_GROUP": pos_group,
        })

    if not features:
        raise ValueError("Không có cặp từ ViCon hợp lệ để kiểm thử.")

    return (
        np.vstack(features),
        np.asarray(labels),
        pd.DataFrame(valid_rows),
        pd.DataFrame(oov_rows),
    )


# Tạo pipeline Logistic Regression
def build_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


# Tính các chỉ số đánh giá
def calculate_metrics(true_labels: np.ndarray, predictions: np.ndarray):
    accuracy = accuracy_score(true_labels, predictions)

    precision = precision_score(
        true_labels,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    return {
        "Accuracy": float(accuracy),
        "Macro Precision": float(precision),
        "Macro Recall": float(recall),
        "Macro F1": float(f1),
    }


# Vẽ và lưu confusion matrix
def save_confusion_matrix(true_labels: np.ndarray, predictions: np.ndarray, output_path: str | Path):
    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=LABELS,
    )

    figure, axis = plt.subplots(figsize=(6, 5))

    image = axis.imshow(matrix, cmap="Blues")

    axis.set_title("Synonym-Antonym Classification")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")

    axis.set_xticks(range(len(LABELS)))
    axis.set_yticks(range(len(LABELS)))

    axis.set_xticklabels(LABELS)
    axis.set_yticklabels(LABELS)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


# Lưu các chỉ số vào file
def save_metrics(metrics: dict[str, float], output_path: str | Path, training_samples: int, testing_samples: int):
    path = Path(output_path)

    with path.open("w", encoding="utf-8") as file:
        for metric_name, metric_value in metrics.items():
            file.write(f"{metric_name}: {metric_value:.4f}\n")

        file.write(f"Training samples: {training_samples}\n")
        file.write(f"Testing samples: {testing_samples}\n")


# Lưu classification report
def save_classification_report(true_labels: np.ndarray, predictions: np.ndarray, output_path: str | Path):
    report = classification_report(
        true_labels,
        predictions,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )

    Path(output_path).write_text(report, encoding="utf-8")


# Huấn luyện trên antonym-synonym set và kiểm thử trên ViCon-400
def run_classification_experiment(embeddings: dict[str, np.ndarray], synonym_path: str | Path, antonym_path: str | Path, vicon_paths: dict[str, str | Path], results_dir: str | Path):
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu huấn luyện
    synonym_samples = load_labeled_pair_file(
        synonym_path,
        "SYN",
    )

    antonym_samples = load_labeled_pair_file(
        antonym_path,
        "ANT",
    )

    training_samples = synonym_samples + antonym_samples

    # Loại cặp trùng và cặp có nhãn mâu thuẫn
    training_samples = remove_duplicate_samples(training_samples)

    # Chuyển dữ liệu huấn luyện thành đặc trưng
    X_train, y_train, training_rows, training_oov = vectorize_training_samples(
        training_samples,
        embeddings,
    )

    print("Số mẫu huấn luyện hợp lệ:", len(y_train))
    print("Số mẫu huấn luyện OOV:", len(training_oov))

    # Huấn luyện mô hình
    model = build_model()
    model.fit(X_train, y_train)

    # Đọc dữ liệu kiểm thử ViCon
    vicon_dataframe = load_vicon_dataset(vicon_paths)

    X_test, y_test, test_rows, test_oov = vectorize_vicon(
        vicon_dataframe,
        embeddings,
    )

    print("Số mẫu ViCon hợp lệ:", len(y_test))
    print("Số mẫu ViCon OOV:", len(test_oov))

    # Dự đoán trên ViCon
    predictions = model.predict(X_test)

    # Tính các chỉ số
    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print("Accuracy:", metrics["Accuracy"])
    print("Macro Precision:", metrics["Macro Precision"])
    print("Macro Recall:", metrics["Macro Recall"])
    print("Macro F1:", metrics["Macro F1"])

    # Lưu kết quả dự đoán
    predictions_output = test_rows.copy()
    predictions_output["Prediction"] = predictions
    predictions_output["Correct"] = predictions_output["Relation"] == predictions_output["Prediction"]

    predictions_output.to_csv(
        output_dir / "vicon_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Lưu các cặp OOV
    training_oov.to_csv(
        output_dir / "training_oov.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_oov.to_csv(
        output_dir / "vicon_oov.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Lưu chỉ số
    save_metrics(
        metrics,
        output_dir / "metrics.txt",
        len(y_train),
        len(y_test),
    )

    # Lưu classification report
    save_classification_report(
        y_test,
        predictions,
        output_dir / "classification_report.txt",
    )

    # Vẽ confusion matrix
    save_confusion_matrix(
        y_test,
        predictions,
        output_dir / "confusion_matrix.png",
    )

    print(f"Đã lưu kết quả tại: {output_dir}")


if __name__ == "__main__":
    word2vec_path = "./data/Word2vec/word2vec.txt"

    synonym_path = "./data/antonym-synonym-set/Synonym_vietnamese.txt"
    antonym_path = "./data/antonym-synonym-set/Antonym_vietnamese.txt"

    vicon_paths = {
        "noun": "./data/ViCon-400/400_noun_pairs.txt",
        "verb": "./data/ViCon-400/400_verb_pairs.txt",
        "adj": "./data/ViCon-400/600_adj_pairs.txt",
    }

    results_dir = "./results/classifier"

    embeddings = load_embeddings(word2vec_path)

    run_classification_experiment(
        embeddings,
        synonym_path,
        antonym_path,
        vicon_paths,
        results_dir,
    )