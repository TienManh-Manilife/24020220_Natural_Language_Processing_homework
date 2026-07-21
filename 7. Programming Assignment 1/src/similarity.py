from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from .utils import ensure_directory, normalize_word, save_json

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray):
    norm1 = float(np.linalg.norm(vector1))
    norm2 = float(np.linalg.norm(vector2))
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Không xác định cosine với vector không")
    return float(np.dot(vector1, vector2) / (norm1 * norm2))


# Load ViSim, nếu không thấy thì báo lỗi
def load_visim(file_path):
    try:
        return pd.read_csv(file_path, sep="\t")
    except Exception as e:
        raise FileNotFoundError(f"Không thấy file ViSim: {file_path}")


def evaluate_word_pairs(dataframe: pd.DataFrame, embeddings: dict[str, np.ndarray],):
    # Lưu kết quả của các cặp từ có đầy đủ vector embedding.
    results: list[dict[str, object]] = []

    # Lưu các cặp từ có ít nhất một từ không tồn tại trong embeddings
    oov_rows: list[dict[str, object]] = [] # OOV = Out Of Vocabulary

    # Duyệt từng dòng trong DataFrame.
    # index=False nghĩa là không lấy cột index của DataFrame.
    for row in dataframe.itertuples(index=False):

        # Lấy Word1 và Word2 rồi chuẩn hóa từ
        word1 = normalize_word(str(row.Word1))
        word2 = normalize_word(str(row.Word2))

        # Kiểm tra từ nào không tồn tại trong từ điển embeddings. Nếu cả hai từ đều có vector thì missing sẽ là danh sách rỗng
        missing = [word for word in (word1, word2) if word not in embeddings]

        # Nếu có từ không tồn tại trong embeddings, lưu OOV. Nếu thiếu nhiều thì nối chúng bằng dấu phẩy. Bỏ qua tính cosine
        if missing:
            oov_rows.append({"Word1": word1, "Word2": word2, "Missing": ",".join(missing),})
            continue

        # Nếu cả hai từ đều có vector thì lưu kết quả đánh giá.
        results.append({"Word1": word1, "Word2": word2, "POS": str(row.POS), "Sim1": float(row.Sim1), 
                        "Sim2": float(row.Sim2), "Cosine": cosine_similarity(embeddings[word1], embeddings[word2],),})
    return pd.DataFrame(results), pd.DataFrame(oov_rows)


def correlation_report(results: pd.DataFrame):
    if results.empty:
        raise ValueError("Không có cặp Visim nào để đánh giá")

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


def run_similarity_experiment(embeddings: dict[str, np.ndarray], visim_path: str | Path, results_dir: str | Path,):
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