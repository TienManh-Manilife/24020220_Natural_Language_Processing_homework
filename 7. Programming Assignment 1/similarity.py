from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from utils import ensure_directory, normalize_word, load_embeddings


# Tính cosine similarity giữa hai vector
def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray):
    norm1 = float(np.linalg.norm(vector1))
    norm2 = float(np.linalg.norm(vector2))
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Không xác định cosine với vector không.")
    return float(np.dot(vector1, vector2) / (norm1 * norm2))


# Đọc dữ liệu ViSim từ file
def load_visim(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file ViSim: {path}")
    return pd.read_csv(path, sep="\t")

# Tính cosine cho các cặp từ và lưu các cặp OOV riêng
def evaluate_word_pairs(dataframe: pd.DataFrame, embeddings: dict[str, np.ndarray]):
    results: list[dict[str, object]] = []
    oov_rows: list[dict[str, object]] = []

    for row in dataframe.itertuples(index=False):
        word1 = normalize_word(str(row.Word1))
        word2 = normalize_word(str(row.Word2))

        missing = [word for word in (word1, word2) if word not in embeddings]

        if missing:
            oov_rows.append({"Word1": word1, "Word2": word2, "Missing": ",".join(missing)})
            continue

        results.append({"Word1": word1, "Word2": word2, "POS": str(row.POS), "Sim1": float(row.Sim1), "Sim2": float(row.Sim2), 
                        "STD": float(row.STD), "Cosine": cosine_similarity(embeddings[word1], embeddings[word2])})

    return pd.DataFrame(results), pd.DataFrame(oov_rows)


# Đọc ViSim, tính cosine và lưu kết quả
def run_similarity_experiment(embeddings: dict[str, np.ndarray], visim_path: str | Path, results_dir: str | Path):
    output_dir = ensure_directory(results_dir)
    visim = load_visim(visim_path)
    predictions, oov = evaluate_word_pairs(visim, embeddings)
    predictions.to_csv(output_dir / "visim_predictions.csv", index=False, encoding="utf-8-sig")
    oov.to_csv(output_dir / "visim_oov.csv", index=False, encoding="utf-8-sig")



# Chạy chương trình
if __name__ == "__main__":
    word2vec_path = "./data/Word2vec/word2vec.txt"
    embeddings = load_embeddings(word2vec_path)

    visim_path = "./data/ViSim-400/Visim-400.txt"
    results_similarity_dir = "./results/similarity"

    run_similarity_experiment(embeddings, visim_path, results_similarity_dir)