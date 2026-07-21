import numpy as np
from pathlib import Path
from similarity import cosine_similarity
from utils import normalize_word, load_embeddings


def nearest_words(word: str, k: int, embeddings: dict[str, np.ndarray]):
    word = normalize_word(word)

    if word not in embeddings:
        raise KeyError(f"Không tìm thấy từ trong embeddings: {word}")
    if k <= 0:
        raise ValueError("k phải lớn hơn 0.")

    query_vector = embeddings[word]
    similarities: list[tuple[str, float]] = []

    for candidate, candidate_vector in embeddings.items():
        # Không so sánh từ với chính nó
        if candidate == word:
            continue

        score = cosine_similarity(query_vector, candidate_vector)

        similarities.append((candidate, score))

    # Sắp xếp theo cosine giảm dần
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:k]


# Chạy chương trình
if __name__ == "__main__":
    word2vec_path = "./data/Word2vec/word2vec.txt"
    embeddings = load_embeddings(word2vec_path)

    words = ["đánh", "nện", "phang", "tấn_công", "giương", "êm_đềm", "đỉnh_đầu", "thuốc_độc"]

    output_dir = Path("./results/nearest")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nearest.txt"
    with output_path.open("w", encoding="utf-8") as file:
        for word in words:
            # Đồng thời lưu vào results/nearest/ và in ra màn hình
            file.write(f"Xét chữ: {word}\n")
            print("Xét chữ: " + word)

            neighbors = nearest_words(word, 5, embeddings)

            for neighbor, score in neighbors:
                file.write(f"{neighbor}: {score:.6f}\n")
            file.write("\n")

            for neighbor, score in neighbors:
                print(neighbor, score)
            print()
    print(f"Đã lưu kết quả tại: {output_path}")