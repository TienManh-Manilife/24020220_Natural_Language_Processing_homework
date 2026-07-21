import numpy as np

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

        score = cosine_similarity(
            query_vector,
            candidate_vector,
        )

        similarities.append((candidate, score))

    # Sắp xếp theo cosine giảm dần
    similarities.sort(key=lambda item: item[1], reverse=True)

    return similarities[:k]


# Chạy chương trình
if __name__ == "__main__":
    word2vec_path = "./data/Word2vec/word2vec.txt"
    embeddings = load_embeddings(word2vec_path)

    neighbors = nearest_words("nện", 5, embeddings)

    for neighbor, score in neighbors:
        print(neighbor, score)