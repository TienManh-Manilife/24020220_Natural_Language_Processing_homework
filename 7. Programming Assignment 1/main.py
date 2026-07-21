from pathlib import Path
from src.classifier import run_classification_experiment
from src.embeddings import load_embeddings
from src.nearest_words import run_nearest_word_search
from src.similarity import run_similarity_experiment
import pandas as pd


if __name__ == "__main__":
    word2vec_path = "./data/Word2vec/word2vec.txt"
    embeddings = load_embeddings(word2vec_path)


    # Tham số bài 1
    visim_path = "./data/ViSim-400/Visim-400.txt"
    results_similarity_dir = "./results/similarity"


    # Tham số bài 2
    visim = pd.read_csv(visim_path, sep="\t")
    query_words = list(dict.fromkeys(visim["Word1"].astype(str).tolist() + visim["Word2"].astype(str).tolist()))
    k = 4
    results_nearest_dir = "./results/nearest"


    # Tham số bài 3
    vicon_paths: dict[str, str | Path] = {"noun": Path("data/ViCon-400/400_noun_pairs.txt"),
                                        "verb": Path("data/ViCon-400/400_verb_pairs.txt"),
                                        "adj": Path("data/ViCon-400/600_adj_pairs.txt"),}
    synonym_path = "./data/antonym-synonym-set/Synonym_vietnamese.txt"
    antonym_path = "./data/antonym-synonym-set/Antonym_vietnamese.txt"
    model_name = "logreg"
    validation_size = 0.2
    random_state = 42
    results_classification_dir = "./results/classification"

    try:
        # Bài 1: Cosine similarity-------------------------------------------------------------------------------------
        run_similarity_experiment(embeddings, visim_path, results_similarity_dir)

        # # Bài 2: K-nearest words---------------------------------------------------------------------------------------
        # run_nearest_word_search(embeddings, query_words, k, results_nearest_dir)

        # # Bài 3: Synonym-antonym classification------------------------------------------------------------------------
        # run_classification_experiment(embeddings, synonym_path, antonym_path, vicon_paths, 
        #                               results_classification_dir, model_name, validation_size, random_state)

        print("Hoàn thành chạy chương trình. Kết quả được lưu trong thư mục 'results'")
    except Exception as error:
        print(error.args)
