from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.classifier import run_classification_experiment
from src.embeddings import load_embeddings
from src.nearest_words import run_nearest_word_search
from src.similarity import run_similarity_experiment


def add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embeddings", required=True, help="Path to Word2Vec text file.")
    parser.add_argument("--dimension", type=int, default=150, help="Embedding dimension.")
    parser.add_argument("--max-words", type=int, default=None, help="Optional limit for debugging.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Programming Assignment 1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    similarity_parser = subparsers.add_parser("similarity", help="Evaluate ViSim-400.")
    add_embedding_arguments(similarity_parser)
    similarity_parser.add_argument("--visim", required=True)
    similarity_parser.add_argument("--results-dir", default="results/similarity")

    nearest_parser = subparsers.add_parser("nearest", help="Find nearest words.")
    add_embedding_arguments(nearest_parser)
    nearest_parser.add_argument("--word", action="append", required=True)
    nearest_parser.add_argument("--k", type=int, default=10)
    nearest_parser.add_argument("--results-dir", default="results/nearest")

    classify_parser = subparsers.add_parser("classify", help="Train/test SYN-ANT classifier.")
    add_embedding_arguments(classify_parser)
    classify_parser.add_argument("--synonyms", required=True)
    classify_parser.add_argument("--antonyms", required=True)
    classify_parser.add_argument("--vicon-noun", required=True)
    classify_parser.add_argument("--vicon-verb", required=True)
    classify_parser.add_argument("--vicon-adj", required=True)
    classify_parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    classify_parser.add_argument("--validation-size", type=float, default=0.2)
    classify_parser.add_argument("--random-state", type=int, default=42)
    classify_parser.add_argument("--results-dir", default="results/classification")

    all_parser = subparsers.add_parser("all", help="Run all assignment parts.")
    add_embedding_arguments(all_parser)
    all_parser.add_argument("--visim", required=True)
    all_parser.add_argument("--synonyms", required=True)
    all_parser.add_argument("--antonyms", required=True)
    all_parser.add_argument("--vicon-noun", required=True)
    all_parser.add_argument("--vicon-verb", required=True)
    all_parser.add_argument("--vicon-adj", required=True)
    all_parser.add_argument("--word", action="append", required=True)
    all_parser.add_argument("--k", type=int, default=10)
    all_parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    all_parser.add_argument("--validation-size", type=float, default=0.2)
    all_parser.add_argument("--random-state", type=int, default=42)
    all_parser.add_argument("--results-dir", default="results")

    return parser


def print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    embeddings = load_embeddings(
        args.embeddings,
        expected_dimension=args.dimension,
        max_words=args.max_words,
    )

    if args.command == "similarity":
        print_json(run_similarity_experiment(embeddings, args.visim, args.results_dir))
        return

    if args.command == "nearest":
        dataframe = run_nearest_word_search(embeddings, args.word, args.k, args.results_dir)
        print(dataframe.to_string(index=False))
        return

    vicon_paths = {
        "NOUN": args.vicon_noun,
        "VERB": args.vicon_verb,
        "ADJ": args.vicon_adj,
    }

    if args.command == "classify":
        print_json(run_classification_experiment(
            embeddings=embeddings,
            synonym_path=args.synonyms,
            antonym_path=args.antonyms,
            vicon_paths=vicon_paths,
            results_dir=args.results_dir,
            model_name=args.model,
            validation_size=args.validation_size,
            random_state=args.random_state,
        ))
        return

    base = Path(args.results_dir)
    similarity_report = run_similarity_experiment(
        embeddings,
        args.visim,
        base / "similarity",
    )
    nearest_results = run_nearest_word_search(
        embeddings,
        args.word,
        args.k,
        base / "nearest",
    )
    classification_summary = run_classification_experiment(
        embeddings=embeddings,
        synonym_path=args.synonyms,
        antonym_path=args.antonyms,
        vicon_paths=vicon_paths,
        results_dir=base / "classification",
        model_name=args.model,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    print("=== Similarity ===")
    print_json(similarity_report)
    print("\n=== Nearest words ===")
    print(nearest_results.to_string(index=False))
    print("\n=== Classification ===")
    print_json(classification_summary)


if __name__ == "__main__":
    main()
