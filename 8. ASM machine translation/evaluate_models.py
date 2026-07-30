#!/usr/bin/env python3
"""Evaluate and compare two machine-translation systems.

The script uses only the Python standard library, so it can run in the
TensorFlow 1.15 Docker container without installing extra packages.

Metrics:
- Corpus BLEU-4 (cased and uncased)
- chrF++ (character 1-6 grams + word 1-2 grams, beta=2)
- Token edit rate based on Levenshtein distance (lower is better)
- Normalized exact-match rate
- Output/reference length ratio
- Empty-output count
- Sentence-level chrF++ wins/ties/losses
- Paired bootstrap confidence interval for the mean sentence chrF++ difference
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def read_lines(path: Path) -> List[str]:
    """Read UTF-8 text while preserving internal empty lines."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.split("\n")
    # A normal final newline is not an additional sentence.
    if lines and lines[-1] == "":
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.strip().split())
    return text.lower() if lowercase else text


def tokenize(text: str, lowercase: bool = False) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(text, lowercase=lowercase))


def ngrams(items: Sequence[str], order: int) -> Counter:
    if len(items) < order:
        return Counter()
    return Counter(tuple(items[i : i + order]) for i in range(len(items) - order + 1))


def corpus_bleu(
    references: Sequence[str], hypotheses: Sequence[str], lowercase: bool = False, max_order: int = 4
) -> float:
    """Compute standard single-reference corpus BLEU with uniform 1-4 gram weights."""
    clipped = [0] * max_order
    totals = [0] * max_order
    ref_length = 0
    hyp_length = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize(ref, lowercase=lowercase)
        hyp_tokens = tokenize(hyp, lowercase=lowercase)
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)

        for order in range(1, max_order + 1):
            ref_counts = ngrams(ref_tokens, order)
            hyp_counts = ngrams(hyp_tokens, order)
            totals[order - 1] += sum(hyp_counts.values())
            clipped[order - 1] += sum(
                min(count, ref_counts.get(gram, 0)) for gram, count in hyp_counts.items()
            )

    if hyp_length == 0:
        return 0.0

    precisions = []
    for match_count, total_count in zip(clipped, totals):
        if total_count == 0 or match_count == 0:
            return 0.0
        precisions.append(match_count / total_count)

    brevity_penalty = 1.0 if hyp_length > ref_length else math.exp(1.0 - ref_length / hyp_length)
    score = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_order)
    return score * 100.0


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def _chrf_stats(
    references: Sequence[str],
    hypotheses: Sequence[str],
    lowercase: bool,
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    scores: List[float] = []

    # chrF character n-grams; whitespace is ignored, matching common chrF settings.
    for order in range(1, char_order + 1):
        matches = hyp_total = ref_total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_chars = list(normalize_text(ref, lowercase=lowercase).replace(" ", ""))
            hyp_chars = list(normalize_text(hyp, lowercase=lowercase).replace(" ", ""))
            ref_counts = ngrams(ref_chars, order)
            hyp_counts = ngrams(hyp_chars, order)
            matches += sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
            hyp_total += sum(hyp_counts.values())
            ref_total += sum(ref_counts.values())
        precision = matches / hyp_total if hyp_total else 0.0
        recall = matches / ref_total if ref_total else 0.0
        scores.append(f_beta(precision, recall, beta=beta))

    # chrF++ adds word n-gram F-scores.
    for order in range(1, word_order + 1):
        matches = hyp_total = ref_total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = tokenize(ref, lowercase=lowercase)
            hyp_tokens = tokenize(hyp, lowercase=lowercase)
            ref_counts = ngrams(ref_tokens, order)
            hyp_counts = ngrams(hyp_tokens, order)
            matches += sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
            hyp_total += sum(hyp_counts.values())
            ref_total += sum(ref_counts.values())
        precision = matches / hyp_total if hyp_total else 0.0
        recall = matches / ref_total if ref_total else 0.0
        scores.append(f_beta(precision, recall, beta=beta))

    return 100.0 * sum(scores) / len(scores) if scores else 0.0


def corpus_chrfpp(references: Sequence[str], hypotheses: Sequence[str], lowercase: bool = False) -> float:
    return _chrf_stats(references, hypotheses, lowercase=lowercase)


def sentence_chrfpp(reference: str, hypothesis: str, lowercase: bool = True) -> float:
    return _chrf_stats([reference], [hypothesis], lowercase=lowercase)


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Memory-efficient Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (token_a != token_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    samples: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean sentence-score difference A - B."""
    if len(scores_a) != len(scores_b) or not scores_a:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(scores_a)
    diffs = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            total += scores_a[i] - scores_b[i]
        diffs.append(total / n)
    diffs.sort()
    observed = sum(a - b for a, b in zip(scores_a, scores_b)) / len(scores_a)
    return observed, percentile(diffs, 0.025), percentile(diffs, 0.975)


def evaluate_model(
    references: Sequence[str], hypotheses: Sequence[str], name: str
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    sentence_rows: List[Dict[str, object]] = []
    total_edit_distance = 0
    total_ref_tokens = 0
    total_hyp_tokens = 0
    normalized_exact = 0
    raw_exact = 0
    empty_outputs = 0
    sent_chrf_scores: List[float] = []

    for index, (ref, hyp) in enumerate(zip(references, hypotheses), start=1):
        ref_tokens = tokenize(ref, lowercase=True)
        hyp_tokens = tokenize(hyp, lowercase=True)
        distance = levenshtein_distance(ref_tokens, hyp_tokens)
        sent_chrf = sentence_chrfpp(ref, hyp, lowercase=True)
        sent_chrf_scores.append(sent_chrf)

        total_edit_distance += distance
        total_ref_tokens += len(ref_tokens)
        total_hyp_tokens += len(hyp_tokens)
        raw_exact += int(ref.strip() == hyp.strip())
        normalized_exact += int(normalize_text(ref, True) == normalize_text(hyp, True))
        empty_outputs += int(not normalize_text(hyp))

        sentence_rows.append(
            {
                "sentence_id": index,
                "reference": ref,
                "hypothesis": hyp,
                "reference_tokens": len(ref_tokens),
                "hypothesis_tokens": len(hyp_tokens),
                "token_edit_distance": distance,
                "token_edit_rate_percent": (100.0 * distance / len(ref_tokens)) if ref_tokens else 0.0,
                "sentence_chrfpp_uncased": sent_chrf,
            }
        )

    n = len(references)
    metrics: Dict[str, float] = {
        "sentences": n,
        "bleu_cased": corpus_bleu(references, hypotheses, lowercase=False),
        "bleu_uncased": corpus_bleu(references, hypotheses, lowercase=True),
        "chrfpp_cased": corpus_chrfpp(references, hypotheses, lowercase=False),
        "chrfpp_uncased": corpus_chrfpp(references, hypotheses, lowercase=True),
        "mean_sentence_chrfpp_uncased": sum(sent_chrf_scores) / len(sent_chrf_scores) if sent_chrf_scores else 0.0,
        "token_edit_rate_percent": (100.0 * total_edit_distance / total_ref_tokens) if total_ref_tokens else 0.0,
        "raw_exact_match_percent": (100.0 * raw_exact / n) if n else 0.0,
        "normalized_exact_match_percent": (100.0 * normalized_exact / n) if n else 0.0,
        "average_reference_tokens": total_ref_tokens / n if n else 0.0,
        "average_hypothesis_tokens": total_hyp_tokens / n if n else 0.0,
        "length_ratio": total_hyp_tokens / total_ref_tokens if total_ref_tokens else 0.0,
        "empty_outputs": empty_outputs,
    }
    return metrics, sentence_rows


def choose_winner(metrics_a: Dict[str, float], metrics_b: Dict[str, float], name_a: str, name_b: str) -> str:
    # chrF++ is the primary metric because it is robust to Vietnamese morphology and tokenization.
    a = metrics_a["chrfpp_uncased"]
    b = metrics_b["chrfpp_uncased"]
    if abs(a - b) < 1e-9:
        return "Hai mô hình hòa theo chrF++ không phân biệt hoa/thường."
    winner = name_a if a > b else name_b
    loser = name_b if a > b else name_a
    return f"{winner} tốt hơn {loser} theo chrF++ không phân biệt hoa/thường."


def write_outputs(
    output_dir: Path,
    references: Sequence[str],
    name_a: str,
    metrics_a: Dict[str, float],
    rows_a: List[Dict[str, object]],
    name_b: str,
    metrics_b: Dict[str, float],
    rows_b: List[Dict[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sent_a = [float(row["sentence_chrfpp_uncased"]) for row in rows_a]
    sent_b = [float(row["sentence_chrfpp_uncased"]) for row in rows_b]
    wins_a = sum(a > b for a, b in zip(sent_a, sent_b))
    wins_b = sum(b > a for a, b in zip(sent_a, sent_b))
    ties = len(sent_a) - wins_a - wins_b
    observed_diff, ci_low, ci_high = paired_bootstrap_ci(
        sent_a, sent_b, samples=bootstrap_samples, seed=seed
    )

    comparison = {
        "primary_metric": "chrF++ uncased",
        "winner_statement": choose_winner(metrics_a, metrics_b, name_a, name_b),
        "sentence_level_chrfpp": {
            f"{name_a}_wins": wins_a,
            f"{name_b}_wins": wins_b,
            "ties": ties,
            f"mean_difference_{name_a}_minus_{name_b}": observed_diff,
            "bootstrap_95_percent_ci": [ci_low, ci_high],
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
    }

    payload = {
        "models": {name_a: metrics_a, name_b: metrics_b},
        "comparison": comparison,
        "metric_notes": {
            "BLEU": "Higher is better; corpus 1-4 gram precision with brevity penalty.",
            "chrF++": "Higher is better; character 1-6 grams and word 1-2 grams, beta=2.",
            "token_edit_rate_percent": "Lower is better; Levenshtein edits divided by reference tokens.",
            "length_ratio": "Closer to 1 is generally preferable; not a quality metric by itself.",
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_order = [
        "sentences",
        "bleu_cased",
        "bleu_uncased",
        "chrfpp_cased",
        "chrfpp_uncased",
        "mean_sentence_chrfpp_uncased",
        "token_edit_rate_percent",
        "raw_exact_match_percent",
        "normalized_exact_match_percent",
        "average_reference_tokens",
        "average_hypothesis_tokens",
        "length_ratio",
        "empty_outputs",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model"] + metric_order)
        for name, metrics in [(name_a, metrics_a), (name_b, metrics_b)]:
            writer.writerow([name] + [metrics[key] for key in metric_order])

    combined_rows = []
    for index, (ref, row_a, row_b) in enumerate(zip(references, rows_a, rows_b), start=1):
        score_a = float(row_a["sentence_chrfpp_uncased"])
        score_b = float(row_b["sentence_chrfpp_uncased"])
        if abs(score_a - score_b) < 1e-12:
            sentence_winner = "Tie"
        else:
            sentence_winner = name_a if score_a > score_b else name_b
        combined_rows.append(
            {
                "sentence_id": index,
                "reference": ref,
                f"{name_a}_hypothesis": row_a["hypothesis"],
                f"{name_b}_hypothesis": row_b["hypothesis"],
                f"{name_a}_sentence_chrfpp": score_a,
                f"{name_b}_sentence_chrfpp": score_b,
                f"chrfpp_difference_{name_a}_minus_{name_b}": score_a - score_b,
                "winner": sentence_winner,
            }
        )

    with (output_dir / "sentence_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows)

    ranked = sorted(
        combined_rows,
        key=lambda row: abs(float(row[f"chrfpp_difference_{name_a}_minus_{name_b}"])),
        reverse=True,
    )
    with (output_dir / "largest_differences.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked[:50])

    def fmt(value: float) -> str:
        return f"{value:.3f}"

    higher_metrics = [
        ("BLEU (cased)", "bleu_cased"),
        ("BLEU (uncased)", "bleu_uncased"),
        ("chrF++ (cased)", "chrfpp_cased"),
        ("chrF++ (uncased)", "chrfpp_uncased"),
        ("Mean sentence chrF++ (uncased)", "mean_sentence_chrfpp_uncased"),
        ("Normalized exact match (%)", "normalized_exact_match_percent"),
    ]
    lower_metrics = [("Token edit rate (%)", "token_edit_rate_percent")]

    lines = [
        "# Kết quả đánh giá hai mô hình dịch máy",
        "",
        f"Số câu đánh giá: **{len(references)}**.",
        "",
        "## Bảng tổng hợp",
        "",
        f"| Độ đo | {name_a} | {name_b} | Tốt hơn |",
        "|---|---:|---:|---|",
    ]
    for label, key in higher_metrics:
        a, b = metrics_a[key], metrics_b[key]
        better = "Hòa" if abs(a - b) < 1e-12 else (name_a if a > b else name_b)
        lines.append(f"| {label} ↑ | {fmt(a)} | {fmt(b)} | {better} |")
    for label, key in lower_metrics:
        a, b = metrics_a[key], metrics_b[key]
        better = "Hòa" if abs(a - b) < 1e-12 else (name_a if a < b else name_b)
        lines.append(f"| {label} ↓ | {fmt(a)} | {fmt(b)} | {better} |")
    lines.extend(
        [
            f"| Length ratio | {fmt(metrics_a['length_ratio'])} | {fmt(metrics_b['length_ratio'])} | Gần 1 hơn |",
            f"| Empty outputs ↓ | {int(metrics_a['empty_outputs'])} | {int(metrics_b['empty_outputs'])} | Ít hơn |",
            "",
            "## So sánh theo từng câu",
            "",
            f"- {name_a} thắng: **{wins_a}** câu.",
            f"- {name_b} thắng: **{wins_b}** câu.",
            f"- Hòa: **{ties}** câu.",
            f"- Chênh lệch trung bình sentence chrF++ ({name_a} − {name_b}): **{observed_diff:.3f}**.",
            f"- Khoảng tin cậy bootstrap 95%: **[{ci_low:.3f}, {ci_high:.3f}]** ({bootstrap_samples} mẫu, seed={seed}).",
            "",
            "## Kết luận",
            "",
            comparison["winner_statement"],
            "BLEU và chrF++ không phân biệt hoa/thường nên được ưu tiên khi so sánh chính trong bài này, vì đầu ra Transformer chủ yếu viết thường. Token edit rate dùng để kiểm tra mức sai khác ở cấp từ; giá trị thấp hơn là tốt hơn.",
            "",
            "## Tệp đầu ra",
            "",
            "- `summary.csv`: bảng chỉ số tổng hợp.",
            "- `metrics.json`: dữ liệu đầy đủ và kết quả bootstrap.",
            "- `sentence_scores.csv`: điểm và mô hình thắng ở từng câu.",
            "- `largest_differences.csv`: 50 câu có chênh lệch chrF++ lớn nhất.",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TensorFlow NMT and Transformer outputs.")
    parser.add_argument("--reference", type=Path, default=Path("data/tst2013.vi"))
    parser.add_argument(
        "--model-a", type=Path, default=Path("results/translation/tensorflow_tst2013.vi")
    )
    parser.add_argument("--model-a-name", default="TensorFlow NMT")
    parser.add_argument(
        "--model-b", type=Path, default=Path("results/translation/transformer_tst2013.vi")
    )
    parser.add_argument("--model-b-name", default="PyTorch Transformer")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.reference, args.model_a, args.model_b]:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    references = read_lines(args.reference)
    hypotheses_a = read_lines(args.model_a)
    hypotheses_b = read_lines(args.model_b)

    counts = {
        "reference": len(references),
        args.model_a_name: len(hypotheses_a),
        args.model_b_name: len(hypotheses_b),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(
            "Ba file phải có cùng số dòng để bảo toàn căn chỉnh câu. "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )

    metrics_a, rows_a = evaluate_model(references, hypotheses_a, args.model_a_name)
    metrics_b, rows_b = evaluate_model(references, hypotheses_b, args.model_b_name)
    write_outputs(
        args.output_dir,
        references,
        args.model_a_name,
        metrics_a,
        rows_a,
        args.model_b_name,
        metrics_b,
        rows_b,
        bootstrap_samples=max(1, args.bootstrap_samples),
        seed=args.seed,
    )

    print(f"Đã đánh giá {len(references)} câu.")
    print(f"Kết quả được lưu tại: {args.output_dir.resolve()}")
    print(f"{args.model_a_name}: BLEU={metrics_a['bleu_uncased']:.3f}, chrF++={metrics_a['chrfpp_uncased']:.3f}")
    print(f"{args.model_b_name}: BLEU={metrics_b['bleu_uncased']:.3f}, chrF++={metrics_b['chrfpp_uncased']:.3f}")
    print(choose_winner(metrics_a, metrics_b, args.model_a_name, args.model_b_name))


if __name__ == "__main__":
    main()
