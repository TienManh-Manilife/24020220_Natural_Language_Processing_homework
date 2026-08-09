from pathlib import Path
import csv
import html
import re
import unicodedata

import sacrebleu


# ============================================================
# ĐƯỜNG DẪN
# ============================================================

PROJECT_DIR = Path(
    r"D:\homework\NLP_homework\8. ASM machine translation"
)

REFERENCE_PATH = PROJECT_DIR / "data" / "tst2013.vi"

PREDICTION_PATH = (
    PROJECT_DIR
    / "results"
    / "translation"
    / "results.csv"
)


# ============================================================
# CHUẨN HÓA VĂN BẢN
# ============================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa nhẹ để tránh lỗi kỹ thuật:
    - Unicode NFC
    - HTML entities
    - khoảng trắng
    - khoảng trắng trước dấu câu

    Không lowercase vì hệ thống chấm có thể phân biệt hoa/thường.
    """
    text = "" if text is None else str(text)

    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)

    # Sửa HTML entity bị tokenizer tách rời.
    text = re.sub(
        r"&\s*quot\s*;?",
        '"',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"&\s*apos\s*;?",
        "'",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"&\s*amp\s*;?",
        "&",
        text,
        flags=re.IGNORECASE,
    )

    # Xóa khoảng trắng thừa trước dấu câu.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Chuẩn hóa nhiều khoảng trắng.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# ĐỌC GROUND TRUTH
# ============================================================

def load_references(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy ground truth: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        references = [
            normalize_text(line.rstrip("\r\n"))
            for line in file
        ]

    return references


# ============================================================
# ĐỌC FILE CSV DỰ ĐOÁN
# ============================================================

def load_predictions(
    path: Path,
    column_name: str = "Vietnamese",
) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file dự đoán: {path}"
        )

    predictions = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "File CSV không có dòng tiêu đề."
            )

        if column_name not in reader.fieldnames:
            raise ValueError(
                f"Không tìm thấy cột '{column_name}'. "
                f"Các cột hiện có: {reader.fieldnames}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            value = row.get(column_name)

            if value is None:
                raise ValueError(
                    f"Dòng CSV {row_number} không có "
                    f"giá trị cột '{column_name}'."
                )

            predictions.append(
                normalize_text(value)
            )

    return predictions


# ============================================================
# LOAD DỮ LIỆU
# ============================================================

references = load_references(
    REFERENCE_PATH
)

predictions = load_predictions(
    PREDICTION_PATH,
    column_name="Vietnamese",
)


# ============================================================
# KIỂM TRA SỐ CÂU
# ============================================================

print("=" * 72)
print("THÔNG TIN DỮ LIỆU")
print("=" * 72)

print("Ground truth :", REFERENCE_PATH)
print("Prediction   :", PREDICTION_PATH)
print("Số câu truth:", len(references))
print("Số câu pred :", len(predictions))

if len(predictions) != len(references):
    raise ValueError(
        "Số câu không khớp nên không thể tính SacreBLEU: "
        f"prediction={len(predictions)}, "
        f"reference={len(references)}"
    )


# ============================================================
# KIỂM TRA DÒNG TRỐNG
# ============================================================

empty_predictions = [
    index + 1
    for index, sentence in enumerate(predictions)
    if not sentence
]

if empty_predictions:
    print(
        f"Cảnh báo: có {len(empty_predictions)} "
        "câu dự đoán trống."
    )
    print(
        "Một số vị trí:",
        empty_predictions[:20],
    )


# ============================================================
# TÍNH SACREBLEU
# ============================================================

bleu = sacrebleu.corpus_bleu(
    predictions,
    [references],
    lowercase=False,
    tokenize="13a",
)

print()
print("=" * 72)
print("KẾT QUẢ ĐÁNH GIÁ")
print("=" * 72)

print(f"SacreBLEU EN-to-VI: {bleu.score:.4f}")
print(f"BP                 : {bleu.bp:.6f}")
print(f"System length      : {bleu.sys_len}")
print(f"Reference length   : {bleu.ref_len}")
print(
    "N-gram precision :",
    ", ".join(
        f"{value:.4f}"
        for value in bleu.precisions
    ),
)

print()
print("Chi tiết SacreBLEU:")
print(bleu)
