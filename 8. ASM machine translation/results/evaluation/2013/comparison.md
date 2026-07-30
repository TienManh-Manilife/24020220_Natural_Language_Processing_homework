# Kết quả đánh giá hai mô hình dịch máy

Số câu đánh giá: **1268**.

## Bảng tổng hợp

| Độ đo | TensorFlow NMT | PyTorch Transformer | Tốt hơn |
|---|---:|---:|---|
| BLEU (cased) ↑ | 20.398 | 24.780 | PyTorch Transformer |
| BLEU (uncased) ↑ | 21.024 | 27.857 | PyTorch Transformer |
| chrF++ (cased) ↑ | 38.550 | 44.074 | PyTorch Transformer |
| chrF++ (uncased) ↑ | 39.108 | 46.354 | PyTorch Transformer |
| Mean sentence chrF++ (uncased) ↑ | 40.699 | 48.191 | PyTorch Transformer |
| Normalized exact match (%) ↑ | 0.552 | 0.000 | TensorFlow NMT |
| Token edit rate (%) ↓ | 66.551 | 58.622 | PyTorch Transformer |
| Length ratio | 0.942 | 0.957 | Gần 1 hơn |
| Empty outputs ↓ | 0 | 0 | Ít hơn |

## So sánh theo từng câu

- TensorFlow NMT thắng: **290** câu.
- PyTorch Transformer thắng: **953** câu.
- Hòa: **25** câu.
- Chênh lệch trung bình sentence chrF++ (TensorFlow NMT − PyTorch Transformer): **-7.491**.
- Khoảng tin cậy bootstrap 95%: **[-8.174, -6.860]** (1000 mẫu, seed=42).

## Kết luận

PyTorch Transformer tốt hơn TensorFlow NMT theo chrF++ không phân biệt hoa/thường.
BLEU và chrF++ không phân biệt hoa/thường nên được ưu tiên khi so sánh chính trong bài này, vì đầu ra Transformer chủ yếu viết thường. Token edit rate dùng để kiểm tra mức sai khác ở cấp từ; giá trị thấp hơn là tốt hơn.

## Tệp đầu ra

- `summary.csv`: bảng chỉ số tổng hợp.
- `metrics.json`: dữ liệu đầy đủ và kết quả bootstrap.
- `sentence_scores.csv`: điểm và mô hình thắng ở từng câu.
- `largest_differences.csv`: 50 câu có chênh lệch chrF++ lớn nhất.
