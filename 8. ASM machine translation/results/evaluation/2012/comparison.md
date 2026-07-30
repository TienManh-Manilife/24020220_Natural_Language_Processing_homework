# Kết quả đánh giá hai mô hình dịch máy

Số câu đánh giá: **1553**.

## Bảng tổng hợp

| Độ đo | TensorFlow NMT | PyTorch Transformer | Tốt hơn |
|---|---:|---:|---|
| BLEU (cased) ↑ | 19.255 | 23.034 | PyTorch Transformer |
| BLEU (uncased) ↑ | 20.016 | 25.459 | PyTorch Transformer |
| chrF++ (cased) ↑ | 37.540 | 42.899 | PyTorch Transformer |
| chrF++ (uncased) ↑ | 38.167 | 44.691 | PyTorch Transformer |
| Mean sentence chrF++ (uncased) ↑ | 38.697 | 45.259 | PyTorch Transformer |
| Normalized exact match (%) ↑ | 0.773 | 0.000 | TensorFlow NMT |
| Token edit rate (%) ↓ | 68.737 | 62.129 | PyTorch Transformer |
| Length ratio | 0.986 | 1.010 | Gần 1 hơn |
| Empty outputs ↓ | 0 | 1 | Ít hơn |

## So sánh theo từng câu

- TensorFlow NMT thắng: **409** câu.
- PyTorch Transformer thắng: **1111** câu.
- Hòa: **33** câu.
- Chênh lệch trung bình sentence chrF++ (TensorFlow NMT − PyTorch Transformer): **-6.562**.
- Khoảng tin cậy bootstrap 95%: **[-7.158, -5.985]** (1000 mẫu, seed=42).

## Kết luận

PyTorch Transformer tốt hơn TensorFlow NMT theo chrF++ không phân biệt hoa/thường.
BLEU và chrF++ không phân biệt hoa/thường nên được ưu tiên khi so sánh chính trong bài này, vì đầu ra Transformer chủ yếu viết thường. Token edit rate dùng để kiểm tra mức sai khác ở cấp từ; giá trị thấp hơn là tốt hơn.

## Tệp đầu ra

- `summary.csv`: bảng chỉ số tổng hợp.
- `metrics.json`: dữ liệu đầy đủ và kết quả bootstrap.
- `sentence_scores.csv`: điểm và mô hình thắng ở từng câu.
- `largest_differences.csv`: 50 câu có chênh lệch chrF++ lớn nhất.
