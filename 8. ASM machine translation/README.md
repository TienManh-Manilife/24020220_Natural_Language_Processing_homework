# ASM machine translation

Kết quả xem tại: [./results/](./results/)

Mô hình transformer ở thư mục results/models/transformer/ không up lên git vì có files quá nặng không up lên được. Có thể lấy mô hình sẵn có từ demo_transformer.ipynb

## 1. Mục tiêu bài tập

Bài tập xây dựng, chạy suy luận và đánh giá hai hệ thống dịch máy từ tiếng Anh sang tiếng Việt trên bộ dữ liệu IWSLT15:

1. **PyTorch Transformer**: mô hình Transformer encoder–decoder được chạy từ notebook `demo_transformer.ipynb`.
2. **TensorFlow NMT**: mô hình sequence-to-sequence dùng RNN và cơ chế attention từ repository `tensorflow/nmt`.

Hai mô hình được đánh giá trên cùng tập `tst2012`. Mỗi dòng trong file đầu ra tương ứng chính xác với cùng dòng trong file dịch chuẩn tiếng Việt.

Mục tiêu chính là:

- So sánh chất lượng bản dịch bằng nhiều độ đo.
- Phân tích ưu, nhược điểm của hai kiến trúc.
- Lưu kết quả ở dạng CSV, JSON và Markdown để có thể đưa trực tiếp vào báo cáo.
- Bảo đảm quy trình có thể chạy lại trong môi trường Docker TensorFlow 1.15/Python 3.6 mà không cần cài thêm thư viện.

## 2. Dữ liệu và cách chia tập

Các file IWSLT15 được sử dụng theo cách sau:

| Mục đích   | Tiếng Anh    | Tiếng Việt   |
| ---------- | ------------ | ------------ |
| Huấn luyện | `train.en`   | `train.vi`   |
| Test       | `tst2012.en` | `tst2012.vi` |
| Test       | `tst2013.en` | `tst2013.vi` |

## 3. Cấu trúc thư mục

```text
8. ASM machine translation/
├── README.md
├── evaluate_models.py
├── data/
│   └── tst2012.vi
└── results/
    ├── translation/
    │   ├── tensorflow_tst2012.vi
    │   └── transformer_tst2012.vi
    ├── evaluation/
    │   ├── 2012/
    │   └── 2013/
    └── translation/
        ├── tensorflow_tst2012.vi
        ├── tensorflow_tst2013.vi
        ├── transformer_tst2012.vi
        └── transformer_tst2013.vi
```

Trong cấu trúc làm việc Docker của bài tập, thư mục tương đương là:

```text
/workspace/
├── tensorflow_nmt/
├── data/
├── results/
└── evaluate_models.py
```

## 4. Hai mô hình được so sánh

### 4.1. PyTorch Transformer

Transformer sử dụng attention để mô hình hóa quan hệ giữa các token mà không cần xử lý tuần tự như RNN. Quy trình tổng quát trong notebook gồm:

1. Đọc dữ liệu song ngữ Anh–Việt.
2. Xây dựng vocabulary từ tập huấn luyện.
3. Chuyển câu thành chuỗi chỉ số token.
4. Huấn luyện hoặc nạp checkpoint Transformer.
5. Dịch từng câu của `tst2012.en`.
6. Ghi mỗi câu dịch vào `transformer_tst2012.vi`.

Đầu ra của mô hình này chủ yếu được viết thường. Vì vậy báo cáo sử dụng thêm các chỉ số không phân biệt chữ hoa/chữ thường để tránh phạt mô hình chỉ vì khác cách viết hoa đầu câu.

### 4.2. TensorFlow NMT

TensorFlow NMT sử dụng mô hình encoder–decoder tuần tự với attention. Cấu hình đã dùng trong lần chạy của bài tập gồm:

- Nguồn: `en`.
- Đích: `vi`.
- 2 lớp.
- 128 hidden units.
- Dropout: 0.2.
- Attention: `scaled_luong`.
- Số bước huấn luyện: 12.000.

Ví dụ lệnh huấn luyện từ `/workspace/tensorflow_nmt`:

```bash
python -m nmt.nmt \
  --src=en \
  --tgt=vi \
  --vocab_prefix=../data/vocab \
  --train_prefix=../data/train \
  --dev_prefix=../data/tst2012 \
  --test_prefix=../data/tst2013 \
  --out_dir=../results/models/tensorflow \
  --num_train_steps=12000 \
  --steps_per_stats=100 \
  --steps_per_external_eval=1000 \
  --num_layers=2 \
  --num_units=128 \
  --dropout=0.2 \
  --attention=scaled_luong
```

Ví dụ sinh bản dịch cho `tst2012.en` bằng checkpoint tốt nhất:

```bash
mkdir -p ../results/translation

python -m nmt.nmt \
  --src=en \
  --tgt=vi \
  --vocab_prefix=../data/vocab \
  --out_dir=../results/models/tensorflow \
  --ckpt=../results/models/tensorflow/best_bleu/translate.ckpt-11000 \
  --inference_input_file=../data/tst2012.en \
  --inference_output_file=../results/translation/tensorflow_tst2012.vi
```

Checkpoint TensorFlow gồm ba file có cùng prefix:

```text
translate.ckpt-11000.data-00000-of-00001
translate.ckpt-11000.index
translate.ckpt-11000.meta
```

Tham số `--ckpt` phải là prefix `translate.ckpt-11000`, không thêm phần mở rộng.

## 5. Cách chạy đánh giá

### 5.1. Chạy trong thư mục dự án kèm theo

```bash
cd mt_translation_assignment
python evaluate_models.py
```

Script mặc định đọc:

```text
data/tst2012.vi
results/translation/tensorflow_tst2012.vi
results/translation/transformer_tst2012.vi
```

và lưu kết quả vào:

```text
results/evaluation/
```

### 5.2. Chạy với đường dẫn tùy chỉnh

```bash
python evaluate_models.py \
  --reference data/tst2012.vi \
  --model-a results/translation/tensorflow_tst2012.vi \
  --model-a-name "TensorFlow NMT" \
  --model-b results/translation/transformer_tst2012.vi \
  --model-b-name "PyTorch Transformer" \
  --output-dir results/evaluation \
  --bootstrap-samples 1000 \
  --seed 42
```

Trong Docker, nên đặt script tại `/workspace/evaluate_models.py`, sau đó chạy:

```bash
cd /workspace
python evaluate_models.py
```

Không cần cài `sacrebleu`, `pandas`, `numpy` hoặc thư viện ngoài. Script chỉ sử dụng Python standard library và tương thích cú pháp Python 3.6.

## 6. Các độ đo đánh giá

### 6.1. BLEU-4

BLEU đo mức trùng khớp n-gram giữa bản dịch mô hình và bản dịch chuẩn, đồng thời áp dụng brevity penalty nếu đầu ra quá ngắn.

- Dùng n-gram từ 1 đến 4.
- Giá trị càng cao càng tốt.
- Báo cáo cả bản **cased** và **uncased**.

Script sử dụng tokenizer Unicode nội bộ, tách từ và dấu câu. Vì tokenizer có thể khác SacreBLEU, không nên so sánh trực tiếp con số này với một bài báo dùng cấu hình SacreBLEU khác. Kết quả vẫn hợp lệ để so sánh hai mô hình trong cùng bài tập vì cả hai được đánh giá bằng cùng một quy trình.

### 6.2. chrF++

chrF++ đo độ trùng khớp ở cấp ký tự và từ:

- Character n-gram từ 1 đến 6.
- Word n-gram từ 1 đến 2.
- `beta = 2`, nhấn mạnh recall hơn precision.
- Giá trị càng cao càng tốt.

chrF++ phù hợp với tiếng Việt vì ít phụ thuộc hơn vào cách tách token và có thể ghi nhận các phần từ/cụm từ trùng khớp ngay cả khi toàn bộ câu chưa giống hoàn toàn.

### 6.3. Token edit rate

Token edit rate được tính bằng tổng số phép chèn, xóa và thay thế token theo khoảng cách Levenshtein, chia cho tổng số token của bản dịch chuẩn.

- Giá trị càng thấp càng tốt.
- Đây là token error rate dựa trên edit distance, không phải TER đầy đủ có phép dịch chuyển cụm từ.

### 6.4. Exact match

Exact match đo tỷ lệ câu đầu ra giống hoàn toàn bản dịch chuẩn.

- `raw_exact_match`: so sánh trực tiếp.
- `normalized_exact_match`: chuẩn hóa khoảng trắng và chữ hoa/thường trước khi so sánh.

Đây là chỉ số rất nghiêm ngặt. Một bản dịch đúng nghĩa nhưng khác dấu câu hoặc dùng từ đồng nghĩa vẫn bị tính là sai hoàn toàn. Vì vậy exact match chỉ là chỉ số phụ.

### 6.5. Length ratio

```text
length ratio = tổng số token đầu ra / tổng số token tham chiếu
```

Giá trị gần 1 cho thấy độ dài tổng thể của đầu ra gần với bản dịch chuẩn. Chỉ số này không tự xác nhận chất lượng bản dịch, nhưng giúp phát hiện mô hình dịch quá ngắn hoặc quá dài.

### 6.6. So sánh từng câu và bootstrap

Mỗi câu được chấm sentence-level chrF++. Script đếm:

- Số câu TensorFlow NMT có điểm cao hơn.
- Số câu Transformer có điểm cao hơn.
- Số câu hòa.

Script cũng lấy mẫu bootstrap có hoàn lại trên các cặp câu để ước lượng khoảng tin cậy 95% của chênh lệch mean sentence chrF++. Vì hai mô hình được chấm trên cùng câu, đây là paired bootstrap.

## 7. Kết quả thực nghiệm

Kết quả trên 1.553 câu `tst2012`:

| Độ đo                          | TensorFlow NMT | PyTorch Transformer | Tốt hơn        |
| ------------------------------ | -------------: | ------------------: | -------------- |
| BLEU cased ↑                   |         19.255 |              23.034 | Transformer    |
| BLEU uncased ↑                 |         20.016 |              25.459 | Transformer    |
| chrF++ cased ↑                 |         37.540 |              42.899 | Transformer    |
| chrF++ uncased ↑               |         38.167 |              44.691 | Transformer    |
| Mean sentence chrF++ uncased ↑ |         38.697 |              45.259 | Transformer    |
| Token edit rate ↓              |        68.737% |             62.129% | Transformer    |
| Normalized exact match ↑       |         0.773% |              0.000% | TensorFlow NMT |
| Length ratio                   |          0.986 |               1.010 | Cả hai gần 1   |
| Số đầu ra rỗng ↓               |              0 |                   1 | TensorFlow NMT |

So sánh theo từng câu:

- TensorFlow NMT thắng: **409 câu**.
- PyTorch Transformer thắng: **1.111 câu**.
- Hòa: **33 câu**.
- Chênh lệch mean sentence chrF++ theo thứ tự TensorFlow − Transformer: **-6.562**.
- Khoảng tin cậy bootstrap 95%: **[-7.158, -5.985]**.

Khoảng tin cậy hoàn toàn nhỏ hơn 0, nên trong tập đánh giá này Transformer có lợi thế ổn định theo sentence chrF++. Đây là kết luận thống kê từ phép bootstrap trên chính 1.553 cặp câu, không phải khẳng định tổng quát cho mọi tập dữ liệu.

## 8. Nhận xét và so sánh

### 8.1. Chất lượng tổng thể

Transformer tốt hơn TensorFlow NMT trên các chỉ số chính:

- BLEU uncased cao hơn khoảng **5,443 điểm**.
- chrF++ uncased cao hơn khoảng **6,524 điểm**.
- Token edit rate thấp hơn khoảng **6,608 điểm phần trăm**.
- Transformer thắng ở khoảng **71,5%** số câu; TensorFlow NMT thắng ở khoảng **26,3%** số câu.

Kết quả cho thấy Transformer tái tạo n-gram và chuỗi ký tự gần bản dịch chuẩn hơn, đồng thời cần ít thao tác chỉnh sửa token hơn.

### 8.2. Ảnh hưởng của chữ hoa/thường

Khoảng cách giữa BLEU cased và uncased của Transformer lớn hơn TensorFlow NMT vì đầu ra Transformer chủ yếu viết thường. Nếu chỉ sử dụng BLEU cased, mô hình bị phạt thêm vì khác cách viết hoa đầu câu dù nội dung token có thể giống nhau.

Do đó:

- Dùng **BLEU uncased** và **chrF++ uncased** làm chỉ số so sánh chính.
- Vẫn lưu chỉ số cased để phản ánh chất lượng bề mặt và khả năng viết hoa.

### 8.3. Exact match không nên là chỉ số chính

TensorFlow NMT có một số câu khớp hoàn toàn, trong khi Transformer không có câu exact match. Tuy nhiên Transformer vẫn tốt hơn rõ rệt theo BLEU, chrF++ và token edit rate. Điều này minh họa rằng exact match quá nghiêm ngặt và không phản ánh đầy đủ chất lượng dịch máy.

### 8.4. Độ dài đầu ra

Cả hai mô hình có length ratio gần 1:

- TensorFlow NMT: 0.986, hơi ngắn hơn tham chiếu.
- Transformer: 1.010, hơi dài hơn tham chiếu.

Không có dấu hiệu nghiêm trọng về việc một mô hình luôn tạo câu quá ngắn hoặc quá dài. Tuy nhiên Transformer có một câu đầu ra rỗng ở dòng 867; cần kiểm tra lỗi dữ liệu hoặc lỗi inference tại câu đó.

## 9. Các file kết quả

### `results/evaluation/summary.csv`

Một dòng cho mỗi mô hình, phù hợp để mở bằng Excel hoặc đưa vào notebook.

### `results/evaluation/metrics.json`

Chứa toàn bộ chỉ số, thông tin so sánh, số câu thắng/hòa/thua và khoảng tin cậy bootstrap.

### `results/evaluation/comparison.md`

Báo cáo Markdown ngắn gọn được sinh tự động từ kết quả.

### `results/evaluation/sentence_scores.csv`

Chứa:

- Bản dịch chuẩn.
- Đầu ra của hai mô hình.
- Sentence chrF++ của từng mô hình.
- Chênh lệch điểm.
- Mô hình thắng ở từng câu.

File này phù hợp cho error analysis.

### `results/evaluation/largest_differences.csv`

Chứa 50 câu có chênh lệch sentence chrF++ lớn nhất. Có thể dùng để chọn ví dụ minh họa cho phần phân tích định tính trong báo cáo.

## 10. Hạn chế

- Chỉ có một bản dịch tham chiếu cho mỗi câu; các bản dịch đồng nghĩa hợp lệ có thể bị điểm thấp.
- BLEU phụ thuộc tokenizer. Kết quả trong script dùng tokenizer Unicode nội bộ, không có SacreBLEU signature.
- chrF++ đo độ giống bề mặt, không trực tiếp đo tính đúng nghĩa hoặc tính tự nhiên.
- Token edit rate không hỗ trợ phép shift như TER chuẩn.
- Tập đánh giá chỉ gồm `tst2012`; kết luận nên được kiểm tra thêm trên `tst2013`.
- Chưa có đánh giá của con người về tính đầy đủ, chính xác và trôi chảy.
- Cấu hình hai mô hình có thể khác nhau về số tham số, thời gian huấn luyện và mức độ tối ưu, nên kết quả không chỉ phản ánh khác biệt kiến trúc.

## 11. Kết luận

Trên tập `tst2012` được cung cấp, **PyTorch Transformer là mô hình tốt hơn** theo BLEU, chrF++, token edit rate và số câu thắng. TensorFlow NMT vẫn tạo được các bản dịch hợp lệ và có một số câu khớp hoàn toàn, nhưng chất lượng tổng thể thấp hơn. Khoảng tin cậy bootstrap của chênh lệch sentence chrF++ không chứa 0, cho thấy lợi thế của Transformer trên tập này khá ổn định.
