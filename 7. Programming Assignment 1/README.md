# Programming Assignment 1:  WORD SIMILARITY AND SYNONYM-ANTONYM CLASSIFICATION
    Sinh viên: Nguyễn Tiến Mạnh
    MSSV: 24020220

**Chú ý:** Vì file word2vec.txt quá lớn nên đang lưu dưới dạng file nén. Trước khi chạy chương trình cần phải giải nén và đặt đúng chỗ cũ (Bên cạnh file zip)  
[Hãy giải nén](./data/word2vec/)  
Có thể đặt các file dữ liệu ở vị trí khác, nhưng cần đổi path trong các file mã nguồn  

**Tải thư viện cần thiết:** pip install -r requirements.txt

**Nguồn tất cả dữ liệu:** [Word-Similarity](https://github.com/NLP-Projects/Word-Similarity)

**Đề bài:** [File PDF](./topic/Programming%20Assignment%201-2026.pdf)

**Cấu trúc thư mục:** [Tree](./tree.txt)

## Chạy chương trình: Cần cd đến thư mục của folder này. Sau đó:  
**Bài 1 - Similarity:** Chạy file similarity.py

```text
Đọc file Word2Vec
        ↓
Đọc dữ liệu ViSim-400
        ↓
Chuẩn hóa các từ
        ↓
Kiểm tra từ có trong embeddings
        ↓
Tính cosine similarity cho từng cặp từ
        ↓
Lưu kết quả vào results/similarity/
```


**Bài 2 - K-nearest words:** Chạy file nearest_words.py 

```text
Đọc file Word2Vec
        ↓
Nhập từ truy vấn w
        ↓
Chuẩn hóa từ
        ↓
Kiểm tra từ có trong embeddings
        ↓
Tính cosine similarity với toàn bộ các từ còn lại
        ↓
Sắp xếp theo cosine giảm dần
        ↓
Lấy k từ có cosine lớn nhất
        ↓
In kết quả ra màn hình và lưu kết quả vào results/nearest/
```


**Bài 3 - Synonym-antonym classification:** Chạy file classifier.py   

```text
Đọc file Word2Vec
        ↓
Đọc tập dữ liệu antonym-synonym set
        ↓
Gán nhãn SYN và ANT
        ↓
Chuẩn hóa các từ
        ↓
Loại các cặp từ trùng lặp hoặc có nhãn mâu thuẫn
        ↓
Kiểm tra từ có trong embeddings
        ↓
Tạo vector đặc trưng cho từng cặp từ
        ↓
Huấn luyện Logistic Regression
        ↓
Đọc bộ dữ liệu ViCon-400
        ↓
Tạo vector đặc trưng cho ViCon-400
        ↓
Dự đoán SYN hoặc ANT
        ↓
Tính Accuracy, Precision, Recall và F1
        ↓
Lưu kết quả vào results/classifier/
```


**Kết quả từng phần:** [Kết quả](./results/)