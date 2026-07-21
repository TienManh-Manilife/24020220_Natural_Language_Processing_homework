# Programming Assignment 1:  WORD SIMILARITY AND SYNONYM-ANTONYM CLASSIFICATION
    Sinh viên: Nguyễn Tiến Mạnh
    MSSV: 24020220

**Chú ý:** Vì file word2vec.txt quá lớn nên đang lưu dưới dạng file nén. Trước khi chạy chương trình cần phải giải nén và đặt đúng chỗ cũ (Bên cạnh file zip)  
[Hãy giải nén](./data/word2vec/)  
Tải thư viện cần thiết: pip install -r requirements.txt

**Nguồn tất cả dữ liệu:** [Word-Similarity](https://github.com/NLP-Projects/Word-Similarity)

**Đề bài:** [File PDF](./topic/Programming%20Assignment%201-2026.pdf)

**Cấu trúc thư mục:** [Tree](./tree.txt)

## Chạy chương trình: Cần cd đến thư mục của folder này. Sau đó:  
**Bài 1 - Similarity:** Chạy file similarity.py

    Đọc embeddings → Đọc ViSim → Tính cosine → Lưu kết quả vào results/similarity/


**Bài 2 - K-nearest words:** Chạy hàm run_nearest_word_search trong main.py  

    Nhận từ (gọi là w) → lấy vector của w → tính cosine với các từ còn lại → sắp xếp giảm dần → lấy k từ đầu tiên

**Chạy Phần 3 của bài tập:**   


**Kết quả từng phần:** [Kết quả](./results/)