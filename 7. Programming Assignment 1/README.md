# Programming Assignment 1:  WORD SIMILARITY AND SYNONYM-ANTONYM CLASSIFICATION
    Sinh viên: Nguyễn Tiến Mạnh
    MSSV: 24020220
**Nguồn dữ liệu:** [Word-Similarity](https://github.com/NLP-Projects/Word-Similarity)

**Đề bài:** [File PDF](./topic/Programming%20Assignment%201-2026.pdf)

**Cấu trúc thư mục:** [Tree](./tree.txt)

**Chạy Phần 1 của bài tập:**   
    python .\main.py similarity `
  --embeddings ".\data\Word2vec\word2vec.txt" `
  --visim ".\data\ViSim-400\Visim-400.txt" `
  --results-dir ".\results\similarity"

**Chạy Phần 2 của bài tập:**   
    python .\main.py nearest `
  --embeddings ".\data\Word2vec\word2vec.txt" `
  --word "sinh_viên" `
  --word "thông_minh" `
  --word "vui_vẻ" `
  --word "học" `
  --word "thành_phố" `
  --k 10 `
  --results-dir ".\results\nearest"

**Chạy Phần 3 của bài tập:**   
    python .\main.py classify `
   --embeddings ".\data\Word2vec\word2vec.txt" `
   --synonyms ".\data\antonym-synonym-set\Synonym_vietnamese.txt" `
   --antonyms ".\data\antonym-synonym-set\Antonym_vietnamese.txt" `
   --vicon-noun ".\data\ViCon-400\400_noun_pairs.txt" `
   --vicon-verb ".\data\ViCon-400\400_verb_pairs.txt" `
   --vicon-adj ".\data\ViCon-400\600_adj_pairs.txt" `
   --model logreg `
   --results-dir ".\results\classification"