# Reflection — Lab 19

**Tên:** Luu Thien Viet Cuong
**Cohort:** A20
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

Trên golden set, hybrid có Precision@10 trung bình cao nhất vì RRF cộng tín hiệu từ BM25 và vector theo rank 1-based. Với `exact`, BM25 rất mạnh vì query chứa đúng thuật ngữ như Kubernetes, OAuth, PostgreSQL. Với `mixed`, hybrid tốt hơn vì query có cả keyword kỹ thuật và phần diễn giải tiếng Việt. Với `paraphrase`, vector hữu ích hơn keyword, nhưng model `bge-small-en` chưa tối ưu tiếng Việt nên hybrid vẫn giúp ổn định kết quả. Tôi sẽ không dùng hybrid khi hệ thống chỉ tìm mã lỗi, tên API, ID, hoặc log token chính xác; lúc đó BM25 đơn giản hơn, nhanh hơn, và dễ debug hơn.

---

## Điều ngạc nhiên nhất khi làm lab này

Hybrid không cần score gốc cùng scale; chỉ cần rank từ từng retriever là đã cải thiện độ ổn định.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [x] Pair work với: Nguyen Hoai Bao Ngoc
