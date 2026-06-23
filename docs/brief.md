# Fake News Detection bằng NLP và Machine Learning

## 1. Tổng quan dự án

### Tên đề tài
**Phát hiện tin giả (Fake News Detection) sử dụng Xử lý ngôn ngữ tự nhiên (NLP) và Học máy (Machine Learning)**

### Học phần
Nhập môn Trí tuệ nhân tạo

### Loại bài toán
Phân loại văn bản (Text Classification)

---

## 2. Bối cảnh và lý do chọn đề tài

Trong thời đại Internet và mạng xã hội phát triển mạnh mẽ, thông tin được lan truyền với tốc độ rất nhanh. Bên cạnh những nguồn tin chính thống, một lượng lớn tin giả (Fake News) cũng xuất hiện và được chia sẻ rộng rãi trên các nền tảng trực tuyến.

Tin giả có thể gây ra nhiều hậu quả tiêu cực như:
- Gây hiểu lầm và lan truyền thông tin sai lệch.
- Ảnh hưởng đến nhận thức và quyết định của người đọc.
- Tác động tiêu cực đến xã hội, kinh tế và chính trị.
- Làm giảm độ tin cậy của các nguồn thông tin trực tuyến.

Việc kiểm chứng thủ công tính xác thực của từng bài báo là tốn thời gian và phụ thuộc nhiều vào chuyên gia. Do đó, việc xây dựng một hệ thống có khả năng tự động phát hiện tin giả bằng các kỹ thuật Trí tuệ nhân tạo là một hướng nghiên cứu có ý nghĩa thực tiễn cao.

## 3. Mục tiêu dự án
- Xây dựng Pipeline NLP hoàn chỉnh: Từ tiền xử lý văn bản, trích xuất đặc trưng (Feature Extraction) cho đến huấn luyện mô hình.
- Đánh giá tính tổng quát (Generalization): Thay vì chỉ chạy trên một tập dữ liệu đơn lẻ (dễ bị overfit), dự án sẽ sử dụng đồng thời 2 dataset khác nhau để huấn luyện và kiểm thử chéo (cross-testing), đánh giá khả năng thực tế của mô hình khi gặp dữ liệu lạ.
- (Mục tiêu quan trọng nhất để làm báo cáo) Thử nghiệm & So sánh hiệu năng: So sánh 4 thuật toán Học máy phổ biến.

## 4. Dataset dự kiến

### Dataset 1: ISOT Fake News Dataset
- Dữ liệu tiếng Anh.
- Bao gồm tin thật và tin giả.
- Có đầy đủ tiêu đề và nội dung bài báo.
- Bộ dữ liệu chuẩn được sử dụng rộng rãi.

### Dataset 2 (Đang cân nhắc): 
*LIAR Dataset* hoặc *WELFake*. Đây là các tập dữ liệu có độ nhiễu cao hơn, ngữ cảnh phức tạp hơn để thử nghiệm độ bền vững (robustness) của mô hình.

## 5. Phương pháp tiếp cận

Quy trình thực hiện:

Thu thập dữ liệu
→ Tiền xử lý văn bản
→ Trích xuất đặc trưng bằng TF-IDF
→ Huấn luyện mô hình Machine Learning
→ Đánh giá kết quả
→ So sánh các thuật toán

## 6. Tiền xử lý dữ liệu (NLP)

- Chuyển về chữ thường.
- Loại bỏ dấu câu.
- Loại bỏ URL.
- Loại bỏ ký tự đặc biệt.
- Tokenization.
- Stopword Removal.
- Lemmatization.

## 7. Trích xuất đặc trưng

### TF-IDF (Term Frequency – Inverse Document Frequency)
### Unigram và Bigram** (`ngram_range`) để bắt được ngữ cảnh của các cụm từ đi liền nhau.

TF-IDF được sử dụng để chuyển đổi dữ liệu văn bản thành vector số phục vụ cho quá trình huấn luyện mô hình Machine Learning.

## 8. Các thuật toán Machine Learning

### Naive Bayes
- Mô hình xác suất.
- Huấn luyện nhanh.
- Hiệu quả trên dữ liệu văn bản.

### Logistic Regression
- Mô hình phân loại tuyến tính.
- Hiệu năng ổn định.
- Dễ triển khai và giải thích.

### Support Vector Machine (SVM)
- Hoạt động tốt trên dữ liệu nhiều chiều.
- Thường đạt kết quả cao trong các bài toán NLP.

### Random Forest
- Thuật toán Ensemble Learning.
- Giảm nguy cơ overfitting.
- Có khả năng học các quan hệ phức tạp.

## 9. Thiết kế thí nghiệm

### Thí nghiệm 1: Đánh giá Nội miền (In-domain Evaluation)
*   **Cách làm:** Train trên ISOT ➔ Test trên ISOT; Train trên WELFake ➔ Test trên WELFake.
*   **Mục tiêu:** Xác định mức trần hiệu năng (baseline) của từng mô hình khi dữ liệu kiểm thử cùng phân phối với dữ liệu huấn luyện.

### Thí nghiệm 2: Đánh giá Chéo miền (Cross-domain Evaluation)
*   **Cách làm:** Train trên ISOT ➔ Test trên WELFake; Train trên WELFake ➔ Test trên ISOT.
*   **Mục tiêu:** Đánh giá khả năng tổng quát hóa (Generalization) khi mô hình phải đối mặt với dữ liệu từ nguồn hoàn toàn mới (kiểm chứng hiện tượng domain shift).

### Thí nghiệm 3: Huấn luyện Gộp (Pooled-Domain Evaluation)
*   **Cách làm:** Gộp tập Train của ISOT và WELFake thành tập huấn luyện chung ➔ Huấn luyện mô hình ➔ Đánh giá độc lập trên tập Test của từng bộ.
*   **Mục tiêu:** Đánh giá hiệu quả của việc gia tăng quy mô và tính đa dạng của dữ liệu huấn luyện lên cả hiệu năng nội miền và khả năng tổng quát hóa.

## 10. Độ đo đánh giá

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 11. Kết quả kỳ vọng

- Xây dựng thành công hệ thống phát hiện tin giả tự động.
- So sánh được hiệu năng của 4 thuật toán Machine Learning.
- Xác định được mô hình phù hợp nhất.
- Đánh giá được khả năng tổng quát hóa của mô hình trên nhiều dataset.

## 12. Công nghệ sử dụng

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Git/GitHub

## 13. Định hướng mở rộng

- Word2Vec.
- GloVe.
- LSTM/BiLSTM.
- BERT.
- Xây dựng Web Demo cho hệ thống phát hiện tin giả.
