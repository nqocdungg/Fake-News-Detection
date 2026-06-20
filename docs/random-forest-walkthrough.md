# Random Forest Walkthrough — Báo cáo Thuật toán Tập hợp

> **Dự án:** Fake News Detection | **Học phần:** Nhập môn Trí tuệ nhân tạo  
> **Notebook:** `notebooks/07_random_forest.ipynb` | **Model:** `models/rf_model.pkl`  
> **Thực hiện bởi:** Thủy (Phase 4)

---

## 1. Lý thuyết Random Forest

### 1.1 Ý tưởng cơ bản
Random Forest là một thuật toán thuộc nhóm **Ensemble Learning** (Học tập hợp), cụ thể là phương pháp **Bagging (Bootstrap Aggregating)**. Thay vì phụ thuộc vào một cây quyết định (Decision Tree) duy nhất (dễ bị overfitting), Random Forest xây dựng một "rừng" gồm hàng trăm cây quyết định độc lập.

Quyết định phân loại cuối cùng (FAKE hay REAL) được đưa ra dựa trên cơ chế **bầu chọn đa số (Majority Voting)** từ tất cả các cây trong rừng.

### 1.2 Hai yếu tố làm nên tính ngẫu nhiên (Randomness)
Để các cây không giống hệt nhau, Random Forest áp dụng hai lớp ngẫu nhiên:
1. **Bagging (Trích mẫu ngẫu nhiên có hoàn lại):** Mỗi cây quyết định chỉ được huấn luyện trên một tập hợp con ngẫu nhiên của dữ liệu huấn luyện.
2. **Feature Randomness (Chọn đặc trưng ngẫu nhiên):** Tại mỗi bước phân nhánh (split node) của một cây, thuật toán không tìm kiếm trên toàn bộ 5,000 từ vựng TF-IDF, mà chỉ chọn ngẫu nhiên một nhóm nhỏ các đặc trưng (thường là $\sqrt{d}$, với bài toán này là $\sqrt{5000} \approx 71$ từ).

Sự ngẫu nhiên này giúp các cây trong rừng **giảm độ tương quan (decorrelation)**, nhờ đó khi kết hợp lại, mô hình sẽ giảm thiểu Variance (tính sai số do overfitting) rất hiệu quả.

### 1.3 Ưu điểm và Nhược điểm đối với Dữ liệu Text (TF-IDF)
- **Ưu điểm:** Rất mạnh mẽ, không cần giả định về phân phối dữ liệu (khác với Naive Bayes), tự động bắt được các quan hệ phi tuyến tính phức tạp giữa các cụm từ mà các mô hình tuyến tính (như Logistic Regression hay LinearSVC) có thể bỏ sót.
- **Nhược điểm:** TF-IDF tạo ra các vector cực kỳ thưa thớt (sparse matrix) với hàng ngàn chiều. Random Forest phải xây dựng vô số các quy tắc IF-ELSE trên các số 0, dẫn đến việc tiêu tốn rất nhiều RAM và **thời gian huấn luyện cực kỳ lâu** so với SVM hay Naive Bayes. Ngoài ra, cây quyết định thường thiên vị các đặc trưng có giá trị liên tục hơn là các vector TF-IDF vốn phần lớn mang giá trị 0.

---

## 2. Giải thích các bước trong Notebook (Phase 4)

File code `07_random_forest.ipynb` đã được thiết kế theo đúng luồng chuẩn để giải bài toán phân loại tin giả:

### Bước 1: Load dữ liệu TF-IDF
- Đọc các file ma trận thưa `.pkl` đã được chuẩn bị từ **Phase 3** (`X_train_tfidf.pkl`, `X_val_tfidf.pkl`...).
- Giữ nguyên cấu trúc dữ liệu không gian 5,000 chiều để bảo đảm tính so sánh công bằng giữa cả 4 mô hình trong dự án.

### Bước 2: Setup GridSearchCV
Thay vì huấn luyện 1 mô hình với tham số mặc định, chúng ta sử dụng `GridSearchCV` với 5-fold Cross Validation để máy tính thử nghiệm 18 cấu hình khác nhau:
```python
param_grid = {
    'n_estimators': [100, 200, 300], # Số lượng cây
    'max_depth': [None, 10, 20],     # Độ sâu của cây
    'min_samples_split': [2, 5]      # Số mẫu tối thiểu để tách cành
}
```
*Ghi chú:* Tham số `n_jobs=-1` được dùng để tận dụng toàn bộ số nhân (cores) của CPU, giúp giảm thời gian training. Tuy nhiên, thời gian huấn luyện ước tính vẫn sẽ mất từ 10 - 20 phút tùy thuộc vào phần cứng.

### Bước 3 & 4: Đánh giá mô hình và Vẽ Confusion Matrix
- Lấy mô hình có điểm `f1_weighted` cao nhất (`grid_search.best_estimator_`).
- Chạy phương thức `.predict(X_val)` để lấy nhãn dự đoán cho tập Validation.
- In ra **Classification Report** (Precision, Recall, F1) và dùng Seaborn để vẽ biểu đồ **Confusion Matrix** (Ma trận nhầm lẫn). Hình ảnh này được tự động xuất ra `reports/rf_confusion_matrix.png`.

### Bước 5: Model Persistence (Lưu mô hình)
- Đây là bước quan trọng nhất từng bị thiếu trong dự án. 
- Sử dụng thư viện `joblib` để lưu cấu trúc cây (model weight) xuống đĩa cứng thành file `models/rf_model.pkl`. 
- Nhờ có file này, Thủy (Phase 8) và các bạn khác có thể tải trực tiếp mô hình Random Forest lên để test chéo trên bộ dữ liệu WELFake mà không cần phải chờ 20 phút để huấn luyện lại từ đầu.

---

## 3. Hướng dẫn chạy
Để sinh ra file `.pkl` và biểu đồ báo cáo:
1. Mở file `notebooks/07_random_forest.ipynb`.
2. Trên thanh công cụ Jupyter, chọn **Cell -> Run All** (hoặc Restart & Run All).
3. Đợi tiến trình `GridSearchCV` hoàn tất.
4. Kiểm tra xem file `models/rf_model.pkl` đã xuất hiện trong hệ thống hay chưa.
