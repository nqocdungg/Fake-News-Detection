# Phase 4 — Model Training: Support Vector Machine (SVM)

> **Trạng thái tổng thể:** 🔄 Đang thực hiện  
> **Notebook output:** `notebooks/06_svm.ipynb`  
> **Model output:** `models/svm_model.pkl`  
> **Cập nhật:** 2026-06-17

---

## Mục tiêu

So sánh hai kernel SVM phổ biến cho bài toán phân loại văn bản:
- **LinearSVC** — kernel tuyến tính, tối ưu cho dữ liệu thưa (sparse) TF-IDF chiều cao
- **SVC(kernel='rbf')** — kernel phi tuyến, dùng subsample để đánh giá tính khả dụng

Chọn ra model tốt hơn, lưu lại để sử dụng ở Phase 5 (comparison).

---

## Dữ liệu đầu vào

| File | Shape | Mô tả |
|------|-------|--------|
| `data/X_train_tfidf.pkl` | (27057, 5000) | Sparse matrix TF-IDF train set |
| `data/X_val_tfidf.pkl` | (5798, 5000) | Sparse matrix TF-IDF val set |
| `data/X_test_tfidf.pkl` | (5798, 5000) | Sparse matrix TF-IDF test set *(không dùng ở phase này)* |
| `data/y_train.pkl` | (27057,) | Nhãn train: 0=FAKE, 1=REAL |
| `data/y_val.pkl` | (5798,) | Nhãn val |

**Phân phối nhãn train:** FAKE=14,837 / REAL=12,220 (tỉ lệ ~55/45, gần balanced)

---

## Các bước thực hiện

---

### Bước 1 — Import & Setup
**Trạng thái:** ✅ Hoàn thành  

**Làm gì:** Import các thư viện cần thiết, set `random_state=42` toàn cục để đảm bảo reproducibility. Ghi lại phiên bản scikit-learn.

**Lý do:** Mọi thí nghiệm phải reproducible. `random_state=42` được dùng nhất quán toàn dự án (xem AGENTS.md).

---

### Bước 2 — Load dữ liệu
**Trạng thái:** ✅ Hoàn thành  

**Làm gì:** Dùng `joblib.load()` để load 4 file pickle (X_train, X_val, y_train, y_val). In ra shape và phân phối nhãn để xác nhận data integrity.

**Lý do:** Các file được lưu bằng `joblib` ở Phase 3. `joblib` xử lý sparse matrix tốt hơn `pickle` thuần. **Không load X_test** — test set chỉ được dùng lần đầu ở Phase 5.

---

### Bước 3 — LinearSVC: Baseline
**Trạng thái:** ✅ Hoàn thành  

**Làm gì:** Train `LinearSVC(C=1.0, max_iter=2000)` không tune, predict trên val set, in classification report nhanh.

**Lý do:** Baseline giúp có điểm tham chiếu trước khi GridSearch. Nếu baseline đã rất cao (>0.95), GridSearch có thể chỉ cải thiện nhỏ. `max_iter=2000` vì LinearSVC đôi khi không hội tụ với 1000 iter mặc định trên dữ liệu văn bản.

---

### Bước 4 — LinearSVC: GridSearchCV
**Trạng thái:** ✅ Hoàn thành  

**Làm gì:** GridSearch trên:
- `C` ∈ [0.01, 0.1, 1, 10] — kiểm soát margin vs misclassification
- `max_iter` = 2000 (fixed)
- `cv=5`, `scoring='f1_weighted'`

Vẽ đồ thị F1 trên val theo từng giá trị C.

**Lý do:** `C` là hyperparameter quan trọng nhất của SVM. C nhỏ → margin rộng, regularization mạnh, có thể underfit. C lớn → margin hẹp, fit sát train data, có thể overfit. Với TF-IDF sparse, C=1 thường là điểm tốt.

---

### Bước 5 — SVC RBF: Đánh giá trên subsample
**Trạng thái:** ✅ Hoàn thành  

**Làm gì:** Lấy stratified subsample 4,000 mẫu từ train set (để đảm bảo tỉ lệ FAKE/REAL). GridSearch trên:
- `C` ∈ [0.1, 1, 10]
- `gamma` ∈ ['scale', 'auto']
- `cv=5`, `scoring='f1_weighted'`

Ghi lại thời gian training.

**Lý do:** `SVC(kernel='rbf')` có độ phức tạp O(n²) đến O(n³). Trên 27K mẫu, GridSearch sẽ mất nhiều giờ. Dùng subsample 4K là đủ để so sánh tính khả dụng. **Kết quả trên subsample sẽ không công bằng hoàn toàn**, nhưng đủ để kết luận về sự phù hợp của RBF cho bài toán này.

---

### Bước 6 — So sánh LinearSVC vs SVC RBF
**Trạng thái:** ✅ Hoàn thành  

**Làm gì:** Bảng so sánh gồm:
- F1 trên val set
- Training time
- Phù hợp với dữ liệu sparse TF-IDF

Kết luận và chọn kernel tốt hơn.

**Lý do:** Với dữ liệu TF-IDF (sparse, chiều cao, tuyến tính separable trong nhiều trường hợp), LinearSVC thường vượt trội SVC RBF cả về accuracy lẫn tốc độ. Đây là lý do có bước so sánh tường minh.

---

### Bước 7 — Đánh giá model tốt nhất trên Validation Set
**Trạng thái:** ✅ Hoàn thành  

**Làm gì:** Dùng model thắng từ Bước 6 (với best params từ GridSearch), predict toàn bộ val set:
- Classification Report (Precision, Recall, F1 per class)
- Confusion Matrix (heatmap)
- Tổng training time

**Lý do:** Đây là đánh giá chính thức trên val set để ghi vào báo cáo. Test set vẫn chưa được dùng.

---

### Bước 8 — Lưu model
**Trạng thái:** ✅ Hoàn thành  

**Làm gì:** `joblib.dump(best_svm, 'models/svm_model.pkl')`. In đường dẫn và kích thước file.

**Lý do:** Model cần được persist để Phase 5 load lại tất cả 4 model và compare trên test set. Dùng `joblib` vì hiệu quả hơn `pickle` với numpy/sparse arrays.

---

## Kết quả kỳ vọng

| Metric | LinearSVC (dự kiến) |
|--------|---------------------|
| Accuracy | > 0.97 |
| F1 (weighted) | > 0.97 |
| Training time | < 30s |

> SVM là một trong những thuật toán mạnh nhất cho NLP text classification với TF-IDF. Kỳ vọng nó sẽ là top performer cùng với Logistic Regression.

---

## Files liên quan

| File | Vai trò |
|------|---------|
| `notebooks/06_svm.ipynb` | Notebook thực thi chính |
| `models/svm_model.pkl` | Model đã train (LinearSVC, best params) |
| `src/preprocessing.py` | Không dùng trực tiếp ở phase này (đã dùng ở Phase 1) |
| `data/X_*_tfidf.pkl` | Input features |
| `data/y_*.pkl` | Input labels |
