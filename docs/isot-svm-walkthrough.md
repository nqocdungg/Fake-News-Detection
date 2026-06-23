# SVM Walkthrough — Báo cáo Thuật toán Support Vector Machine

> **Dự án:** Fake News Detection | **Học phần:** Nhập môn Trí tuệ nhân tạo  
> **Notebook:** `notebooks/06_svm.ipynb` | **Model:** `models/svm_model.pkl`

---

## 1. Lý thuyết Support Vector Machine

### 1.1 Ý tưởng cơ bản

Support Vector Machine (SVM) là thuật toán học có giám sát tìm **siêu phẳng phân tách** (hyperplane) tối ưu giữa hai lớp dữ liệu.

Với bài toán phân loại nhị phân (FAKE/REAL), SVM tìm hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ sao cho **margin** — khoảng cách từ hyperplane đến điểm dữ liệu gần nhất của mỗi lớp — là **lớn nhất có thể**.

```
      REAL (class 0) |       | FAKE (class 1)
                  |         |
       ●   ● ●   |         |   ○  ○
         ●    ● ←|— margin →|→ ○   ○
                  |         |
              hyperplane  support vectors
```

Những điểm dữ liệu nằm ngay trên biên margin được gọi là **support vectors** — đây là những điểm duy nhất quyết định vị trí hyperplane.

### 1.2 Bài toán tối ưu hóa

**Hard-margin SVM** (dữ liệu linearly separable):

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t. } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \forall i$$

**Soft-margin SVM** (thực tế — cho phép một số điểm vi phạm margin):

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{s.t. } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Trong đó:
- $\xi_i$ là **slack variable** — đo mức độ vi phạm margin của điểm $i$
- $C$ là **regularization parameter** kiểm soát đánh đổi giữa margin rộng và lỗi phân loại

### 1.3 Tham số C

| Giá trị C | Ý nghĩa | Hệ quả |
|-----------|---------|--------|
| C nhỏ (0.01) | Regularization mạnh | Margin rộng, chấp nhận nhiều lỗi hơn → dễ underfit |
| C = 1 | Cân bằng (mặc định) | Thường cho kết quả tốt ở hầu hết bài toán |
| C lớn (10+) | Ít regularization | Cố gắng phân loại đúng tất cả → dễ overfit |

### 1.4 LinearSVC vs SVC với Kernel RBF

#### LinearSVC

`LinearSVC` sử dụng thuật toán **liblinear** để giải trực tiếp bài toán SVM tuyến tính:

- Độ phức tạp: **O(n × d)** — tuyến tính theo số mẫu và số chiều
- Tận dụng **sparse matrix** (TF-IDF) hiệu quả
- Không xây dựng kernel matrix → rất nhanh

#### SVC với Kernel RBF

Kernel RBF (Radial Basis Function) ánh xạ dữ liệu vào không gian Hilbert chiều vô hạn:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

- Độ phức tạp: **O(n² × d)** để xây kernel matrix → không khả thi với 27K mẫu
- Tham số **gamma** kiểm soát "bán kính ảnh hưởng":
  - `gamma='scale'` = $\frac{1}{d \cdot \text{Var}(X)}$ → phù hợp với normalized features
  - `gamma='auto'` = $\frac{1}{d}$ = $\frac{1}{5000}$ = 0.0002 → **quá nhỏ** với TF-IDF: kernel gần 0, mọi điểm đều "xa nhau" → model degenerate

---

## 2. Ứng dụng vào Fake News Detection

### 2.1 Tại sao SVM phù hợp với Text Classification?

Sau khi vectorize bằng TF-IDF, mỗi bài báo trở thành vector thưa (sparse) trong không gian 5,000 chiều. SVM hoạt động tốt trong trường hợp này vì:

1. **Chiều cao, mẫu ít hơn chiều** — SVM tìm max-margin hyperplane, hiệu quả ngay cả khi $d \gg n$
2. **Linearly separable trong không gian TF-IDF** — Fake news và real news thường dùng từ ngữ khác nhau rõ rệt, tạo ra ranh giới tuyến tính rõ trong không gian TF-IDF
3. **Robust với nhiễu** — Margin maximization giúp model ít bị ảnh hưởng bởi outlier

### 2.2 Pipeline

```
Bài báo thô
  → preprocess_text()     [src/preprocessing.py]
  → TF-IDF transform()    [models/tfidf_vectorizer.pkl — CHỈ transform, không fit lại]
  → LinearSVC.predict()   [models/svm_model.pkl]
  → REAL (0) / FAKE (1)
```

---

## 3. Thiết kế Thí nghiệm

### 3.1 Dữ liệu

| Split | Số mẫu | REAL | FAKE |
|-------|--------|------|------|
| Train | 27,057 | 14,837 (54.8%) | 12,220 (45.2%) |
| Val   | 5,798  | 3,179  (54.8%) | 2,619  (45.2%) |
| Test  | 5,798  | — | — *(chưa dùng)* |

Dataset gần **balanced** → không cần class weighting đặc biệt.

### 3.2 GridSearchCV — LinearSVC

```python
GridSearchCV(
    LinearSVC(max_iter=2000, random_state=42),
    param_grid = {'C': [0.01, 0.1, 1, 10]},
    cv          = 5,
    scoring     = 'f1_weighted',
    n_jobs      = -1
)
```

Kết quả từng giá trị C (5-fold cross-validation):

| C | CV F1 (mean) | CV F1 (std) |
|---|:---:|:---:|
| 0.01 | 0.9648 | ±0.0040 |
| 0.1 | 0.9812 | ±0.0013 |
| **1** | **0.9852** | ±0.0017 |
| 10 | 0.9815 | ±0.0012 |

**Nhận xét:** C=1 cho F1 cao nhất. C=10 giảm nhẹ — có dấu hiệu overfit nhỏ. Điều này xác nhận regularization mức vừa phải là tối ưu cho ISOT dataset.

### 3.3 Phân tích SVC RBF

Do hạn chế tính toán (O(n²)), SVC RBF được đánh giá trên subsample 400 mẫu:

| C | gamma | Val F1 (mini) |
|---|-------|:---:|
| 0.1 | scale | 0.3874 |
| 0.1 | auto | 0.3874 |
| 1 | scale | 0.9147 |
| 1 | auto | 0.3874 |
| **10** | **scale** | **0.9323** |
| 10 | auto | 0.3874 |

**Nhận xét về gamma='auto':** F1 = 0.3874 ≈ tỉ lệ lớp thiểu số → model predict toàn bộ về một class. Nguyên nhân: $\gamma = 1/5000 = 0.0002$ quá nhỏ, $K(\mathbf{x}_i, \mathbf{x}_j) = e^{-0.0002 \|\Delta\|^2} \approx 0$ với mọi cặp điểm → không phân biệt được.

---

## 4. Kết quả

### 4.1 Metric trên Validation Set

**Model:** LinearSVC, C=1, max_iter=2000, random_state=42

| Metric | REAL (0) | FAKE (1) | Weighted Avg |
|--------|:--------:|:--------:|:------------:|
| Precision | 0.99 | 0.99 | 0.99 |
| Recall | 0.99 | 0.98 | 0.99 |
| F1-score | **0.99** | **0.99** | **0.9869** |
| Support | 3,179 | 2,619 | 5,798 |

**Accuracy: 0.9869** (5,722 / 5,798 mẫu phân loại đúng)

### 4.2 Confusion Matrix

```
                  Predicted
               REAL    FAKE
Actual  REAL  [3149]   [30]   ← 30 tin thật bị nhận nhầm là tin giả
        FAKE  [ 46]  [2573]   ← 46 tin giả bị nhận nhầm là tin thật
```

**Phân tích lỗi:**
- **30 False Positives (REAL → FAKE):** Tin thật bị mô hình cảnh báo nhầm là tin giả.
- **46 False Negatives (FAKE → REAL):** Tin giả bị nhận nhầm là tin thật. Đây là dạng lỗi nguy hiểm hơn vì người đọc có thể tin vào tin giả.

### 4.3 So sánh LinearSVC vs SVC RBF

| | LinearSVC (full) | SVC RBF (sub 400) |
|--|:--:|:--:|
| Val F1 | **0.9869** | ~0.93 |
| Scalability | ✅ O(n·d) | ❌ O(n²·d) |
| Sparse support | ✅ | ❌ Cần dense |
| Nhạy với gamma | — | ⚠️ Rất cao |

**Kết luận:** LinearSVC vượt trội về cả hiệu năng lẫn tốc độ. Đây là lựa chọn chuẩn cho text classification với TF-IDF.

---

## 5. Phân tích Thêm

### 5.1 Tại sao LinearSVC không overfit dù 5000 features?

TF-IDF với `max_features=5000` tạo ra vector rất thưa — hầu hết entry bằng 0. LinearSVC với $C=1$ áp dụng regularization L2 ngầm, penalize các trọng số $w_i$ lớn. Kết hợp với kích thước train set lớn (27K mẫu), model có đủ dữ liệu để học ranh giới tốt.

### 5.2 Interpretability

LinearSVC có vector trọng số $\mathbf{w} \in \mathbb{R}^{5000}$. Với `classes_ = [0, 1]`, từ có $w_i > 0$ → hướng đến FAKE (class 1); $w_i < 0$ → hướng đến REAL (class 0). Đây là input cho Phase 6 (Feature Analysis).

### 5.3 Hướng cải thiện

- **Thử `max_features` lớn hơn** (10,000, 20,000) — nhiều đặc trưng hơn có thể giúp LinearSVC
- **Title + Text concatenation** — gộp tiêu đề vào nội dung trước khi vectorize
- **Phase 7:** Kiểm tra LinearSVC trên Dataset 2 để đánh giá cross-domain generalization

---

## 6. Tóm tắt

| Mục | Nội dung |
|-----|----------|
| Thuật toán | LinearSVC (liblinear solver) |
| Best params | C=1, max_iter=2000, random_state=42 |
| GridSearch CV F1 | 0.9852 (5-fold, f1_weighted) |
| Val Accuracy | **0.9869** |
| Val F1 (weighted) | **0.9869** |
| Training time | ~0.62s |
| Model file | `models/svm_model.pkl` (39.8 KB) |
| Biểu đồ | `reports/svm_linearsvc_c_tuning.png` |
| | `reports/svm_kernel_comparison.png` |
| | `reports/svm_confusion_matrix.png` |
