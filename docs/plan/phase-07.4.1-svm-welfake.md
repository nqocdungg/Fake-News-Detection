# Phase 7.4.1 — SVM trên WELFake

> **Trạng thái:** ✅ Hoàn thành
>
> **Notebook:** `notebooks/13_welfake_svm.ipynb`
>
> **Model:** `models/svm_welfake_model.pkl`
> **Chạy và kiểm chứng:** 2026-06-21

---

## Mục tiêu

1. Train baseline LinearSVC trên WELFake.
2. Tune `C` bằng GridSearchCV.
3. Đánh giá best LinearSVC trên validation set.
4. Thử SVC RBF trên subsample để đánh giá tính khả dụng.
5. Lưu model cho Phase 8.

Test set không được load hoặc sử dụng.

---

## Dữ liệu

| File | Shape |
|------|-------|
| `data/welfake/X_train_tfidf.pkl` | (50,451, 5,000) |
| `data/welfake/X_val_tfidf.pkl` | (10,811, 5,000) |
| `data/welfake/y_train.pkl` | (50,451,) |
| `data/welfake/y_val.pkl` | (10,811,) |

| Split | REAL (0) | FAKE (1) |
|-------|---------:|---------:|
| Train | 25,932 | 24,519 |
| Validation | 5,557 | 5,254 |

---

## Thiết kế thí nghiệm

### LinearSVC

```python
GridSearchCV(
    LinearSVC(max_iter=2000, random_state=42),
    {"C": [0.01, 0.1, 1, 10]},
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
)
```

### SVC RBF phụ trợ

- Stratified train subsample: 400.
- Stratified validation subsample: 400.
- `C ∈ [0.1, 1, 10]`.
- `gamma ∈ ['scale', 'auto']`.
- Kết quả này chỉ dùng để đánh giá tính khả dụng, không phải full-dataset benchmark.

---

## Kết quả thực tế

### Baseline

| Metric | Giá trị |
|--------|--------:|
| C | 1 |
| Training time (lần chạy cuối) | 2.34s |
| Val Accuracy | 0.9451 |
| Val F1 weighted | 0.9451 |

### GridSearchCV

| C | CV F1 mean | CV F1 std |
|---:|-----------:|----------:|
| 0.01 | 0.9095 | 0.0025 |
| 0.1 | 0.9395 | 0.0023 |
| **1** | **0.9441** | **0.0025** |
| 10 | 0.9358 | 0.0019 |

- Best params: `C=1`.
- GridSearch time ở lần chạy kiểm chứng cuối: 21.0 giây.
- C=10 giảm F1, cho thấy regularization yếu hơn làm giảm generalization.

### Validation chính thức

| Metric | Giá trị |
|--------|--------:|
| Accuracy | 0.945148 |
| F1 weighted | 0.945133 |
| F1 macro | 0.945078 |

Classification report:

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| REAL (0) | 0.94 | 0.95 | 0.95 | 5,557 |
| FAKE (1) | 0.95 | 0.94 | 0.94 | 5,254 |

Confusion matrix:

| | Pred REAL | Pred FAKE |
|---|---:|---:|
| Actual REAL | 5,302 | 255 |
| Actual FAKE | 338 | 4,916 |

Số tin giả bị nhận nhầm là thật (338) cao hơn số tin thật bị cảnh báo nhầm là giả (255). Đây là điểm cần phân tích ở Phase 8/9.

### SVC RBF trên subsample

| C | gamma | F1 weighted |
|---:|-------|------------:|
| 0.1 | scale | 0.3501 |
| 0.1 | auto | 0.3501 |
| 1 | scale | 0.8347 |
| 1 | auto | 0.3501 |
| **10** | **scale** | **0.8424** |
| 10 | auto | 0.3501 |

`gamma='auto'=1/5000` quá nhỏ, làm kernel RBF gần 1 với phần lớn cặp điểm; các mẫu khó phân biệt và model suy biến về gần một lớp.

Không so sánh trực tiếp F1 `0.8424` với LinearSVC `0.9451` như hai benchmark ngang hàng vì:

- RBF chỉ train trên 400 mẫu.
- LinearSVC train trên 50,451 mẫu.
- Hai kết quả có quy mô train khác nhau.

Kết quả vẫn đủ để kết luận RBF không phù hợp với yêu cầu scalability hiện tại.

---

## Đánh giá chất lượng triển khai

| Kiểm tra | Kết quả |
|----------|---------|
| Dùng đúng WELFake paths | ✅ |
| Không dùng test set | ✅ |
| Không fit lại vectorizer | ✅ |
| LinearSVC giữ sparse CSR | ✅ |
| GridSearchCV `cv=5`, `f1_weighted` | ✅ |
| `random_state=42` | ✅ |
| Nhãn `REAL=0`, `FAKE=1` | ✅ |
| Model lưu và load lại được | ✅ |
| Model `classes_=[0,1]` | ✅ |
| Metrics kiểm chứng độc lập khớp notebook | ✅ |
| Charts được tạo thành công | ✅ |

Notebook có thể xuất cảnh báo `joblib resource_tracker KeyError` sau quá trình parallel GridSearch trên Windows. Trong lần chạy kiểm chứng:

- `nbconvert` trả exit code 0.
- Mọi code cell thực thi đủ.
- GridSearch, model, metrics và charts đều được tạo.

Do đó đây là cảnh báo cleanup của worker, không phải lỗi kết quả. Nếu muốn output sạch hơn, có thể đổi `n_jobs=1` với đánh đổi runtime dài hơn.

---

## So sánh với SVM ISOT

| Metric | ISOT | WELFake |
|--------|-----:|--------:|
| Train samples | 27,057 | 50,451 |
| Best C | 1 | 1 |
| CV F1 | 0.9852 | 0.9441 |
| Val F1 | 0.9869 | 0.9451 |
| Chênh lệch Val F1 | — | -0.0418 |

WELFake có nhiều mẫu hơn nhưng F1 thấp hơn, cho thấy dữ liệu đa nguồn/nhiễu hơn và khó phân loại hơn ISOT. Đây là kết quả hợp lý, không phải dấu hiệu notebook chạy sai.

---

## Artifacts

| File | Trạng thái |
|------|------------|
| `notebooks/13_welfake_svm.ipynb` | ✅ Executed |
| `models/svm_welfake_model.pkl` | ✅ |
| `reports/welfake_svm_c_tuning.png` | ✅ |
| `reports/welfake_svm_kernel_comparison.png` | ✅ |
| `reports/welfake_svm_confusion_matrix.png` | ✅ |

---

## Kết luận

`LinearSVC(C=1)` là model SVM được chọn cho WELFake. Notebook đã chạy đúng và đủ điều kiện làm input cho Phase 8 sau khi các model WELFake còn lại hoàn thành.
