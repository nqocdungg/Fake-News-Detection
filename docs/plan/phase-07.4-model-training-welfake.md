# Phase 7.4 — Model Training trên WELFake

> **Trạng thái tổng thể:** 🔄 Đang thực hiện — 2/4 model hoàn thành
> **Cập nhật:** 2026-06-22

---

## Mục tiêu

Train cùng bốn thuật toán đã dùng trên ISOT bằng WELFake TF-IDF splits:

1. LinearSVC
2. Naive Bayes
3. Logistic Regression
4. Random Forest

Phase này chỉ dùng train/validation. Test set được giữ nguyên cho Phase 8.

---

## Cấu trúc triển khai

Do mỗi thuật toán có grid, thời gian chạy và phân tích riêng, Phase 7.4 được tách thành notebook theo model:

| Sub-phase | Model | Notebook | Trạng thái |
|-----------|-------|----------|------------|
| 7.4.1 | LinearSVC | `13_welfake_svm.ipynb` | ✅ Hoàn thành |
| 7.4.2 | Multinomial Naive Bayes | `14_welfake_naive_bayes.ipynb` | ⏳ Chưa tạo |
| 7.4.3 | Logistic Regression | `15_welfake_logistic_regression.ipynb` | ✅ Hoàn thành |
| 7.4.4 | Random Forest | `16_welfake_random_forest.ipynb` | ⏳ Chưa tạo |

Plan cũ dùng một notebook `13_welfake_model_training.ipynb` cho cả bốn model không còn phù hợp với triển khai thực tế và được thay bằng cấu trúc trên.

---

## Dữ liệu đầu vào dùng chung

| File | Shape | Mô tả |
|------|-------|-------|
| `data/welfake/X_train_tfidf.pkl` | (50,451, 5,000) | Train sparse CSR |
| `data/welfake/X_val_tfidf.pkl` | (10,811, 5,000) | Validation sparse CSR |
| `data/welfake/y_train.pkl` | (50,451,) | `0=REAL`, `1=FAKE` |
| `data/welfake/y_val.pkl` | (10,811,) | `0=REAL`, `1=FAKE` |

Không load `X_test_tfidf.pkl` hoặc `y_test.pkl` trong các notebook 7.4.x.

Phân phối train:

| Nhãn | Số mẫu | Tỉ lệ |
|------|-------:|------:|
| REAL (0) | 25,932 | 51.4% |
| FAKE (1) | 24,519 | 48.6% |

---

## Cấu hình và trạng thái từng model

### 7.4.1 — LinearSVC ✅

Plan chi tiết: `phase-07.4.1-svm-welfake.md`.

| Thuộc tính | Kết quả |
|------------|--------:|
| Grid C | [0.01, 0.1, 1, 10] |
| Best C | 1 |
| CV F1 weighted | 0.9441 |
| Val Accuracy | 0.9451 |
| Val F1 weighted | 0.9451 |
| Model | `models/svm_welfake_model.pkl` |

### 7.4.2 — Naive Bayes ⏳

Notebook dự kiến: `14_welfake_naive_bayes.ipynb`.

```python
param_grid = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
    "fit_prior": [True, False],
}
```

- `GridSearchCV(cv=5, scoring="f1_weighted")`
- Output: `models/nb_welfake_model.pkl`
- Báo cáo Accuracy, Precision, Recall, F1 weighted/macro và confusion matrix trên validation.

### 7.4.3 — Logistic Regression ⏳

Notebook dự kiến: `15_welfake_logistic_regression.ipynb`.

Grid phải dùng các tổ hợp solver/penalty hợp lệ. `saga` hỗ trợ `l1` và `l2`; `liblinear` cũng hỗ trợ binary `l1/l2`.

```python
param_grid = [
    {
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"],
        "C": [0.1, 1, 5, 10, 20],
        "class_weight": [None, "balanced"],
        "max_iter": [500],
    },
    {
        "solver": ["saga"],
        "penalty": ["l1", "l2"],
        "C": [0.1, 1, 5, 10, 20],
        "class_weight": [None, "balanced"],
        "max_iter": [500],
    },
]
```

- Output: `models/lr_welfake_model.pkl`
- Ghi convergence warnings nếu có; không im lặng coi model chưa hội tụ là hợp lệ.

### 7.4.4 — Random Forest ⏳

Notebook dự kiến: `16_welfake_random_forest.ipynb`.

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}
```

- 12 cấu hình × 5 folds = 60 fits.
- Có thể rút gọn grid nếu runtime không phù hợp, nhưng phải ghi rõ thay đổi trong notebook và CHANGELOG.
- Output: `models/rf_welfake_model.pkl`.

---

## Quy trình bắt buộc cho mỗi notebook

1. Load đúng `data/welfake/`.
2. Xác minh shape và nhãn `{0,1}`.
3. Train baseline.
4. GridSearchCV trên train set.
5. Chọn best estimator bằng CV.
6. Đánh giá một lần trên validation set.
7. Báo cáo:
   - Accuracy
   - Precision/Recall/F1 per class
   - F1 weighted và macro
   - Confusion matrix
   - Training/GridSearch time
8. Lưu model bằng `joblib.dump`.
9. Không đọc hoặc predict test set.

---

## Tổng hợp validation

| Model | Best Params | CV F1 | Val Accuracy | Val F1 weighted | Trạng thái |
|-------|-------------|:-----:|:------------:|:---------------:|------------|
| LinearSVC | `C=1` | 0.9441 | 0.9451 | 0.9451 | ✅ |
| Naive Bayes | — | — | — | — | ⏳ |
| Logistic Regression | `C=5.0`, `solver='saga'`, `max_iter=500` (penalty='l2' mặc định) | 0.9447 | 0.9459 | 0.9459 | ✅ |
| Random Forest | — | — | — | — | ⏳ |

Không chọn “model tốt nhất cuối cùng” bằng test set trong Phase 7.4. Bảng này chỉ là validation summary.

---

## Output

```text
models/
├── svm_welfake_model.pkl   ✅
├── nb_welfake_model.pkl    ⏳
├── lr_welfake_model.pkl    ✅
└── rf_welfake_model.pkl    ⏳
```

Phase 7.4 hoàn thành khi đủ bốn model, đủ validation metrics và CHANGELOG đã ghi best params/F1 cho từng model.

---

## Lưu ý cho Phase 8

- WELFake model chỉ nhận vector từ `tfidf_vectorizer_welfake.pkl`.
- Khi WELFake → ISOT, phải reconstruct đúng ISOT test raw texts rồi gọi `tfidf_vectorizer_welfake.transform()`.
- Không được dùng sẵn `data/X_test_tfidf.pkl` của ISOT cho WELFake model vì matrix đó thuộc vocabulary ISOT.
