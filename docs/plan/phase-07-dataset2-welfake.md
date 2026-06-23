# Phase 7 — Dataset 2 Integration: WELFake

> **Trạng thái tổng thể:** 🔄 Đang thực hiện  
> **Tiến độ:** 7.1–7.3 hoàn thành; 7.4 hoàn thành 1/4 model
> **Cập nhật:** 2026-06-21

---

## Mục tiêu

Tích hợp WELFake thành pipeline độc lập song song với ISOT:

```text
KaggleHub
  → Preprocessing
  → EDA
  → TF-IDF riêng
  → Train 4 model trên WELFake
  → Chuẩn bị Phase 8 cross-dataset evaluation
```

Phase 7 chỉ sử dụng train/validation để chọn model. WELFake test set được giữ lại cho đánh giá cuối.

---

## Dataset và quy ước nhãn

| Thuộc tính | Giá trị |
|------------|---------|
| Dataset | WELFake |
| Kaggle handle | `saurabhshahane/fake-news-classification` |
| File nguồn | `WELFake_Dataset.csv` |
| Dữ liệu gốc | 72,134 rows |
| Sau preprocessing | 72,074 rows |
| Cột nguồn | `Unnamed: 0`, `title`, `text`, `label` |
| Nhãn nguồn | `0=FAKE`, `1=REAL` |
| Nhãn project | `0=REAL`, `1=FAKE` |

Notebook 10 tải dữ liệu tự động bằng `kagglehub`. Nhãn nguồn được lưu tạm trong `source_label`, sau đó map:

```python
df["label"] = df["source_label"].map({0: 1, 1: 0})
```

Phân phối sau xử lý:

| Nhãn | Số mẫu | Tỉ lệ |
|------|-------:|------:|
| REAL (0) | 37,046 | 51.4% |
| FAKE (1) | 35,028 | 48.6% |

---

## Cấu trúc sub-phase chính xác

| Sub-phase | Nội dung | Notebook/Plan | Trạng thái |
|-----------|----------|---------------|------------|
| 7.1 | Preprocessing | `10_welfake_preprocessing.ipynb` / `phase-07.1-preprocessing.md` | ✅ |
| 7.2 | EDA | `11_welfake_eda.ipynb` / `phase-07.2-eda.md` | ✅ |
| 7.3 | TF-IDF Vectorization | `12_welfake_vectorization.ipynb` / `phase-07.3-vectorization.md` | ✅ |
| 7.4 | Model Training WELFake | `phase-07.4-model-training-welfake.md` | 🔄 3/4 |
| 7.4.1 | LinearSVC | `13_welfake_svm.ipynb` / `phase-07.4.1-svm-welfake.md` | ✅ |
| 7.4.2 | Naive Bayes | `14_welfake_naive_bayes.ipynb` | ⏳ |
| 7.4.3 | Logistic Regression | `15_welfake_logistic_regression.ipynb` | ✅ |
| 7.4.4 | Random Forest | `16_welfake_random_forest.ipynb` | ✅ |

Không dùng tên `13_welfake_model_training.ipynb`: notebook số 13 hiện đã được triển khai riêng cho SVM.

---

## Kết quả từng sub-phase

### 7.1 — Preprocessing ✅

- Tải WELFake bằng KaggleHub.
- Drop dòng thiếu cả title và text; fill null đơn lẻ.
- Gộp `title + text`.
- Dùng duy nhất `preprocess_text()` từ `src/preprocessing.py`.
- Chuẩn hóa nhãn về `REAL=0`, `FAKE=1`.
- Output: `data/processed/preprocessed_welfake_full.csv`.

### 7.2 — EDA ✅

- Phân phối nhãn, độ dài text, word cloud/top n-grams.
- So sánh ISOT và WELFake để mô tả domain shift.
- Charts lưu tại `reports/welfake_*.png`.

### 7.3 — TF-IDF ✅

| Thiết lập | Giá trị |
|-----------|---------|
| Split | 70/15/15 stratified |
| random_state | 42 |
| max_features | 5000 |
| ngram_range | (1,2) |

| Split | Shape |
|-------|-------|
| Train | (50,451, 5,000) |
| Validation | (10,811, 5,000) |
| Test | (10,812, 5,000) |

Vectorizer chỉ fit trên train; validation/test chỉ dùng `transform()`.

### 7.4.1 — SVM ✅

Notebook `13_welfake_svm.ipynb` đã chạy thành công ngày 2026-06-21.

| Kết quả | Giá trị |
|---------|--------:|
| Best C | 1 |
| Best CV F1 weighted | 0.9441 |
| Validation Accuracy | 0.9451 |
| Validation F1 weighted | 0.9451 |
| Validation F1 macro | 0.9451 |
| GridSearch time (lần chạy cuối) | 21.0s |

Confusion matrix validation:

| | Pred REAL | Pred FAKE |
|---|---:|---:|
| Actual REAL | 5,302 | 255 |
| Actual FAKE | 338 | 4,916 |

RBF chỉ là thí nghiệm phụ trên subsample 400:

- Best: `C=10`, `gamma='scale'`
- F1 weighted: `0.8424`
- Không so sánh trực tiếp như một đánh giá full-dataset.

Artifacts:

- `models/svm_welfake_model.pkl`
- `reports/welfake_svm_c_tuning.png`
- `reports/welfake_svm_kernel_comparison.png`
- `reports/welfake_svm_confusion_matrix.png`

---

## Output structure

```text
data/
├── processed/
│   └── preprocessed_welfake_full.csv
└── welfake/
    ├── X_train_tfidf.pkl
    ├── X_val_tfidf.pkl
    ├── X_test_tfidf.pkl
    ├── y_train.pkl
    ├── y_val.pkl
    └── y_test.pkl

models/
├── tfidf_vectorizer_welfake.pkl
├── svm_welfake_model.pkl
├── nb_welfake_model.pkl       ← pending
├── lr_welfake_model.pkl       ← pending
└── rf_welfake_model.pkl       ← pending
```

---

## Điều kiện hoàn thành Phase 7

Phase 7 chỉ được đánh dấu hoàn thành khi:

- [x] Preprocessing WELFake hoàn tất.
- [x] EDA WELFake hoàn tất.
- [x] TF-IDF splits và vectorizer WELFake được lưu.
- [x] SVM WELFake được train, đánh giá và lưu.
- [ ] Naive Bayes WELFake được train, đánh giá và lưu.
- [ ] Logistic Regression WELFake được train, đánh giá và lưu.
- [ ] Random Forest WELFake được train, đánh giá và lưu.
- [ ] CHANGELOG ghi best params và validation F1 của đủ 4 model.

---

## Nguyên tắc kỹ thuật

1. `REAL=0`, `FAKE=1` trong mọi artifact của project.
2. Không dùng WELFake test set để tune model.
3. Không fit lại vectorizer trong notebook model training.
4. Mọi model dùng cùng WELFake TF-IDF feature space.
5. `random_state=42`; GridSearchCV dùng `cv=5`, metric chính `f1_weighted`.
6. Phase 8 phải dùng vectorizer của domain train để transform raw text của domain test; tuyệt đối không fit trên test domain.
