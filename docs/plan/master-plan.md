# Master Plan — Fake News Detection

> **Học phần:** Nhập môn Trí tuệ nhân tạo  
> **Loại bài toán:** Phân loại văn bản nhị phân (FAKE / REAL)  
> **Cập nhật lần cuối:** 2026-06-17

---

## Tổng quan kiến trúc

```
Thu thập dữ liệu (2 datasets)
  → Tiền xử lý NLP
  → TF-IDF Feature Extraction
  → Huấn luyện 4 mô hình ML
  → Đánh giá trong cùng dataset
  → Cross-Dataset Evaluation (Generalization)
  → So sánh & Phân tích
  → Báo cáo
```

---

## Trạng thái tổng quan

| Phase | Tên | Trạng thái |
|-------|-----|------------|
| 0 | Project Setup | ✅ Hoàn thành |
| 1 | Data & Preprocessing (ISOT) | ✅ Hoàn thành |
| 2 | Exploratory Data Analysis | ✅ Hoàn thành |
| 3 | TF-IDF Vectorization | ✅ Hoàn thành |
| 4 | Model Training (4 thuật toán) | 🔄 Đang thực hiện |
| 5 | Model Comparison (same-dataset) | ⏳ Chưa bắt đầu |
| 6 | Feature & Error Analysis | ⏳ Chưa bắt đầu |
| 7 | Dataset 2 Integration (WELFake) | 🔄 Đang thực hiện |
| 8 | Cross-Dataset Evaluation | ⏳ Chưa bắt đầu |
| 9 | Report & Documentation | ⏳ Chưa bắt đầu |

---

## Chi tiết từng Phase

---

### Phase 0 — Project Setup
**Trạng thái:** ✅ Hoàn thành  
**File kế hoạch:** —

**Deliverables đã có:**
- `README.md` — hướng dẫn cài đặt & chạy
- `requirements.txt` — pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud, joblib, jupyter, ipykernel
- `.gitignore`
- Cấu trúc thư mục: `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `models/`, `reports/`, `docs/`

---

### Phase 1 — Data & Preprocessing (ISOT)
**Trạng thái:** ✅ Hoàn thành  
**File kế hoạch:** `docs/plan/phase-1-preprocessing.md`

**Dataset:** ISOT Fake News Dataset (~21K real, ~23K fake, tiếng Anh)  
**Nguồn:** https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset

**Pipeline NLP:**
1. Chuyển về chữ thường
2. Loại bỏ URL, ký tự đặc biệt, dấu câu
3. Fix Reuters leakage (`re.sub(r'\(Reuters\)|\bReuters\b', '', text)`)
4. Giữ lại negation words (no, not, never, neither) trước khi xóa stopwords
5. Tokenization
6. Stopword Removal
7. Lemmatization

**Deliverables đã có:**
- `src/preprocessing.py` — hàm `preprocess_text(text)` tái sử dụng
- `notebooks/01_preprocessing.ipynb`
- `data/processed/preprocessed_isot_full.csv`

---

### Phase 2 — Exploratory Data Analysis
**Trạng thái:** ✅ Hoàn thành  
**File kế hoạch:** `docs/plan/phase-2-eda.md`

**Nội dung EDA:**
- Phân phối nhãn fake/real (pie chart + bar)
- Histogram độ dài text (theo từ) của fake vs real
- Word cloud fake và real
- Top 20 unigrams và bigrams mỗi class
- Kiểm tra null, duplicate, sample text

**Deliverables đã có:**
- `notebooks/02_eda.ipynb`
- Biểu đồ xuất vào `reports/`

---

### Phase 3 — TF-IDF Vectorization
**Trạng thái:** ✅ Hoàn thành  
**File kế hoạch:** `docs/plan/phase-3-vectorization.md`

**Chi tiết:**
- Split: 70% train / 15% val / 15% test (stratified)
- Fit vectorizer **chỉ trên train set** (tránh data leakage)
- `max_features`: thử 3000 / 5000 / 10000
- `ngram_range`: thử (1,1) / (1,2) — bigram bắt ngữ cảnh cụm từ
- Lưu vectorizer để tái sử dụng

**Deliverables đã có:**
- `notebooks/03_vectorization.ipynb`
- `data/X_train_tfidf.pkl`, `data/X_val_tfidf.pkl`, `data/X_test_tfidf.pkl`
- `data/y_train.pkl`, `data/y_val.pkl`, `data/y_test.pkl`
- `models/tfidf_vectorizer.pkl` *(cần xác nhận)*

---

### Phase 4 — Model Training
**Trạng thái:** 🔄 Đang thực hiện
**File kế hoạch:** `docs/plan/phase-4-model-training.md`

Mỗi model dùng chung flow: load TF-IDF splits → GridSearchCV (cv=5) → best params → predict val → classification report + confusion matrix + training time → lưu `.pkl`.

| Model | Notebook | Output | Trạng thái |
|-------|----------|--------|------------|
| Naive Bayes | `04_naive_bayes.ipynb` | `models/naive_bayes_model.pkl` | 🔄 |
| Logistic Regression | `05_logistic_regression.ipynb` | `models/lr_model.pkl` | 🔄 |
| SVM | `06_svm.ipynb` | `models/svm_model.pkl` | 🔄 |
| Random Forest | `07_random_forest.ipynb` | `models/rf_model.pkl` | 🔄 |

**Hyperparameter search:**

- **Naive Bayes:** `alpha` ∈ [0.01, 0.1, 0.5, 1.0, 2.0], `fit_prior` ∈ [True, False]
- **Logistic Regression:** `C` ∈ [0.1, 0.5, 1.0, 2.0, 5.0], `solver` ∈ ['lbfgs', 'saga'], `max_iter` ∈ [500, 1000]
- **SVM:** `LinearSVC` (C ∈ [0.01, 0.1, 1, 10]) vs `SVC(kernel='rbf')` (C ∈ [0.1, 1, 10], gamma ∈ ['scale', 'auto'])
- **Random Forest:** `n_estimators` ∈ [100, 200, 300], `max_depth` ∈ [None, 10, 20], `min_samples_split` ∈ [2, 5]

---

### Phase 5 — Model Comparison (Same-Dataset)
**Trạng thái:** ⏳ Chưa bắt đầu  
**File kế hoạch:** `docs/plan/phase-5-comparison.md`

**Thí nghiệm 1:** Train ISOT → Test ISOT (dùng test set lần đầu tiên)

**Nội dung:**
- Load cả 4 model đã lưu
- Predict trên test set ISOT
- Bảng tổng hợp: Accuracy / Precision / Recall / F1 / Training time
- Bar chart so sánh F1 của 4 model
- Nhận xét: model nào tốt nhất, đánh đổi accuracy vs speed

**Deliverables:**
- `notebooks/08_comparison.ipynb`
- Bảng metrics xuất vào `reports/`

---

### Phase 6 — Feature & Error Analysis
**Trạng thái:** ⏳ Chưa bắt đầu  
**File kế hoạch:** `docs/plan/phase-6-analysis.md`

**Nội dung:**
- **Feature analysis:** Top 20 từ quan trọng nhất cho FAKE và REAL (LR coefficients hoặc NB log-probabilities), vẽ bar chart horizontal
- **Error analysis:** 10-15 sample bị predict sai của model tốt nhất, nhận xét pattern (bài quá ngắn? ngôn ngữ trung lập?)

**Deliverables:**
- `notebooks/09_feature_error_analysis.ipynb`

---

### Phase 7 — Dataset 2 Integration *(MỚI)*
**Trạng thái:** 🔄 Đang thực hiện

**File kế hoạch:** `docs/plan/phase-07-dataset2-welfake.md`

**Mục đích:** Đánh giá tính tổng quát (Generalization) của mô hình khi gặp dữ liệu lạ.

**Dataset 2 đang cân nhắc:**
- **LIAR Dataset** — câu tuyên bố chính trị có độ nhiễu cao, 6 nhãn (cần binary mapping)
- **WELFake** — ~72K bài, gộp từ nhiều nguồn, gần với ISOT hơn về format

> **Khuyến nghị:** WELFake vì có format tiêu đề + nội dung tương đồng ISOT, dễ áp dụng cùng pipeline hơn.

**Pipeline áp dụng lại:**
1. Tự động tải WELFake bằng KaggleHub
2. Chuẩn hóa nhãn nguồn về `REAL=0`, `FAKE=1` giống ISOT
3. Áp dụng `preprocess_text()` từ `src/preprocessing.py`
4. Tạo TF-IDF riêng cho WELFake (fit trên train set)
5. Split 70/15/15 tương tự Phase 3

**Deliverables:**
- `notebooks/10_welfake_preprocessing.ipynb`
- `notebooks/11_welfake_eda.ipynb`
- `notebooks/12_welfake_vectorization.ipynb`
- `data/processed/preprocessed_welfake_full.csv`
- `data/welfake/X_train_tfidf.pkl`, `X_val_tfidf.pkl`, `X_test_tfidf.pkl`
- `data/welfake/y_train.pkl`, `y_val.pkl`, `y_test.pkl`
- `models/tfidf_vectorizer_welfake.pkl`

---

### Phase 8 — Cross-Dataset Evaluation *(MỚI)*
**Trạng thái:** ⏳ Chưa bắt đầu  
**File kế hoạch:** `docs/plan/phase-8-cross-eval.md`

**Thí nghiệm 2:** Cross-Dataset Evaluation

| Scenario | Train | Test |
|----------|-------|------|
| ISOT → D2 | ISOT | Dataset 2 |
| D2 → ISOT | Dataset 2 | ISOT |

**Lưu ý kỹ thuật:** Khi cross-test, cần dùng vectorizer của tập train (không re-fit). Apply `transform()` (không `fit_transform()`) trên test set đến từ dataset kia.

**Nội dung phân tích:**
- So sánh performance in-distribution vs out-of-distribution
- Nhận xét sự sụt giảm F1 khi cross-test (domain shift)
- Model nào robust nhất trước domain shift?

**Deliverables:**
- `notebooks/13_cross_dataset_eval.ipynb`
- Bảng so sánh 2x2 (4 models × 2 scenarios)

---

### Phase 9 — Report & Documentation
**Trạng thái:** ⏳ Chưa bắt đầu  
**File kế hoạch:** `docs/plan/phase-9-report.md`

**Cấu trúc báo cáo:**
1. Tổng quan (Bối cảnh, Mục tiêu, Ý nghĩa)
2. Phương pháp (Datasets, Preprocessing, TF-IDF, 4 thuật toán)
3. Kết quả (Thí nghiệm 1 + Thí nghiệm 2 + Feature/Error analysis)
4. Kết luận & Hướng phát triển
5. Hướng dẫn chạy hệ thống

**Tech:** Python / NLTK / Scikit-learn / Matplotlib / Seaborn

**Deliverables:**
- Báo cáo PDF hoàn chỉnh trong `reports/`
- `README.md` cập nhật (thêm Dataset 2, thứ tự notebook mới)

---

## Phân công nhóm

| Thành viên | Phase chính |
|------------|-------------|
| Hưng | Phase 1 (Preprocessing), Phase 4 (Naive Bayes), Phase 5+6 |
| Xuân | Phase 2 (EDA), Phase 4 (Logistic Regression), Phase 5+6 |
| Dung | Phase 3 (Vectorization), Phase 4 (SVM), Phase 7 |
| Thủy | Phase 0 (Setup), Phase 4 (Random Forest), Phase 8, Phase 9 |

---

## Dependency Graph

```
Phase 0 ──► Phase 1 ──► Phase 2
                  └──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
                                                              │
                                         Phase 7 ──► Phase 8─┘
                                                              │
                                                         Phase 9
```

---

## Ghi chú kỹ thuật

- **Data leakage prevention:** Vectorizer luôn được `fit` chỉ trên train set, dùng `transform` cho val/test.
- **Reproducibility:** Set `random_state=42` cho tất cả model và split.
- **Model persistence:** Dùng `joblib.dump` / `joblib.load`.
- **Metrics chính:** F1-score (vì dataset có thể imbalanced); báo cáo đủ Accuracy, Precision, Recall, F1, Confusion Matrix.
