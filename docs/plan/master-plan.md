# Master Plan — Fake News Detection

> **Học phần:** Nhập môn Trí tuệ nhân tạo  
> **Loại bài toán:** Phân loại văn bản nhị phân — **0 = REAL, 1 = FAKE** (nhất quán toàn dự án)  
> **Cập nhật lần cuối:** 2026-06-21

---

## Tổng quan kiến trúc

```
Thu thập dữ liệu (ISOT + WELFake)
  → Tiền xử lý NLP  →  TF-IDF Feature Extraction
  → Huấn luyện 4 mô hình ML (ISOT)
  → Đánh giá trong cùng dataset (Thí nghiệm 1)
  → Huấn luyện 4 mô hình ML (WELFake)
  → Cross-Dataset Evaluation / Generalization (Thí nghiệm 2)
  → So sánh & Phân tích  →  Báo cáo
```

---

## Quy ước nhãn (QUAN TRỌNG)

| Nhãn số | Ý nghĩa | Áp dụng cho |
|---------|---------|-------------|
| **0** | **REAL** (tin thật) | ISOT & WELFake |
| **1** | **FAKE** (tin giả) | ISOT & WELFake |

`notebooks/06_svm.ipynb` đã được đồng bộ theo quy ước `REAL=0`, `FAKE=1`, bao gồm classification report và confusion matrix.

---

## Trạng thái tổng quan

| Phase | Tên | Trạng thái |
|-------|-----|------------|
| 0 | Project Setup | ✅ Hoàn thành |
| 1 | Preprocessing ISOT | ✅ Hoàn thành |
| 2 | EDA ISOT | ✅ Hoàn thành |
| 3 | TF-IDF Vectorization ISOT | ✅ Hoàn thành |
| 4 | Model Training ISOT (4 thuật toán) | ✅ Hoàn thành |
| 5 | Model Comparison ISOT (same-dataset) | ⏳ Chờ Phase 4 hoàn thành |
| 6 | Feature & Error Analysis | ⏳ Chờ Phase 5 |
| 7 | Dataset 2 Integration (WELFake) | 🔄 7.1–7.3 + SVM xong; 3 model còn thiếu |
| 8 | Cross-Dataset Evaluation | ⏳ Chờ Phase 4 RF + Phase 7.4 đủ 4 model |
| 9 | Report & Documentation | ⏳ Chờ tất cả |

---

## Chi tiết từng Phase

---

### Phase 0 — Project Setup
**Trạng thái:** ✅ Hoàn thành

**Deliverables:**
- `README.md`, `requirements.txt`, `.gitignore`, `AGENTS.md`
- Cấu trúc thư mục: `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `docs/`

---

### Phase 1 — Preprocessing ISOT
**Trạng thái:** ✅ Hoàn thành  
**Notebook:** `01_preprocessing.ipynb`

**Dataset:** ISOT Fake News Dataset  
**Số mẫu sau xử lý:** 38,653 (REAL=21,196 / FAKE=17,457)  
**Nguồn:** KaggleHub `rahulogoel/isot-fake-news-dataset`

**Pipeline NLP** (trong `src/preprocessing.py`):
1. Fix Reuters leakage → lowercase → strip URL/email/digits/punct → tokenize → lemmatize → remove stopwords (giữ negation words)

**Output:**
- `src/preprocessing.py` — `preprocess_text(text)`
- `data/processed/preprocessed_isot_full.csv` (38,653 rows)

---

### Phase 2 — EDA ISOT
**Trạng thái:** ✅ Hoàn thành  
**Notebook:** `02_eda.ipynb`

**Output:** Biểu đồ phân phối nhãn, độ dài text, word cloud, top n-grams trong `reports/`

---

### Phase 3 — TF-IDF Vectorization ISOT
**Trạng thái:** ✅ Hoàn thành  
**Notebook:** `03_vectorization.ipynb`

**Settings chọn cuối:** `max_features=5000`, `ngram_range=(1,2)`  
**Split:** 70/15/15, `stratify=y`, `random_state=42`

**Kết quả splits:**

| Split | Samples | Shape |
|-------|---------|-------|
| Train | 27,057 | (27057, 5000) |
| Val | 5,798 | (5798, 5000) |
| Test | 5,798 | (5798, 5000) |

**Output:**
- `data/X_train_tfidf.pkl`, `X_val_tfidf.pkl`, `X_test_tfidf.pkl`
- `data/y_train.pkl`, `y_val.pkl`, `y_test.pkl`
- `models/tfidf_vectorizer.pkl` — ⚠️ hiện chưa có trong workspace, cần chạy lại/lưu từ notebook 03 trước Phase 8

---

### Phase 4 — Model Training ISOT
**Trạng thái:** ✅ Hoàn thành

**Kết quả thực nghiệm (Validation Set ISOT):**

| Model | Best Params | Val Accuracy | Val F1 (weighted) | Notebook | Status |
|-------|-------------|:---:|:---:|----------|--------|
| Naive Bayes | `alpha=0.01, fit_prior=True` | 0.9415 | 0.9416 | `04_naive_bayes.ipynb` | ✅ |
| Logistic Regression | `C=20.0, penalty=l2, solver=saga, balanced` | ~0.99 | ~0.98 | `05_logistic_regression.ipynb` | ✅ |
| SVM (LinearSVC) | `C=1, max_iter=2000` | 0.9869 | 0.9869 | `06_svm.ipynb` | ✅ |
| Random Forest | `max_depth=None, min_samples_split=5, n_estimators=300` | 0.98 | 0.98 | `07_random_forest.ipynb` | ✅ |

**SVM notebook:** nhãn hiển thị đã được sửa thành `REAL(0)/FAKE(1)`.

---

### Phase 5 — Model Comparison ISOT (Same-Dataset)
**Trạng thái:** ⏳ Chờ Phase 4 hoàn thành (RF)  
**Notebook:** `08_comparison.ipynb` *(có markdown, chưa có code)*

**Nội dung:**
- Load 4 models, predict trên **test set ISOT** (lần đầu tiên dùng test set)
- Bảng: Accuracy / Precision / Recall / F1 (weighted) / Training time
- Bar chart so sánh F1

---

### Phase 6 — Feature & Error Analysis
**Trạng thái:** ⏳ Chờ Phase 5  
**Notebook:** `09_feature_error_analysis.ipynb` *(có markdown, chưa có code)*

**Nội dung:**
- Top 20 từ quan trọng FAKE vs REAL (LR coefficients)
- 10–15 sample bị predict sai của model tốt nhất

---

### Phase 7 — Dataset 2 Integration: WELFake
**Trạng thái:** 🔄 7.1–7.3 ✅ / 7.4 hoàn thành 1/4 model
**Plan tổng thể:** `docs/plan/phase-07-dataset2-welfake.md`

#### Sub-phase 7.1 — Preprocessing WELFake
**Trạng thái:** ✅ Hoàn thành  
**Notebook:** `10_welfake_preprocessing.ipynb`  
**Plan:** `docs/plan/phase-07.1-preprocessing.md`

**Dataset:** WELFake — ~72,134 rows (tải qua KaggleHub)  
**Sau xử lý:** 72,074 rows (60 dropped — empty sau preprocessing)  
**Label mapping:** WELFake gốc (0=FAKE, 1=REAL) → project (0=REAL, 1=FAKE)  
**Output:** `data/processed/preprocessed_welfake_full.csv` (155 MB)

| Nhãn | Sau xử lý |
|------|-----------|
| REAL (0) | 37,046 (51.4%) |
| FAKE (1) | 35,028 (48.6%) |

#### Sub-phase 7.2 — EDA WELFake
**Trạng thái:** ✅ Hoàn thành  
**Notebook:** `11_welfake_eda.ipynb`  
**Plan:** `docs/plan/phase-07.2-eda.md`

**Key findings:**
- Imbalance ratio: 1.058 (gần balanced hơn ISOT)
- FAKE mean words: 330 / REAL mean words: 283 (FAKE dài hơn một chút)
- Word cloud bỏ qua (wordcloud library chưa cài)
- Output charts: `welfake_label_dist.png`, `welfake_text_length_by_label.png`, `welfake_top_ngrams.png`, `welfake_isot_comparison.png`

**So sánh ISOT vs WELFake:**

| Metric | ISOT | WELFake |
|--------|------|---------|
| Tổng mẫu | 38,653 | 72,074 |
| FAKE | 17,457 | 35,028 |
| REAL | 21,196 | 37,046 |
| Tỉ lệ FAKE | 45.2% | 48.6% |
| Mean words | — | FAKE=330, REAL=283 |

#### Sub-phase 7.3 — TF-IDF Vectorization WELFake
**Trạng thái:** ✅ Hoàn thành  
**Notebook:** `12_welfake_vectorization.ipynb`  
**Plan:** `docs/plan/phase-07.3-vectorization.md`

**Settings:** `max_features=5000`, `ngram_range=(1,2)` — nhất quán với ISOT  
**Split:** 70/15/15, `stratify=y`, `random_state=42`

| Split | Samples | Shape |
|-------|---------|-------|
| Train | 50,451 | (50451, 5000) |
| Val | 10,811 | (10811, 5000) |
| Test | 10,812 | (10812, 5000) |

**Output:**
- `models/tfidf_vectorizer_welfake.pkl` (184 KB)
- `data/welfake/X_train_tfidf.pkl` (95 MB), `X_val_tfidf.pkl`, `X_test_tfidf.pkl`
- `data/welfake/y_train.pkl`, `y_val.pkl`, `y_test.pkl`

#### Sub-phase 7.4 — Model Training WELFake
**Trạng thái:** 🔄 Đang thực hiện — SVM và LR hoàn thành, NB/RF còn thiếu
**Plan:** `docs/plan/phase-07.4-model-training-welfake.md`

| Sub-phase | Model | Notebook | Val F1 weighted | Trạng thái |
|-----------|-------|----------|:---------------:|------------|
| 7.4.1 | LinearSVC | `13_welfake_svm.ipynb` | **0.9451** | ✅ |
| 7.4.2 | Naive Bayes | `14_welfake_naive_bayes.ipynb` | — | ⏳ |
| 7.4.3 | Logistic Regression | `15_welfake_logistic_regression.ipynb` | **0.9436** | ✅ |
| 7.4.4 | Random Forest | `16_welfake_random_forest.ipynb` | — | ⏳ |

**SVM WELFake đã xác minh:**
- Best `C=1`
- CV F1 weighted = `0.9441`
- Validation Accuracy/F1 weighted = `0.9451`
- Confusion matrix = `[[5302,255],[338,4916]]`
- Model: `models/svm_welfake_model.pkl`

**Logistic Regression WELFake đã xác minh:**
- Best `C=10.0`, `class_weight='balanced'`, `solver='saga'`, `penalty='l2'`
- CV F1 weighted = `0.9419`
- Validation Accuracy/F1 weighted = `0.9436`
- Confusion matrix = `[[5307,250],[325,4929]]`
- Model: `models/lr_welfake_model.pkl`

**Output Phase 7.4:**
- `models/nb_welfake_model.pkl`
- `models/lr_welfake_model.pkl`
- `models/svm_welfake_model.pkl` ✅
- `models/rf_welfake_model.pkl`

---

### Phase 8 — Cross-Dataset Evaluation
**Trạng thái:** ⏳ Chờ Phase 4 RF + Phase 7.4.2–7.4.4

**Notebook:** `17_cross_dataset_eval.ipynb` *(chưa tạo)*
**Plan:** `docs/plan/phase-08-cross-eval.md`

**Hai thí nghiệm:**

| Scenario | Train Set | Test Set | Vectorizer dùng |
|----------|-----------|----------|-----------------|
| ISOT → WELFake | ISOT | WELFake test | ISOT vectorizer `.transform()` |
| WELFake → ISOT | WELFake | ISOT test | WELFake vectorizer `.transform()` |

**Kỹ thuật quan trọng:** Không `fit_transform()` lại — chỉ `.transform()` với vectorizer của tập train nguồn.

**Output:** Bảng 4 model × 2 scenarios, nhận xét domain shift.

**Prerequisites cần xử lý trước khi bắt đầu:**
- Tạo lại `models/tfidf_vectorizer.pkl` của ISOT vì artifact này hiện chưa có trong workspace.
- Reconstruct chính xác raw text test split của ISOT và WELFake bằng cùng quy trình split (`70/15/15`, stratify, `random_state=42`) hoặc lưu text split riêng.
- Không đưa matrix `X_test_tfidf.pkl` của dataset A trực tiếp vào model của dataset B vì hai matrix thuộc hai vocabulary khác nhau.

---

### Phase 9 — Report & Documentation
**Trạng thái:** ⏳ Chờ tất cả  
**Plan:** `docs/plan/phase-09-report.md`

**Cấu trúc báo cáo:**
1. Tổng quan — Bối cảnh, Mục tiêu, Ý nghĩa
2. Phương pháp — Datasets, Preprocessing, TF-IDF, 4 thuật toán
3. Kết quả — Thí nghiệm 1 (same-dataset) + Thí nghiệm 2 (cross-dataset) + Feature/Error analysis
4. Kết luận & Hướng phát triển
5. Hướng dẫn chạy hệ thống

---

## Thứ tự Notebooks

| # | Notebook | Input | Output |
|---|----------|-------|--------|
| 01 | `01_preprocessing.ipynb` | KaggleHub ISOT | `preprocessed_isot_full.csv` |
| 02 | `02_eda.ipynb` | processed CSV | charts `reports/` |
| 03 | `03_vectorization.ipynb` | processed CSV | ISOT TF-IDF splits |
| 04 | `04_naive_bayes.ipynb` | ISOT splits | `naive_bayes_model.pkl` ✅ |
| 05 | `05_logistic_regression.ipynb` | ISOT splits | `lr_model.pkl` ✅ |
| 06 | `06_svm.ipynb` | ISOT splits | `svm_model.pkl` ✅ |
| 07 | `07_random_forest.ipynb` | ISOT splits | `rf_model.pkl` ✅ |
| 08 | `08_comparison.ipynb` | 4 models + ISOT test | metrics bảng ⏳ |
| 09 | `09_feature_error_analysis.ipynb` | best model | feature charts ⏳ |
| 10 | `10_welfake_preprocessing.ipynb` | KaggleHub WELFake | `preprocessed_welfake_full.csv` ✅ |
| 11 | `11_welfake_eda.ipynb` | WELFake CSV | WELFake charts ✅ |
| 12 | `12_welfake_vectorization.ipynb` | WELFake CSV | WELFake TF-IDF splits ✅ |
| 13 | `13_welfake_svm.ipynb` | WELFake train/val | `svm_welfake_model.pkl` ✅ |
| 14 | `14_welfake_naive_bayes.ipynb` | WELFake train/val | `nb_welfake_model.pkl` ⏳ |
| 15 | `15_welfake_logistic_regression.ipynb` | WELFake train/val | `lr_welfake_model.pkl` ⏳ |
| 16 | `16_welfake_random_forest.ipynb` | WELFake train/val | `rf_welfake_model.pkl` ⏳ |
| 17 | `17_cross_dataset_eval.ipynb` | all models + exact raw test splits | cross-eval table ⏳ |

---

## Phân công nhóm

| Thành viên | Phase chính |
|------------|-------------|
| Hưng | Phase 1, Phase 4 (NB ✅), Phase 5+6 |
| Xuân | Phase 2, Phase 4 (LR ✅), Phase 5+6 |
| Dung | Phase 3, Phase 4 (SVM ✅), Phase 7 |
| Thủy | Phase 0, Phase 4 (RF ✅), Phase 8, Phase 9 |

---

## Dependency Graph

```
Phase 0 → Phase 1 → Phase 2
                 └→ Phase 3 → Phase 4 (NB✅ LR✅ SVM✅ RF✅)
                                         ↓
                                      Phase 5 → Phase 6
                                                    ↓
          Phase 7.1✅→7.2✅→7.3✅→7.4 (SVM✅, NB/LR/RF⏳) → Phase 8 → Phase 9
```

---

## Ghi chú kỹ thuật

- **Label convention:** 0=REAL, 1=FAKE — áp dụng toàn bộ ISOT và WELFake
- **Data leakage:** Vectorizer chỉ `fit` trên train set; `transform` cho val/test/cross-test
- **Reproducibility:** `random_state=42` cho mọi model và split
- **Model persistence:** `joblib.dump` / `joblib.load`
- **Metrics chính:** F1-score (weighted); báo cáo đủ Accuracy, Precision, Recall, F1, Confusion Matrix
- **WELFake lớn hơn ISOT:** Train 50K vs 27K → RF GridSearch trên WELFake sẽ rất chậm, cân nhắc giảm n_estimators grid
