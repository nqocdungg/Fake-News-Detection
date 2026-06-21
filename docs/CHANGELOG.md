# CHANGELOG

---

## [2026-06-21] Phase 7 — Cập nhật trạng thái & Lên kế hoạch Phase 7.4

### Hoàn thành
- `docs/plan/master-plan.md` — cập nhật toàn diện: Phase 4 results thực tế (NB✅ LR✅ SVM✅ RF❌), Phase 7.1–7.3 ✅ với số liệu thực tế, thêm Phase 7.4 + notebook 14
- `docs/plan/phase-07.4-model-training-welfake.md` — kế hoạch 8 bước chi tiết cho training 4 model trên WELFake

### Kế hoạch Phase 7.4
- **Notebook:** `notebooks/13_welfake_model_training.ipynb` (chưa tạo)
- **Input:** `data/welfake/X_train_tfidf.pkl` (50451×5000), `X_val_tfidf.pkl` (10811×5000)
- **Output:** `models/{nb,lr,svm,rf}_welfake_model.pkl` + `reports/welfake_model_comparison_val.png`
- **Lưu ý RF:** GridSearch 60 fits trên 50K samples ước tính 30–90 phút; cần chạy qua đêm

### Trạng thái hiện tại (2026-06-21)
- Phase 7.1 (preprocessing) ✅ — 72,074 rows, REAL=37,046/FAKE=35,028
- Phase 7.2 (EDA) ✅ — 4 charts sinh ra, FAKE mean=330 words
- Phase 7.3 (vectorization) ✅ — splits: 50451/10811/10812 × 5000
- Phase 7.4 (model training) ⏳ — kế hoạch đã có, chưa implement

---

## [2026-06-20] Phase 4 — Model Training: Tinh chỉnh Logistic Regression & Đồng bộ tài liệu

### Hoàn thành
- `notebooks/05_logistic_regression.ipynb` — Tinh chỉnh tham số mở rộng, vẽ Confusion Matrix + ROC Curve song song, thêm các cell giải thích Markdown.
- `notebooks/02_eda.ipynb` — Định dạng và làm đẹp các cell Markdown bằng các khối alert/callout.
- `models/lr_model.pkl` — Mô hình Logistic Regression tối ưu nhất sau tinh chỉnh (40.9 KB).
- `reports/lr_evaluation_plots.png` — Biểu đồ Confusion Matrix và ROC Curve song song trên tập Validation.

### Kết quả tinh chỉnh Logistic Regression
- **Không gian GridSearch mới:** 350 lượt fit (70 combinations × 5 folds).
- **Siêu tham số tối ưu:** `{'C': 20.0, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga', 'tol': 0.01}`.
- **Hiệu năng:**
  - Điểm F1 tốt nhất trên Cross-Validation đạt **98.35%** (tăng **0.11%** so với baseline 98.24%).
  - Đạt điểm F1/Accuracy khoảng **98.64%** trên tập Validation.
  - Chỉ số AUC đạt **0.9991** (phân loại gần như hoàn hảo).
- **Thời gian chạy:** **372.53 giây** (~6 phút 12 giây), tối ưu hóa tốc độ nhờ thiết lập `tol=0.01` và `max_iter=100` cho solver `saga`.

### Ghi chú kỹ thuật
- Mặc dù hỗ trợ `l1` (Lasso) và `elasticnet`, GridSearchCV vẫn chọn `l2` (Ridge) làm penalty tối ưu, cho thấy việc giữ lại thông tin từ mọi từ khóa trong từ vựng 5,000 từ vẫn hiệu quả hơn.
- `class_weight='balanced'` giúp điều chỉnh ngưỡng phân lớp tối ưu hơn đối với sự mất cân bằng nhẹ của dữ liệu huấn luyện.

---

## [2026-06-18] Phase 7 — Dataset 2 Integration: WELFake (7.0 → 7.3)

### Hoàn thành
- `docs/plan/phase-07-dataset2-welfake.md` — kế hoạch tổng thể Phase 7, lý do chọn WELFake, output files
- `docs/plan/phase-07.1-preprocessing.md` — kế hoạch 7 bước preprocessing WELFake
- `docs/plan/phase-07.2-eda.md` — kế hoạch EDA với 6 bước phân tích
- `docs/plan/phase-07.3-vectorization.md` — kế hoạch vectorization độc lập
- `notebooks/10_welfake_preprocessing.ipynb` — pipeline: load → xử lý null → gộp title+text → preprocess_text() → lưu CSV
- `notebooks/11_welfake_eda.ipynb` — label dist, text length, word cloud, top n-grams, so sánh ISOT vs WELFake
- `notebooks/12_welfake_vectorization.ipynb` — TF-IDF config comparison, fit/transform, lưu splits + vectorizer
- `data/welfake/` — thư mục chứa splits WELFake (tạo khi chạy notebook 12)

### Thông tin Dataset
- **WELFake**: ~72,134 rows, nguồn gốc 0=FAKE (~35,028), 1=REAL (~37,106)
- **Nguồn**: Kaggle (saurabhshahane/fake-news-classification) — notebook 10 tự tải và cache bằng KaggleHub
- **Cột**: `title` + `text` → gộp thành `full_text` trước preprocessing
- **Label output**: đổi nhãn nguồn thành `0=REAL`, `1=FAKE` để nhất quán với pipeline ISOT hiện tại

### Quyết định kỹ thuật
- Vectorizer fit **độc lập** (`tfidf_vectorizer_welfake.pkl`) — không dùng lại ISOT vectorizer
- Hyperparameters nhất quán: `max_features=5000`, `ngram_range=(1,2)` để cross-eval công bằng
- Split 70/15/15 stratified, `random_state=42`
- Cùng `preprocess_text()` từ `src/preprocessing.py` (Reuters leakage fix giữ lại — WELFake chứa McIntire/Reuters articles)

### Ghi chú
- Notebooks chưa chạy; notebook 10 sẽ tự tải WELFake từ Kaggle khi thực thi
- Mọi code đã được viết hoàn chỉnh; cần cài dependencies từ `requirements.txt` trước khi chạy
- Phase 7 sẽ đánh dấu ✅ sau khi toàn bộ 3 notebooks chạy thành công

---

## [2026-06-17] Phase 4 — Model Training: SVM (LinearSVC)

### Hoàn thành
- `notebooks/06_svm.ipynb` — notebook đầy đủ: LinearSVC GridSearchCV, SVC RBF subsample analysis, evaluation, confusion matrix
- `models/svm_model.pkl` — LinearSVC (C=1, max_iter=2000) trained trên 27,057 mẫu (39.8 KB)
- `reports/svm_linearsvc_c_tuning.png` — biểu đồ CV F1 theo C
- `reports/svm_kernel_comparison.png` — biểu đồ so sánh LinearSVC vs SVC RBF
- `reports/svm_confusion_matrix.png` — confusion matrix trên validation set
- `docs/plan/phase-04-model-training-SVM.md` — kế hoạch chi tiết phase 4 SVM
- `docs/SVM-walkthrough.md` — báo cáo lý thuyết và kết quả SVM

### Kết quả thực nghiệm

**GridSearchCV LinearSVC (cv=5, scoring=f1_weighted) trên 27,057 mẫu:**

| C | CV F1 (mean) | CV F1 (std) |
|---|:---:|:---:|
| 0.01 | 0.9648 | ±0.0040 |
| 0.1 | 0.9812 | ±0.0013 |
| **1** | **0.9852** | ±0.0017 |
| 10 | 0.9815 | ±0.0012 |

**Model tốt nhất: LinearSVC (C=1)**

| Metric | Giá trị |
|--------|---------|
| Val Accuracy | **0.9869** |
| Val F1 (weighted) | **0.9869** |
| Val F1 (macro) | 0.9868 |
| Precision FAKE | 0.99 |
| Recall FAKE | 0.99 |
| Precision REAL | 0.99 |
| Recall REAL | 0.98 |
| Training time | ~0.62s |

**Confusion Matrix (Validation Set):**
- TN (FAKE→FAKE): 3,149 ✅
- FP (FAKE→REAL): 30 ← tin giả bị nhận nhầm
- FN (REAL→FAKE): 46 ← tin thật bị nhận nhầm
- TP (REAL→REAL): 2,573 ✅

**SVC RBF (subsample 400 mẫu, best: C=10, gamma=scale):** F1 ≈ 0.9323 — không scalable, không dùng cho production.

### Ghi chú
- SVC RBF với `gamma='auto'` cho F1 ≈ 0.39 do γ=1/5000=0.0002 quá nhỏ, kernel degenerates.
- LinearSVC được chọn vì O(n·d) complexity, tận dụng sparse TF-IDF, và vượt trội về cả F1 lẫn tốc độ.
- Test set chưa được dùng — reserved cho Phase 5 (08_comparison.ipynb).

---

## [2026-06-17] Khởi tạo tài liệu dự án

### Hoàn thành
- `docs/plan/master-plan.md` — kế hoạch tổng thể 10 phases (Phase 0–9)
