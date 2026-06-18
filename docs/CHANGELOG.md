# CHANGELOG

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
