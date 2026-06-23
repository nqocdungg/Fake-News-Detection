# CHANGELOG

---

## [2026-06-23] Phase 7.4.4 — Random Forest WELFake hoàn thành & Nén mô hình

### Hoàn thành
- Tạo mới và chạy đầy đủ notebook [16_welfake_random_forest.ipynb](file:///c:/Users/sf/Documents/GitHub/Fake-News-Detection/notebooks/16_welfake_random_forest.ipynb).
- Sử dụng GridSearchCV với Random Forest trên tập dữ liệu WELFake.
- Nén trực tiếp file model `.pkl` bằng tham số `compress=3` của `joblib`, giảm kích thước từ **198.65 MB** xuống còn **~46 MB**, qua đó giải quyết thành công vấn đề quá giới hạn 100MB của GitHub mà không cần dùng đến Git LFS.
- Cập nhật tài liệu lý thuyết Random Forest (trong file artifact ngoài dự án).
- Cập nhật trạng thái trong [master-plan.md](file:///c:/Users/sf/Documents/GitHub/Fake-News-Detection/docs/plan/master-plan.md).

### Kết quả
- Best params: `max_depth=None`, `min_samples_split=2`, `n_estimators=300`
- Best CV F1 weighted: **0.9471**
- Validation Accuracy: **0.9492**
- Validation F1 weighted: **0.9492**
- Lần chạy cuối: GridSearchCV hoàn thành trong **6683.34 giây** (~111 phút). Quá trình mất thời gian rất dài do sự chênh lệch lớn về độ đa dạng của dữ liệu (WELFake vs ISOT) khiến cây quyết định phải phân nhánh rất sâu.
- Confusion matrix: `[[5279, 278], [263, 4991]]`

---

## [2026-06-23] Phase 8 — Triển khai đánh giá chéo miền (Cross-Dataset Evaluation) cho LR & SVM thành công

### Hoàn thành
- Sửa đổi notebook [10_welfake_preprocessing.ipynb](file:///d:/PROJECT_GIT/Fake-News-Detection/notebooks/10_welfake_preprocessing.ipynb) loại bỏ việc đảo ngược nhãn vô tình để đồng bộ nhãn chuẩn (`0 = REAL`, `1 = FAKE`).
- Chạy lại các notebook Vectorization (12), SVM (13) và Logistic Regression (15) cho WELFake với nhãn đã được chuẩn hóa.
- Tạo mới và chạy thành công notebook [17_cross_dataset_eval.ipynb](file:///d:/PROJECT_GIT/Fake-News-Detection/notebooks/17_cross_dataset_eval.ipynb).
- Thực hiện đánh giá hiệu năng nội miền (In-domain) và chéo miền (Cross-domain) cho cả hai mô hình Logistic Regression và SVM.
- Xuất các đồ thị confusion matrix và bảng so sánh F1-score vào thư mục `reports/`.

### Kết quả
1. **Logistic Regression (LR)**:
   - ISOT ➔ ISOT (In-domain): F1-score (weighted) = **99.43%**
   - WELFake ➔ WELFake (In-domain): F1-score (weighted) = **94.34%**
   - ISOT ➔ WELFake (Cross-domain): F1-score (weighted) = **85.43%**
   - WELFake ➔ ISOT (Cross-domain): F1-score (weighted) = **98.29%**
2. **SVM (LinearSVC)**:
   - ISOT ➔ ISOT (In-domain): F1-score (weighted) = **99.55%**
   - WELFake ➔ WELFake (In-domain): F1-score (weighted) = **94.51%**
   - ISOT ➔ WELFake (Cross-domain): F1-score (weighted) = **85.38%**
   - WELFake ➔ ISOT (Cross-domain): F1-score (weighted) = **98.26%**

### Nhận xét
- Sự chênh lệch lớn giữa hai chiều đánh giá chéo: WELFake ➔ ISOT đạt F1 cực kỳ cao (~98.3%), chứng tỏ WELFake là tập dữ liệu tổng quát tốt. Trong khi đó, chiều ngược lại ISOT ➔ WELFake bị giảm sút do hiện tượng domain shift (ISOT thiên lệch về phong cách của Reuters).

---

## [2026-06-23] Phase 4 — Random Forest ISOT hoàn thành

### Hoàn thành
- Đã chạy thành công notebook `07_random_forest.ipynb`.
- Thực hiện GridSearchCV tìm kiếm siêu tham số tốt nhất cho Random Forest trên dữ liệu TF-IDF của ISOT.
- Cập nhật mô hình `models/rf_model.pkl` và kết quả đánh giá (Confusion Matrix).

### Kết quả
- Siêu tham số tối ưu: `max_depth=None, min_samples_split=5, n_estimators=300`.
- Điểm Accuracy trên tập Validation: **0.98**
- Điểm F1-score (weighted): **0.98**
- Lần chạy cuối: GridSearchCV hoàn thành trong **1687.11 giây** (~28 phút).
- Confusion Matrix: `[[3179, 0], [46, 2573]]`

---

## [2026-06-22] Phase 7.4.3 — Logistic Regression WELFake hoàn thành

### Hoàn thành
- Tạo mới và chạy đầy đủ notebook [15_welfake_logistic_regression.ipynb](file:///d:/PROJECT_GIT/Fake-News-Detection/notebooks/15_welfake_logistic_regression.ipynb)
- Sử dụng GridSearchCV với lưới siêu tham số tối ưu (`liblinear` và `saga` với `max_iter=100`, `tol=0.01` để tăng tốc)
- Khởi chạy song song với `n_jobs=2` giúp tránh hiện tượng tràn bộ nhớ (OOM) và tự động dừng tiến trình của hệ điều hành Windows trên tập dữ liệu lớn của WELFake
- Lưu mô hình tối ưu tại [lr_welfake_model.pkl](file:///d:/PROJECT_GIT/Fake-News-Detection/models/lr_welfake_model.pkl)
- Cập nhật trạng thái trong [master-plan.md](file:///d:/PROJECT_GIT/Fake-News-Detection/docs/plan/master-plan.md) và [phase-07.4-model-training-welfake.md](file:///d:/PROJECT_GIT/Fake-News-Detection/docs/plan/phase-07.4-model-training-welfake.md)

### Kết quả
- Best params: `C=5.0`, `solver='saga'`, `max_iter=500` (penalty='l2' mặc định)
- Best CV F1 weighted: **0.9447**
- Validation Accuracy: **0.9459**
- Validation F1 weighted: **0.9459**
- Lần chạy cuối: GridSearchCV hoàn thành trong **75.39 giây**
- Confusion matrix: `[[5291, 266], [319, 4935]]`

### Ghi chú
- Logistic Regression đạt hiệu năng gần tương đồng với LinearSVC (F1 ~0.944) trên tập validation nhưng có khả năng phân phối xác suất và điều chuẩn linh hoạt hơn.

---


## [2026-06-21] Phase 7.4.1 — SVM WELFake hoàn thành

### Hoàn thành
- Chạy đầy đủ `notebooks/13_welfake_svm.ipynb`
- GridSearchCV LinearSVC: `C=[0.01,0.1,1,10]`, `cv=5`, `scoring=f1_weighted`
- Lưu `models/svm_welfake_model.pkl`
- Tạo:
  - `reports/welfake_svm_c_tuning.png`
  - `reports/welfake_svm_kernel_comparison.png`
  - `reports/welfake_svm_confusion_matrix.png`
- Cập nhật master plan, plan tổng Phase 7, Phase 7.4 và Phase 7.4.1 theo triển khai thực tế

### Kết quả
- Best params: `C=1`, `max_iter=2000`, `random_state=42`
- Best CV F1 weighted: **0.9441**
- Validation Accuracy: **0.9451**
- Validation F1 weighted: **0.9451**
- Validation F1 macro: **0.9451**
- Lần chạy cuối: baseline **2.34s**, GridSearchCV **21.0s**
- Confusion matrix: `[[5302,255],[338,4916]]`
- SVC RBF subsample 400 tốt nhất: `C=10`, `gamma=scale`, F1 weighted **0.8424**

### Ghi chú
- Không sử dụng WELFake test set.
- Sửa giải thích `gamma='auto'`: gamma quá nhỏ làm kernel gần 1 và giảm khả năng phân biệt, không phải kernel gần 0.
- Cảnh báo cleanup `joblib resource_tracker` trên Windows không làm notebook thất bại; model và metrics đã được kiểm chứng độc lập.

---

## [2026-06-21] Fix label convention + Kế hoạch Phase 7.4.1

### Sửa lỗi label (REAL=0, FAKE=1)
- Audit toàn bộ project xác nhận canonical mapping: `0=REAL`, `1=FAKE`
- Kiểm tra artifacts:
  - ISOT processed: 38,653 rows — REAL(0)=21,196, FAKE(1)=17,457
  - WELFake processed: 72,074 rows — REAL(0)=37,046, FAKE(1)=35,028
  - Tất cả `y_train/y_val/y_test.pkl` của hai dataset chỉ chứa `{0,1}` và giữ đúng phân phối
  - `models/svm_model.pkl` có `classes_=[0,1]`
- `notebooks/06_svm.ipynb` — 4 chỗ sai:
  - Cell 4: `label_map` {0:'FAKE',1:'REAL'} → {0:'REAL',1:'FAKE'} + output text
  - Cell 15: `target_names` ['FAKE (0)','REAL (1)'] → ['REAL (0)','FAKE (1)'] + output text
  - Cell 16: `display_labels` ['FAKE','REAL'] → ['REAL','FAKE']
  - Cell 16: Comment TN/FP/FN/TP hoán vị → đúng theo 0=REAL,1=FAKE
- `notebooks/02_eda.ipynb` — 3 chỗ sai:
  - Cell 17: `fake_text` dùng label==0 (REAL), `real_text` dùng label==1 (FAKE) → đã hoán vị đúng
  - Cell 23: `X_fake` dùng label==0 → đã sửa thành label==1
  - Cell 31: `common_bigrams_real` dùng label==1, `common_bigrams_fake` dùng label==0 → đã hoán vị + sửa comment
  - Tiêu đề sentiment đã sửa thành `REAL (0) vs FAKE (1)`; output cũ của cell đã được xóa
- Chạy lại `notebooks/06_svm.ipynb` và tái tạo `reports/svm_confusion_matrix.png` với thứ tự trục `REAL`, `FAKE`
- Đồng bộ `docs/SVM-walkthrough.md`, `docs/plan/phase-04-model-training-SVM.md` và `docs/plan/master-plan.md`

### Kế hoạch mới
- `docs/plan/phase-07.4.1-svm-welfake.md` — 8 bước chi tiết train SVM (LinearSVC + SVC RBF subsample) trên WELFake 50K×5K
  - Notebook: `13_welfake_svm.ipynb`
  - Output: `models/svm_welfake_model.pkl`, 3 report charts
  - Bao gồm bảng so sánh đầy đủ với Phase 4 SVM (ISOT)

---

## [2026-06-21] Phase 7 — Cập nhật trạng thái & Lên kế hoạch Phase 7.4

### Hoàn thành
- `docs/plan/master-plan.md` — cập nhật toàn diện: Phase 4 results thực tế (NB✅ LR✅ SVM✅ RF❌), Phase 7.1–7.3 ✅ với số liệu thực tế, thêm Phase 7.4 + notebook 14
- `docs/plan/phase-07.4-model-training-welfake.md` — kế hoạch 8 bước chi tiết cho training 4 model trên WELFake

### Kế hoạch Phase 7.4
- Kế hoạch một notebook `13_welfake_model_training.ipynb` đã được thay bằng notebook riêng theo model
- `13_welfake_svm.ipynb` ✅; `14_welfake_naive_bayes.ipynb`, `15_welfake_logistic_regression.ipynb`, `16_welfake_random_forest.ipynb` ⏳
- **Input:** `data/welfake/X_train_tfidf.pkl` (50451×5000), `X_val_tfidf.pkl` (10811×5000)
- **Output:** `models/{nb,lr,svm,rf}_welfake_model.pkl`
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
| Recall FAKE | 0.98 |
| Precision REAL | 0.99 |
| Recall REAL | 0.99 |
| Training time | ~0.62s |

**Confusion Matrix (Validation Set):**
- TN (REAL→REAL): 3,149 ✅
- FP (REAL→FAKE): 30 ← tin thật bị nhận nhầm là giả
- FN (FAKE→REAL): 46 ← tin giả bị nhận nhầm là thật
- TP (FAKE→FAKE): 2,573 ✅

**SVC RBF (subsample 400 mẫu, best: C=10, gamma=scale):** F1 ≈ 0.9323 — không scalable, không dùng cho production.

### Ghi chú
- SVC RBF với `gamma='auto'` cho F1 ≈ 0.39 do γ=1/5000=0.0002 quá nhỏ, kernel degenerates.
- LinearSVC được chọn vì O(n·d) complexity, tận dụng sparse TF-IDF, và vượt trội về cả F1 lẫn tốc độ.
- Test set chưa được dùng — reserved cho Phase 5 (08_comparison.ipynb).

---

## [2026-06-17] Khởi tạo tài liệu dự án

### Hoàn thành
- `docs/plan/master-plan.md` — kế hoạch tổng thể 10 phases (Phase 0–9)
