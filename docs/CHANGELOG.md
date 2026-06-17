# CHANGELOG

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
- `AGENTS.md` — hướng dẫn cho AI agents: conventions, commands, CHANGELOG policy
