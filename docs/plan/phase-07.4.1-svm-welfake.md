# Phase 7.4.1 — Model Training: SVM trên WELFake

> **Trạng thái tổng thể:** ⏳ Chưa bắt đầu  
> **Notebook output:** `notebooks/13_welfake_svm.ipynb`  
> **Model output:** `models/svm_welfake_model.pkl`  
> **Cập nhật:** 2026-06-21

---

## Mục tiêu

Train và đánh giá SVM trên **WELFake TF-IDF splits** (từ Phase 7.3). Cụ thể:
- **LinearSVC** — GridSearchCV tìm C tối ưu trên 50K samples
- **SVC RBF** — Phân tích tính khả dụng với subsample (xác nhận lại kết luận từ ISOT)

Kết quả phục vụ hai mục đích:
1. Đánh giá SVM trong cùng dataset WELFake → để Phase 7.4 tổng hợp bộ 4 model
2. Lưu `svm_welfake_model.pkl` → Phase 8 dùng để test **WELFake → ISOT** cross-domain

---

## Dữ liệu đầu vào

| File | Shape | Mô tả |
|------|-------|--------|
| `data/welfake/X_train_tfidf.pkl` | (50451, 5000) | Sparse CSR TF-IDF train |
| `data/welfake/X_val_tfidf.pkl` | (10811, 5000) | Sparse CSR TF-IDF val |
| `data/welfake/X_test_tfidf.pkl` | (10812, 5000) | *(reserved — KHÔNG dùng ở notebook này)* |
| `data/welfake/y_train.pkl` | (50451,) | Nhãn: 0=REAL, 1=FAKE |
| `data/welfake/y_val.pkl` | (10811,) | Nhãn val |

**Phân phối nhãn train (ước tính từ tỉ lệ 70% stratified):**

| Nhãn | Tổng dataset | Train (~70%) | Val (~15%) |
|------|:---:|:---:|:---:|
| REAL (0) | 37,046 | ~25,932 | ~5,557 |
| FAKE (1) | 35,028 | ~24,519 | ~5,254 |
| **Tỉ lệ FAKE** | **48.6%** | **~48.6%** | **~48.6%** |

> **So sánh với ISOT:** WELFake gần balanced hơn (48.6% vs 45.2% FAKE). Train set lớn gần gấp đôi (50K vs 27K). Không cần `class_weight='balanced'`.

---

## GridSearch Config

| Param | Values | Ghi chú |
|-------|--------|---------|
| `C` | [0.01, 0.1, 1, 10] | Nhất quán với ISOT để so sánh |
| `max_iter` | 2000 (fixed) | Đảm bảo hội tụ trên 50K samples |
| `cv` | 5 | Stratified K-Fold |
| `scoring` | `f1_weighted` | Metric chính |
| `n_jobs` | -1 | Song song hóa |

---

## Các bước thực hiện

---

### Bước 1 — Import & Setup
**Trạng thái:** ⏳

**Làm gì:** Import thư viện, set `RANDOM_STATE = 42`, ghi lại phiên bản scikit-learn.

**Lý do:** Reproducibility. `random_state=42` bắt buộc cho mọi model và split trong toàn dự án (xem `AGENTS.md`).

---

### Bước 2 — Load Dữ liệu
**Trạng thái:** ⏳

**Làm gì:** Dùng `joblib.load()` load 4 files từ `data/welfake/`. In shape và phân phối nhãn để xác nhận đang dùng đúng WELFake splits (không phải ISOT).

```python
DATA_DIR = Path('../data/welfake')

X_train = joblib.load(DATA_DIR / 'X_train_tfidf.pkl')
X_val   = joblib.load(DATA_DIR / 'X_val_tfidf.pkl')
y_train = np.array(joblib.load(DATA_DIR / 'y_train.pkl'))
y_val   = np.array(joblib.load(DATA_DIR / 'y_val.pkl'))

label_map = {0: 'REAL', 1: 'FAKE'}  # Quy ước toàn dự án
```

**Checkpoint sau bước này:**
- X_train.shape == (50451, 5000) ✓
- unique(y_train) == [0, 1] ✓
- Tỉ lệ FAKE xấp xỉ 48.6% ✓

**Lý do:** Tường minh đường dẫn `data/welfake/` (khác với ISOT ở `data/`). Không load X_test — reserved cho Phase 8.

---

### Bước 3 — LinearSVC: Baseline
**Trạng thái:** ⏳

**Làm gì:** Train `LinearSVC(C=1.0, max_iter=2000, random_state=42)`, predict trên val set, ghi F1 và thời gian.

```python
baseline = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
```

**Ước tính thời gian:** 1–3 giây (WELFake 50K × 5K sparse — LinearSVC vẫn rất nhanh).

**Lý do:** Baseline thiết lập điểm tham chiếu. Nếu baseline với C=1 đã rất cao (>0.95), GridSearch sẽ xác nhận hoặc cải thiện nhỏ.

---

### Bước 4 — LinearSVC: GridSearchCV
**Trạng thái:** ⏳

**Làm gì:**
- Grid: `C` ∈ [0.01, 0.1, 1, 10] — 4 combos × 5 folds = **20 fits**
- In kết quả theo từng C, ghi best params và best CV F1
- Vẽ đồ thị F1 theo C (giống `06_svm.ipynb` cell 8–9)
- Lưu chart ra `reports/welfake_svm_c_tuning.png`

```python
param_grid = {'C': [0.01, 0.1, 1, 10]}
gs = GridSearchCV(
    LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
    param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
)
gs.fit(X_train, y_train)
```

**Ước tính thời gian:** 30–60 giây (`n_jobs=-1` song song 4 folds).

**Lý do:** C=1 thường tối ưu cho TF-IDF text (đã xác nhận trên ISOT). Thí nghiệm này xác nhận lại cho WELFake — domain khác có thể cần C khác do vocabulary distribution khác.

**So sánh với ISOT:** Nếu C tối ưu trên WELFake khác ISOT → phân tích lý do (domain shift, vocabulary distribution).

---

### Bước 5 — SVC RBF: Phân tích trên Subsample
**Trạng thái:** ⏳

**Làm gì:** Lấy stratified subsample 400 mẫu. Test SVC RBF với:
- `C` ∈ [0.1, 1, 10]
- `gamma` ∈ ['scale', 'auto']

Ghi thời gian và F1 từng combo. Ghi nhận đặc biệt kết quả `gamma='auto'`.

```python
N_SUB = 400
sss = StratifiedShuffleSplit(n_splits=1, train_size=N_SUB, random_state=RANDOM_STATE)
idx_tr, _ = next(sss.split(X_train, y_train))
X_sub = X_train[idx_tr].toarray()
y_sub = y_train[idx_tr]
```

**Ước tính thời gian:** < 20 giây (6 combos × 1 fit × 400 samples).

**Kết quả kỳ vọng:**
- `gamma='auto'` → F1 ≈ 0.39 (predict all one class) — **cùng nguyên nhân với ISOT**: γ = 1/5000 = 0.0002 → kernel ≈ 0
- `gamma='scale'` → F1 có thể tốt hơn (~0.90–0.93 trên subsample)

**Lý do:** WELFake có cùng `max_features=5000` → gamma='auto' = 1/5000 = 0.0002 → kernel degeneracy y hệt ISOT. Bước này xác nhận tính tổng quát của kết luận: *LinearSVC vượt trội SVC RBF cho TF-IDF sparse, bất kể dataset*.

---

### Bước 6 — So sánh LinearSVC vs SVC RBF
**Trạng thái:** ⏳

**Làm gì:** Bảng so sánh và bar chart:

| Model | Dataset train | F1 val | Thời gian | Scalable? |
|-------|:---:|:---:|:---:|:---:|
| LinearSVC (best C) | 50,451 | — | — | ✅ |
| SVC RBF (best, subsample) | 400 | — | ~Xs/fit | ❌ |

Lưu chart ra `reports/welfake_svm_kernel_comparison.png`.

**Lý do:** So sánh tường minh, cùng format với `06_svm.ipynb` cell 12–13, phục vụ báo cáo cuối (Phase 9).

---

### Bước 7 — Đánh giá Chính Thức trên Validation Set
**Trạng thái:** ⏳

**Làm gì:** Dùng LinearSVC với best params từ GridSearch, predict toàn bộ `X_val`:

```python
best_model = gs.best_estimator_
y_pred = best_model.predict(X_val)
```

In:
- Accuracy, F1 (weighted), F1 (macro)
- `classification_report(y_val, y_pred, target_names=['REAL (0)', 'FAKE (1)'])`
- Confusion Matrix heatmap → `reports/welfake_svm_confusion_matrix.png`
- In giải thích TN/FP/FN/TP đúng quy ước:

```python
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
# tn = REAL→REAL, fp = REAL→FAKE, fn = FAKE→REAL, tp = FAKE→FAKE
print(f'TN (REAL→REAL): {tn:,}  | FP (REAL→FAKE): {fp:,}  ← tin thật bị nhận nhầm là giả')
print(f'FN (FAKE→REAL): {fn:,}  | TP (FAKE→FAKE): {tp:,}  ← tin giả bị nhận nhầm là thật')
```

**Lý do:** Đây là đánh giá chính thức lưu vào báo cáo. Test set vẫn chưa được dùng.

---

### Bước 8 — Lưu Model
**Trạng thái:** ⏳

**Làm gì:**

```python
model_path = Path('../models/svm_welfake_model.pkl')
joblib.dump(best_model, model_path)
```

In đường dẫn, kích thước file, type, C, max_iter.

**Lý do:** `svm_welfake_model.pkl` sẽ được dùng ở Phase 8 để dự đoán trên ISOT test set (cross-domain evaluation). Đặt tên `_welfake_` để phân biệt với `svm_model.pkl` (ISOT).

---

## Kết quả kỳ vọng

| Metric | WELFake (kỳ vọng) | ISOT (thực tế) | Ghi chú |
|--------|:---:|:---:|---------|
| Best C | 1 | 1 | Có thể khác nếu domain WELFake khác |
| CV F1 (5-fold) | > 0.97 | 0.9852 | WELFake 50K → nhiều data hơn |
| Val Accuracy | > 0.97 | 0.9869 | |
| Val F1 (weighted) | > 0.97 | 0.9869 | |
| Training time (GridSearch) | ~30–60s | ~16.9s | 50K vs 27K samples |

> WELFake có nhiều nguồn hơn ISOT (McIntire, GossipCop, PolitiFact...) → vocabulary phong phú hơn nhưng cũng noise hơn. F1 có thể thấp hơn một chút (~0.96–0.98) hoặc tương đương.

---

## Output Files

| File | Mô tả |
|------|-------|
| `notebooks/13_welfake_svm.ipynb` | Notebook chính |
| `models/svm_welfake_model.pkl` | LinearSVC tối ưu trên WELFake |
| `reports/welfake_svm_c_tuning.png` | F1 theo C (GridSearch) |
| `reports/welfake_svm_kernel_comparison.png` | LinearSVC vs SVC RBF |
| `reports/welfake_svm_confusion_matrix.png` | Confusion Matrix val set |

---

## So sánh với Phase 4 SVM (ISOT)

| Khía cạnh | Phase 4 (ISOT) | Phase 7.4.1 (WELFake) |
|-----------|:--------------:|:---------------------:|
| Notebook | `06_svm.ipynb` | `13_welfake_svm.ipynb` |
| Train size | 27,057 | 50,451 |
| Val size | 5,798 | 10,811 |
| Label balance | 54.8% REAL / 45.2% FAKE | 51.4% REAL / 48.6% FAKE |
| Data path | `data/` | `data/welfake/` |
| Model output | `svm_model.pkl` | `svm_welfake_model.pkl` |
| GridSearch time | ~17s | ~30–60s (est.) |
| SVC RBF subsample | 400 mẫu | 400 mẫu (same) |

---

## Lưu ý quan trọng

- **Quy ước nhãn:** 0=REAL, 1=FAKE — bắt buộc nhất quán (đã fix lỗi trong `06_svm.ipynb`)
- **Không** dùng X_test ở notebook này
- **Không** fit lại vectorizer — TF-IDF đã xong từ Phase 7.3
- **Không** dùng vectorizer ISOT để transform WELFake (khác vocabulary)
- `display_labels=['REAL', 'FAKE']` — index 0=REAL, index 1=FAKE cho ConfusionMatrixDisplay
- `target_names=['REAL (0)', 'FAKE (1)']` cho classification_report
