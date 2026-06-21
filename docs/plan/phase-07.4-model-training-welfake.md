# Phase 7.4 — Model Training: WELFake Dataset

> **Trạng thái tổng thể:** ⏳ Chưa bắt đầu  
> **Notebook output:** `notebooks/13_welfake_model_training.ipynb`  
> **Cập nhật:** 2026-06-21

---

## Mục tiêu

Train cùng 4 thuật toán ML như ISOT (Naive Bayes, Logistic Regression, LinearSVC, Random Forest) nhưng trên **WELFake TF-IDF splits** đã có từ Phase 7.3.

Kết quả của phase này phục vụ hai mục đích:
1. So sánh hiệu năng trong cùng dataset WELFake (baseline cho Phase 8)
2. Chuẩn bị model để đánh giá cross-dataset (Phase 8): WELFake models → dự đoán ISOT test

---

## Dữ liệu đầu vào

| File | Shape | Mô tả |
|------|-------|--------|
| `data/welfake/X_train_tfidf.pkl` | (50451, 5000) | Sparse matrix TF-IDF train WELFake |
| `data/welfake/X_val_tfidf.pkl` | (10811, 5000) | Sparse matrix TF-IDF val WELFake |
| `data/welfake/X_test_tfidf.pkl` | (10812, 5000) | *(Không dùng ở phase này — reserved Phase 8)* |
| `data/welfake/y_train.pkl` | (50451,) | Nhãn train: 0=REAL, 1=FAKE |
| `data/welfake/y_val.pkl` | (10811,) | Nhãn val |

**Phân phối nhãn train (ước tính):**  
WELFake tổng: REAL=37,046 (51.4%) / FAKE=35,028 (48.6%) → gần balanced → không cần `class_weight` để xử lý imbalance như ISOT.

> **Quy ước nhãn:** 0=REAL, 1=FAKE — nhất quán toàn dự án.

---

## Hyperparameter Grids

Giữ **nguyên các grid** đã dùng cho ISOT để so sánh công bằng. Điều chỉnh nhỏ ở RF để tránh timeout trên 50K samples.

| Model | Hyperparameter Grid |
|-------|---------------------|
| Naive Bayes | `alpha` ∈ [0.001, 0.01, 0.1, 1.0]; `fit_prior` ∈ [True, False] |
| Logistic Regression | `C` ∈ [0.01, 0.1, 1, 5, 10, 20]; `penalty` ∈ ['l1','l2']; `solver` ∈ ['liblinear','saga']; `class_weight` ∈ [None,'balanced'] |
| LinearSVC | `C` ∈ [0.01, 0.1, 1, 10]; `max_iter=2000` (fixed) |
| Random Forest | `n_estimators` ∈ [**100, 200**]; `max_depth` ∈ [None, 10, 20]; `min_samples_split` ∈ [2, 5] |

**GridSearchCV config toàn bộ:** `cv=5`, `scoring='f1_weighted'`, `n_jobs=-1`

---

## Các bước thực hiện

---

### Bước 1 — Import & Setup
**Trạng thái:** ⏳

**Làm gì:** Import thư viện, set `random_state=42`. In phiên bản scikit-learn và numpy để ghi lại reproducibility.

**Lý do:** Nhất quán với mọi notebook trước. `random_state=42` bắt buộc cho mọi model và split trong toàn dự án.

---

### Bước 2 — Load dữ liệu
**Trạng thái:** ⏳

**Làm gì:** Dùng `joblib.load()` load 4 file: `X_train`, `X_val`, `y_train`, `y_val` từ `data/welfake/`. In shape và phân phối nhãn (`value_counts()`) để xác nhận đúng file WELFake.

```python
import joblib
X_train = joblib.load('data/welfake/X_train_tfidf.pkl')
X_val   = joblib.load('data/welfake/X_val_tfidf.pkl')
y_train = joblib.load('data/welfake/y_train.pkl')
y_val   = joblib.load('data/welfake/y_val.pkl')
```

**Lý do:** Phân biệt rõ với ISOT splits (lưu ở `data/`). **Không load X_test** — dành cho Phase 8.

---

### Bước 3 — Naive Bayes: GridSearchCV
**Trạng thái:** ⏳

**Làm gì:**
1. Baseline `MultinomialNB()` (default alpha=1.0), ghi F1 val baseline.
2. GridSearchCV trên `alpha` ∈ [0.001, 0.01, 0.1, 1.0] × `fit_prior` ∈ [True, False] → 8 combos × 5 folds = 40 fits.
3. Ghi lại best params, CV F1, thời gian.

**Lý do:** NB nhanh nhất — chạy trước để có kết quả tham chiếu sớm. `alpha` kiểm soát Laplace smoothing; trên WELFake (đa nguồn, vocab phong phú hơn ISOT) `alpha=0.01` hoặc `alpha=0.1` có thể tối ưu hơn `alpha=1.0`.

**Ước tính thời gian:** < 2 phút.

---

### Bước 4 — Logistic Regression: GridSearchCV
**Trạng thái:** ⏳

**Làm gì:**
1. Baseline `LogisticRegression(C=1.0, max_iter=500)`, ghi F1 val.
2. GridSearchCV theo grid ở bảng trên. Lưu ý: `penalty='l1'` không hợp lệ với `solver='saga'`+`class_weight='balanced'` trong một số config — scikit-learn sẽ tự bỏ các combo không hợp lệ nếu dùng `ParameterGrid` đúng cách; hoặc dùng `error_score=np.nan` để bỏ qua. **Khuyến nghị:** dùng hai GridSearch riêng:
   - Grid A: `solver=liblinear`, `penalty=['l1','l2']`, `C=[...]`, `class_weight=[None,'balanced']`
   - Grid B: `solver=saga`, `penalty=['l1','l2']`, `C=[...]`, `class_weight=[None,'balanced']`, `max_iter=500`
3. Vẽ đồ thị F1 theo C với từng penalty.

**Lý do:** WELFake gần balanced nên `class_weight='balanced'` ít cần thiết hơn ISOT, nhưng vẫn nên thử để xác nhận. `saga` hỗ trợ cả `l1` và `l2`, tốt cho dataset lớn.

**Ước tính thời gian:** 15–40 phút (70 combos × 5 folds × 50K samples, giảm với `n_jobs=-1`).

---

### Bước 5 — SVM (LinearSVC): GridSearchCV
**Trạng thái:** ⏳

**Làm gì:**
1. Baseline `LinearSVC(C=1, max_iter=2000)`, ghi F1 val.
2. GridSearchCV: `C` ∈ [0.01, 0.1, 1, 10], `cv=5` → 20 fits.
3. Vẽ đồ thị F1 theo C (giống bước 4 của `06_svm.ipynb`).

**Lý do:** LinearSVC O(n·d) → phù hợp với 50K×5K sparse. Không thử SVC RBF trên WELFake vì đã xác nhận từ Phase 4 ISOT rằng RBF không scalable.

**Ước tính thời gian:** < 3 phút.

---

### Bước 6 — Random Forest: GridSearchCV
**Trạng thái:** ⏳

> ⚠️ **CẢNH BÁO HIỆU NĂNG:** RF GridSearch trên 50K samples rất chậm.  
> `n_estimators=[100,200]` × `max_depth=[None,10,20]` × `min_samples_split=[2,5]` = 12 combos × 5 folds = **60 fits**.  
> Với `n_jobs=-1`, ước tính **30–90 phút** tùy CPU.  
> **Cần chạy notebook qua đêm hoặc trên máy đủ mạnh.**

**Làm gì:**
1. Baseline `RandomForestClassifier(n_estimators=100, random_state=42)`, ghi F1 val và thời gian.
2. GridSearchCV theo grid (2×3×2 = 12 combos).
3. In best params, CV F1, val F1.

**Kỹ thuật giảm thời gian (tùy chọn):**
- Giảm grid xuống: `n_estimators=[100]`, `max_depth=[None, 10]`, `min_samples_split=[2]` → 2 combos × 5 = 10 fits nếu máy quá chậm.
- Ghi rõ trong notebook nếu đã rút gọn grid.

**Lý do:** RF thường cho kết quả tốt nhưng chậm. Vẫn cần train để hoàn thiện bộ 4 model cho Phase 8.

---

### Bước 7 — Tổng hợp kết quả trên Validation Set
**Trạng thái:** ⏳

**Làm gì:**
- In bảng tổng hợp 4 model:

| Model | Best Params | Val Accuracy | Val F1 (weighted) | Train time |
|-------|-------------|:---:|:---:|:---:|
| Naive Bayes | ... | ... | ... | ... |
| Logistic Regression | ... | ... | ... | ... |
| LinearSVC | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... |

- Vẽ bar chart F1 của 4 model (horizontal bar chart).
- Lưu chart ra `reports/welfake_model_comparison_val.png`.

**Lý do:** Cung cấp cái nhìn tổng quan trước khi Phase 8 dùng test set. Validation set đã được dùng để select hyperparams nên không thể dùng làm kết quả final — đó là lý do Phase 8 dùng test set.

---

### Bước 8 — Lưu 4 models
**Trạng thái:** ⏳

**Làm gì:** Lưu từng model bằng `joblib.dump()`:

```python
joblib.dump(best_nb,  'models/nb_welfake_model.pkl')
joblib.dump(best_lr,  'models/lr_welfake_model.pkl')
joblib.dump(best_svm, 'models/svm_welfake_model.pkl')
joblib.dump(best_rf,  'models/rf_welfake_model.pkl')
```

In tên file và kích thước. In summary cuối: 4 model đã sẵn sàng cho Phase 8.

**Lý do:** Phase 8 sẽ load các file này cùng với `tfidf_vectorizer_welfake.pkl` để predict trên ISOT test set. Đặt tên rõ ràng `_welfake_` tránh nhầm với ISOT models.

---

## Output Files

| File | Mô tả |
|------|-------|
| `models/nb_welfake_model.pkl` | MultinomialNB tối ưu trên WELFake |
| `models/lr_welfake_model.pkl` | LogisticRegression tối ưu trên WELFake |
| `models/svm_welfake_model.pkl` | LinearSVC tối ưu trên WELFake |
| `models/rf_welfake_model.pkl` | RandomForestClassifier tối ưu trên WELFake |
| `reports/welfake_model_comparison_val.png` | Bar chart F1 val 4 models |

---

## Kết quả kỳ vọng

WELFake gần balanced hơn ISOT và lớn hơn (50K vs 27K train) → F1 dự kiến cao, nhưng cũng có thể thấp hơn nếu WELFake phức tạp hơn (nhiều nguồn, nhiều domain hơn).

| Model | Val F1 (dự kiến) | Ghi chú |
|-------|:---:|---------|
| Naive Bayes | ~0.93–0.96 | Hiệu quả với sparse TF-IDF |
| Logistic Regression | ~0.97–0.99 | Thường là top performer |
| LinearSVC | ~0.97–0.99 | Comparable với LR |
| Random Forest | ~0.95–0.98 | Chậm nhưng robust |

---

## Lưu ý quan trọng

- **KHÔNG** dùng `X_test` ở phase này — test set reserved cho Phase 8
- **KHÔNG** dùng vectorizer ISOT để transform WELFake data — đã có `tfidf_vectorizer_welfake.pkl` riêng
- **KHÔNG** `fit_transform` lại vectorizer — data đã được transform từ Phase 7.3
- Nếu RF quá chậm, ghi rõ thời gian và có thể rút gọn grid (document lại)
- Phase 8 sẽ là lần đầu tiên dùng `X_test_tfidf.pkl` của cả ISOT và WELFake
