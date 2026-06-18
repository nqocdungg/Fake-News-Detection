# Phase 7.3 — WELFake TF-IDF Vectorization

> **Trạng thái:** ✅ Hoàn thành  
> **Notebook:** `notebooks/12_welfake_vectorization.ipynb`  
> **Input:** `data/processed/preprocessed_welfake_full.csv`  
> **Cập nhật:** 2026-06-18

---

## Mục tiêu

Tạo TF-IDF features độc lập cho WELFake, chuẩn bị splits cho Phase 8 (Cross-Dataset Evaluation).

**Quy ước nhãn đầu vào:** `REAL=0`, `FAKE=1`, giống dữ liệu ISOT hiện tại.

---

## Nguyên tắc quan trọng

**Vectorizer phải được fit độc lập** — không dùng lại `tfidf_vectorizer.pkl` (ISOT). Lý do:

- Mỗi vectorizer học vocabulary từ dataset của nó
- Khi cross-test (ISOT → WELFake), chủ động dùng ISOT vectorizer `.transform()` lên WELFake text
- Điều này phản ánh đúng scenario thực tế: model không biết vocabulary của domain mới

---

## Các bước thực hiện

### Bước 1 — Load & Split ✅
**Làm gì:** Load CSV, `train_test_split` 70/15/15, `stratify=y`, `random_state=42`. In shapes.

**Lý do:** Nhất quán với ISOT split strategy. Stratify đảm bảo tỉ lệ FAKE/REAL đều nhau trong cả 3 splits.

### Bước 2 — TF-IDF Hyperparameter Selection ✅
**Làm gì:** Thử 3 cấu hình, đánh giá bằng `LogisticRegression` nhanh (cv=3):
- `max_features=5000, ngram_range=(1,2)` ← giống ISOT (baseline)
- `max_features=10000, ngram_range=(1,2)`
- `max_features=5000, ngram_range=(1,1)`

**Lý do:** Xác nhận cấu hình tối ưu cho WELFake. Dùng cùng settings với ISOT giúp so sánh công bằng hơn ở Phase 8; tuy nhiên WELFake có vocabulary phong phú hơn nên `max_features=10000` có thể tốt hơn.

### Bước 3 — Fit Vectorizer & Transform ✅
**Làm gì:** Fit `TfidfVectorizer` trên train set. `transform()` (không fit) val và test.

**Lý do:** Data leakage prevention — vectorizer không được "thấy" val/test vocabulary.

### Bước 4 — Lưu Splits & Vectorizer ✅
**Làm gì:** `joblib.dump()` tất cả artifacts vào `data/welfake/` và `models/`.

**Lý do:** Tách riêng `data/welfake/` tránh ghi đè ISOT splits ở `data/`.

---

## Hyperparameters (nhất quán với ISOT)

| Parameter | Giá trị | Lý do |
|-----------|---------|-------|
| `max_features` | 5000 | Nhất quán với ISOT; nếu WELFake tốt hơn với 10K → ghi note |
| `ngram_range` | (1, 2) | Bắt cụm từ 2 chữ; đã xác nhận tốt hơn unigram ở ISOT |
| `random_state` | 42 | Reproducibility |
| Split ratio | 70/15/15 | Nhất quán với ISOT |

---

## Output Files

| File | Mô tả |
|------|-------|
| `models/tfidf_vectorizer_welfake.pkl` | Vectorizer fit trên WELFake train |
| `data/welfake/X_train_tfidf.pkl` | (N_train, 5000) sparse matrix |
| `data/welfake/X_val_tfidf.pkl` | (N_val, 5000) sparse matrix |
| `data/welfake/X_test_tfidf.pkl` | (N_test, 5000) sparse matrix |
| `data/welfake/y_train.pkl` | Labels train |
| `data/welfake/y_val.pkl` | Labels val |
| `data/welfake/y_test.pkl` | Labels test |
