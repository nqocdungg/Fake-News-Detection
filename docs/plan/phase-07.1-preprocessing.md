# Phase 7.1 — WELFake Preprocessing

> **Trạng thái:** ✅ Hoàn thành  
> **Notebook:** `notebooks/10_welfake_preprocessing.ipynb`  
> **Output:** `data/processed/preprocessed_welfake_full.csv`  
> **Cập nhật:** 2026-06-18

---

## Mục tiêu

Xây dựng dataset WELFake đã tiền xử lý, sẵn sàng cho EDA và vectorization. Tái sử dụng `preprocess_text()` từ ISOT pipeline.

---

## Input

| Nguồn | Mô tả |
|------|-------|
| KaggleHub | Tải `saurabhshahane/fake-news-classification` và dùng cache cục bộ |
| `src/preprocessing.py` | Hàm `preprocess_text()` tái sử dụng |

---

## Các bước thực hiện

### Bước 1 — Tải, Load & Kiểm tra dữ liệu gốc ✅
**Làm gì:** Dùng `kagglehub.dataset_download()`, tìm `WELFake_Dataset.csv` trong thư mục cache, sau đó load CSV và kiểm tra shape, dtypes, null, sample rows.

**Lý do:** WELFake có thể chứa null trong cột `text` hoặc `title`. Cần biết extent của vấn đề trước khi xử lý.

**Kết quả kỳ vọng:**
- ~72,134 rows × 4 columns (`Unnamed: 0`, `title`, `text`, `label`)
- ~150 null trong `text`, ~15 null trong `title`

---

### Bước 2 — Xử lý giá trị thiếu ✅
**Làm gì:**
1. Dòng có null cả `title` lẫn `text` → drop
2. Dòng null `text` nhưng có `title` → fill `text = ""` (title đã đủ content)
3. Dòng null `title` nhưng có `text` → fill `title = ""`

**Lý do:** Không drop toàn bộ dòng null text vì nhiều bài chỉ có title (đặc biệt fake news ngắn). Tận dụng title làm thêm signal.

---

### Bước 3 — Gộp title + text ✅
**Làm gì:** Tạo cột `full_text = title.str.strip() + " " + text.str.strip()`, strip whitespace.

**Lý do:** WELFake tách biệt `title` và `text` (khác ISOT đã gộp sẵn). Gộp cả hai cho pipeline nhất quán và tận dụng thông tin từ tiêu đề.

---

### Bước 4 — Kiểm tra và chuẩn hóa nhãn ✅
**Làm gì:** Verify nhãn nguồn chỉ gồm `{0, 1}`, giữ lại trong `source_label`, sau đó map `{0: 1, 1: 0}` sang cột `label`.

**Lý do:** WELFake gốc dùng `FAKE=0`, `REAL=1`, trong khi pipeline ISOT hiện tại dùng `REAL=0`, `FAKE=1`. Cần thống nhất trước EDA, training và cross-dataset evaluation.

---

### Bước 5 — Áp dụng preprocessing pipeline ✅
**Làm gì:** Apply `preprocess_text()` lên cột `full_text` với `tqdm` progress bar. Lưu vào cột `processed_text`.

**Lý do:** Cùng hàm preprocessing với ISOT đảm bảo feature space nhất quán khi cross-test. Reuters leakage fix cũng cần thiết vì WELFake chứa McIntire (Reuters-sourced) articles.

**Runtime:** ~72K bài × preprocessing → khoảng 3-5 phút.

---

### Bước 6 — Kiểm tra kết quả ✅
**Làm gì:** Đếm số dòng sau preprocessing, kiểm tra empty string, so sánh độ dài text trước/sau, xem 5 sample.

**Lý do:** Đảm bảo không có bài bị "blank" sau khi xóa stopwords (bài quá ngắn, toàn stopwords).

---

### Bước 7 — Lưu kết quả ✅
**Làm gì:** Save DataFrame với cột `['processed_text', 'label']` vào `data/processed/preprocessed_welfake_full.csv`.

**Lý do:** Giữ format nhất quán với ISOT (`preprocessed_isot_full.csv`). Các phase sau chỉ cần đọc file này.

---

## Output

| File | Columns | Mô tả |
|------|---------|-------|
| `data/processed/preprocessed_welfake_full.csv` | `processed_text`, `label` | ~72K rows đã tiền xử lý |
