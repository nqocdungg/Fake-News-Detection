# Phase 7.2 — WELFake Exploratory Data Analysis

> **Trạng thái:** ✅ Hoàn thành  
> **Notebook:** `notebooks/11_welfake_eda.ipynb`  
> **Input:** `data/processed/preprocessed_welfake_full.csv`  
> **Cập nhật:** 2026-06-18

---

## Mục tiêu

Khám phá đặc trưng ngôn ngữ của WELFake, so sánh với ISOT để hiểu sự khác biệt domain — thông tin quan trọng để diễn giải kết quả cross-dataset evaluation ở Phase 8.

**Quy ước nhãn đầu vào:** `REAL=0`, `FAKE=1`, đã được chuẩn hóa trong notebook 10.

---

## Các bước thực hiện

### Bước 1 — Load & Thống kê cơ bản ✅
**Làm gì:** Load CSV, in shape, phân phối nhãn, null check.

### Bước 2 — Phân phối nhãn ✅
**Làm gì:** Pie chart + bar chart FAKE vs REAL.  
**Lý do:** WELFake gần balanced (35K fake / 37K real) — so sánh với ISOT (23K fake / 21K real để biết class imbalance ảnh hưởng ra sao.

### Bước 3 — Phân phối độ dài văn bản ✅
**Làm gì:** Histogram số từ (sau preprocessing) của FAKE vs REAL, so sánh với ISOT.  
**Lý do:** WELFake từ nhiều nguồn → có thể có phân phối độ dài khác ISOT. Nếu FAKE articles ngắn hơn nhiều, độ dài là leaky feature cần biết.

### Bước 4 — Word Cloud ✅
**Làm gì:** 2 word cloud: từ hay gặp ở FAKE vs REAL (sau preprocessing).  
**Lý do:** Visual hóa sự khác biệt từ vựng giữa hai lớp trong WELFake, so sánh với ISOT.

### Bước 5 — Top Unigrams & Bigrams ✅
**Làm gì:** Bar chart top 20 unigrams và bigrams mỗi class dùng `CountVectorizer`.  
**Lý do:** Xác định từ/cụm từ đặc trưng của WELFake fake news — có thể khác ISOT (domain shift).

### Bước 6 — So sánh WELFake vs ISOT ✅
**Làm gì:** Bảng thống kê so sánh: số mẫu, tỉ lệ nhãn, độ dài trung bình, vocabulary size.  
**Lý do:** Cung cấp context để diễn giải domain shift trong Phase 8 cross-evaluation.

---

## Output Files

| File | Mô tả |
|------|-------|
| `reports/welfake_label_dist.png` | Phân phối nhãn |
| `reports/welfake_text_length_by_label.png` | Độ dài text FAKE vs REAL |
| `reports/welfake_wordcloud_fake.png` | Word cloud FAKE |
| `reports/welfake_wordcloud_real.png` | Word cloud REAL |
| `reports/welfake_top_ngrams.png` | Top unigrams & bigrams |
| `reports/welfake_isot_comparison.png` | So sánh hai dataset |
