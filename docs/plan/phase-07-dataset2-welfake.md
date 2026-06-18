# Phase 7 — Dataset 2 Integration: WELFake

> **Trạng thái tổng thể:** 🔄 Đang thực hiện  
> **Cập nhật:** 2026-06-18

---

## Tổng quan

Phase 7 tích hợp **WELFake Dataset** vào hệ thống, xây dựng pipeline độc lập (preprocessing → EDA → vectorization) song song với ISOT. Mục tiêu cuối là chuẩn bị dữ liệu cho Phase 8 (Cross-Dataset Evaluation).

---

## Tại sao chọn WELFake?

| Tiêu chí | LIAR | WELFake ✅ |
|----------|------|-----------|
| Format | Câu tuyên bố ngắn (claim) | Bài báo đầy đủ (title + text) |
| Nhãn | 6 nhãn (cần map về binary) | Binary sẵn; chuẩn hóa về `REAL=0`, `FAKE=1` |
| Kích thước | ~12K | ~72K |
| Tương đồng với ISOT | Thấp (domain khác) | Cao (cùng format bài báo) |
| Pipeline reuse | Khó | Dễ — dùng lại preprocessing.py |

WELFake được tổng hợp từ 4 nguồn: Kaggle, McIntire, Reuters, BuzzFeed → độ đa dạng cao hơn ISOT, phù hợp để test generalization.

---

## Thông tin Dataset

| Thuộc tính | Giá trị |
|------------|---------|
| Tên | WELFake Dataset |
| Nguồn | Kaggle: [saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| File | `WELFake_Dataset.csv` |
| Kích thước | ~72,134 rows |
| FAKE (nhãn nguồn=0 → nhãn project=1) | ~35,028 |
| REAL (nhãn nguồn=1 → nhãn project=0) | ~37,106 |
| Cột | `Unnamed: 0`, `title`, `text`, `label` |
| Ngôn ngữ | Tiếng Anh |
| Nguồn gốc | Kaggle + McIntire + Reuters + BuzzFeed |

### Tải dữ liệu

Notebook 10 tự động tải phiên bản mới nhất và dùng lại cache cục bộ bằng KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "saurabhshahane/fake-news-classification"
)
```

Không cần tải hoặc đặt thủ công `WELFake_Dataset.csv` trong `data/raw/`.

---

## Cấu trúc Sub-phases

| Sub-phase | Tên | Notebook | Plan file | Trạng thái |
|-----------|-----|----------|-----------|------------|
| 7.1 | Preprocessing | `10_welfake_preprocessing.ipynb` | `phase-07.1-preprocessing.md` | ✅ Hoàn thành |
| 7.2 | EDA | `11_welfake_eda.ipynb` | `phase-07.2-eda.md` | ✅ Hoàn thành |
| 7.3 | TF-IDF Vectorization | `12_welfake_vectorization.ipynb` | `phase-07.3-vectorization.md` | ✅ Hoàn thành |

---

## Output Files

```
data/
├── processed/
│   └── preprocessed_welfake_full.csv     ← Phase 7.1 output
└── welfake/                              ← Phase 7.3 output
    ├── X_train_tfidf.pkl
    ├── X_val_tfidf.pkl
    ├── X_test_tfidf.pkl
    ├── y_train.pkl
    ├── y_val.pkl
    └── y_test.pkl

models/
└── tfidf_vectorizer_welfake.pkl          ← Phase 7.3 output
```

---

## Điểm khác biệt so với ISOT Pipeline

| | ISOT | WELFake |
|--|------|---------|
| Nguồn raw | KaggleHub cache | KaggleHub cache |
| Cột text | `text` (đã gộp title bên trong) | `title` + `text` riêng → cần gộp |
| Null values | Không đáng kể | Có null trong `text` (~vài trăm dòng) |
| Reuters leakage | Có (fix trong preprocessing.py) | Có (WELFake bao gồm McIntire/Reuters) |
| Kích thước | ~44K | ~72K |
| Vectorizer | `tfidf_vectorizer.pkl` | `tfidf_vectorizer_welfake.pkl` (fit độc lập) |
| Split artifacts | `data/X_*_tfidf.pkl` | `data/welfake/X_*_tfidf.pkl` |

---

## Nguyên tắc kỹ thuật

1. **Vectorizer fit độc lập:** `tfidf_vectorizer_welfake.pkl` được fit chỉ trên WELFake train set — không dùng ISOT vectorizer.
2. **Hyperparameters nhất quán:** `max_features=5000`, `ngram_range=(1,2)`, `random_state=42` — giống ISOT để so sánh công bằng.
3. **Cùng `preprocess_text()`:** Import từ `src/preprocessing.py` — không viết lại.
4. **Cùng split ratio:** 70/15/15, `stratify=y`, `random_state=42`.
5. **Cùng nhãn ISOT:** Đổi nhãn nguồn WELFake `{0: FAKE, 1: REAL}` thành `{0: REAL, 1: FAKE}` trước khi lưu processed CSV.
