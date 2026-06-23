# Phase 8 — Evaluation & Comparison (Chiến lược đánh giá và so sánh)

> **Trạng thái:** ⏳ Chờ Phase 4 RF + Phase 7.4.2–7.4.4 hoàn thành
> **Cập nhật:** 2026-06-23

---

## Mục tiêu

Thực hiện đánh giá hiệu năng và so sánh 4 thuật toán Học máy (Naive Bayes, Logistic Regression, SVM, Random Forest) dưới 3 kịch bản thực nghiệm khác nhau nhằm kiểm chứng tính bền vững, khả năng tổng quát hóa và sự đánh đổi giữa các nguồn dữ liệu khác nhau.

---

## Chi tiết 3 kịch bản thực nghiệm

### 1. Kịch bản 1: Đánh giá Nội miền (In-domain Evaluation)
*   **Cách làm:**
    *   Huấn luyện trên `ISOT Train` ➔ Đánh giá trên `ISOT Test`.
    *   Huấn luyện trên `WELFake Train` ➔ Đánh giá trên `WELFake Test`.
*   **Vectorizer:** Mỗi mô hình sử dụng Vectorizer tương ứng của tập dữ liệu nguồn đó (được `fit` trên tập train tương ứng).
*   **Ý nghĩa:** Đo lường mức trần hiệu năng (baseline tốt nhất) khi mô hình được học và kiểm tra trên dữ liệu có cùng phân phối từ vựng và chủ đề.

### 2. Kịch bản 2: Đánh giá Chéo miền (Cross-domain Evaluation)
*   **Cách làm:**
    *   Sử dụng mô hình huấn luyện trên `ISOT Train` ➔ Đánh giá trên `WELFake Test`.
    *   Sử dụng mô hình huấn luyện trên `WELFake Train` ➔ Đánh giá trên `ISOT Test`.
*   **Kỹ thuật quan trọng (Tránh rò rỉ dữ liệu):**
    *   Không được `fit_transform()` lại vectorizer trên tập dữ liệu đích.
    *   Phải dùng đúng Vectorizer của tập train nguồn để `.transform()` văn bản gốc của tập test đích về định dạng ma trận đặc trưng phù hợp với mô hình nguồn.
*   **Ý nghĩa:** Trọng tâm đánh giá khả năng tổng quát hóa (generalization) của thuật toán và độ bền vững trước hiện tượng lệch phân phối dữ liệu (domain shift).

### 3. Kịch bản 3: Huấn luyện Gộp (Pooled-Domain Evaluation)
*   **Cách làm:**
    *   Gộp tập Train của `ISOT` và `WELFake` thành một tập train lớn (Pooled Train).
    *   Huấn luyện mô hình chung trên tập Pooled Train.
    *   Đánh giá độc lập trên tập Test của `ISOT` và tập Test của `WELFake` để so sánh với Kịch bản 1 & 2.
*   **Kỹ thuật quan trọng:**
    *   Vì từ vựng của hai bộ khác nhau, nhóm bắt buộc phải `fit` một Vectorizer TF-IDF mới trên tập Pooled Train để xây dựng bộ từ vựng gộp đại diện.
    *   Dùng Vectorizer mới này để `.transform()` tập Pooled Train trước khi huấn luyện mô hình, và dùng để `.transform()` hai tập test riêng lẻ khi đánh giá.
*   **Ý nghĩa:** Kiểm tra sự đánh đổi (trade-off) giữa khả năng khái quát hóa đa miền và độ chính xác chuyên biệt khi tăng quy mô và tính đa dạng của tập huấn luyện.

---

## Phân chia Notebooks

Để đảm bảo các file không quá lớn và dễ theo dõi, Phase 8 được chia làm 2 notebook:

| Notebook | Tác vụ | Dữ liệu đầu vào | Kết quả đầu ra |
|----------|--------|-----------------|----------------|
| `17_cross_dataset_eval.ipynb` | Kịch bản 1 & Kịch bản 2 | 8 mô hình đã lưu + raw test splits | Bảng so sánh In-domain vs Cross-domain |
| `18_pooled_training_eval.ipynb` | Kịch bản 3 | Tập train gộp (raw text) + raw test splits | 4 mô hình gộp mới + Bảng so sánh Pooled-domain |

---

## Chỉ số Đánh giá (Metrics)

Quy ước chung toàn dự án: **0 = REAL (Tin thật), 1 = FAKE (Tin giả)**. Các chỉ số được trích xuất trên lớp tin giả (lớp 1) bao gồm:
*   **Accuracy:** Tỷ lệ dự đoán đúng tổng thể.
*   **Precision:** Tỷ lệ tin giả thực sự trong số tin mô hình cảnh báo là giả (tránh lọc nhầm tin thật).
*   **Recall:** Tỷ lệ tin giả bị phát hiện trong thực tế (tránh bỏ sót tin giả).
*   **F1-Score (weighted & macro):** Chỉ số trung bình hài hòa cân bằng.
*   **Confusion Matrix:** Thể hiện rõ các lỗi False Positive (báo nhầm) và False Negative (bỏ sót).
