# SVM trên WELFake — Báo cáo Chi tiết

> **Dự án:** Fake News Detection | **Học phần:** Nhập môn Trí tuệ nhân tạo  
> **Phase:** 7.4.1 | **Notebook:** `notebooks/13_welfake_svm.ipynb`  
> **Model:** `models/svm_welfake_model.pkl` | **Hoàn thành:** 2026-06-21

---

## 1. Bối cảnh và Mục tiêu

### 1.1 Vị trí trong dự án

Phase 7.4.1 là bước đầu tiên trong chuỗi huấn luyện bốn thuật toán trên dataset thứ hai (WELFake). Mục tiêu kép của phase này là:

1. **Xây dựng model SVM cho WELFake** — đủ chất lượng để tham gia đánh giá chéo dataset ở Phase 8.
2. **Đánh giá khả năng tổng quát hóa của SVM** — xem liệu một thuật toán đạt F1 = 0.9869 trên ISOT có duy trì được hiệu năng tương đương khi đối mặt với dữ liệu đa nguồn, nhiều nhiễu hơn không.

Kết quả ở phase này bổ sung một cột quan trọng vào bảng so sánh tổng hợp (Phase 5 và Phase 8) và là kiểm chứng thực nghiệm đầu tiên về tính bền vững (robustness) của mô hình.

> **Phạm vi:** Phase 7.4.1 chỉ sử dụng tập train và validation. Test set được giữ nguyên cho Phase 8 (cross-dataset evaluation).

### 1.2 Lý do chọn LinearSVC

Toàn bộ pipeline text classification của dự án dùng TF-IDF sparse matrix (50,451 × 5,000 cho WELFake). LinearSVC là lựa chọn tự nhiên vì:

- **Độ phức tạp O(n·d)** — tuyến tính theo cả số mẫu lẫn số chiều, so với O(n²) của SVC kernel.
- **Tận dụng cấu trúc thưa (sparse)** — LinearSVC (liblinear solver) tính trực tiếp trên sparse CSR matrix, không cần dense conversion.
- **Precedent từ Phase 4** — LinearSVC đạt F1 = 0.9869 trên ISOT trong chưa đầy 1 giây, thiết lập một đường cơ sở (baseline) vững chắc cho WELFake.

---

## 2. Lý thuyết Support Vector Machine

> Phần này tóm tắt các khái niệm cốt lõi trực tiếp liên quan đến thí nghiệm. Trình bày toán học đầy đủ xem tại `docs/SVM-walkthrough.md`.

### 2.1 Hyperplane và Margin

SVM tìm **siêu phẳng phân tách** (hyperplane) $\mathbf{w}^T \mathbf{x} + b = 0$ sao cho **margin** — khoảng cách từ hyperplane đến support vector gần nhất của mỗi lớp — là cực đại:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \;\frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{s.t. } \quad y_i\left(\mathbf{w}^T \mathbf{x}_i + b\right) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i$$

Tham số $C$ kiểm soát đánh đổi giữa margin rộng và lỗi phân loại:

| $C$ | Ý nghĩa | Hệ quả thực tế |
|-----|---------|----------------|
| $C \ll 1$ | Regularization mạnh | Margin rộng, chấp nhận nhiều lỗi → underfit khi dữ liệu phức tạp |
| $C = 1$ | Cân bằng (mặc định) | Điểm khởi đầu tốt cho hầu hết bài toán NLP |
| $C \gg 1$ | Ít regularization | Cố phân loại đúng mọi điểm train → nguy cơ overfit |

### 2.2 Tại sao SVM phù hợp với TF-IDF?

Sau khi vectorize, mỗi bài báo là một vector thưa trong $\mathbb{R}^{5000}$. Ba đặc tính của SVM làm cho nó hiệu quả trong không gian này:

1. **$d \gg n$ không phải vấn đề** — SVM tìm max-margin hyperplane dựa trên support vectors, không phụ thuộc vào số chiều.
2. **Linear separability trong không gian TF-IDF** — Fake news và real news dùng từ ngữ đặc trưng khác nhau; ranh giới tuyến tính thường đủ tốt.
3. **Regularization L2 ngầm** — $\|\mathbf{w}\|^2$ trong hàm mục tiêu penalize trọng số lớn, kiểm soát overfitting mà không cần thêm tham số.

### 2.3 Vấn đề với SVC Kernel RBF trên TF-IDF

Kernel RBF ánh xạ dữ liệu qua hàm:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\!\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

Với TF-IDF 5,000 chiều, `gamma='auto'` = $1/5000 = 0.0002$. Vì vector TF-IDF rất thưa, khoảng cách $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ giữa hai bài báo bất kỳ thường rất lớn, dẫn đến:

$$K(\mathbf{x}_i, \mathbf{x}_j) = e^{-0.0002 \cdot \|\Delta\|^2} \approx 0 \quad \forall i \neq j$$

Toàn bộ kernel matrix xấp xỉ ma trận đơn vị — model không còn khả năng phân biệt giữa các mẫu, suy biến về dự đoán một lớp duy nhất (F1 ≈ tỉ lệ lớp đa số). Sử dụng `gamma='scale'` khắc phục một phần, nhưng vẫn không đủ competitive với LinearSVC khi train trên toàn bộ dữ liệu.

---

## 3. Dataset WELFake

### 3.1 Nguồn gốc và cấu trúc

WELFake (Fake News Detection Dataset — Verma et al., 2021) tổng hợp bài báo từ bốn nguồn: **Kaggle**, **McIntire**, **Reuters**, và **BuzzFeed Political**. Đây là điểm khác biệt căn bản so với ISOT (chỉ có Reuters và PolitiFact):

| Đặc điểm | ISOT | WELFake |
|-----------|------|---------|
| Số bài sau preprocessing | 38,653 | 72,074 |
| Số nguồn | 2 | 4+ |
| Phong cách viết | Đồng nhất | Đa dạng |
| Thách thức chính | Phân loại tin thật từ một nguồn duy nhất | Phân loại tin từ nhiều nguồn, phong cách khác nhau |

Đặc tính đa nguồn tạo ra nhiễu cả ở nội dung lẫn phong cách văn phong, khiến ranh giới REAL/FAKE kém tách biệt hơn trong không gian TF-IDF.

### 3.2 Nhãn và phân phối

Dataset gốc dùng quy ước `0=FAKE, 1=REAL` (ngược với pipeline dự án). Notebook `10_welfake_preprocessing.ipynb` ánh xạ lại:

$$\text{WELFake gốc} \xrightarrow{\text{remap}} \text{Pipeline: } 0 = \text{REAL},\; 1 = \text{FAKE}$$

Phân phối sau remapping và split:

| Split | Tổng | REAL (0) | FAKE (1) | Tỉ lệ FAKE |
|-------|-----:|--------:|--------:|:-----------:|
| Train | 50,451 | 25,932 | 24,519 | 48.6% |
| Validation | 10,811 | 5,557 | 5,254 | 48.6% |
| Test *(reserved)* | 10,812 | — | — | — |

Dataset gần **balanced** (tỉ lệ FAKE ≈ 48.6%) — không cần `class_weight='balanced'` hay oversampling.

### 3.3 Đặc trưng đầu vào (TF-IDF)

Vectorizer WELFake được fit **độc lập** trên train set WELFake (`tfidf_vectorizer_welfake.pkl`, 185 KB), với cùng siêu tham số như ISOT:

```python
TfidfVectorizer(
    max_features = 5000,
    ngram_range  = (1, 2),   # unigrams + bigrams
    sublinear_tf = True,     # log(1 + tf) thay vì tf thuần
)
```

Việc dùng vectorizer riêng biệt đảm bảo từ vựng phản ánh đặc trưng ngôn ngữ của WELFake, và tránh data leakage khi đánh giá chéo dataset ở Phase 8.

---

## 4. Thiết kế Thí nghiệm

### 4.1 Quy trình tổng thể

```
data/welfake/
  ├── X_train_tfidf.pkl  (50,451 × 5,000)   ─┐
  ├── X_val_tfidf.pkl    (10,811 × 5,000)    │  Input
  ├── y_train.pkl        (50,451,)           │
  └── y_val.pkl          (10,811,)           ─┘
         │
         ▼
  [Bước 1] Baseline LinearSVC(C=1)
         │
         ▼
  [Bước 2] GridSearchCV — tìm C tối ưu
         │
         ▼
  [Bước 3] Đánh giá best model trên val set
         │
         ▼
  [Bước 4] SVC RBF subsample — đánh giá tính khả dụng
         │
         ▼
  models/svm_welfake_model.pkl
```

> Test set (`X_test_tfidf.pkl`, `y_test.pkl`) **không được load hoặc sử dụng** trong toàn bộ phase này.

### 4.2 Baseline LinearSVC

Trước GridSearch, train một model với tham số mặc định để có điểm tham chiếu:

```python
baseline = LinearSVC(C=1, max_iter=2000, random_state=42)
baseline.fit(X_train, y_train)
```

Mục đích: nếu baseline đã đạt F1 tốt, GridSearch chỉ cần tinh chỉnh nhỏ; nếu baseline yếu, cần mở rộng không gian tìm kiếm.

### 4.3 GridSearchCV — LinearSVC

```python
param_grid = {'C': [0.01, 0.1, 1, 10]}

gs = GridSearchCV(
    LinearSVC(max_iter=2000, random_state=42),
    param_grid,
    cv      = 5,
    scoring = 'f1_weighted',
    n_jobs  = -1,
)
gs.fit(X_train, y_train)
```

**Lý do chọn không gian tìm kiếm:**
- 4 giá trị C trải đều trên thang logarithmic (0.01 → 10), đủ để quan sát xu hướng underfitting và overfitting.
- `f1_weighted` phù hợp với bài toán phân loại nhị phân gần balanced.
- `cv=5` là tiêu chuẩn trong dự án, nhất quán với Phase 4.

Tổng số lần fit: 4 giá trị × 5 folds = **20 fits**.

### 4.4 SVC RBF trên Subsample

Do hạn chế tính toán ($O(n^2)$ kernel matrix), SVC RBF được đánh giá trên subsample stratified 400 mẫu:

```python
param_grid_rbf = {
    'C':     [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
}
# Subsample 400 mẫu (stratified theo y_train)
```

Kết quả không so sánh trực tiếp với LinearSVC (scale train khác nhau), nhưng đủ để kết luận về tính khả dụng của RBF trên bài toán này.

---

## 5. Kết quả

### 5.1 GridSearchCV — LinearSVC

Kết quả 5-fold cross-validation trên toàn bộ train set (50,451 mẫu):

| C | CV F1 weighted (mean) | CV F1 weighted (std) | Nhận xét |
|--:|:---------------------:|:--------------------:|----------|
| 0.01 | 0.9095 | ±0.0025 | Regularization quá mạnh → underfit rõ |
| 0.1 | 0.9395 | ±0.0023 | Tốt hơn nhưng vẫn chưa đạt tối ưu |
| **1** | **0.9441** | **±0.0025** | **Tốt nhất — cân bằng bias-variance** |
| 10 | 0.9358 | ±0.0019 | Giảm nhẹ — regularization yếu hơn làm giảm generalization |

**Nhận xét về xu hướng:**

Đường cong CV F1 theo C có dạng lồi với đỉnh tại C=1. C=0.01 dưới-fit rõ rệt (F1 chênh tới 0.0346 so với C=1). C=10 thay vì tiếp tục tăng lại giảm nhẹ — cho thấy với dữ liệu WELFake, phân loại quá cứng nhắc các điểm training không giúp cải thiện generalization.

**Model tốt nhất:** `LinearSVC(C=1, max_iter=2000, random_state=42)`.

Thời gian GridSearch (lần chạy kiểm chứng): **21.0 giây** — tất cả 20 fits trong vòng nửa phút trên 50K mẫu, khẳng định tính scalable của LinearSVC.

### 5.2 Đánh giá chính thức trên Validation Set

| Metric | Giá trị |
|--------|--------:|
| Accuracy | **0.9451** |
| F1 weighted | **0.9451** |
| F1 macro | **0.9451** |

Classification report đầy đủ:

| Class | Precision | Recall | F1-score | Support |
|-------|:---------:|:------:|:--------:|--------:|
| REAL (0) | 0.94 | 0.95 | **0.95** | 5,557 |
| FAKE (1) | 0.95 | 0.94 | **0.94** | 5,254 |
| **Weighted avg** | **0.95** | **0.95** | **0.95** | 10,811 |

**F1 macro = F1 weighted = 0.9451** — hai chỉ số trùng nhau cho thấy hiệu năng trên hai lớp gần cân bằng, phù hợp với phân phối dữ liệu (REAL 51.4% / FAKE 48.6%).

### 5.3 Confusion Matrix

```
                       Predicted
                  REAL (0)    FAKE (1)
Actual  REAL (0)   5,302        255    ← 255 tin thật bị cảnh báo nhầm
        FAKE (1)     338      4,916    ← 338 tin giả bị nhận nhầm là thật
```

| Ô | Ký hiệu | Số lượng | Ý nghĩa |
|---|---------|--------:|---------|
| True Negative | TN | 5,302 | Tin thật được nhận dạng đúng ✅ |
| False Positive | FP | 255 | Tin thật bị cảnh báo nhầm là giả |
| False Negative | FN | 338 | Tin giả bị nhận nhầm là thật ⚠️ |
| True Positive | TP | 4,916 | Tin giả được nhận dạng đúng ✅ |

**Phân tích lỗi:**

- **FP = 255 (REAL → FAKE):** 4.6% tin thật bị cảnh báo nhầm. Người dùng thấy tin thật nhưng hệ thống đánh dấu là giả — ảnh hưởng đến trải nghiệm nhưng không gây hại về mặt thông tin.
- **FN = 338 (FAKE → REAL):** 6.4% tin giả qua lọt. Đây là loại lỗi nghiêm trọng hơn về mặt xã hội — tin giả được phân loại sai thành tin thật, người đọc có thể tin tưởng và lan truyền.
- **Tỉ lệ FN > FP** (338 > 255) — mô hình thiên hướng conservative, thà bỏ sót một ít tin giả hơn là cảnh báo nhầm tin thật.

### 5.4 SVC RBF trên Subsample (400 mẫu)

| C | gamma | F1 weighted (subsample) |
|--:|-------|:-----------------------:|
| 0.1 | scale | 0.3501 |
| 0.1 | `auto` | 0.3501 |
| 1 | scale | 0.8347 |
| 1 | `auto` | 0.3501 |
| **10** | **scale** | **0.8424** |
| 10 | `auto` | 0.3501 |

**Phân tích `gamma='auto'`:** F1 = 0.3501 ≈ tỉ lệ lớp thiểu số — model dự đoán hầu hết mọi mẫu về một lớp duy nhất. Nguyên nhân đã phân tích ở mục 2.3: $\gamma = 1/5000 = 0.0002$ làm kernel matrix suy biến.

**Phân tích `gamma='scale'`:** Best F1 = 0.8424 ở C=10 trên 400 mẫu train. Không so sánh trực tiếp với LinearSVC (0.9451 trên 50,451 mẫu), nhưng ngay cả với lợi thế đánh giá trên subsample nhỏ, RBF vẫn kém hơn đáng kể và không khả thi về mặt tính toán ở quy mô đầy đủ.

---

## 6. Phân tích Feature Weights

LinearSVC học một vector trọng số $\mathbf{w} \in \mathbb{R}^{5000}$ trên từ vựng WELFake. Với `classes_ = [0, 1]`:
- $w_i > 0$: feature $i$ hướng về **FAKE** (class 1)
- $w_i < 0$: feature $i$ hướng về **REAL** (class 0)

### 6.1 Từ/bigram đặc trưng cho FAKE (w > 0)

| Rank | Feature | $w_i$ |
|-----:|---------|------:|
| 1 | `breitbart` | +7.38 |
| 2 | `president donald` | +7.25 |
| 3 | `com` | +6.72 |
| 4 | `follow` | +6.66 |
| 5 | `said` | +5.65 |
| 6 | `pic twitter` | +5.24 |
| 7 | `york time` | +4.87 |
| 8 | `tuesday` | +3.65 |
| 9 | `thursday` | +3.64 |
| 10 | `said statement` | +3.37 |

**Nhận xét:**
- `breitbart` — Breitbart News là nguồn tin thường bị đánh dấu là phân cực hoặc không đáng tin cậy trong các bộ dataset kiểm chứng sự thật; xuất hiện nhiều trong các bài báo được gán nhãn FAKE.
- `president donald`, `york time` — các bài fake news thường trích dẫn sai hoặc xuyên tạc phát biểu của nhân vật nổi tiếng và các tờ báo lớn.
- `pic twitter`, `follow`, `com` — ngôn ngữ đặc trưng của nội dung lan truyền qua mạng xã hội, thường xuất hiện trong fake news được chia sẻ từ Twitter/Facebook.
- Các từ chỉ ngày trong tuần (`tuesday`, `thursday`, `wednesday`) — có thể phản ánh phong cách viết của một số nguồn fake news cụ thể trong WELFake.

### 6.2 Từ/bigram đặc trưng cho REAL (w < 0)

| Rank | Feature | $w_i$ |
|-----:|---------|------:|
| 1 | `via` | −13.29 |
| 2 | `image via` | −6.45 |
| 3 | `october` | −6.41 |
| 4 | `image` | −5.71 |
| 5 | `breaking` | −5.69 |
| 6 | `video` | −4.22 |
| 7 | `november` | −4.17 |
| 8 | `entire` | −3.96 |
| 9 | `pictwittercom` | −3.91 |
| 10 | `donate` | −3.85 |

**Nhận xét:**
- `via`, `image via` có trọng số rất âm (−13.29, −6.45). Trong WELFake, đây có thể là dấu hiệu của định dạng bài báo cụ thể từ một số nguồn được gán nhãn REAL — ví dụ bài chia sẻ hình ảnh kèm chú thích attribution ("Image via [tên nguồn]").
- Sự xuất hiện của `breaking`, `donate` trong nhóm REAL phản ánh bản chất đa nguồn của WELFake: một số nguồn dùng ngôn ngữ mang tính kêu gọi hoặc giật tít nhưng vẫn được gán nhãn là tin thật trong dataset.
- Đây là minh chứng cho thách thức chính của WELFake — ranh giới ngôn ngữ giữa REAL và FAKE kém sắc nét hơn so với ISOT.

### 6.3 Hàm ý cho Interpretability

Vector $\mathbf{w}$ của LinearSVC cung cấp khả năng giải thích tường minh — tính năng không có ở SVC RBF hay Random Forest. Đây là input trực tiếp cho **Phase 6 (Feature & Error Analysis)**: đối chiếu top features của WELFake model và ISOT model để hiểu sự khác biệt về "từ điển" giữa hai dataset.

---

## 7. So sánh với SVM trên ISOT

### 7.1 Hiệu năng

| Metric | ISOT (Phase 4) | WELFake (Phase 7.4.1) | Chênh lệch |
|--------|:--------------:|:---------------------:|:----------:|
| Train samples | 27,057 | 50,451 | +86.3% |
| Best C | 1 | 1 | — |
| CV F1 weighted | 0.9852 | 0.9441 | **−0.0411** |
| Val Accuracy | 0.9869 | 0.9451 | **−0.0418** |
| Val F1 weighted | 0.9869 | 0.9451 | **−0.0418** |
| Val F1 macro | 0.9868 | 0.9451 | −0.0417 |
| Training time (GridSearch) | ~5s | ~21s | +4× |

### 7.2 Phân tích lỗi so sánh

| Ô confusion matrix | ISOT | WELFake |
|-------------------|-----:|--------:|
| TN (REAL → REAL) | 3,149 | 5,302 |
| FP (REAL → FAKE) | 30 | 255 |
| FN (FAKE → REAL) | 46 | 338 |
| TP (FAKE → FAKE) | 2,573 | 4,916 |
| FP rate (REAL bị nhầm) | 0.94% | 4.59% |
| FN rate (FAKE bị nhầm) | 1.73% | 6.44% |

WELFake có FP rate tăng gấp ~5 lần và FN rate tăng gấp ~4 lần so với ISOT. Điều này phản ánh trực tiếp chất lượng phân tách trong không gian TF-IDF.

### 7.3 Giải thích sự chênh lệch hiệu năng

WELFake có nhiều mẫu train hơn gần gấp đôi (50,451 so với 27,057) nhưng F1 lại thấp hơn đáng kể (0.9451 so với 0.9869). Điều này trực quan có vẻ mâu thuẫn, nhưng được giải thích bởi:

1. **Đa nguồn tạo nhiễu nhãn** — WELFake tổng hợp từ 4+ nguồn với tiêu chí đánh nhãn có thể không hoàn toàn nhất quán. Một số bài báo từ nguồn partisan (ủng hộ một phía) có thể nằm ở vùng ranh giới giữa REAL và FAKE.

2. **Phong cách viết đa dạng** — ISOT chủ yếu từ Reuters (tin tức chuẩn mực, văn phong nhất quán) và PolitiFact (fact-checking có cấu trúc rõ ràng). WELFake bao gồm cả mạng xã hội, blog, và các trang thông tin phi truyền thống — phong cách viết biến thiên lớn.

3. **Vocabulary overlap** — Khi REAL và FAKE news cùng xuất phát từ một số nguồn (ví dụ, cả bài thật và bài giả đều từ môi trường Twitter), TF-IDF khó phân biệt hơn vì vocabulary trùng nhau nhiều.

4. **Không phải dấu hiệu lỗi** — F1 = 0.9451 là kết quả hợp lý và được kỳ vọng cho bài toán multi-source fake news classification. Đây là thông tin quan trọng cho Phase 8: không nên kỳ vọng cross-dataset performance vượt quá single-dataset performance.

---

## 8. Kiểm tra Chất lượng Triển khai

Sau khi chạy notebook, thực hiện kiểm chứng độc lập (separate script) để xác nhận kết quả:

| Kiểm tra | Kết quả |
|----------|:-------:|
| Load đúng `data/welfake/` | ✅ |
| Không load test set | ✅ |
| Không fit lại vectorizer | ✅ |
| LinearSVC nhận sparse CSR (không dense conversion) | ✅ |
| `GridSearchCV(cv=5, scoring='f1_weighted')` | ✅ |
| `random_state=42` | ✅ |
| Nhãn `{0: REAL, 1: FAKE}`, `classes_=[0, 1]` | ✅ |
| Model load lại từ pkl và cho kết quả nhất quán | ✅ |
| Metrics kiểm chứng độc lập khớp output notebook | ✅ |
| Tất cả charts được tạo trong `reports/` | ✅ |

**Ghi chú về `joblib resource_tracker`:** Khi chạy `GridSearchCV(n_jobs=-1)` trên Windows, Python worker processes có thể log cảnh báo `KeyError` từ `resource_tracker` sau khi kết thúc. Đây là vấn đề cleanup của hệ điều hành, không ảnh hưởng đến kết quả tính toán. Model, metrics và charts đều được kiểm chứng là chính xác. Nếu cần output sạch hơn, có thể đặt `n_jobs=1` với đánh đổi runtime dài hơn.

---

## 9. Artifacts

| File | Kích thước | Mô tả |
|------|:----------:|-------|
| `notebooks/13_welfake_svm.ipynb` | — | Notebook thực thi chính |
| `models/svm_welfake_model.pkl` | 40 KB | LinearSVC(C=1) trained trên 50,451 mẫu |
| `models/tfidf_vectorizer_welfake.pkl` | 185 KB | Vectorizer WELFake (dùng để transform trong Phase 8) |
| `reports/welfake_svm_c_tuning.png` | — | CV F1 theo C (GridSearchCV) |
| `reports/welfake_svm_kernel_comparison.png` | — | LinearSVC vs SVC RBF (bar chart) |
| `reports/welfake_svm_confusion_matrix.png` | — | Confusion matrix trên validation set |

> **Lưu ý Phase 8:** `svm_welfake_model.pkl` chỉ nhận vector từ `tfidf_vectorizer_welfake.pkl`. Khi đánh giá cross-domain (WELFake model → ISOT test), phải reconstruct ISOT test raw texts rồi gọi `tfidf_vectorizer_welfake.transform()` — không dùng `data/X_test_tfidf.pkl` của ISOT vì đó là vocabulary khác.

---

## 10. Tóm tắt

| Mục | Nội dung |
|-----|----------|
| Thuật toán | LinearSVC (liblinear solver) |
| Dataset | WELFake — 72,074 bài, 4+ nguồn |
| Best params | `C=1`, `max_iter=2000`, `random_state=42` |
| CV F1 weighted (5-fold) | **0.9441** ± 0.0025 |
| Val Accuracy | **0.9451** |
| Val F1 weighted | **0.9451** |
| Val F1 macro | **0.9451** |
| Thời gian GridSearch | ~21s (20 fits, 50K mẫu) |
| FP (tin thật nhận nhầm là giả) | 255 (4.6%) |
| FN (tin giả nhận nhầm là thật) | 338 (6.4%) |
| Model file | `models/svm_welfake_model.pkl` (40 KB) |

LinearSVC với C=1 là model SVM được chọn cho WELFake. Hiệu năng F1 = 0.9451, thấp hơn 0.0418 so với ISOT (0.9869), là kết quả hợp lý và phản ánh độ khó cao hơn của bài toán phân loại tin tức đa nguồn. Model đủ điều kiện làm input cho Phase 8 (cross-dataset evaluation) sau khi các model WELFake còn lại (Naive Bayes, Logistic Regression, Random Forest) hoàn thành.
