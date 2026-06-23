# Current Progress & Report Guide — Fake News Detection

> Cập nhật theo branch `results` sau khi chuẩn hóa pipeline code, model outputs và thư mục kết quả.  
> Mục tiêu file này: đối chiếu lộ trình ban đầu với tình hình code hiện tại, chỉ ra phần đã làm/chưa làm, và mô tả định hướng phân tích để viết báo cáo.

---

## 1. Tổng Quan Tình Hình Hiện Tại

Project hiện đã có pipeline Machine Learning tương đối đầy đủ cho bài toán **Fake News Detection** trên bộ dữ liệu **ISOT Fake News Dataset**.

Các phần code chính đã chạy được:

```text
Raw ISOT data
  -> preprocessing
  -> EDA
  -> TF-IDF vectorization
  -> train 4 models
  -> compare on test set
```

Các notebook core đã được chuẩn hóa:

| Notebook | Vai trò | Trạng thái |
|---|---|---|
| `notebooks/01_preprocessing.ipynb` | Tiền xử lý ISOT | Đã chạy được |
| `notebooks/02_eda.ipynb` | EDA ISOT | Đã chạy được |
| `notebooks/03_vectorization.ipynb` | TF-IDF + split train/val/test | Đã chạy được |
| `notebooks/04_naive_bayes.ipynb` | Train Naive Bayes | Đã chạy được |
| `notebooks/05_logistic_regression.ipynb` | Train Logistic Regression | Đã chạy được |
| `notebooks/06_svm.ipynb` | Train SVM, so sánh Linear/RBF | Đã chạy được |
| `notebooks/07_random_forest.ipynb` | Train Random Forest | Đã chạy được |
| `notebooks/08_comparison.ipynb` | So sánh 4 model trên test set | Đã chạy được |
| `notebooks/09_feature_error_analysis.ipynb` | Feature/error analysis | Chưa làm, mới có tiêu đề |

Các kết quả/log đã được chia folder rõ:

```text
reports/
├── 01_preprocessing/
├── 02_eda/
├── 03_vectorization/
├── 04_naive_bayes/
├── 05_logistic_regression/
├── 06_svm/
├── 07_random_forest/
└── 08_comparison/
```

---

## 2. Bài Toán Đang Làm

### 2.1 Tên bài toán

**Phát hiện tin giả (Fake News Detection) bằng NLP và Machine Learning.**

Đây là bài toán **phân loại văn bản nhị phân**:

| Nhãn | Ý nghĩa |
|---|---|
| `0` | REAL - tin thật |
| `1` | FAKE - tin giả |

### 2.2 Input và output

Input của hệ thống là nội dung bài báo, gồm:

- `title`: tiêu đề bài báo
- `text`: nội dung bài báo
- có thêm metadata như `subject`, `date`, nhưng phần model hiện chủ yếu dùng text đã gộp từ title + text

Output của model:

```text
REAL hoặc FAKE
```

### 2.3 Ý nghĩa thực tiễn

Tin giả lan truyền nhanh trên mạng xã hội và các nền tảng trực tuyến. Nếu không có công cụ hỗ trợ, việc kiểm chứng từng bài viết bằng tay rất tốn thời gian. Mục tiêu của project là xây dựng pipeline tự động để phân loại một bài báo là tin thật hay tin giả dựa trên nội dung văn bản.

### 2.4 Định hướng kỹ thuật

Project đi theo hướng Machine Learning truyền thống:

```text
Text preprocessing
  -> TF-IDF vectorization
  -> Classical ML classifiers
```

Các model được so sánh:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine
- Random Forest

Lý do chọn hướng này:

- Dễ giải thích trong báo cáo.
- Phù hợp học phần Nhập môn Trí tuệ nhân tạo.
- TF-IDF + model tuyến tính thường rất mạnh với text classification.
- Có thể so sánh rõ accuracy, precision, recall, F1, confusion matrix và training time.

---

## 3. Dữ Liệu Và Preprocessing Hiện Tại

### 3.1 Dataset

Dataset chính hiện dùng là **ISOT Fake News Dataset**.

Theo kết quả preprocessing:

| Thông tin | Giá trị |
|---|---:|
| Fake raw rows | 23,481 |
| True raw rows | 21,417 |
| Fake duplicated rows | 3 |
| True duplicated rows | 206 |
| Final rows sau xử lý | 38,651 |
| REAL sau xử lý | 21,195 |
| FAKE sau xử lý | 17,456 |
| Duplicate `full_text` đã loại | 5,402 |
| Empty processed text đã loại | 5 |

File log:

```text
reports/01_preprocessing/preprocessing_summary.json
```

### 3.2 Preprocessing pipeline

Hàm tái sử dụng:

```text
src/preprocessing.py
```

Các bước chính trong `preprocess_text(text)`:

1. Ép input về string.
2. Xóa Reuters pattern:

```python
re.sub(r'\(Reuters\)|\bReuters\b', ' ', text, flags=re.IGNORECASE)
```

3. Lowercase.
4. Xóa URL, email, số, dấu câu, ký tự đặc biệt.
5. Xóa stopwords.
6. Giữ lại negation words:

```text
no, not, never, neither, nor
```

7. Lemmatization.
8. Join token thành `processed_text`.

### 3.3 Vì sao phải fix Reuters leakage?

Trong dataset ISOT, nhiều tin thật đến từ nguồn Reuters. Nếu giữ nguyên từ `Reuters`, model có thể học mẹo rằng cứ có từ Reuters thì là tin thật. Đây là **data leakage** vì model không thật sự hiểu nội dung bài báo, mà dựa vào dấu hiệu nguồn.

Vì vậy cần xóa `Reuters` để model tập trung vào nội dung văn bản.

### 3.4 Vì sao giữ negation words?

Các từ phủ định như `not`, `never`, `no` có ý nghĩa quan trọng trong tin tức.

Ví dụ:

```text
claim is true
claim is not true
```

Nếu xóa `not`, hai câu có thể trở nên gần giống nhau về mặt token, làm giảm chất lượng phân loại.

---

## 4. EDA Hiện Tại

Notebook:

```text
notebooks/02_eda.ipynb
```

Output:

```text
reports/02_eda/
```

Đã có:

| Yêu cầu EDA | Trạng thái | Output |
|---|---|---|
| Distribution fake/real pie + bar | Đã làm | `label_distribution_bar_pie.png` |
| Histogram độ dài text | Đã làm | `text_length_histogram.png` |
| Word cloud fake | Đã làm | `wordcloud_fake.png` |
| Word cloud real | Đã làm | `wordcloud_real.png` |
| Top 20 unigrams/bigrams mỗi class | Đã làm | `top_unigrams_bigrams.png`, `.csv` |
| Null check | Đã làm | `null_summary.csv` |
| Duplicate check | Đã làm | `duplicate_summary.csv` |
| Sample text ví dụ | Đã làm | `sample_texts.csv` |

### Định hướng viết báo cáo EDA

Trong báo cáo, phần EDA nên viết theo cấu trúc:

1. Mô tả tổng quan dataset:
   - nguồn dữ liệu
   - số lượng tin thật/tin giả
   - các cột dữ liệu
   - ngôn ngữ tiếng Anh

2. Phân phối nhãn:
   - dán `label_distribution_bar_pie.png`
   - nhận xét dataset không cân bằng tuyệt đối nhưng vẫn đủ hai lớp

3. Độ dài văn bản:
   - dán `text_length_histogram.png`
   - nhận xét tin thật/tin giả có thể khác nhau về độ dài

4. Word cloud:
   - dán wordcloud REAL/FAKE
   - nhận xét các từ nổi bật ở từng class

5. Top n-grams:
   - dán `top_unigrams_bigrams.png`
   - giải thích unigram/bigram giúp hiểu từ/cụm từ hay xuất hiện

---

## 5. Vectorization Hiện Tại

Notebook:

```text
notebooks/03_vectorization.ipynb
```

Output:

```text
reports/03_vectorization/
models/tfidf_vectorizer.pkl
data/X_train_tfidf.pkl
data/X_val_tfidf.pkl
data/X_test_tfidf.pkl
data/y_train.pkl
data/y_val.pkl
data/y_test.pkl
```

### 5.1 Split dữ liệu

| Split | Số mẫu |
|---|---:|
| Train | 27,055 |
| Validation | 5,798 |
| Test | 5,798 |

Tỷ lệ:

```text
70% train / 15% validation / 15% test
```

Có dùng `stratify=y`, nên tỷ lệ REAL/FAKE được giữ tương đối ổn định giữa các tập.

### 5.2 TF-IDF config đã thử

Yêu cầu ban đầu:

```text
max_features = 3000 / 5000 / 10000
ngram_range = (1,1) / (1,2)
```

Notebook hiện tại đã thử đủ các cấu hình này và lưu ở:

```text
reports/03_vectorization/tfidf_config_results.csv
```

Best config hiện tại:

| Tham số | Giá trị |
|---|---|
| `max_features` | 10,000 |
| `ngram_range` | `(1, 2)` |
| Validation accuracy khi chọn config | 0.9515 |
| Validation weighted F1 khi chọn config | 0.9516 |

### 5.3 Giải thích TF-IDF để viết báo cáo

TF-IDF là phương pháp biến văn bản thành vector số.

Ý tưởng:

- TF cao: từ xuất hiện nhiều trong một văn bản.
- IDF cao: từ hiếm trong toàn bộ corpus, có khả năng phân biệt tốt.
- TF-IDF cao: từ vừa quan trọng trong bài hiện tại, vừa không quá phổ biến ở mọi bài.

Công thức có thể trình bày:

```text
TF-IDF(t, d) = TF(t, d) * IDF(t)
IDF(t) = log(N / DF(t))
```

Trong đó:

- `t`: term/từ
- `d`: document/bài báo
- `N`: số document trong corpus
- `DF(t)`: số document chứa từ `t`

### 5.4 Vì sao fit vectorizer chỉ trên train?

Vectorizer chỉ được `fit` trên train để tránh data leakage. Nếu fit trên cả validation/test, model đã gián tiếp nhìn thấy vocabulary của dữ liệu đánh giá, làm kết quả không còn khách quan.

---

## 6. Model Training Hiện Tại

Tất cả model hiện đã train lại theo TF-IDF mới với 10,000 features.

Kiểm tra tương thích:

| Model | File | `n_features_in_` |
|---|---|---:|
| Naive Bayes | `models/naive_bayes_model.pkl` | 10,000 |
| Logistic Regression | `models/lr_model.pkl` | 10,000 |
| SVM | `models/svm_model.pkl` | 10,000 |
| Random Forest | `models/rf_model.pkl` | 10,000 |

---

## 7. Chi Tiết Từng Model

### 7.1 Naive Bayes

Notebook:

```text
notebooks/04_naive_bayes.ipynb
```

Output:

```text
reports/04_naive_bayes/
models/naive_bayes_model.pkl
```

GridSearchCV:

```python
alpha = [0.01, 0.1, 0.5, 1.0, 2.0]
fit_prior = [True, False]
cv = 5
scoring = "accuracy"
```

Kết quả:

| Metric | Giá trị |
|---|---:|
| Best params | `alpha=0.01, fit_prior=True` |
| Best CV accuracy | 0.9488 |
| Validation accuracy | 0.9533 |
| Validation weighted F1 | 0.9533 |
| Training time | 2.43s |

Confusion matrix validation:

```text
[[3029, 150],
 [ 121, 2498]]
```

Nhận xét để viết báo cáo:

- Naive Bayes train rất nhanh.
- Là baseline tốt cho text classification.
- Kết quả thấp hơn Logistic/SVM vì giả định các từ độc lập với nhau, chưa bắt tốt ngữ cảnh.
- `alpha=0.01` tốt nhất, tức smoothing nhỏ phù hợp với TF-IDF đã khá giàu thông tin.

### 7.2 Logistic Regression

Notebook:

```text
notebooks/05_logistic_regression.ipynb
```

Output:

```text
reports/05_logistic_regression/
models/lr_model.pkl
```

GridSearchCV đúng theo lộ trình:

```python
C = [0.1, 0.5, 1.0, 2.0, 5.0]
solver = ["lbfgs", "saga"]
max_iter = [500, 1000]
cv = 5
```

Kết quả:

| Metric | Giá trị |
|---|---:|
| Best params | `C=5.0, max_iter=500, solver=saga` |
| Best CV weighted F1 | 0.9850 |
| Validation accuracy | 0.9876 |
| Validation weighted F1 | 0.9876 |
| Training time | 10.56s |

Confusion matrix validation:

```text
[[3155, 24],
 [  48, 2571]]
```

Nhận xét để viết báo cáo:

- Logistic Regression rất phù hợp với TF-IDF vì dữ liệu text sau vector hóa thường phân tách tuyến tính tốt.
- `C=5.0` nghĩa là regularization nhẹ hơn so với `C=1`, giúp model fit tốt hơn.
- Hiệu năng rất gần SVM nhưng thời gian train lâu hơn Naive Bayes.

### 7.3 SVM

Notebook:

```text
notebooks/06_svm.ipynb
```

Output:

```text
reports/06_svm/
models/svm_model.pkl
```

Thử:

- `LinearSVC(C=[0.01, 0.1, 1, 10])`
- `SVC(kernel='rbf')` trên subsample với `C=[0.1, 1, 10]`, `gamma=['scale', 'auto']`

Kết quả:

| Metric | Giá trị |
|---|---:|
| Selected model | LinearSVC |
| Best params | `C=1` |
| Best CV weighted F1 | 0.9871 |
| Validation accuracy | 0.9886 |
| Validation weighted F1 | 0.9886 |
| Training time GridSearch | 4.91s |
| Best RBF subsample F1 | 0.9397 |

Confusion matrix validation:

```text
[[3153, 26],
 [  40, 2579]]
```

Nhận xét để viết báo cáo:

- LinearSVC là model tốt nhất trên validation và test.
- Text TF-IDF là dữ liệu sparse/high-dimensional, thường phù hợp với linear classifier.
- RBF kernel không hiệu quả bằng trên bài này vì tốn tính toán và không tận dụng tốt sparse TF-IDF.
- SVM cân bằng tốt giữa tốc độ và hiệu năng.

### 7.4 Random Forest

Notebook:

```text
notebooks/07_random_forest.ipynb
```

Output:

```text
reports/07_random_forest/
models/rf_model.pkl
```

GridSearchCV:

```python
n_estimators = [100, 200, 300]
max_depth = [None, 10, 20]
min_samples_split = [2, 5]
cv = 5
```

Kết quả:

| Metric | Giá trị |
|---|---:|
| Best params | `max_depth=None, min_samples_split=5, n_estimators=200` |
| Best CV weighted F1 | 0.9788 |
| Validation accuracy | 0.9778 |
| Validation weighted F1 | 0.9777 |
| Training time | 235.92s |

Confusion matrix validation:

```text
[[3142, 37],
 [  92, 2527]]
```

Nhận xét để viết báo cáo:

- Random Forest đạt kết quả tốt nhưng kém SVM/Logistic Regression.
- Thời gian train lâu nhất.
- Với TF-IDF 10,000 chiều, dữ liệu rất thưa, Random Forest không tận dụng sparse linear structure tốt như SVM/Logistic.
- Có thể dùng để so sánh với nhóm model tuyến tính.

---

## 8. So Sánh Model Trên Test Set

Notebook:

```text
notebooks/08_comparison.ipynb
```

Output:

```text
reports/08_comparison/
```

Test set chỉ được dùng ở bước này, đúng với lộ trình.

Kết quả:

| Model | Accuracy | Precision weighted | Recall weighted | F1 weighted | Training time |
|---|---:|---:|---:|---:|---:|
| SVM (LinearSVC) | 0.9865 | 0.9866 | 0.9865 | 0.9865 | 4.91s |
| Logistic Regression | 0.9834 | 0.9835 | 0.9834 | 0.9834 | 10.56s |
| Random Forest | 0.9741 | 0.9744 | 0.9741 | 0.9741 | 235.92s |
| Naive Bayes | 0.9502 | 0.9501 | 0.9502 | 0.9501 | 2.43s |

Best model:

```text
SVM (LinearSVC)
```

Nhận xét chính:

- SVM đạt F1 cao nhất.
- Logistic Regression bám sát SVM.
- Random Forest tốt nhưng chậm và không vượt được model tuyến tính.
- Naive Bayes nhanh nhất nhưng kết quả thấp nhất.
- Với dữ liệu TF-IDF, model tuyến tính là lựa chọn phù hợp nhất.

---

## 9. Đối Chiếu Lộ Trình Ban Đầu

### 9.1 Tuần 1 - Data, Preprocessing & Khung Báo Cáo

| Người | Yêu cầu | Trạng thái | Ghi chú |
|---|---|---|---|
| Người 1 - Hưng | `src/preprocessing.py` + `01_preprocessing.ipynb` | Đã làm | Pipeline chạy được, có Reuters fix, giữ negation words, có summary JSON |
| Người 2 - Xuân | `02_eda.ipynb` + EDA plots | Đã làm | Đã có plots/csv trong `reports/02_eda/` |
| Người 3 - Dung | Repo structure, README, requirements, `03_vectorization.ipynb`, vectorizer | Đã làm | Đã thử đủ TF-IDF configs và chọn best |
| Người 4 - Thủy | File Word/Google Docs khung báo cáo | Chưa thấy trong repo | Không có `.docx`, `.pdf`, hoặc Google Docs export |
| Báo cáo mục 1, 2.1, 2.2, 4 | Viết nội dung báo cáo | Chưa hoàn chỉnh | Có thể dùng file này làm nền để viết |

### 9.2 Tuần 2 - Model Training

| Người | Model | Trạng thái | Ghi chú |
|---|---|---|---|
| Người 1 | Naive Bayes | Đã làm | Đúng grid, có accuracy theo alpha |
| Người 2 | Logistic Regression | Đã làm | Đã sửa đúng grid theo lộ trình |
| Người 3 | SVM | Đã làm | Có LinearSVC, RBF subsample, chọn LinearSVC |
| Người 4 | Random Forest | Đã làm | Đúng grid, có log/metrics/confusion matrix |
| Báo cáo mục 2.3.x | Nội dung lý thuyết + kết quả từng model | Một phần | Có walkthrough cho NB/SVM/RF, thiếu report unified và LR walkthrough riêng |

### 9.3 Tuần 3 - So Sánh & Phân Tích

| Nhóm việc | Yêu cầu | Trạng thái | Ghi chú |
|---|---|---|---|
| Người 1 + 2 | `08_comparison.ipynb` | Đã làm | Đã so sánh 4 model trên test set |
| Người 1 + 2 | Bảng Accuracy/Precision/Recall/F1/Training time | Đã làm | `reports/08_comparison/test_metrics.csv` |
| Người 1 + 2 | Bar chart F1 | Đã làm | `reports/08_comparison/comparison_f1_score.png` |
| Người 3 + 4 | `09_feature_error_analysis.ipynb` | Chưa làm | Notebook mới có markdown tiêu đề |
| Feature analysis | Top 20 từ quan trọng cho FAKE/REAL | Chưa làm | Có thể dùng LR coefficients hoặc NB log probabilities |
| Error analysis | 10-15 sample predict sai của best model | Chưa làm | Cần raw text/test index để phân tích tốt hơn |
| Báo cáo mục 3 | Kết quả + feature/error analysis | Một phần | So sánh model đã có, feature/error chưa có |

### 9.4 Tuần 4 - Hoàn Thiện Báo Cáo

| Yêu cầu | Trạng thái | Ghi chú |
|---|---|---|
| Merge draft từng người | Chưa làm | Chưa thấy file báo cáo hoàn chỉnh |
| Cover page HUST | Chưa làm | Cần Word/Google Docs |
| Mục lục tự động | Chưa làm | Cần Word/Google Docs |
| Đánh số trang, căn lề, font chuẩn | Chưa làm | Cần xử lý trong bản báo cáo |
| Kết luận + hướng phát triển | Chưa làm | Có thể viết dựa trên kết quả SVM best |
| Xuất PDF | Chưa làm | Chưa thấy file PDF |

---

## 10. Những Gì Đã Làm Được

### Code/pipeline

- Có hàm `preprocess_text()` tái sử dụng.
- Có notebook preprocessing chạy tuần tự.
- Có EDA đầy đủ theo yêu cầu.
- Có TF-IDF split 70/15/15.
- Có thử nghiệm TF-IDF config.
- Có lưu vectorizer.
- Có train đủ 4 model.
- Có GridSearchCV đúng yêu cầu cho từng model.
- Có lưu model bằng `joblib.dump`.
- Có logs/metrics/plots tách folder rõ.
- Có comparison trên test set.

### Kết quả thực nghiệm

- TF-IDF best config: `max_features=10000`, `ngram_range=(1,2)`.
- SVM là model tốt nhất.
- Logistic Regression đứng thứ hai và rất sát SVM.
- Naive Bayes nhanh nhưng kém hơn.
- Random Forest chậm nhất, hiệu năng thấp hơn SVM/Logistic.

---

## 11. Những Gì Chưa Làm Được / Cần Làm Tiếp

Ưu tiên cao:

1. Hoàn thiện `notebooks/09_feature_error_analysis.ipynb`.
2. Tạo feature importance:
   - Top 20 features cho FAKE/REAL bằng Logistic Regression coefficients.
   - Hoặc dùng Naive Bayes log probabilities.
3. Làm error analysis:
   - lấy các mẫu test bị SVM dự đoán sai
   - in raw/processed text
   - nhận xét pattern lỗi
4. Tạo file báo cáo Word/Google Docs/PDF.
5. Viết mục lý thuyết Logistic Regression riêng, vì hiện chưa có walkthrough `.md` riêng cho LR.
6. Chuẩn hóa lại toàn bộ nội dung report theo một format: bảng tham số, bảng kết quả, confusion matrix, caption, nhận xét.

Ưu tiên vừa:

1. Lưu thêm raw test split hoặc test dataframe để error analysis dễ đọc raw text hơn.
2. Cân nhắc Git LFS cho `models/rf_model.pkl` vì file lớn hơn 50MB.
3. Dọn các artefact WELFake nếu không thuộc phạm vi báo cáo chính.

---

## 12. Định Hướng Viết Báo Cáo Chi Tiết

### Mục 1 - Tổng quan

Nên viết:

- Bối cảnh fake news.
- Tác hại của tin giả.
- Vì sao cần phát hiện tự động.
- Mục tiêu project:
  - phân loại FAKE/REAL
  - input là text bài báo
  - so sánh 4 thuật toán ML
- Ý nghĩa thực tiễn.

Metrics cần giải thích:

| Metric | Ý nghĩa | Vì sao dùng |
|---|---|---|
| Accuracy | Tỷ lệ dự đoán đúng | Dễ hiểu, phù hợp khi class không quá lệch |
| Precision | Trong các mẫu đoán FAKE/REAL, bao nhiêu mẫu đúng | Quan trọng khi muốn giảm cảnh báo sai |
| Recall | Trong các mẫu thật sự thuộc một class, model bắt được bao nhiêu | Quan trọng khi không muốn bỏ sót tin giả |
| F1-score | Trung hòa precision và recall | Hữu ích khi cần đánh giá cân bằng |
| Confusion matrix | Cho biết sai ở đâu | Dễ phân tích false positive/false negative |

### Mục 2.1 - Tập dữ liệu

Nên đưa:

- ISOT dataset.
- Nguồn real: Reuters.
- Nguồn fake: unreliable websites.
- Số lượng ban đầu và sau xử lý.
- Features: `title`, `text`, `subject`, `date`.
- Ngôn ngữ: English.
- Biểu đồ phân phối nhãn.
- Histogram độ dài.
- Word clouds.
- Top n-grams.

### Mục 2.2 - Vectorization

Nên đưa:

- Công thức TF-IDF.
- Giải thích TF, IDF.
- Lý do dùng TF-IDF thay vì Bag-of-Words thuần.
- Quy trình split 70/15/15.
- Nhấn mạnh fit vectorizer chỉ trên train.
- Bảng thử nghiệm TF-IDF configs.
- Lý do chọn `max_features=10000`, `ngram_range=(1,2)`.

### Mục 2.3 - Các thuật toán

Mỗi model nên có cùng format:

```text
1. Lý thuyết ngắn
2. Hyperparameters đã tune
3. Best params
4. Classification report
5. Confusion matrix
6. Nhận xét
```

#### Naive Bayes

Nội dung lý thuyết:

- Định lý Bayes.
- Multinomial Naive Bayes.
- Giả định độc lập giữa các từ.
- Laplace smoothing.

Nhận xét:

- Nhanh, baseline tốt.
- Kém hơn model tuyến tính vì giả định độc lập quá mạnh.

#### Logistic Regression

Nội dung lý thuyết:

- Sigmoid function.
- Xác suất class.
- Binary cross-entropy.
- Regularization và tham số `C`.

Nhận xét:

- Rất mạnh với TF-IDF.
- Hiệu năng gần SVM.
- Dễ giải thích bằng trọng số feature.

#### SVM

Nội dung lý thuyết:

- Hyperplane.
- Margin.
- Support vectors.
- Kernel trick.
- Vì sao LinearSVC phù hợp với TF-IDF.

Nhận xét:

- Tốt nhất trong 4 model.
- LinearSVC nhanh và hiệu quả hơn RBF trong bối cảnh sparse text.

#### Random Forest

Nội dung lý thuyết:

- Decision tree.
- Bagging.
- Random feature selection.
- Majority voting.

Nhận xét:

- Train lâu.
- Không tận dụng tốt sparse high-dimensional TF-IDF bằng model tuyến tính.

### Mục 3 - Kết quả thu được

Nên đưa:

- Bảng test metrics từ `reports/08_comparison/test_metrics.csv`.
- Bar chart F1 từ `reports/08_comparison/comparison_f1_score.png`.
- Nhận xét model tốt nhất.
- Phân tích trade-off:
  - Naive Bayes nhanh nhất nhưng thấp nhất.
  - SVM tốt nhất và vẫn nhanh.
  - Logistic rất mạnh nhưng chậm hơn SVM.
  - Random Forest chậm và không vượt được linear models.

Sau khi hoàn thiện notebook 09, bổ sung:

- Top feature FAKE/REAL.
- Các sample bị dự đoán sai.
- Nhận xét lỗi.

### Mục 4 - Hướng dẫn chạy hệ thống

Nên lấy từ README và bổ sung:

```bash
git clone <repo>
cd Fake-News-Detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Thứ tự chạy:

```text
01_preprocessing.ipynb
02_eda.ipynb
03_vectorization.ipynb
04_naive_bayes.ipynb
05_logistic_regression.ipynb
06_svm.ipynb
07_random_forest.ipynb
08_comparison.ipynb
09_feature_error_analysis.ipynb
```

### Mục 5 - Kết luận và hướng phát triển

Kết luận nên viết:

- TF-IDF + LinearSVC là lựa chọn tốt nhất trong project.
- Các model tuyến tính phù hợp với text classification.
- Preprocessing đúng giúp giảm leakage và giữ ý nghĩa phủ định.
- Test F1 của SVM đạt khoảng 0.9865.

Hướng phát triển:

- Dùng thêm dataset khác để kiểm tra generalization.
- Thử transformer models như BERT.
- Thử giải thích mô hình bằng SHAP/LIME.
- Xây dựng web demo.
- Cập nhật dữ liệu mới theo thời gian thực.

---

## 13. Kết Luận Tiến Độ

Nếu chỉ xét phần code chính từ Tuần 1 đến Tuần 3:

```text
Đã hoàn thành khoảng 80-85%.
```

Đã xong:

- preprocessing
- EDA
- vectorization
- 4 model training
- comparison trên test set
- logs/results folder rõ ràng

Chưa xong:

- feature/error analysis
- báo cáo Word/PDF hoàn chỉnh
- phần polish cuối cùng cho report

Việc nên làm tiếp ngay:

```text
Hoàn thiện notebooks/09_feature_error_analysis.ipynb
```

Sau đó có thể bắt đầu merge nội dung thành báo cáo cuối.
