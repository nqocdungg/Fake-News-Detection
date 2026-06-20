# Naive Bayes Walkthrough — `notebooks/04_naive_bayes.ipynb`

> **Dự án:** Fake News Detection  
> **Notebook:** `notebooks/04_naive_bayes.ipynb`  
> **Model output:** `models/naive_bayes_model.pkl`  
> **Mục tiêu của phần này:** dùng dữ liệu TF-IDF đã tạo ở Tuần 1 để huấn luyện, tune và đánh giá mô hình Multinomial Naive Bayes.

---

## 1. Ý tưởng tổng thể của bài

Bài toán là **phân loại văn bản nhị phân**:

| Nhãn | Ý nghĩa |
|---|---|
| `0` | REAL - tin thật |
| `1` | FAKE - tin giả |

Pipeline chính của project:

```text
Tin tức thô
  -> Tiền xử lý văn bản
  -> Chia train / validation / test
  -> TF-IDF vectorization
  -> Huấn luyện nhiều mô hình ML
  -> Đánh giá validation
  -> So sánh cuối cùng trên test set
```

Notebook `04_naive_bayes.ipynb` nằm ở bước **huấn luyện mô hình đầu tiên**. Nó không xử lý text thô nữa, mà dùng trực tiếp các file `.pkl` đã được tạo từ các notebook trước.

---

## 2. Các phần trước làm gì để dẫn đến notebook 04?

### 2.1 `01_preprocessing.ipynb` - tiền xử lý dữ liệu

Mục đích:

- Tải / đọc dataset ISOT gồm tin thật và tin giả.
- Gán nhãn:
  - `REAL = 0`
  - `FAKE = 1`
- Gộp `title` và `text` thành một trường văn bản đầy đủ.
- Xóa dữ liệu trùng, dữ liệu rỗng.
- Làm sạch text:
  - lowercase
  - xóa URL, email, số, dấu câu, ký tự đặc biệt
  - xóa dấu hiệu nguồn như `Reuters` để giảm leakage
  - xóa stopwords nhưng giữ các từ phủ định như `no`, `not`, `never`
  - lemmatization
- Kết quả mong muốn: một file dữ liệu đã xử lý, ví dụ `data/preprocessed_isot_full.csv`, có ít nhất:
  - `processed_text`
  - `label`

Ý nghĩa với notebook 04: nếu preprocessing sai, TF-IDF và model sau đó sẽ học trên dữ liệu nhiễu hoặc bị leakage.

### 2.2 `02_eda.ipynb` - phân tích dữ liệu

Mục đích:

- Kiểm tra phân phối nhãn REAL / FAKE.
- Xem độ dài bài viết.
- Xem các từ / cụm từ phổ biến.
- Phát hiện bất thường trong dữ liệu.

Ý nghĩa với notebook 04: EDA giúp hiểu dữ liệu có cân bằng không, text có quá ngắn / quá dài không, và có nên dùng unigram + bigram trong TF-IDF không.

### 2.3 `03_vectorization.ipynb` - tạo đặc trưng TF-IDF

Mục đích:

- Đọc `processed_text` và `label`.
- Chia dữ liệu theo tỷ lệ:
  - train: 70%
  - validation: 15%
  - test: 15%
- Dùng `stratify=y` để giữ tỷ lệ nhãn ở các split.
- Fit `TfidfVectorizer` **chỉ trên train set** để tránh data leakage.
- Transform validation và test bằng vectorizer đã fit từ train.
- Lưu output cho notebook 04:

```text
data/X_train_tfidf.pkl
data/X_val_tfidf.pkl
data/y_train.pkl
data/y_val.pkl
```

Ý nghĩa với notebook 04: Naive Bayes không nhận text thô trực tiếp, mà nhận ma trận số TF-IDF. Mỗi dòng là một bài báo, mỗi cột là một từ / cụm từ, mỗi giá trị là trọng số TF-IDF.

---

## 3. Notebook 04 đang làm gì?

### Bước 1 - Import thư viện

Notebook import:

- `joblib`: load dữ liệu `.pkl` và lưu model.
- `time`: đo thời gian huấn luyện.
- `numpy`, `pandas`: xử lý mảng và bảng kết quả.
- `matplotlib`, `seaborn`: vẽ biểu đồ.
- `MultinomialNB`: mô hình Naive Bayes phù hợp với dữ liệu dạng tần suất / TF-IDF không âm.
- `GridSearchCV`: thử nhiều bộ hyperparameter bằng cross-validation.
- `classification_report`, `confusion_matrix`: đánh giá mô hình.

### Bước 2 - Load dữ liệu từ Tuần 1

Notebook load:

```python
X_train = joblib.load('../data/X_train_tfidf.pkl')
X_val = joblib.load('../data/X_val_tfidf.pkl')
y_train = joblib.load('../data/y_train.pkl')
y_val = joblib.load('../data/y_val.pkl')
```

Kết quả kiểm tra thực tế trong repo:

| Biến | Shape |
|---|---:|
| `X_train` | `(27057, 5000)` |
| `X_val` | `(5798, 5000)` |
| `y_train` | `(27057,)` |
| `y_val` | `(5798,)` |

Nghĩa là train có 27,057 bài, validation có 5,798 bài, mỗi bài được biểu diễn bằng 5,000 feature TF-IDF.

Notebook đang dùng:

```text
DATA_DIR = ../data
```

Vì vậy model đang train trên bộ TF-IDF chính của Tuần 1/ISOT đã lưu trong thư mục `data/`, **không phải** bộ WELFake trong `data/welfake/`. Đây là dữ liệu đã vector hóa thật của project, không phải dữ liệu demo.

Phân phối nhãn kiểm tra được:

| Split | Nhãn `0` | Nhãn `1` |
|---|---:|---:|
| Train | 14,837 | 12,220 |
| Validation | 3,179 | 2,619 |

Theo quy ước trong project: `REAL=0`, `FAKE=1`.

### Bước 3 - GridSearchCV cho MultinomialNB

Yêu cầu đề bài:

```python
alpha = [0.01, 0.1, 0.5, 1.0, 2.0]
fit_prior = [True, False]
cv = 5
```

Notebook hiện tại đã khai báo đúng:

```python
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}
```

Ý nghĩa tham số:

| Tham số | Ý nghĩa |
|---|---|
| `alpha` | Hệ số smoothing. Giúp tránh xác suất bằng 0 khi một từ chưa xuất hiện trong class nào đó. |
| `fit_prior=True` | Model học xác suất xuất hiện ban đầu của mỗi class từ dữ liệu train. |
| `fit_prior=False` | Model giả định các class có prior bằng nhau. |
| `cv=5` | Chia train set thành 5 phần, train/validate luân phiên để chọn tham số ổn định hơn. |

### Bước 4 - Đánh giá trên validation set

Sau khi GridSearchCV xong:

```python
best_nb_model = grid_search.best_estimator_
y_pred = best_nb_model.predict(X_val)
```

Notebook in:

- `classification_report`
- accuracy
- weighted F1-score
- confusion matrix

Kết quả kiểm tra lại bằng script tương đương:

| Metric | Giá trị |
|---|---:|
| Best params | `{'alpha': 0.01, 'fit_prior': True}` |
| Best CV Accuracy | `0.942898` |
| Validation accuracy | `0.941532` |
| Validation F1 weighted | `0.941562` |
| Training time kiểm tra lại | khoảng `0.77s` với `n_jobs=1` |

Confusion matrix:

```text
[[2993  186]
 [ 153 2466]]
```

Diễn giải theo nhãn `REAL=0`, `FAKE=1`:

| | Predict REAL | Predict FAKE |
|---|---:|---:|
| Actual REAL | 2993 | 186 |
| Actual FAKE | 153 | 2466 |

### Bước 5 - Lưu model

Notebook lưu:

```python
joblib.dump(best_nb_model, '../models/naive_bayes_model.pkl')
```

File hiện đã tồn tại:

```text
models/naive_bayes_model.pkl
```

### Bước 6 - Vẽ biểu đồ theo `alpha`

Notebook hiện tại lấy `cv_results_` từ GridSearchCV và vẽ score theo `alpha`, tách 2 đường:

- `fit_prior=True`
- `fit_prior=False`

File output hiện đã tồn tại:

```text
reports/naive_bayes_alpha_tuning.png
```

---

## 4. Audit code hiện tại: đã đúng 100% chưa?

Kết luận ngắn: **logic chính của notebook 04 là đúng và đã được chỉnh để khớp yêu cầu "accuracy theo alpha"**.

| Hạng mục yêu cầu | Trạng thái | Nhận xét |
|---|---|---|
| Load `X_train_tfidf`, `X_val_tfidf`, `y_train`, `y_val` | Đúng | File tồn tại và load được. |
| Dùng `MultinomialNB` | Đúng | Phù hợp với TF-IDF không âm. |
| GridSearchCV trên `alpha` đúng list yêu cầu | Đúng | `[0.01, 0.1, 0.5, 1.0, 2.0]`. |
| GridSearchCV trên `fit_prior` đúng list yêu cầu | Đúng | `[True, False]`. |
| `cv=5` | Đúng | Đã dùng 5-fold CV. |
| Predict trên validation set | Đúng | Dùng `best_estimator_`. |
| In classification report | Đúng | Có `classification_report`. |
| In / vẽ confusion matrix | Đúng | Có confusion matrix trên validation set. |
| Đo training time | Đúng | Đo thời gian chạy GridSearchCV. Nên ghi rõ đây là thời gian tuning + training, không chỉ final fit. |
| Lưu model bằng `joblib.dump` | Đúng | Đã lưu `models/naive_bayes_model.pkl`. |
| Vẽ đồ thị accuracy theo alpha | Đúng | Notebook đã dùng `scoring='accuracy'`, nên `mean_test_score` trong biểu đồ là Mean CV Accuracy. |
| Đường dẫn chạy notebook | Cần lưu ý | Code dùng `../data`, `../models`, `../reports`, nên đúng khi kernel chạy từ thư mục `notebooks/`. Nếu chạy từ root repo thì đường dẫn sẽ sai. |
| `n_jobs=-1` | Chấp nhận được | Trên máy local thường ổn. Trong sandbox Codex bị lỗi quyền joblib worker, nên lúc kiểm tra phải đổi tạm thành `n_jobs=1`. |

---

## 5. Việc đã sửa để notebook 04 khớp đề bài hơn

### Đã đổi GridSearchCV sang accuracy

Notebook đã được đổi từ `scoring='f1_weighted'` sang:

```python
grid_search = GridSearchCV(
    estimator=nb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
```

Nhãn biểu đồ cũng đã được đổi sang:

```python
plt.title('Naive Bayes — Accuracy theo tham số alpha', fontsize=12)
plt.ylabel('Mean CV Accuracy', fontsize=11)
```

Phần báo cáo F1-score trên validation set vẫn được giữ lại để có thêm metric so sánh.

### Việc nên làm để báo cáo đẹp hơn

Nên lưu thêm confusion matrix thành file ảnh:

```python
plt.savefig('../reports/naive_bayes_confusion_matrix.png', dpi=150, bbox_inches='tight')
```

Nên lưu bảng kết quả GridSearchCV để dễ đưa vào báo cáo:

```python
cv_results.to_csv('../reports/naive_bayes_gridsearch_results.csv', index=False)
```

Nên in rõ training time:

```python
print(f"GridSearchCV training/tuning time: {training_time:.2f} seconds")
```

---

## 6. Checklist cho phần Người 1 - Naive Bayes

### Đã hoàn thành

- [x] Có notebook `notebooks/04_naive_bayes.ipynb`.
- [x] Load đúng TF-IDF train / validation.
- [x] Dùng `MultinomialNB`.
- [x] Dùng `GridSearchCV`.
- [x] Search đúng `alpha`.
- [x] Search đúng `fit_prior`.
- [x] Dùng `cv=5`.
- [x] Predict trên validation set.
- [x] Có classification report.
- [x] Có confusion matrix.
- [x] Có đo training time.
- [x] Có lưu model `models/naive_bayes_model.pkl`.
- [x] Có lưu biểu đồ `reports/naive_bayes_alpha_tuning.png`.

### Cần làm thêm nếu muốn báo cáo đẹp hơn

- [ ] Lưu thêm confusion matrix image nếu muốn đưa vào báo cáo.
- [ ] Kiểm tra notebook 03 có tạo đủ `models/tfidf_vectorizer.pkl` nếu các phần sau cần phân tích feature.
- [ ] Chuẩn hóa output notebook 01/03 để khi chạy lại từ đầu vẫn tạo đúng file trong `data/`.

---

## 7. Lưu ý về các phần trước

Trong repo hiện tại đã có các file TF-IDF `.pkl`, nên notebook 04 có thể chạy được.

Tuy nhiên, nếu muốn chạy lại toàn bộ pipeline từ notebook 01 đến 04, cần kiểm tra lại:

- `03_vectorization.ipynb` đang đọc `../data/preprocessed_isot_full.csv`, nhưng trong `data/` hiện không thấy file CSV này.
- `models/tfidf_vectorizer.pkl` chưa thấy trong thư mục `models/`, trong khi một số notebook sau có thể cần file này.
- `01_preprocessing.ipynb` có dấu hiệu là notebook thử nghiệm nhiều bước, cần đảm bảo bản cuối cùng chạy tuần tự từ đầu đến cuối và lưu output đúng vào `data/`.

Nói cách khác: **notebook 04 hiện ổn với dữ liệu `.pkl` có sẵn**, nhưng pipeline từ đầu cần được dọn lại để người khác clone repo và chạy tuần tự không bị thiếu file.

---

## 8. Tóm tắt để trình bày

Phần Naive Bayes sử dụng dữ liệu đã được biểu diễn bằng TF-IDF từ Tuần 1. Mỗi bài báo được chuyển thành vector 5,000 chiều, sau đó `MultinomialNB` học xác suất từ / cụm từ xuất hiện trong từng class REAL hoặc FAKE. `GridSearchCV` được dùng để thử nhiều giá trị `alpha` và `fit_prior`, chọn mô hình tốt nhất qua 5-fold cross-validation. Mô hình tốt nhất sau đó được đánh giá trên validation set bằng classification report, confusion matrix, accuracy và F1-score, rồi lưu lại bằng `joblib.dump` để dùng trong bước so sánh các mô hình.

Kết quả kiểm tra hiện tại cho thấy `alpha=0.01`, `fit_prior=True` là cấu hình tốt nhất, validation accuracy khoảng `94.15%`. Notebook đã được chỉnh để GridSearchCV và biểu đồ tuning dùng **accuracy theo alpha**, đúng với yêu cầu.
