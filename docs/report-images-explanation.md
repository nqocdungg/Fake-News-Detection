# Report Images Explanation

File này mô tả các hình ảnh hiện có trong thư mục `reports/`, tên ảnh, nên đặt ở mục nào trong báo cáo, và ảnh đó chứng minh/giải thích vấn đề gì.

---

## 1. Ảnh EDA ISOT

### `reports/02_eda/label_distribution_bar_pie.png`

**Nên dùng ở:** Mục 2.1 - Tập dữ liệu.

**Nội dung ảnh:** Biểu đồ cột và biểu đồ tròn thể hiện số lượng/tỷ lệ tin thật và tin giả trong dataset ISOT sau preprocessing.

**Chứng minh điều gì:**

- Dataset có đủ hai lớp `REAL` và `FAKE`.
- Phân phối nhãn không bị lệch nghiêm trọng.
- Có thể dùng các metric như accuracy, precision, recall, F1 để đánh giá tương đối công bằng.

**Nhận xét gợi ý:** Tập dữ liệu sau xử lý vẫn giữ được số lượng mẫu lớn ở cả hai class, giúp mô hình học được đặc trưng của cả tin thật và tin giả.

---

### `reports/02_eda/text_length_histogram.png`

**Nên dùng ở:** Mục 2.1 - Tập dữ liệu.

**Nội dung ảnh:** Histogram số lượng từ sau preprocessing, tách theo nhãn `REAL` và `FAKE`.

**Chứng minh điều gì:**

- Độ dài văn bản giữa các bài báo không đồng đều.
- Có thể quan sát sự khác biệt về phân phối độ dài giữa tin thật và tin giả.
- Độ dài văn bản là một đặc điểm cần chú ý khi phân tích dữ liệu text.

**Nhận xét gợi ý:** Nếu hai class có phân phối độ dài khác nhau, model có thể gián tiếp học một phần tín hiệu từ độ dài văn bản. Vì vậy cần dùng thêm TF-IDF để tập trung vào nội dung từ vựng, không chỉ độ dài.

---

### `reports/02_eda/wordcloud_real.png`

**Nên dùng ở:** Mục 2.1 - Tập dữ liệu.

**Nội dung ảnh:** Word cloud các từ xuất hiện nổi bật trong class `REAL`.

**Chứng minh điều gì:**

- Tin thật có một số nhóm từ/cụm từ xuất hiện thường xuyên.
- Có thể nhận diện sơ bộ domain/ngữ cảnh của tin thật trong ISOT.
- Cho thấy dữ liệu đã được preprocessing thành token dễ phân tích hơn.

**Nhận xét gợi ý:** Word cloud giúp nhìn nhanh các từ phổ biến, nhưng không nên dùng nó làm bằng chứng định lượng chính vì word cloud chỉ mang tính trực quan.

---

### `reports/02_eda/wordcloud_fake.png`

**Nên dùng ở:** Mục 2.1 - Tập dữ liệu.

**Nội dung ảnh:** Word cloud các từ xuất hiện nổi bật trong class `FAKE`.

**Chứng minh điều gì:**

- Tin giả có các từ/cụm từ nổi bật riêng.
- Có sự khác biệt từ vựng giữa hai class.
- Điều này hỗ trợ lý do dùng mô hình dựa trên đặc trưng văn bản như TF-IDF.

**Nhận xét gợi ý:** Nếu word cloud của FAKE khác REAL, điều đó gợi ý rằng phân loại bằng đặc trưng từ vựng là khả thi.

---

### `reports/02_eda/top_unigrams_bigrams.png`

**Nên dùng ở:** Mục 2.1 - Tập dữ liệu hoặc Mục 3 - Phân tích kết quả.

**Nội dung ảnh:** Top 20 unigram và bigram xuất hiện nhiều nhất trong từng class.

**Chứng minh điều gì:**

- Không chỉ từ đơn, mà cả cụm 2 từ cũng mang thông tin phân biệt.
- Đây là cơ sở để chọn `ngram_range=(1,2)` trong TF-IDF.
- Cho thấy một số pattern ngôn ngữ khác nhau giữa `REAL` và `FAKE`.

**Nhận xét gợi ý:** Bigram giúp giữ lại một phần ngữ cảnh, ví dụ các cụm từ mang nghĩa rõ hơn từ đơn. Vì vậy việc thử `ngram_range=(1,2)` là hợp lý.

---

## 2. Ảnh Vectorization

### `reports/03_vectorization/tfidf_config_accuracy.png`

**Nên dùng ở:** Mục 2.2 - Vectorization (TF-IDF).

**Nội dung ảnh:** So sánh validation accuracy giữa các cấu hình TF-IDF khác nhau.

Các cấu hình đã thử:

- `max_features=3000`, `ngram_range=(1,1)`
- `max_features=3000`, `ngram_range=(1,2)`
- `max_features=5000`, `ngram_range=(1,1)`
- `max_features=5000`, `ngram_range=(1,2)`
- `max_features=10000`, `ngram_range=(1,1)`
- `max_features=10000`, `ngram_range=(1,2)`

**Chứng minh điều gì:**

- Nhóm không chọn TF-IDF config theo cảm tính mà có thử nghiệm.
- Cấu hình `max_features=10000`, `ngram_range=(1,2)` cho kết quả tốt nhất trên validation.
- Bigram và số lượng feature lớn hơn giúp model có thêm thông tin phân biệt.

**Nhận xét gợi ý:** Việc chọn 10,000 features và unigram+bigram giúp biểu diễn văn bản giàu thông tin hơn, cải thiện kết quả so với cấu hình ít feature hơn.

---

## 3. Ảnh Naive Bayes

### `reports/04_naive_bayes/alpha_accuracy.png`

**Nên dùng ở:** Mục 2.3.1 - Naive Bayes.

**Nội dung ảnh:** Accuracy trung bình qua 5-fold Cross Validation theo từng giá trị `alpha`, tách hai trường hợp `fit_prior=True` và `fit_prior=False`.

**Chứng minh điều gì:**

- Có tune hyperparameter `alpha` và `fit_prior`.
- `alpha=0.01`, `fit_prior=True` là cấu hình tốt nhất.
- Khi `alpha` tăng, accuracy có xu hướng giảm nhẹ, cho thấy smoothing quá mạnh có thể làm mất tín hiệu từ đặc trưng TF-IDF.

**Nhận xét gợi ý:** Naive Bayes nhạy với tham số smoothing. Với dữ liệu đã preprocessing và TF-IDF, smoothing nhỏ cho kết quả tốt hơn.

---

### `reports/04_naive_bayes/confusion_matrix.png`

**Nên dùng ở:** Mục 2.3.1 - Naive Bayes.

**Nội dung ảnh:** Ma trận nhầm lẫn của Naive Bayes trên validation set.

**Chứng minh điều gì:**

- Model không chỉ có accuracy tổng quát mà còn có thể xem sai ở class nào.
- Có thể phân tích số mẫu REAL bị dự đoán nhầm thành FAKE và FAKE bị dự đoán nhầm thành REAL.
- Giúp đánh giá chất lượng model chi tiết hơn một con số accuracy.

**Nhận xét gợi ý:** Naive Bayes có tốc độ nhanh nhưng số lỗi vẫn nhiều hơn Logistic Regression và SVM, phù hợp vai trò baseline.

---

### `reports/naive_bayes_alpha_tuning.png`

**Ghi chú:** Đây là ảnh compatibility ở cấp `reports/`, nội dung tương đương `reports/04_naive_bayes/alpha_accuracy.png`.

**Nên dùng:** Ưu tiên dùng ảnh trong folder `reports/04_naive_bayes/` để báo cáo có cấu trúc rõ ràng.

---

## 4. Ảnh Logistic Regression

### `reports/05_logistic_regression/top_gridsearch_configs.png`

**Nên dùng ở:** Mục 2.3.2 - Logistic Regression.

**Nội dung ảnh:** Top các cấu hình GridSearchCV tốt nhất của Logistic Regression theo mean CV weighted F1.

**Chứng minh điều gì:**

- Logistic Regression đã được tune theo nhiều giá trị `C`, `solver`, `max_iter`.
- Cấu hình tốt nhất là `C=5.0`, `solver='saga'`, `max_iter=500`.
- Regularization nhẹ hơn có thể giúp model fit tốt hơn trên TF-IDF.

**Nhận xét gợi ý:** Logistic Regression đạt hiệu năng rất cao vì dữ liệu TF-IDF thường phân tách tuyến tính tốt.

---

### `reports/05_logistic_regression/confusion_matrix.png`

**Nên dùng ở:** Mục 2.3.2 - Logistic Regression.

**Nội dung ảnh:** Ma trận nhầm lẫn của Logistic Regression trên validation set.

**Chứng minh điều gì:**

- Logistic Regression giảm số lỗi so với Naive Bayes.
- Model phân loại tốt cả hai class.
- Đây là bằng chứng trực quan cho kết quả validation accuracy/F1 cao.

**Nhận xét gợi ý:** Số lượng dự đoán sai thấp cho thấy Logistic Regression là model mạnh cho bài toán text classification với TF-IDF.

---

### `reports/lr_evaluation_plots.png`

**Ghi chú:** Đây là ảnh compatibility ở cấp `reports/`, hiện tương ứng với biểu đồ top GridSearch config của Logistic Regression.

**Nên dùng:** Ưu tiên dùng `reports/05_logistic_regression/top_gridsearch_configs.png`.

---

## 5. Ảnh SVM

### `reports/06_svm/linearsvc_c_tuning.png`

**Nên dùng ở:** Mục 2.3.3 - SVM.

**Nội dung ảnh:** Mean CV weighted F1 của LinearSVC theo từng giá trị `C`.

**Chứng minh điều gì:**

- Có tune tham số `C` của LinearSVC.
- `C=1` là cấu hình tốt nhất.
- Nếu `C` quá nhỏ có thể underfit, nếu quá lớn có thể không cải thiện thêm hoặc dễ overfit.

**Nhận xét gợi ý:** `C=1` cân bằng tốt giữa margin rộng và lỗi phân loại, phù hợp với dữ liệu TF-IDF.

---

### `reports/06_svm/kernel_comparison.png`

**Nên dùng ở:** Mục 2.3.3 - SVM.

**Nội dung ảnh:** So sánh F1 giữa LinearSVC full dataset và SVC RBF trên subsample.

**Chứng minh điều gì:**

- LinearSVC tốt hơn và phù hợp hơn với dữ liệu TF-IDF sparse.
- RBF kernel không phải lựa chọn tối ưu trong bối cảnh dữ liệu nhiều chiều và thưa.
- Lý do chọn LinearSVC là có căn cứ thực nghiệm.

**Nhận xét gợi ý:** Text classification với TF-IDF thường có ranh giới tuyến tính rõ, nên LinearSVC vừa nhanh vừa chính xác.

---

### `reports/06_svm/confusion_matrix.png`

**Nên dùng ở:** Mục 2.3.3 - SVM.

**Nội dung ảnh:** Ma trận nhầm lẫn của LinearSVC trên validation set.

**Chứng minh điều gì:**

- SVM có số lỗi rất thấp trên validation.
- Đây là model có kết quả tốt nhất trong nhóm.
- Confusion matrix giúp chứng minh model không chỉ tốt về F1 mà còn cân bằng giữa hai class.

**Nhận xét gợi ý:** SVM đạt hiệu quả cao nhờ khả năng tìm hyperplane phân tách tốt trong không gian TF-IDF 10,000 chiều.

---

### `reports/svm_linearsvc_c_tuning.png`, `reports/svm_kernel_comparison.png`, `reports/svm_confusion_matrix.png`

**Ghi chú:** Đây là các ảnh compatibility ở cấp `reports/`, nội dung tương đương ảnh trong `reports/06_svm/`.

**Nên dùng:** Ưu tiên dùng ảnh trong `reports/06_svm/`.

---

## 6. Ảnh Random Forest

### `reports/07_random_forest/top_gridsearch_configs.png`

**Nên dùng ở:** Mục 2.3.4 - Random Forest.

**Nội dung ảnh:** Top cấu hình GridSearchCV tốt nhất của Random Forest.

**Chứng minh điều gì:**

- Random Forest đã được tune theo `n_estimators`, `max_depth`, `min_samples_split`.
- Cấu hình tốt nhất hiện tại là `max_depth=None`, `min_samples_split=5`, `n_estimators=200`.
- Model cây cần nhiều thời gian train hơn các model tuyến tính.

**Nhận xét gợi ý:** Random Forest có thể học quan hệ phi tuyến, nhưng với TF-IDF sparse high-dimensional thì không hiệu quả bằng LinearSVC/Logistic Regression.

---

### `reports/07_random_forest/confusion_matrix.png`

**Nên dùng ở:** Mục 2.3.4 - Random Forest.

**Nội dung ảnh:** Ma trận nhầm lẫn của Random Forest trên validation set.

**Chứng minh điều gì:**

- Random Forest phân loại tương đối tốt nhưng vẫn kém SVM/Logistic Regression.
- Có thể nhìn trực tiếp số lỗi ở từng class.
- Dùng để so sánh với confusion matrix của các model khác.

**Nhận xét gợi ý:** Random Forest có kết quả khá tốt nhưng thời gian huấn luyện dài, không phải lựa chọn tối ưu nhất cho bài toán này.

---

### `reports/rf_confusion_matrix.png`

**Ghi chú:** Đây là ảnh compatibility ở cấp `reports/`, nội dung tương đương `reports/07_random_forest/confusion_matrix.png`.

---

## 7. Ảnh So Sánh 4 Model

### `reports/08_comparison/comparison_f1_score.png`

**Nên dùng ở:** Mục 3 - Kết quả thu được.

**Nội dung ảnh:** Bar chart so sánh weighted F1-score của 4 model trên test set.

**Chứng minh điều gì:**

- SVM là model tốt nhất trong 4 model.
- Logistic Regression đứng thứ hai và rất gần SVM.
- Random Forest thấp hơn hai model tuyến tính.
- Naive Bayes thấp nhất nhưng train nhanh nhất.

**Nhận xét gợi ý:** Kết quả cho thấy với dữ liệu TF-IDF, các model tuyến tính như LinearSVC và Logistic Regression phù hợp hơn Random Forest.

---

### `reports/comparison_f1_score.png`

**Ghi chú:** Đây là ảnh compatibility ở cấp `reports/`, nội dung tương đương `reports/08_comparison/comparison_f1_score.png`.

**Nên dùng:** Ưu tiên dùng ảnh trong `reports/08_comparison/`.

---

## 8. Ảnh WELFake / Dataset 2

Những ảnh dưới đây thuộc phần WELFake/dataset 2. Nếu báo cáo hiện tại chỉ tập trung ISOT và 4 model chính, có thể đưa vào phần mở rộng hoặc hướng phát triển. Nếu nhóm có làm cross-dataset/generalization thì mới nên đưa vào phần kết quả chính.

### `reports/welfake_label_dist.png`

**Nội dung ảnh:** Phân phối nhãn REAL/FAKE trong WELFake.

**Chứng minh điều gì:** WELFake cũng có đủ hai class và tương đối cân bằng.

---

### `reports/welfake_text_length.png`

**Nội dung ảnh:** Phân phối độ dài văn bản WELFake.

**Chứng minh điều gì:** WELFake có đặc điểm độ dài văn bản riêng, có thể khác ISOT.

---

### `reports/welfake_text_length_by_label.png`

**Nội dung ảnh:** Độ dài văn bản WELFake tách theo nhãn.

**Chứng minh điều gì:** Có thể xem sự khác biệt giữa REAL/FAKE trong dataset thứ hai.

---

### `reports/welfake_top_ngrams.png`

**Nội dung ảnh:** Top unigram/bigram trong WELFake.

**Chứng minh điều gì:** Từ vựng nổi bật của WELFake có thể khác ISOT, gợi ý khả năng domain shift.

---

### `reports/welfake_isot_comparison.png`

**Nội dung ảnh:** So sánh thống kê giữa WELFake và ISOT.

**Chứng minh điều gì:** Hai dataset có phân phối/đặc trưng khác nhau, phù hợp để bàn về generalization.

---

### `reports/welfake_tfidf_config.png`

**Nội dung ảnh:** Kết quả thử cấu hình TF-IDF cho WELFake.

**Chứng minh điều gì:** Pipeline vectorization cũng được thử nghiệm trên dataset thứ hai.

---

### `reports/welfake_svm_c_tuning.png`

**Nội dung ảnh:** Tune tham số `C` của LinearSVC trên WELFake.

**Chứng minh điều gì:** SVM WELFake cũng được tuning tương tự ISOT.

---

### `reports/welfake_svm_kernel_comparison.png`

**Nội dung ảnh:** So sánh LinearSVC và RBF SVC trên WELFake.

**Chứng minh điều gì:** Kiểm tra xem kết luận LinearSVC tốt hơn RBF có còn đúng trên dataset khác không.

---

### `reports/welfake_svm_confusion_matrix.png`

**Nội dung ảnh:** Confusion matrix của SVM trên WELFake validation set.

**Chứng minh điều gì:** Đánh giá chi tiết lỗi của SVM trên dataset thứ hai.

---

## 9. Ảnh Nên Đưa Vào Báo Cáo Chính

Nếu báo cáo bị giới hạn số trang, nên ưu tiên các ảnh sau:

1. `reports/02_eda/label_distribution_bar_pie.png`
2. `reports/02_eda/text_length_histogram.png`
3. `reports/02_eda/top_unigrams_bigrams.png`
4. `reports/03_vectorization/tfidf_config_accuracy.png`
5. `reports/04_naive_bayes/alpha_accuracy.png`
6. `reports/06_svm/linearsvc_c_tuning.png`
7. `reports/06_svm/kernel_comparison.png`
8. `reports/08_comparison/comparison_f1_score.png`

Confusion matrix nên đưa theo bảng/model nếu còn chỗ:

- `reports/04_naive_bayes/confusion_matrix.png`
- `reports/05_logistic_regression/confusion_matrix.png`
- `reports/06_svm/confusion_matrix.png`
- `reports/07_random_forest/confusion_matrix.png`

---

## 10. Cách Viết Caption Cho Ảnh

Mẫu caption nên dùng:

```text
Hình X. Phân phối nhãn của tập dữ liệu ISOT sau preprocessing.
Biểu đồ cho thấy dữ liệu có đủ hai lớp REAL và FAKE, không bị mất cân bằng quá nghiêm trọng, phù hợp cho huấn luyện mô hình phân loại nhị phân.
```

```text
Hình X. So sánh F1-score của bốn mô hình trên test set.
Kết quả cho thấy LinearSVC đạt F1-score cao nhất, chứng minh mô hình tuyến tính phù hợp với đặc trưng TF-IDF trong bài toán phát hiện tin giả.
```

Nguyên tắc:

- Caption không chỉ nói ảnh là gì.
- Caption phải nói ảnh chứng minh điều gì.
- Sau mỗi ảnh nên có 2-4 câu nhận xét ngắn.
