# Fake News Detection

Hệ thống phát hiện tin giả sử dụng Machine Learning trên bộ dữ liệu ISOT.
Pipeline gồm:
- preprocessing
- exploratory data analysis (EDA)
- TF-IDF vectorization
- huấn luyện và đánh giá nhiều mô hình phân loại
- phân tích lỗi mô hình

## Tech Stack

- Python 3.12
- pandas
- scikit-learn
- matplotlib
- seaborn
- nltk

## Dataset

- ISOT Fake News Dataset: https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset
- WELFake Dataset: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Quy ước nhãn dùng trong project: `REAL=0`, `FAKE=1`.
  
## Cấu trúc thư mục

```text
data/
├── raw/                # Dữ liệu gốc
└── processed/          # Dữ liệu đã tiền xử lý

docs/                   # Tài liệu dự án
models/                 # Mô hình huấn luyện
notebooks/              # Notebook
reports/                # Báo cáo
src/                    # Mã nguồn tái sử dụng

.gitignore
requirements.txt
README.md
```

## Clone và cài đặt

```bash
git clone <repository-url>
cd Fake-News-Detection
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```
## Run

Mở Jupyter Notebook:

```bash
jupyter notebook
```

## Thứ tự chạy notebook

1. `notebooks/01_preprocessing.ipynb`: lam sach text, xu ly Reuters leakage, tao du lieu tien xu ly.
2. `notebooks/02_eda.ipynb`: truc quan hoa phan bo nhan, do dai text, word cloud, top unigram/bigram.
3. `notebooks/03_vectorization.ipynb`: chia train/validation/test, fit TF-IDF tren train set, luu vectorizer.
4. `notebooks/04_naive_bayes.ipynb`: train va tune Multinomial Naive Bayes.
5. `notebooks/05_logistic_regression.ipynb`: train va tune Logistic Regression.
6. `notebooks/06_svm.ipynb`: train va tune LinearSVC/SVC.
7. `notebooks/07_random_forest.ipynb`: train va tune Random Forest.
8. `notebooks/08_comparison.ipynb`: danh gia 4 model tren test set va so sanh metric.
9. `notebooks/09_feature_error_analysis.ipynb`: phan tich tu quan trong va cac mau du doan sai.
10. `notebooks/10_welfake_preprocessing.ipynb`: tai WELFake bang KaggleHub, chuan hoa nhan va tien xu ly.
11. `notebooks/11_welfake_eda.ipynb`: EDA WELFake va so sanh voi ISOT.
12. `notebooks/12_welfake_vectorization.ipynb`: chia 70/15/15 va tao TF-IDF rieng cho WELFake.
13. `notebooks/13_welfake_svm.ipynb`: train va tune LinearSVC tren WELFake.
14. `notebooks/14_welfake_naive_bayes.ipynb`: train Naive Bayes WELFake (chua tao).
15. `notebooks/15_welfake_logistic_regression.ipynb`: train Logistic Regression WELFake (chua tao).
16. `notebooks/16_welfake_random_forest.ipynb`: train Random Forest WELFake (chua tao).
17. `notebooks/17_cross_dataset_eval.ipynb`: danh gia cross-dataset (chua tao).

## Kết quả đầu ra

- `models/tfidf_vectorizer.pkl`
- `models/naive_bayes_model.pkl`
- `models/lr_model.pkl`
- `models/svm_model.pkl`
- `models/rf_model.pkl`
- Bang metric va hinh anh bao cao trong `reports/`
