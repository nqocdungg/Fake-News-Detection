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

- ISOT Fake News Dataset
- Source: https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset
  
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

## Kết quả đầu ra

- `models/tfidf_vectorizer.pkl`
- `models/naive_bayes_model.pkl`
- `models/lr_model.pkl`
- `models/svm_model.pkl`
- `models/rf_model.pkl`
- Bang metric va hinh anh bao cao trong `reports/`
