# Fake News Detection

Du an phat hien tin gia bang Machine Learning. Pipeline chinh gom: preprocessing, EDA, TF-IDF vectorization, huan luyen 4 mo hinh, so sanh ket qua va phan tich loi.

## Cau truc thu muc

```text
data/
  raw/                 # Du lieu goc, vi du Fake.csv va True.csv cua ISOT
  processed/           # Du lieu da tien xu ly va train/test split
docs/                  # Ghi chu tham chieu tu tai lieu du an
models/                # Vectorizer va model da train
notebooks/             # Notebook chay theo tung buoc
reports/               # Hinh anh, bang ket qua, bao cao
src/                   # Ma nguon tai su dung
requirements.txt
```

## Clone va cai dat

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

Neu may co `uv`, nen dung cach nay tren Windows:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Du lieu

Dat du lieu ISOT goc vao `data/raw/`. Neu da co file tien xu ly, cac file CSV nam trong `data/processed/`.

## Thu tu chay notebook

1. `notebooks/01_preprocessing.ipynb`: lam sach text, xu ly Reuters leakage, tao du lieu tien xu ly.
2. `notebooks/02_eda.ipynb`: truc quan hoa phan bo nhan, do dai text, word cloud, top unigram/bigram.
3. `notebooks/03_vectorization.ipynb`: chia train/validation/test, fit TF-IDF tren train set, luu vectorizer.
4. `notebooks/04_naive_bayes.ipynb`: train va tune Multinomial Naive Bayes.
5. `notebooks/05_logistic_regression.ipynb`: train va tune Logistic Regression.
6. `notebooks/06_svm.ipynb`: train va tune LinearSVC/SVC.
7. `notebooks/07_random_forest.ipynb`: train va tune Random Forest.
8. `notebooks/08_comparison.ipynb`: danh gia 4 model tren test set va so sanh metric.
9. `notebooks/09_feature_error_analysis.ipynb`: phan tich tu quan trong va cac mau du doan sai.

## Ket qua dau ra

- `models/tfidf_vectorizer.pkl`
- `models/naive_bayes_model.pkl`
- `models/lr_model.pkl`
- `models/svm_model.pkl`
- `models/rf_model.pkl`
- Bang metric va hinh anh bao cao trong `reports/`
