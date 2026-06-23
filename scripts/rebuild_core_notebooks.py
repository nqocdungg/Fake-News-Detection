from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, path)
    print(f"wrote {path.relative_to(ROOT)}")


def build_01():
    cells = [
        md(
            """
            # 01 - Preprocessing ISOT

            Output:
            - `src/preprocessing.py`
            - `data/processed/preprocessed_isot_full.csv`
            - `data/preprocessed_isot_full.csv` for backward compatibility
            - `reports/01_preprocessing/preprocessing_summary.json`

            This notebook downloads/loads ISOT, fixes Reuters leakage, keeps negation words through
            `src.preprocessing.preprocess_text`, and saves a reusable processed dataset.
            """
        ),
        code(
            """
            import json
            import sys
            from pathlib import Path

            import pandas as pd

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            sys.path.append(str(ROOT))

            from src.preprocessing import preprocess_text

            DATA_RAW = ROOT / "data" / "raw"
            DATA_PROCESSED = ROOT / "data" / "processed"
            REPORT_DIR = ROOT / "reports" / "01_preprocessing"

            DATA_RAW.mkdir(parents=True, exist_ok=True)
            DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            """
        ),
        code(
            """
            def find_isot_files():
                candidates = [
                    DATA_RAW,
                    DATA_RAW / "isot",
                    DATA_RAW / "News_Dataset",
                    DATA_RAW / "isot-fake-news-dataset",
                ]
                for base in candidates:
                    fake_path = base / "Fake.csv"
                    true_path = base / "True.csv"
                    if fake_path.exists() and true_path.exists():
                        return fake_path, true_path
                return None, None


            fake_path, true_path = find_isot_files()

            if fake_path is None or true_path is None:
                import kagglehub

                kaggle_path = Path(kagglehub.dataset_download("rahulogoel/isot-fake-news-dataset"))
                possible_pairs = [
                    (kaggle_path / "News_Dataset" / "Fake.csv", kaggle_path / "News_Dataset" / "True.csv"),
                    (kaggle_path / "Fake.csv", kaggle_path / "True.csv"),
                ]
                for fake_candidate, true_candidate in possible_pairs:
                    if fake_candidate.exists() and true_candidate.exists():
                        fake_path, true_path = fake_candidate, true_candidate
                        break

            if fake_path is None or true_path is None:
                raise FileNotFoundError(
                    "Could not find ISOT Fake.csv/True.csv. Put them in data/raw/isot/ "
                    "or allow kagglehub to download rahulogoel/isot-fake-news-dataset."
                )

            print("Fake CSV:", fake_path)
            print("True CSV:", true_path)

            fake_df = pd.read_csv(fake_path)
            true_df = pd.read_csv(true_path)

            print("Fake shape:", fake_df.shape)
            print("True shape:", true_df.shape)
            display(fake_df.head(2))
            display(true_df.head(2))
            """
        ),
        code(
            """
            raw_summary = {
                "fake_shape": list(fake_df.shape),
                "true_shape": list(true_df.shape),
                "fake_nulls": fake_df.isnull().sum().to_dict(),
                "true_nulls": true_df.isnull().sum().to_dict(),
                "fake_duplicates": int(fake_df.duplicated().sum()),
                "true_duplicates": int(true_df.duplicated().sum()),
            }

            fake_df = fake_df.drop_duplicates().copy()
            true_df = true_df.drop_duplicates().copy()

            fake_df = fake_df[fake_df["text"].astype(str).str.strip() != ""].copy()
            true_df = true_df[true_df["text"].astype(str).str.strip() != ""].copy()

            true_df["label"] = 0  # REAL
            fake_df["label"] = 1  # FAKE

            df = pd.concat([true_df, fake_df], ignore_index=True)
            df["title"] = df["title"].fillna("").astype(str)
            df["text"] = df["text"].fillna("").astype(str)
            df["full_text"] = (df["title"].str.strip() + " " + df["text"].str.strip()).str.strip()

            df = df[df["full_text"] != ""].copy()
            duplicated_full_text = int(df.duplicated(subset=["full_text"]).sum())
            conflicting_texts = int((df.groupby("full_text")["label"].nunique() > 1).sum())
            df = df.drop_duplicates(subset=["full_text"], keep="first").reset_index(drop=True)

            print("After basic cleaning:", df.shape)
            print(df["label"].value_counts().sort_index())
            print("Duplicated full_text removed:", duplicated_full_text)
            print("Conflicting full_text count:", conflicting_texts)
            """
        ),
        code(
            """
            sample = "WASHINGTON (Reuters) - Trump did not listen to the warnings, never neither!"
            print("Original :", sample)
            print("Processed:", preprocess_text(sample))

            df["processed_text"] = df["full_text"].apply(preprocess_text)
            empty_processed = int((df["processed_text"].astype(str).str.strip() == "").sum())
            df = df[df["processed_text"].astype(str).str.strip() != ""].reset_index(drop=True)

            final_cols = ["title", "text", "subject", "date", "full_text", "processed_text", "label"]
            final_cols = [col for col in final_cols if col in df.columns]
            df = df[final_cols]

            print("Final shape:", df.shape)
            print(df["label"].value_counts().sort_index())
            display(df[["full_text", "processed_text", "label"]].sample(3, random_state=42))
            """
        ),
        code(
            """
            processed_path = DATA_PROCESSED / "preprocessed_isot_full.csv"
            compat_path = ROOT / "data" / "preprocessed_isot_full.csv"
            compact_path = DATA_PROCESSED / "preprocessed_isot.csv"

            df.to_csv(processed_path, index=False)
            df.to_csv(compat_path, index=False)
            df[["processed_text", "label"]].to_csv(compact_path, index=False)

            summary = {
                **raw_summary,
                "final_shape": list(df.shape),
                "final_label_distribution": df["label"].value_counts().sort_index().to_dict(),
                "empty_processed_removed": empty_processed,
                "duplicated_full_text_removed": duplicated_full_text,
                "conflicting_full_text_count": conflicting_texts,
                "processed_path": str(processed_path),
                "compat_path": str(compat_path),
            }

            with open(REPORT_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print("Saved:", processed_path)
            print("Saved:", compat_path)
            print("Saved:", REPORT_DIR / "preprocessing_summary.json")
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "01_preprocessing.ipynb", cells)


def build_02():
    cells = [
        md(
            """
            # 02 - EDA ISOT

            Saves all EDA outputs to `reports/02_eda/`.
            """
        ),
        code(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from sklearn.feature_extraction.text import CountVectorizer
            from wordcloud import WordCloud

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            REPORT_DIR = ROOT / "reports" / "02_eda"
            REPORT_DIR.mkdir(parents=True, exist_ok=True)

            sns.set_style("whitegrid")
            LABEL_NAMES = {0: "REAL", 1: "FAKE"}
            """
        ),
        code(
            """
            data_path = ROOT / "data" / "processed" / "preprocessed_isot_full.csv"
            if not data_path.exists():
                data_path = ROOT / "data" / "preprocessed_isot_full.csv"

            if not data_path.exists():
                raise FileNotFoundError(
                    "Run notebooks/01_preprocessing.ipynb first to create preprocessed_isot_full.csv."
                )

            df = pd.read_csv(data_path)
            df = df.dropna(subset=["processed_text", "label"]).reset_index(drop=True)
            df["label"] = df["label"].astype(int)
            df["label_name"] = df["label"].map(LABEL_NAMES)
            df["word_count"] = df["processed_text"].astype(str).str.split().str.len()

            print("Dataset shape:", df.shape)
            print(df["label"].value_counts().sort_index())
            print("Nulls:")
            display(df.isnull().sum())
            print("Duplicate rows:", df.duplicated().sum())

            df.sample(10, random_state=42).to_csv(REPORT_DIR / "sample_texts.csv", index=False)
            df.isnull().sum().rename("null_count").to_csv(REPORT_DIR / "null_summary.csv")
            pd.Series({"duplicate_rows": int(df.duplicated().sum())}).to_csv(REPORT_DIR / "duplicate_summary.csv")
            """
        ),
        code(
            """
            label_counts = df["label_name"].value_counts().reindex(["REAL", "FAKE"])
            label_counts.to_csv(REPORT_DIR / "label_distribution.csv", header=["count"])

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.barplot(x=label_counts.index, y=label_counts.values, ax=axes[0], palette=["royalblue", "tomato"])
            axes[0].set_title("ISOT label distribution")
            axes[0].set_xlabel("Label")
            axes[0].set_ylabel("Count")

            axes[1].pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%", colors=["royalblue", "tomato"])
            axes[1].set_title("ISOT label ratio")

            plt.tight_layout()
            plt.savefig(REPORT_DIR / "label_distribution_bar_pie.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.histplot(data=df, x="word_count", hue="label_name", bins=60, kde=True, element="step")
            plt.title("Text length by label")
            plt.xlabel("Number of words after preprocessing")
            plt.ylabel("Article count")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "text_length_histogram.png", dpi=150, bbox_inches="tight")
            plt.show()

            df.groupby("label_name")["word_count"].describe().to_csv(REPORT_DIR / "text_length_summary.csv")
            """
        ),
        code(
            """
            for label_value, label_name in LABEL_NAMES.items():
                corpus = " ".join(df.loc[df["label"] == label_value, "processed_text"].astype(str))
                wc = WordCloud(width=1200, height=650, background_color="white", collocations=False).generate(corpus)
                plt.figure(figsize=(12, 6))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"Word cloud - {label_name}")
                plt.tight_layout()
                out = REPORT_DIR / f"wordcloud_{label_name.lower()}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                plt.show()
                print("Saved:", out)
            """
        ),
        code(
            """
            def top_ngrams(texts, ngram_range, top_n=20):
                vec = CountVectorizer(ngram_range=ngram_range, stop_words="english", max_features=50000)
                X = vec.fit_transform(texts.astype(str))
                counts = X.sum(axis=0).A1
                out = pd.DataFrame({"ngram": vec.get_feature_names_out(), "count": counts})
                return out.sort_values("count", ascending=False).head(top_n)


            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            rows = []
            for col_idx, (label_value, label_name) in enumerate(LABEL_NAMES.items()):
                corpus = df.loc[df["label"] == label_value, "processed_text"]
                for row_idx, (name, ngram_range) in enumerate([("unigram", (1, 1)), ("bigram", (2, 2))]):
                    ngram_df = top_ngrams(corpus, ngram_range)
                    ngram_df["label"] = label_name
                    ngram_df["type"] = name
                    rows.append(ngram_df)
                    ax = axes[row_idx, col_idx]
                    sns.barplot(data=ngram_df, x="count", y="ngram", ax=ax, palette="viridis")
                    ax.set_title(f"Top 20 {name}s - {label_name}")
                    ax.set_xlabel("Count")
                    ax.set_ylabel("")

            top_df = pd.concat(rows, ignore_index=True)
            top_df.to_csv(REPORT_DIR / "top_unigrams_bigrams.csv", index=False)
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "top_unigrams_bigrams.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "02_eda.ipynb", cells)


def build_03():
    cells = [
        md(
            """
            # 03 - TF-IDF Vectorization

            Split 70/15/15, fit vectorizer only on train, evaluate TF-IDF configs, save final vectorizer and matrices.
            Results are saved to `reports/03_vectorization/`.
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            import joblib
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import MultinomialNB

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            REPORT_DIR = ROOT / "reports" / "03_vectorization"
            MODEL_DIR = ROOT / "models"
            DATA_DIR = ROOT / "data"

            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            sns.set_style("whitegrid")
            """
        ),
        code(
            """
            data_path = ROOT / "data" / "processed" / "preprocessed_isot_full.csv"
            if not data_path.exists():
                data_path = ROOT / "data" / "preprocessed_isot_full.csv"

            if not data_path.exists():
                raise FileNotFoundError("Run notebooks/01_preprocessing.ipynb before vectorization.")

            df = pd.read_csv(data_path).dropna(subset=["processed_text", "label"]).reset_index(drop=True)
            X = df["processed_text"].astype(str)
            y = df["label"].astype(int)

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
            )

            print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
            print("Train labels:")
            print(y_train.value_counts().sort_index())
            """
        ),
        code(
            """
            configs = [
                {"max_features": max_features, "ngram_range": ngram_range}
                for max_features in [3000, 5000, 10000]
                for ngram_range in [(1, 1), (1, 2)]
            ]

            rows = []
            for cfg in configs:
                vectorizer = TfidfVectorizer(
                    max_features=cfg["max_features"],
                    ngram_range=cfg["ngram_range"],
                )
                X_train_cfg = vectorizer.fit_transform(X_train)
                X_val_cfg = vectorizer.transform(X_val)

                clf = MultinomialNB(alpha=0.1)
                clf.fit(X_train_cfg, y_train)
                y_pred = clf.predict(X_val_cfg)

                rows.append({
                    "max_features": cfg["max_features"],
                    "ngram_range": str(cfg["ngram_range"]),
                    "val_accuracy": accuracy_score(y_val, y_pred),
                    "val_f1_weighted": f1_score(y_val, y_pred, average="weighted"),
                    "n_features_actual": len(vectorizer.get_feature_names_out()),
                })

            results = pd.DataFrame(rows).sort_values(
                ["val_accuracy", "val_f1_weighted"], ascending=False
            ).reset_index(drop=True)
            results.to_csv(REPORT_DIR / "tfidf_config_results.csv", index=False)
            display(results)

            best = results.iloc[0].to_dict()
            print("Best TF-IDF config:", best)
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            plot_df = results.copy()
            plot_df["config"] = plot_df["max_features"].astype(str) + " | " + plot_df["ngram_range"]
            sns.barplot(data=plot_df, x="config", y="val_accuracy", palette="mako")
            plt.xticks(rotation=30, ha="right")
            plt.title("TF-IDF config validation accuracy")
            plt.xlabel("max_features | ngram_range")
            plt.ylabel("Validation accuracy")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "tfidf_config_accuracy.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
        code(
            """
            best_max_features = int(best["max_features"])
            best_ngram_range = tuple(int(x) for x in best["ngram_range"].strip("()").split(",") if x.strip())

            vectorizer = TfidfVectorizer(max_features=best_max_features, ngram_range=best_ngram_range)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_val_tfidf = vectorizer.transform(X_val)
            X_test_tfidf = vectorizer.transform(X_test)

            joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
            joblib.dump(X_train_tfidf, DATA_DIR / "X_train_tfidf.pkl")
            joblib.dump(X_val_tfidf, DATA_DIR / "X_val_tfidf.pkl")
            joblib.dump(X_test_tfidf, DATA_DIR / "X_test_tfidf.pkl")
            joblib.dump(y_train, DATA_DIR / "y_train.pkl")
            joblib.dump(y_val, DATA_DIR / "y_val.pkl")
            joblib.dump(y_test, DATA_DIR / "y_test.pkl")

            summary = {
                "train_size": int(len(X_train)),
                "val_size": int(len(X_val)),
                "test_size": int(len(X_test)),
                "best_config": {
                    "max_features": best_max_features,
                    "ngram_range": list(best_ngram_range),
                    "val_accuracy": float(best["val_accuracy"]),
                    "val_f1_weighted": float(best["val_f1_weighted"]),
                },
                "train_shape": list(X_train_tfidf.shape),
                "val_shape": list(X_val_tfidf.shape),
                "test_shape": list(X_test_tfidf.shape),
            }
            with open(REPORT_DIR / "split_vectorization_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print("Saved vectorizer:", MODEL_DIR / "tfidf_vectorizer.pkl")
            print("Saved TF-IDF matrices and labels in:", DATA_DIR)
            print(summary)
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "03_vectorization.ipynb", cells)


def build_04():
    cells = [
        md(
            """
            # 04 - Multinomial Naive Bayes

            Common model template: load TF-IDF splits, GridSearchCV, best params, validation report,
            confusion matrix, training time, and model persistence.

            Outputs are saved to `reports/04_naive_bayes/` and `models/naive_bayes_model.pkl`.
            """
        ),
        code(
            """
            import json
            import time
            from pathlib import Path

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
            )
            from sklearn.model_selection import GridSearchCV
            from sklearn.naive_bayes import MultinomialNB

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            DATA_DIR = ROOT / "data"
            MODELS_DIR = ROOT / "models"
            REPORT_DIR = ROOT / "reports" / "04_naive_bayes"

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            sns.set_style("whitegrid")
            LABELS = ["REAL (0)", "FAKE (1)"]
            """
        ),
        code(
            """
            X_train = joblib.load(DATA_DIR / "X_train_tfidf.pkl")
            X_val = joblib.load(DATA_DIR / "X_val_tfidf.pkl")
            y_train = np.array(joblib.load(DATA_DIR / "y_train.pkl"))
            y_val = np.array(joblib.load(DATA_DIR / "y_val.pkl"))

            print(f"Data directory: {DATA_DIR.resolve()}")
            print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
            print("Train labels:", dict(zip(*np.unique(y_train, return_counts=True))))
            print("Val labels  :", dict(zip(*np.unique(y_val, return_counts=True))))
            """
        ),
        code(
            """
            param_grid = {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
                "fit_prior": [True, False],
            }

            grid_search = GridSearchCV(
                estimator=MultinomialNB(),
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time

            print(f"Training/GridSearch time: {training_time:.2f} seconds")
            print("Best params:", grid_search.best_params_)
            print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
            """
        ),
        code(
            """
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val)

            val_accuracy = accuracy_score(y_val, y_pred)
            val_f1_weighted = f1_score(y_val, y_pred, average="weighted")
            report_dict = classification_report(y_val, y_pred, target_names=LABELS, output_dict=True)
            report_text = classification_report(y_val, y_pred, target_names=LABELS)
            cm = confusion_matrix(y_val, y_pred)

            print(report_text)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1: {val_f1_weighted:.4f}")
            print(cm)
            """
        ),
        code(
            """
            cv_results = pd.DataFrame(grid_search.cv_results_)
            cv_results.to_csv(REPORT_DIR / "gridsearch_results.csv", index=False)

            report_df = pd.DataFrame(report_dict).transpose()
            report_df.to_csv(REPORT_DIR / "classification_report.csv")

            metrics = {
                "best_params": grid_search.best_params_,
                "best_cv_accuracy": float(grid_search.best_score_),
                "validation_accuracy": float(val_accuracy),
                "validation_f1_weighted": float(val_f1_weighted),
                "training_time_seconds": float(training_time),
                "confusion_matrix": cm.tolist(),
            }
            with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            with open(REPORT_DIR / "training_log.txt", "w", encoding="utf-8") as f:
                f.write(f"Training/GridSearch time: {training_time:.2f} seconds\\n")
                f.write(f"Best params: {grid_search.best_params_}\\n")
                f.write(f"Best CV accuracy: {grid_search.best_score_:.6f}\\n")
                f.write(f"Validation accuracy: {val_accuracy:.6f}\\n")
                f.write(f"Validation weighted F1: {val_f1_weighted:.6f}\\n\\n")
                f.write(report_text)

            model_path = MODELS_DIR / "naive_bayes_model.pkl"
            joblib.dump(best_model, model_path)
            print("Saved model:", model_path)
            print("Saved reports:", REPORT_DIR)
            """
        ),
        code(
            """
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
            ax.set_title("Naive Bayes - Validation Confusion Matrix")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
        code(
            """
            results_true = cv_results[cv_results["param_fit_prior"] == True].sort_values("param_alpha")
            results_false = cv_results[cv_results["param_fit_prior"] == False].sort_values("param_alpha")

            plt.figure(figsize=(10, 5))
            plt.plot(
                results_true["param_alpha"].astype(float),
                results_true["mean_test_score"],
                marker="o",
                label="fit_prior=True",
                linewidth=2,
            )
            plt.plot(
                results_false["param_alpha"].astype(float),
                results_false["mean_test_score"],
                marker="s",
                label="fit_prior=False",
                linewidth=2,
            )
            plt.title("Naive Bayes - Accuracy by alpha")
            plt.xlabel("Alpha")
            plt.ylabel("Mean CV accuracy")
            plt.xscale("log")
            plt.xticks([0.01, 0.1, 0.5, 1.0, 2.0], ["0.01", "0.1", "0.5", "1.0", "2.0"])
            plt.legend()
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "alpha_accuracy.png", dpi=150, bbox_inches="tight")
            plt.savefig(ROOT / "reports" / "naive_bayes_alpha_tuning.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "04_naive_bayes.ipynb", cells)


def build_05():
    cells = [
        md(
            """
            # 05 - Logistic Regression

            Required grid:
            - `C = [0.1, 0.5, 1.0, 2.0, 5.0]`
            - `solver = ['lbfgs', 'saga']`
            - `max_iter = [500, 1000]`
            - `cv = 5`

            Outputs are saved to `reports/05_logistic_regression/` and `models/lr_model.pkl`.
            """
        ),
        code(
            """
            import json
            import time
            from pathlib import Path

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
            )
            from sklearn.model_selection import GridSearchCV

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            DATA_DIR = ROOT / "data"
            MODELS_DIR = ROOT / "models"
            REPORT_DIR = ROOT / "reports" / "05_logistic_regression"

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            sns.set_style("whitegrid")
            LABELS = ["REAL (0)", "FAKE (1)"]
            """
        ),
        code(
            """
            X_train = joblib.load(DATA_DIR / "X_train_tfidf.pkl")
            X_val = joblib.load(DATA_DIR / "X_val_tfidf.pkl")
            y_train = np.array(joblib.load(DATA_DIR / "y_train.pkl"))
            y_val = np.array(joblib.load(DATA_DIR / "y_val.pkl"))

            print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
            print("Train labels:", dict(zip(*np.unique(y_train, return_counts=True))))
            print("Val labels  :", dict(zip(*np.unique(y_val, return_counts=True))))
            """
        ),
        code(
            """
            lr = LogisticRegression(random_state=42)

            param_grid = {
                "C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "solver": ["lbfgs", "saga"],
                "max_iter": [500, 1000],
            }

            grid_search = GridSearchCV(
                estimator=lr,
                param_grid=param_grid,
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=1,
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time

            print(f"Training/GridSearch time: {training_time:.2f} seconds")
            print("Best params:", grid_search.best_params_)
            print(f"Best CV weighted F1: {grid_search.best_score_:.4f}")
            """
        ),
        code(
            """
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val)

            val_accuracy = accuracy_score(y_val, y_pred)
            val_f1_weighted = f1_score(y_val, y_pred, average="weighted")
            report_dict = classification_report(y_val, y_pred, target_names=LABELS, output_dict=True)
            report_text = classification_report(y_val, y_pred, target_names=LABELS)
            cm = confusion_matrix(y_val, y_pred)

            print(report_text)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1: {val_f1_weighted:.4f}")
            print(cm)
            """
        ),
        code(
            """
            cv_results = pd.DataFrame(grid_search.cv_results_)
            cv_results.to_csv(REPORT_DIR / "gridsearch_results.csv", index=False)

            pd.DataFrame(report_dict).transpose().to_csv(REPORT_DIR / "classification_report.csv")

            metrics = {
                "best_params": grid_search.best_params_,
                "best_cv_f1_weighted": float(grid_search.best_score_),
                "validation_accuracy": float(val_accuracy),
                "validation_f1_weighted": float(val_f1_weighted),
                "training_time_seconds": float(training_time),
                "confusion_matrix": cm.tolist(),
            }
            with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            with open(REPORT_DIR / "training_log.txt", "w", encoding="utf-8") as f:
                f.write(f"Training/GridSearch time: {training_time:.2f} seconds\\n")
                f.write(f"Best params: {grid_search.best_params_}\\n")
                f.write(f"Best CV weighted F1: {grid_search.best_score_:.6f}\\n")
                f.write(f"Validation accuracy: {val_accuracy:.6f}\\n")
                f.write(f"Validation weighted F1: {val_f1_weighted:.6f}\\n\\n")
                f.write(report_text)

            model_path = MODELS_DIR / "lr_model.pkl"
            joblib.dump(best_model, model_path)
            print("Saved model:", model_path)
            print("Saved reports:", REPORT_DIR)
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
                cmap="Blues", ax=ax, values_format="d", colorbar=False
            )
            ax.set_title("Logistic Regression - Validation Confusion Matrix")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.show()

            # Compatibility image for older report links.
            fig, ax = plt.subplots(figsize=(8, 4))
            top = cv_results.sort_values("mean_test_score", ascending=False).head(10).copy()
            top["params_label"] = top["params"].astype(str)
            sns.barplot(data=top, y="params_label", x="mean_test_score", ax=ax, palette="mako")
            ax.set_title("Top Logistic Regression GridSearchCV configs")
            ax.set_xlabel("Mean CV weighted F1")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "top_gridsearch_configs.png", dpi=150, bbox_inches="tight")
            plt.savefig(ROOT / "reports" / "lr_evaluation_plots.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "05_logistic_regression.ipynb", cells)


def build_06():
    cells = [
        md(
            """
            # 06 - Support Vector Machine

            Tries `LinearSVC` on full TF-IDF data and `SVC(kernel='rbf')` on a stratified subsample.
            Saves outputs to `reports/06_svm/` and `models/svm_model.pkl`.
            """
        ),
        code(
            """
            import json
            import time
            from pathlib import Path

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
            )
            from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
            from sklearn.svm import LinearSVC, SVC

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            DATA_DIR = ROOT / "data"
            MODELS_DIR = ROOT / "models"
            REPORT_DIR = ROOT / "reports" / "06_svm"

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            sns.set_style("whitegrid")
            LABELS = ["REAL (0)", "FAKE (1)"]
            RANDOM_STATE = 42
            """
        ),
        code(
            """
            X_train = joblib.load(DATA_DIR / "X_train_tfidf.pkl")
            X_val = joblib.load(DATA_DIR / "X_val_tfidf.pkl")
            y_train = np.array(joblib.load(DATA_DIR / "y_train.pkl"))
            y_val = np.array(joblib.load(DATA_DIR / "y_val.pkl"))

            print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
            print("Train labels:", dict(zip(*np.unique(y_train, return_counts=True))))
            print("Val labels  :", dict(zip(*np.unique(y_val, return_counts=True))))
            """
        ),
        code(
            """
            baseline = LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE)
            t0 = time.time()
            baseline.fit(X_train, y_train)
            baseline_time = time.time() - t0
            baseline_pred = baseline.predict(X_val)

            print(f"Baseline time: {baseline_time:.2f}s")
            print(f"Baseline accuracy: {accuracy_score(y_val, baseline_pred):.4f}")
            print(f"Baseline weighted F1: {f1_score(y_val, baseline_pred, average='weighted'):.4f}")
            """
        ),
        code(
            """
            param_grid_linear = {"C": [0.01, 0.1, 1, 10]}

            gs_linear = GridSearchCV(
                LinearSVC(max_iter=3000, random_state=RANDOM_STATE),
                param_grid=param_grid_linear,
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=1,
            )

            t0 = time.time()
            gs_linear.fit(X_train, y_train)
            training_time = time.time() - t0

            print(f"LinearSVC GridSearch time: {training_time:.2f}s")
            print("Best params:", gs_linear.best_params_)
            print(f"Best CV weighted F1: {gs_linear.best_score_:.4f}")
            """
        ),
        code(
            """
            linear_cv = pd.DataFrame(gs_linear.cv_results_)
            linear_cv.to_csv(REPORT_DIR / "linearsvc_gridsearch_results.csv", index=False)

            plt.figure(figsize=(8, 4))
            plt.errorbar(
                linear_cv["param_C"].astype(float),
                linear_cv["mean_test_score"],
                yerr=linear_cv["std_test_score"],
                marker="o",
                capsize=5,
                linewidth=2,
            )
            plt.xscale("log")
            plt.title("LinearSVC - weighted F1 by C")
            plt.xlabel("C")
            plt.ylabel("Mean CV weighted F1")
            plt.grid(True, which="both", linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "linearsvc_c_tuning.png", dpi=150, bbox_inches="tight")
            plt.savefig(ROOT / "reports" / "svm_linearsvc_c_tuning.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
        code(
            """
            N_SUB = min(400, X_train.shape[0], X_val.shape[0])
            splitter_train = StratifiedShuffleSplit(n_splits=1, train_size=N_SUB, random_state=RANDOM_STATE)
            splitter_val = StratifiedShuffleSplit(n_splits=1, train_size=N_SUB, random_state=RANDOM_STATE)
            idx_train, _ = next(splitter_train.split(X_train, y_train))
            idx_val, _ = next(splitter_val.split(X_val, y_val))

            X_sub = X_train[idx_train].toarray()
            y_sub = y_train[idx_train]
            X_val_sub = X_val[idx_val].toarray()
            y_val_sub = y_val[idx_val]

            rbf_rows = []
            for C in [0.1, 1, 10]:
                for gamma in ["scale", "auto"]:
                    t0 = time.time()
                    model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=RANDOM_STATE)
                    model.fit(X_sub, y_sub)
                    pred = model.predict(X_val_sub)
                    rbf_rows.append({
                        "C": C,
                        "gamma": gamma,
                        "val_f1_weighted": f1_score(y_val_sub, pred, average="weighted", zero_division=0),
                        "val_accuracy": accuracy_score(y_val_sub, pred),
                        "time_seconds": time.time() - t0,
                    })

            rbf_results = pd.DataFrame(rbf_rows).sort_values("val_f1_weighted", ascending=False)
            rbf_results.to_csv(REPORT_DIR / "rbf_subsample_results.csv", index=False)
            display(rbf_results)
            """
        ),
        code(
            """
            best_model = gs_linear.best_estimator_
            y_pred = best_model.predict(X_val)

            val_accuracy = accuracy_score(y_val, y_pred)
            val_f1_weighted = f1_score(y_val, y_pred, average="weighted")
            report_dict = classification_report(y_val, y_pred, target_names=LABELS, output_dict=True)
            report_text = classification_report(y_val, y_pred, target_names=LABELS)
            cm = confusion_matrix(y_val, y_pred)

            print(report_text)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1: {val_f1_weighted:.4f}")
            """
        ),
        code(
            """
            pd.DataFrame(report_dict).transpose().to_csv(REPORT_DIR / "classification_report.csv")

            metrics = {
                "selected_model": "LinearSVC",
                "best_params": gs_linear.best_params_,
                "best_cv_f1_weighted": float(gs_linear.best_score_),
                "baseline_time_seconds": float(baseline_time),
                "training_time_seconds": float(training_time),
                "validation_accuracy": float(val_accuracy),
                "validation_f1_weighted": float(val_f1_weighted),
                "confusion_matrix": cm.tolist(),
                "rbf_best_subsample": rbf_results.iloc[0].to_dict(),
            }
            with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            with open(REPORT_DIR / "training_log.txt", "w", encoding="utf-8") as f:
                f.write(f"LinearSVC GridSearch time: {training_time:.2f}s\\n")
                f.write(f"Best params: {gs_linear.best_params_}\\n")
                f.write(f"Best CV weighted F1: {gs_linear.best_score_:.6f}\\n")
                f.write(f"Validation accuracy: {val_accuracy:.6f}\\n")
                f.write(f"Validation weighted F1: {val_f1_weighted:.6f}\\n\\n")
                f.write(report_text)

            model_path = MODELS_DIR / "svm_model.pkl"
            joblib.dump(best_model, model_path)
            print("Saved model:", model_path)
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
                cmap="Blues", ax=ax, values_format="d", colorbar=False
            )
            ax.set_title("LinearSVC - Validation Confusion Matrix")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.savefig(ROOT / "reports" / "svm_confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.show()

            fig, ax = plt.subplots(figsize=(7, 4))
            comparison = pd.DataFrame({
                "model": ["LinearSVC full", "Best RBF subsample"],
                "weighted_f1": [val_f1_weighted, float(rbf_results.iloc[0]["val_f1_weighted"])],
            })
            sns.barplot(data=comparison, x="model", y="weighted_f1", ax=ax, palette=["steelblue", "tomato"])
            ax.set_ylim(0.80, 1.02)
            ax.set_title("LinearSVC vs RBF SVC")
            ax.set_ylabel("Validation weighted F1")
            ax.set_xlabel("")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "kernel_comparison.png", dpi=150, bbox_inches="tight")
            plt.savefig(ROOT / "reports" / "svm_kernel_comparison.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "06_svm.ipynb", cells)


def build_07():
    cells = [
        md(
            """
            # 07 - Random Forest

            Required grid:
            - `n_estimators = [100, 200, 300]`
            - `max_depth = [None, 10, 20]`
            - `min_samples_split = [2, 5]`
            - `cv = 5`

            Outputs are saved to `reports/07_random_forest/` and `models/rf_model.pkl`.
            """
        ),
        code(
            """
            import json
            import time
            from pathlib import Path

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
            )
            from sklearn.model_selection import GridSearchCV

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            DATA_DIR = ROOT / "data"
            MODELS_DIR = ROOT / "models"
            REPORT_DIR = ROOT / "reports" / "07_random_forest"

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            sns.set_style("whitegrid")
            LABELS = ["REAL (0)", "FAKE (1)"]
            """
        ),
        code(
            """
            X_train = joblib.load(DATA_DIR / "X_train_tfidf.pkl")
            X_val = joblib.load(DATA_DIR / "X_val_tfidf.pkl")
            y_train = np.array(joblib.load(DATA_DIR / "y_train.pkl"))
            y_val = np.array(joblib.load(DATA_DIR / "y_val.pkl"))

            print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
            print("Train labels:", dict(zip(*np.unique(y_train, return_counts=True))))
            print("Val labels  :", dict(zip(*np.unique(y_val, return_counts=True))))
            """
        ),
        code(
            """
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)

            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            }

            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=2,
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time

            print(f"Training/GridSearch time: {training_time:.2f} seconds")
            print("Best params:", grid_search.best_params_)
            print(f"Best CV weighted F1: {grid_search.best_score_:.4f}")
            """
        ),
        code(
            """
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val)

            val_accuracy = accuracy_score(y_val, y_pred)
            val_f1_weighted = f1_score(y_val, y_pred, average="weighted")
            report_dict = classification_report(y_val, y_pred, target_names=LABELS, output_dict=True)
            report_text = classification_report(y_val, y_pred, target_names=LABELS)
            cm = confusion_matrix(y_val, y_pred)

            print(report_text)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1: {val_f1_weighted:.4f}")
            """
        ),
        code(
            """
            cv_results = pd.DataFrame(grid_search.cv_results_)
            cv_results.to_csv(REPORT_DIR / "gridsearch_results.csv", index=False)
            pd.DataFrame(report_dict).transpose().to_csv(REPORT_DIR / "classification_report.csv")

            metrics = {
                "best_params": grid_search.best_params_,
                "best_cv_f1_weighted": float(grid_search.best_score_),
                "training_time_seconds": float(training_time),
                "validation_accuracy": float(val_accuracy),
                "validation_f1_weighted": float(val_f1_weighted),
                "confusion_matrix": cm.tolist(),
            }
            with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            with open(REPORT_DIR / "training_log.txt", "w", encoding="utf-8") as f:
                f.write(f"Training/GridSearch time: {training_time:.2f} seconds\\n")
                f.write(f"Best params: {grid_search.best_params_}\\n")
                f.write(f"Best CV weighted F1: {grid_search.best_score_:.6f}\\n")
                f.write(f"Validation accuracy: {val_accuracy:.6f}\\n")
                f.write(f"Validation weighted F1: {val_f1_weighted:.6f}\\n\\n")
                f.write(report_text)

            model_path = MODELS_DIR / "rf_model.pkl"
            joblib.dump(best_model, model_path)
            print("Saved model:", model_path)
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
                cmap="Greens", ax=ax, values_format="d", colorbar=False
            )
            ax.set_title("Random Forest - Validation Confusion Matrix")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.savefig(ROOT / "reports" / "rf_confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.show()

            top = cv_results.sort_values("mean_test_score", ascending=False).head(10).copy()
            top["params_label"] = top["params"].astype(str)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=top, y="params_label", x="mean_test_score", ax=ax, palette="crest")
            ax.set_title("Top Random Forest GridSearchCV configs")
            ax.set_xlabel("Mean CV weighted F1")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "top_gridsearch_configs.png", dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "07_random_forest.ipynb", cells)


def build_08():
    cells = [
        md(
            """
            # 08 - Model Comparison

            Loads all four ISOT models and evaluates on the held-out test set.
            Outputs are saved to `reports/08_comparison/`.
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            DATA_DIR = ROOT / "data"
            MODELS_DIR = ROOT / "models"
            REPORT_DIR = ROOT / "reports" / "08_comparison"
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            sns.set_style("whitegrid")
            """
        ),
        code(
            """
            X_test = joblib.load(DATA_DIR / "X_test_tfidf.pkl")
            y_test = np.array(joblib.load(DATA_DIR / "y_test.pkl"))

            models = {
                "Naive Bayes": joblib.load(MODELS_DIR / "naive_bayes_model.pkl"),
                "Logistic Regression": joblib.load(MODELS_DIR / "lr_model.pkl"),
                "SVM (LinearSVC)": joblib.load(MODELS_DIR / "svm_model.pkl"),
                "Random Forest": joblib.load(MODELS_DIR / "rf_model.pkl"),
            }

            print("X_test:", X_test.shape)
            for name, model in models.items():
                print(name, "n_features_in_=", getattr(model, "n_features_in_", None))
                if getattr(model, "n_features_in_", X_test.shape[1]) != X_test.shape[1]:
                    raise ValueError(f"{name} is incompatible with X_test feature count.")
            """
        ),
        code(
            """
            report_dirs = {
                "Naive Bayes": ROOT / "reports" / "04_naive_bayes" / "metrics.json",
                "Logistic Regression": ROOT / "reports" / "05_logistic_regression" / "metrics.json",
                "SVM (LinearSVC)": ROOT / "reports" / "06_svm" / "metrics.json",
                "Random Forest": ROOT / "reports" / "07_random_forest" / "metrics.json",
            }

            rows = []
            for name, model in models.items():
                y_pred = model.predict(X_test)
                train_time = None
                if report_dirs[name].exists():
                    with open(report_dirs[name], "r", encoding="utf-8") as f:
                        train_time = json.load(f).get("training_time_seconds")

                rows.append({
                    "model": name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "training_time_seconds": train_time,
                })

            comparison = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
            comparison.to_csv(REPORT_DIR / "test_metrics.csv", index=False)
            display(comparison)
            """
        ),
        code(
            """
            plt.figure(figsize=(9, 5))
            sns.barplot(data=comparison, x="model", y="f1_weighted", palette="viridis")
            plt.ylim(0.90, 1.00)
            plt.title("Model comparison - Test weighted F1")
            plt.xlabel("")
            plt.ylabel("Weighted F1")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "comparison_f1_score.png", dpi=150, bbox_inches="tight")
            plt.savefig(ROOT / "reports" / "comparison_f1_score.png", dpi=150, bbox_inches="tight")
            plt.show()

            with open(REPORT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_model": comparison.iloc[0]["model"],
                        "best_f1_weighted": float(comparison.iloc[0]["f1_weighted"]),
                        "rows": comparison.to_dict(orient="records"),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            """
        ),
    ]
    write_notebook(NOTEBOOKS / "08_comparison.ipynb", cells)


if __name__ == "__main__":
    build_01()
    build_02()
    build_03()
    build_04()
    build_05()
    build_06()
    build_07()
    build_08()
