import argparse
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_text


LABEL_NAMES = {
    0: "REAL",
    1: "FAKE",
}


def read_input_text(args):
    if args.text:
        return args.text

    if args.file:
        return Path(args.file).read_text(encoding="utf-8")

    print("Paste news text, then press Ctrl-D (macOS/Linux) or Ctrl-Z then Enter (Windows):")
    return sys.stdin.read()


def load_artifacts(model_name):
    model_paths = {
        "svm": ROOT / "models" / "svm_model.pkl",
        "lr": ROOT / "models" / "lr_model.pkl",
        "nb": ROOT / "models" / "naive_bayes_model.pkl",
        "rf": ROOT / "models" / "rf_model.pkl",
    }

    vectorizer_path = ROOT / "models" / "tfidf_vectorizer.pkl"
    model_path = model_paths[model_name]

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Missing vectorizer: {vectorizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    return joblib.load(vectorizer_path), joblib.load(model_path), model_path


def predict(text, model_name):
    vectorizer, model, model_path = load_artifacts(model_name)
    processed_text = preprocess_text(text)

    if not processed_text:
        raise ValueError("Text is empty after preprocessing. Please provide a longer news article.")

    X = vectorizer.transform([processed_text])
    pred = int(model.predict(X)[0])

    result = {
        "model_path": model_path,
        "raw_length_chars": len(text),
        "processed_word_count": len(processed_text.split()),
        "processed_preview": processed_text[:350],
        "prediction_id": pred,
        "prediction_label": LABEL_NAMES.get(pred, str(pred)),
        "probabilities": None,
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["probabilities"] = {
            LABEL_NAMES.get(i, str(i)): float(score)
            for i, score in enumerate(proba)
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a news article is REAL or FAKE using the trained TF-IDF model."
    )
    parser.add_argument("--text", help="News text to classify.")
    parser.add_argument("--file", help="Path to a text file containing the news article.")
    parser.add_argument(
        "--model",
        choices=["svm", "lr", "nb", "rf"],
        default="svm",
        help="Model to use. Default: svm.",
    )
    args = parser.parse_args()

    raw_text = read_input_text(args).strip()
    if not raw_text:
        raise SystemExit("No input text provided.")

    result = predict(raw_text, args.model)

    print()
    print("=== Fake News Detection Prediction ===")
    print(f"Model              : {args.model} ({result['model_path'].name})")
    print(f"Raw length         : {result['raw_length_chars']} characters")
    print(f"Processed words    : {result['processed_word_count']}")
    print(f"Processed preview  : {result['processed_preview']}")
    print()
    print(f"Prediction         : {result['prediction_label']} ({result['prediction_id']})")

    if result["probabilities"]:
        print("Probabilities      :")
        for label, score in result["probabilities"].items():
            print(f"  {label:<5} {score:.4f}")
    else:
        print("Probabilities      : not available for this model")


if __name__ == "__main__":
    main()
