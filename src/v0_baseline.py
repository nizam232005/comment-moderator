"""
Version 0: TF-IDF + Logistic Regression baseline for comment moderation.

Setup:
    pip install pandas scikit-learn joblib
    Download Jigsaw "Toxic Comment Classification Challenge" train.csv
    (Kaggle) and put it at data/train.csv

Run:
    python v0_baseline.py
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/train.csv")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

ALL_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# DOCUMENT THIS CHOICE IN YOUR README:
# "any"   -> binary label = 1 if ANY of the six categories is 1
# "toxic" -> binary label = Jigsaw's own `toxic` column only
BINARY_DEFINITION = "any"

VERSION = "v0_binary"  # later: "v4_multilabel", etc.
SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    """Load ALL six label columns; never throw them away."""
    df = pd.read_csv(DATA_PATH)
    return df["comment_text"].fillna(""), df[ALL_LABELS]


def make_binary_label(labels: pd.DataFrame):
    if BINARY_DEFINITION == "any":
        return (labels.sum(axis=1) > 0).astype(int)
    return labels["toxic"].astype(int)


# ---------------------------------------------------------------------------
# Train / evaluate: these don't care if y has one column or six
# ---------------------------------------------------------------------------
def train(X, y):
    """X: sparse features. y: 1D array (binary) or 2D array (multi-label)."""
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    model = base if y.ndim == 1 else OneVsRestClassifier(base)
    model.fit(X, y)
    return model


def evaluate(model, X, y, label_names):
    """Per-label precision/recall/F1 plus macro-F1."""
    preds = model.predict(X)
    report = classification_report(
        y, preds, target_names=label_names, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "per_label": {name: report[name] for name in label_names},
    }


def save_results(metrics: dict, extra: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"version": VERSION, "binary_definition": BINARY_DEFINITION, **extra, **metrics}
    path = RESULTS_DIR / f"{VERSION}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved metrics -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    texts, labels = load_data()
    y = make_binary_label(labels)  # swap for labels.values in the multi-label version

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=SEED, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=3
    )
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)

    t0 = time.time()
    model = train(X_train, y_train.values)
    train_seconds = time.time() - t0

    # Latency per comment is a number you'll want later for the hybrid comparison
    t0 = time.time()
    model.predict(X_test)
    ms_per_comment = (time.time() - t0) / X_test.shape[0] * 1000

    metrics = evaluate(model, X_test, y_test.values, ["not_toxic", "toxic"])
    print(json.dumps(metrics, indent=2))

    save_results(
        metrics,
        {
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "train_seconds": round(train_seconds, 2),
            "predict_ms_per_comment": round(ms_per_comment, 4),
        },
    )

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, MODELS_DIR / f"{VERSION}.joblib")


if __name__ == "__main__":
    main()