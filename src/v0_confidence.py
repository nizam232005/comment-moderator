"""
V0 follow-up: how confident is the classifier, and where do its errors live?

Answers the question the hybrid design depends on:
  "If I only escalate uncertain comments to an LLM, how much traffic do I
   still resolve cheaply, how accurate is that part, and how many of the
   classifier's mistakes land in the escalated slice?"

Run AFTER v0_baseline.py (needs models/v0_binary.joblib), from the same folder:
    python v0_confidence.py
"""

import json

import joblib
import numpy as np
from sklearn.model_selection import train_test_split


from v0_baseline import (
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    VERSION,
    load_data,
    make_binary_label,
)

DECISION_THRESHOLD = 0.5  # where "toxic" is declared; not tuned or calibrated yet
BAND_EDGES = [0.0, 0.1, 0.3, 0.7, 0.9, 1.0]


def get_test_scores():
    """Recreate the EXACT split from v0_baseline.py (same seed, same stratify)."""
    texts, labels = load_data()
    y = make_binary_label(labels)
    _, X_test_txt, _, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=SEED, stratify=y
    )
    bundle = joblib.load(MODELS_DIR / f"{VERSION}.joblib")
    X_test = bundle["vectorizer"].transform(X_test_txt)
    probs = bundle["model"].predict_proba(X_test)[:, 1]
    return probs, y_test.values


def band_table(probs, y):
    pred = (probs >= DECISION_THRESHOLD).astype(int)
    rows = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        in_band = (probs >= lo) & ((probs < hi) if hi < 1.0 else (probs <= hi))
        n = int(in_band.sum())
        rows.append(
            {
                "band": f"{lo:.1f}-{hi:.1f}",
                "share_of_traffic": round(n / len(y), 4),
                "n": n,
                "actual_toxic_rate": round(float(y[in_band].mean()), 4) if n else None,
                "accuracy": round(float((pred == y)[in_band].mean()), 4) if n else None,
            }
        )
    return rows


def router_sweep(probs, y):
    """Simulate: below `low` -> allow, above `high` -> remove, else -> escalate."""
    pred = (probs >= DECISION_THRESHOLD).astype(int)
    wrong = pred != y
    total_errors = int(wrong.sum())
    rows = []
    for low in [0.05, 0.1, 0.2, 0.3]:
        for high in [0.7, 0.8, 0.9, 0.95]:
            auto = (probs <= low) | (probs >= high)
            esc = ~auto
            rows.append(
                {
                    "low": low,
                    "high": high,
                    "auto_resolved_share": round(float(auto.mean()), 4),
                    "escalated_share": round(float(esc.mean()), 4),
                    "accuracy_on_auto": round(float((pred == y)[auto].mean()), 4),
                    "errors_left_in_auto": int((wrong & auto).sum()),
                    "share_of_all_errors_escalated": round(
                        float((wrong & esc).sum() / total_errors), 4
                    ),
                }
            )
    return rows


def print_table(rows, cols):
    print("  ".join(f"{c:>26}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>26}" for c in cols))
    print()


def main():
    probs, y = get_test_scores()

    bands = band_table(probs, y)
    print("== Score bands ==")
    print_table(bands, ["band", "share_of_traffic", "actual_toxic_rate", "accuracy"])

    sweep = router_sweep(probs, y)
    print("== Router simulation (low/high cutoffs) ==")
    print_table(
        sweep,
        [
            "low",
            "high",
            "auto_resolved_share",
            "accuracy_on_auto",
            "errors_left_in_auto",
            "share_of_all_errors_escalated",
        ],
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{VERSION}_confidence.json"
    out.write_text(json.dumps({"bands": bands, "router_sweep": sweep}, indent=2))
    print(f"Saved -> {out}")

    # Optional histogram for the README
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.hist(probs[y == 0], bins=50, alpha=0.6, label="actually not toxic")
        plt.hist(probs[y == 1], bins=50, alpha=0.6, label="actually toxic")
        plt.yscale("log")
        plt.xlabel("classifier score")
        plt.ylabel("comments (log scale)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{VERSION}_score_hist.png", dpi=150)
        print("Saved histogram.")
    except ImportError:
        print("matplotlib not installed; skipped histogram.")


if __name__ == "__main__":
    main()