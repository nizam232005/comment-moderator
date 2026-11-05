"""
V1 evaluation: classifier vs. LLM (plus a preview of the hybrid) on the SAME 1,000 comments.

Run after v1_llm_baseline.py has collected all answers:
    python src/v1_evaluate.py
    python src/v1_evaluate.py --show 15     # also print 15 disagreement cases (with text)

What it reports
  1. Per-group results (random sample, uncertain sample) with 95% bootstrap intervals.
  2. An estimate for OVERALL traffic. The uncertain sample is deliberately over-represented,
     so results are reweighted by how common each kind of comment really is (stratified
     estimate: "escalated band" vs "everything else").
  3. A PREVIEW of the hybrid (classifier outside the band, LLM inside it). The 0.2/0.9
     cutoffs were explored on this same test set, so treat that row as exploratory,
     not as a final result. V3 should tune thresholds on a separate validation split.
  4. Cost and latency, and how much the 'blocked' answers could change things.

Writes: results/v1_evaluation.json and results/v1_disagreements.csv (ids only, no text).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from v0_baseline import RESULTS_DIR, VERSION as V0_VERSION  # noqa: E402
from v0_confidence import DECISION_THRESHOLD  # noqa: E402
from v1_llm_baseline import (  # noqa: E402
    ANSWERS_PATH,
    MANIFEST_PATH,
    MODEL_ID,
    PRICE_IN_PER_M,
    PRICE_OUT_PER_M,
    UNCERTAIN_HIGH,
    UNCERTAIN_LOW,
    load_test_set,
)

N_BOOT = 2000
BOOT_SEED = 123


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def counts(y, p):
    """tp, fp, fn, tn for the 'toxic' (=1) class."""
    y, p = np.asarray(y), np.asarray(p)
    return (
        int(((y == 1) & (p == 1)).sum()),
        int(((y == 0) & (p == 1)).sum()),
        int(((y == 1) & (p == 0)).sum()),
        int(((y == 0) & (p == 0)).sum()),
    )


def scores_from_counts(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    total = tp + fp + fn + tn
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
    }


def ci(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def group_scores(y, p, rng):
    """Scores on one simple sample, with bootstrap intervals for F1 and recall."""
    y, p = np.asarray(y), np.asarray(p)
    out = scores_from_counts(*counts(y, p))
    n = len(y)
    f1s, recs = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        s = scores_from_counts(*counts(y[idx], p[idx]))
        f1s.append(s["f1"])
        recs.append(s["recall"])
    out["f1_ci"], out["recall_ci"] = ci(f1s), ci(recs)
    out["n"], out["n_toxic"] = int(n), int(y.sum())
    return out


def stratified_scores(strata, names, rng, boot=True):
    """Overall-traffic estimate: scale each stratum's confusion counts to its real size."""

    def point(sts):
        res = {}
        for nm in names:
            total = np.zeros(4)
            for s in sts:
                total += np.array(counts(s["y"], s[nm]), dtype=float) / len(s["y"]) * s["N"]
            res[nm] = scores_from_counts(*total)
        return res

    base = point(strata)
    if not boot:
        return base

    draws = {nm: {"f1": [], "recall": [], "precision": []} for nm in names}
    for _ in range(N_BOOT):
        resampled = []
        for s in strata:
            idx = rng.integers(0, len(s["y"]), len(s["y"]))
            resampled.append(
                {"y": s["y"][idx], "N": s["N"], **{nm: s[nm][idx] for nm in names}}
            )
        pt = point(resampled)
        for nm in names:
            for k in draws[nm]:
                draws[nm][k].append(pt[nm][k])
    for nm in names:
        for k in draws[nm]:
            base[nm][f"{k}_ci"] = ci(draws[nm][k])
    return base


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_joined():
    manifest = pd.DataFrame(json.loads(MANIFEST_PATH.read_text()))
    answers = pd.DataFrame(
        [json.loads(l) for l in ANSWERS_PATH.read_text().splitlines() if l.strip()]
    )
    answers = answers.drop_duplicates("id", keep="last").drop(columns=["group"])
    df = manifest.merge(answers, on="id", how="inner")
    missing = len(manifest) - len(df)
    df["clf_pred"] = (df.classifier_prob >= DECISION_THRESHOLD).astype(int)
    df["escalated"] = (df.classifier_prob > UNCERTAIN_LOW) & (df.classifier_prob < UNCERTAIN_HIGH)
    return df, missing


def llm_pred(df, blocked_as):
    """toxic -> 1, not_toxic -> 0; blocked/unparseable answers use the chosen fallback."""
    fallback = 1 if blocked_as == "toxic" else 0
    return np.where(df.verdict == "toxic", 1, np.where(df.verdict == "not_toxic", 0, fallback))


def build_strata(df, n_esc_pop, n_rest_pop, blocked_as):
    df = df.copy()
    df["llm_pred"] = llm_pred(df, blocked_as)
    df["hybrid_pred"] = np.where(df.escalated, df.llm_pred, df.clf_pred)
    strata = []
    for esc, pop in [(True, n_esc_pop), (False, n_rest_pop)]:
        sub = df[df.escalated == esc]
        if len(sub) == 0:
            continue
        strata.append(
            {
                "y": sub.label.values,
                "clf": sub.clf_pred.values,
                "llm": sub.llm_pred.values,
                "hybrid": sub.hybrid_pred.values,
                "N": pop,
            }
        )
    return df, strata


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def fmt_ci(s, key):
    if f"{key}_ci" not in s:
        return f"{s[key]:.3f}"
    lo, hi = s[f"{key}_ci"]
    return f"{s[key]:.3f} [{lo:.3f}-{hi:.3f}]"


def print_row(label, s):
    print(
        f"  {label:<18} P {s['precision']:.3f} | R {fmt_ci(s, 'recall'):<19} | "
        f"F1 {fmt_ci(s, 'f1'):<19} | acc {s['accuracy']:.3f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0, help="print N disagreement cases with text")
    args = ap.parse_args()
    rng = np.random.default_rng(BOOT_SEED)

    df, missing = load_joined()
    if missing:
        print(f"WARNING: {missing} sampled comments have no LLM answer yet; evaluating the rest.\n")

    test_df = load_test_set()
    p = test_df.classifier_prob
    n_esc_pop = int(((p > UNCERTAIN_LOW) & (p < UNCERTAIN_HIGH)).sum())
    n_rest_pop = len(test_df) - n_esc_pop
    share_esc = n_esc_pop / len(test_df)

    # --- 1. Data check -------------------------------------------------------
    print("== Data check ==")
    print(f"Model: {MODEL_ID} | answers evaluated: {len(df)}")
    for g in ["random", "uncertain"]:
        sub = df[df.group == g]
        print(f"  {g:<10} n={len(sub)}  truly toxic={int(sub.label.sum())}")
    print(f"  LLM verdicts: {df.verdict.value_counts().to_dict()}")
    print(
        f"  Escalated band ({UNCERTAIN_LOW}-{UNCERTAIN_HIGH}) is {share_esc:.1%} of all test traffic.\n"
    )

    # Primary analysis: blocked/unparseable answers counted as "toxic" (a refusal gets flagged).
    dfp, strata = build_strata(df, n_esc_pop, n_rest_pop, "toxic")
    results = {"model": MODEL_ID, "blocked_treated_as": "toxic", "share_escalated": share_esc}

    # --- 2. Per-group --------------------------------------------------------
    print("== Per-group results (95% bootstrap intervals in brackets) ==")
    results["groups"] = {}
    for g in ["random", "uncertain"]:
        sub = dfp[dfp.group == g]
        note = "typical traffic" if g == "random" else "hard cases only"
        print(f"{g} sample ({note}, {int(sub.label.sum())} truly toxic of {len(sub)}):")
        results["groups"][g] = {}
        for name, col in [("classifier", "clf_pred"), ("LLM", "llm_pred")]:
            s = group_scores(sub.label.values, sub[col].values, rng)
            results["groups"][g][name] = s
            print_row(name, s)
        print()

    # --- 3. Overall estimate + hybrid preview -------------------------------
    print("== Estimated overall performance on ALL test traffic (reweighted) ==")
    overall = stratified_scores(strata, ["clf", "llm", "hybrid"], rng)
    results["overall"] = overall
    print_row("classifier only", overall["clf"])
    print_row("LLM only", overall["llm"])
    print_row("hybrid (PREVIEW)", overall["hybrid"])
    print("  (hybrid cutoffs were explored on this test set: exploratory, not final)\n")

    # --- 4. Cost and latency -------------------------------------------------
    df["cost"] = df.input_tokens / 1e6 * PRICE_IN_PER_M + df.output_tokens / 1e6 * PRICE_OUT_PER_M
    rnd, esc = df[df.group == "random"], df[df.escalated]
    llm_cost_1k = float(rnd.cost.mean() * 1000)
    hybrid_cost_1k = float(share_esc * esc.cost.mean() * 1000)
    clf_ms = None
    try:
        clf_ms = json.loads((RESULTS_DIR / f"{V0_VERSION}.json").read_text())["predict_ms_per_comment"]
    except Exception:  # noqa: BLE001
        pass

    print("== Cost and latency ==")
    print(f"  Estimated cost per 1,000 comments (list price, free tier not billed):")
    print(f"    classifier only: ~$0   | LLM only: ${llm_cost_1k:.3f} | hybrid (preview): ${hybrid_cost_1k:.3f}")
    print(
        f"  LLM latency per comment: median {df.latency_s.median():.2f}s, "
        f"95th percentile {df.latency_s.quantile(0.95):.2f}s, mean {df.latency_s.mean():.2f}s"
    )
    if clf_ms is not None:
        print(f"  Classifier latency: about {clf_ms:.4f} ms per comment")
    print(f"  Hybrid sends {share_esc:.1%} of traffic to the LLM.\n")
    results["cost_latency"] = {
        "llm_cost_per_1000": llm_cost_1k,
        "hybrid_cost_per_1000": hybrid_cost_1k,
        "llm_latency_median_s": float(df.latency_s.median()),
        "llm_latency_p95_s": float(df.latency_s.quantile(0.95)),
        "llm_latency_mean_s": float(df.latency_s.mean()),
        "classifier_ms_per_comment": clf_ms,
        "price_in_per_m": PRICE_IN_PER_M,
        "price_out_per_m": PRICE_OUT_PER_M,
    }

    # --- 5. Sensitivity to blocked answers -----------------------------------
    n_odd = int((~df.verdict.isin(["toxic", "not_toxic"])).sum())
    _, strata_alt = build_strata(df, n_esc_pop, n_rest_pop, "not_toxic")
    alt = stratified_scores(strata_alt, ["clf", "llm", "hybrid"], rng, boot=False)
    print("== Sensitivity: blocked/unparseable answers ==")
    print(f"  {n_odd} answers were blocked or unparseable.")
    print(
        f"  Overall LLM F1: {overall['llm']['f1']:.3f} (counted as toxic) vs "
        f"{alt['llm']['f1']:.3f} (counted as not toxic)\n"
    )
    results["blocked_sensitivity"] = {"n": n_odd, "llm_f1_alt": alt["llm"]["f1"]}

    # --- 6. Disagreements ----------------------------------------------------
    dis = dfp[dfp.clf_pred != dfp.llm_pred]
    RESULTS_DIR.mkdir(exist_ok=True)

    # Add original comment text to the disagreement DataFrame
    dis = dis.copy()
    dis["comment_text"] = dis["id"].map(test_df["text"])

    # Export disagreements with original comments
    dis[[
        "id",
        "group",
        "label",
        "comment_text",
        "classifier_prob",
        "clf_pred",
        "verdict",
        "reason"
    ]].to_csv(
        RESULTS_DIR / "v1_disagreements.csv",
        index=False
    )

    print(
        f"Classifier and LLM disagree on {len(dis)} of {len(dfp)} comments "
        "-> results/v1_disagreements.csv"
    )

    (RESULTS_DIR / "v1_evaluation.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
    print("Saved -> results/v1_evaluation.json")

    if args.show > 0 and len(dis) > 0:
        print(
            f"\n== {min(args.show, len(dis))} random disagreements "
            "(for your own error analysis) =="
        )

        for _, r in dis.sample(
            min(args.show, len(dis)), random_state=1
        ).iterrows():

            text = str(r["comment_text"]).replace("\n", " ")[:300]
            truth = "TOXIC" if r["label"] == 1 else "not toxic"

            print(
                f"\n[id {r['id']}] truth: {truth} "
                f"| classifier score {r['classifier_prob']:.2f} "
                f"| LLM: {r['verdict']} ({r['reason']})\n  \"{text}\""
            )

if __name__ == "__main__":
    main()