"""
V1: LLM-only baseline with Gemini 3.5 Flash-Lite.

Sends ONE comment per request to the LLM on two fixed samples drawn from the
same test split the classifier was evaluated on:
  - 500 random comments            (overall performance)
  - 500 "uncertain" comments       (classifier score between 0.2 and 0.9)

Design goals for a free-tier quota (500 requests/day, 15/minute):
  - Resumable: every answer is appended to a JSONL file; re-running skips done ids.
  - Stops cleanly at DAILY_CAP so you never burn the quota on retries.
  - Fixed sample: the manifest is saved once and reused on every run.
  - No comment text is saved, only ids (Jigsaw's license restricts redistribution).

Setup:
    pip install google-genai pandas scikit-learn joblib
    Set your key as an environment variable (never put it in the file):
        Windows (new terminal after running):  setx GEMINI_API_KEY "your-key"
        Mac/Linux:                             export GEMINI_API_KEY="your-key"
    Needs data/train.csv and models/v0_binary.joblib in the project root.

Run from the project root, once per day until it says all done:
    python src/v1_llm_baseline.py
"""

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from v0_baseline import (  # noqa: E402
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    VERSION as V0_VERSION,
    load_data,
    make_binary_label,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "gemini-3.5-flash-lite"

N_RANDOM = 500
N_UNCERTAIN = 500
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.2, 0.9  # classifier score band to escalate

DAILY_CAP = 450  # free tier allows 500/day; keep a margin for retries
MIN_SECONDS_BETWEEN_CALLS = 4.5  # free tier allows 15/minute
MAX_CHARS = 1500  # truncate very long comments to control token cost

# List prices per 1M tokens, used ONLY for a cost ESTIMATE (free tier isn't billed).
# Check Google's pricing page; prices change.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50

MANIFEST_PATH = RESULTS_DIR / "v1_sample.json"
ANSWERS_PATH = RESULTS_DIR / "v1_llm_answers.jsonl"

SYSTEM_PROMPT = """You are a content moderation classifier for an online comment section.

Label a comment "toxic" if it contains ANY of: general toxicity (rude, disrespectful, or hostile language), severe toxicity, obscene language, threats, insults, or identity-based hate. Otherwise label it "not_toxic".

Judge the comment itself. Quoting or discussing offensive content, or mentioning a group neutrally, is not toxic by itself.

Respond with JSON only, in this exact shape:
{"verdict": "toxic" or "not_toxic", "reason": "<at most 15 words>"}"""

# Moderation work involves toxic text; default safety filters can block it.
SAFETY_SETTINGS = [
    types.SafetySetting(category=c, threshold="BLOCK_NONE")
    for c in [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
]


# ---------------------------------------------------------------------------
# Data: recreate the exact V0 test split and attach classifier scores
# ---------------------------------------------------------------------------
def load_test_set() -> pd.DataFrame:
    texts, labels = load_data()
    y = make_binary_label(labels)
    _, X_test_txt, _, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=SEED, stratify=y
    )
    bundle = joblib.load(MODELS_DIR / f"{V0_VERSION}.joblib")
    probs = bundle["model"].predict_proba(bundle["vectorizer"].transform(X_test_txt))[:, 1]
    return pd.DataFrame(
        {"text": X_test_txt.values, "label": y_test.values, "classifier_prob": probs},
        index=X_test_txt.index,
    )


def build_manifest(df: pd.DataFrame) -> list:
    """Fixed, disjoint samples. Saved once; reused on every later run."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())

    rng = np.random.default_rng(SEED)
    random_ids = rng.choice(df.index.values, size=N_RANDOM, replace=False)

    pool = df[
        (df.classifier_prob > UNCERTAIN_LOW)
        & (df.classifier_prob < UNCERTAIN_HIGH)
        & ~df.index.isin(random_ids)
    ]
    uncertain_ids = rng.choice(pool.index.values, size=min(N_UNCERTAIN, len(pool)), replace=False)

    def rows(ids, group):
        return [
            {
                "id": int(i),
                "group": group,
                "label": int(df.loc[i, "label"]),
                "classifier_prob": round(float(df.loc[i, "classifier_prob"]), 6),
            }
            for i in ids
        ]

    r, u = rows(random_ids, "random"), rows(uncertain_ids, "uncertain")
    # Interleave so that a stopped run still has both groups covered.
    manifest = [x for pair in zip(r, u) for x in pair] + r[len(u):] + u[len(r):]

    RESULTS_DIR.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    print(f"Created sample manifest: {len(r)} random + {len(u)} uncertain -> {MANIFEST_PATH}")
    return manifest


def load_done_ids() -> set:
    if not ANSWERS_PATH.exists():
        return set()
    return {json.loads(line)["id"] for line in ANSWERS_PATH.read_text().splitlines() if line.strip()}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def classify(client, text: str):
    """Returns a result dict, or None if the API kept failing (caller should stop)."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        safety_settings=SAFETY_SETTINGS,
        # Note: no temperature/top_p/top_k. Google deprecated those for this model family.
    )
    contents = f"Comment:\n{text[:MAX_CHARS]}"

    for attempt in range(3):
        start = time.time()
        try:
            resp = client.models.generate_content(model=MODEL_ID, contents=contents, config=config)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            wait = 65 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 5
            print(f"  API error (attempt {attempt + 1}/3): {msg[:120]}  -> waiting {wait}s")
            time.sleep(wait)
            continue

        latency = time.time() - start
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = (getattr(usage, "candidates_token_count", 0) or 0) + (
            getattr(usage, "thoughts_token_count", 0) or 0
        )

        try:
            raw = resp.text
        except Exception:  # noqa: BLE001
            raw = None

        verdict, reason = "blocked", ""
        if raw:
            try:
                data = json.loads(raw)
                verdict = data.get("verdict", "unparseable")
                reason = str(data.get("reason", ""))[:200]
                if verdict not in ("toxic", "not_toxic"):
                    verdict = "unparseable"
            except json.JSONDecodeError:
                verdict = "unparseable"

        return {
            "verdict": verdict,
            "reason": reason,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_s": round(latency, 3),
        }
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Set the GEMINI_API_KEY environment variable first (see the top of this file).")

    df = load_test_set()
    manifest = build_manifest(df)
    done = load_done_ids()
    todo = [m for m in manifest if m["id"] not in done]
    print(f"Sample size: {len(manifest)} | already done: {len(done)} | remaining: {len(todo)}")
    if not todo:
        print("All done. Next step: write the evaluation script.")
        return

    client = genai.Client()
    RESULTS_DIR.mkdir(exist_ok=True)
    calls_today = 0
    last_call = 0.0

    with ANSWERS_PATH.open("a") as f:
        for m in todo:
            if calls_today >= DAILY_CAP:
                print(f"Reached daily cap of {DAILY_CAP} calls. Run again tomorrow.")
                break

            wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.time()
            calls_today += 1

            result = classify(client, df.loc[m["id"], "text"])
            if result is None:
                print("Stopping: the API kept failing (probably a quota limit). Run again later.")
                break

            record = {"id": m["id"], "group": m["group"], "model": MODEL_ID, **result}
            f.write(json.dumps(record) + "\n")
            f.flush()

            if calls_today % 25 == 0:
                print(f"  {calls_today} calls this session...")

    # Summary of everything collected so far
    rows = [json.loads(l) for l in ANSWERS_PATH.read_text().splitlines() if l.strip()]
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    est_cost = tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M
    counts = pd.Series([r["verdict"] for r in rows]).value_counts().to_dict()
    print("\n== Progress ==")
    print(f"Answers collected: {len(rows)} / {len(manifest)}")
    print(f"Verdicts: {counts}  (watch 'blocked' and 'unparseable')")
    print(f"Tokens: {tin} in, {tout} out | estimated cost so far: ${est_cost:.4f} (list price, not billed)")
    if rows:
        print(f"Mean latency: {np.mean([r['latency_s'] for r in rows]):.2f}s per comment")


if __name__ == "__main__":
    main()