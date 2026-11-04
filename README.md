# Comment Moderator (hybrid classifier + LLM)

A comment moderation system built in small, measured versions. The goal is to test one idea: **a cheap trained classifier can resolve most comments on its own, and only the uncertain ones need an LLM.** Each version adds one piece and reports numbers, so every claim here is backed by a result in `results/`.

**Status:** V0 complete. LLM baseline (V1) is next.

## Headline result so far

On the Jigsaw Toxic Comment test split (31,915 comments), a TF-IDF + logistic regression baseline gets a toxic-class F1 of **0.75**. More useful for the hybrid idea: in a simulated router, the classifier resolves about **79% of comments alone at about 99% accuracy**, and the remaining ~21% (the slice that would go to an LLM) contains about **88% of the classifier's mistakes**.

That last number is a hypothesis check, not a final result. It shows where the errors are. Whether an LLM actually fixes them is what V1 will measure.

## Versions

| Version | What it adds | Status | Key numbers |
|---|---|---|---|
| V0 | TF-IDF + logistic regression, binary toxic vs. not | Done | Toxic F1 0.748 (P 0.658, R 0.866), macro-F1 0.857 |
| V0.1 | Confidence analysis and router simulation | Done | See below |
| V1 | LLM-only baseline on a fixed sample | Planned | |
| V2 | Router: classifier first, LLM for uncertain cases | Planned | |
| V3 | Probability calibration and validation-set threshold tuning | Planned | |
| V4+ | One of: human review queue, plain-language policy config, LLM-labeled retraining loop | Planned | |

## V0: baseline

- **Data:** Jigsaw Toxic Comment Classification Challenge, `train.csv` (about 160k comments), split 80/20 with a fixed seed, stratified on the label.
- **Label definition:** a comment is "toxic" if **any** of the six Jigsaw categories is 1 (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`). This differs slightly from Jigsaw's own `toxic` column.
- **Model:** TF-IDF (50k features, 1-2 grams) into logistic regression with `class_weight="balanced"`.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| not toxic | 0.984 | 0.949 | 0.966 | 28,670 |
| toxic | 0.658 | 0.866 | 0.748 | 3,245 |

The model catches most toxic comments but about one in three flags is a false alarm. Part of that is a deliberate trade-off from `class_weight="balanced"`, which pushes toward recall.

## V0.1: where does the classifier know what it's doing?

Accuracy by score band (score is the classifier's predicted probability of "toxic"):

| Score band | Share of traffic | Actually toxic | Accuracy |
|---|---|---|---|
| 0.0 - 0.1 | 56.4% | 0.2% | 99.8% |
| 0.1 - 0.3 | 23.6% | 2.2% | 97.8% |
| 0.3 - 0.7 | 10.6% | 17.7% | 67.3% |
| 0.7 - 0.9 | 3.3% | 56.1% | 56.1% |
| 0.9 - 1.0 | 6.2% | 93.8% | 93.8% |

Router simulation example (allow if score ≤ 0.2, remove if score ≥ 0.9, otherwise escalate):

- Resolved by the classifier alone: **78.9%** of traffic, at **99.1%** accuracy
- Escalated to the LLM: **21.1%** of traffic
- Share of all classifier errors inside the escalated slice: **87.8%**
- Confident errors the LLM would never see: 231

The full 16-combination sweep is in `results/v0_binary_confidence.json`, and the score histogram is `results/v0_binary_score_hist.png`.

## What failed or surprised me

- The 0.7-0.9 band is only 56% toxic, so a score of 0.8 does not mean 80% likely. The probabilities are distorted by class weighting and need calibration (planned for V3).
- The top band (0.9-1.0) is wrong about 6% of the time. Auto-removing there would silently delete a meaningful number of clean comments, so a real system should send some of that band to human review rather than removing automatically.

## Limitations

- **Thresholds were explored on the test set.** Comparing 16 cutoff pairs and reading results from the same data flatters the numbers. V3 will tune on a separate validation split.
- **Jigsaw labels are noisy**, which caps how well any model can score.
- **Known dataset bias:** Jigsaw-style data tends to over-flag comments that merely mention identity terms. This has not been tested here yet.
- **English only, binary only.** Multi-label categories and other languages are out of scope for now.
- **Single random split, no repeated runs**, so treat small differences as noise.

## Reproduce

```bash
pip install -r requirements.txt
# Download train.csv from the Kaggle Jigsaw Toxic Comment Classification
# Challenge (accept the competition rules first) and place it at data/train.csv

python src/v0_baseline.py     # trains, saves results/v0_binary.json and models/
python src/v0_confidence.py   # needs the saved model; writes confidence results
```

Run both from the project root. `data/` and `models/` are not committed.

## Repo layout

```
src/        training and analysis scripts
results/    metrics per version (JSON) and figures
```
