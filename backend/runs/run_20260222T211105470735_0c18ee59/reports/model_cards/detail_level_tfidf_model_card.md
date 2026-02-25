# Model Card: Detail Level — tfidf

**Run ID:** `run_20260222T211105470735_0c18ee59`
**Generated:** 2026-02-22T21:13:36.656789+00:00

---

## Model Details

- **Task:** detail_level
- **Model Type:** tfidf
- **Framework:** scikit-learn

### Hyperparameters

```json
{
  "model_type": "tfidf_logreg",
  "max_features": 50000,
  "ngram_range": [
    1,
    2
  ],
  "sublinear_tf": true,
  "C": 1.0,
  "max_iter": 1000,
  "class_weight": "balanced",
  "solver": "saga"
}
```

---

## Performance (Validation Set)

| Metric | Value |
|--------|-------|
| macro_f1 | 0.8946 |
| accuracy | 0.8929 |
| per_class_f1/long | 0.8928 |
| per_class_f1/medium | 0.8875 |
| per_class_f1/short | 0.9035 |

---

## Intended Use

- This model classifies student feedback text by detail_level.
- Intended for aggregate reporting and quality monitoring only.
- **Not for individual-level decisions about students or staff.**

## Limitations

- Trained on Russian/Kazakh educational feedback. May not generalize to other domains.
- Performance varies by language (see stratified metrics in evaluation report).
- Class imbalance: the training set has detail_level-specific class imbalance (see report).
- Balanced class weights used during training to mitigate imbalance effects.

## No Causal Claims

This model identifies patterns in text. It does not establish causation.

## Training Data

- Dataset: Multilingual student feedback (Russian/Kazakh/mixed)
- Labels: Pre-existing annotations (language, sentiment, detail level)
- Split: 80% train, 10% validation, 10% test (stratified by sentiment class)