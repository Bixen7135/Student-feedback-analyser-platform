# Model Card: Sentiment — tfidf

**Run ID:** `run_20260228T132912051916_0c18ee59`
**Generated:** 2026-02-28T13:31:48.252608+00:00

---

## Model Details

- **Task:** sentiment
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
| macro_f1 | 0.9975 |
| accuracy | 0.9968 |
| per_class_f1/negative | 1.0000 |
| per_class_f1/neutral | 0.9953 |
| per_class_f1/positive | 0.9971 |

---

## Intended Use

- This model classifies student feedback text by sentiment.
- Intended for aggregate reporting and quality monitoring only.
- **Not for individual-level decisions about students or staff.**

## Limitations

- Trained on Russian/Kazakh educational feedback. May not generalize to other domains.
- Performance varies by language (see stratified metrics in evaluation report).
- Class imbalance: the training set has sentiment-specific class imbalance (see report).
- Balanced class weights used during training to mitigate imbalance effects.

## No Causal Claims

This model identifies patterns in text. It does not establish causation.

## Training Data

- Dataset: Multilingual student feedback (Russian/Kazakh/mixed)
- Labels: Pre-existing annotations (language, sentiment, detail level)
- Split: 80% train, 10% validation, 10% test (stratified by sentiment class)