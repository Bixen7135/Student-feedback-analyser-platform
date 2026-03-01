# Model Card: Language — char_ngram

**Run ID:** `run_20260228T132912051916_0c18ee59`
**Generated:** 2026-02-28T13:31:48.248443+00:00

---

## Model Details

- **Task:** language
- **Model Type:** char_ngram
- **Framework:** scikit-learn

### Hyperparameters

```json
{
  "model_type": "char_ngram_logreg",
  "analyzer": "char_wb",
  "ngram_range": [
    2,
    5
  ],
  "max_features": 100000,
  "sublinear_tf": true,
  "C": 1.0,
  "max_iter": 1000,
  "class_weight": "balanced"
}
```

---

## Performance (Validation Set)

| Metric | Value |
|--------|-------|
| macro_f1 | 0.9905 |
| accuracy | 0.9940 |
| per_class_f1/kz | 0.9890 |
| per_class_f1/mixed | 0.9834 |
| per_class_f1/ru | 0.9991 |

---

## Intended Use

- This model classifies student feedback text by language.
- Intended for aggregate reporting and quality monitoring only.
- **Not for individual-level decisions about students or staff.**

## Limitations

- Trained on Russian/Kazakh educational feedback. May not generalize to other domains.
- Performance varies by language (see stratified metrics in evaluation report).
- Class imbalance: the training set has language-specific class imbalance (see report).
- Balanced class weights used during training to mitigate imbalance effects.

## No Causal Claims

This model identifies patterns in text. It does not establish causation.

## Training Data

- Dataset: Multilingual student feedback (Russian/Kazakh/mixed)
- Labels: Pre-existing annotations (language, sentiment, detail level)
- Split: 80% train, 10% validation, 10% test (stratified by sentiment class)