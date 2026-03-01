# Evaluation Report

**Run ID:** `run_20260228T132912051916_0c18ee59`
**Generated:** 2026-02-28T13:31:48.229517+00:00
**Config Hash:** `0c18ee591d2d8696`
**Data Snapshot:** `902f20578fb5bac8`
**Random Seed:** 50

---

> **Disclaimers:**
> - This system is for analytical reporting only.
> - Not for individual-level decision making.
> - No causal claims are made.
> - Latent quality factors are derived from survey items only, not from text.

---

## 1. Psychometrics

**Method:** semopy_cfa
**Observations:** 18476

### Fit Statistics

| Statistic | Value |
|-----------|-------|
| cfi | 0.9999 |
| rmsea | 0.0064 |
| gfi | 0.9998 |
| agfi | 0.9997 |
| nfi | 0.9998 |
| tli | 0.9999 |
| chi2 | 41.9743 |
| chi2_df | 24.0000 |
| chi2_p | 0.0130 |

### Reliability

| Factor | Cronbach α | McDonald ω |
|--------|-----------|-----------|
| program_quality | 0.9240 | 0.5979 |
| resources_teaching | 0.9211 | 0.5984 |
| digital_assessment | 0.9225 | 0.6002 |

---

## 2. Text Classification

All tasks trained independently. No shared gradients or loss functions.

### Language

**tfidf:** Macro F1 = 0.9664, Accuracy = 0.9811
**char_ngram:** Macro F1 = 0.9951, Accuracy = 0.9973

### Sentiment

**tfidf:** Macro F1 = 0.9958, Accuracy = 0.9957
**char_ngram:** Macro F1 = 0.9951, Accuracy = 0.9957

### Detail Level

**tfidf:** Macro F1 = 0.8903, Accuracy = 0.8885
**char_ngram:** Macro F1 = 0.9540, Accuracy = 0.9529

---

## 3. Fusion Experiments

Regression targets: psychometric factor scores (standardized).
Model: HuberRegressor. Lower MAE is better. Higher R² is better.

| Model | MAE | R² |
|-------|-----|-----|
| survey only | 0.7261 | -0.0297 |
| text only | 0.8406 | -0.3913 |
| late fusion | 0.7733 | -0.1653 |

### Deltas (fusion - survey_only)

Negative Δ MAE = improvement. Null results reported without spin.

| Factor | Δ MAE | Δ R² |
|--------|-------|------|
| program_quality | +0.0459 | -0.1379 |
| resources_teaching | +0.0496 | -0.1388 |
| digital_assessment | +0.0460 | -0.1300 |

---

## 4. Contradiction Monitoring

> Monitoring only. Predictions are not altered.

**Overall contradiction rate:** 0.07413419913419914
**Total flagged:** 137 / 1848

| Type | Rate |
|------|------|
| type_a_negative_high_score | 0.0146 |
| type_b_positive_low_score | 0.0595 |

---
