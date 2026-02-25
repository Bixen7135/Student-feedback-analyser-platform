# Evaluation Report

**Run ID:** `run_20260222T211105470735_0c18ee59`
**Generated:** 2026-02-22T21:13:36.641256+00:00
**Config Hash:** `0c18ee591d2d8696`
**Data Snapshot:** `53f553a7172324dc`
**Random Seed:** 42

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
| chi2 | 41.9394 |
| chi2_df | 24.0000 |
| chi2_p | 0.0131 |

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

**tfidf:** Macro F1 = 0.9747, Accuracy = 0.9838
**char_ngram:** Macro F1 = 0.9891, Accuracy = 0.9930

### Sentiment

**tfidf:** Macro F1 = 0.9951, Accuracy = 0.9957
**char_ngram:** Macro F1 = 0.9938, Accuracy = 0.9951

### Detail Level

**tfidf:** Macro F1 = 0.8687, Accuracy = 0.8701
**char_ngram:** Macro F1 = 0.9293, Accuracy = 0.9286

---

## 3. Fusion Experiments

Regression targets: psychometric factor scores (standardized).
Model: HuberRegressor. Lower MAE is better. Higher R² is better.

| Model | MAE | R² |
|-------|-----|-----|
| survey only | 0.7259 | -0.0296 |
| text only | 0.8442 | -0.3870 |
| late fusion | 0.7861 | -0.1772 |

### Deltas (fusion - survey_only)

Negative Δ MAE = improvement. Null results reported without spin.

| Factor | Δ MAE | Δ R² |
|--------|-------|------|
| program_quality | +0.0662 | -0.1629 |
| resources_teaching | +0.0554 | -0.1344 |
| digital_assessment | +0.0591 | -0.1455 |

---

## 4. Contradiction Monitoring

> Monitoring only. Predictions are not altered.

**Overall contradiction rate:** 0.07738095238095238
**Total flagged:** 143 / 1848

| Type | Rate |
|------|------|
| type_a_negative_high_score | 0.0157 |
| type_b_positive_low_score | 0.0617 |

---
