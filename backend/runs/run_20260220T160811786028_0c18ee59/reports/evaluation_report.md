# Evaluation Report

**Run ID:** `run_20260220T160811786028_0c18ee59`
**Generated:** 2026-02-20T16:09:30.895392+00:00
**Config Hash:** `0c18ee591d2d8696`
**Data Snapshot:** `7828d96cb3ec0dfc`
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
| error | float() argument must be a string or a real number, not 'Series' |

### Reliability

| Factor | Cronbach α | McDonald ω |
|--------|-----------|-----------|
| program_quality | 0.9240 | 0.0000 |
| resources_teaching | 0.9211 | 0.0000 |
| digital_assessment | 0.9225 | 0.0000 |

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
| survey only | 0.0000 | 1.0000 |
| text only | 0.0000 | 1.0000 |
| late fusion | 0.0000 | 1.0000 |

### Deltas (fusion - survey_only)

Negative Δ MAE = improvement. Null results reported without spin.

| Factor | Δ MAE | Δ R² |
|--------|-------|------|
| program_quality | +0.0000 | +0.0000 |
| resources_teaching | +0.0000 | +0.0000 |
| digital_assessment | +0.0000 | +0.0000 |

---

## 4. Contradiction Monitoring

> Monitoring only. Predictions are not altered.

**Overall contradiction rate:** 0.6520562770562771
**Total flagged:** 1205 / 1848

| Type | Rate |
|------|------|
| type_a_negative_high_score | 0.0947 |
| type_b_positive_low_score | 0.5574 |

---
