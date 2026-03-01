# Data Dictionary
**Generated:** 2026-03-01T10:34:04.765334+00:00

## Dataset: Student Feedback Survey

### Overview
- **Source:** Student questionnaire responses
- **Language:** Russian (60.8%), Kazakh (24.2%), Mixed (15.1%)
- **Total Records:** 18,476
- **Total Columns:** 14

---

## Column Definitions

### Metadata
| Column (English) | Column (Russian) | Type | Description |
|-----------------|-----------------|------|-------------|
| survey_id | № анкеты | Integer | Unique questionnaire identifier |

### Ordinal Survey Items (Psychometric Indicators)
| Column | Question | Scale | Factor |
|--------|----------|-------|--------|
| item_1 | Expectation fulfillment (educational program choice) | 1–10 | Program Quality |
| item_2 | Sufficiency of theoretical knowledge and skills | 0–10 | Program Quality |
| item_3 | Alignment of program goals with curriculum content | 0–10 | Program Quality |
| item_4 | Faculty quality (knowledge, pedagogy, objectivity) | 0–10 | Resources & Teaching |
| item_5 | Infrastructure and material-technical resources | 0–10 | Resources & Teaching |
| item_6 | Learning materials (textbooks, guides) | 0–10 | Resources & Teaching |
| item_7 | E-learning environment (internet, Moodle, Platonus) | 1–10 | Digital & Assessment |
| item_8 | Practical training quality (internships/practicum) | 0–10 | Digital & Assessment |
| item_9 | Knowledge–assessment alignment (diploma/exam) | 0–10 | Digital & Assessment |

**Note:** Items with scale 0–10 allow a zero rating. Items with scale 1–10 begin at 1.
**Note:** Survey items are the ONLY input for latent factor derivation. Text is never used.

### Text and Labels
| Column (English) | Column (Russian) | Type | Values | Description |
|-----------------|-----------------|------|--------|-------------|
| text_feedback | Ваши пожелания и предложения | Text | Free text | Open-ended student comment |
| language | Язык | Categorical | ru, kz, mixed | Language of the feedback text |
| detail_level | Длина | Categorical | short, medium, long | Annotated length/detail level |
| sentiment_class | Тональность_класс | Categorical | positive, neutral, negative | Sentiment class label |

---

## Class Distributions

### Language (Язык)
| Class | Count | Fraction |
|-------|-------|----------|
| ru (Russian) | ~11,234 | 60.8% |
| kz (Kazakh) | ~4,462 | 24.2% |
| mixed | ~2,780 | 15.1% |

### Detail Level (Длина)
| Class | Count | Fraction |
|-------|-------|----------|
| short | ~3,935 | 21.3% |
| medium | ~9,108 | 49.3% |
| long | ~5,433 | 29.4% |

### Sentiment Class (Тональность_класс)
| Class | Count | Fraction |
|-------|-------|----------|
| positive | ~10,301 | 55.8% |
| neutral | ~6,426 | 34.8% |
| negative | ~1,749 | 9.5% |

**Note:** Sentiment is class-imbalanced. Balanced class weights are used in training.

---

## Data Quality Notes

- Missing values: minimal (~0.01% in text_feedback, 0% elsewhere)
- PII redaction: emails, phone numbers, URLs, and student IDs are replaced with [REDACTED]
- Unicode normalization: NFC applied to all text
- Punctuation normalization: typographic quotes/dashes converted to ASCII equivalents

---

## Limitations and Caveats

- This data represents a specific institution and time period — generalization to other contexts is uncertain.
- Sentiment labels are human annotations and may contain inter-annotator disagreements.
- Mixed-language texts may confuse language-specific models.
- Not for individual-level decision making about students or staff.
- No causal claims should be derived from model outputs.
