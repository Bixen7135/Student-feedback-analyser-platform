"""Dataset loader — reads CSV, renames Russian columns to English, validates schema."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# Mapping from Russian CSV column names to English internal names.
# Column 10 has a trailing newline in the original CSV — we strip all names.
COLUMN_MAP: dict[str, str] = {
    "№ анкеты": "survey_id",
    "1) Оцените, в какой мере оправдались Ваши ожидания, связанные с выбором образовательной программы": "item_1",
    "2) Оцените достаточность полученных теоретических знаний и приобретенных умений и навыков для осуществления эффективной профессиональной деятельности": "item_2",
    "3) Оцените соответствие целей, результатов обучения по ОП содержанию ОП (подбор дисциплин, кол-во кредитов, методы обучения, процедуры оценивания и т.п)": "item_3",
    "4) Оцените качество профессорско-преподавательского состава (знания и квалификация ППС, педагогические качества, объективность оценивания и др.)": "item_4",
    "5) Оцените качество материально-технической ресурсов необходимой для реализации ОП (здания и сооружения, состояние аудиторного фонда, обеспеченность лабораторным оборудованием и т.д.)": "item_5",
    "6) Оцените наличие, качество и достаточность учебной и учебно-методической литературы по дисциплинам ОП (учебников, учебно-методических пособий, методических рекомендаций по выполнению заданий СРО, курсовых и дипломных работ и т.д.)": "item_6",
    "7) Оцените удовлетворенность электронной информационно-образовательной средой (широкополосного доступа в Интернет, системами Moodle, Platonus, электронной библиотеки и др.)": "item_7",
    "8) Оцените качество организации, проведения и содержание профессиональных практик": "item_8",
    "9) Как Вы считаете, на сколько соответствуют знания, умения и навыки, полученные обучающимися при освоении образовательной программы, оценке, полученной на итоговой аттестации (при защите дипломной работы / комплексного экзамена)": "item_9",
    "Ваши пожелания и предложения": "text_feedback",
    "Язык": "language",
    "Длина": "detail_level",
    "Тональность_класс": "sentiment_class",
}

ITEM_COLS = [f"item_{i}" for i in range(1, 10)]
LABEL_COLS = ["language", "detail_level", "sentiment_class"]
EXPECTED_COLS = ["survey_id"] + ITEM_COLS + ["text_feedback"] + LABEL_COLS


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the student feedback CSV, rename columns, validate schema, and return DataFrame."""
    log.info("loading_dataset", path=str(path))
    df = pd.read_csv(path, encoding="utf-8", dtype=str)

    # Strip whitespace (including newlines) from column names
    df.columns = [c.strip() for c in df.columns]

    # Rename columns
    df = df.rename(columns=COLUMN_MAP)

    # Drop any columns not in our expected set
    known_cols = [c for c in EXPECTED_COLS if c in df.columns]
    df = df[known_cols]

    # Cast item columns to Int64 (nullable integer)
    for col in ITEM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Cast survey_id to Int64
    df["survey_id"] = pd.to_numeric(df["survey_id"], errors="coerce").astype("Int64")

    # Fill missing text with empty string
    df["text_feedback"] = df["text_feedback"].fillna("").astype(str)

    validate_schema(df)
    log.info("dataset_loaded", rows=len(df), cols=list(df.columns))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Assert that the DataFrame has all required columns and valid value ranges."""
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after renaming: {missing}")

    # Validate item ranges (allow 0-10)
    for col in ITEM_COLS:
        invalid = df[col].dropna()
        out_of_range = invalid[(invalid < 0) | (invalid > 10)]
        if len(out_of_range) > 0:
            raise ValueError(f"Column {col} has {len(out_of_range)} values outside [0, 10]")

    # Validate categorical columns
    valid_language = {"ru", "kz", "mixed"}
    valid_detail = {"short", "medium", "long"}
    valid_sentiment = {"positive", "neutral", "negative"}

    _check_cats(df, "language", valid_language)
    _check_cats(df, "detail_level", valid_detail)
    _check_cats(df, "sentiment_class", valid_sentiment)


def _check_cats(df: pd.DataFrame, col: str, valid: set[str]) -> None:
    actual = set(df[col].dropna().unique())
    unexpected = actual - valid
    if unexpected:
        raise ValueError(f"Column {col} has unexpected values: {unexpected}")
