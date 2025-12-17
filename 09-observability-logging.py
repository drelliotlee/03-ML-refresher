"""
observable_pipeline_only.py

Observability-only pandas pipeline.
No validation. No semantic correctness checks. Never raises.
"""

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from typing import Callable

import numpy as np
import pandas as pd


# ======================================================
# Logging setup
# ======================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = RotatingFileHandler(
    LOG_DIR / "pipeline.log",
    maxBytes=5_000_000,
    backupCount=5,
)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ======================================================
# Observability decorator
# ======================================================

def observe_step(step_name: str) -> Callable:
    def decorator(fn: Callable[[pd.DataFrame], pd.DataFrame]):
        def wrapper(df: pd.DataFrame) -> pd.DataFrame:
            rows_before = len(df)
            out = fn(df)
            rows_after = len(out)

            logger.info(
                "%s | rows_before=%d rows_after=%d delta=%+d",
                step_name,
                rows_before,
                rows_after,
                rows_after - rows_before,
            )

            if out.empty:
                logger.warning("%s | empty_dataframe", step_name)

            null_rates = out.isna().mean()
            high_nulls = null_rates[null_rates > 0.2]
            if not high_nulls.empty:
                logger.warning(
                    "%s | high_null_rates=%s",
                    step_name,
                    high_nulls.to_dict(),
                )

            numeric = out.select_dtypes(include="number")
            if not numeric.empty:
                stats = numeric.agg(["min", "max", "mean"]).to_dict()
                logger.info(
                    "%s | numeric_summary=%s",
                    step_name,
                    stats,
                )

            return out
        return wrapper
    return decorator

# ======================================================
# Pipeline steps
# ======================================================

@observe_step("record_baseline")
def record_baseline(df: pd.DataFrame) -> pd.DataFrame:
    return df

@observe_step("clean_country")
def clean_country(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        country=df["country"].str.upper().str.strip()
    )

@observe_step("filter_active")
def filter_active(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_active"]]

@observe_step("add_features")
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        is_senior=df["age"] >= 65,
    )

# ======================================================
# Pipeline execution
# ======================================================

def run_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df
        .pipe(record_baseline)
        .pipe(clean_country)
        .pipe(filter_active)
        .pipe(add_features)
    )