"""
Utility helpers: logging, timing, serialization.
"""

import logging
import time
import functools
import json
from typing import Any, Dict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import Settings

settings = Settings()


# ── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(settings.LOG_FORMAT))
        logger.addHandler(ch)
        # File handler
        fh = logging.FileHandler(
            Path(settings.LOG_DIR) / "pipeline.log", mode="a"
        )
        fh.setFormatter(logging.Formatter(settings.LOG_FORMAT))
        logger.addHandler(fh)
    return logger


# ── Timing decorator ────────────────────────────────────────────────────────

def timer(func):
    """Decorator that logs execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"⏱  {func.__qualname__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ── Model I/O ───────────────────────────────────────────────────────────────

def save_model(model: Any, filename: str) -> Path:
    """Persist a model to disk."""
    path = Path(settings.MODEL_DIR) / filename
    joblib.dump(model, path)
    get_logger(__name__).info(f"💾 Model saved → {path}")
    return path


def load_model(filename: str) -> Any:
    """Load a persisted model."""
    path = Path(settings.MODEL_DIR) / filename
    model = joblib.load(path)
    get_logger(__name__).info(f"📦 Model loaded ← {path}")
    return model


# ── Metrics I/O ─────────────────────────────────────────────────────────────

def save_metrics(metrics: Dict[str, Any], filename: str) -> Path:
    """Save evaluation metrics as JSON."""
    path = Path(settings.OUTPUT_DIR) / filename
    # Convert numpy types to native Python types
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    get_logger(__name__).info(f"📊 Metrics saved → {path}")
    return path


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to CSV."""
    path = Path(settings.OUTPUT_DIR) / filename
    df.to_csv(path, index=False)
    get_logger(__name__).info(f"📄 DataFrame saved → {path} ({len(df)} rows)")
    return path


# ── Numpy / Pandas helpers ──────────────────────────────────────────────────

def describe_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of null values per column."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    summary = pd.DataFrame({
        "null_count": null_counts,
        "null_pct": null_pct.round(2),
        "dtype": df.dtypes,
    })
    return summary[summary["null_count"] > 0].sort_values(
        "null_pct", ascending=False
    )


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Return boolean mask of IQR-based outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


def cramers_v(confusion_matrix: np.ndarray) -> float:
    """Compute Cramér's V statistic for categorical association."""
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    denom = min(k_corr - 1, r_corr - 1)
    return np.sqrt(phi2_corr / denom) if denom > 0 else 0.0
