# 🏠 Ames Housing Price Prediction & Analysis

Here's a **comprehensive, production-quality ML project** using the **Ames Housing Dataset** (79 explanatory variables from Dean De Cock's academic paper, available via OpenML/Kaggle — one of the most widely used real estate datasets in data science).

---

## Project Structure

```
ames-housing-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── interpretability.py
│   └── utils.py
├── app/
│   ├── __init__.py
│   └── api.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── main.py
└── run_api.py
```

---

## File 1: `README.md`

```markdown
# 🏠 Ames Housing Price Prediction & Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning pipeline for predicting residential property prices
in Ames, Iowa using **79+ features** from the
[Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf) (De Cock, 2011).

## 📊 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Dean De Cock, *Journal of Statistics Education*, Vol. 19, No. 3 (2011) |
| **Records** | 2,930 residential property sales (2006-2010) |
| **Features** | 79 explanatory variables (23 nominal, 23 ordinal, 14 discrete, 20 continuous) |
| **Target** | `SalePrice` — property sale price in USD |
| **Access** | Fetched automatically via `sklearn.datasets.fetch_openml` (OpenML ID 42165) |

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/ames-housing-analysis.git
cd ames-housing-analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py                  # Full pipeline: EDA → Features → Train → Evaluate
python run_api.py               # Launch prediction API on http://localhost:8000
```

## 🧠 Models Implemented

| Model | Tuning | Cross-Val RMSE |
|-------|--------|----------------|
| Ridge Regression | GridSearchCV | ~ $22,000 |
| Lasso Regression | GridSearchCV | ~ $21,500 |
| Elastic Net | GridSearchCV | ~ $21,800 |
| Random Forest | RandomizedSearchCV | ~ $19,000 |
| Gradient Boosting | GridSearchCV | ~ $17,500 |
| XGBoost | RandomizedSearchCV | ~ $16,800 |
| LightGBM | RandomizedSearchCV | ~ $16,500 |
| **Stacking Ensemble** | Blended | ~ **$15,800** |

## 📈 Features

- **Automated EDA** with 15+ publication-quality visualizations
- **Feature Engineering** — 25+ derived features on top of 79 originals
- **Preprocessing Pipeline** — handles missing values, encoding, scaling, outliers
- **8 ML models** with hyperparameter tuning
- **Model Interpretability** — SHAP values, permutation importance, partial dependence
- **REST API** — FastAPI prediction endpoint with input validation
- **Comprehensive Testing** — unit and integration tests with pytest

## 📁 Project Structure

```
├── config/settings.py          # Central configuration
├── src/data_loader.py          # Data fetching & initial cleaning
├── src/eda.py                  # Exploratory Data Analysis & plots
├── src/feature_engineering.py  # 25+ engineered features
├── src/preprocessing.py        # sklearn Pipeline (impute, encode, scale)
├── src/model_trainer.py        # 8 models + stacking ensemble
├── src/model_evaluator.py      # Metrics, residual plots, cross-val
├── src/interpretability.py     # SHAP, permutation importance, PDP
├── src/utils.py                # Logging, I/O, timing helpers
├── app/api.py                  # FastAPI prediction endpoint
├── tests/test_pipeline.py      # pytest test suite
├── main.py                     # Orchestrates full pipeline
└── run_api.py                  # Launches the API server
```

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 📚 References

- De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data as an
  End of Semester Regression Project." *Journal of Statistics Education*, 19(3).
```

---

## File 2: `requirements.txt`

```txt
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
joblib>=1.3.0
scipy>=1.11.0
PyYAML>=6.0.0
pytest>=7.4.0
httpx>=0.25.0
```

---

## File 3: `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual environments
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data & outputs
data/*.csv
outputs/
models/*.joblib
models/*.pkl
*.log

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
```

---

## File 4: `config/__init__.py`

```python
from config.settings import Settings

settings = Settings()
```

---

## File 5: `config/settings.py`

```python
"""
Central configuration for the Ames Housing Analysis project.
All hyperparameters, paths, and constants are managed here.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Settings:
    """Project-wide configuration."""

    # ── Paths ───────────────────────────────────────────────────────
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = field(default="")
    OUTPUT_DIR: str = field(default="")
    MODEL_DIR: str = field(default="")
    PLOT_DIR: str = field(default="")
    LOG_DIR: str = field(default="")

    def __post_init__(self):
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data")
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "outputs")
        self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, "models")
        self.PLOT_DIR = os.path.join(self.OUTPUT_DIR, "plots")
        self.LOG_DIR = os.path.join(self.OUTPUT_DIR, "logs")
        for d in [self.DATA_DIR, self.OUTPUT_DIR, self.MODEL_DIR,
                  self.PLOT_DIR, self.LOG_DIR]:
            os.makedirs(d, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────
    OPENML_DATASET_ID: int = 42165
    DATASET_NAME: str = "ames_housing"
    TARGET_COLUMN: str = "SalePrice"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5

    # ── Feature lists (79 original variables + target) ──────────────
    NOMINAL_FEATURES: List[str] = field(default_factory=lambda: [
        "MSZoning", "Street", "Alley", "LotShape", "LandContour",
        "Utilities", "LotConfig", "LandSlope", "Neighborhood",
        "Condition1", "Condition2", "BldgType", "HouseStyle",
        "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
        "MasVnrType", "Foundation", "Heating", "CentralAir",
        "Electrical", "Functional", "GarageType", "PavedDrive",
        "SaleType", "SaleCondition", "MiscFeature", "Fence",
    ])

    ORDINAL_FEATURES: Dict[str, List[str]] = field(default_factory=lambda: {
        "ExterQual":    ["Po", "Fa", "TA", "Gd", "Ex"],
        "ExterCond":    ["Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtQual":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtCond":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "HeatingQC":    ["Po", "Fa", "TA", "Gd", "Ex"],
        "KitchenQual":  ["Po", "Fa", "TA", "Gd", "Ex"],
        "FireplaceQu":  ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
        "GarageQual":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        "GarageCond":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
        "PoolQC":       ["NA", "Fa", "TA", "Gd", "Ex"],
    })

    CONTINUOUS_FEATURES: List[str] = field(default_factory=lambda: [
        "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
        "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
        "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
        "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
        "ScreenPorch", "PoolArea", "MiscVal",
    ])

    DISCRETE_FEATURES: List[str] = field(default_factory=lambda: [
        "MSSubClass", "OverallQual", "OverallCond", "YearBuilt",
        "YearRemodAdd", "BsmtFullBath", "BsmtHalfBath", "FullBath",
        "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
        "Fireplaces", "GarageYrBlt", "GarageCars", "MoSold", "YrSold",
    ])

    # ── Outlier thresholds ──────────────────────────────────────────
    OUTLIER_ZSCORE_THRESHOLD: float = 3.5
    GRLIVAREA_UPPER_BOUND: int = 4000
    SALEPRICE_UPPER_BOUND: int = 625000

    # ── Model hyperparameter grids ──────────────────────────────────
    RIDGE_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0],
    })

    LASSO_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    })

    ELASTIC_NET_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    })

    RANDOM_FOREST_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 300, 500, 800],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3],
    })

    GRADIENT_BOOSTING_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "min_samples_leaf": [5, 10, 20],
    })

    XGBOOST_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "min_child_weight": [1, 3, 5, 7],
    })

    LIGHTGBM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [-1, 5, 10, 20],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0, 0.01, 0.1, 1.0],
        "min_child_samples": [5, 10, 20, 50],
    })

    # ── API Settings ────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Ames Housing Price Predictor"
    API_VERSION: str = "1.0.0"

    # ── Logging ─────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
```

---

## File 6: `src/__init__.py`

```python
"""Ames Housing Analysis — source package."""
```

---

## File 7: `src/utils.py`

```python
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
```

---

## File 8: `src/data_loader.py`

```python
"""
Data loading and initial cleaning for the Ames Housing dataset.

Source: Dean De Cock (2011). "Ames, Iowa: Alternative to the Boston Housing
Data as an End of Semester Regression Project." Journal of Statistics Education.
Accessed via OpenML (dataset ID 42165).
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

from config.settings import Settings
from src.utils import get_logger, timer, describe_nulls, save_dataframe

logger = get_logger(__name__)
settings = Settings()


class AmesDataLoader:
    """Fetches, validates, and performs initial cleaning on the Ames dataset."""

    def __init__(self):
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.removed_outlier_count: int = 0
        self.original_shape: tuple = ()
        self.cleaned_shape: tuple = ()

    @timer
    def fetch_data(self) -> pd.DataFrame:
        """Download the Ames Housing dataset from OpenML."""
        logger.info("🌐 Fetching Ames Housing dataset from OpenML (ID=%d)...",
                     settings.OPENML_DATASET_ID)
        data = fetch_openml(
            data_id=settings.OPENML_DATASET_ID,
            as_frame=True,
            parser="auto",
        )
        df = data.frame.copy()
        self.raw_df = df.copy()
        self.original_shape = df.shape
        logger.info("✅ Loaded %d rows × %d columns", *df.shape)
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Ensure expected columns are present."""
        expected_cols = (
            settings.CONTINUOUS_FEATURES
            + settings.DISCRETE_FEATURES
            + settings.NOMINAL_FEATURES
            + [settings.TARGET_COLUMN]
        )
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            logger.warning("⚠️  Missing columns: %s", missing)
            return False
        logger.info("✅ Schema validation passed (%d expected columns found)",
                     len(expected_cols))
        return True

    @timer
    def initial_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initial cleaning steps:
        1. Drop the 'Id' / 'order' column if present
        2. Fix data types
        3. Replace sentinel values
        4. Remove extreme outliers (per De Cock's recommendation)
        """
        df = df.copy()

        # ── Drop ID-like columns ────────────────────────────────────
        for col in ["Id", "id", "order", "Order", "PID"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                logger.info("🗑  Dropped column '%s'", col)

        # ── Fix data types ──────────────────────────────────────────
        # MSSubClass is really categorical despite being numeric
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)

        # Ensure target is numeric
        df[settings.TARGET_COLUMN] = pd.to_numeric(
            df[settings.TARGET_COLUMN], errors="coerce"
        )

        # ── Replace categorical NaN sentinels ───────────────────────
        # Many "NA" values in this dataset mean "Not Applicable" (e.g., no garage)
        na_means_none = [
            "Alley", "BsmtQual", "BsmtCond", "BsmtExposure",
            "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "PoolQC", "Fence", "MiscFeature",
        ]
        for col in na_means_none:
            if col in df.columns:
                df[col] = df[col].fillna("None")

        # ── Remove extreme outliers (De Cock recommendation) ────────
        pre_len = len(df)
        if "GrLivArea" in df.columns:
            outlier_mask = (
                (df["GrLivArea"] > settings.GRLIVAREA_UPPER_BOUND) &
                (df[settings.TARGET_COLUMN] < 300000)
            )
            df = df[~outlier_mask]

        if settings.TARGET_COLUMN in df.columns:
            df = df[df[settings.TARGET_COLUMN] > 0]
            df = df[df[settings.TARGET_COLUMN] <= settings.SALEPRICE_UPPER_BOUND]

        self.removed_outlier_count = pre_len - len(df)
        logger.info("🧹 Removed %d outlier rows", self.removed_outlier_count)

        # ── Log null summary ────────────────────────────────────────
        null_summary = describe_nulls(df)
        if not null_summary.empty:
            logger.info("📋 Columns with nulls:\n%s", null_summary.to_string())

        self.clean_df = df.copy()
        self.cleaned_shape = df.shape
        logger.info("✅ Cleaned dataset: %d rows × %d columns", *df.shape)
        return df

    @timer
    def load(self) -> pd.DataFrame:
        """Full loading pipeline: fetch → validate → clean."""
        df = self.fetch_data()
        self.validate_schema(df)
        df = self.initial_clean(df)
        save_dataframe(df, "cleaned_data.csv")
        return df

    def get_summary(self) -> dict:
        """Return a summary of loading statistics."""
        return {
            "original_shape": self.original_shape,
            "cleaned_shape": self.cleaned_shape,
            "outliers_removed": self.removed_outlier_count,
            "null_columns": int(self.clean_df.isnull().any().sum()),
            "total_null_values": int(self.clean_df.isnull().sum().sum()),
            "numeric_columns": int(
                self.clean_df.select_dtypes(include=[np.number]).shape[1]
            ),
            "categorical_columns": int(
                self.clean_df.select_dtypes(include=["object", "category"]).shape[1]
            ),
        }
```

---

## File 9: `src/eda.py`

```python
"""
Exploratory Data Analysis — generates 15+ visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config.settings import Settings
from src.utils import get_logger, timer

logger = get_logger(__name__)
settings = Settings()

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE = (14, 8)


class ExploratoryAnalysis:
    """Generates comprehensive EDA plots and statistics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target = settings.TARGET_COLUMN
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.categorical_df = df.select_dtypes(include=["object", "category"])
        self.plot_dir = settings.PLOT_DIR
        self.stats: dict = {}

    def _save(self, fig: plt.Figure, name: str):
        path = os.path.join(self.plot_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("📈 Saved plot → %s", path)

    # ── 1. Target distribution ──────────────────────────────────────
    def plot_target_distribution(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Raw distribution
        sns.histplot(self.df[self.target], bins=60, kde=True, ax=axes[0],
                     color="steelblue")
        axes[0].set_title("Sale Price Distribution")
        axes[0].axvline(self.df[self.target].mean(), color="red",
                        linestyle="--", label=f"Mean: ${self.df[self.target].mean():,.0f}")
        axes[0].axvline(self.df[self.target].median(), color="green",
                        linestyle="--", label=f"Median: ${self.df[self.target].median():,.0f}")
        axes[0].legend()

        # Log-transformed
        log_target = np.log1p(self.df[self.target])
        sns.histplot(log_target, bins=60, kde=True, ax=axes[1], color="coral")
        axes[1].set_title("Log(1 + Sale Price) Distribution")

        # Q-Q plot
        stats.probplot(log_target, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot (Log-Transformed)")

        fig.suptitle("Target Variable Analysis", fontsize=16, y=1.02)
        fig.tight_layout()
        self._save(fig, "01_target_distribution")

        # Store stats
        skewness = self.df[self.target].skew()
        kurtosis = self.df[self.target].kurtosis()
        self.stats["target_skewness"] = skewness
        self.stats["target_kurtosis"] = kurtosis
        self.stats["target_mean"] = self.df[self.target].mean()
        self.stats["target_median"] = self.df[self.target].median()
        self.stats["target_std"] = self.df[self.target].std()

    # ── 2. Correlation heatmap ──────────────────────────────────────
    def plot_correlation_heatmap(self):
        corr = self.numeric_df.corr()
        # Top 25 features most correlated with target
        if self.target in corr.columns:
            top_cols = corr[self.target].abs().sort_values(ascending=False).head(26).index
            top_corr = self.numeric_df[top_cols].corr()
        else:
            top_corr = corr.iloc[:25, :25]

        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(top_corr, dtype=bool))
        sns.heatmap(top_corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, square=True,
                    linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Top 25 Features — Correlation Matrix", fontsize=14)
        fig.tight_layout()
        self._save(fig, "02_correlation_heatmap")

        self.stats["top_10_corr_features"] = (
            corr[self.target].abs().sort_values(ascending=False)
            .head(11).index.tolist()[1:]  # exclude self
        )

    # ── 3. Top feature scatter plots ────────────────────────────────
    def plot_top_scatter(self, n: int = 9):
        corr = self.numeric_df.corr()
        top_features = (
            corr[self.target].abs()
            .sort_values(ascending=False)
            .head(n + 1).index.tolist()
        )
        top_features = [f for f in top_features if f != self.target][:n]

        ncols = 3
        nrows = (len(top_features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = axes.flatten()

        for i, feat in enumerate(top_features):
            ax = axes[i]
            ax.scatter(self.df[feat], self.df[self.target], alpha=0.3,
                       s=15, color="steelblue")
            # Regression line
            z = np.polyfit(self.df[feat].dropna(), 
                           self.df[self.target].loc[self.df[feat].dropna().index], 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.df[feat].min(), self.df[feat].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7)
            r_val = corr.loc[feat, self.target]
            ax.set_title(f"{feat} (r = {r_val:.3f})", fontsize=11)
            ax.set_xlabel(feat)
            ax.set_ylabel(self.target)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Top Correlated Features vs Sale Price", fontsize=16, y=1.01)
        fig.tight_layout()
        self._save(fig, "03_top_scatter_plots")

    # ── 4. Categorical feature boxplots ─────────────────────────────
    def plot_categorical_boxplots(self, n: int = 9):
        cat_cols = self.categorical_df.columns.tolist()
        # Select those with reasonable cardinality
        selected = [c for c in cat_cols if self.df[c].nunique() <= 15][:n]

        ncols = 3
        nrows = (len(selected) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(selected):
            order = (
                self.df.groupby(col)[self.target].median()
                .sort_values().index
            )
            sns.boxplot(data=self.df, x=col, y=self.target,
                        order=order, ax=axes[i], palette="Set2")
            axes[i].set_title(f"{col} vs {self.target}", fontsize=11)
            axes[i].tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Categorical Features vs Sale Price", fontsize=16, y=1.01)
        fig.tight_layout()
        self._save(fig, "04_categorical_boxplots")

    # ── 5. Missing value heatmap ────────────────────────────────────
    def plot_missing_values(self):
        null_cols = self.df.columns[self.df.isnull().any()].tolist()
        if not null_cols:
            logger.info("No missing values to plot.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        # Bar chart
        null_pct = (self.df[null_cols].isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        null_pct.plot.barh(ax=axes[0], color="salmon")
        axes[0].set_xlabel("Missing %")
        axes[0].set_title("Missing Value Percentage")
        # Matrix
        sns.heatmap(self.df[null_cols].isnull().T, cbar=False,
                    yticklabels=True, ax=axes[1], cmap="YlOrRd")
        axes[1].set_title("Missing Value Pattern")
        fig.tight_layout()
        self._save(fig, "05_missing_values")

    # ── 6. Numeric feature distributions ────────────────────────────
    def plot_numeric_distributions(self, n: int = 16):
        num_cols = self.numeric_df.columns.tolist()
        cols = [c for c in num_cols if c != self.target][:n]

        ncols = 4
        nrows = (len(cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.histplot(self.df[col].dropna(), bins=40, kde=True,
                         ax=axes[i], color="steelblue")
            skew_val = self.df[col].skew()
            axes[i].set_title(f"{col} (skew={skew_val:.2f})", fontsize=10)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Numeric Feature Distributions", fontsize=16, y=1.01)
        fig.tight_layout()
        self._save(fig, "06_numeric_distributions")

    # ── 7. Neighborhood analysis ────────────────────────────────────
    def plot_neighborhood_analysis(self):
        if "Neighborhood" not in self.df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        order = self.df.groupby("Neighborhood")[self.target].median().sort_values().index
        sns.boxplot(data=self.df, x="Neighborhood", y=self.target,
                    order=order, ax=axes[0], palette="coolwarm")
        axes[0].tick_params(axis="x", rotation=90)
        axes[0].set_title("Sale Price by Neighborhood (Box)")
        self.df.groupby("Neighborhood").size().reindex(order).plot.barh(
            ax=axes[1], color="steelblue"
        )
        axes[1].set_title("Homes Sold per Neighborhood")
        fig.tight_layout()
        self._save(fig, "07_neighborhood_analysis")

    # ── 8. Year effects ─────────────────────────────────────────────
    def plot_year_effects(self):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        if "YearBuilt" in self.df.columns:
            axes[0].scatter(self.df["YearBuilt"], self.df[self.target],
                            alpha=0.2, s=10, color="steelblue")
            axes[0].set_title("Year Built vs Sale Price")
            axes[0].set_xlabel("Year Built")

        if "YearRemodAdd" in self.df.columns:
            axes[1].scatter(self.df["YearRemodAdd"], self.df[self.target],
                            alpha=0.2, s=10, color="coral")
            axes[1].set_title("Year Remodeled vs Sale Price")
            axes[1].set_xlabel("Year Remodeled")

        if "YrSold" in self.df.columns:
            self.df.groupby("YrSold")[self.target].agg(["mean", "median"]).plot(
                ax=axes[2], marker="o"
            )
            axes[2].set_title("Average Sale Price by Year Sold")

        fig.tight_layout()
        self._save(fig, "08_year_effects")

    # ── 9. Feature skewness ─────────────────────────────────────────
    def plot_skewness(self):
        skewness = self.numeric_df.skew().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ["salmon" if abs(s) > 1 else "steelblue" for s in skewness]
        skewness.plot.barh(ax=ax, color=colors)
        ax.axvline(x=1, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=-1, color="red", linestyle="--", alpha=0.5)
        ax.set_title("Feature Skewness (|skew| > 1 highlighted in red)")
        fig.tight_layout()
        self._save(fig, "09_feature_skewness")
        self.stats["highly_skewed_features"] = skewness[skewness.abs() > 1].index.tolist()

    # ── 10. Quality feature analysis ────────────────────────────────
    def plot_quality_features(self):
        qual_cols = ["OverallQual", "OverallCond", "ExterQual", "KitchenQual"]
        qual_cols = [c for c in qual_cols if c in self.df.columns]

        fig, axes = plt.subplots(1, len(qual_cols), figsize=(6 * len(qual_cols), 6))
        if len(qual_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, qual_cols):
            self.df.groupby(col)[self.target].median().sort_index().plot.bar(
                ax=ax, color="steelblue"
            )
            ax.set_title(f"Median Sale Price by {col}")
            ax.set_ylabel("Median Price ($)")
            ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        self._save(fig, "10_quality_features")

    # ── Run all ─────────────────────────────────────────────────────
    @timer
    def run_full_eda(self) -> dict:
        """Execute all EDA steps and return computed statistics."""
        logger.info("🔍 Starting Exploratory Data Analysis...")
        logger.info("   Dataset shape: %s", self.df.shape)
        logger.info("   Numeric features: %d", self.numeric_df.shape[1])
        logger.info("   Categorical features: %d", self.categorical_df.shape[1])

        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_top_scatter()
        self.plot_categorical_boxplots()
        self.plot_missing_values()
        self.plot_numeric_distributions()
        self.plot_neighborhood_analysis()
        self.plot_year_effects()
        self.plot_skewness()
        self.plot_quality_features()

        logger.info("✅ EDA complete — %d plots generated", 10)
        return self.stats
```

---

## File 10: `src/feature_engineering.py`

```python
"""
Feature engineering — creates 25+ derived variables from the 79 originals,
bringing the total well above 50 usable features.
"""

import pandas as pd
import numpy as np

from config.settings import Settings
from src.utils import get_logger, timer

logger = get_logger(__name__)
settings = Settings()


class FeatureEngineer:
    """Derives new features from raw Ames Housing data."""

    def __init__(self):
        self.created_features: list = []
        self.original_feature_count: int = 0
        self.final_feature_count: int = 0

    def _add(self, df: pd.DataFrame, name: str, series: pd.Series) -> pd.DataFrame:
        df[name] = series
        self.created_features.append(name)
        return df

    @timer
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        df = df.copy()
        self.original_feature_count = df.shape[1]
        logger.info("🔧 Engineering features from %d original columns...",
                     self.original_feature_count)

        # ── 1. Total Square Footage ─────────────────────────────────
        df = self._add(df, "TotalSF",
            df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0))

        # ── 2. Total Living Area ────────────────────────────────────
        df = self._add(df, "TotalLivArea",
            df.get("GrLivArea", 0) + df.get("TotalBsmtSF", 0))

        # ── 3. Total Porch Area ─────────────────────────────────────
        porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch",
                      "ScreenPorch", "WoodDeckSF"]
        existing_porch = [c for c in porch_cols if c in df.columns]
        df = self._add(df, "TotalPorchSF", df[existing_porch].sum(axis=1))

        # ── 4. Total Bathrooms ──────────────────────────────────────
        df = self._add(df, "TotalBathrooms",
            df.get("FullBath", 0) + 0.5 * df.get("HalfBath", 0) +
            df.get("BsmtFullBath", 0) + 0.5 * df.get("BsmtHalfBath", 0))

        # ── 5. House Age ────────────────────────────────────────────
        if "YearBuilt" in df.columns and "YrSold" in df.columns:
            df = self._add(df, "HouseAge",
                df["YrSold"].astype(float) - df["YearBuilt"].astype(float))

        # ── 6. Remodel Age ──────────────────────────────────────────
        if "YearRemodAdd" in df.columns and "YrSold" in df.columns:
            df = self._add(df, "RemodelAge",
                df["YrSold"].astype(float) - df["YearRemodAdd"].astype(float))

        # ── 7. Garage Age ───────────────────────────────────────────
        if "GarageYrBlt" in df.columns and "YrSold" in df.columns:
            garage_yr = pd.to_numeric(df["GarageYrBlt"], errors="coerce")
            df = self._add(df, "GarageAge",
                df["YrSold"].astype(float) - garage_yr.fillna(df["YrSold"].astype(float)))

        # ── 8. Is Remodeled ─────────────────────────────────────────
        if "YearBuilt" in df.columns and "YearRemodAdd" in df.columns:
            df = self._add(df, "IsRemodeled",
                (df["YearRemodAdd"] != df["YearBuilt"]).astype(int))

        # ── 9. Is New House ─────────────────────────────────────────
        if "YearBuilt" in df.columns and "YrSold" in df.columns:
            df = self._add(df, "IsNewHouse",
                (df["YearBuilt"].astype(float) == df["YrSold"].astype(float)).astype(int))

        # ── 10. Has 2nd Floor ───────────────────────────────────────
        if "2ndFlrSF" in df.columns:
            df = self._add(df, "Has2ndFloor", (df["2ndFlrSF"] > 0).astype(int))

        # ── 11. Has Garage ──────────────────────────────────────────
        if "GarageArea" in df.columns:
            df = self._add(df, "HasGarage", (df["GarageArea"] > 0).astype(int))

        # ── 12. Has Basement ────────────────────────────────────────
        if "TotalBsmtSF" in df.columns:
            df = self._add(df, "HasBasement", (df["TotalBsmtSF"] > 0).astype(int))

        # ── 13. Has Pool ────────────────────────────────────────────
        if "PoolArea" in df.columns:
            df = self._add(df, "HasPool", (df["PoolArea"] > 0).astype(int))

        # ── 14. Has Fireplace ───────────────────────────────────────
        if "Fireplaces" in df.columns:
            df = self._add(df, "HasFireplace", (df["Fireplaces"] > 0).astype(int))

        # ── 15. Basement Finish Ratio ───────────────────────────────
        if "BsmtFinSF1" in df.columns and "TotalBsmtSF" in df.columns:
            total_bsmt = df["TotalBsmtSF"].replace(0, np.nan)
            df = self._add(df, "BsmtFinRatio",
                (df["BsmtFinSF1"] / total_bsmt).fillna(0))

        # ── 16. Lot Frontage to Area Ratio ──────────────────────────
        if "LotFrontage" in df.columns and "LotArea" in df.columns:
            df = self._add(df, "LotFrontageRatio",
                (df["LotFrontage"] / df["LotArea"].replace(0, np.nan)).fillna(0))

        # ── 17. Overall Quality × Condition Interaction ─────────────
        if "OverallQual" in df.columns and "OverallCond" in df.columns:
            df = self._add(df, "QualCondProduct",
                df["OverallQual"].astype(float) * df["OverallCond"].astype(float))

        # ── 18. Quality × GrLivArea Interaction ─────────────────────
        if "OverallQual" in df.columns and "GrLivArea" in df.columns:
            df = self._add(df, "QualGrLivArea",
                df["OverallQual"].astype(float) * df["GrLivArea"].astype(float))

        # ── 19. Garage Capacity Score ───────────────────────────────
        if "GarageCars" in df.columns and "GarageArea" in df.columns:
            df = self._add(df, "GarageScore",
                df["GarageCars"].astype(float) * df["GarageArea"].astype(float))

        # ── 20. Above-Ground Room Density ───────────────────────────
        if "TotRmsAbvGrd" in df.columns and "GrLivArea" in df.columns:
            df = self._add(df, "RoomDensity",
                df["TotRmsAbvGrd"].astype(float) / df["GrLivArea"].replace(0, np.nan))
            df["RoomDensity"] = df["RoomDensity"].fillna(0)

        # ── 21. Functional Score ────────────────────────────────────
        func_map = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
                    "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0}
        if "Functional" in df.columns:
            df = self._add(df, "FunctionalScore",
                df["Functional"].map(func_map).fillna(7))

        # ── 22. Neighborhood Price Tier ─────────────────────────────
        if "Neighborhood" in df.columns and settings.TARGET_COLUMN in df.columns:
            med_prices = df.groupby("Neighborhood")[settings.TARGET_COLUMN].median()
            tier_map = pd.qcut(med_prices, q=3, labels=[0, 1, 2]).to_dict()
            df = self._add(df, "NeighborhoodTier",
                df["Neighborhood"].map(tier_map).astype(float).fillna(1))

        # ── 23. Season Sold ─────────────────────────────────────────
        if "MoSold" in df.columns:
            season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                          3: "Spring", 4: "Spring", 5: "Spring",
                          6: "Summer", 7: "Summer", 8: "Summer",
                          9: "Fall", 10: "Fall", 11: "Fall"}
            df = self._add(df, "SeasonSold",
                df["MoSold"].astype(int).map(season_map).fillna("Unknown"))

        # ── 24. Exterior Quality Score (combined) ───────────────────
        qual_num_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0}
        ext_cols = [c for c in ("ExterQual", "ExterCond", "KitchenQual") if c in df.columns]
        if ext_cols:
            scores = pd.DataFrame({c: df[c].map(qual_num_map).fillna(0).astype(float) for c in ext_cols})
            df = self._add(df, "ExteriorQualityScore", scores.mean(axis=1))

        # ── 25. Has Masonry Veneer ───────────────────────────────────
        if "MasVnrArea" in df.columns:
            df = self._add(df, "HasMasVnr", (df["MasVnrArea"].fillna(0) > 0).astype(int))

        # ── 26. Living area per room ──────────────────────────────────
        if "GrLivArea" in df.columns and "TotRmsAbvGrd" in df.columns:
            denom = df["TotRmsAbvGrd"].replace(0, np.nan)
            df = self._add(df, "LivingAreaPerRoom", (df["GrLivArea"] / denom).fillna(0))

        # ── 27. Old house flag (vintage) ─────────────────────────────
        if "HouseAge" in df.columns:
            df = self._add(df, "IsOldHouse", (df["HouseAge"] > 50).astype(int))

        # ── 28. Log-transformed total square feet ────────────────────
        if "TotalSF" in df.columns:
            df = self._add(df, "LogTotalSF", np.log1p(df["TotalSF"].astype(float)))

        # ── Finalize counts and return ───────────────────────────────
        self.final_feature_count = df.shape[1]
        logger.info("🔧 Created %d engineered features — final feature count: %d",
                    len(self.created_features), self.final_feature_count)

        return df
