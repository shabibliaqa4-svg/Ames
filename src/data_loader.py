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
        # Note: avoid passing `parser` to maintain compatibility across
        # scikit-learn versions.
        data = fetch_openml(data_id=settings.OPENML_DATASET_ID, as_frame=True)
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
        self.missing_columns = missing
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
        valid = self.validate_schema(df)
        if not valid:
            raise ValueError(f"Schema validation failed; missing columns: {getattr(self, 'missing_columns', [])}")
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
