"""
Feature engineering  creates 25+ derived variables from the 79 originals,
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
        logger.info("🔧 Created %d engineered features  final feature count: %d",
                    len(self.created_features), self.final_feature_count)

        return df
