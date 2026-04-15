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
