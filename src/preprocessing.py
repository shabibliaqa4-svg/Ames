"""
Preprocessing pipeline for the Ames Housing dataset.
Provides a ColumnTransformer that handles numeric, ordinal and nominal features.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from config.settings import Settings

settings = Settings()


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer based on columns present in `df` and project settings.

    Returns a fitted (not-fitted) ColumnTransformer instance.
    """
    # Numeric features: continuous + discrete
    numeric_features = [c for c in (settings.CONTINUOUS_FEATURES + settings.DISCRETE_FEATURES) if c in df.columns]

    # Ordinal features and their categories
    ordinal_cols = [c for c in settings.ORDINAL_FEATURES.keys() if c in df.columns]
    ordinal_cats = [settings.ORDINAL_FEATURES[c] for c in ordinal_cols]

    # Nominal (categorical) features
    nominal_features = [c for c in settings.NOMINAL_FEATURES if c in df.columns]

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ord", OrdinalEncoder(categories=ordinal_cats, dtype=float)),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", num_pipeline, numeric_features))
    if ordinal_cols:
        transformers.append(("ord", ord_pipeline, ordinal_cols))
    if nominal_features:
        transformers.append(("cat", cat_pipeline, nominal_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
