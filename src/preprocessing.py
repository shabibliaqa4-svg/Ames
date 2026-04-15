"""
Preprocessing pipeline for the Ames Housing dataset.
Provides a ColumnTransformer that handles numeric, ordinal and nominal features.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

from config.settings import Settings

settings = Settings()


def _replace_literal_none(X):
    """Replace literal 'None' strings (and Python None / pd.NA) with np.nan.

    This helps encoders and imputers treat textual 'None' values as missing.
    Returns a numpy array compatible with scikit-learn pipelines.
    """
    try:
        df = pd.DataFrame(X)
        # Build replacement dict based on pandas version
        repl_dict = {"None": np.nan, None: np.nan}
        if hasattr(pd, "NA"):
            repl_dict[pd.NA] = np.nan
        return df.replace(repl_dict).values
    except Exception:
        # Fallback: convert to array and manually replace
        arr = np.array(X, dtype=object)
        try:
            mask = arr == "None"
            arr[mask] = np.nan
        except Exception:
            pass
        return arr


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
        # Normalize literal 'None' then impute missing ordinals to 'NA'
        ("fix_none", FunctionTransformer(_replace_literal_none, validate=False)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
        ("ord", OrdinalEncoder(categories=ordinal_cats, dtype=float)),
    ])

    # Create OneHotEncoder in a compatibility-friendly way to support
    # both older and newer scikit-learn versions (sparse vs sparse_output).
    def _make_onehot():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    cat_pipeline = Pipeline([
        # Normalize literal 'None' to np.nan so imputers behave consistently
        ("fix_none", FunctionTransformer(_replace_literal_none, validate=False)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", _make_onehot()),
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
