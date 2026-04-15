"""
Model evaluation helpers: metrics and cross-validation summaries.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X, y) -> Dict[str, Any]:
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
