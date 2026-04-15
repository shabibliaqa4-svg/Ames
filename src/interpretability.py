"""
Model interpretability helpers (SHAP wrappers).
"""

from typing import Any
import numpy as np

try:
    import shap
except Exception:
    shap = None


def explain_model_shap(model: Any, X) -> Any:
    """Return SHAP values/explainer output if SHAP is available."""
    if shap is None:
        raise RuntimeError("shap is not installed")
    explainer = shap.Explainer(model, X)
    return explainer(X)
