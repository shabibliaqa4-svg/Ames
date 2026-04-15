"""
Simple model training utilities.
Provides light-weight training routines for quick development.
"""

from typing import Dict, Any
from config.settings import Settings
from src.utils import get_logger, timer, save_model

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

settings = Settings()
logger = get_logger(__name__)


@timer
def train_basic_models(X, y) -> Dict[str, Any]:
    """Train a couple of baseline models and persist them.

    Returns a dict of trained models.
    """
    models = {}

    # Ridge baseline
    # Ridge does not accept `random_state` in many sklearn versions; keep default.
    ridge = Ridge()
    ridge.fit(X, y)
    save_model(ridge, "ridge.joblib")
    models["ridge"] = ridge

    # Random Forest baseline
    rf = RandomForestRegressor(n_estimators=100, random_state=settings.RANDOM_STATE)
    rf.fit(X, y)
    save_model(rf, "random_forest.joblib")
    models["random_forest"] = rf

    logger.info("Trained %d models", len(models))
    return models
