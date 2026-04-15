"""
Orchestrator script to run the full pipeline: load data, EDA, features, train, evaluate.
"""

from src.data_loader import AmesDataLoader
from src.eda import ExploratoryAnalysis
from src.feature_engineering import FeatureEngineer
from src.preprocessing import build_preprocessor
from src.model_trainer import train_basic_models
from src.model_evaluator import evaluate_model
from src.utils import get_logger, save_metrics

logger = get_logger(__name__)


def run_pipeline():
    loader = AmesDataLoader()
    df = loader.load()

    # EDA
    eda = ExploratoryAnalysis(df)
    eda_stats = eda.run_full_eda()

    # Feature engineering
    fe = FeatureEngineer()
    df_fe = fe.transform(df)

    # Preprocessing (build transformer)
    preproc = build_preprocessor(df_fe)

    # Training (apply preprocessor so all features are numeric)
    target = df_fe["SalePrice"]
    X = df_fe.drop(columns=["SalePrice"]) if "SalePrice" in df_fe.columns else df_fe

    # Fit and transform the feature matrix using the pipeline
    X_processed = preproc.fit_transform(X)

    models = train_basic_models(X_processed, target)

    # Evaluate first model on the processed features
    first_model = list(models.values())[0]
    metrics = evaluate_model(first_model, X_processed, target)
    save_metrics(metrics, "metrics.json")
    logger.info("Pipeline complete")


if __name__ == "__main__":
    run_pipeline()
