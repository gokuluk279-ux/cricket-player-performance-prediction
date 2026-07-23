"""Baseline model-training helpers for the cricket analytics project."""

from __future__ import annotations

from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Create a reproducible preprocessing and Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_regression(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Return standard regression metrics."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
