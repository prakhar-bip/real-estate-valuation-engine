"""Model Training and Interpretation Module.

This module provides functions to train a VotingRegressor ensemble
(comprising XGBoost and LightGBM models), evaluate it with R2 metrics,
and fit a SHAP TreeExplainer for feature explainability.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import shap
from typing import Dict, Any, Tuple

def train_and_evaluate_ensemble(
    train_scaled_path: str = "data/train_scaled.csv",
    test_scaled_path: str = "data/test_scaled.csv",
    model_path: str = "models/ensemble_model.joblib",
    explainer_path: str = "models/shap_explainer.pkl"
) -> Tuple[VotingRegressor, Dict[str, float]]:
    """Train the ensemble model and fit a SHAP TreeExplainer.

    Args:
        train_scaled_path: Path to scaled training data.
        test_scaled_path: Path to scaled test data.
        model_path: Path to save the trained ensemble model.
        explainer_path: Path to save the serialized SHAP explainer.

    Returns:
        A tuple of (trained_ensemble, dictionary_of_metrics).
    """
    # Load scaled data
    print("Loading scaled datasets...")
    train_df = pd.read_csv(train_scaled_path)
    test_df = pd.read_csv(test_scaled_path)

    # Separate features and target
    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    # Define base tree-based models
    print("Initializing base estimators (XGBoost and LightGBM)...")
    xgb = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    lgb = LGBMRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )

    # Create the Voting Ensemble
    print("Training VotingRegressor Ensemble...")
    ensemble = VotingRegressor(
        estimators=[("xgb", xgb), ("lgb", lgb)],
        weights=[0.5, 0.5]
    )
    
    ensemble.fit(X_train, y_train)

    # Make predictions and evaluate
    train_preds = ensemble.predict(X_train)
    test_preds = ensemble.predict(X_test)

    metrics = {
        "train_r2": float(r2_score(y_train, train_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
        "train_mae": float(mean_absolute_error(y_train, train_preds)),
        "test_mae": float(mean_absolute_error(y_test, test_preds))
    }

    print(f"Model Metrics: {metrics}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(explainer_path), exist_ok=True)

    # Save the trained ensemble model
    joblib.dump(ensemble, model_path)
    print(f"Ensemble model saved to {model_path}")

    # For fast, real-time SHAP values, we fit a TreeExplainer on the XGBoost component.
    # The XGBoost model represents a major part of the ensemble and shares the same features.
    print("Fitting SHAP TreeExplainer on XGBoost component...")
    trained_xgb = ensemble.named_estimators_["xgb"]
    explainer = shap.TreeExplainer(trained_xgb)
    
    # Save the explainer with pickle for robust serialization
    with open(explainer_path, "wb") as f:
        pickle.dump(explainer, f)
    print(f"SHAP explainer saved to {explainer_path}")

    return ensemble, metrics

def explain_prediction(
    explainer_path: str,
    feature_row: pd.DataFrame
) -> Dict[str, float]:
    """Calculate SHAP values for a single prediction row.

    Args:
        explainer_path: Path to the serialized explainer.
        feature_row: Single-row DataFrame containing scaled features.

    Returns:
        A dictionary mapping feature names to their SHAP values (price contributions).
    """
    with open(explainer_path, "rb") as f:
        explainer = pickle.load(f)
        
    shap_values = explainer(feature_row)
    
    # Extract the SHAP values (contributions) for the single row
    # shap_values.values shape is (1, num_features) or (num_features,)
    values = shap_values.values[0] if len(shap_values.values.shape) > 1 else shap_values.values
    
    # Combine feature names with SHAP values
    explanations = {}
    for feature_name, value in zip(feature_row.columns, values):
        explanations[feature_name] = float(value)
        
    return explanations
