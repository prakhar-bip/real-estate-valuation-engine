"""Main Model Training Pipeline.

This script runs the end-to-end training pipeline:
1. Downloads and splits the California Housing dataset.
2. Engineers geospatial features (KMeans clusters and distances to cities).
3. Fits the RobustScaler and scales the data.
4. Trains the XGBoost + LightGBM VotingRegressor ensemble.
5. Serializes the trained models and the SHAP TreeExplainer.
"""

import pandas as pd
from src.data_prep import load_and_prepare_data
from src.features import fit_and_save_geospatial_models, transform_features
from src.model_utils import train_and_evaluate_ensemble

def run_pipeline() -> None:
    """Run the end-to-end data preparation, feature engineering, and training pipeline."""
    print("=== Step 1: Data Ingestion & Splitting ===")
    raw_path = "data/raw_data.csv"
    train_path = "data/train_data.csv"
    test_path = "data/test_data.csv"
    
    # Download and split raw dataset
    load_and_prepare_data(
        raw_path=raw_path,
        train_path=train_path,
        test_path=test_path
    )

    print("\n=== Step 2: Feature Engineering & Scaling ===")
    # Load splits
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Fit KMeans clusterer and RobustScaler on training set
    kmeans_path = "models/kmeans_model.joblib"
    scaler_path = "models/robust_scaler.joblib"
    kmeans, scaler = fit_and_save_geospatial_models(
        train_df=train_df,
        kmeans_path=kmeans_path,
        scaler_path=scaler_path
    )

    # Transform training and test datasets
    print("Transforming and scaling features...")
    train_scaled = transform_features(train_df, kmeans, scaler)
    test_scaled = transform_features(test_df, kmeans, scaler)

    # Save scaled datasets
    train_scaled_path = "data/train_scaled.csv"
    test_scaled_path = "data/test_scaled.csv"
    train_scaled.to_csv(train_scaled_path, index=False)
    test_scaled.to_csv(test_scaled_path, index=False)
    print(f"Saved scaled datasets to {train_scaled_path} and {test_scaled_path}")

    print("\n=== Step 3: Model Training & Interpretation ===")
    model_path = "models/ensemble_model.joblib"
    explainer_path = "models/shap_explainer.pkl"
    
    # Train ensemble and fit SHAP explainer
    ensemble, metrics = train_and_evaluate_ensemble(
        train_scaled_path=train_scaled_path,
        test_scaled_path=test_scaled_path,
        model_path=model_path,
        explainer_path=explainer_path
    )
    
    print("\n=== End-to-End Pipeline Complete ===")
    print(f"Final Train R2: {metrics['train_r2']:.4f}")
    print(f"Final Test R2:  {metrics['test_r2']:.4f}")
    print(f"Final Train MAE: ${metrics['train_mae']:,.2f}")
    print(f"Final Test MAE:  ${metrics['test_mae']:,.2f}")

if __name__ == "__main__":
    run_pipeline()
