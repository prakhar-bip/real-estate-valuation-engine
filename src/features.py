"""Feature Engineering and Scaling Module.

This module contains functions to calculate geospatial distances (Haversine formula),
assign spatial cluster labels using KMeans, and scale numerical features.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict

# Major California economic hub coordinates
HUBS = {
    "los_angeles": (34.0522, -118.2437),
    "san_francisco": (37.7749, -122.4194),
    "san_diego": (32.7157, -117.1611),
    "san_jose": (37.3382, -121.8863)
}

def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the great-circle distance between two points in kilometers.

    Args:
        lat1: Latitude of first point.
        lon1: Longitude of first point.
        lat2: Latitude of second point.
        lon2: Longitude of second point.

    Returns:
        Distance in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Earth's radius in kilometers
    
    return float(c * r)

def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distances to major California cities and nearest economic hub.

    Args:
        df: Input DataFrame containing latitude and longitude.

    Returns:
        DataFrame with added distance features.
    """
    df = df.copy()
    
    # Compute distance to each hub
    for name, coords in HUBS.items():
        hub_lat, hub_lon = coords
        df[f"dist_{name}"] = df.apply(
            lambda row: haversine_distance(row["latitude"], row["longitude"], hub_lat, hub_lon),
            axis=1
        )
    
    # Distance to the nearest hub
    dist_cols = [f"dist_{name}" for name in HUBS.keys()]
    df["dist_nearest_hub"] = df[dist_cols].min(axis=1)
    
    return df

def fit_and_save_geospatial_models(
    train_df: pd.DataFrame,
    kmeans_path: str = "models/kmeans_model.joblib",
    scaler_path: str = "models/robust_scaler.joblib"
) -> Tuple[KMeans, RobustScaler]:
    """Train and serialize the KMeans clustering model and RobustScaler.

    Args:
        train_df: Training DataFrame containing engineered features.
        kmeans_path: Path to save the trained KMeans model.
        scaler_path: Path to save the RobustScaler model.

    Returns:
        A tuple of (trained_kmeans, fitted_scaler).
    """
    import os
    os.makedirs(os.path.dirname(kmeans_path), exist_ok=True)
    
    # 1. Fit KMeans clusterer on Latitude and Longitude to group neighborhoods
    print("Fitting KMeans spatial clusterer...")
    coords = train_df[["latitude", "longitude"]].values
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(coords)
    joblib.dump(kmeans, kmeans_path)
    print(f"KMeans model saved to {kmeans_path}")
    
    # Add geospatial distance features first
    train_df_engineered = add_geospatial_features(train_df)
    
    # Add cluster feature to dataframe to fit the scaler
    train_df_engineered["cluster"] = kmeans.labels_
    
    # List features to scale (excluding the target column 'price' if present)
    feature_cols = [col for col in train_df_engineered.columns if col != "price"]
    
    # 2. Fit RobustScaler
    print("Fitting RobustScaler...")
    scaler = RobustScaler()
    scaler.fit(train_df_engineered[feature_cols])
    joblib.dump(scaler, scaler_path)
    print(f"RobustScaler saved to {scaler_path}")
    
    return kmeans, scaler

def transform_features(
    df: pd.DataFrame,
    kmeans: KMeans,
    scaler: RobustScaler
) -> pd.DataFrame:
    """Apply distance calculations, KMeans clustering, and robust scaling.

    Args:
        df: Input DataFrame (must contain raw features).
        kmeans: Fitted KMeans model.
        scaler: Fitted RobustScaler.

    Returns:
        DataFrame containing scaled features.
    """
    # 1. Add geospatial distances
    df_engineered = add_geospatial_features(df)
    
    # 2. Add spatial clusters
    coords = df_engineered[["latitude", "longitude"]].values
    df_engineered["cluster"] = kmeans.predict(coords)
    
    # 3. Separate target price if it is in the dataframe
    has_price = "price" in df_engineered.columns
    if has_price:
        prices = df_engineered["price"].copy()
        features = df_engineered.drop(columns=["price"])
    else:
        features = df_engineered
        
    # Scale features
    scaled_values = scaler.transform(features)
    df_scaled = pd.DataFrame(scaled_values, columns=features.columns, index=features.index)
    
    if has_price:
        df_scaled["price"] = prices
        
    return df_scaled
