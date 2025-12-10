"""Data Ingestion and Preparation Module.

This script downloads the California Housing dataset from scikit-learn,
filters the data by geographic bounds, scales the target price to USD,
and splits the data into train and test sets.
"""

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_and_prepare_data(
    raw_path: str = "data/raw_data.csv",
    train_path: str = "data/train_data.csv",
    test_path: str = "data/test_data.csv"
) -> None:
    """Download California Housing dataset, filter geographic bounds, and split.

    Args:
        raw_path: Path to save the filtered raw dataset.
        train_path: Path to save the training split.
        test_path: Path to save the test split.
    """
    print("Fetching California Housing dataset...")
    # Fetch dataset from sklearn
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # Rename columns to cleaner, property-centric names
    df = df.rename(columns={
        "MedInc": "median_income",
        "HouseAge": "house_age",
        "AveRooms": "rooms",
        "AveBedrms": "bedrooms",
        "Population": "population",
        "AveOccup": "occupancy",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "MedHouseVal": "price"
    })

    # Convert median house value to actual USD (original is in units of $100k)
    df["price"] = df["price"] * 100000.0

    # Define California coordinate bounds
    min_lat, max_lat = 32.5, 42.5
    min_lon, max_lon = -124.5, -114.0

    print(f"Original dataset size: {len(df)}")
    
    # Filter rows that fall within geographic bounds
    df = df[
        (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat) &
        (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
    ]
    
    print(f"Filtered dataset size (within California bounds): {len(df)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    # Save complete cleaned/filtered raw dataset
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")

    # Split into train and test sets (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save train and test sets
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train data to {train_path} and test data to {test_path}")

if __name__ == "__main__":
    load_and_prepare_data()
