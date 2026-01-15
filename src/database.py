"""Database Logging Module.

Uses SQLite to persist incoming prediction requests, the predicted price,
and the primary SHAP feature driver for audit and monitoring purposes.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

DB_PATH = "data/predictions.db"

def init_db(db_path: str = DB_PATH) -> None:
    """Initialize the SQLite database and create the log table if it doesn't exist.

    Args:
        db_path: Filepath to the SQLite database.
    """
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the prediction_logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            house_age REAL NOT NULL,
            rooms REAL NOT NULL,
            bedrooms REAL NOT NULL,
            population REAL NOT NULL,
            occupancy REAL NOT NULL,
            median_income REAL NOT NULL,
            predicted_price REAL NOT NULL,
            top_feature TEXT NOT NULL,
            top_feature_effect REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

def log_prediction(
    data: Dict[str, Any],
    predicted_price: float,
    top_feature: str,
    top_feature_effect: float,
    db_path: str = DB_PATH
) -> None:
    """Insert a prediction log entry into the database.

    Args:
        data: Dict containing input features.
        predicted_price: The model's prediction.
        top_feature: Name of the feature with the largest absolute SHAP value.
        top_feature_effect: The SHAP value (influence) of the top feature.
        db_path: Filepath to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO prediction_logs (
            timestamp, latitude, longitude, house_age, rooms, bedrooms,
            population, occupancy, median_income, predicted_price,
            top_feature, top_feature_effect
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        data["latitude"],
        data["longitude"],
        data["house_age"],
        data["rooms"],
        data["bedrooms"],
        data["population"],
        data["occupancy"],
        data["median_income"],
        predicted_price,
        top_feature,
        top_feature_effect
    ))
    
    conn.commit()
    conn.close()

def get_recent_predictions(limit: int = 10, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Fetch recent prediction logs from the database.

    Args:
        limit: Number of records to return.
        db_path: Filepath to the SQLite database.

    Returns:
        List of dicts representing the log records.
    """
    if not os.path.exists(db_path):
        return []
        
    conn = sqlite3.connect(db_path)
    # Set row factory to return dictionaries instead of tuples
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM prediction_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
