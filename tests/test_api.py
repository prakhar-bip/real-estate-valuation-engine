"""API Integration Tests.

This file uses FastAPI's TestClient to verify the endpoints of our API.
It verifies the health check, the prediction/interpretability endpoint,
and the SQLite history logs retrieval.
"""

import os
import sys
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app, load_models_and_db

# Force load models for the test environment
load_models_and_db()

client = TestClient(app)

def test_health_endpoint():
    """Verify that the health check returns a successful status."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "ready" in json_data["message"]

def test_predict_endpoint_success():
    """Verify that the prediction endpoint works with valid inputs."""
    payload = {
        "latitude": 37.88,
        "longitude": -122.23,
        "house_age": 41.0,
        "rooms": 6.98,
        "bedrooms": 1.02,
        "population": 322.0,
        "occupancy": 2.55,
        "median_income": 8.32
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    json_data = response.json()
    assert "predicted_price" in json_data
    assert isinstance(json_data["predicted_price"], float)
    assert json_data["predicted_price"] > 0.0
    
    assert "shap_explanations" in json_data
    shap_exps = json_data["shap_explanations"]
    assert isinstance(shap_exps, list)
    assert len(shap_exps) > 0
    
    # Check structure of the first SHAP explanation item
    first_exp = shap_exps[0]
    assert "feature" in first_exp
    assert "shap_value" in first_exp
    assert "effect_usd" in first_exp
    assert "description" in first_exp
    
    print(f"\nTest Predict Output predicted price: ${json_data['predicted_price']:,.2f}")
    print("Top contributing factor:")
    print(f"  - Feature: {first_exp['feature']}")
    print(f"  - Effect USD: ${first_exp['effect_usd']:,.2f}")
    print(f"  - Description: {first_exp['description']}")

def test_predict_endpoint_validation_error():
    """Verify that the API rejects invalid coordinate boundaries (out of California)."""
    # Coordinates for Seattle, Washington (outside California bounds)
    payload = {
        "latitude": 47.6062, 
        "longitude": -122.3321,
        "house_age": 10.0,
        "rooms": 5.0,
        "bedrooms": 1.0,
        "population": 1000.0,
        "occupancy": 3.0,
        "median_income": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predictions_history_endpoint():
    """Verify that we can retrieve recent prediction history from SQLite."""
    # First make a prediction to ensure there is at least one log
    payload = {
        "latitude": 37.88,
        "longitude": -122.23,
        "house_age": 41.0,
        "rooms": 6.98,
        "bedrooms": 1.02,
        "population": 322.0,
        "occupancy": 2.55,
        "median_income": 8.32
    }
    client.post("/predict", json=payload)
    
    # Query history
    response = client.get("/predictions?limit=5")
    assert response.status_code == 200
    json_data = response.json()
    assert "count" in json_data
    assert "history" in json_data
    assert isinstance(json_data["history"], list)
    assert json_data["count"] >= 1
    
    # Check that keys match our DB fields
    first_record = json_data["history"][0]
    assert "id" in first_record
    assert "timestamp" in first_record
    assert "predicted_price" in first_record
    assert "top_feature" in first_record
    assert "top_feature_effect" in first_record
    assert first_record["latitude"] == 37.88
    
    print("\nTest Prediction History:")
    print(f"  - Total records found: {json_data['count']}")
    print(f"  - Latest predicted price in database: ${first_record['predicted_price']:,.2f}")
    print(f"  - Top driver recorded in database: {first_record['top_feature']} (${first_record['top_feature_effect']:,.2f})")

if __name__ == "__main__":
    print("Running API Integration Tests...")
    test_health_endpoint()
    print("Health check endpoint: PASSED")
    test_predict_endpoint_success()
    print("Prediction endpoint success: PASSED")
    test_predict_endpoint_validation_error()
    print("Prediction endpoint validation: PASSED")
    test_predictions_history_endpoint()
    print("Predictions history database retrieval: PASSED")
    print("All integration tests passed successfully!")
