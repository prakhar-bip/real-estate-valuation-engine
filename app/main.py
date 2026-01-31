"""FastAPI Web Server for Real Estate Price Prediction.

This server loads the trained ensemble model, scaling models, and SHAP explainer,
initializes the SQLite log database, logs predictions, and exposes endpoints,
including a premium web dashboard interface served at the root route.
"""

import os
import pickle
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.schemas import PropertyPredictRequest, PropertyPredictResponse, ShapExplanation
from src.features import add_geospatial_features, transform_features
from src.database import init_db, log_prediction, get_recent_predictions

app = FastAPI(
    title="California Real Estate Market Intelligence API",
    description="A FastAPI service providing ensemble-based real estate price predictions, SQLite logs, and SHAP-based interpretations.",
    version="1.2.0"
)

# Global variables to store loaded models
ensemble_model = None
robust_scaler = None
kmeans_model = None
shap_explainer = None

# Paths to serialized models
ENSEMBLE_PATH = "models/ensemble_model.joblib"
SCALER_PATH = "models/robust_scaler.joblib"
KMEANS_PATH = "models/kmeans_model.joblib"
EXPLAINER_PATH = "models/shap_explainer.pkl"

@app.on_event("startup")
def load_models_and_db():
    """Load model artifacts and initialize database on server startup."""
    global ensemble_model, robust_scaler, kmeans_model, shap_explainer
    
    # Initialize the prediction log database
    try:
        init_db()
    except Exception as e:
        print(f"Error initializing database: {e}")
    
    # Check if model files exist
    missing_files = [
        path for path in [ENSEMBLE_PATH, SCALER_PATH, KMEANS_PATH, EXPLAINER_PATH]
        if not os.path.exists(path)
    ]
    
    if missing_files:
        print(f"Warning: Serialized model files are missing: {missing_files}")
        print("Please run the training pipeline first: 'python -m src.train'")
        return

    print("Loading serialized models and preprocessing pipelines...")
    ensemble_model = joblib.load(ENSEMBLE_PATH)
    robust_scaler = joblib.load(SCALER_PATH)
    kmeans_model = joblib.load(KMEANS_PATH)
    
    with open(EXPLAINER_PATH, "rb") as f:
        shap_explainer = pickle.load(f)
        
    print("All models loaded successfully!")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the interactive web dashboard interface."""
    template_path = os.path.join("app", "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard UI template file not found.")
        
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
def health_check():
    """Simple API health check endpoint."""
    if ensemble_model is None or robust_scaler is None or kmeans_model is None or shap_explainer is None:
        return {"status": "unhealthy", "message": "Models not loaded. Run training script first."}
    return {"status": "healthy", "message": "API is ready for inference."}

@app.post("/predict", response_model=PropertyPredictResponse)
def predict_price(request: PropertyPredictRequest):
    """Predict real estate price and return SHAP-based feature explanations.

    Accepts property characteristics and coordinates, performs geospatial calculations,
    scales features, runs ensemble predictions, logs results to SQLite, and returns interpretations.
    """
    # Verify models are loaded
    if ensemble_model is None or robust_scaler is None or kmeans_model is None or shap_explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first by running python -m src.train"
        )

    # 1. Convert Pydantic request to pandas DataFrame
    input_data = {
        "median_income": [request.median_income],
        "house_age": [request.house_age],
        "rooms": [request.rooms],
        "bedrooms": [request.bedrooms],
        "population": [request.population],
        "occupancy": [request.occupancy],
        "latitude": [request.latitude],
        "longitude": [request.longitude]
    }
    input_df = pd.DataFrame(input_data)

    try:
        # 2. Apply feature engineering: distances to hubs and KMeans spatial cluster
        input_engineered = add_geospatial_features(input_df)
        
        # predict the KMeans cluster
        coords = input_engineered[["latitude", "longitude"]].values
        input_engineered["cluster"] = kmeans_model.predict(coords)

        # 3. Scale input using RobustScaler
        scaled_values = robust_scaler.transform(input_engineered)
        scaled_df = pd.DataFrame(scaled_values, columns=input_engineered.columns)

        # 4. Predict property price
        predicted_val = ensemble_model.predict(scaled_df)
        price_prediction = float(predicted_val[0])

        # 5. Calculate SHAP values
        shap_result = shap_explainer(scaled_df)
        raw_shap_values = shap_result.values[0] if len(shap_result.values.shape) > 1 else shap_result.values

        # 6. Build structured SHAP explanations list
        explanations = []
        for col_name, shap_val in zip(scaled_df.columns, raw_shap_values):
            effect_usd = float(shap_val)
            readable_name = col_name.replace("_", " ").title()
            
            if effect_usd > 0:
                desc = f"Increases estimated price by ${effect_usd:,.2f} due to {readable_name}."
            else:
                desc = f"Decreases estimated price by ${abs(effect_usd):,.2f} due to {readable_name}."
                
            explanations.append(ShapExplanation(
                feature=col_name,
                shap_value=effect_usd,
                effect_usd=effect_usd,
                description=desc
            ))

        # Sort explanations by absolute impact
        explanations.sort(key=lambda x: abs(x.effect_usd), reverse=True)

        # Log prediction query and results to SQLite for audit/monitoring
        top_feature = explanations[0].feature
        top_feature_effect = explanations[0].effect_usd
        
        try:
            log_prediction(
                data={
                    "latitude": request.latitude,
                    "longitude": request.longitude,
                    "house_age": request.house_age,
                    "rooms": request.rooms,
                    "bedrooms": request.bedrooms,
                    "population": request.population,
                    "occupancy": request.occupancy,
                    "median_income": request.median_income
                },
                predicted_price=price_prediction,
                top_feature=top_feature,
                top_feature_effect=top_feature_effect
            )
        except Exception as db_err:
            print(f"Failed to log prediction to database: {db_err}")

        return PropertyPredictResponse(
            predicted_price=price_prediction,
            shap_explanations=explanations
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

@app.get("/predictions")
def get_prediction_history(limit: int = 10):
    """Retrieve recent prediction history logged in the SQLite database."""
    try:
        history = get_recent_predictions(limit=limit)
        return {"count": len(history), "history": history}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch prediction history: {str(e)}"
        )
