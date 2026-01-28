# California Real Estate Price Predictor & Interpreter

This project is a real estate intelligence and price prediction engine focused on the state of **California**. It combines supervised machine learning (an ensemble of XGBoost and LightGBM) with geospatial feature engineering, SQLite database logging, and SHAP (SHapley Additive exPlanations) values to explain exactly what factors drive a property's estimated market value.

The project is fully productionized with a **FastAPI** web API, **SQLite** database logging, and containerized using **Docker** and **Docker Compose**.

---

## 🎯 Key Features

- **Geospatial Feature Engineering**: Calculates Haversine distances to major California economic hubs (Los Angeles, San Francisco, San Diego, San Jose) and performs spatial clustering using KMeans.
- **Ensemble Regression**: Trains a `VotingRegressor` combining XGBoost and LightGBM regressor models.
- **Model Interpretation**: Integrates a SHAP `TreeExplainer` to extract the individual feature contributions for each prediction in USD.
- **SQLite Prediction Logger**: Automatically stores request payloads, predicted valuations, and the top SHAP-driver features into a local database for monitoring and audit.
- **Production Serving**: Implements a FastAPI endpoint `/predict` which processes property attributes, computes geospatial metrics dynamically, scales them, logs them, and returns predictions with explanations.
- **History Endpoint**: Exposes a `/predictions` endpoint to inspect recently logged inference events.
- **CI/CD Integration**: Pre-configured GitHub Actions workflow for automated pipeline training and API integration tests.
- **Containerization**: Packaged with Docker and Docker Compose for easy deployment.

---

## 📊 Dataset & Mapping

The project utilizes the **California Housing Dataset** from `scikit-learn` (originally compiled from the 1990 US Census). We map census block-group level attributes to clean property-like parameters:

| Input Variable | Description |
|---|---|
| `latitude` | Property latitude (must be within California bounds: `[32.5, 42.5]`) |
| `longitude` | Property longitude (must be within California bounds: `[-124.5, -114.0]`) |
| `house_age` | Median age of houses in the block group |
| `rooms` | Average number of rooms per household |
| `bedrooms` | Average number of bedrooms per household |
| `population` | Total neighborhood population |
| `occupancy` | Average number of household members |
| `median_income` | Median household income in tens of thousands of dollars (e.g. `8.3` = $83,000) |

---

## 📁 Project Structure

```
real-estate-market-intelligence-engine/
├── .github/
│   └── workflows/
│       └── test.yml            # CI/CD GitHub Actions workflow
├── app/                        # FastAPI Application
│   ├── __init__.py
│   ├── main.py                 # FastAPI router, model loader, prediction logic
│   └── schemas.py              # Pydantic input/output schemas
├── src/                        # Core ML Package
│   ├── __init__.py
│   ├── data_prep.py            # Ingestion, cleaning, and train/test splitting
│   ├── database.py             # SQLite predictions logging and audit module
│   ├── features.py             # Haversine distance, KMeans, and RobustScaling
│   ├── model_utils.py          # Ensemble training, metrics, and SHAP calculations
│   └── train.py                # Main training pipeline orchestrator
├── models/                     # Serialized artifacts (created after training)
│   ├── ensemble_model.joblib   # Trained ensemble model (XGBoost + LightGBM)
│   ├── robust_scaler.joblib    # Fitted RobustScaler
│   ├── kmeans_model.joblib     # Fitted KMeans coordinate clusterer
│   └── shap_explainer.pkl      # Serialized SHAP TreeExplainer
├── data/                       # Local data cache (gitignored)
│   ├── predictions.db          # SQLite prediction log file
│   ├── raw_data.csv
│   ├── train_data.csv
│   └── test_data.csv
├── tests/                      # Testing directory
│   └── test_api.py             # FastAPI endpoint, validation, and database tests
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Setup & Installation

### Option 1: Running Locally

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone https://github.com/shaleenswarup/real-estate-market-intelligence-engine.git
   cd real-estate-market-intelligence-engine
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training pipeline** to download the dataset, fit the feature pipeline, and train the model:
   ```bash
   python -m src.train
   ```

5. **Start the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

6. The API docs will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Option 2: Running with Docker (Recommended)

1. Make sure **Docker** and **Docker Compose** are installed and running on your system.
2. Build and spin up the containerized service:
   ```bash
   docker-compose up --build
   ```
3. The server will launch and bind to port `8000` on your host machine.
4. Verify the API health check:
   ```bash
   curl http://localhost:8000/health
   ```

---

## 📡 API Documentation

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "message": "API is ready for inference."
  }
  ```

### 2. Predict Price
- **Endpoint**: `POST /predict`
- **Request Body**:
  ```json
  {
    "latitude": 37.88,
    "longitude": -122.23,
    "house_age": 41.0,
    "rooms": 6.98,
    "bedrooms": 1.02,
    "population": 322.0,
    "occupancy": 2.55,
    "median_income": 8.32
  }
  ```
- **Response Body**:
  ```json
  {
    "predicted_price": 417205.67,
    "shap_explanations": [
      {
        "feature": "median_income",
        "shap_value": 187465.38,
        "effect_usd": 187465.38,
        "description": "Increases estimated price by $187,465.38 due to Median Income."
      },
      {
        "feature": "dist_los_angeles",
        "shap_value": -32658.20,
        "effect_usd": -32658.20,
        "description": "Decreases estimated price by $32,658.20 due to Dist Los Angeles."
      }
    ]
  }
  ```

### 3. Prediction Logs (Audit/Monitoring)
- **Endpoint**: `GET /predictions`
- **Query Params**: `limit=10` (default: 10)
- **Response Body**:
  ```json
  {
    "count": 1,
    "history": [
      {
        "id": 1,
        "timestamp": "2026-06-07T08:45:00.000000",
        "latitude": 37.88,
        "longitude": -122.23,
        "house_age": 41.0,
        "rooms": 6.98,
        "bedrooms": 1.02,
        "population": 322.0,
        "occupancy": 2.55,
        "median_income": 8.32,
        "predicted_price": 417205.67,
        "top_feature": "median_income",
        "top_feature_effect": 187465.38
      }
    ]
  }
  ```

---

## 🧪 Running Tests

A test suite is included to check API health, validate schema validation constraints, verify the model prediction output format, and test SQLite database logging. Run it with:
```bash
python tests/test_api.py
```
