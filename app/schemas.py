"""FastAPI Request and Response Pydantic Schemas.

Defines schemas for the real estate price prediction request
and the corresponding explanation-augmented response.
"""

from pydantic import BaseModel, Field
from typing import List

class PropertyPredictRequest(BaseModel):
    latitude: float = Field(
        ..., 
        description="Latitude of the property in California (e.g., 37.88)",
        ge=32.5,
        le=42.5
    )
    longitude: float = Field(
        ..., 
        description="Longitude of the property in California (e.g., -122.23)",
        ge=-124.5,
        le=-114.0
    )
    house_age: float = Field(
        ..., 
        description="Median age of houses in the block group (e.g., 41.0)",
        ge=0.0
    )
    rooms: float = Field(
        ..., 
        description="Average number of rooms per household (e.g., 6.98)",
        ge=0.0
    )
    bedrooms: float = Field(
        ..., 
        description="Average number of bedrooms per household (e.g., 1.02)",
        ge=0.0
    )
    population: float = Field(
        ..., 
        description="Total population of the block group (e.g., 322.0)",
        ge=0.0
    )
    occupancy: float = Field(
        ..., 
        description="Average number of household members (e.g., 2.55)",
        ge=0.0
    )
    median_income: float = Field(
        ..., 
        description="Median neighborhood household income in tens of thousands (e.g., 8.3 = $83,000)",
        ge=0.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "latitude": 37.88,
                "longitude": -122.23,
                "house_age": 41.0,
                "rooms": 6.98,
                "bedrooms": 1.02,
                "population": 322.0,
                "occupancy": 2.55,
                "median_income": 8.32
            }
        }
    }

class ShapExplanation(BaseModel):
    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="Raw SHAP value from model")
    effect_usd: float = Field(..., description="Financial effect of feature in USD (added/subtracted)")
    description: str = Field(..., description="User-friendly text explanation of the impact")

class PropertyPredictResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted median property price in USD")
    shap_explanations: List[ShapExplanation] = Field(
        ..., 
        description="List of feature contributions explaining the prediction"
    )
