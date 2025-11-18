import pickle
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
import urllib.request
from fastapi import FastAPI
import uvicorn

# Create class for each person to enforce data standardisation rules prior to data being fed to model service
class Person(BaseModel):
    model_config = ConfigDict(extra="forbid") # To make Pydantic raise an error, we add model_config
    stress_level: int = Field(..., ge=1, le=10)
    laptop_usage_hours: float = Field(..., ge=0.0)
    daily_screen_time_hours: float = Field(...)
    physical_activity_hours_per_week: float = Field(..., ge=0.0)
    gaming_hours: float = Field(..., ge=0.0)
    mindfulness_minutes_per_day: float = Field(..., ge=0.0)
    entertainment_hours: float = Field(..., ge=0.0)
    age: int = Field(..., ge=0) 
    sleep_quality: float = Field(..., ge=1.0, le=5.0)
    phone_usage_hours: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    prediction: float
    severity_level: str

app = FastAPI(title="person-weekly-depression-anxiety-score-prediction")

# Load the model
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Load the severity mapping dictionary
with open('mapping_dict.pkl', 'rb') as f_map:  
    mapping_dict = pickle.load(f_map)

# Helper function to get severity level
def get_severity_level(score: float) -> str:
    """Convert numeric score to severity level using mapping dictionary"""
    # Round to nearest integer and convert to float format that matches dict keys
    rounded_score = float(round(score))
    # Look up in mapping dictionary
    return mapping_dict.get(rounded_score, "unknown")

# apply the model to single person
def predict_single(person):
    result = pipeline.predict([person])
    return result


@app.post("/predict")
def predict(person: Person) -> PredictResponse:
    regression = predict_single(person.model_dump())
    prediction_score = float(regression[0])
    severity = get_severity_level(prediction_score)
    
    return PredictResponse(
        prediction=prediction_score,
        severity_level=severity
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)