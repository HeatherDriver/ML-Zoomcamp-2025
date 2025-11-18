import pickle
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
import urllib.request
from fastapi import FastAPI
import uvicorn

# Create class for each person to enforce data standardisation rules prior to data being fed to model service
class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")
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

app = FastAPI(title="person-weekly-depression-anxiety-score-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# apply the model to single person
def predict_single(person):
    result = pipeline.predict(person)
    return result


@app.post("/predict")