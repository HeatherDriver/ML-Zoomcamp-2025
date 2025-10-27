import pickle
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
import urllib.request
from fastapi import FastAPI
import uvicorn



class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_source: Literal["paid_ads", "organic_search"]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    convert_probability: float
    convert: bool


app = FastAPI(title="customer-convert-prediction")

url = 'https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin'
urllib.request.urlretrieve(url, 'model_trained.bin')
print('Model downloaded successfully!')

with open('model_trained.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        convert_probability=prob,
        convert=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)



