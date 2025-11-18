# Description of the problem
Considering that mental health conditions often develop gradually, early intervention might be a good strategy for treating these conditions. 
This dataset from Kaggle lists different behaviours and quantifies the time spent doing various daily activities.
We could analyse this weekly behavioral and mood data to forecast the combined depression and anxiety scores.
Building a predictive model from the data would enable proactive mental healthcare in the following way:
1. Users could complete a brief weekly survey through a mobile app, reporting sleep patterns, mood ratings, activity levels, and social interactions
2. Data would be sent to a containerized ML service via secure API
3. The model built here would return personalized depression and anxiety risk score, and give this meaning based on its bracket relative to the population.
4. Healthcare providers would receive automated alerts for concerning scores, enabling timely intervention.
5. Users would receive personalized wellness recommendations based on their risk profile

This system could transform reactive mental healthcare into preventive care. The containerized deployment would ensure scalable and reliable service delivery across healthcare networks while maintaining data privacy and security standards required for such medical applications.

# Instructions on how to run the project
## Notebook.ipynb
This notebook contains the process followed for:
1. Data preparation and data cleaning
2. Exploratory data analysis and feature importance analysis
3. Model selection process and parameter tuning

## train.py
This script:
1. Trains the final model
2. Saves it to a pickle file

## predict.py
This script:
1. Loads the trained model and severity mapping dictionary from pickle files
2. Defines data validation rules using Pydantic to ensure incoming data meets requirements (e.g., stress_level between 1-10, non-negative hours)
3. Creates a FastAPI web service that accepts POST requests with person data
4. Makes predictions by processing input through the ML pipeline
5. Maps numeric predictions to severity levels (e.g., score 6.7 → "moderate")
6. Returns both the raw prediction score and human-readable severity level as JSON
7. Serves the API on port 9696, making it accessible for other applications to consume

## Turn predict.py into a web service
As in the [workshop](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment/workshop), FastAPI and uvicorn are used to turn predict.py into a web service. Use the following for uvicorn:

```bash
# Install dependencies
uv sync

# Run the service locally
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload

# Test the service (in a separate terminal)
uv run python test.py

# Close uvicorn down in its terminal with Ctrl + C
```

And this for Docker:

```bash
# Build the Docker image in the local folder
docker build -t depression-anxiety-scoring-model .

# Run the containerized service from the image
docker run -it -p 9696:9696 depression-anxiety-scoring-model

# Test the service (in a separate terminal)
python test.py

# Close Docker down in its terminal with Ctrl + C
```
