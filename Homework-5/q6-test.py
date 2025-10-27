import requests

# Question 6
url = "http://localhost:9696/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
result = requests.post(url, json=client).json()
print(result)