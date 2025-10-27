import requests

url = 'http://localhost:9696/predict'


# Question 3
customer = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

response = requests.post(url, json=customer)

predictions = response.json()
print(f"Question 3 prediction:\n{predictions['convert']}: {predictions['convert_probability']:.4f}")

# Question 4
customer = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=customer).json()

print(f"Question 4 prediction:\n{response['convert']}: {response['convert_probability']:.4f}")