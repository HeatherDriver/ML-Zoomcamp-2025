import requests

url = 'http://localhost:9696/predict'

person = {
    'stress_level' : 10,
    'laptop_usage_hours' : 0,
    'daily_screen_time_hours' : 5.3,
    'physical_activity_hours_per_week' : 0,
    'gaming_hours' : 1.6,
    'mindfulness_minutes_per_day' : 6.5,
    'entertainment_hours' : 0.8,
    'age' : 16,
    'sleep_quality' : 3,
    'phone_usage_hours' : 3.2
}

response = requests.post(url, json=person)

predictions = response.json()

# print(predictions): Max score is 40 so score is reported out of 40 and the corresponding bracket is given

print(f"Predicted Weekly Depression and Anxiety Score:   {predictions['prediction']:.4f} / 40")
print(f"Prediction Bracket:                              *** {predictions['severity_level'].upper()} ***")