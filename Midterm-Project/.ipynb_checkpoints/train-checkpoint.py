#!/usr/bin/env python
# coding: utf-8

# This script (1) loads the data, (2) trains the final model and (3) saves it to a pickle file. It was built from _train.py

import os
import pandas as pd
import numpy as np
import zipfile
import subprocess
import sys

import sklearn
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

import pickle

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


def load_data():
    # Downloads the data from Kaggle and returns a Pandas dataframe of it
    try:
        result = subprocess.run([
            'kaggle', 'datasets', 'download', 
            'nagpalprabhavalkar/tech-use-and-stress-wellness'
        ], check=True, capture_output=True, text=True)
        print("Dataset downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

    with zipfile.ZipFile('tech-use-and-stress-wellness.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    df = pd.read_csv('Tech_Use_Stress_Wellness.csv')
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical = list(df.dtypes[df.dtypes == 'object'].index)
    
    #String formatting
    for col in categorical:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    return df

def train_model(df):
    # Assume y is the sum of weekly_anxiety_score and weekly_depression_score
    df['weekly_depression_anxiety_score'] = df.weekly_anxiety_score + df.weekly_depression_score
    y_train = df.weekly_depression_anxiety_score
    
    x_columns = ['stress_level', 'laptop_usage_hours', 'daily_screen_time_hours', 'physical_activity_hours_per_week', 
                    'gaming_hours', 'mindfulness_minutes_per_day', 'entertainment_hours', 'age', 'sleep_quality', 'phone_usage_hours']
    
    train_dict = df[x_columns].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(sparse=False), 
        preprocessing.StandardScaler(),
        RandomForestRegressor(
            max_depth=3, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            n_estimators=200
        )
    )
    pipeline.fit(train_dict, y_train)

    return pipeline

def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')