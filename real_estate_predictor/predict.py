import pandas as pd
import numpy as np
import requests
import os
import pickle
key = os.environ['REPLIERS_KEY']

from real_estate_predictor.utils.validate_input import process_input


from real_estate_predictor.config.config import MODEL_FILE
model_path = MODEL_FILE
model = pickle.load(open(model_path, 'rb'))

def extract_input(mlsNumber: str):
    url = f"https://api.repliers.io/listings/{mlsNumber}"
    headers = {'repliers-api-key': key}
    r = requests.request("GET",url, headers=headers)
    data = r.json()
    return mlsNumber, data

def predict(data):
    df = process_input(data)
    prediction = model.predict(df) 
    
    return prediction
    