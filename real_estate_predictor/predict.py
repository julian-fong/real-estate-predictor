import pandas as pd
import numpy as np
import requests
import os
import pickle
key = os.environ['REPLIERS_KEY']

from real_estate_predictor.utils.validate_input import process_input


from real_estate_predictor.config.config import SALE_MODEL_FILE
sale_model_path = SALE_MODEL_FILE
sale_model = pickle.load(open(sale_model_path, 'rb'))

def extract_input(mlsNumber: str):
    url = f"https://api.repliers.io/listings/{mlsNumber}"
    headers = {'repliers-api-key': key}
    r = requests.request("GET",url, headers=headers)
    data = r.json()
    return mlsNumber, data

def predict(data, type):
    if type == 'sale':
        model = sale_model
    else:
        model = lease_model
    df = process_input(data, model)
    prediction = model.predict(df) 
    
    return prediction
    