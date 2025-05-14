import requests
import os
import pickle

key = os.environ["REPLIERS_KEY"]

from real_estate_predictor.utils.validate_input import process_input


from real_estate_predictor.config.config import SALE_MODEL_FILE
from real_estate_predictor.config.config import LEASE_MODEL_FILE

sale_model_path = SALE_MODEL_FILE
sale_model = pickle.load(open(sale_model_path, "rb"))

lease_model_path = LEASE_MODEL_FILE
lease_model = pickle.load(open(sale_model_path, "rb"))


def extract_input(mlsNumber: str):
    url = f"https://api.repliers.io/listings/{mlsNumber}"
    headers = {"repliers-api-key": key}
    r = requests.request("GET", url, headers=headers, timeout=10)
    data = r.json()
    return mlsNumber, data


def predict(data, listing_type):
    if listing_type == "sale":
        model = sale_model
    else:
        model = lease_model
    df = process_input(data, model)
    prediction = model.predict(df)

    return prediction
