import os
import pathlib
import pickle

# Ensure compatibility with different operating systems when using pathlib
import platform

import requests

from real_estate_predictor.config.config import LEASE_MODEL_FILE, SALE_MODEL_FILE
from real_estate_predictor.utils.validate_input import process_input

key = os.environ["REPLIERS_KEY"]

plt = platform.system()
if plt == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
elif plt == "Linux":
    pathlib.WindowsPath = pathlib.PosixPath

sale_model_path = SALE_MODEL_FILE
with open(sale_model_path, "rb") as f:
    sale_model = pickle.load(f)

lease_model_path = LEASE_MODEL_FILE
with open(lease_model_path, "rb") as f:
    lease_model = pickle.load(f)


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


def predict_sale_listing(mlsNumber, listing_type="sale"):
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data, listing_type)[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}


def predict_lease_listing(mlsNumber, listing_type="lease"):
    mlsNumber, data = extract_input(mlsNumber)
    prediction = predict(data, listing_type)[0].item()
    return {"mlsNumber": mlsNumber, "prediction": prediction}
