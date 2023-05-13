from flask import Flask, request
from flask_cors import CORS
import pickle
import pandas as pd
import ast
import datetime as dt
import numpy as np
import requests as r
import os

from scripts import api_helper_functions

# import argparse
# parser  = argparse.ArgumentParser(description = 'Enter the amount of days to retrieve information from')
# parser.add_argument("mlsNumber", help='A natural number for the number of days to retrieve information from', metavar='days')
# args = parser.parse_args()
# listing_df, type_ = api_helper_functions.get_listing(args.mlsNumber)
# if type_ == 'lease':
#     df_listing = api_helper_functions.feature_engineer_single_listing(listing_df, lease_columns, lease_feat_index)
#     model = lease_model
# else:
#     df_listing = api_helper_functions.feature_engineer_single_listing(listing_df, sale_columns, sale_feat_index)
#     model = sale_model
# value = predict_listing(df_listing, model)
# print(str(value[0]))

def predict_listing(df, model):
    print(model)
    y_pred = model.predict(df)

    return str(np.round(y_pred[0], 0))


sale_file = pickle.load(open(os.getcwd()+"\\models\\sale_model_parameters.sav", 'rb'))
lease_file = pickle.load(open(os.getcwd()+"\\models\\lease_model_parameters.sav", 'rb'))

sale_model = sale_file['model']
sale_columns = sale_file['columns']
sale_feat_index = sale_file['idx']

lease_model = lease_file['model']
lease_columns = lease_file['columns']
lease_feat_index = lease_file['idx']

print(sale_model)
print(lease_model)

app = Flask(__name__)

CORS(app)

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    data = request.args.get('mlsNumber')
    listing_df, type_ = api_helper_functions.get_listing(data)
    print(type_)
    if type_.lower() == 'lease':
        df_listing = api_helper_functions.feature_engineer_single_listing(listing_df, lease_columns, lease_feat_index)
        model = lease_model
        print(model)
    else:
        df_listing = api_helper_functions.feature_engineer_single_listing(listing_df, sale_columns, sale_feat_index)
        model = sale_model

    value = predict_listing(df_listing, model)

    return {"prediction": value}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8000)