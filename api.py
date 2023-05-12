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

sale_file = pickle.load(open(os.getcwd()+"\\models\\sale_model_parameters.sav", 'rb'))
lease_file = pickle.load(open(os.getcwd()+"\\models\\lease_model_parameters.sav", 'rb'))

sale_model = sale_file['model']
sale_columns = sale_file['columns']
sale_feat_index = sale_file['idx']

lease_model = lease_file['model']
lease_columns = lease_file['columns']
lease_feat_index = lease_file['idx']

# app = Flask(__name__)

# CORS(app)

if __name__ == "__main__":
    for col in sale_columns:
        print(col)