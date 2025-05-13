import pandas as pd
import numpy as np
import pytest
from real_estate_predictor.models.model import *
from sklearn.model_selection import train_test_split

data = {'listPrice': [2200.0, 2400.0, 850000.0, 2700.0, 750000.0, 2750000.0, 700000.0, 2900.0, 3500.0, 3100.0],
 'soldPrice': [2200.0, 2400.0, 820000.0, 2600.0, 570000.0, 2600000.0, 710000.0, 2800.0, 690000.0, 3000.0],
 'latitude': [43.7, 43.71, 43.72, 43.73, 43.74, 43.75, 43.76, 43.77, 43.78, 43.79],
 'longitude': [-79.4, -79.41, -79.42, -79.43, -79.44, -79.45, -79.46, -79.47, -79.48, -79.49],
 'numBathrooms': [1, 2, 3, 1, 2, 4, 3, 2, 1, 2],
 'numBedrooms': [1, 2, 3, 1, 2, 4, 3, 2, 1, 3],
 'numKitchens': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 'numRooms': [5, 4, 8, 4, 5, 9, 7, 6, 5, 4],
 'numParkingSpaces': [0, 1, 4, 2, 3, 4, 2, 3, 2, 1],
 'numGarageSpaces': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
 'numDrivewaySpaces': [0, 1, 2, 0, 1, 0, 2, 0, 2, 0],
 'class_CondoProperty': [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
 'class_ResidentialProperty': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
 'type_Lease': [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
 'type_Sale': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
 'city_Toronto': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
 'area_Toronto': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
 'district_Toronto C01': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'district_Toronto C02': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'district_Toronto C03': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'district_Toronto C04': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'district_Toronto C05': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'district_Toronto C06': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
 'district_Toronto C07': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 'district_Toronto C08': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 'district_Toronto C09': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 'district_Toronto C10': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 'neighborhood_Neighborhood A': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'neighborhood_Neighborhood B': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'neighborhood_Neighborhood C': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'neighborhood_Neighborhood D': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'neighborhood_Neighborhood E': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'neighborhood_Neighborhood F': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
 'neighborhood_Neighborhood G': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 'neighborhood_Neighborhood H': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 'neighborhood_Neighborhood I': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 'neighborhood_Neighborhood J': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 'zip_M5P0A0': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
 'zip_M5P1A1': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'zip_M5P2A2': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'zip_M5P3A3': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'zip_M5P4A4': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'zip_M5P5A5': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'zip_M5P6A6': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
 'zip_M5P7A7': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 'zip_M5P8A8': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 'zip_M5P9A9': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 'style_2-Storey': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'style_Apartment': [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
 'style_Bungalow': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 'style_Detached': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
 'sqft_0-499': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'sqft_500-599': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 'sqft_600-699': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
 'sqft_700-799': [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
 'sqft_800-899': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 'propertyType_Comm Element Condo': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'propertyType_Condo Apt': [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
 'propertyType_Detached': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]}

test_df = pd.DataFrame(data)

X = test_df.drop(columns=['soldPrice'])
y = test_df['soldPrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
        'n_estimators': [100, 300, 1000],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [7, 10, 15],
        'lambda': [5, 7, 10, 15],
        }

def test_instantiation_model():
    model = XGBoostRegressor(params=params)
    assert isinstance(model, BaseModel)
    assert model.params
    
def test_fit_and_predict():
    model = XGBoostRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(y_test)
    assert isinstance(predictions, np.ndarray)
    
def test_gridsearch():
    model = XGBoostRegressor(param_grid=params)
    grid_search_results_df, best_params = model.grid_search(X_train, y_train)
    assert isinstance(grid_search_results_df, pd.DataFrame)
    assert isinstance(best_params, dict)
    
    model.set_model_params(**best_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)