"""Test file for repliers generate_dataset."""
import os
import pytest
import pandas as pd
import datetime as dt
from real_estate_predictor.utils.generate_dataset import retrieve_repliers_listing_request, retrieve_repliers_neighbourhood_request
from real_estate_predictor.config.api_config import listings_parameters, neighbourhoods_parameters
key = os.environ['REPLIERS_KEY']

@pytest.fixture
def repliers_request():
    today = dt.date.today()
    month = today.month
    year = today.year
    years_behind = 2
    start_date = dt.date(year-years_behind, 1, 1)
    end_date = dt.date(year-years_behind, 12, 31)
    neighbourhood_minSoldDate = "2021-01-01"
    neighbourhood_maxSoldDate = "2023-12-31"
    neighbourhood_minListDate = "2021-01-01"
    neighbourhood_maxListDate = "2023-12-31"
    
    return {
            "start_date": start_date,
            "end_date": end_date,
            "neighbourhood_maxListDate": neighbourhood_maxListDate, 
            "neighbourhood_maxSoldDate": neighbourhood_maxSoldDate, 
            "neighbourhood_minListDate": neighbourhood_minListDate, 
            "neighbourhood_minSoldDate": neighbourhood_minSoldDate
            }
    
def test_repliers_listing_request(repliers_request):
    request_params = repliers_request
    r, numPages, data = retrieve_repliers_listing_request(request_params["start_date"], repliers_request["end_date"], 1, listings_parameters, include_listings=False)
    
    assert r.status_code == 200
    assert numPages != 0
    assert numPages == 2205
    assert data["listings"] == []
    
    
def test_repliers_listings_request_no_listings(repliers_request):
    request_params = repliers_request
    r, numPages, data = retrieve_repliers_listing_request(request_params["start_date"], repliers_request["end_date"], 1, listings_parameters)

    assert r.status_code == 200
    assert numPages != 0
    assert len(data["listings"]) != 0
    
    
def test_repliers_neighbourhoods_request(repliers_request):
    request_params = repliers_request
    r, numPages, data = retrieve_repliers_neighbourhood_request(request_params["start_date"], repliers_request["end_date"], 1, listings_parameters, include_listings=False)
    
    assert r.status_code == 200
    assert numPages != 0
    assert numPages == 2205
    assert data["listings"] == []