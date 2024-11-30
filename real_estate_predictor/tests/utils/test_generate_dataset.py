"""Test file for repliers generate_dataset."""
import os
import pytest
import pandas as pd
import datetime as dt
from real_estate_predictor.utils.generate_dataset import retrieve_repliers_listing_request, retrieve_repliers_neighbourhood_request
from real_estate_predictor.config.api_config import LISTING_PARAMETERS, NEIGHBOURHOOD_PARAMETERS, NEIGHBOURHOOD_KEYS, NEIGHBOURHOODS, NEIGHBOURHOOD_NUMBEDROOMS, NEIGHBOURHOOD_TYPES
key = os.environ['REPLIERS_KEY']

@pytest.fixture
def repliers_request():
    start_date = "2024-01-01"
    end_date = "2024-01-02"
    return {
            "start_date": start_date,
            "end_date": end_date,
            }
    
def test_repliers_listing_request(repliers_request):
    request_params = repliers_request
    r, numPages, data = retrieve_repliers_listing_request(request_params["start_date"], repliers_request["end_date"], 1, LISTING_PARAMETERS, include_listings=False)
    
    assert r.status_code == 200
    assert numPages != 0
    assert data["listings"] == []
    
    
def test_repliers_listings_request_with_listings(repliers_request):
    request_params = repliers_request
    r, numPages, data = retrieve_repliers_listing_request(request_params["start_date"], repliers_request["end_date"], 1, LISTING_PARAMETERS)

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