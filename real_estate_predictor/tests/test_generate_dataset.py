"""Test file for repliers generate_dataset."""

import os

import pytest

from real_estate_predictor.config.config import (
    LISTING_PARAMETERS,
    NEIGHBOURHOOD_PARAMETERS,
)
from real_estate_predictor.utils.generate_dataset import (
    retrieve_repliers_listing_request,
    retrieve_repliers_neighbourhood_request,
)

key = os.getenv("REPLIERS_KEY")


@pytest.fixture
def repliers_request():
    start_date = "2024-01-01"
    end_date = "2024-01-02"
    neighbourhood_start_date = "2024-01-01"
    neighbourhood_end_date = "2024-03-01"
    return {
        "start_date": start_date,
        "end_date": end_date,
        "neighbourhood_start_date": neighbourhood_start_date,
        "neighbourhood_end_date": neighbourhood_end_date,
    }


def test_repliers_listing_request(repliers_request):
    r, numPages, data = retrieve_repliers_listing_request(
        repliers_request["start_date"],
        repliers_request["end_date"],
        1,
        LISTING_PARAMETERS,
        include_listings=False,
    )

    assert r.status_code == 200
    assert numPages != 0
    assert data["listings"] == []


def test_repliers_listings_request_with_listings(repliers_request):
    r, numPages, data = retrieve_repliers_listing_request(
        repliers_request["start_date"],
        repliers_request["end_date"],
        1,
        LISTING_PARAMETERS,
    )

    assert r.status_code == 200
    assert numPages != 0
    assert len(data["listings"]) != 0


def test_repliers_neighbourhoods_request(repliers_request):
    r, data = retrieve_repliers_neighbourhood_request(
        repliers_request["neighbourhood_start_date"],
        repliers_request["neighbourhood_end_date"],
        listing_type="lease",
        payload=NEIGHBOURHOOD_PARAMETERS,
        neighbourhood="Waterfront Communities C1",
        numBedroom=1,
    )

    assert r.status_code == 200
    assert data["statistics"]
