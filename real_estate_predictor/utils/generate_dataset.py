import os
import requests
import urllib.parse
key = os.environ['REPLIERS_KEY']

import pandas as pd

def retrieve_repliers_listing_request(start_date: str, end_date: str, page_num: int, fields: dict):
    print(f"the start date is: {start_date} and the end date is: {end_date}")
    url = f"https://api.repliers.io/listings?resultsPerPage=100&type=lease&type=sale&fields=soldDate,address.city,address.area,address.district,address.neighborhood,details.numBathrooms,details.numBedrooms,details.style,listPrice,listDate,daysOnMarket,details.sqft,details.propertyType,type,class,map,soldPrice&pageNum=1&minSoldDate={start_date}&maxSoldDate={end_date}&class=condo&class=residential&status=U&lastStatus=Lsd&lastStatus=Sld&listings=false&page=2"
    payload = fields
    headers = {'repliers-api-key': key}
    r = requests.request("GET",url, params=payload, headers=headers)
    print(r)
    data = r.json()
    numPages = data['numPages']
    numPages 

def retrieve_repliers_neighbourhood_request(start_date: str, end_date: str, fields: dict):
    print(f"the start date is: {start_date} and the end date is: {end_date}")
    url = f"https://api.repliers.io/listings?resultsPerPage=100&type=lease&type=sale&fields=soldDate,address.city,address.area,address.district,address.neighborhood,details.numBathrooms,details.numBedrooms,details.style,listPrice,listDate,daysOnMarket,details.sqft,details.propertyType,type,class,map,soldPrice&pageNum=1&minSoldDate={start_date}&maxSoldDate={end_date}&class=condo&class=residential&status=U&lastStatus=Lsd&lastStatus=Sld&listings=false&page=2"
    payload = fields
    headers = {'repliers-api-key': key}
    r = requests.request("GET",url, params=payload, headers=headers)
    print(r)
    data = r.json()
    numPages = data['numPages']
    numPages     

def extract_raw_data_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts relevant data from the raw dataframe.
    Parameters
    ----------
    
    df : pd.DataFrame
        Incoming raw dataframe containing listings from a specific
        period. Assumes the following columns exists
        - details
        - address
        - condominium
        - nearby
        - map
    
    Returns
    -------
    df : pd.DataFrame
        Extracted data
    
    """
    
    keys = ['details', 'address', 'condominium', 'nearby','map']
    
    for key in keys:
        if key not in df.columns:
            raise ValueError(f"missing key {key} in dataframe columns")
        
    details_df = pd.DataFrame.from_records(df['details'])
    address_df = pd.DataFrame.from_records(df['address'])
    condo_df = pd.DataFrame.from_records(df['condominium'])
    nearby_df = pd.DataFrame.from_records(df['nearby'])
    map_df = pd.DataFrame.from_records(df['map'])

    df['city'] = address_df['city']
    df['area'] = address_df['area']
    df['district'] = address_df['district']
    df['neighborhood'] = address_df['neighborhood']
    df['zip'] = address_df['zip']

    df['latitude'] = map_df['latitude']
    df['longitude'] = map_df['longitude']

    df['fees'] = condo_df['fees']
    df['condo_ammenities'] = condo_df['ammenities']

    df['ammenities'] = nearby_df['ammenities']

    df['numBathrooms'] = details_df['numBathrooms']
    df['numBedrooms'] = details_df['numBedrooms']
    df['style'] = details_df['style']
    df['numKitchens'] = details_df['numKitchens']
    df['numRooms'] = details_df['numRooms']
    df['numParkingSpaces'] = details_df['numParkingSpaces']
    df['sqft'] = details_df['sqft']

    df['description'] = details_df['description']
    df['extras'] = details_df['extras']
    df['propertyType'] = details_df['propertyType']
    df['numGarageSpaces'] = details_df['numGarageSpaces']
    df['numDrivewaySpaces'] = details_df['numDrivewaySpaces']


    df = df.drop(columns=['details', 'address','condominium','map','nearby'])
    df = df.map(str)
    
    return df