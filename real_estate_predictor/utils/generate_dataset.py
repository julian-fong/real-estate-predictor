import os
import time
import pathlib
import requests
key = os.environ['REPLIERS_KEY']

import pandas as pd

#for function subtract_months
import datetime as dt
from dateutil.relativedelta import relativedelta

from pathlib import Path

def retrieve_repliers_listing_request(
    start_date: str, 
    end_date: str, 
    page_num: int, 
    payload: dict, 
    include_listings = True, 
    verbose = False
    ):
    
    if verbose:
        start_time = time.time()
    url = f"https://api.repliers.io/listings?&pageNum={page_num}&minSoldDate={start_date}&maxSoldDate={end_date}"
    if not include_listings:
        url += "&listings=False"
    payload = payload
    headers = {'repliers-api-key': key}
    r = requests.request("GET",url, params=payload, headers=headers)
    if r.status_code != 200:
        print(f"{r.status_code} error returned from repliers: {r.json()[0]['msg']}")
    data = r.json()
    numPages = data['numPages']
    if verbose:
        end_time = time.time()
        print(f"url: {r.url}")
        print(f"index {page_num} took {end_time - start_time} seconds with a response of {r}")
    return r, numPages, data

def retrieve_repliers_neighbourhood_request(
    start_date: str, 
    end_date: str, 
    payload: dict,
    type: str,
    neighbourhood: str,
    numBedroom: int,
    verbose = False
    ):
    if type == "lease":
        lastStatus = "Lsd"
    else:
        lastStatus = "Sld"
    if verbose:
        print(f"the start date is: {start_date} and the end date is: {end_date}")
        start_time = time.time()
        
    url = (
        f"https://api.repliers.io/listings?minSoldDate={start_date}&maxSoldDate={end_date}"
        f"&minBeds={numBedroom}&maxBeds={numBedroom}"
        f"&minListDate={start_date}&maxListDate={end_date}&type={type}"
        f"&neighborhood={neighbourhood}&lastStatus={lastStatus}"
    )
    payload = payload
    headers = {'repliers-api-key': key}
    r = requests.request("GET",url, params=payload, headers=headers)
    data = r.json()
    if verbose:
        end_time = time.time()
        print(f"url: {r.url}")
        print(f"data of {neighbourhood, numBedroom, type} took {end_time - start_time} seconds with a response of {r}")
        if r.status_code != 200:
            print(r.json())
            
    return r, data

def save_raw_dataset(
    df: pd.DataFrame,
    format: str,
    path: str = None, 
    is_listings_dataset = True, 
    ):
    df.reset_index(drop = True)
    if is_listings_dataset:
        type = "listing"
    else:
        type = "neighbourhood"
        
    #if path is not specified, put it in the storage/datasets folder
    if not path:
        path = pathlib.Path(__file__).parent.parent.absolute().joinpath('storage', 'datasets')
    else:
        path = Path(path)
    path = str(path).replace("\\\\", "\\")+"\\"
    
    #create the file name
    date = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    file_name = f"{type}_dataset_{date}"
    
    if format == "json":
        file_name+=".json"
        full_path = path+file_name
        df.to_json(full_path, index = False)
    elif format == "csv":
        file_name+=".csv"
        full_path = path+file_name
        df.to_csv(full_path, index=False)
    else:
        raise ValueError(f"Unknown format {format}")
    
    
def save_dataset(
    df: pd.DataFrame,
    format: str,
    path: str = None,
    file_name: str = None,
    ):
        
    #if path is not specified, put it in the storage/datasets folder
    if not path:
        path = pathlib.Path(__file__).parent.parent.absolute().joinpath('storage', 'datasets')
    else:
        path = Path(path)
    path = str(path).replace("\\\\", "\\")+"\\"
    
    #create the file name
    date = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if file_name:
        file_name += f"_{date}"
    else:
        file_name = f"dataset_{date}"
    
    if format == "json":
        file_name+=".json"
        full_path = path+file_name
        df.to_json(full_path, index = False)
    elif format == "csv":
        file_name+=".csv"
        full_path = path+file_name
        df.to_csv(full_path, index=False)
    else:
        raise ValueError(f"Unknown format {format}")