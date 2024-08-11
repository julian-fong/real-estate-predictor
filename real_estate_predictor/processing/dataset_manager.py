import requests
import pandas as pd
import datetime as dt
import os
from ..utils.generate_dataset import extract_raw_data
key = os.environ['REPLIERS_KEY']

#GLOBAL VARIABLES
today = dt.date.today().strftime("%d/%m/%Y")
month = today.month
year = today.year
years_behind = 2

neighbourhood_minSoldDate = "2021-01-01"
neighbourhood_maxSoldDate = "2023-12-31"
neighbourhood_minListDate = "2021-01-01"
neighbourhood_maxListDate = "2023-12-31"

def generate_listings_dataset():
    import time
    raw_df = pd.DataFrame()
    for year_ in range(1, years_behind+1):
        start_date = dt.date(year-year_, 1, 1)
        end_date = dt.date(year-year_, 12, 31)
        print(f"the start date is: {start_date} and the end date is: {end_date}")
        url = f"https://api.repliers.io/listings?resultsPerPage=100&type=lease&type=sale&fields=soldDate,address.city,address.area,address.district,address.neighborhood,details.numBathrooms,details.numBedrooms,details.style,listPrice,listDate,daysOnMarket,details.sqft,details.propertyType,type,class,map,soldPrice&pageNum=1&minSoldDate={start_date}&maxSoldDate={end_date}&class=condo&class=residential&status=U&lastStatus=Lsd&lastStatus=Sld&listings=false&page=2"
        payload = {}
        headers = {'repliers-api-key': key}
        r = requests.request("GET",url, params=payload, headers=headers)
        print(r)
        data = r.json()
        numPages = data['numPages']
        print(numPages)
        
        for i in range(1, numPages+1):
            url = f"https://api.repliers.io/listings?resultsPerPage=100&type=lease&type=sale&fields=soldDate,address.city,address.area,address.district,address.neighborhood,address.zip,details.numBathrooms,details.numBedrooms,details.style,listPrice,listDate,details.sqft,details.propertyType,details.numParkingSpace,details.numGarageSpaces,details.numKitchens,details.numDrivewaySpaces,details.description,details.numParkingSpaces,details.extras,details.numRooms,condominium.ammenities,condominium.fees,nearby.ammenities,type,class,map,soldPrice&pageNum={i}&minSoldDate={start_date}&maxSoldDate={end_date}&class=condo&class=residential&status=U&lastStatus=Lsd&lastStatus=Sld"
            data = None
            df = None
            start_time = time.time()
            payload = {}
            headers = {'repliers-api-key': key}
            r = requests.request("GET",url, params=payload, headers=headers)
            end_time = time.time()
            print(f"index {i} took {end_time - start_time} seconds")

            try:
                data = r.json()

                df = pd.DataFrame(data['listings'])
                df = extract_raw_data(df)
                raw_df = pd.concat([raw_df, df], axis = 0)
                
            except:
                print(f"could not get page {i}")
                continue

            if i % 10 == 0:
                time.sleep(5)

def generate_neighbourhoods_dataset():
    start_date = dt.date(year-years_behind, 1, 1)
    end_date = dt.date(year-years_behind, 12, 31)    
    pass

def save_dataset(df):
    pass

