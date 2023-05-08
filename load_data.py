import pandas as pd
import numpy as np
import requests
import os
import datetime as dt

def get_dates(days):
    date_list = []
    today = dt.datetime.now()
    daysBehind = dt.timedelta(days = days)
    start_raw = today-daysBehind
    start_date = str(start_raw).split(" ")[0]

    current_date = start_raw
    date_list.append(start_date)
    while current_date <= (today- dt.timedelta(days = 14)):
        current_date = current_date + dt.timedelta(days = 7)
        date_list.append(str(current_date).split(" ")[0])

    return date_list

def get_data(type_, date_list, api_key):
    if type_.lower() == 'lease':
        lastStatus = ""
    else:
        lastStatus = "&lastStatus=Lsd"

    raw_df = pd.DataFrame()
    for i in range(len(date_list)):
        start_date = date_list[i]
        end_date = dt.datetime.strptime(start_date, '%Y-%m-%d') + dt.timedelta(days = 6)
        end_date = dt.datetime.strftime(end_date, '%Y-%m-%d')
        url = f'https://api.repliers.io/listings?resultsPerPage=10000&minSoldDate={start_date}&maxSoldDate={end_date}&status=U'+lastStatus+f"&type={type_}"

        payload = {}
        headers = {'repliers-api-key': api_key}
        r = requests.request("GET",url, params=payload, headers=headers)
        data = r.json()
        df = pd.DataFrame(data['listings'])
        #Raw Dataset
        raw_df = pd.concat([raw_df, df],axis = 0, ignore_index= True)
        print(len(raw_df))

    return raw_df

def import_data(days_amount, type_, api_key, save = False):
    dates = get_dates(days_amount)
    if not type_:
        raw_lease_data = get_data('lease', dates, api_key)
        raw_sale_data = get_data('sale', dates, api_key)
        if save:
            raw_lease_data.to_csv(os.getcwd()+"\\data\\raw_lease_data.csv", index = False)
            raw_sale_data.to_csv(os.getcwd()+"\\data\\raw_sale_data.csv", index = False)
        else:
            return raw_lease_data, raw_sale_data
    else:
        raw_data = get_data(type_, dates, api_key)
        if save:
            raw_data.to_csv(os.getcwd()+f"\\data\\raw_{type_}_data.csv", index = False)
        else:
            return raw_data
        
if __name__ == "__main__":

    import_data(1460, None, os.environ['REPLIERS_KEY'], True)