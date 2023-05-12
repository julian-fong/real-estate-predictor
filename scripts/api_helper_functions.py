import pandas as pd
import numpy as np
import ast
import requests
import os
import datetime as dt

def createDOM():
    pass

def dfExtractFeatures(df: pd.DataFrame, key: str, columns: list):
    if type(df[key].values[0]) == str:
        try:
            serie = df[key].apply(lambda x: ast.literal_eval(str(x)))
            d = pd.DataFrame.from_dict(serie.tolist())
        except:
            print("String does not match proper conversion methods")
    elif type(df[key].values[0]) == dict:
        serie = pd.Series.to_dict(df[key])
        d = pd.DataFrame.from_dict(serie, orient='index')
    for i in range(len(columns)):
        df[columns[i]] = d[columns[i]]

    return df

def grab_average_area(df):
    url = f"https://api.repliers.io/listings?resultsPerPage=1000"
    payload = {'area': df['area'].values[0], 'type': df['type'].values[0], 'status': 'A'}
    headers = {'repliers-api-key': os.environ['REPLIERS_KEY']}
    r = requests.request("GET",url, params=payload, headers=headers)
    data = r.json()
    df_area = pd.DataFrame(data['listings'])
    columns = [col for col in df_area.columns if col != 'listPrice']
    df_area = df_area.drop(columns, axis = 1)
    df['listPrice_by_area'] = np.mean(df_area['listPrice'].astype("float").values)


def grab_average_city(df):
    url = f"https://api.repliers.io/listings?resultsPerPage=1000"
    payload = {'city': df['city'].values[0], 'type': df['type'].values[0], 'status': 'A'}
    headers = {'repliers-api-key': os.environ['REPLIERS_KEY']}
    r = requests.request("GET",url, params=payload, headers=headers)
    data = r.json()
    df_city = pd.DataFrame(data['listings'])
    columns = [col for col in df_city.columns if col != 'listPrice']
    df_city = df_city.drop(columns, axis = 1)
    df['listPrice_by_city'] = np.mean(df_city['listPrice'].astype("float").values)  

def feature_engineer_single_listing(df, model_columns, model_idx):
    #Replace any empty strings or nan values that are not specific to the pandas DataFrame
    df.replace({"": np.nan}, inplace = True)
    df.replace({float("nan"): np.nan}, inplace = True)

    df['listPrice'] = df['listPrice'].astype("float")
    df['soldPrice'] = df['soldPrice'].astype("float")
    
    #Replace any empty strings or nan values that are not specific to the pandas DataFrame
    df.replace({"": np.nan}, inplace = True)
    df.replace({float("nan"): np.nan}, inplace = True)
    
    #Encode map coordinates as floats
    df['latitude'] = df['latitude'].astype("float")
    df['longitude'] = df['longitude'].astype("float")

    #dom
    today = dt.datetime.now()
    if 'listDate' in df.columns:
        try:
            listDate = dt.datetime.strptime(df['listDate'].values[0].split("T")[0], '%Y-%m-%d')
            dom = (today-listDate).day
        except:
            dom = np.nan
    else:
        dom = np.nan 

    df['DOM'] = dom

    #bed bath ratio
    try:
        bedtobath = df['numBathrooms'].astype('float')/df['numBedrooms'].astype('float')
    except:
        bedtobath = np.nan
    
    df['bathtobed_ratio'] = bedtobath

    #sqft and price per sqft
    if '-' in df['sqft']:
        average_sqft = (int(df['sqft'].split("-")[0])+int(df['sqft'].split("-")[1]))/2
        df['avg_sqft'] = average_sqft

        pp_sqft = df['listPrice']/average_sqft
        df['ppsqft'] = pp_sqft
    else:
        avg_sqft = np.nan
        pp_sqft = np.nan

        df['avg_sqft'] = avg_sqft
        df['ppsqft'] = pp_sqft


    #average listing of city
    grab_average_city(df)

    #average listing of area
    grab_average_area(df)

    columns_to_keep = ['originalPrice', 'soldPrice', 'class', 'listPrice', 'area', 'city', 'district', 'neighborhood', 'numBathrooms', 'numBedrooms', 'sqft', 'style', 'latitude', 'longitude', 'DOM', 'avg_sqft', 'ppsqft', 'bathtobed_ratio', 'listPrice_by_area', 'listPrice_by_city']
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
    df_agg = df.drop(columns = columns_to_drop, axis = 1)

    df_agg_cats = pd.get_dummies(df_agg)

    df_model_columns = pd.DataFrame(df_agg_cats, columns = model_columns)

    df_listing = pd.DataFrame(df_model_columns, columns = df_model_columns.columns[model_idx])

    return df_listing

