import pandas as pd
import numpy as np
import requests
import os
import datetime as dt
import re
import ast
from scripts import load_data

pd.set_option('display.max_columns', 999)
#Helper Functions

def convert_to_num(df):
    for col in df.columns:
        try:
            pd.to_numeric(df[col])
        except:
            print(f"Column {col} cannot be converted to numeric")
            continue

def labelDom(row):
    if row['DOM'] < 14:
        return True
    else:
        return False

def binaryDom(df):
    df['binaryDOM'] = df.apply(lambda row: labelDom(row), axis = 1)

def createDOM(df: pd.DataFrame, timestamp = False, convert = False):
    dom = df['soldDate'] - df['listDate']
    df['DOM'] = dom.astype("string")
    df.dropna(subset=['DOM'], inplace = True)
    df['DOM'] = df['DOM'].str.slice_replace(start=2)
    df['DOM'] = df['DOM'].astype("int")

def dfRemoveNA(df):
    df = df.replace('', np.nan)
    if sum(df.isna().sum()) != 0:
        df = df.dropna()
    return df

def dfFillNA(df, column):
    df = df.replace('', np.nan)
    df = df.fillna(value = {column: "No "+column})
    return df

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

def dfPriceDifference(df: pd.DataFrame):
    #should only be for clustering purposes only
    df['PriceDifference'] = df['soldPrice'] - df['listPrice']

def dfDatesRemoveTimeStamp(df: pd.DataFrame):
    df['listDate'] = df['listDate'].str.slice_replace(start = 10, repl="")
    df['soldDate'] = df['soldDate'].str.slice_replace(start = 10, repl="")

def dfConvertToDatetime(df: pd.DataFrame, timestamp = False):
    if type(df['listDate'].values[0]) == str:
        dfDatesRemoveTimeStamp(df)
    df['listDate'] = pd.to_datetime(df['listDate'])
    df['soldDate'] = pd.to_datetime(df['soldDate'])


#Data Augmentation


def removeOutliers(df):
    percentile90 = df['soldPrice'].quantile(0.90)
    percentile10 = df['soldPrice'].quantile(0.10)

    iqr = percentile90 - percentile10

    upper_limit = percentile90 + 3 * iqr
    lower_limit = percentile10 - 3 * iqr

    df = df[df['soldPrice'] < upper_limit]
    df = df[df['soldPrice'] > lower_limit]

    return df


def feature_engineering(days = None, type_ = None):
    raw_df = load_data.import_data(days, type_, os.environ['REPLIERS_KEY'])
    
    if raw_df.empty:
        print("Specified days range does not retrieve any data for specific type")
    
    address = ['area', 'city', 'district', 'neighborhood']
    details = ['numBathrooms','numBedrooms','sqft','style',]
    map_ = ['latitude', 'longitude']

    #Extract column values off of dictionary
    df = dfExtractFeatures(raw_df, key = 'address', columns = address)
    df = dfExtractFeatures(df, key = 'details', columns = details)
    df = dfExtractFeatures(df, key = 'map', columns = map_)
    
    #Remove every listing which had a sold price of 0
    df = df[df['soldPrice'] != 0]

    #Replace any empty strings or nan values that are not specific to the pandas DataFrame
    df.replace({"": np.nan}, inplace = True)
    df.replace({float("nan"): np.nan}, inplace = True)

    #Create Days on market variable
    dfConvertToDatetime(df)
    createDOM(df)

    #Create average square feet variable
    df['avg_sqft'] = [(int(value.split("-")[0])+int(value.split("-")[1]))/2 if not isinstance(value, float) and '-' in value else np.nan for value in df['sqft']]
    df['avg_sqft']

    #Encode listPrice and soldPrice as float
    df['listPrice'] = df['listPrice'].astype("float")
    df['soldPrice'] = df['soldPrice'].astype("float")
    df['originalPrice'] = df['originalPrice'].astype("float")

    #Create the price per square feet variable
    df['ppsqft'] = df['listPrice']/df['avg_sqft']

    #Create the bed to bath ratio
    df['bathtobed_ratio'] = df['numBathrooms'].astype("float")/df['numBedrooms'].astype("float")

    #Encode map coordinates as floats
    df['latitude'] = df['latitude'].astype("float")
    df['longitude'] = df['longitude'].astype("float")

    #Create the average price by area variable (calculates the average price of every listing in the area in the dataframe)
    avg_price_by_area = df.groupby('area').agg('mean')['listPrice']

    #Create the average price by city variable (calculates the average price of every listing in the city in the dataframe)
    avg_price_by_city = df.groupby('city').agg('mean')['listPrice']

    #Join in the average price by city and average price by area to the main dataframe
    df_agg = df.join(avg_price_by_area, on = 'area', rsuffix='_by_area')
    df_agg = df_agg.join(avg_price_by_city, on = 'city', rsuffix='_by_city')

    #drop the remaining unnecessary columns
    columns_to_keep = ['originalPrice', 'soldPrice', 'class', 'listPrice', 'area', 'city', 'district', 'neighborhood', 'numBathrooms', 'numBedrooms', 'sqft', 'style', 'latitude', 'longitude', 'DOM', 'avg_sqft', 'ppsqft', 'bathtobed_ratio', 'listPrice_by_area', 'listPrice_by_city']
    columns_to_drop = [col for col in df_agg.columns if col not in columns_to_keep]
    df_agg = df_agg.drop(columns = columns_to_drop, axis = 1)

    #Remove any outliers
    df_agg = removeOutliers(df_agg)
    
    #Encode the categorical variables
    df_agg_cats = pd.get_dummies(df_agg)

    print(f"Feature engineering complete. Dataset for type {type_} has size {df_agg_cats.shape}")

    return df_agg_cats

