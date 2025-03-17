import datetime as dt 
import requests
import pandas as pd
import os
import re
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import ast

#Feature Engineering

## Sqft

def helper_create_avg_sqft(x):
    try:
        if x is not None and x != '':
            if "-" in x:
                return (int(x.split("-")[0]) + int(x.split("-")[1]))/2
            else:
                pass
                # pattern = r"\d+"
                # matches = re.findall(pattern, x)
                # return int(matches[0])
    except:
        return np.nan
    
def create_sqft_avg_column(df):
    """
    Assumes the existence of a column named `sqft` in the dataframe.
    """
    df['sqft_avg'] = pd.to_numeric(df['sqft'].apply(helper_create_avg_sqft), errors = 'coerce')

    return df

def create_ppsqft_column(df):
    """
    Assumes the existence of a column named `listPrice` and `sqft_avg` in the dataframe.
    """
    df['ppsqft'] = df['listPrice']/df['sqft_avg']
    return df

## Bed and Bath

def create_bedbathRatio_column(df):
    """
    Assumes the existence of the columns `numBedrooms` and `numBathrooms` in the dataframe of type float.
    """
    #bed bath ratio
    #change dtype of numBed and numBath to numeric while filling errors with np.NaN, divide them and fill any errors with NaN
    
    if not isinstance(df['numBedrooms'][0], float):
        _numBedrooms = pd.to_numeric(df['numBedrooms'], errors='coerce')
    else:
        _numBedrooms = df['numBedrooms']
        
    if not isinstance(df['numBathrooms'][0], float):
        _numBathrooms = pd.to_numeric(df['numBathrooms'], errors='coerce')
    else:
        _numBathrooms = df['numBathrooms']
    
    df['bedbathRatio'] = _numBedrooms.div(_numBathrooms).replace([np.inf, -np.inf], np.nan)
    return df

## Ammenities and Condo Ammenities

def convert_pd_convert_literal_eval_to_list(df, column):
    """
    Assumes that the values in the columns are all of type string
    """
    df[column] = df[column].apply(lambda x: ast.literal_eval(x))
    return df[column]

def helper_get_unique_ammenities(df, column):
    """
    Assumes the existence of columns `condo_ammenities` and `ammenities` inside the dataframe.
    
    """
    ammenities = []
    if column !=  "condo_ammenities" and column != "ammenities":
        raise ValueError("incorrect column name, expected condo_ammenities or ammenities")

    for i in range(len(df)):
        if df[column].values[i]:
            for item in df[column].values[i]:
                if item not in ammenities:
                    ammenities.append(item)
                    
    return ammenities

def helper_find_existence_of_ammenity(x: list, value, errors = 'coerce'):
    """
    Given a list of ammenities of format ["item1", "item2", ...]
    check to see if `value` exists in the list
    
    if the ammenities list is None i.e [] for the given row, we return np.nan
    """
    if not x or len(x) == 0:
        return np.nan
    if value in x:
        return True
    
    return False

def create_ammenities_flag_columns(df, ammenities = None, drop_after = True):
    """
    Creates new flag columns based on a passed `list_of_ammenities`.
    eg. suppose `list_of_ammnenities` = ["Park", "School"]
    Then the dataframe will contain new columns `hasPark` and `hasSchool`
    
    ammenities: one of `ammenities` or `condo_ammenities`
    
    Correspondings to `has_ammenities_flags` and `has_condo_ammenities_flags` in the config.
    
    Assumes the columns `ammenities` and `condo_ammenities` are already present in the dataframe.
    """
    # if ammenities != "ammenities" and ammenities != "condo_ammenities":
    #     raise ValueError("invalid ammenities passed, can only be `ammenities` or `condo_ammenities`")
    
    if not ammenities:
        #first check to see if either column is present
        available_ammenity_columns = []
        if "ammenities" in df.columns:
            available_ammenity_columns.append("ammenities")
        if "condo_ammenities" in df.columns:
            available_ammenity_columns.append("condo_ammenities")
        
        if not available_ammenity_columns:
            raise ValueError("no ammenities column found in the dataframe")    
            
        for type_of_ammenity in available_ammenity_columns:
            list_of_ammenities = helper_get_unique_ammenities(df, type_of_ammenity)
           
            for ammenity in list_of_ammenities:
                df[f"has{ammenity.capitalize()}"] = df[type_of_ammenity].apply(helper_find_existence_of_ammenity, args = (ammenity,)).astype(bool)
        
            if drop_after:
                df.drop(columns = [type_of_ammenity], inplace = True)
    
    else:
        if ammenities != "ammenities" and ammenities != "condo_ammenities":
            raise ValueError("invalid ammenities passed, can only be `ammenities` or `condo_ammenities`")
        
        list_of_ammenities = helper_get_unique_ammenities(df, ammenities)
        
        for ammenity in list_of_ammenities:
            df[f"has{ammenity.capitalize()}"] = df[ammenities].apply(helper_find_existence_of_ammenity, args = (ammenity,)).astype(bool)
            
        if drop_after:
            df.drop(columns = [ammenities], inplace = True)
        
    return df
    

def helper_calculate_num_ammenities(x):
    count = 0
    try:
        for item in x:
            if item != '':
                count += 1
        return count
    except:
        return np.nan


def create_num_ammenities_column(df):
    """
    Assumes the existence of columns `ammenities` and `condo_ammenities` in the dataframe.
    """

    df['numAmmenities'] = df['ammenities'].apply(helper_calculate_num_ammenities)
    df['numCondoAmmenities'] = df['condo_ammenities'].apply(helper_calculate_num_ammenities)

    return df

## Postal Code

def helper_split_postal_code(x, num_chars = 2):
    
    try:
        x = x[:num_chars]
    except:
        return np.nan
    
    return x

def create_split_postalcode_column(df, num_chars = 2):
    """
    Assumes existance of a column named zip
    """
    if num_chars != 3 and num_chars != 2:
        raise ValueError("invalid int passed, can only be 2 or 3")
    
    df[f"postal_code_split_{num_chars}"] = df["zip"].apply(lambda x: helper_split_postal_code(x, num_chars))
    
    return df

## Datetime

def create_dom_column(df):
    """
    Assumes the columns `soldDate` and `listDate` are present in the dataframe of pandas datetime formatting.
    """
    #calculate DOM
    # df['soldDate_pd'] = pd.to_datetime(df['soldDate'])
    # df['listDate_pd'] = pd.to_datetime(df['listDate'])
    df['daysOnMarket'] = df['soldDate'].sub(df['listDate'], fill_value = np.nan)
    df['daysOnMarket'] = df['daysOnMarket'].div(np.timedelta64(1, 'D'), fill_value = np.nan)

    return df

## neighbourhoods dataset

def create_previous_month_ppsqft_columns(df):
    """
    Assumes the existence of columns: 
        `avg_soldPrice_currentL*M` where * = 1, 3, 6
        `med_soldPrice_currentL*M` where * = 1, 3, 6
        `avg_listPrice_currentL*M` where * = 1, 3, 6
        `med_listPrice_currentL*M` where * = 1, 3, 6 
        `sqft_avg` in the dataframe
    """
    df['avg_soldPrice_ppsqft_currentL1M'] = pd.to_numeric(df['avg_soldPrice_currentL1M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['med_soldPrice_ppsqft_currentL1M'] = pd.to_numeric(df['med_soldPrice_currentL1M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['avg_listPrice_ppsqft_currentL1M'] = pd.to_numeric(df['avg_listPrice_currentL1M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['med_listPrice_ppsqft_currentL1M'] = pd.to_numeric(df['med_listPrice_currentL1M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")

    df['avg_soldPrice_ppsqft_currentL3M'] = pd.to_numeric(df['avg_soldPrice_currentL3M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['med_soldPrice_ppsqft_currentL3M'] = pd.to_numeric(df['med_soldPrice_currentL3M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['avg_listPrice_ppsqft_currentL3M'] = pd.to_numeric(df['avg_listPrice_currentL3M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['med_listPrice_ppsqft_currentL3M'] = pd.to_numeric(df['med_listPrice_currentL3M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")

    df['avg_soldPrice_ppsqft_currentL6M'] = pd.to_numeric(df['avg_soldPrice_currentL6M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['med_soldPrice_ppsqft_currentL6M'] = pd.to_numeric(df['avg_soldPrice_currentL6M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['avg_listPrice_ppsqft_currentL6M'] = pd.to_numeric(df['avg_listPrice_currentL6M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")
    df['med_listPrice_ppsqft_currentL6M'] = pd.to_numeric(df['avg_listPrice_currentL6M'].div(df['sqft_avg']).replace([np.inf, -np.inf], np.nan), errors = "coerce")

    return df

def create_difference_bymonth_columns(df):
    """
    Assumes the existence of columns: 
    Assumes the existence of columns: 
        `avg_soldPrice_currentL*M` where * = 1, 3, 6
        `avg_listPrice_currentL*M` where * = 1, 3, 6
        `med_soldPrice_currentL*M` where * = 1, 3, 6
        `med_listPrice_currentL*M` where * = 1, 3, 6
        `count_soldPrice_currentL*M` where * = 1, 3, 6
        `count_listPrice_currentL*M` where * = 1, 3, 6
    """
    df['avg_soldPrice_difference_1M_3M'] = df['avg_soldPrice_currentL1M'] - df['avg_soldPrice_currentL3M']
    df['avg_soldPrice_difference_3M_6M'] = df['avg_soldPrice_currentL3M'] - df['avg_soldPrice_currentL6M']
    df['avg_listPrice_difference_1M_3M'] = df['avg_listPrice_currentL1M'] - df['avg_listPrice_currentL3M']
    df['avg_listPrice_difference_3M_6M'] = df['avg_listPrice_currentL3M'] - df['avg_listPrice_currentL6M']

    df['med_soldPrice_difference_1M_3M'] = df['med_soldPrice_currentL1M'] - df['med_soldPrice_currentL3M']
    df['med_soldPrice_difference_3M_6M'] = df['med_soldPrice_currentL3M'] - df['med_soldPrice_currentL6M']
    df['med_listPrice_difference_1M_3M'] = df['med_listPrice_currentL1M'] - df['med_listPrice_currentL3M']
    df['med_listPrice_difference_3M_6M'] = df['med_listPrice_currentL3M'] - df['med_listPrice_currentL6M']

    df['count_soldPrice_difference_1M_3M'] = df['count_soldPrice_currentL1M'] - df['count_soldPrice_currentL3M']
    df['count_soldPrice_difference_3M_6M'] = df['count_soldPrice_currentL3M'] - df['count_soldPrice_currentL6M']
    df['count_listPrice_difference_1M_3M'] = df['count_listPrice_currentL1M'] - df['count_listPrice_currentL3M']
    df['count_listPrice_difference_3M_6M'] = df['count_listPrice_currentL3M'] - df['count_listPrice_currentL6M']

    return df

def create_ratio_bymonth_columns(df):
    """
    Assumes the existence of columns: 
        `avg_soldPrice_currentL*M` where * = 1, 3, 6
        `avg_listPrice_currentL*M` where * = 1, 3, 6
        `med_soldPrice_currentL*M` where * = 1, 3, 6
        `med_listPrice_currentL*M` where * = 1, 3, 6
        `count_soldPrice_currentL*M` where * = 1, 3, 6
        `count_listPrice_currentL*M` where * = 1, 3, 6
    """
    df['avg_soldPrice_ratio_1M_3M'] = df['avg_soldPrice_currentL1M'].div(df['avg_soldPrice_currentL3M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['avg_soldPrice_ratio_3M_6M'] = df['avg_soldPrice_currentL3M'].div(df['avg_soldPrice_currentL6M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['avg_listPrice_ratio_1M_3M'] = df['avg_listPrice_currentL1M'].div(df['avg_soldPrice_currentL3M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['avg_listPrice_ratio_3M_6M'] = df['avg_listPrice_currentL3M'].div(df['avg_soldPrice_currentL6M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)

    df['med_soldPrice_ratio_1M_3M'] = df['med_soldPrice_currentL1M'].div(df['med_soldPrice_currentL3M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['med_soldPrice_ratio_3M_6M'] = df['med_soldPrice_currentL3M'].div(df['med_soldPrice_currentL6M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['med_listPrice_ratio_1M_3M'] = df['med_listPrice_currentL1M'].div(df['med_listPrice_currentL3M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['med_listPrice_ratio_3M_6M'] = df['med_listPrice_currentL3M'].div(df['med_listPrice_currentL6M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)

    df['count_soldPrice_ratio_1M_3M'] = df['count_soldPrice_currentL1M'].div(df['count_soldPrice_currentL3M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['count_soldPrice_ratio_3M_6M'] = df['count_soldPrice_currentL3M'].div(df['count_soldPrice_currentL6M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['count_listPrice_ratio_1M_3M'] = df['count_listPrice_currentL1M'].div(df['count_listPrice_currentL3M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)
    df['count_listPrice_ratio_3M_6M'] = df['count_listPrice_currentL3M'].div(df['count_listPrice_currentL6M'], fill_value = np.NaN).replace([np.inf, -np.inf], np.nan)

    return df