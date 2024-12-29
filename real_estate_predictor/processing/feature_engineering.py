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
    df['sqft_avg'] = df['sqft'].apply(helper_create_avg_sqft)

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
    df['bedbathRatio'] = df['numBedrooms'].div(df['numBathrooms'])
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

def create_ammenities_flag_columns(df, ammenities, list_of_ammenities = None):
    """
    Creates new flag columns based on a passed `list_of_ammenities`.
    eg. suppose `list_of_ammnenities` = ["Park", "School"]
    Then the dataframe will contain new columns `hasPark` and `hasSchool`
    
    ammenities: one of `ammenities` or `condo_ammenities`
    
    Correspondings to `has_ammenities_flags` and `has_condo_ammenities_flags` in the config.
    
    Assumes the columns `ammenities` and `condo_ammenities` are already present in the dataframe.
    """
    
    list_of_ammenities = helper_get_unique_ammenities(df, ammenities)
        
    for ammenity in list_of_ammenities:
        df[f"has{ammenity.capitalize()}"] = df[ammenities].apply(helper_find_existence_of_ammenity, args = (ammenity,))
        
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
    df['daysOnMarket'] = df['soldDate'] - df['listDate']
    df['daysOnMarket'] = df['daysOnMarket'] / np.timedelta64(1, 'D')

    return df

## neighbourhoods dataset

def create_neighbourhood_key_column(df):
    """
    Assumes the existence of the keys `numBedroom`, `type`, `neighborhood`, `listDate` in the dataframe.
    """
    
    year_month_series = df['listDate'].astype(str).apply(lambda x: x.split('T')[0][:7])
    bedrooms_series = df['numBathrooms'].astype(str, errors = "ignore").apply(lambda x: x[:1])
    df["neighborhood_key"] = bedrooms_series + "_" + df['type'].apply(lambda x: x.lower()) + "_" + df['neighborhood'] + "_" + year_month_series
    
    return df


def create_previous_month_columns(df, other_df, key = "neighborhood_key", drop_key_after = True):
    """
    Assumes the existence of a column named `neighborhood_key` in the dataframe, containing values of formatting
        `numBedroom_type_neighborhood_year-month`.
        eg: "1_sale_Waterfront Communities C1_2022-01"
    
    Parameters
    ----------
    
    df : pd.DataFrame
        The dataframe to which the columns will be added.
    other_df : pd.DataFrame
        The dataframe from which the columns will be taken.
    key : str
        The key on which to join the two dataframes.
    drop_key_after : bool
        If True, the key column will be dropped from the resulting dataframe.
    """
    
    df = df.merge(other_df, on = key, how = "left")
    
    if drop_key_after:
        df = df.drop(columns = [key], axis = 1)
    
    return df

def create_previous_month_ppsqft(df):
    """
    Assumes the existence of columns: 
        `avg_soldPrice_currentL*M` where * = 1, 3, 6
        `med_soldPrice_currentL*M` where * = 1, 3, 6
        `avg_listPrice_currentL*M` where * = 1, 3, 6
        `med_listPrice_currentL*M` where * = 1, 3, 6 
        `sqft_avg` in the dataframe
    """
    df['avg_soldPrice_ppsqft_currentL1M'] = df['avg_soldPrice_currentL1M']/df['sqft_avg']
    df['med_soldPrice_ppsqft_currentL1M'] = df['med_soldPrice_currentL1M']/df['sqft_avg']
    df['avg_listPrice_ppsqft_currentL1M'] = df['avg_listPrice_currentL1M']/df['sqft_avg']
    df['med_listPrice_ppsqft_currentL1M'] = df['med_listPrice_currentL1M']/df['sqft_avg']

    df['avg_soldPrice_ppsqft_currentL3M'] = df['avg_soldPrice_currentL3M']/df['sqft_avg']
    df['med_soldPrice_ppsqft_currentL3M'] = df['med_soldPrice_currentL3M']/df['sqft_avg']
    df['avg_listPrice_ppsqft_currentL3M'] = df['avg_listPrice_currentL3M']/df['sqft_avg']
    df['med_listPrice_ppsqft_currentL3M'] = df['med_listPrice_currentL3M']/df['sqft_avg']

    df['avg_soldPrice_ppsqft_currentL6M'] = df['avg_soldPrice_currentL6M']/df['sqft_avg']
    df['med_soldPrice_ppsqft_currentL6M'] = df['avg_soldPrice_currentL6M']/df['sqft_avg']
    df['avg_listPrice_ppsqft_currentL6M'] = df['avg_listPrice_currentL6M']/df['sqft_avg']
    df['med_listPrice_ppsqft_currentL6M'] = df['avg_listPrice_currentL6M']/df['sqft_avg']

    return df

def create_difference_bymonth(df):
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

def create_ratio_bymonth(df):
    """
    Assumes the existence of columns: 
        `avg_soldPrice_currentL*M` where * = 1, 3, 6
        `avg_listPrice_currentL*M` where * = 1, 3, 6
        `med_soldPrice_currentL*M` where * = 1, 3, 6
        `med_listPrice_currentL*M` where * = 1, 3, 6
        `count_soldPrice_currentL*M` where * = 1, 3, 6
        `count_listPrice_currentL*M` where * = 1, 3, 6
    """
    df['avg_soldPrice_ratio_1M_3M'] = df['avg_soldPrice_currentL1M'].div(df['avg_soldPrice_currentL3M'], fill_value = np.NaN)
    df['avg_soldPrice_ratio_3M_6M'] = df['avg_soldPrice_currentL3M'].div(df['avg_soldPrice_currentL6M'], fill_value = np.NaN)
    df['avg_listPrice_ratio_1M_3M'] = df['avg_listPrice_currentL1M'].div(df['avg_soldPrice_currentL3M'], fill_value = np.NaN)
    df['avg_listPrice_ratio_3M_6M'] = df['avg_listPrice_currentL3M'].div(df['avg_soldPrice_currentL6M'], fill_value = np.NaN)

    df['med_soldPrice_ratio_1M_3M'] = df['med_soldPrice_currentL1M'].div(df['med_soldPrice_currentL3M'], fill_value = np.NaN)
    df['med_soldPrice_ratio_3M_6M'] = df['med_soldPrice_currentL3M'].div(df['med_soldPrice_currentL6M'], fill_value = np.NaN)
    df['med_listPrice_ratio_1M_3M'] = df['med_listPrice_currentL1M'].div(df['med_listPrice_currentL3M'], fill_value = np.NaN)
    df['med_listPrice_ratio_3M_6M'] = df['med_listPrice_currentL3M'].div(df['med_listPrice_currentL6M'], fill_value = np.NaN)

    df['count_soldPrice_ratio_1M_3M'] = df['count_soldPrice_currentL1M'].div(df['count_soldPrice_currentL3M'], fill_value = np.NaN)
    df['count_soldPrice_ratio_3M_6M'] = df['count_soldPrice_currentL3M'].div(df['count_soldPrice_currentL6M'], fill_value = np.NaN)
    df['count_listPrice_ratio_1M_3M'] = df['count_listPrice_currentL1M'].div(df['count_listPrice_currentL3M'], fill_value = np.NaN)
    df['count_listPrice_ratio_3M_6M'] = df['count_listPrice_currentL3M'].div(df['count_listPrice_currentL6M'], fill_value = np.NaN)

    return df