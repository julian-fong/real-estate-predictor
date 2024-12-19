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

## Ammenities / Condo Ammenities Global Variables

AMMENITIES = ['School',
                'Library',
                'Public Transit',
                'Clear View',
                'Park',
                'Golf',
                'Arts Centre',
                'Hospital',
                'Place Of Worship',
                'Cul De Sac',
                'Ravine',
                'Fenced Yard',
                'Rec Centre',
                'School Bus Route',
                'Beach',
                'Lake/Pond',
                'Skiing',
                'Waterfront',
                'Electric Car Charger',
                'Grnbelt/Conserv',
                'Campground',
                'Marina',
                'Other',
                'Level',
                'Wooded/Treed',
                'Lake Access',
                'River/Stream',
                'Island',
                'Terraced',
                'Rolling',
                'Part Cleared',
                'Sloping',
                'Lake Backlot',
                'Tiled',
                'Lake/Pond/River',
                'Tiled/Drainage',
                'Electric Car Charg'
            ]


CONDO_AMMENITIES = ['Concierge',
        'Exercise Room',
        'Party/Meeting Room',
        'Recreation Room',
        'Rooftop Deck/Garden',
        'Visitor Parking',
        'Gym',
        'Security Guard',
        'Guest Suites',
        'Outdoor Pool',
        'Sauna',
        'Security System',
        'Car Wash',
        'Games Room',
        'Indoor Pool',
        'Media Room',
        'Bbqs Allowed',
        'Bus Ctr (Wifi Bldg)',
        'Bike Storage',
        'Tennis Court',
        'Squash/Racquet Court',
        'Lap Pool',
        'Satellite Dish'
        ]

## Sqft

def helper_calculate_avg_sqft(x):
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
    
def calculate_sqft(df):
    df['sqft_avg'] = df['sqft'].apply(helper_calculate_avg_sqft)

    return df

def calculate_ppsqft(df):
    df['ppsqft'] = df['listPrice']/df['sqft_avg']
    return df

## Bed and Bath

def calculate_bbratio(df):
    #bed bath ratio
    #change dtype of numBed and numBath to numeric while filling errors with np.NaN, divide them and fill any errors with NaN
    df['numBedrooms'] = pd.to_numeric(df['numBedrooms'], errors = 'coerce')
    df['numBathrooms'] = pd.to_numeric(df['numBathrooms'], errors = 'coerce')
    df['bedbathRatio'] = pd.to_numeric(df['numBedrooms'], errors = 'coerce').div(pd.to_numeric(df['numBathrooms'], errors = 'coerce'), fill_value=np.NaN)
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
    if column !=  "condo_ammenities" or column != "ammenities":
        raise ValueError("incorrect column name, expected condo_ammenities or ammenities")

    for i in range(len(df)):
        if df[column].values[i]:
            for item in df[column].values[i]:
                if item not in ammenities:
                    ammenities.append(item)

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
    """
    if not list_of_ammenities:
        list_of_ammenities = AMMENITIES
        
    for ammenity in list_of_ammenities:
        df[f"has{ammenity.capitalize()}"] = df[ammenities].apply(helper_find_existence_of_ammenity, args = (ammenity,))
    

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
    #numAmmenities
    df['numAmmenities'] = df['ammenities'].apply(helper_calculate_num_ammenities)
    df['numCondoAmmenities'] = df['condo_ammenities'].apply(helper_calculate_num_ammenities)

    return df

## Postal Code

def calculate_split_postalcode(df, int = 2):
    """
    Assumes existance of a column named postal_code
    """
    if int != 3 or int != 2:
        raise ValueError("invalid int passed, can only be 2 or 3")
    
    df[f"postal_code_split_{int}"] = df["zip"].apply(lambda x: x[:int])

## Datetime

def calculate_dom(df):
    #calculate DOM
    df['soldDate_pd'] = pd.to_datetime(df['soldDate'])
    df['listDate_pd'] = pd.to_datetime(df['listDate'])
    df['daysOnMarket'] = df['soldDate_pd'] - df['listDate_pd']
    df['daysOnMarket'] = df['daysOnMarket'] / np.timedelta64(1, 'D')

    return df

## Outliers

def removeOutliers(df):
    for col in df.select_dtypes(include=['number']).columns:
        if col == "soldPrice" or col == "listPrice":
            percentile90 = df[col].quantile(0.95)
            percentile10 = df[col].quantile(0.05)

            iqr = percentile90 - percentile10

            upper_limit = percentile90 + 3 * iqr
            lower_limit = percentile10 - 3 * iqr

            if lower_limit != upper_limit:
                df = df[df[col] < upper_limit]
                df = df[df[col] > lower_limit]

            print(col, len(df))

    return df

## neighbourhoods dataset

def calculate_previous_month_ppsqft(df):
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

def calculate_difference_bymonth(df):
    df['avg_soldPrice_difference_13M'] = df['avg_soldPrice_currentL1M'] - df['avg_soldPrice_currentL3M']
    df['avg_soldPrice_difference_36M'] = df['avg_soldPrice_currentL3M'] - df['avg_soldPrice_currentL6M']
    df['avg_listPrice_difference_13M'] = df['avg_soldPrice_currentL1M'] - df['avg_soldPrice_currentL3M']
    df['avg_listPrice_difference_36M'] = df['avg_soldPrice_currentL3M'] - df['avg_soldPrice_currentL6M']

    df['med_soldPrice_difference_13M'] = df['med_soldPrice_currentL1M'] - df['med_soldPrice_currentL3M']
    df['med_soldPrice_difference_36M'] = df['med_soldPrice_currentL3M'] - df['med_soldPrice_currentL6M']
    df['med_listPrice_difference_13M'] = df['med_soldPrice_currentL1M'] - df['med_soldPrice_currentL3M']
    df['med_listPrice_difference_36M'] = df['med_soldPrice_currentL3M'] - df['med_soldPrice_currentL6M']

    df['count_soldPrice_difference_13M'] = df['count_soldPrice_currentL1M'] - df['count_soldPrice_currentL3M']
    df['count_soldPrice_difference_36M'] = df['count_soldPrice_currentL3M'] - df['count_soldPrice_currentL6M']
    df['count_listPrice_difference_13M'] = df['count_soldPrice_currentL1M'] - df['count_soldPrice_currentL3M']
    df['count_listPrice_difference_36M'] = df['count_soldPrice_currentL3M'] - df['count_soldPrice_currentL6M']

    return df

def calculate_ratio_bymonth(df):
    df['avg_soldPrice_ratio_13M'] = df['avg_soldPrice_currentL1M'].div(df['avg_soldPrice_currentL3M'], fill_value = np.NaN)
    df['avg_soldPrice_ratio_36M'] = df['avg_soldPrice_currentL3M'].div(df['avg_soldPrice_currentL6M'], fill_value = np.NaN)
    df['avg_listPrice_ratio_13M'] = df['avg_soldPrice_currentL1M'].div(df['avg_soldPrice_currentL3M'], fill_value = np.NaN)
    df['avg_listPrice_ratio_36M'] = df['avg_soldPrice_currentL3M'].div(df['avg_soldPrice_currentL6M'], fill_value = np.NaN)

    df['med_soldPrice_ratio_13M'] = df['med_soldPrice_currentL1M'].div(df['med_soldPrice_currentL3M'], fill_value = np.NaN)
    df['med_soldPrice_ratio_36M'] = df['med_soldPrice_currentL3M'].div(df['med_soldPrice_currentL6M'], fill_value = np.NaN)
    df['med_listPrice_ratio_13M'] = df['med_soldPrice_currentL1M'].div(df['med_soldPrice_currentL3M'], fill_value = np.NaN)
    df['med_listPrice_ratio_36M'] = df['med_soldPrice_currentL3M'].div(df['med_soldPrice_currentL6M'], fill_value = np.NaN)

    df['count_soldPrice_ratio_13M'] = df['count_soldPrice_currentL1M'].div(df['count_soldPrice_currentL3M'], fill_value = np.NaN)
    df['count_soldPrice_ratio_36M'] = df['count_soldPrice_currentL3M'].div(df['count_soldPrice_currentL6M'], fill_value = np.NaN)
    df['count_listPrice_ratio_13M'] = df['count_soldPrice_currentL1M'].div(df['count_soldPrice_currentL3M'], fill_value = np.NaN)
    df['count_listPrice_ratio_36M'] = df['count_soldPrice_currentL3M'].div(df['count_soldPrice_currentL6M'], fill_value = np.NaN)

    return df