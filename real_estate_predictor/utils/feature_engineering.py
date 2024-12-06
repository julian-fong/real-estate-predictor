import datetime as dt 
import requests
import pandas as pd
import os
import re
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

#Feature Engineering

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

def helper_calculate_num_ammenities(x):
    count = 0
    try:
        for item in x:
            if item != '':
                count += 1
        return count
    except:
        return np.nan

def calculate_sqft(df):
    df['sqft_avg'] = df['sqft_range'].apply(helper_calculate_avg_sqft)

    return df

def calculate_ppsqft(df):
    df['ppsqft'] = df['listPrice']/df['sqft_avg']
    return df

def calculate_bbratio(df):
    #bed bath ratio
    #change dtype of numBed and numBath to numeric while filling errors with np.NaN, divide them and fill any errors with NaN
    df['numBedrooms'] = pd.to_numeric(df['numBedrooms'], errors = 'coerce')
    df['numBathrooms'] = pd.to_numeric(df['numBathrooms'], errors = 'coerce')
    df['bedbathRatio'] = pd.to_numeric(df['numBedrooms'], errors = 'coerce').div(pd.to_numeric(df['numBathrooms'], errors = 'coerce'), fill_value=np.NaN)
    return df

def calculate_ammenities(df):
    #numAmmenities
    df['numAmmenities'] = df['ammenities'].apply(helper_calculate_num_ammenities)
    df['numCondoAmmenities'] = df['condo_ammenities'].apply(helper_calculate_num_ammenities)

    return df

def calculate_dom(df):
    #calculate DOM
    df['soldDate_pd'] = pd.to_datetime(df['soldDate'])
    df['listDate_pd'] = pd.to_datetime(df['listDate'])
    df['daysOnMarket'] = df['soldDate_pd'] - df['listDate_pd']
    df['daysOnMarket'] = df['daysOnMarket'] / np.timedelta64(1, 'D')

    return df

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