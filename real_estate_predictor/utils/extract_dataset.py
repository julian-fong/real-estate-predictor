from datetime import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
import ast

from real_estate_predictor.config.config import LISTING_COLUMN_TO_DTYPE_MAPPING, LISTING_EXPECTED_COLUMNS


MISSING_VALUES = [
        '', ' ','nan', 'NaN', 'NA', 'na', 'N/A', 'n/a',
        'null', 'NULL', 'none', 'None', 'missing',
        'Missing', 'MISSING', 'unknown', 'Unknown',
        '?', '-', '--', '---', 'None', None, 'NAN'
    ]

def extract_raw_data_listings(raw_df: pd.DataFrame, inplace = True, verbose = False) -> pd.DataFrame:
    """Extracts relevant data from the raw dataframe.
    Parameters
    ----------
    
    df : pd.DataFrame
        Incoming raw dataframe containing listings from a specific
        period. Assumes the following columns exist
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
    if inplace:
        df = raw_df
    else:
        df = raw_df.copy(deep = True)
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

    #df['fees'] = condo_df['fees']
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
    
    if 'fees' in LISTING_EXPECTED_COLUMNS:
        LISTING_EXPECTED_COLUMNS.remove('fees')
    
    assert all([True for col in LISTING_EXPECTED_COLUMNS if col in df.columns]), "some not found in dataset that are expected"
    
    #remove any extra columns that we don't need
    df = df[LISTING_EXPECTED_COLUMNS]
    
    datetime_cols = [col for col in LISTING_COLUMN_TO_DTYPE_MAPPING.keys() if LISTING_COLUMN_TO_DTYPE_MAPPING[col] == np.datetime64]
    numerical_cols = [col for col in LISTING_COLUMN_TO_DTYPE_MAPPING.keys() if LISTING_COLUMN_TO_DTYPE_MAPPING[col] == float]
    list_cols = [col for col in LISTING_COLUMN_TO_DTYPE_MAPPING.keys() if LISTING_COLUMN_TO_DTYPE_MAPPING[col] == list]
    str_cols = [col for col in LISTING_COLUMN_TO_DTYPE_MAPPING.keys() if LISTING_COLUMN_TO_DTYPE_MAPPING[col] == str]
    int_cols = [col for col in LISTING_COLUMN_TO_DTYPE_MAPPING.keys() if LISTING_COLUMN_TO_DTYPE_MAPPING[col] == int]
    #dict_cols = [col for col in LISTING_COLUMN_TO_DTYPE_MAPPING.keys() if LISTING_COLUMN_TO_DTYPE_MAPPING[col] == dict]
    
    convert_col_dtype(df, datetime_cols, "datetime")
    convert_col_dtype(df, numerical_cols, "numeric")
    convert_col_dtype(df, list_cols, "list")
    convert_col_dtype(df, str_cols, "str")
    convert_col_dtype(df, int_cols, "int")
    
    if verbose:
        for col in df.columns:
            unique_types = set(type(value) for value in df[col].values)
            #print(f"Column '{col}' contains the following data types: {unique_types}")
            # Count occurrences of each type
            type_counts = df[col].apply(type).value_counts(normalize=True)
            
            # Print the unique types and their ratios
            print(f"Column '{col}' contains the following data types: {unique_types} and ratios:")
            for dtype, ratio in type_counts.items():
                print(f"  - {dtype.__name__}: {ratio:.2%}")
                
    #try to standardize the missing values in the dataframe
    df = df.replace(MISSING_VALUES, np.nan)
                
    return df

### Neighborhood dataset

def extract_neighborhood_data(data: dict, location, bed, type, verbose = False):
    
    soldPrice_mth_stats = data['statistics']['soldPrice']['mth'] #gives avg, count and med
    listPrice_mth_stats = data['statistics']['listPrice']['mth'] #gives avg, count and med
    available_mth_stats = data['statistics']['available']['mth'] #gives count 
    daysOnMarket_mth_stats = data['statistics']['daysOnMarket']['mth'] #gives avg, count and med
    closed_mth_stats = data['statistics']['closed']['mth']

    soldPrice_mth_stats_df = pd.DataFrame(soldPrice_mth_stats)
    listPrice_mth_stats_df = pd.DataFrame(listPrice_mth_stats)
    available_mth_stats_df = pd.DataFrame([available_mth_stats], index = ['count'])
    daysOnMarket_mth_stats_df = pd.DataFrame(daysOnMarket_mth_stats)
    closed_mth_stats_df = pd.DataFrame(closed_mth_stats)

    for index in soldPrice_mth_stats_df.index:
        soldPrice_mth_stats_df = soldPrice_mth_stats_df.rename(index = {index: f'{index}_{bed}_{type}_{location}'})
        listPrice_mth_stats_df = listPrice_mth_stats_df.rename(index = {index: f'{index}_{bed}_{type}_{location}'})
        daysOnMarket_mth_stats_df = daysOnMarket_mth_stats_df.rename(index = {index: f'{index}_{bed}_{type}_{location}_'})
        
    available_mth_stats_df = available_mth_stats_df.rename(index = {"count": f'count_{bed}_{type}_{location}'})

    try:
        soldPrice_mth_stats_df = extract_neighbourhood_df(soldPrice_mth_stats_df, "soldPrice")
        listPrice_mth_stats_df = extract_neighbourhood_df(listPrice_mth_stats_df, "listPrice")
        available_mth_stats_df = extract_neighbourhood_df(available_mth_stats_df, "available")
        daysOnMarket_mth_stats_df = extract_neighbourhood_df(daysOnMarket_mth_stats_df, "daysOnMarket")

        merged_inital_df = soldPrice_mth_stats_df.merge(listPrice_mth_stats_df, on='key').merge(available_mth_stats_df, on='key').merge(daysOnMarket_mth_stats_df, on='key')

        merged_inital_df['Date_1M'] = merged_inital_df['key'].apply(lambda x: subtract_months(x, 1))
        date_1m_df = merged_inital_df.merge(left_on = "Date_1M", right_on = "key", suffixes=[None, "L1M"], right = merged_inital_df, how = "left").drop(columns = ['keyL1M','Date_1ML1M','Date_1M'] + [col for col in merged_inital_df.columns if "current" in col])
        merged_inital_df['Date_3M'] = merged_inital_df['key'].apply(lambda x: subtract_months(x, 3))
        date_3m_df = merged_inital_df.merge(left_on = "Date_3M", right_on = "key", suffixes=[None, "L3M"], right = merged_inital_df, how = "left").drop(columns = ['keyL3M','Date_1ML3M', 'Date_3ML3M','Date_1M','Date_3M'] + [col for col in merged_inital_df.columns if "current" in col])
        merged_inital_df['Date_6M'] = merged_inital_df['key'].apply(lambda x: subtract_months(x, 6))
        date_6m_df = merged_inital_df.merge(left_on = "Date_6M", right_on = "key", suffixes=[None, "L6M"], right = merged_inital_df, how = "left").drop(columns = ['keyL6M','Date_1ML6M', 'Date_3ML6M', 'Date_6ML6M','Date_1M','Date_3M','Date_6M'] + [col for col in merged_inital_df.columns if "current" in col])

        merged_final_df = date_1m_df.merge(date_3m_df, on='key').merge(date_6m_df, on='key')
        
        return merged_final_df
    except:
        if verbose:
            print(f"unable to get data for {(location, bed, type)}")
        return None
    
    

def extract_neighbourhood_df(df, metric):
    df = df.rename_axis('key').reset_index()
    df = df.melt(id_vars = ["key"], var_name="Date",value_name="value")
    # Extract the metric from the 'key' column
    # print(df['key'].values[0])
    df['index'] = df['key'].str.split("_").str[1] +"_"+ df['key'].str.split("_").str[2] +"_"+ df['key'].str.split("_").str[3]
    df['new_index'] = df['index']+"_"+df['Date']
    df['metric'] = df['key'].str.split("_").str[0]
    metric_columns = ['key'] + [f'{agg}_{metric}_current' for agg in sorted(set(df['metric'].values))]
    df = df.drop(columns = ["key"])
    # Pivot the DataFrame
    df = df.pivot(index=['new_index'], columns=['metric'], values='value')

    # Reset the index to make 'Date' a column again
    df.reset_index(inplace=True)
    # Rename the columns for better clarity
    df.columns.name = None  # Remove the column name generated by pivot
    # df.columns = ['key',f'avg_value_{metric}', f'count_value_{metric}', f'med_value_{metric}']
    df.columns = metric_columns
    return df

def subtract_months(col, num_months = 1):
    date_str = col.split("_")[-1]
    
    # Convert the input date string to a datetime object
    date_obj = dt.strptime(date_str, '%Y-%m')

    # Subtract the specified number of months
    new_date_obj = date_obj - relativedelta(months=num_months)
    new_date_obj = new_date_obj.strftime('%Y-%m')
    new_date = col.replace(date_str, new_date_obj)
    # Format the result as "YYYY-MM" and return
    return new_date

## Helper functions

def convert_col_dtype(df: pd.DataFrame, columns: list, convert_to_type: str, errors: str = "coerce"):
    """
    Eligible values for convert_to_type are:
        numeric: pd.to_numerical
        datetime: pd.to_datetime
        str: series.as_type(str)
        list: ast.literal_eval
        dict: ast.literal_eval
    
    Parameters
    ----------
    
    df : pd.DataFrame
    
    columns : list
    
    errors : str
        how to handle errors in. Possible values
            'raise': If `raise`, then invalid parsing will raise an exception.
            'coerce': If `coerce`, then invalid parsing will be set as NaN or NaT
            'ignore': If `ignore`, then invalid parsing will return the input.
            
        Only applicable currently to numeric and datetime types
    
    """
    if not columns:
        #if no columns are provided, don't raise an error
        return
    
    if convert_to_type == "str":
        for col in columns:
            df[col] = df[col].astype("str")
    elif convert_to_type == "numeric":
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors = errors)
    elif convert_to_type == "int":
        for col in columns:
            #np.nan is not a valid value for Int32, will convert to pd.NA
            df[col] = pd.to_numeric(df[col], errors = errors).astype('Int32', errors = "ignore")
    elif convert_to_type == "datetime":
        for col in columns:
            df[col] = pd.to_datetime(df[col], errors = errors)      
    elif convert_to_type == "list":
        for col in columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x))
    elif convert_to_type == "dict":
        for col in columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x))
    else:
        raise ValueError(f"Unexpected convert_to_type {convert_to_type}"
                     " available types are ['numeric','datetime','str', 'list', 'dict', 'int']")