import pandas as pd

def helper_construct_neighbourhood_key_column(df):
    """
    Assumes the existence of the keys `numBedroom`, `type`, `neighborhood`, `listDate` in the dataframe.
    """
    
    year_month_series = df['listDate'].astype(str).apply(lambda x: x.split('T')[0][:7])
    bedrooms_series = df['numBedroom'].astype(str, errors = "ignore").apply(lambda x: x[:1])
    df["neighborhood_key"] = bedrooms_series + "_" + df['type'].apply(lambda x: x.lower()) + "_" + df['neighborhood'] + "_" + year_month_series
    
    return df


def merge_neighborhood_previous_columns(df, other_df, key = "neighborhood_key", drop_key_after = True):
    """
    Assumes the existence of a column named `neighborhood_key` in the dataframe, containing values of formatting
        `numBedroom_type_neighborhood_year-month`.
        eg: "1_sale_Waterfront Comm unities C1_2022-01"
    
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