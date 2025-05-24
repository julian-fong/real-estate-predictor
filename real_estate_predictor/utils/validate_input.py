import os
import pickle

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier, XGBRegressor

key = os.environ["REPLIERS_KEY"]


from real_estate_predictor.config import (DATACLEANER_FILE,
                                          FEATURE_ENGINEERING_FILE,
                                          PREPROCESSOR_FILE)
from real_estate_predictor.processing.processor import (DataCleaner,
                                                        FeatureEngineering,
                                                        Processor)
from real_estate_predictor.utils.extract_dataset import (
    extract_raw_data_listings, subtract_months)

processor_path = PREPROCESSOR_FILE
processor = Processor.load(processor_path)

datacleaner_path = DATACLEANER_FILE
cleaner = DataCleaner.load(datacleaner_path)

feature_path = FEATURE_ENGINEERING_FILE
feature = FeatureEngineering.load(feature_path)


def transform_listings_input(data):
    """
    Assumes data is of type dict containing a single observation, will
    obtain the listings data
    """
    # Transform the input data into the format required by the model

    input_df = pd.DataFrame.from_dict(data, orient="index").T
    input_df = extract_raw_data_listings(input_df)
    return input_df


def transform_neighborhood_input(data):
    """
    Assumes data is of type dict containing a single observation, will
    obtain the neighbourhood data
    """

    # obtain the neighborhood parameters
    end_date = data["listDate"].split("T")[0]
    end_date_ym = end_date[:7]
    start_date = subtract_months(end_date_ym, 6) + "-01"
    numBedroom = data["details"]["numBedrooms"]

    listing_type = data["type"]
    if listing_type.lower() == "lease":
        lastStatus = "Lsd"
    else:
        lastStatus = "Sld"

    neighbourhood = data["address"]["neighborhood"]

    # obtain the relative months from the listing date
    month_1_key = subtract_months(end_date_ym, 1)
    month_3_key = subtract_months(end_date_ym, 3)
    month_6_key = subtract_months(end_date_ym, 6)

    url = (
        f"https://api.repliers.io/listings?minSoldDate={start_date}&maxSoldDate={end_date}"
        f"&minBeds={numBedroom}&maxBeds={numBedroom}"
        f"&minListDate={start_date}&maxListDate={end_date}&type={listing_type}"
        f"&neighborhood={neighbourhood}&lastStatus={lastStatus}"
    )

    NEIGHBOURHOOD_PARAMETERS = {
        "listings": "false",
        "status": "U",
        "statistics": "grp-mth,avg-listPrice,"
        "avg-soldPrice,cnt-available,cnt-closed,"
        "med-daysOnMarket,avg-daysOnMarket,med-soldPrice,med-listPrice",
    }

    headers = {"repliers-api-key": key}
    r = requests.request(
        "GET", url, params=NEIGHBOURHOOD_PARAMETERS, headers=headers, timeout=10
    )
    n_data = r.json()

    mapping = {
        "avg_soldPrice_current": np.nan,
        "count_soldPrice_current": np.nan,
        "med_soldPrice_current": np.nan,
        "avg_listPrice_current": np.nan,
        "count_listPrice_current": np.nan,
        "med_listPrice_current": np.nan,
        "count_available_current": np.nan,
        "avg_daysOnMarket_current": np.nan,
        "count_daysOnMarket_current": np.nan,
        "med_daysOnMarket_current": np.nan,
        "avg_soldPrice_currentL1M": np.nan,
        "count_soldPrice_currentL1M": np.nan,
        "med_soldPrice_currentL1M": np.nan,
        "avg_listPrice_currentL1M": np.nan,
        "count_listPrice_currentL1M": np.nan,
        "med_listPrice_currentL1M": np.nan,
        "count_available_currentL1M": np.nan,
        "avg_daysOnMarket_currentL1M": np.nan,
        "count_daysOnMarket_currentL1M": np.nan,
        "med_daysOnMarket_currentL1M": np.nan,
        "avg_soldPrice_currentL3M": np.nan,
        "count_soldPrice_currentL3M": np.nan,
        "med_soldPrice_currentL3M": np.nan,
        "avg_listPrice_currentL3M": np.nan,
        "count_listPrice_currentL3M": np.nan,
        "med_listPrice_currentL3M": np.nan,
        "count_available_currentL3M": np.nan,
        "avg_daysOnMarket_currentL3M": np.nan,
        "count_daysOnMarket_currentL3M": np.nan,
        "med_daysOnMarket_currentL3M": np.nan,
        "avg_soldPrice_currentL6M": np.nan,
        "count_soldPrice_currentL6M": np.nan,
        "med_soldPrice_currentL6M": np.nan,
        "avg_listPrice_currentL6M": np.nan,
        "count_listPrice_currentL6M": np.nan,
        "med_listPrice_currentL6M": np.nan,
        "count_available_currentL6M": np.nan,
        "avg_daysOnMarket_currentL6M": np.nan,
        "count_daysOnMarket_currentL6M": np.nan,
        "med_daysOnMarket_currentL6M": np.nan,
    }

    soldPrice_mth_stats = n_data["statistics"]["soldPrice"][
        "mth"
    ]  # gives avg, count and med
    listPrice_mth_stats = n_data["statistics"]["listPrice"][
        "mth"
    ]  # gives avg, count and med
    available_mth_stats = n_data["statistics"]["available"]["mth"]  # gives count
    daysOnMarket_mth_stats = n_data["statistics"]["daysOnMarket"][
        "mth"
    ]  # gives avg, count and med

    if end_date_ym in soldPrice_mth_stats:
        mapping["avg_soldPrice_current"] = soldPrice_mth_stats[end_date_ym]["avg"]
        mapping["count_soldPrice_current"] = soldPrice_mth_stats[end_date_ym]["count"]
        mapping["med_soldPrice_current"] = soldPrice_mth_stats[end_date_ym]["med"]

    if end_date_ym in listPrice_mth_stats:
        mapping["avg_listPrice_current"] = listPrice_mth_stats[end_date_ym]["avg"]
        mapping["count_listPrice_current"] = listPrice_mth_stats[end_date_ym]["count"]
        mapping["med_listPrice_current"] = listPrice_mth_stats[end_date_ym]["med"]

    if end_date_ym in available_mth_stats:
        mapping["count_available_current"] = available_mth_stats[end_date_ym]

    if end_date_ym in daysOnMarket_mth_stats:
        mapping["avg_daysOnMarket_current"] = daysOnMarket_mth_stats[end_date_ym]["avg"]
        mapping["count_daysOnMarket_current"] = daysOnMarket_mth_stats[end_date_ym][
            "count"
        ]
        mapping["med_daysOnMarket_current"] = daysOnMarket_mth_stats[end_date_ym]["med"]

    if month_1_key in soldPrice_mth_stats:
        mapping["avg_soldPrice_currentL1M"] = soldPrice_mth_stats[month_1_key]["avg"]
        mapping["count_soldPrice_currentL1M"] = soldPrice_mth_stats[month_1_key][
            "count"
        ]
        mapping["med_soldPrice_currentL1M"] = soldPrice_mth_stats[month_1_key]["med"]

    if month_1_key in listPrice_mth_stats:
        mapping["avg_listPrice_currentL1M"] = listPrice_mth_stats[month_1_key]["avg"]
        mapping["count_listPrice_currentL1M"] = listPrice_mth_stats[month_1_key][
            "count"
        ]
        mapping["med_listPrice_currentL1M"] = listPrice_mth_stats[month_1_key]["med"]

    if month_1_key in available_mth_stats:
        mapping["count_available_currentL1M"] = available_mth_stats[month_1_key]

    if month_1_key in daysOnMarket_mth_stats:
        mapping["avg_daysOnMarket_currentL1M"] = daysOnMarket_mth_stats[month_1_key][
            "avg"
        ]
        mapping["count_daysOnMarket_currentL1M"] = daysOnMarket_mth_stats[month_1_key][
            "count"
        ]
        mapping["med_daysOnMarket_currentL1M"] = daysOnMarket_mth_stats[month_1_key][
            "med"
        ]

    if month_3_key in soldPrice_mth_stats:
        mapping["avg_soldPrice_currentL3M"] = soldPrice_mth_stats[month_3_key]["avg"]
        mapping["count_soldPrice_currentL3M"] = soldPrice_mth_stats[month_3_key][
            "count"
        ]
        mapping["med_soldPrice_currentL3M"] = soldPrice_mth_stats[month_3_key]["med"]

    if month_3_key in listPrice_mth_stats:
        mapping["avg_listPrice_currentL3M"] = listPrice_mth_stats[month_3_key]["avg"]
        mapping["count_listPrice_currentL3M"] = listPrice_mth_stats[month_3_key][
            "count"
        ]
        mapping["med_listPrice_currentL3M"] = listPrice_mth_stats[month_3_key]["med"]

    if month_3_key in available_mth_stats:
        mapping["count_available_currentL3M"] = available_mth_stats[month_3_key]

    if month_3_key in daysOnMarket_mth_stats:
        mapping["avg_daysOnMarket_currentL3M"] = daysOnMarket_mth_stats[month_3_key][
            "avg"
        ]
        mapping["count_daysOnMarket_currentL3M"] = daysOnMarket_mth_stats[month_3_key][
            "count"
        ]
        mapping["med_daysOnMarket_currentL3M"] = daysOnMarket_mth_stats[month_3_key][
            "med"
        ]

    if month_6_key in soldPrice_mth_stats:
        mapping["avg_soldPrice_currentL6M"] = soldPrice_mth_stats[month_6_key]["avg"]
        mapping["count_soldPrice_currentL6M"] = soldPrice_mth_stats[month_6_key][
            "count"
        ]
        mapping["med_soldPrice_currentL6M"] = soldPrice_mth_stats[month_6_key]["med"]

    if month_6_key in listPrice_mth_stats:
        mapping["avg_listPrice_currentL6M"] = listPrice_mth_stats[month_6_key]["avg"]
        mapping["count_listPrice_currentL6M"] = listPrice_mth_stats[month_6_key][
            "count"
        ]
        mapping["med_listPrice_currentL6M"] = listPrice_mth_stats[month_6_key]["med"]

    if month_6_key in available_mth_stats:
        mapping["count_available_currentL6M"] = available_mth_stats[month_6_key]

    if month_6_key in daysOnMarket_mth_stats:
        mapping["avg_daysOnMarket_currentL6M"] = daysOnMarket_mth_stats[month_6_key][
            "avg"
        ]
        mapping["count_daysOnMarket_currentL6M"] = daysOnMarket_mth_stats[month_6_key][
            "count"
        ]
        mapping["med_daysOnMarket_currentL6M"] = daysOnMarket_mth_stats[month_6_key][
            "med"
        ]

    input_stats = pd.DataFrame(mapping, index=[0])
    return input_stats


def transform_input(data):
    """
    Assumes data is of type dict containing a single observation, will
    obtain the listings and neighbourhood data
    """
    listings_data = transform_listings_input(data)
    neighbourhood_data = transform_neighborhood_input(data)

    return pd.concat([listings_data, neighbourhood_data], axis=1)


def process_input(data, model):
    """
    Assumes data is of type dict containing a single observation, will
    obtain the listings and neighbourhood data, and process the data
    to be passed into the model
    """
    df = transform_input(data)
    if isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
        model_cols = model.get_booster().feature_names
    else:
        model_cols = model._model.get_booster().feature_names

    # fix the dtype of soldDate if its not the correct datetime format since its NaT
    df = fix_nat_dtype(df)

    # set the dataframe for the feature module
    feature.set_df(df)

    feature.create_features_old()
    # correct the columns of the dataframe in case it does not match the processor columns
    df = fill_missing_columns(df, processor.feature_names_)
    df = remove_extra_columns(df, processor.feature_names_)
    df = processor.transform(df)
    # correct the columns of the dataframe in case it does not match the model columns
    df = remove_extra_columns(df, model_cols)
    df = fill_missing_columns(df, model_cols)
    df = reindex_columns(df, model_cols)
    return df


# helper functions for process_input


def fix_nat_dtype(df, column="soldDate"):
    df[column] = df[column].dt.tz_localize("UTC")
    return df


def fill_missing_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan  # Fill missing columns with a default value

    return df


def remove_extra_columns(df, columns):
    extra_cols = []
    for col in df.columns:
        if col not in columns:
            extra_cols.append(col)

    df.drop(columns=extra_cols, inplace=True)

    return df


def reindex_columns(df, columns):
    df = df[columns]
    return df
