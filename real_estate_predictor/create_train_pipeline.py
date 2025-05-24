import pathlib
from copy import deepcopy

import pandas as pd
import yaml

from real_estate_predictor.models.model import XGBoostRegressor
from real_estate_predictor.processing.processor import (
    DataCleaner,
    FeatureEngineering,
    Processor,
)
from real_estate_predictor.utils.extract_dataset import extract_raw_data_listings
from real_estate_predictor.utils.merge_datasets import (
    merge_neighborhood_previous_columns,
)
from real_estate_predictor.utils.pandas import pandas_read_filepath


def run_datacleaner(df: pd.DataFrame, config: dict, save=False) -> pd.DataFrame:
    """
    Cleans the dataset based on the provided configuration.

    Args:
        df (pd.DataFrame): The DataFrame to be cleaned
        config (dict): Configuration dictionary containing DataCleaner parameters
        save (bool): Whether to save the cleaned DataFrame to a file.
        Defaults to False.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    cleaner = DataCleaner(df)
    for key in config.keys():
        assert key in dir(
            cleaner
        ), f"Function Name {key} not found in DataCleaner class"

    for key, value in config.items():
        f = getattr(cleaner, key)
        df = f(**value)

    if save:
        cleaner.save()

    return df


def run_feature_engineering(df: pd.DataFrame, config: dict, save=False) -> pd.DataFrame:
    """
    Performs feature engineering on the dataset based on the provided configuration.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        config (dict): Configuration dictionary containing FeatureEngineer parameters.
        save (bool): Whether to save the processed DataFrame to a file.
        Defaults to False.

    Returns:
        pd.DataFrame: The processed DataFrame with new features.
    """
    feature_engineer = FeatureEngineering(df)
    for key in config.keys():
        assert key in dir(
            feature_engineer
        ), f"Function Name {key} not found in FeatureEngineer class"

    for key, value in config.items():
        f = getattr(feature_engineer, key)
        f(**value)
    if save:
        feature_engineer.save()

    return df


def run_preprocessor(df: pd.DataFrame, config: dict, save=False) -> tuple:
    """
    Preprocesses the dataset based on the provided configuration.

    Args:
        df (pd.DataFrame): The DataFrame to be preprocessed.
        config (dict): Configuration dictionary containing Preprocessor parameters.
        save (bool): Whether to save the processed DataFrame to a file.
        Defaults to False.

    Returns:
        tuple: Tuple containing the preprocessed features and target variables.
    """
    preprocessor = Processor(df)
    _config = deepcopy(config)
    assert "target" in _config.keys(), "The config file must contain a 'target' key."
    target = _config["target"]
    _config.pop("target")
    for key in _config.keys():
        assert key in dir(
            preprocessor
        ), f"Function Name {key} not found in Preprocessor class"

    assert (
        target in df.columns
    ), f"The target column '{target}' is not present in the DataFrame."

    if "train_test_split_df" in _config.keys():
        _config.pop("train_test_split_df")
    if "drop_columns" in _config.keys():
        preprocessor.drop_columns(**_config["drop_columns"])
        _config.pop("drop_columns")

    X, y = preprocessor.train_test_split_df(target)
    if "train_test_split" in _config.keys():
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(
            X, y, **_config["train_test_split"]
        )
        _config.pop("train_test_split")
    else:
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

    for key, value in _config.items():
        f = getattr(preprocessor, key)
        f(**value)

    preprocessor.apply_transformer()
    preprocessor.fit(X_train)

    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    if save:
        preprocessor.save()

    return X_train, y_train, X_test, y_test


def run_train_pipeline(config=None, save=False):
    if not config:
        config = str(pathlib.Path.cwd().joinpath("config", "sample_lease_config.yaml"))

    with open(config, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    raw_df = pandas_read_filepath(data["listings_filepath"])
    df = extract_raw_data_listings(raw_df)

    if "type" not in data.keys():
        raise ValueError("The config file must contain a 'type' key.")

    listing_type = data["type"]
    df = df[df["type"] == listing_type]

    if "neighbourhoods_filepath" in data.keys():
        neighbourhoods_df = pandas_read_filepath(data["neighbourhoods_filepath"])
        # merge the listings with the neighbourhoods
        df = merge_neighborhood_previous_columns(df, neighbourhoods_df)

    # check and clean the data
    if "DataCleaner" in data.keys():
        df = run_datacleaner(df, data["DataCleaner"], save=save)

    # check and feature engineer the data
    if "FeatureEngineering" in data.keys():
        df = run_feature_engineering(df, data["FeatureEngineering"], save=save)

    # check for a Preprocessor
    if not data["Processor"]:
        raise ValueError("The config file must contain a 'Processor' key.")
    else:
        X_train, y_train, _, _ = run_preprocessor(df, data["Processor"], save=save)

    # load and training the model

    if "model_type" not in data.keys():
        raise ValueError("The config file must contain a 'model_type' key.")

    model_type = data["model_type"]
    if model_type not in ["XGBoostRegressor", "XGBRegressor"]:
        raise ValueError(f"Model type '{model_type}' is not supported.")

    if "param_grid" in data.keys():
        param_grid = data["param_grid"]
    else:
        param_grid = None

    # initialize the model
    model = XGBoostRegressor(param_grid=param_grid)

    if "model" in data.keys():
        params = data["model"]
        model.set_model_params(**params)

    model.fit(X_train, y_train)

    if save:
        model.save_model(override=True)


if __name__ == "__main__":
    pipeline_config = str(
        pathlib.Path.cwd().joinpath("config", "sample_lease_config.yaml")
    )
    run_train_pipeline(pipeline_config, save=True)
