import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

from real_estate_predictor.processing.processor import Processor


# need this function to compare arrays with nans since nans are not equal to themselves
# check if arrays are equal ignoring np.nan
def arrays_equal_ignore_nan(arr1, arr2):
    if arr1.shape != arr2.shape:
        return False
    # Compare elements, treating nans as equal
    return np.all((arr1 == arr2) | (pd.isna(arr1) & pd.isna(arr2)))


data = {
    "class": [
        "CondoProperty",
        "CondoProperty",
        "ResidentialProperty",
        "CondoProperty",
        "ResidentialProperty",
        "CondoProperty",
        "ResidentialProperty",
        "CondoProperty",
        "ResidentialProperty",
        "CondoProperty",
    ],
    "type": [
        "Lease",
        "Lease",
        "Sale",
        "Lease",
        "Sale",
        "Lease",
        "Sale",
        "Lease",
        "Sale",
        "Lease",
    ],
    "listPrice": [
        2200.0,
        2400.0,
        850000.0,
        2700.0,
        np.nan,
        2750000.0,
        700000.0,
        2900.0,
        3500.0,
        3100.0,
    ],
    "listDate": [
        "2023-12-01 00:00:00+00:00",
        "2023-12-02 00:00:00+00:00",
        "2023-11-05 00:00:00+00:00",
        "2023-10-15 00:00:00+00:00",
        "2023-10-20 00:00:00+00:00",
        "2023-12-03 00:00:00+00:00",
        "2023-12-06 00:00:00+00:00",
        "2023-12-07 00:00:00+00:00",
        "2023-11-10 00:00:00+00:00",
        "2023-12-08 00:00:00+00:00",
    ],
    "soldPrice": [
        2200.0,
        2400.0,
        820000.0,
        2600.0,
        570000.0,
        2600000.0,
        710000.0,
        2800.0,
        np.nan,
        3000.0,
    ],
    "soldDate": ["2023-12-31 00:00:00+00:00"] * 10,
    "city": ["Toronto"] * 10,
    "area": ["Toronto"] * 10,
    "district": [
        "Toronto C01",
        "Toronto C02",
        "Toronto C03",
        "Toronto C04",
        "Toronto C05",
        "Toronto C06",
        "Toronto C07",
        "Toronto C08",
        "Toronto C09",
        "Toronto C10",
    ],
    "neighborhood": [
        "Neighborhood A",
        "Neighborhood B",
        "Neighborhood C",
        "Neighborhood D",
        "Neighborhood E",
        "Neighborhood F",
        "Neighborhood G",
        "Neighborhood H",
        "Neighborhood I",
        "Neighborhood J",
    ],
    "zip": [
        "M5P1A1",
        "M5P2A2",
        "M5P3A3",
        "M5P4A4",
        "M5P5A5",
        np.nan,
        "M5P7A7",
        "M5P8A8",
        "M5P9A9",
        "M5P0A0",
    ],
    "latitude": [43.7, 43.71, 43.72, 43.73, 43.74, 43.75, 43.76, 43.77, 43.78, 43.79],
    "longitude": [
        -79.4,
        -79.41,
        -79.42,
        -79.43,
        -79.44,
        -79.45,
        -79.46,
        -79.47,
        -79.48,
        -79.49,
    ],
    "numBathrooms": [1, 2, 3, 1, 2, 4, 3, 2, 1, 2],
    "numBedrooms": [1, 2, 3, 1, 2, 4, 3, 2, 1, 3],
    "style": [
        "Apartment",
        "2-Storey",
        "Bungalow",
        "Apartment",
        "Apartment",
        "Detached",
        "Apartment",
        "Bungalow",
        "Detached",
        "Apartment",
    ],
    "numKitchens": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "numRooms": [5, 4, 8, 4, 5, 9, 7, np.nan, 5, 4],
    "numParkingSpaces": [0, 1, 4, 2, 3, 4, 2, 3, 2, 1],
    "sqft": [
        "500-599",
        "700-799",
        "600-699",
        "0-499",
        np.nan,
        "700-799",
        np.nan,
        "500-599",
        "800-899",
        "600-699",
    ],
    "propertyType": [
        "Condo Apt",
        "Detached",
        "Comm Element Condo",
        "Condo Apt",
        "Detached",
        "Condo Apt",
        "Detached",
        "Condo Apt",
        "Detached",
        "Condo Apt",
    ],
    "numGarageSpaces": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    "numDrivewaySpaces": [np.nan] * 10,
}

test_df = pd.DataFrame(data)


# Test Cases


def test_apply_transformer():
    # create some sample columns
    feature_transform_numerical1 = ["listPrice"]
    feature_transform_numerical2 = ["numBathrooms", "numBedrooms"]
    feature_impute_numerical1 = ["listPrice"]
    feature_impute_categorical1 = ["zip"]
    feature_encode_categorical1 = ["style"]
    feature_impute_encode_categorical1 = ["sqft"]
    process = Processor(test_df, "soldPrice")
    process.transform_numerical(
        strategy="default", columns=feature_transform_numerical1
    )
    process.transform_numerical(
        strategy="normalize", columns=feature_transform_numerical2
    )
    process.impute_numerical(strategy="mean", columns=feature_impute_numerical1)
    process.impute_categorical(
        columns=feature_impute_categorical1, f=SimpleImputer(strategy="most_frequent")
    )
    process.encode_categorical(columns=feature_encode_categorical1, strategy="onehot")
    process.encode_categorical(
        columns=feature_impute_encode_categorical1, strategy="onehot"
    )
    process.impute_categorical(
        columns=feature_impute_encode_categorical1,
        f=SimpleImputer(strategy="most_frequent"),
    )
    process.apply_transformer()

    test_numerical_transformer1 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    test_numerical_transformer2 = Pipeline(steps=[("scaler", Normalizer())])
    test_categorical_transformer1 = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    test_categorical_transformer2 = Pipeline(
        steps=[("encoder", OneHotEncoder(sparse_output=False))]
    )
    test_categorical_transformer3 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False)),
        ]
    )

    # Check if the column transformer is the same as the one created via Processor object
    Processor_object_processor1 = process
    Processor_object_processor2 = process
    column_transformer_processor = ColumnTransformer(
        transformers=[
            (
                "num_transformer1",
                test_numerical_transformer1,
                feature_impute_numerical1,
            ),
            (
                "num_transformer2",
                test_numerical_transformer2,
                feature_transform_numerical2,
            ),
            (
                "cat_transformer1",
                test_categorical_transformer1,
                feature_impute_categorical1,
            ),
            (
                "cat_transformer3",
                test_categorical_transformer3,
                feature_impute_encode_categorical1,
            ),
            (
                "cat_transformer2",
                test_categorical_transformer2,
                feature_encode_categorical1,
            ),
        ],
        remainder="passthrough",
    )
    Processor_object_processor1.preprocessor.set_output(transform="pandas")
    Processor_object_processor2.preprocessor.set_output(transform="pandas")
    column_transformer_processor.set_output(transform="pandas")

    # using fit_transform
    X_transformed_object = Processor_object_processor1.fit_transform(test_df)

    # using fit and then transform
    Processor_object_processor2.fit(test_df)
    X_transformed_object2 = Processor_object_processor2.transform(test_df)

    # comparing with original sklearn processor
    X_transformed_orig = column_transformer_processor.fit_transform(test_df)

    # Check if the transformed dataframes are the same
    assert arrays_equal_ignore_nan(
        X_transformed_object.values, X_transformed_orig.values
    )
    assert arrays_equal_ignore_nan(
        X_transformed_object.values, X_transformed_object2.values
    )
    assert len(X_transformed_object.columns) == len(X_transformed_orig.columns)


def test_split_df():
    process = Processor(test_df, "soldPrice")
    (
        X,
        y,
    ) = process.train_test_split_df(target="soldPrice")
    assert len(X) == len(y)


def test_train_test_split():
    process = Processor(test_df, "soldPrice")
    (
        X,
        y,
    ) = process.train_test_split_df(target="soldPrice")
    X_train, X_test, y_train, y_test = process.train_test_split(X, y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(X)
