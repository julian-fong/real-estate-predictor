"""Main Module for methods to handle data cleaning, data analysis, and data transformation"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.pyplot import legend

# Data Exploration / Analysis


def show_dtypes(df):
    print(df.dtypes)


def show_missing_values(df):
    """
    df: DataFrame

    Prints the missing perentage of every column
    """
    na_dict = {}
    for col in df.columns:
        na = df[col].isna().sum()
        total = len(df[col])

        na_dict[col] = f"{round(100*na/total,2)}%"

    print(pd.Series(na_dict).sort_values(ascending=False))


def show_cat_col_values(df, columns=None):
    """
    df: DataFrame
    cols: list/Dataframe

    Displays every unique categorical for each categorical variable
    """
    if columns:
        for col in columns:
            ratios = []
            values = pd.DataFrame(df[col].value_counts())
            for value in values["count"].values:
                total = np.sum(values["count"].values)
                ratios.append(value / total)

            values["Ratio"] = ratios
            values["Percentage"] = values["Ratio"] * 100
            print(values.sort_values(by="Ratio", ascending=False))
            print("\n")
    else:
        cats = df.select_dtypes(include=["object"])
        for col in cats:
            ratios = []
            values = pd.DataFrame(df[col].value_counts())
            for value in values["count"].values:
                total = np.sum(values["count"].values)
                ratios.append(value / total)

            values["Ratio"] = ratios
            values["Percentage"] = values["Ratio"] * 100
            print(values.sort_values(by="Ratio", ascending=False))
            print("\n")


def show_unique_values_per_column(df, path=None, columns=None):
    """
    Creates an entirely new dataframe which contains all unique values for each column

    """
    new_df = pd.DataFrame()
    if not columns:
        columns = df.select_dtypes(include=["object"]).columns

    for col in columns:
        new_column = pd.DataFrame(df[col].value_counts().index)

        new_df = pd.concat([new_df, new_column], axis=1)

    if path:
        new_df.to_csv(path, index=False)

    return new_df


def show_corr(df, columns):
    f = plt.figure(figsize=(106, 106))
    if not columns:
        columns = df.select_dtypes(include=np.number).columns
    plt.matshow(df[columns].corr(), fignum=f.number)
    plt.xticks(
        range(df[columns].shape[1]), df[columns].columns, fontsize=14, rotation=45
    )
    plt.yticks(range(df[columns].shape[1]), df[columns].columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Correlation Matrix", fontsize=16)


def show_num_values(df):
    for col in df.select_dtypes(["number"]):
        print(df.select_dtypes(["number"])[col].value_counts())
        print("\n")


def show_num_values_col(df, col):
    for value in df[col].value_counts().sort_index().index:
        print(value)


def plot_timeplots(df, by_, agg_dict, x_axis_name, y_axis_name, y=None):
    """
    df: main dataframw where the data will be parsed
    by_: time column in data series format i.e df['soldDate']
    agg: dictionary of aggregates that follows the pandas agg function format
    x_axis_name: name of the x axis that will be displayed for all plots
    y_axis_name: name fo the y axis that will be displayed for all plots
    label: name of the label you want in the legend if neccessary
    y: this is the variable that will be displayed as a second line to compare with the keys in the list 'keys'
    """
    df_avg = df.groupby(by=by_).agg(agg_dict)
    df_avg["Date"] = pd.to_datetime(df_avg.index, format="%Y-%m-%dT%H:%M:%S.%fZ")

    plt.figure(figsize=(20, 10))
    keys = list(agg_dict.keys())
    if y:
        keys.remove(y)
    for i in range(len(keys)):
        plt.subplot(len(keys), 1, i + 1)
        if y:
            plt.plot(df_avg["Date"], df_avg[y], color="blue", alpha=0.5, label=y)
        plt.plot(df_avg["Date"], df_avg[keys[i]], color="red", alpha=0.5, label=keys[i])
        plt.gca().xaxis.set_major_locator(MonthLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter("%b %Y"))
        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        legend()

    plt.tight_layout()
    plt.show()


def plot_frequency_histogram(
    df, col, bins, xlabel=None, ylabel=None, weights=None, edgecolor="black"
):
    plt.hist(df[col], bins=bins, weights=weights, edgecolor=edgecolor)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{col} Relative Frequency Histogram")
    plt.show()


def plot_scatterplot(df, col1, col2, label1=None, label2=None):
    plt.scatter(df[col1], df[col2], alpha=0.5)
    if not label1:
        plt.xlabel(col1)
    else:
        plt.xlabel(label1)

    if not label2:
        plt.ylabel(col2)
    else:
        plt.ylabel(label2)

    plt.show()


def show_pairplot(df, pairplot_cols, class_col):
    sns.pairplot(df[pairplot_cols], hue=class_col)


def plot_kde(df, col):
    na = df[col].isna().sum()
    total = len(df[col])
    missing = round(100 * na / total, 2)

    if "TARGET" in df.columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.kdeplot(df.loc[df["TARGET"] == 0, col], label="TARGET == 0")
        sns.kdeplot(df.loc[df["TARGET"] == 1, col], label="TARGET == 1")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"{col} Distribution, {missing}% Missing")
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.kdeplot(df[col].values, label="All")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"{col} Distribution, {missing}% Missing")
        plt.legend()

        plt.show()
    else:
        plt.subplot(1, 1, 1)
        sns.kdeplot(df[col].values, label="All")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"{col} Distribution, {missing}% Missing")
        plt.legend()

        plt.show()


def kde_plots(df, cols):
    if isinstance(cols, list):
        for col in cols:
            plot_kde(df, col)
    else:
        plot_kde(df, col)


# Data cleaning

# Removing Duplicates


def remove_duplicates(df: pd.DataFrame, columns=None, inplace=False, ignore_index=True):
    """
    Make sure to assign this to a new variable if inplace = False
    """
    if not inplace:
        df = df.drop_duplicates(
            subset=columns, inplace=inplace, ignore_index=ignore_index
        )
        return df
    else:
        df = df.drop_duplicates(
            subset=columns, inplace=inplace, ignore_index=ignore_index
        )


# Handling Missing/Bad Values


def helper_calculate_missing_values_percentage_by_col(df: pd.DataFrame, column):
    """
    Calculates the missing values percentage by column

    Parameters
    ----------

    df : pd.DataFrame
        input pandas dataframe

    column : str
        column name to calculate missing values percentage

    Returns
    -------
    float
        missing values percentage
    """
    return round(100 * df[column].isna().sum() / len(df[column]), 2)


def remove_na_values_by_col(df: pd.DataFrame, strategy, columns=None, threshold=None):
    """
    Removes na values by column

    Parameters
    ----------

    df : pd.DataFrame
        input pandas dataframe

    strategy : str
        Available parameters:
            columns - will completely drop whatever column is inside the `columns` parameter
            rows - will drop rows if there are any missing values in the passed columns
            columns_threshold - will completely drop the column if the number of missing values exceed the passed threshold

    columns : list
        default = None
        list of columns used to drop missing values

    threshold : float
        default = None
        threshold used to drop columns if the missing values exceed the threshold. Between 0 and 1
    """
    if strategy == "columns":
        if not columns:
            raise ValueError(f"missing columns, got {columns}")
        df = df.drop(columns=columns, axis=1)
    elif strategy == "rows":
        df = df.dropna(subset=columns, ignore_index=True)
    elif strategy == "columns_threshold":
        if not columns:
            raise ValueError(f"missing columns, got {columns}")
        if not threshold:
            raise ValueError(f"missing mandatory field threshold, got {threshold}")
        elif threshold < 0 or threshold > 1:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

        for col in columns:
            missing_values_percentage = (
                helper_calculate_missing_values_percentage_by_col(df, col)
            )
            if missing_values_percentage > threshold:
                df = df.drop(columns=col, axis=1)
    else:
        raise ValueError(
            f"unexpected strategy {strategy}, available strategies are ['columns', 'rows', 'columns_threshold']"
        )
    return df


def replace_values(df: pd.DataFrame, column, value, replacement=np.nan):
    """
    Replaces values in a given column

    Parameters
    ----------

    df : pd.DataFrame
        input pandas dataframe

    strategy : str
        Available strategies:
            replace: replaces all occurences of `value` with the replacement value

    column : str
        column name to replace values

    value : str
        value to replace

    replacement : str
        replacement value
    """
    df[column] = df[column].replace(value, replacement)
    return df


# Standardizing Text


def replace_special_text(text, keep_hyphens=False, space_mode="underscore"):
    """
    Cleans a given text by replacing special characters and optionally modifying spaces.

    Parameters:
        text (str): The input text to clean.
        keep_hyphens (bool): If True, hyphens are not removed.
        space_mode (str): How to handle spaces. Options are:
                          - "underscore": Replace spaces with underscores.
                          - "remove": Remove spaces entirely.
                          - "keep": Keep spaces unchanged.

    Returns:
        str: The cleaned text.
    """
    if not keep_hyphens:
        text = re.sub(
            r"[^a-zA-Z0-9\s]", "", text
        )  # Remove special characters, keep spaces
    else:
        text = re.sub(
            r"[^a-zA-Z0-9\s\-]", "", text
        )  # Remove special characters but keep hyphens

    if space_mode == "underscore":
        text = re.sub(r"\s+", "_", text)  # Replace spaces with underscores
    elif space_mode == "remove":
        text = re.sub(r"\s+", "", text)  # Remove spaces entirely

    return text


def clean_string(text, keep_hyphens=False, space_mode="underscore", errors="raise"):
    """
    Cleans a given text by replacing special characters and optionally modifying spaces.

    Parameters
    ----------

    text (str): The input text to clean.
    keep_hyphens (bool): If True, hyphens are not removed.
    space_mode (str): How to handle spaces. Options are:
                        - "underscore": Replace spaces with underscores.
                        - "remove": Remove spaces entirely.
                        - "keep": Keep spaces unchanged.
    errors : str
        how to handle encountered errors. Possible values:
            'raise': If `raise`, then invalid parsing will raise an exception.
            'coerce': If `coerce`, then invalid parsing will be set as np.nan
            'ignore': If `ignore`, then invalid parsing will return the input.

    Returns
    -------
        str: The cleaned text or the original text if an error is encountered with errors = 'ignore'.
        np.nan: If the input text is invalid.
    """
    try:
        text = text.strip().lower()
        text = replace_special_text(
            text, keep_hyphens=keep_hyphens, space_mode=space_mode
        )
    except Exception as e:
        if errors == "ignore":
            return text
        elif errors == "coerce":
            return np.nan
        elif errors == "raise":
            raise e

    return text


# Individual Predictors

# Postal Code


def helper_standardize_postal_code(x: str):
    # convert from ### ### to ######
    try:
        x = x.replace(" ", "")
    except ValueError:
        x = np.nan
        return x

    if len(x) != 6:
        x = np.nan

    return x


def standardize_postal_code(df: pd.DataFrame, col: str = "zip"):
    """
    Standardizes a postal code column by removing spaces and ensuring it is 6 characters long.

    Assumes the column name zip is in the columns

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the postal code column to standardize.

    Returns:
        pd.Series: The standardized postal code column.
    """
    df[col] = df[col].apply(helper_standardize_postal_code)

    return df


# Ammenities / Condo Ammenities


def helper_clean_ammenities_list(x: list):
    """
    Assumes that the input x is a list of strings
    """
    cleaned_ammenities_list = []
    if not x:
        return []

    for value in x:
        value = clean_string(
            value, keep_hyphens=False, space_mode="underscore", errors="coerce"
        )
        cleaned_ammenities_list.append(value)

    return cleaned_ammenities_list


def standardize_ammenities_text(df):
    """
    Removes special characters and spaces between categories

    Assumes the columns `ammenities` and `condo_ammenities` are in the dataframe
    """

    df["ammenities"] = df["ammenities"].apply(helper_clean_ammenities_list)
    df["condo_ammenities"] = df["condo_ammenities"].apply(helper_clean_ammenities_list)

    return df


def standardize_locations_text(df):
    """
    Removes any special characters between categories for locations and standardizes the text

    Assumes the columns `area`, `city`, and `neighborhood` are in the dataframe
    """

    df["area"] = df["area"].apply(
        lambda x: clean_string(x, keep_hyphens=True, space_mode="keep", errors="coerce")
    )
    df["city"] = df["city"].apply(
        lambda x: clean_string(x, keep_hyphens=True, space_mode="keep", errors="coerce")
    )
    df["district"] = df["district"].apply(
        lambda x: clean_string(x, keep_hyphens=True, space_mode="keep", errors="coerce")
    )
    df["neighborhood"] = df["neighborhood"].apply(
        lambda x: clean_string(x, keep_hyphens=True, space_mode="keep", errors="coerce")
    )

    return df


def standardize_style_text(df):
    """
    Removes any special characters between categories for style and standardizes the text

    Assumes the columns `style` is in the dataframe
    """

    df["style"] = df["style"].apply(
        lambda x: clean_string(x, keep_hyphens=True, space_mode="keep", errors="coerce")
    )

    return df


def standardize_propertyType_text(df):
    """
    Removes any special characters between categories for style and standardizes the text

    Assumes the columns `propertyType` is in the dataframe
    """

    df["propertyType"] = df["propertyType"].apply(
        lambda x: clean_string(x, keep_hyphens=True, space_mode="keep", errors="coerce")
    )

    return df


# Reducing cardinality of categorical columns


def reduce_cardinality(df, columns, threshold=0.01):
    minimum_observations = threshold * len(df)

    for col in columns:
        value_counts = df[col].value_counts()
        df[col] = df[col].apply(
            lambda x: x if value_counts[x] > minimum_observations else "Other"
        )

    return df


# Data Analysis

# Outliers


def removeOutliers(df, strategy="all", columns=None, threshold=None, multiplier=None):
    """
    Removes outliers from the dataframe based on a 95% and 5% quantile

    Parameters
    ----------
    df : pd.DataFrame

    strategy : str
        Default = "all"
        Available strategies:
            all - removes outliers from all numerical columns
            columns - removes outliers from the specified columns in the columns parameter

    columns : list
        List of columns to apply outlier removal from
    """
    if strategy == "columns":
        if not columns:
            raise ValueError(
                "Columns must be specified when using the columns strategy"
            )
    if not multiplier:
        multiplier = 3
    if not threshold:
        upper_threshold = 0.95
        lower_threshold = 1 - upper_threshold
    else:
        if isinstance(threshold, float) and threshold > 0 and threshold < 1:
            upper_threshold = threshold
            lower_threshold = 1 - upper_threshold
        else:
            raise ValueError("parameter `threshold` must be a float between 0 and 1")

    if strategy == "all":
        for col in df.select_dtypes(include=["number"]).columns:
            if col != "index":
                percentile90 = df[col].quantile(upper_threshold)
                percentile10 = df[col].quantile(lower_threshold)

                iqr = percentile90 - percentile10

                upper_limit = percentile90 + multiplier * iqr
                lower_limit = percentile10 - multiplier * iqr

                if lower_limit != upper_limit:
                    df = df[df[col] < upper_limit]
                    df = df[df[col] > lower_limit]

                print(col, len(df))

    elif strategy == "columns":
        for col in columns:
            percentile90 = df[col].quantile(upper_threshold)
            percentile10 = df[col].quantile(lower_threshold)

            iqr = percentile90 - percentile10

            upper_limit = percentile90 + multiplier * iqr
            lower_limit = percentile10 - multiplier * iqr

            if lower_limit != upper_limit:
                df = df[df[col] < upper_limit]
                df = df[df[col] > lower_limit]

    return df
