import datetime as dt 
import requests
import pandas as pd
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import re

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.pyplot import legend
import seaborn as sns
import numpy as np

#Data exploration

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

    print(pd.Series(na_dict).sort_values(ascending = False))

def show_cat_col_values(df, cols = None):
    """ 
    df: DataFrame
    cols: list/Dataframe

    Displays every unique categorical for each categorical variable
    """
    if cols:
        for col in cols:
            ratios = []
            values = pd.DataFrame(df[col].value_counts())
            for value in values['count'].values:
                total = np.sum(values['count'].values)
                ratios.append(value/total)
            
            values['Ratio'] = ratios

            print(values.sort_values(by = 'Ratio', ascending = False))
            print("\n")
    else:
        cats = df.select_dtypes(include = ['object'])
        for col in cats:
            ratios = []
            values = pd.DataFrame(df[col].value_counts())
            for value in values[col].values:
                total = np.sum(values[col].values)
                ratios.append(value/total)

            values['Ratio'] = ratios

            print(values.sort_values(by = 'Ratio', ascending = False))
            print("\n")


def show_corr(df):
    f = plt.figure(figsize=(106, 106))
    plt.matshow(df.select_dtypes(include = np.number).corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)


def show_num_values(df):
    for col in df.select_dtypes(['number']):
        print(df.select_dtypes(['number'])[col].value_counts())
        print("\n")


def show_num_values_col(df, col):
    for value in df[col].value_counts().sort_index().index:
        print(value)

def show_timeplots(df, by_, agg_dict, x_axis_name, y_axis_name, y = None, label = None):
    """
    df: main dataframw where the data will be parsed
    by_: time column in data series format i.e df['soldDate']
    agg: dictionary of aggregates that follows the pandas agg function format
    x_axis_name: name of the x axis that will be displayed for all plots
    y_axis_name: name fo the y axis that will be displayed for all plots
    label: name of the label you want in the legend if neccessary
    y: this is the variable that will be displayed as a second line to compare with the keys in the list 'keys'
    """
    df_avg = df.groupby(by = by_).agg(agg_dict)
    df_avg['Date'] = pd.to_datetime(df_avg.index, format='%Y-%m-%dT%H:%M:%S.%fZ')

    plt.figure(figsize=(20,10))
    keys = list(agg_dict.keys())
    if y:
        keys.remove(y)
    for i in range(len(keys)):
        plt.subplot(len(keys), 1, i+1)
        if y:
            plt.plot(df_avg['Date'], df_avg[y], color='blue', alpha=0.5, label = y)
        plt.plot(df_avg['Date'], df_avg[keys[i]], color='red', alpha=0.5, label = keys[i])
        plt.gca().xaxis.set_major_locator(MonthLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))
        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        legend()

    plt.tight_layout()
    plt.show()

def show_histogram(df, col):
    pass 

def show_scatterplot(df, col1, col2, label1 = None, label2 = None):
    plt.scatter(df[col1], df[col2], alpha = 0.5)
    if not label1:
        plt.xlabel(col1)
    else:
        plt.xlabel(label1)

    if not label2:
        plt.ylabel(col2)
    else:
        plt.ylabel(label2)

    plt.show()

def show_boxplot(df, col):
    pass

def show_barplot(df, col):
    pass

def show_pairplot(df, pairplot_cols, class_col):
    sns.pairplot(df[pairplot_cols], hue = class_col)


def plot_kde(df, col):
    na = df[col].isna().sum()
    total = len(df[col])
    missing = round(100*na/total, 2)

    if 'TARGET' in df.columns:
        plt.figure(figsize = (12, 6))
        plt.subplot(1,2,1)
        sns.kdeplot(df.loc[df['TARGET'] == 0, col], label = 'TARGET == 0')
        sns.kdeplot(df.loc[df['TARGET'] == 1, col], label = 'TARGET == 1')
        plt.xlabel(col); plt.ylabel('Density'); plt.title(f"{col} Distribution, {missing}% Missing")
        plt.legend()

        plt.subplot(1,2,2)
        sns.kdeplot(df[col].values, label = 'All')
        plt.xlabel(col); plt.ylabel('Density'); plt.title(f"{col} Distribution, {missing}% Missing")
        plt.legend()

        plt.show()
    else:
        plt.subplot(1,1,1)
        sns.kdeplot(df[col].values, label = 'All')
        plt.xlabel(col); plt.ylabel('Density'); plt.title(f"{col} Distribution, {missing}% Missing")
        plt.legend()

        plt.show()

def kde_plots(df, cols):
    if isinstance(cols, list):
        for col in cols:
            plot_kde(df,col)
    else:
        plot_kde(df, col)

# Data cleaning

## Removing Duplicates

def remove_duplicates(df: pd.DataFrame):
    pass

## Handling Missing Values

def remove_na_values(df: pd.DataFrame):
    pass

## Standardizing Text

def remove_special_characters(df: pd.DataFrame):
    pass

def standardize_text(df: pd.DataFrame):
    pass

#Data Transformation

def convert_col_dtype(df: pd.DataFrame, columns: list, convert_to_type: str):
    """
    Eligible types:
        numeric: pd.to_numerical
        datetime: pd.to_datetime
        str: series.as_type(str)
    
    """
    if convert_to_type == "str":
        for col in columns:
            df[col] = df[col].as_type("str")
    elif convert_to_type == "numeric":
        for col in columns:
            df[col] = pd.to_numeric(df[col])
    elif convert_to_type == "datetime":
        for col in columns:
            df[col] = pd.to_datetime(df[col])      
    else:
        raise ValueError(f"Unexpected convert_to_type {convert_to_type}"
                     "available types are ['numeric','datetime','str']")