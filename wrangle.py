# To get rid of those blocks of red warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Imports
import numpy as np
from scipy import stats
import pandas as pd
from math import sqrt
import os
from sklearn.cluster import KMeans
import requests
import json
import time
from scipy.stats import spearmanr

# Vis Imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.plotting import register_matplotlib_converters


# Modeling Imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import statsmodels.api as sm
from statsmodels.tsa.api import Holt

def acquire_df():
    df1=pd.read_csv("GlobalLandTemperaturesByCity.csv")
    df2=pd.read_csv("GlobalTemperatures.csv")
    df = pd.merge(df1, df2, how="left", on=["dt"])
    return df

def prepare_df(df):
    df.dt = pd.to_datetime(df.dt)
    df_cols = ['date', 'avg_temp', 'avg_temp_uncertainty', 'city',
       'country', 'lat', 'long', 'land_avg_temp',
       'land_avg_temp_uncertainty', 'land_max_temp',
       'land_max_temp_uncertainty', 'land_min_temp',
       'land_min_temp_uncertainty', 'land_ocean_avg_temp',
       'land_ocean_avg_temp_uncertainty']
    df.columns = df_cols
    df = df[(df['date'].dt.year > 1849)]
    df = df[(df['date'].dt.year != 2013)]
    df = df.dropna()
    df = df.set_index('date')
    df['lat'] = df['lat'].map(lambda x: x.rstrip('NESW'))
    df['long'] = df['long'].map(lambda x: x.rstrip('NESW'))
    df['lat'] = df['lat'].astype(float)
    df['long'] = df['long'].astype(float)
    df = df.loc[(df['country'] == 'United States')]
    df = df.loc[(df['city'] == 'Columbus')]
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]

    y = train.avg_temp

    return df, train, validate, test

def vis_1(train):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='avg_temp', y='land_avg_temp', data=train)
    plt.xlabel('Avg Temp C')
    plt.ylabel('Global Avg Land Temp')
    plt.title('Average Global Land Temp is related to the Average Temp in Columbus')


def stats_1(train):
    corr, p = spearmanr(train['avg_temp'], train['land_avg_temp'])
    print('Correlation:', corr)
    print('P-value:', p)

    alpha = 0.05
    Null = 'There is no relationship between the Global Land Average Temperature and the Average Temperature in Columbus, Ohio.'

    Alt = 'There is a relationship between the Global Land Average Temperature and the Average Temperature in Columbus, Ohio.'

    if p < alpha:
        print('We reject the null hypothesis that', Null)
        print(Alt)
    else:
        print('We fail to reject the null hypothesis that', Null)

def vis_2(train):
    train = train.resample('Y').mean()
    # create a categorical feature
    train['temp_bin'] = pd.qcut(train.avg_temp, 4, labels=['cold', 'cool', 'warm', 'hot'])
    train.groupby('temp_bin').mean()

    (train.groupby('temp_bin')
    .resample('Y')
    .size()
    .unstack(0)
    .apply(lambda row: row / row.sum(), axis=1)
    .plot.area(figsize=(10, 10), title='More days got hotter as the years progressed'))
    plt.ylabel('% of days in the year')
    plt.show()

def vis_3(train):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='avg_temp', y='land_ocean_avg_temp', data=train)
    plt.xlabel('Avg Temp C')
    plt.ylabel('Global Avg Land and Ocean Temp')
    plt.title('Average Global Land and Ocean Temp is related to the Average Temp in Columbus')

def stats_3(train):
    corr, p = spearmanr(train['avg_temp'], train['land_ocean_avg_temp'])
    print('Correlation:', corr)
    print('P-value:', p)

    alpha = 0.05
    Null = 'There is no relationship between the Global Land and Ocean Average Temperature and the Average Temperature in Columbus, Ohio.'

    Alt = 'There is a relationship between the Global Land and Ocean Average Temperature and the Average Temperature in Columbus, Ohio.'

    if p < alpha:
        print('We reject the null hypothesis that', Null)
        print(Alt)
    else:
        print('We fail to reject the null hypothesis that', Null)

def baseline_rmse(train):
    temp_pred_mean = train['avg_temp'].mean()
    train['temp_pred_mean'] = temp_pred_mean
    baseline_rmse = mean_squared_error(train.avg_temp, train.temp_pred_mean)**(1/2)
    print('Baseline RMSE:', baseline_rmse)

