import pandas as pd
import numpy as np
import streamlit as st

import streamlit
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder
def calculate_mad(df):
  mean_value = np.mean(df)
  absolute_deviations = [abs(x - mean_value) for x in df]
  mad = np.mean(absolute_deviations)
  return mad

def process_data(data,drop_columns,target_column,transformation_columns,fillna=True,train=False):
    '''Returns processed data for model

    args: data , train(optional)
    returns: dataframe
    '''

    '''
    Simple processing since vertex Ai does the preprocessing and analyzing in vertex ai platform for best model results
    '''

    data.drop(drop_columns, axis=1, inplace=True)
    numerical_columns = data.select_dtypes(include='number').columns
    categorical_columns = data.select_dtypes(include='object').columns
    outlier_columns = []

    for column in numerical_columns:
        if column != 'Churn':
            zscores = np.abs((data[column] - data[column].median()) / calculate_mad(data[column]))
            thesh = 3.5
            if (zscores > thesh).any():
                outlier_columns.append(column)

    for column in outlier_columns:
        twenty_per = np.percentile(data[column], 25)
        seventy_five = np.percentile(data[column], 75)
        IQR = seventy_five - twenty_per
        data[column] = data[column].clip(lower=-1.5 * IQR, upper=1.5 * IQR)

    normalizing_cols = []

    numerical_stats = data[numerical_columns].describe()

    for column in numerical_columns:
        if numerical_stats.loc['max', column] > 10 * numerical_stats.loc['min', column] and column != 'Churn':
            normalizing_cols.append(column)

    if normalizing_cols:
        scaler = MinMaxScaler()
        for column in normalizing_cols:
            data[column] = scaler.fit_transform(data[column].values.reshape(-1,1))

    cat_encoder = OneHotEncoder()
    encoded_features = cat_encoder.fit_transform(data[categorical_columns]).toarray()
    encoded_df = pd.DataFrame(encoded_features)

    # Combining numeric and encoded categorical features
    processed_data = pd.concat([data[numerical_columns], encoded_df], axis=1)

    if fillna:
        processed_data[numerical_columns] = processed_data[numerical_columns].fillna(0)
    else:
        data.dropna(inplace=True)

    processed_data.columns = [str(column).replace(' ','_') for column in processed_data.columns]

    for column in transformation_columns:
        if column not in processed_data.columns:
            processed_data[column] = float(0)

    #handling missing feature data if any
    processed_data.fillna(float(0),inplace=True)
    if train:
        return processed_data
    else:
        processed_data.drop(target_column,axis=1,inplace=True)
        return processed_data

