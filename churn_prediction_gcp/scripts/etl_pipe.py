import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
def calculate_mad(df):
  mean_value = np.mean(df)
  absolute_deviations = [abs(x - mean_value) for x in df]
  mad = np.mean(absolute_deviations)
  return mad

def process_data(data,train=False):
    '''Returns processed data for model

    args: data , train(optional)
    returns: dataframe
    '''
    if train:
        pass

    data = pd.read_csv(data)
    data.drop(['CustomerID','Churn'], axis=1, inplace=True)
    numerical_columns = data.select_dtypes(include='number').columns
    categorical_columns = data.select_dtypes(include='object').columns
    outlier_columns = []


    for column in numerical_columns:
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
            data[column] = scaler.fit_transform(data[column])

    data.dropna(inplace=True)
    data.columns = ['_'.join(column.split(' ')) if len(column.split(' ')) == 2 else column for column in
                    data.columns]

    return data

