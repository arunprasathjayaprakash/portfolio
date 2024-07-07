import pandas as pd
import numpy as np
from io import BytesIO

def process_data(data):
    '''Returns processed data for model

    args: data , train(optional)
    returns: dataframe
    '''
    data = pd.read_csv(data)
    numerical_df = data.select_dtypes(include='number')
    numerical_df.drop('CustomerID', axis=1, inplace=True)
    numerical_df.dropna(inplace=True)
    outlier_columns = []

    for columns in numerical_df:
        md_values = np.ma.array(numerical_df[columns].values).compressed()
        median_values = numerical_df[columns].median()
        temp_df = pd.DataFrame()
        temp_df['temp_values'] = md_values - median_values
        z_scores = temp_df['temp_values'].median()
        if z_scores > 3:
            outlier_columns.append(columns)

    data.dropna(inplace=True)
    data.columns = ['_'.join(column.split(' ')) if len(column.split(' ')) == 2 else column for column in
                    data.columns]

    return data

