import pandas as pd
import numpy as np

def process_data(data,train=False):

    # data = pd.read_csv(data_path)
    numerical_df = data.select_dtypes(include='number')
    numerical_df.drop('CustomerID', axis=1, inplace=True)
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
    categorical_colums = data.select_dtypes(include='object').columns
    numerical_columns = data.select_dtypes(include='number').columns

    # transformations = []
    #
    # for column in categorical_colums:
    #     transformations.append({'categorical': {'column_name': column}})
    #
    # for num_column in numerical_columns:
    #     transformations.append({'numerical': {'column_name': num_column}})

    if train:
        features = data.drop('Churn', axis=1)
        labels = data['Churn']

        return features , labels
    else:
        return data

