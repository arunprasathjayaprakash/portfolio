from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import visual
import json
import os

def transform_infer(df):
    ''' Return converted data for inference pipeline
    
    args: dataframe
    return: converted dataframe
    '''

    cat_columns = df.select_dtypes(include='object').columns
    numeric_vars = list(set(df.columns.values.tolist()) - set(cat_columns))

    cat_data = pd.get_dummies(df[cat_columns])
    numeric_data = df[numeric_vars].copy()
    numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)

    with open(os.path.join(os.path.dirname(os.getcwd()), 'Yolo_object_detection/feature_store/feature_set.json'), 'r') as feature:
        req_cols = json.load(feature)['columns']

    lt = list((set(req_cols) - set(numeric_cat_data.columns)))

    for col in lt:
        numeric_cat_data[col] = [0] * len(numeric_cat_data)

    # numeric_cat_data[lt] = [0] * len(numeric_cat_data)
    numeric_cat_data = numeric_cat_data[req_cols]

    labels = df['label']
    le = LabelEncoder()
    le.fit(labels)
    integer_labels = le.transform(labels)

    new_data = xgb.DMatrix(numeric_cat_data,integer_labels)

    return new_data


def train_and_infer(df, classification="Binary"):
    ''' Returns binary or multiclass classification report and model based on the option selected
    from UI

    args: Dataframe , classification metrics
    returns: Metrics , report
    
    '''
    cat_vars = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    cat_data = pd.get_dummies(df[cat_vars])

    numeric_vars = list(set(df.columns.values.tolist()) - set(cat_vars))
    numeric_vars.remove('label')
    numeric_data = df[numeric_vars].copy()

    numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)

    labels = df['label'].copy()
    le = LabelEncoder()
    le.fit(labels)
    integer_labels = le.transform(labels)

    params = {
        'num_rounds':        10,
        'n_estimators': 100,
        'max_depth':         6,
        'max_leaves':        2**4,
        'alpha':             0.9,
        'eta':               0.1,
        'gamma':             0.1,
        'learning_rate':     0.1,
        'subsample':         1,
        'reg_lambda':        1,
        'scale_pos_weight':  2,
        'objective':         'multi:softprob',
        'verbose':           True
    }

    params['num_class'] = len(le.classes_)


    #storing feature set for inference
    with open(os.path.join(os.path.dirname(os.getcwd()), 'Yolo_object_detection/feature_store/feature_set.json'), 'w') as feature:
        json.dump({"columns": numeric_cat_data.columns.tolist()}, feature)


    x_train, x_test, y_train, y_test = train_test_split(numeric_cat_data, integer_labels, test_size=0.25, random_state=42)

    dtrain = xgb.DMatrix(x_train,label=y_train)
    dtest = xgb.DMatrix(x_test,label=y_test)
    evals = evals = [(dtest, 'test',), (dtrain, 'train')]
    model = xgb.train(params,dtrain,params['num_rounds'],evals)

    y_pred = model.predict(dtest)

    # Predict probabilities
    y_pred_prob = model.predict(dtest)

    # Convert probabilities to class labels
    y_pred = np.argmax(y_pred_prob, axis=1)

    present_classes = np.unique(y_test)
    target_names = le.inverse_transform(present_classes)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=target_names)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    return model , x_train

if __name__ == "__main__":
    '''
    Static for trianing model in local since container fails when training due to docker limitations
    '''
    import os
    
    DATA_PATH = ''

    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
    ]
    df = pd.read_csv(DATA_PATH, header=None, names=col_names, index_col=False)
    model , train_data = train_and_infer(df)
    import pickle
    with open('E:\portfolio_projects\portfolio\hyper_application\yolo_object_detection\models/xgboost_model.pkl', 'wb') as file:
        pickle.dump(model, file)