from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import visual

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
        'max_depth':         8,
        'max_leaves':        2**8,
        'alpha':             0.9,
        'eta':               0.1,
        'gamma':             0.1,
        'learning_rate':     0.1,
        'subsample':         1,
        'reg_lambda':        1,
        'scale_pos_weight':  2,
        'tree_method': 'hist',
        'device':       'cuda',
        'n_gpus':            1,
        'objective':         'multi:softprob',
        'verbose':           True
    }

    params['num_class'] = len(le.classes_)

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
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    # plt.figure(figsize=(10,10))
    # visual.plot_confusion_matrix(cm, target_names)
    # st.pyplot(plt)

    return report, model