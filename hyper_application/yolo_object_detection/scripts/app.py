import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import visual
import numpy as np
import os
from stats_dt import statistics , remove_high_correlation
from train import transform_infer , LabelEncoder
import json


DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'Yolo_object_detection/data/kddcup.data.corrected')

col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

@st.cache_data
def load_data():
    '''Returns Data loaded before with the first request

    args: None
    returns: None
    
    '''
    #Sampling subset of data visual inference
    data = pd.read_csv(DATA_PATH, header=None, names=col_names, index_col=False)[:10000]
    return data

def main():
    st.title("Anomaly Detection with XGBoost")
    st.info("This application demonstrates anomaly detection using the KDD Cup dataset and XGBoost.")
    data = load_data()
    menu = ["Explore Data", "Visualize Dataset", "Load model and infer new data"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Explore Data":
        st.subheader("Explore Data")
        with st.spinner("Loading Data"):
            st.info("Dataset loaded successfully.")
            st.dataframe(data.head())
            st.write(f"Shape of the dataset: {data.shape}")
            st.write("Summary statistics:")
            st.write(data.describe())

        # Correlation Analysis and Removal
        st.subheader("Remove Highly Correlated Features")
        if st.checkbox("Enable Correlation-Based Feature Removal"):
            threshold = st.slider("Set Correlation Threshold", 0.0, 1.0, 0.9, 0.01)
            reduced_data, removed_features = remove_high_correlation(data, threshold)
            data = reduced_data
            st.write(f"Features removed with correlation above {threshold}: {removed_features}")
            st.write(f"Shape after removal: {reduced_data.shape}")
            st.dataframe(reduced_data.head())

        with st.spinner("Loading statistics.Please wait...."):
            statistics(data)

    elif choice == "Visualize Dataset":
        st.subheader("Visualize Dataset")
        visual.plot_data_distribution(data)
        visual.cat_distribution(data)

    elif choice == "Load model and infer new data":
        st.subheader("Load model and infer new data")

        st.info("""
            ** 99% Accurate XGBoost model is loaded and feature importance is reflected below** 
        """)
        
        import pickle

        with open(os.path.join(os.path.dirname(os.getcwd()) ,'Yolo_object_detection/models/xgboost_model.pkl'), 'rb') as file:
            loaded_model = pickle.load(file)
        
        sample_data = data.sample(1)
        st.info("""
                ** Sample data loaded **
                """)
        inf_data = sample_data.drop('label',axis=1)
        json_data = inf_data.to_json()


        infer_matrix = transform_infer(sample_data)
        pred = loaded_model.predict(infer_matrix)

        # Predict probabilities
        pred_prob= loaded_model.predict(infer_matrix)

        # Convert probabilities to class labels
        json_data = json.loads(json_data)
        json_data["Prediction"] = data['label'].values[np.argmax(pred_prob, axis=1)][0]
        st.json(json_data) # returning data with prediction

        st.write("Feature Importances:")
        fig, ax = plt.subplots(figsize=(10, 10))  
        xgb.plot_importance(loaded_model, ax=ax)      
        st.pyplot(fig)   


if __name__ == "__main__":
    main()
