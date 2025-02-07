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

        st.subheader("Explanation of Sample Data")

        # Overview
        st.write("### Overview")
        st.write("""
        This dataset is part of a network traffic classification problem. 
        Each row represents details about a network connection, and the goal is to predict its behavior, 
        such as whether it is 'normal' or potentially malicious.
        """)

        # Key Features Explained
        st.write("### Key Features Explained")

        # Basic Connection Features
        st.write("#### Basic Connection Features:")
        st.write("""
        - **duration**: Time (in seconds) the connection lasted.
        - **protocol_type**: Protocol used for communication (e.g., TCP, UDP, ICMP).
        - **service**: Type of network service accessed (e.g., HTTP, FTP).
        - **flag**: Status flag of the connection (e.g., SF indicates 'successful connection').
        """)

        # Traffic Statistics
        st.write("#### Traffic Statistics:")
        st.write("""
        - **src_bytes** and **dst_bytes**: Bytes sent from the source and destination.
        - **count**: Number of connections to the same host in the last 2 seconds.
        - **srv_count**: Number of connections to the same service in the last 2 seconds.
        """)

        # Error Rates
        st.write("#### Error Rates:")
        st.write("""
        - **serror_rate** and **srv_serror_rate**: Fraction of connections with SYN errors.
        - **rerror_rate** and **srv_rerror_rate**: Fraction of connections with REJ errors.
        """)

        # Host-Based Features
        st.write("#### Host-Based Features:")
        st.write("""
        - **dst_host_count**: Number of connections made to the destination host.
        - **dst_host_same_srv_rate**: Fraction of connections to the same service by the destination.
        """)

        # Sample Data Row Breakdown
        st.write("### Sample Data Row Breakdown")
        st.write("""
        | Feature           Explanation                                  |
        |----------------------------------------------------------------|
        | **duration**   --   Connection lasted for 0 seconds.             |
        | **protocol_type** -- Communication used the TCP protocol.         |
        | **service**       -- The service accessed was HTTP (web traffic). |
        | **src_bytes**     -- bytes were sent from the source.             |
        | **dst_bytes**     -- bytes were sent to the destination.          |
        | **logged_in**     -- The user successfully logged in.             |
        | **Prediction**    -- The connection is classified as normal.      |
        """)

        # Convert probabilities to class labels
        json_data = json.loads(json_data)
        json_data["Prediction"] = data['label'].values[np.argmax(pred_prob, axis=1)][0]
        st.json(json_data) # returning data with prediction

        # Feature Importance Visualization
        st.subheader("Feature Importance Visualization")

        # What Does the Image Show?
        st.write("### What Does the Image Show?")
        st.write("""
                The bar chart displays the importance of each feature in making predictions for the model. 
                The longer the bar, the more significant the feature is in influencing predictions.
                """)

        # Top Features
        st.write("#### Top Features:")
        st.write("""
                1. **src_bytes (175):** The number of bytes sent from the source is the most critical factor.
                2. **dst_host_count (112):** The number of connections to the destination host is the second most influential feature.
                3. **dst_host_same_src_port_rate (81):** The rate of connections using the same source port is highly significant.
                """)

        # How to Interpret This?
        st.write("### How to Interpret This?")
        st.write("""
                Features like **src_bytes** or **dst_host_count** are key indicators of network behavior. For example:
                - A high number of bytes (**src_bytes**) might indicate a large file upload, which could be normal or suspicious based on the context.
                - A high **dst_host_count** might indicate a potential DoS (Denial-of-Service) attack if many connections are directed to one host.
                """)

        st.write("Feature Importances:")
        fig, ax = plt.subplots(figsize=(10, 10))  
        xgb.plot_importance(loaded_model, ax=ax)      
        st.pyplot(fig)   


if __name__ == "__main__":
    main()
