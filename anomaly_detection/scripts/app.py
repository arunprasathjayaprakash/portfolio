import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


import os
from stats_dt import statistics , remove_high_correlation
import visual
from train import train_and_infer

#static path for serving data will be updated to GCP bigquery
DATA_PATH = r"E:\portfolio_projects\portfolio\hyper_application\yolo_object_detection\data\kddcup.data.corrected"

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
    return pd.read_csv(DATA_PATH, header=None, names=col_names, index_col=False)


def main():
    st.title("Anomaly Detection with XGBoost")
    st.info("This application demonstrates anomaly detection using the KDD Cup dataset and XGBoost.")

    data = load_data()
    st.info("Dataset loaded successfully.")

    menu = ["Explore Data", "Visualize Dataset", "Make Inference"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Explore Data":
        st.subheader("Explore Data")
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

    elif choice == "Make Inference":
        st.subheader("Make Inference")
        
        with st.spinner("Training the model..."):
            report, model = train_and_infer(data)
            st.success("Model trained successfully!")
            st.json(report)

            st.write("Feature Importances:")
            fig, ax = plt.subplots(figsize=(10, 10))  
            xgb.plot_importance(model, ax=ax)      
            st.pyplot(fig)   


if __name__ == "__main__":
    main()
