import streamlit as st
import pandas as pd
import os
from gcloud_connect import get_credentials , retrive_connection , get_endpoints , login_gcloud
from google.cloud import storage , aiplatform
from etl_pipe import process_data
from google.protobuf import json_format


def highlight_churn(s):
    """
    Highlight cells in the 'Churn' column based on the condition
    """
    if s == 'Churn':
        return 'background-color: red'
    else:
        return 'background-color: green'

def app():
    st.title('Churn Prediction')
    st.subheader('Analytics By Vertex AI')

    login_holder = st.empty()

    if st.checkbox('Login to GCloud'):
        with login_holder.form('login input',clear_on_submit=True):
            login_option = st.selectbox('Do you want to login to gcloud now?',options=['Yes','No'])
            login_submit = st.form_submit_button()

        if login_submit and login_option == 'Yes':
            credentials , endpoint = login_gcloud()
            st.info("Successfully logged in GCloud")
            login_holder.empty()


    with st.form('Input',clear_on_submit=True):
        uploaded_file = st.file_uploader('Upload Your Test data',type='csv')
        st.write('Copy paste your key value pair (Optional)')
        st.text_input('Enter Key value Pairs for test data')
        st.info('Select checkbox and model if you want to retrain')
        retrain_option = st.checkbox('Retrain model (if You have selected checkbox)')
        model_value = st.selectbox('Select Which model of your choice',['Decision Tree','Random Forest','XGBoost'])
        submit = st.form_submit_button()

        if submit:
            #check for retrain_model
            if retrain_option:
                #Yet to be developed for v0.1
                model = model_value
                pass

            contents = pd.read_csv(uploaded_file)
            processed_test_data = process_data(contents)
            with st.spinner("Predictions are underway"):
                endpoint_info = get_endpoints()
                instance = processed_test_data.astype(str).to_dict(orient='records')
                # instances = [instance]
                predictions = endpoint_info.predict(instances=instance)
                score_index = [v.index(max(v)) for values in predictions.predictions for k,v in values.items() if k=='scores']
                classes = [v for values in predictions.predictions for k,v in values.items() if k=='classes']
                predicted_classes = [classes[idx][values] for idx,values in enumerate(score_index)]
                processed_test_data['Predicted_Churn'] = predicted_classes
                processed_test_data['Predicted_Churn'] = processed_test_data['Predicted_Churn'].map({'0.0':'No Churn','1.0':"Churn"})
                styled_df = processed_test_data.style.applymap(highlight_churn, subset=['Predicted_Churn'])
                st.dataframe(styled_df)

if __name__ == "__main__":
    app()