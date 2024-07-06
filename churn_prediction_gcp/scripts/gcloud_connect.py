from google.cloud import storage , aiplatform
import streamlit as st
import os
import json

def retrive_connection():
    '''Returns bucket names verifying gcloud credentials

    args: None
    returns: bucket names in gcloud for default cloud authentication
    '''
    try:
        storage_unit = storage.Client()
        buckets = storage_unit.list_buckets()

        bucket_names = []

        for bucket in buckets:
            bucket_names.append(bucket.name)

        return bucket_names
    except:
        raise "Check gcloud intialization credentials"
def get_credentials(storage_client,bucket_name,blob_name='portfolio_buckets_2024'):
    ''' Returns credential data from SSO default google authentication

    args: string storage , bucket name , blob name
    returns: json
    '''
    # bucket = storage_client.bucket(bucket_name[-1])
    data_blob = storage_client.list_blobs(bucket_name[-1])

    for blob in data_blob:
        if blob.name == blob_name:
            credentials_data = json.loads(blob.download_as_text())
            return credentials_data

#Module testing function
def get_endpoints():
    '''Returns endpoint object for model in vertex ai

    args:None
    returns: endpoint object
    '''
    # bucket = storage_client.bucket(bucket_name[-1])
    endpoint = aiplatform.Endpoint.list(filter="display_name=Churn_model_endpoint")
    try:
        endpoint_client = aiplatform.Endpoint(endpoint[0].name)
        return endpoint_client
    except:
        with st.form("Could Not find any endpoint for prediction.",clear_on_submit=True):
            option = st.radio('Do you want to train new model with data?',
                     ('Yes','No'))
            submit = st.form_submit_button()
            if submit and option == 'Yes':
                st.session_state.page = "main"
                return
            elif submit and option == 'No':
                st.info('No prediction endpoints were found select retrain option and upload data to train'
                        )
                return None



def login_gcloud():
    ''' Returns credentials and endpoints

    args: None
    returns: credentials json , endpoints json
    '''
    with st.spinner("Please wait while we intialize GCloud..."):
        os.system("gcloud auth application-default login")
        login = False

    with st.spinner("Retrieving Cloud Credentials"):
        storage_client = storage.Client()
        buckets = retrive_connection()
        credentials = get_credentials(storage_client, buckets)
        endpoint = get_endpoints()

    return credentials , endpoint

