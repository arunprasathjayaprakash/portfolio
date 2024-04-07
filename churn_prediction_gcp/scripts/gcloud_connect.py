from google.cloud import storage , aiplatform
import streamlit as st
import os
import json

def retrive_connection():
    try:
        storage_unit = storage.Client()
        buckets = storage_unit.list_buckets()

        bucket_names = []

        for bucket in buckets:
            bucket_names.append(bucket.name)

        return bucket_names
    except:
        raise "Check gcloud intialization credentials"
def get_credentials(storage_client,bucket_name,blob_name='credentials_2024_educative'):
    # bucket = storage_client.bucket(bucket_name[-1])
    data_blob = storage_client.list_blobs(bucket_name[-1])

    for blob in data_blob:
        if blob.name == blob_name:
            credentials_data = json.loads(blob.download_as_text())
            return credentials_data

def get_endpoints():
    # bucket = storage_client.bucket(bucket_name[-1])
    endpoint = aiplatform.Endpoint.list(filter="display_name=Churn_model_endpoint")
    endpoint_client = aiplatform.Endpoint(endpoint[0].name)
    return endpoint_client

def login_gcloud():
    with st.spinner("Please wait while we intialize GCloud..."):
        os.system("gcloud auth application-default login")
        login = False

    with st.spinner("Retrieving Cloud Credentials"):
        storage_client = storage.Client()
        buckets = retrive_connection()
        credentials = get_credentials(storage_client, buckets)
        endpoint = get_endpoints()

    return credentials , endpoint

