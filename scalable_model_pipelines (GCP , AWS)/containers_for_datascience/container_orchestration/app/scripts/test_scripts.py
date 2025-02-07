from google.cloud import aiplatform , storage
import pandas as pd
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import streamlit as st
from gcloud_connect import retrive_buckets
from gcloud_services import upload_to_bucket
import json

def get_endpoints():
    '''Returns endpoint object for model in vertex ai

    args:None
    returns: endpoint object
    '''
    # bucket = storage_client.bucket(bucket_name[-1])
    endpoint = aiplatform.Endpoint.list()
    if endpoint:
        endpoint_client = aiplatform.Endpoint(endpoint[0].name)
        return endpoint_client
    else:
        endpoint_client = aiplatform.Endpoint.create(display_name="default")
        return endpoint_client

def create_dataset_artifact(bucket_name , source_file ,file_name,display_name,project_id):
    '''Creates and Returns tabular datset artifact from gcp bucket

    args: Bucket name , file name , display nam
    returns: tabular dataframe
    '''
    bucket_names = retrive_buckets()
    # upload column_transformation to gc bucket to retrive during testing
    column_transformation = json.dumps(list(source_file.columns)).encode('utf-8')
    if bucket_name in bucket_names:
        blob_name = upload_to_bucket(bucket_name,
                                     source_file,
                                     column_transformation,
                                     file_name,
                                     display_name)
        data_path = f'gs://{bucket_name}/{blob_name}'
        cloud_dataset = aiplatform.TabularDataset.create(
            display_name=f"{display_name}",
            gcs_source=[data_path]
        )
    else:
        create_bucket_if_not_exists(bucket_name,project_id)
        blob_name = upload_to_bucket(bucket_name,source_file,column_transformation, display_name,display_name)
        data_path = f'gs://{bucket_name}/{blob_name}'
        cloud_dataset = aiplatform.TabularDataset.create(
            display_name=f"{display_name}",
            gcs_source=[data_path]
        )
    return cloud_dataset


def download_json_from_gcs(bucket_name, source_blob_name):
    # Initialize the GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file contents as a string
    file_contents = blob.download_as_string()

    # Parse the JSON content
    json_data = json.loads(file_contents)
    return json_data
def create_bucket_if_not_exists(bucket_name):
    """Creates a new bucket if it does not already exist."""
    storage_client = storage.Client(client_options={
        'api_endpoint': 'us-central1-aiplatform.googleapis.com'
    })
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        st.warning('No Bucket were found with the given name. Creating One')
        bucket = storage_client.create_bucket(bucket_name)
        st.success(f"Bucket {bucket_name} created.")
        return bucket
    else:
        st.info(f"Bucket {bucket_name} already exists.")
        return bucket_name

if __name__ == "__main__":
    from etl_pipe import process_data
    df = pd.read_csv(r'C:\csulb_projects\portfolio_projects\churn_prediction_gcp\data\customer_churn_dataset.csv')
    # processed_test_data = process_data(df,['CustomerID'],'Churn')
    # create_dataset_artifact('portfolio_buckets_2024', processed_test_data, 'customer_churn_dataset.csv'
    #                         , 'customer_churn_dataset.json', "complete-will-428105-h5")
    download_json_from_gcs('portfolio_buckets_2024', 'customer_churn_dataset.json')
    # endpoint_info = get_endpoints()
    # instance = processed_test_data.to_dict(orient="records")
    # json_format.ParseDict(instance, Value())
    # create_bucket_if_not_exists('churn_bucket')
    # # instances = [instance]
    # predictions = endpoint_info.predict(instances=instance)
    # score_index = [v.index(max(v)) for values in predictions.predictions for k, v in values.items() if k == 'scores']
    # classes = [v for values in predictions.predictions for k, v in values.items() if k == 'classes']
    # predicted_classes = [classes[idx][values] for idx, values in enumerate(score_index)]
    # processed_test_data['Predicted_Churn'] = predicted_classes
    # processed_test_data['Predicted_Churn'] = processed_test_data['Predicted_Churn'].map(
    #     {'0.0': 'No Churn', '1.0': "Churn"})
    # styled_df = processed_test_data.style.applymap(highlight_churn, subset=['Predicted_Churn'])
    # st.dataframe(styled_df)
