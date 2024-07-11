from google.cloud import aiplatform ,aiplatform_v1, storage
from gcloud_connect import retrive_buckets,create_bucket_if_not_exists
import streamlit as st
import io
import json
def upload_to_bucket(bucket_name,
                     file,column_file,
                     destination_blob_name,
                     column_display_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    json_blob = bucket.blob(column_display_name)
    csv_bytes = io.BytesIO()
    file.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    columns_file = io.BytesIO()
    columns_file.write(column_file)
    columns_file.seek(0)
    # Check if the file already exists
    if blob.exists() and json_blob.exists() :
        st.warning(f"File {destination_blob_name} and json {column_display_name} already exists in the bucket {bucket_name}.")
        return destination_blob_name

    # Upload the file if it doesn't exist
    if not blob.exists():
        blob.upload_from_file(csv_bytes, rewind=True)
    if not json_blob.exists():
        json_blob.upload_from_file(columns_file, rewind=True)

    st.success(f"File uploaded to {bucket_name}/{destination_blob_name}.")
    st.success(f"File uploaded to {bucket_name}/{column_display_name}.")
    return destination_blob_name

def create_endpoint(endpoint_name,location_path):
    endpoint_client = aiplatform_v1.EndpointServiceClient()

    endpoint = {
        "display_name": endpoint_name
    }

    response = endpoint_client.create_endpoint(parent=location_path, endpoint=endpoint)
    print("Waiting for operation to complete...")
    endpoint_result = response.result()
    return endpoint_result

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
        create_bucket_if_not_exists(bucket_name, project_id)
        blob_name = upload_to_bucket(bucket_name, source_file, column_transformation, file_name, display_name)
        data_path = f'gs://{bucket_name}/{blob_name}'
        cloud_dataset = aiplatform.TabularDataset.create(
            display_name=f"{display_name}",
            gcs_source=[data_path]
        )
    return cloud_dataset

def download_json_from_gcs(bucket_name, source_blob_name):
    # Initialize the GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name[-1])
    blob = bucket.blob(source_blob_name)

    # Download the file contents as a string
    file_contents = blob.download_as_string()

    # Parse the JSON content
    json_data = json.loads(file_contents)
    return json_data


def initialize_job(dataset,model_type):
    '''Returns training job

    args: dataset , model-type classfication or regression
    return: training job object , dataset object
    '''
    dataset = aiplatform.TabularDataset(dataset.resource_name)

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="train-automl",
        optimization_prediction_type=f"{model_type}"
    )

    return job ,dataset

def train_model(job,dataset):
    '''Returns model object from artifacts

    args: job details, dataset
    returns: model object
    '''
    model = job.run(
        dataset=dataset,
        target_column="Churn",
        training_fraction_split=0.6,
        validation_fraction_split=0.2,
        test_fraction_split=0.2,
        budget_milli_node_hours=1000,
        model_display_name="Churn_model",
        disable_early_stopping=False
    )

    return model
def deploy_model_to_endpoint(model, endpoint):
    endpoint_client = aiplatform_v1.EndpointServiceClient(
        client_options={'api_endpoint': 'us-central1-aiplatform.googleapis.com'}
    )

    dedicated_resources = aiplatform_v1.DedicatedResources(
        machine_spec={
            "machine_type": "n1-standard-4"  # Specify the machine type
        },
        min_replica_count=1,
        max_replica_count=1
    )

    deployed_model = {
        "model": model.name,
        "display_name": "deployed-model",
        "dedicated_resources": dedicated_resources
    }

    traffic_split = {"0": 100}

    response = endpoint_client.deploy_model(
        endpoint=endpoint.name,
        deployed_model=deployed_model,
        traffic_split=traffic_split
    )

    with st.spinner("Waiting for model deployment to complete..."):
        deploy_model_result = response.result()

    return deploy_model_result

def create_endpoint(endpoint_name,location_path):
    endpoint_client = aiplatform_v1.EndpointServiceClient(
        client_options={'api_endpoint': 'us-central1-aiplatform.googleapis.com'}
    )

    endpoint = {
        "display_name": endpoint_name
    }

    response = endpoint_client.create_endpoint(parent=location_path, endpoint=endpoint)
    with st.spinner("Waiting for operation to complete..."):
        endpoint_result = response.result()

    return endpoint_result

def deploy(model,deploy_name):
    '''Returns deployed endpoint from deployment endpooint

    args: trained model , deployment name
    return: deployment endpoint
    '''
    DEPLOYED_NAME = f"{deploy_name}"
    endpoint = model.deploy(deployed_model_display_name=DEPLOYED_NAME)

    return endpoint