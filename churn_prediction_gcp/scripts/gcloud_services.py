from google.cloud import aiplatform ,aiplatform_v1, storage
from gcloud_connect import retrive_buckets,create_bucket_if_not_exists
import streamlit as st
import io
def upload_to_bucket(bucket_name, file, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    csv_bytes = io.BytesIO()
    file.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    # Check if the file already exists
    if blob.exists():
        st.warning(f"File {destination_blob_name} already exists in the bucket {bucket_name}.")
        return

    # Upload the file if it doesn't exist
    blob.upload_from_file(csv_bytes, rewind=True)
    st.success(f"File uploaded to {bucket_name}/{destination_blob_name}.")


def create_endpoint(endpoint_name,location_path):
    endpoint_client = aiplatform_v1.EndpointServiceClient()

    endpoint = {
        "display_name": endpoint_name
    }

    response = endpoint_client.create_endpoint(parent=location_path, endpoint=endpoint)
    print("Waiting for operation to complete...")
    endpoint_result = response.result()
    return endpoint_result

def create_dataset_artifact(bucket_name , source_file ,file_name,display_name):
    '''Creates and Returns tabular datset artifact from gcp bucket

    args: Bucket name , file name , display nam
    returns: tabular dataframe
    '''
    bucket_names = retrive_buckets()
    if bucket_name in bucket_names:
        data_path = f'gs://{bucket_name}/{file_name}'
        upload_to_bucket(bucket_name,source_file,display_name)
        cloud_dataset = aiplatform.TabularDataset.create(
            display_name=f"{display_name}",
            gcs_source=[data_path]
        )
    else:
        create_bucket_if_not_exists(bucket_name)
        data_path = f'gs://{bucket_name}/{file_name}'
        upload_to_bucket(bucket_name, source_file, display_name)
        cloud_dataset = aiplatform.TabularDataset.create(
            display_name=f"{display_name}",
            gcs_source=[data_path]
        )
    return cloud_dataset

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