import os
import google.auth
from google.auth.transport.requests import Request
from google.cloud.storage import Client, transfer_manager
from google.cloud import storage

def get_cloud_credentials():
    ''' Returns Storage client object

    args: None gets default credentials from gcloud init autorization
    return: Storage client object
    '''
    try:
        credentials, _ = google.auth.default()
        auth_request = Request()
        credentials.refresh(auth_request)

        storage_client = storage.Client(project=credentials.quota_project_id,
                                         credentials=credentials)
        return storage_client
    except Exception as e:
        raise e
def create_buckets(storage_name):
    ''' Creates bucket with the given name

    args: Name
    return: storage object
    '''
    storage_buckets = get_cloud_credentials()

    if storage_buckets.list_blobs(storage_name):
        raise ValueError("Please select another bucket name to create as it already exists")
    else:
        try:
            storage_buckets.create_bucket(storage_name)
        except Exception as e:
            raise e
        
def upload_to_storage(storage_name, source_directory="", workers=8):
    ''' Uploads files using Google transfer manager

    args: bucket-name , folder path , workers default to 8
    returns: None
    '''

    storage_client = get_cloud_credentials()
    storage_bucket = storage_client.get_bucket(storage_name)

    file_names = [file for file in os.listdir(source_directory)]

    results = transfer_manager.upload_many_from_filenames(
        storage_bucket, file_names, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(file_names, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, storage_bucket.name))

if __name__ == "__main__":
    storage_name = "model-pytorch"
    folder_path = os.path.join(os.path.dirname(os.getcwd()),'models/model_V1')
    upload_to_storage(storage_name, source_directory=folder_path, workers=8)