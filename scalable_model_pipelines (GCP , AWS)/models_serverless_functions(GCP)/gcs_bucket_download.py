import os.path
import mlflow
import flask
import ast
import pandas as pd
import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage

MODEL = None
bucket_name = None
model_blob = None
Model_tmp_path = '/tmp'

app = flask.Flask(__name__)

def get_cloud_credentials():
    ''' Returns Storage client object

    args: None gets default credentials from gcloud init authorization
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
def download_model(bucket_name,blob_name):
    try:
        storage_client = get_cloud_credentials()
        bucket = storage_client.get_bucket(bucket_name)

        os.makedirs(Model_tmp_path,exist_ok=True)

        list_blob = bucket.list_blobs(prefix=blob_name)
        model_local_path = os.path.join(Model_tmp_path, blob_name)
        os.makedirs(model_local_path, exist_ok=True)

        for blob in list_blob:
            local_file_path = os.path.join(Model_tmp_path, blob.name)

            # Ensure the subdirectories exist before downloading
            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Download the file from GCS to the local path
            print(f"Downloading {blob.name} to {local_file_path}")
            blob.download_to_filename(local_file_path)

        return model_local_path
    except Exception as e:
        raise e

def load_model():
    global MODEL
    # Load the model if it's not already loaded
    if MODEL is None:
        try:
            #Testing with static blobnames and bucket names
            model_path = download_model('model-pytorch', 'mlflow/')
            # Load the model using MLflow from the downloaded path
            MODEL = mlflow.sklearn.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            flask.abort(500, description="Error loading model.")

def return_predictions():
    '''
    Returns predicted probabilities from model loaded with mlflow

    args: request
    returns: Predicted probabilities
    '''
    import json
    global MODEL
    # sample request with data of {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    # in raw format
    params = {
        "msg": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    }

    if params:
        # params = params.decode('utf-8')
        # params = ast.literal_eval(params)
        # data_values = params.values()

        #Updating code for getting the data
        # new_data = {}
        #
        # for k , v in params.items():
        #     new_data[k] = v

        data_values = pd.DataFrame.from_dict(params['msg'],
                                  orient = "index").transpose()

        predictions = MODEL.predict_proba([list(data_values)])

        return flask.jsonify({"Predicted_probability": predictions[0][0]})


if __name__ == "__main__":
    # download_model('model-pytorch', 'mlflow/')
    load_model()
    return_predictions()