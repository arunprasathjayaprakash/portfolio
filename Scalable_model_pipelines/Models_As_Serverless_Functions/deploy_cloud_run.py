import os.path
import mlflow
import flask
import ast
import pandas as pd
import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage

MODEL = None
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
    global MODEL
    try:
        storage_client = get_cloud_credentials()
        bucket = storage_client.get_bucket(bucket_name)

        os.makedirs(Model_tmp_path, exist_ok=True)

        #retriving blob with the prefix that has been created with.
        #Check function documentation for how prefix works
        list_blob = bucket.list_blobs(prefix=blob_name)

        # Download all files in the folder (simulating a folder structure in GCS)
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

        MODEL = mlflow.sklearn.load_model(model_local_path)
        return MODEL
    except Exception as e:
        raise e

@app.route('/hello',methods=['GET','POST'])
def testing_value():
    return flask.jsonify({"msg":"Hello"})

@app.route('/predict',methods=['GET','POST'])
def return_predictions():
    '''
    Returns predicted probabilities from model loaded with mlflow

    args: request
    returns: Predicted probabilities
    '''
    global MODEL

    try:
        if not MODEL:
            MODEL = download_model('model-pytorch','mlflow-1/')
    except Exception as e:
        return flask.abort(300, description=f"{e}")

    try:
        # sample request with data of {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        # in raw format
        params = flask.request.get_json()

        if params:

            #Updating code for getting the data
            new_data = []

            for k , v in params.items():
                new_data.append(v)

            predictions = MODEL.predict_proba([list(new_data)])

            return flask.jsonify({"Predicted_probability": max(predictions[0]),
                                  "Class": list(predictions[0]).index(max(predictions[0]))})
        else:
            return flask.abort(404, description="Please send required data")
    except Exception as e:
        return flask.abort(404,{})

if __name__ == "__main__":
    '''
    Containerazie using docker commands
    docker build -t location-docker.pkg.dev/project_id/repo_id/application:tag .
    docker push location-docker.pkg.dev/project_id/repo_id/application:tag
    
    Deploy to gcloud using
    gcloud run deploy --image docker image tag
    
    '''
    app.run(host='0.0.0.0',port=8080)