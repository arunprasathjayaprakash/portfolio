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

        local_model_path = os.path.join([Model_tmp_path,bucket_name,mlflow])
        list_blob = bucket.list_blobs(blob_name)

        for blob in list_blob:
            local_file_path = os.path.join(Model_tmp_path, blob.name)

            # Ensure the subdirectories exist before downloading
            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Download the file from GCS to the local path
            print(f"Downloading {blob.name} to {local_file_path}")
            blob.download_to_filename(local_file_path)

        return flask.jsonify('All files have been downloaded'), 200
    except Exception as e:
        raise e

@app.before_request
def load_model():
    global MODEL
    #Making default values for getting model data
    #For Production run use GCP secrets for getting environment values
    model_path = download_model('model-pytorch','mlflow')
    MODEL = mlflow.sklearn.load_model(model_path)
@app.route('/predict', methods=['GET', 'POST'])
def return_predictions():
    '''
    Returns predicted probabilities from model loaded with mlflow

    args: request
    returns: Predicted probabilities
    '''
    global MODEL
    try:
        # sample request with data of {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        # in raw format
        params = flask.request.get_json()

        if params:
            # params = params.decode('utf-8')
            # params = ast.literal_eval(params)
            # data_values = params.values()

            #Updating code for getting the data
            new_data = {}

            for k , v in params.items():
                new_data[k] = v

            data_values = pd.DataFrame.from_dict(new_data,
                                      orient = "index").transpose()

            predictions = MODEL.predict_proba([list(data_values)])

            return flask.jsonify({"Predicted_probability": predictions[0][0]})
        else:
            return flask.abort(404, description="Please send required data")
    except Exception as e:
        return flask.abort(400, description=f"{e}")

if __name__ == "__main__":
    '''
    Run gcloud and docker commands 
    gcloud auth configure-docker
    docker build -t us-docker.pkg.dev/YOUR_PROJECT_ID/REPO/prediction-app:latest
    docker push us-docker.pkg.dev/YOUR_PROJECT_ID/REPO/prediction-app:latest
    
    Deploy using gcloud CLI

    Create Repository using artifact registry if you need to have repository
    
    gcloud run deploy prediction-app \
    --image <location>.pkg.dev/YOUR_PROJECT_ID/prediction-app:latest

    '''
    download_model('model-pytorch','mlflow')
    app.run(host="0.0.0.0", port=8080)