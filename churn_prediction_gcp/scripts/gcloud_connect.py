from google.cloud import storage , aiplatform
from google.cloud import aiplatform_v1
import streamlit as st
import os
import json
from google.protobuf.json_format import MessageToDict
def check_running_jobs(project_id, region):
    """Check if there are any running training jobs in the specified project and region."""
    finished_pipelines = []
    client = aiplatform_v1.PipelineServiceClient(client_options={
        'api_endpoint': f'{region}-aiplatform.googleapis.com'
    })
    parent = f'projects/{project_id}/locations/{region}'

    request = aiplatform_v1.ListTrainingPipelinesRequest(
        parent=parent
    )
    page_result = client.list_training_pipelines(request=request)
    for response in page_result:
        response_dict = MessageToDict(response._pb)
        state = response_dict.get('state')
        if state in ['PIPELINE_STATE_RUNNING', 'PIPELINE_STATE_PENDING']:
            print(f"Found a running pipeline: {response_dict['displayName']} with state: {state}")
            return True
        elif state == 'PIPELINE_STATE_SUCCEEDED':
            finished_pipelines.append(response_dict.get('modelToUpload')['name'])

    return False , finished_pipelines

    # for job in jobs:
    #     job_dict = MessageToDict(job._pb)
    #     state = job_dict.get('state')
    #     if state in ['JOB_STATE_PENDING', 'JOB_STATE_RUNNING']:
    #         print(f"Found a running job: {job_dict['displayName']} with state: {state}")
    #         return True
    # return False

def create_bucket_if_not_exists(bucket_name):
    """Creates a new bucket if it does not already exist."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        st.warning('No Bucket were found with the given name. Creating One')
        bucket = storage_client.create_bucket(bucket_name)
        st.success(f"Bucket {bucket_name} created.")
        return bucket
    else:
        st.info(f"Bucket {bucket_name} already exists.")
        return bucket_name

def retrive_buckets():
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
# def check_running_jobs(project_id,region):
#     ''' Returns credential data from SSO default google authentication
#
#     args: string storage , bucket name , blob name
#     returns: json
#     '''
#     # bucket = storage_client.bucket(bucket_name[-1])
#     client = aiplatform.JobServiceClient(client_options={
#         'api_endpoint': f'{region}-aiplatform.googleapis.com'
#     })
#
#     parent = f'projects/{project_id}/locations/{region}'
#     jobs = client.list_custom_jobs(parent=parent)
#
#     for job in jobs:
#         job_dict = MessageToDict(job._pb)
#         state = job_dict.get('state')
#         if state in ['JOB_STATE_PENDING', 'JOB_STATE_RUNNING']:
#             print(f"Found a running job: {job_dict['displayName']} with state: {state}")
#             return True
#     return False

#Module testing function
def get_endpoints():
    '''Returns endpoint object for model in vertex ai

    args:None
    returns: endpoint object
    '''
    # bucket = storage_client.bucket(bucket_name[-1])
    endpoint = aiplatform.Endpoint.list(filter="display_name=Churn_model_endpoint")
    endpoint_client = aiplatform.Endpoint(endpoint[0].name)
    return endpoint_client
    # except:
    #     with st.form("Could Not find any endpoint for prediction.",clear_on_submit=True):
    #         option = st.radio('Do you want to train new model with data?',
    #                  ('Yes','No'))
    #         submit = st.form_submit_button()
    #         if submit and option == 'Yes':
    #             st.session_state.page = "main"
    #             return
    #         elif submit and option == 'No':
    #             st.info('No prediction endpoints were found select retrain option and upload data to train'
    #                     )
    #             return None



def login_gcloud():
    ''' Returns credentials and endpoints

    args: None
    returns: credentials json , endpoints json
    '''
    with st.spinner("Please wait while we intialize GCloud..."):
        os.system("gcloud auth application-default login")

    with st.spinner("Retrieving Cloud Credentials"):
        storage_client = storage.Client()
        buckets = retrive_buckets()
        credentials = get_credentials(storage_client, buckets)
        endpoint = get_endpoints()

    return credentials , endpoint

