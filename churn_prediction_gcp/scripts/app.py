import streamlit as st
import pandas as pd
import json
import os
from gcloud_connect import get_endpoints , login_gcloud , check_running_jobs
from gcloud_services import (create_dataset_artifact , initialize_job , train_model , deploy , create_endpoint ,
                             deploy_model_to_endpoint)
from google.cloud import aiplatform_v1 ,aiplatform
from etl_pipe import process_data
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def highlight_churn(s):
    """
    Highlight cells in the 'Churn' column based on the condition
    """
    if s == 'Churn':
        return 'background-color: red'
    else:
        return 'background-color: green'

def app():
    '''Hosts streamlit application on default port

    args: None
    returns: None
    '''
    st.title('Churn Prediction')
    st.subheader('Analytics By Vertex AI')

    login_holder = st.empty()

    # Create three columns
    col1, col2, col3 , col4= st.columns(4)

    # Place widgets in each column
    with col1:
        loginbox = st.checkbox("Login to GCloud")

    with col2:
        custombox = st.checkbox("Custom Training")

    with col3:
        deploybox = st.checkbox("Deploy")

    with col4:
        undeploy = st.checkbox("Undo Deploy")

    try:
        if len(os.listdir(os.path.join(os.path.dirname(os.getcwd()),'credentials'))) != 0:
            credentials = json.load(open(os.path.join(os.path.dirname(os.getcwd()),'credentials/portfolio_project_key.json')))
            project_id = credentials['project_id']
            location_path = f"projects/{project_id}/locations/us-central1"
    except:
        st.error('Unable to find credentials file. Please check your credentials login , try the login option or '
                 'download the credentials file to credentials folder')

    if loginbox:
        with login_holder.form('login input',clear_on_submit=True):
            login_option = st.selectbox('Do you want to login to gcloud now?',options=['Yes','No'])
            login_submit = st.form_submit_button()

        if login_submit and login_option == 'Yes':
            credentials , endpoint = login_gcloud()
            st.info("Successfully logged in GCloud")
            login_holder.empty()

    if custombox:
        # Write your own custom model training here and use deployment functions available in scripts
        # Model select box, activated only if custom training is selected
        model_value = st.selectbox(
            'Select Which model of your choice',
            ['Decision Tree', 'Random Forest', 'XGBoost'],
        )


    if deploybox:
        try:
            with st.form('deploy_input',clear_on_submit=True):
                selection = st.radio('Do you want to deploy a pretrained model from gcs bucket?',options=['Yes','No'],index=1)
                submit = st.form_submit_button()

                if submit and selection == 'Yes':
                    with st.spinner("Deploying pretrained from your default bucket."):
                        client = aiplatform_v1.ModelServiceClient(
                            client_options={'api_endpoint': 'us-central1-aiplatform.googleapis.com'}
                        )
                        _ , finished_pipeline_model_id = check_running_jobs(project_id, 'us-central1')
                        model = client.get_model(name=finished_pipeline_model_id[0])
                        with st.spinner("Creating Endpoint.."):
                            endpoint = create_endpoint('churn_online_prediction',location_path)
                        with st.spinner(f"Deploying Model to endpoint ..{endpoint}"):
                            deploy_model_to_endpoint(model, endpoint)
                        st.info(f'Successfully deployed latest model to endpoint : {endpoint}')
        except Exception as e:
            st.error(
                e
            )

    if undeploy:
        st.info("Currently in development")
    #     try:
    #         with st.form('un_deploy_input',clear_on_submit=True):
    #             selection = st.radio('Do you want to remove deployment and delete endpoint ?',options=['Yes','No'],index=1)
    #             submit = st.form_submit_button()
    #
    #             if submit and selection == 'Yes':
    #                 with st.spinner("Deploying pretrained from your default bucket."):
    #                     client = aiplatform_v1.Endpoint()
    #                     _ , finished_pipeline_model_id = check_running_jobs(project_id, 'us-central1')
    #                     location_path = f"projects/{project_id}/locations/us-central1"
    #                     model = client.get_model(name=finished_pipeline_model_id[0])
    #                     with st.spinner("Creating Endpoint.."):
    #                         endpoint = create_endpoint('churn_online_prediction',location_path)
    #                     with st.spinner(f"Deploying Model to endpoint ..{endpoint}"):
    #                         deploy_model_to_endpoint(model, endpoint)
    #                     st.info(f'Successfully deployed latest model to endpoint : {endpoint}')
    #     except Exception as e:
    #         st.error(
    #             e
    #         )


    with st.form('File Input', clear_on_submit=True):
        uploaded_file = st.file_uploader('Upload Your Test or Training data (Retraining Needs to be Enabled for trainig data)',
                                         type='csv')
        submit = st.form_submit_button()


    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        with st.form('Input',clear_on_submit=False):
            drop_column = st.multiselect("Which columns do you want to drop?",options=data.columns)
            fillna_selection = st.selectbox('Do you want to fill null values or drop null values?',['Fillna','DropNa'])
            target_column = st.selectbox("Select target column",options=data.columns)
            if fillna_selection == 'DropNA':
                fillna_selection = False
            st.write('Copy paste your key value pair (Optional)')
            st.text_input('Enter Key value Pairs for test data')
            st.info('Select checkbox to retrain')
            retrain_option = st.checkbox('Retrain model')
            bucket_name = st.text_input('Enter a valid bucket name in GCS')
            input_submit = st.form_submit_button()

            if input_submit:
                #check for retrain_model
                if retrain_option and uploaded_file!= None:
                    try:
                        with st.spinner("Please Wait while we retrain the model...."):
                            processed_data = process_data(data,drop_column ,target_column,
                                                          fillna=fillna_selection,train=True)
                            dataset = create_dataset_artifact(bucket_name,processed_data,
                                                              uploaded_file.name,uploaded_file.name,project_id)
                            job , dataset = initialize_job(dataset,'classification')
                            current_training_jobs , finished_pipelines = check_running_jobs(project_id,
                                                                                            'us-central1')
                            if not current_training_jobs:
                                with st.spinner("Model Training Now...."):
                                    model = train_model(job , dataset)
                            else:
                                st.info("There are current running retraining jobs,Try again later once jobs have completed")

                            with st.spinner("Model Has been trained. Deploying to endpoint now"):
                                deploy(model,'churn_model')

                        st.info("Successfully trained and deployed to the endpoint. Please upload the test data for prediction")
                        return
                    except Exception as e:
                        st.error(
                            e
                        )
                        return
                else:
                    processed_test_data = process_data(data,drop_column ,target_column,fillna=fillna_selection)
                    with st.spinner("Predictions are underway"):
                        # endpoint_info = get_endpoints(project_id)
                        # instance = processed_test_data.astype(str).values.tolist()
                        # instance = processed_test_data.to_dict(orient='records')
                        # # instances = {key: value for key, value in processed_test_data.items()}
                        # # instances = [instance]
                        # endpoint = aiplatform.Endpoint(endpoint_info[0])
                        # # instance = json_format.ParseDict(instance, Value())
                        # instances = [instance]
                        # prediction = endpoint.predict(instances=instances)
                        # print(prediction)
                        # client = aiplatform.gapic.PredictionServiceClient(
                        #     client_options={'api_endpoint': 'us-central1-aiplatform.googleapis.com'}
                        # )
                        # instance = json_format.ParseDict(instance, Value())
                        # instances = [instance]
                        # endpoint = client.endpoint_path(
                        #     project=project_id, location='us-central1', endpoint=endpoint_info[0]
                        # )
                        # try:
                        #     response = client.predict(
                        #         endpoint=endpoint_info[0], instances=instances
                        #     )
                        # except Exception as e:
                        #     print(f"InvalidArgument error: {e}")
                        #     # Print more details if available
                        #     if hasattr(e, 'errors'):
                        #         for error in e.errors:
                        #             print(error)

                        # prediction = endpoint.predict(instances=instances)
                        # # print(prediction)
                        # prediction_client = aiplatform_v1.PredictionServiceClient(
                        #     client_options={'api_endpoint': 'us-central1-aiplatform.googleapis.com'}
                        # )
                        #
                        # if endpoint_info:
                        #     prediction_request = aiplatform_v1.PredictRequest(
                        #         endpoint=endpoint_info[0],
                        #         instances=instances[0]
                        #     )
                        # else:
                        #     endpoint_info = create_endpoint('default', location_path)
                        #     prediction_request = aiplatform_v1.PredictRequest(
                        #         endpoint=endpoint_info,
                        #         instances=instance
                        #     )
                        # predictions = prediction_client.predict(
                        #     request=prediction_request
                        # )
                        # score_index = [v.index(max(v)) for values in predictions.predictions for k,v in values.items() if k=='scores']
                        # classes = [v for values in predictions.predictions for k,v in values.items() if k=='classes']
                        # predicted_classes = [classes[idx][values] for idx,values in enumerate(score_index)]
                        # processed_test_data['Predicted_Churn'] = predicted_classes
                        # processed_test_data['Predicted_Churn'] = processed_test_data['Predicted_Churn'].map({'0.0':'No Churn','1.0':"Churn"})
                        # styled_df = processed_test_data.style.applymap(highlight_churn, subset=['Predicted_Churn'])
                        # st.dataframe(styled_df)
                        endpoint_info = get_endpoints()
                        if endpoint_info:
                            instance = processed_test_data.astype(str).to_dict(orient='records')
                            # instances = [instance]
                            predictions = endpoint_info.predict(instances=instance)
                            score_index = [v.index(max(v)) for values in predictions.predictions for k, v in values.items()
                                           if k == 'scores']
                            classes = [v for values in predictions.predictions for k, v in values.items() if k == 'classes']
                            predicted_classes = [classes[idx][values] for idx, values in enumerate(score_index)]
                            processed_test_data['Predicted_Churn'] = predicted_classes
                            processed_test_data['Predicted_Churn'] = processed_test_data['Predicted_Churn'].map(
                                {'0.0': 'No Churn', '1.0': "Churn"})
                            styled_df = processed_test_data.style.applymap(highlight_churn, subset=['Predicted_Churn'])
                            st.dataframe(styled_df)
                        else:
                            st.error("Please deploy a model to make predictions")

if __name__ == "__main__":
    app()