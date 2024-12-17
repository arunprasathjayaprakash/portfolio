import streamlit as st
import pandas as pd
import json
import os
from gcloud_connect import get_endpoints , login_gcloud , check_running_jobs
from gcloud_services import (create_dataset_artifact , initialize_job , train_model , deploy , create_endpoint ,
                             deploy_model_to_endpoint,download_json_from_gcs)
from google.cloud import aiplatform_v1
from etl_pipe import process_data
from gcloud_connect import retrive_buckets

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
        st.info("You can write your own training here. Head to the code section and explore the code")


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
                            processed_data = process_data(data,drop_column ,target_column, None,
                                                          fillna=fillna_selection,train=True)
                            dataset = create_dataset_artifact(bucket_name,processed_data,
                                                              uploaded_file.name,uploaded_file.name.split('.')[0]+'.json',project_id)
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
                    with st.spinner('Retriving transformation information from GCP. Please Wait..'):
                        existing_buckets = retrive_buckets()
                        transformation_columns = download_json_from_gcs(existing_buckets,
                                                                        'customer_churn_dataset.json')
                    processed_test_data = process_data(data,drop_column ,target_column,transformation_columns,fillna=fillna_selection)
                    with st.spinner("Predictions are underway"):
                        endpoint_info = get_endpoints()
                        if endpoint_info:
                            instance = processed_test_data.astype(str).to_dict(orient='records')
                            # instances = [instance]
                            predictions = endpoint_info.predict(instances=instance)
                            score_index = [v.index(max(v)) for values in predictions.predictions for k, v in values.items()
                                           if k == 'scores']
                            classes = [v for values in predictions.predictions for k, v in values.items() if k == 'classes']
                            predicted_classes = [classes[idx][values] for idx, values in enumerate(score_index)]
                            display_data = data.drop('Churn',axis=1)
                            display_data['Predicted_Churn'] = predicted_classes
                            display_data['Predicted_Churn'] = display_data['Predicted_Churn'].map(
                                {'0.0': 'No Churn', '1.0': "Churn"})
                            styled_df = display_data.style.applymap(highlight_churn, subset=['Predicted_Churn'])
                            st.dataframe(styled_df)
                        else:
                            st.error("Please deploy a model to make predictions")

if __name__ == "__main__":
    app()