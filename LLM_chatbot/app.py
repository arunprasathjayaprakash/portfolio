import streamlit as st

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
        cluster_box = st.checkbox("Clustering")

    with col2:
        data_insight = st.checkbox("Data Insights")


    if cluster_box:
        with login_holder.form('login input',clear_on_submit=True):
            login_option = st.text_input()
            login_submit = st.form_submit_button()

        if login_submit and login_option == 'Yes':
            credentials , endpoint = login_gcloud()
            st.info("Successfully logged in GCloud")
            login_holder.empty()

    if data_insight:
        # Write your own custom model training here and use deployment functions available in scripts
        # Model select box, activated only if custom training is selected
        model_value = st.selectbox(
            'Select Which model of your choice',
            ['Decision Tree', 'Random Forest', 'XGBoost'],
        )
        st.info("You can write your own training here. Head to the code section and explore the code")


    #
    # with st.form('File Input', clear_on_submit=True):
    #     uploaded_file = st.file_uploader('Upload Your Test or Training data (Retraining Needs to be Enabled for trainig data)',
    #                                      type='csv')
    #     submit = st.form_submit_button()
    #
    #
    # if uploaded_file:
    #     data = pd.read_csv(uploaded_file)
    #     with st.form('Input',clear_on_submit=False):
    #         drop_column = st.multiselect("Which columns do you want to drop?",options=data.columns)
    #         fillna_selection = st.selectbox('Do you want to fill null values or drop null values?',['Fillna','DropNa'])
    #         target_column = st.selectbox("Select target column",options=data.columns)
    #         if fillna_selection == 'DropNA':
    #             fillna_selection = False
    #         st.write('Copy paste your key value pair (Optional)')
    #         st.text_input('Enter Key value Pairs for test data')
    #         st.info('Select checkbox to retrain')
    #         retrain_option = st.checkbox('Retrain model')
    #         bucket_name = st.text_input('Enter a valid bucket name in GCS')
    #         input_submit = st.form_submit_button()
    #
    #         if input_submit:
    #             #check for retrain_model
    #             if retrain_option and uploaded_file!= None:
    #                 try:
    #                     with st.spinner("Please Wait while we retrain the model...."):
    #                         processed_data = process_data(data,drop_column ,target_column, None,
    #                                                       fillna=fillna_selection,train=True)
    #                         dataset = create_dataset_artifact(bucket_name,processed_data,
    #                                                           uploaded_file.name,uploaded_file.name.split('.')[0]+'.json',project_id)
    #                         job , dataset = initialize_job(dataset,'classification')
    #                         current_training_jobs , finished_pipelines = check_running_jobs(project_id,
    #                                                                                         'us-central1')
    #                         if not current_training_jobs:
    #                             with st.spinner("Model Training Now...."):
    #                                 model = train_model(job , dataset)
    #                         else:
    #                             st.info("There are current running retraining jobs,Try again later once jobs have completed")
    #
    #                         with st.spinner("Model Has been trained. Deploying to endpoint now"):
    #                             deploy(model,'churn_model')
    #
    #                     st.info("Successfully trained and deployed to the endpoint. Please upload the test data for prediction")
    #                     return
    #                 except Exception as e:
    #                     st.error(
    #                         e
    #                     )
    #                     return
    #             else:
    #                 with st.spinner('Retriving transformation information from GCP. Please Wait..'):
    #                     existing_buckets = retrive_buckets()
    #                     transformation_columns = download_json_from_gcs(existing_buckets,
    #                                                                     'customer_churn_dataset.json')
    #                 processed_test_data = process_data(data,drop_column ,target_column,transformation_columns,fillna=fillna_selection)
    #                 with st.spinner("Predictions are underway"):
    #                     endpoint_info = get_endpoints()
    #                     if endpoint_info:
    #                         instance = processed_test_data.astype(str).to_dict(orient='records')
    #                         # instances = [instance]
    #                         predictions = endpoint_info.predict(instances=instance)
    #                         score_index = [v.index(max(v)) for values in predictions.predictions for k, v in values.items()
    #                                        if k == 'scores']
    #                         classes = [v for values in predictions.predictions for k, v in values.items() if k == 'classes']
    #                         predicted_classes = [classes[idx][values] for idx, values in enumerate(score_index)]
    #                         display_data = data.drop('Churn',axis=1)
    #                         display_data['Predicted_Churn'] = predicted_classes
    #                         display_data['Predicted_Churn'] = display_data['Predicted_Churn'].map(
    #                             {'0.0': 'No Churn', '1.0': "Churn"})
    #                         styled_df = display_data.style.applymap(highlight_churn, subset=['Predicted_Churn'])
    #                         st.dataframe(styled_df)
    #                     else:
    #                         st.error("Please deploy a model to make predictions")

if __name__ == "__main__":
    app()