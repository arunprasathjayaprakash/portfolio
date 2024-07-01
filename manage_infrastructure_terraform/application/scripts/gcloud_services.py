from google.cloud import aiplatform
def create_dataset_artifact(bucket_name , file_name,display_name):
    '''Creates and Returns tabular datset artifact from gcp bucket

    args: Bucket name , file name , display nam
    returns: tabular dataframe
    '''
    data_path = f'gs://{bucket_name}/{file_name}'
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
def deploy(model,deploy_name):
    '''Returns deployed endpoint from deployment endpooint

    args: trained model , deployment name
    return: deployment endpoint
    '''
    DEPLOYED_NAME = f"{deploy_name}"
    endpoint = model.deploy(deployed_model_display_name=DEPLOYED_NAME)

    return endpoint