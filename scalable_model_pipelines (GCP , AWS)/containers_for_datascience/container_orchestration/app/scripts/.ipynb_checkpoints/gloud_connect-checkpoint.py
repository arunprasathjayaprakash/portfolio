from google.cloud import aiplatform


dataset = aiplatform.TabularDataset('projects/my-project/location/us-central1/datasets/{DATASET_ID}')

job = aiplatform.AutoMLTabularTrainingJob(
  display_name="train-automl",
  optimization_prediction_type="classification",
  optimization_objective="-",
)

model = job.run(
    dataset=dataset,
    target_column="churn",
    training_fraction_split=0.6,
    validation_fraction_split=0.2,
    test_fraction_split=0.2,
    budget_milli_node_hours=1000,
    model_display_name="Churn_model",
    disable_early_stopping=False,
)
