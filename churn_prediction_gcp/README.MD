# Churn Prediction App

This repository contains a Streamlit application for predicting customer churn using machine learning models. The app integrates with Google Cloud Platform services like Vertex AI for model training and prediction, and Google Cloud Storage for data retrieval and credential management.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/arunprasathjayaprakash/portfolio/tree/b96aa2b5db63771b43d6cb519ae714a7c522e9f1/churn_prediction_gcp
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up Google Cloud credentials. Ensure that you have the Google Cloud SDK installed and authenticated (skip this to login using UI):

    ```bash
    gcloud auth application-default login
    ```

## Usage

1. Start the Streamlit application:

    ```bash
    streamlit run Scripts/app.py
    ```

2. Use the Streamlit interface to:
    - Log in to Google Cloud.
    - Upload test data in CSV format.
    - Optionally, provide key-value pairs for the test data.
    - Choose to retrain the model and select a model type (Decision Tree, Random Forest, or XGBoost).
    - View the predictions and highlighted churn results.

## Features

- **Google Cloud Authentication**: Secure login to Google Cloud for accessing storage and Vertex AI services.
- **Data Upload**: Upload CSV files for test data.
- **Model Selection and Retraining**: Select machine learning models and opt to retrain them.
- **Data Processing**: Automated processing of input data.
- **Churn Prediction**: Display predictions with highlighted churn status.
- **Integration with Vertex AI**: Utilize Vertex AI for making predictions and potentially training models.

## Project Structure

```
churn-prediction-app/
│
├── Data/                    # Directory to store data files
├── Scripts/                 # Directory containing all scripts
│   ├── app.py               # Main Streamlit application file
│   ├── etl_pipe.py          # ETL (Extract, Transform, Load) pipeline for data processing
│   ├── gcloud_connect.py    # Google Cloud related functions
│   └── gcloud_services.py   # Additional Google Cloud services functions
├── requirements.txt         # Python dependencies
├── README.md                # Project README file

```

---

## Detailed Code Description

### `Scripts/app.py`

The main application file that uses Streamlit to create a web interface for churn prediction.

- **highlight_churn(s)**: Function to highlight cells in the 'Churn' column based on the condition.
- **app()**: Main function to render the Streamlit application.

### `Scripts/gcloud_connect.py`

Contains functions to interact with Google Cloud services.

- **get_credentials(storage_client, bucket_name, blob_name)**: Retrieve credentials from Google Cloud Storage.
- **retrive_connection()**: Retrieve the list of storage buckets.
- **get_endpoints()**: Retrieve endpoint information for model predictions.
- **login_gcloud()**: Authenticate and log in to Google Cloud.

### `Scripts/etl_pipe.py`

Handles the data processing pipeline.

- **process_data(data, train=False)**: Process input data for predictions or training.

### `Scripts/gcloud_services.py`

Contains additional Google Cloud service functions.

- **create_dataset_artifact(bucket_name, file_name, display_name)**: Create a dataset artifact in Vertex AI.
- **initialize_job(dataset, model_type)**: Initialize an AutoML training job.
- **train_model(job, dataset)**: Train a model using AutoML.
- **deploy(model, deploy_name)**: Deploy the trained model.