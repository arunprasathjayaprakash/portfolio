# Cloud Run Deployment and Prediction Functions

This repository contains a collection of scripts and tools to deploy and run prediction functions on Google Cloud Run. It includes dynamic GCS bucket storage handling, Cloud Function integrations, and deployment automation.

## Project Structure

- **`Deploying Cloud Run.png`**  
  A visual reference illustrating the Cloud Run deployment process.

- **`Dockerfile`**  
  Contains the configuration for building the Docker image for deploying the prediction function on Cloud Run.

- **`bucket_storage_gcs.py`**  
  Script to manage GCS bucket storage dynamically for prediction workflows.

- **`cloud_function_call.py`**  
  Functions to handle calls between Google Cloud Functions and Cloud Run services.

- **`command_references.txt`**  
  A guide with references to commands used for deploying, building, and interacting with the services.

- **`deploy_cloud_run.py`**  
  Automates the process of deploying the application to Cloud Run. Includes updates for dynamically managing GCS buckets and prediction functionality.

- **`gcs_bucket_download.py`**  
  Contains utility functions to handle downloads from Google Cloud Storage buckets, optimized for use within the Cloud Run prediction function.

## Key Features

- **Dynamic GCS Bucket Handling**: Scripts for efficient interaction with Google Cloud Storage (GCS) to manage and retrieve data dynamically.
- **Prediction Service Deployment**: Tools and configurations for deploying machine learning prediction services on Google Cloud Run.
- **Interoperability**: Facilitates communication between Cloud Functions and Cloud Run services.
- **Command References**: Easy-to-follow deployment and management instructions.

## Prerequisites

- [Docker](https://www.docker.com/)
- [Google Cloud CLI](https://cloud.google.com/sdk)
- Google Cloud Project with billing enabled
- IAM permissions to deploy Cloud Run and access GCS

## Setup and Usage

1. **Clone the repository**  
   ```bash
   git clone <repository-url>
   cd <repository-name>
