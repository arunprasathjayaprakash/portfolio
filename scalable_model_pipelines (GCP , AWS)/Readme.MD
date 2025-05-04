# Model Pipelines at Scale

This repository contains workflow implementations for data science workflows that should be used for production pipelines with various options for running production grade pipelines.

## Folder Structure

### `containers_for_datascience`
Contains Dockerfiles and container orchestrations to set up containers for running data science workflows in production environment. 
These containers may include pre-installed libraries and tools required for the project.

### `models_as_endpoints`
Contains scripts and configurations to deploy models as REST API endpoints, enabling integration with other applications.

### `models_serverless_functions (GCP)`
Holds serverless functions implemented on Google Cloud Platform (GCP) for model deployment or other serverless workflows.

### `serverless_functions (AWS)`
Contains serverless functions implemented on Amazon Web Services (AWS), typically using services like AWS Lambda, for handling lightweight tasks or deployments.

## Usage
1. **Setup**: Start by setting up the environment using the configurations in the `containers_for_datascience` folder.

2. **Deploy Models**:
   - Use `models_as_endpoints` for REST API-based deployments.
   - Use `models_serverless_functions (GCP)` or `serverless_functions (AWS)` for serverless cloud deployments.
3. **Manage Credentials**: Place secure access keys in the `credentials` folder and ensure proper security practices.

## Notes
- Always follow best practices for securing sensitive information.
- Ensure all dependencies and configurations are correctly set up before running any code.
- Refer to individual folders for more detailed documentation.
- Refer .IPYNB files for quick prototyping modules
