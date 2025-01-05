# Containerized Application Deployment on Google Cloud Platform (GCP)

This project demonstrates how to streamline containerized application in our case the Churn prediction with streamlit
deployment on the Google Cloud Platform (GCP) using a combination of CLI and Terraform scripts using Google Cloud Run.

Note: Credentials.JSON , Service Account Details and terraform plan files are not included.Create
files with GCP account during runtime

## Table of Contents
- [Prerequisites](#prerequisites)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Project Tasks](#project-tasks)
  - [Getting Started](#getting-started)
  - [Build Infrastructure as Code (IaC)](#build-infrastructure-as-code-iac)
  - [Deploying Application to Cloud Run](#deploying-application-to-cloud-run)
- [Skills](#skills)
- [Setup Instructions](#setup-instructions)

## Prerequisites
- A GCP account with billing enabled
- Understanding (Intermediate level) of Terraform
- Understanding of gcloud CLI
- Understanding of clusters (GKE) , storage buckets , Dockers 

## Technologies
- [Terraform](https://www.terraform.io/)
- [Cloud Run](https://cloud.google.com/run)
- [Kubernetes](https://kubernetes.io/)
- [gCloud CLI](https://cloud.google.com/sdk/gcloud)
- [Artifact Registry](https://cloud.google.com/artifact-registry)

## Project Structure
````
manage_infrastructure_terraform/
├── application/
│ ├── data/
│ ├── scripts/
    │ ├── .ipynb_checkpoints/
    │ ├── app.py
    │ ├── etl_pipe.py
    │ ├── gcloud_connect.py
    │ ├── gcloud_services.py
│ ├── docker_commands.txt
│ ├── Dockerfile
│ └── requirements.txt
│
├── command_references.txt
├── main.tf
├── output.tf
├── plan_config
├── README.md
└── variables.tf
````

## Skills
- Cloud Deployment
- Docker Containers
- DevOps

## Setup Instructions
1. **Clone the Repository**
    ```sh
    git clone https://github.com/arunprasathjayaprakash/portfolio/tree/b579b694fa86b5902fdcefb94dffc8b3eca91316/manage_infrastructure_terraform
    cd manage_infrastructure_terraform
    ```

2. **Install Dependencies**
- Ensure you have Docker, Terraform, and the gcloud CLI installed. Install Python dependencies:
    ```sh
    pip install -r application/scripts/requirements.txt
    ```

3. **Authenticate gcloud**
    ```sh
    gcloud auth login
    gcloud auth application-default login
    ```

4. **Initialize Terraform**
    ```sh
    terraform init
    ```

5. **Apply Terraform Configuration**
    ```sh
    terraform apply <config file>
    ```

6. **Build and Push Docker Image**
    ```sh
    docker build -t gcr.io/<your-project-id>/your-app:latest -f application/scripts/Dockerfile .
    docker push gcr.io/<your-project-id>/your-app:latest
    ```

7. **Deploy on Cloud Run**
    - Update the terraform configuration file to include google cloud run as resource
    - Plan and apply terraform configuration
   ```sh
    terraform apply <config file>
    ```


8. **Clean Up Resources**
    ```sh
    terraform destroy
    ```

## Note
This project will incur minimal costs on your GCP account. Make sure to destroy all resources once you are done to 
avoid unnecessary charges or restrict usages under GCP administration.
