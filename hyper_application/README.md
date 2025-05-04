# Portfolio Projects

This repository contains multiple machine learning and deep learning projects hosted under one Streamlit application to host and showcase projects in one application. Each project is self-contained and includes its own scripts, Dockerfiles, and dependencies. Below is an overview of the projects included in this repository.

---

## 1. **Adversarial Robustness**

### Description:
This project explores techniques to improve the robustness of machine learning models against adversarial attacks.

### Directory Structure:
```
adversarial_robustness/
├── scripts/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
```

### Setup Instructions:
1. Install dependencies using the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```
2. Build and run the Docker container:
   ```bash
   docker-compose up --build
   ```

---

## 2. **Anomaly Detection**

### Description:
Implements machine learning models to detect anomalies in data using statistical and machine learning approaches.

### Directory Structure:
```
anomaly_detection/
├── scripts/
│   ├── app.py
│   ├── stats_dt.py
│   ├── train.py
│   └── visual.py
```

### Setup Instructions:
1. Navigate to the `anomaly_detection` directory.
2. Run the `train.py` script to train the anomaly detection model:
   ```bash
   python scripts/train.py
   ```
3. Use `app.py` to deploy the model or analyze results.

---

## 3. **Automating Contracts**

### Description:
Automates contract-related processes, including data extraction and document generation.

### Directory Structure:
```
automating_contracts/
├── data/
├── refered_publications/
├── scripts/
├── docker-compose.yml
├── Dockerfile
├── README.MD
├── requirements.txt
```

### Setup Instructions:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the automation scripts available in the `scripts/` folder.

---

## 4. **Churn Prediction**

### Description:
Builds machine learning models to predict customer churn using structured datasets.

### Directory Structure:
```
churn_prediction/
├── scripts/
├── docker-compose.yml
├── Dockerfile
├── README.MD
```

### Setup Instructions:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the churn prediction model by running the scripts.

---

## 5. **Self-Supervised Contrastive Learning**

### Description:
Explores contrastive learning techniques, including implementations of NT-Xent (Normalized Temperature-Scaled Cross-Entropy Loss).

### Directory Structure:
```
self_supervised_contrastive_learning/
├── scripts/
├── docker-compose.yml
├── Dockerfile
├── NT-Xent Loss Explanation.pdf
├── README.md
├── requirements.txt
```

### Setup Instructions:
1. Build the Docker container:
   ```bash
   docker-compose up --build
   ```
2. Run the training scripts for contrastive learning available in the `scripts/` folder.

---

## 6. **YOLO Object Detection**

### Description:
Implements object detection using the YOLO (You Only Look Once) architecture.

### Directory Structure:
```
yolo_object_detection/
├── data/
├── input_data/
├── models/
├── output/
├── scripts/
├── docker-compose.yml
├── Dockerfile
├── path_file.json
├── deployments.yaml
```

### Setup Instructions:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure the deployment using `deployments.yaml` or `path_file.json`.
3. Run the YOLO object detection pipeline via the `scripts` folder.

---

## General Notes:
- Each project contains its own `requirements.txt` file for managing dependencies.
- Docker support is provided for easy containerization and deployment.
- Kubenetes scripts for deploying in GKS is available in project main directory
- GKS needs gcloud credentials in OS environment and set up with "gcloud auth login" for the scripts to dynamically retrive details
- GKS config files template can be created with the "generate_deployment_config.py" files
