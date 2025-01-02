# Projects Overview

This repository contains several projects organized into directories. Each project has its own docker file for containerization and the containers are deployed with Google kubernetes Service  using GKE deployment script. 

## 1. `adversarial_robustness`
This folder contains projects and code related to evaluating and improving the robustness of machine learning models against adversarial attacks.

### Key Features:
- Adversarial attack implementations (e.g., FGSM, PGD).
- Robustness evaluation techniques.
- Defenses and mitigation strategies.

---

## 2. `anomaly_detection` (In development)
This folder includes projects on identifying anomalies in datasets using advanced machine learning and deep learning techniques.

### Key Features:
- XGBoost for anomaly detection.
- Autoencoders and GAN-based methods.
- GPU-accelerated model training.

---

## 3. `automating_contracts`
This folder focuses on automating the processing and analysis of contracts using machine learning and natural language processing (NLP).

### Key Features:
- Automated extraction of key contract terms.
- Legal text classification.
- Tools for contract digitization.

---

## 4. `churn_prediction`
This folder contains a real-time churn prediction project that predicts vendor churn in various domains.

### Key Features:
- Churn prediction using Vertex AI.
- Streamlit UI for monitoring and retraining.
- Explainability and data drift detection.

---

## 5. `self_supervised_contrastive_learning`
This folder includes projects implementing self-supervised learning methods inspired by frameworks like SimCLR.

### Key Features:
- Learning image representations via contrastive learning.
- Self-supervised training on the CIFAR-10 dataset.
- PyTorch implementation.

---

## 6. `yolo_object_detection`
This folder contains YOLO-based object detection projects for real-time detection tasks.

### Key Features:
- YOLO model training and evaluation.
- Custom dataset preparation.
- Real-time detection demos.

---

## How to Use
1. Clone this repository and run through the code with docker-compose:
   ```bash
   git clone https://github.com/arunprasathjayaprakash/portfolio.git
