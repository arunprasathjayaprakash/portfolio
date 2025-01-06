# Anomaly Detection

This repository contains a Jupyter notebook for performing anomaly detection for various image and tabular data.
Detecting anomaly with XGboost for detecting network intrutions and other is an CNN based malware classfication that 
detects malware in .exe files by converting it into Grayscale images.

Note: Refer .ipynb files for implmentation and conceptuall understanding. Run scripts for streamlit application which follows software development standards

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)

## Installation

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Open the Jupyter notebook:

    ```bash
    jupyter notebook Image_CNN.ipynb
    jupyter notebook XGBoost.ipynb
    ```

2. Follow the instructions in the notebook to:
    - Load and preprocess the data.
    - Define and train the autoencoder model.
    - Use the trained model to detect anomalies.
    - Visualize the results.

## Features

- **Data Preprocessing**: Load and preprocess data for training and testing.
- **Model Training**: Define and train an autoencoder model.
- **Anomaly Detection**: Detect anomalies using the trained autoencoder model.
- **Visualization**: Visualize the reconstruction error and detected anomalies.

## Project Structure

```
anomaly_detection_with_auto_encoders/
│                 
├──scripts/ 
│   - notebook Image_CNN.ipynb
│   - notebook Xgboost.ipynb
├── requirements.txt         # Python dependencies
├── README.md                # Project README file
```

---

## Detailed Code Description

### `Xgboost.ipynb  - Binary and multiclass`

The main notebook file that demonstrates anomaly detection using autoencoders.

- **Data Loading and Preprocessing**:
  - Load the dataset.
  - Preprocess the data (normalization, splitting into training and test sets).

- **Model Definition**:
  - Initialize Xgboost with Nvidia Rapids.
  - Utilize DMatrix for faster higher accurate model

- **Model Training**:
  - Compile and train the autoencoder model.
  - Save the trained model.

- **Anomaly Detection**:
  - Detect the type of anomaly with Xgboost model.

### `Image_CNN.ipynb - Detect and Classify malware in .exe files

The main notebook file that demonstrates anomaly detection using autoencoders.

- **Data Loading and Preprocessing**:
  - Load the dataset.
  - Preprocess the data (normalization, splitting into training and test sets).

- **Model Definition**:
  - Initialize CNN with Pytorch and simpler layers

- **Model Training**:
  - Compile and train the PyTorch model.
  - Save the trained model.

- **Anomaly Detection**:
  - Detect the type of anomaly with Pytorch model.
