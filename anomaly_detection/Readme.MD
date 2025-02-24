# Anomaly Detection

This repository contains scripts for performing anomaly detection. optionally has jupyter files for prototyping and understanding

Note: Refer .ipynb files for implmentation and conceptual understanding. Run scripts for streamlit application which follows software development standards

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
    or

    ```bash
    app.py
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