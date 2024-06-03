# Adversarial Robustness of Neural Networks

This repository contains an implementation for evaluating and improving the adversarial robustness of neural networks using PyTorch and the Adversarial Robustness Toolbox (ART).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Tasks Overview](#tasks-overview)
- [Contributing](#contributing)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/arunprasathjayaprakash/portfolio/tree/3d42c9587706f96dc323f93e7cd95728fdb35ff3/adversial_robustness
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Load the Jupyter notebook:

    ```bash
    jupyter notebook Adverserial_robustness.ipynb
    ```

2. Follow the steps in the notebook to:
    - Import necessary libraries.
    - Load and modify pretrained models.
    - Load and preprocess datasets.
    - Perform adversarial attacks.
    - Visualize adversarial images.
    - Evaluate model performance on adversarial and clean images.
    - Train a classifier to detect adversarial examples.
    - Evaluate model's performance after adversarial training.

## Project Structure

```
Adversarial_Robustness/
│
├── files/                   # Directory containing model weight files
├── Adverserial_robustness.ipynb  # Jupyter notebook for adversarial robustness tasks
├── requirements.txt         # Python dependencies
└── README.md                # Project README file
```

## Tasks Overview

### Task 1: Import Libraries

Import necessary libraries and check the current working directory.

### Task 2: Load the Pretrained Model

Load the pretrained ResNet-34 model and modify it with custom weights.

### Task 3: Load the Dataset

Load and preprocess the CIFAR-10 dataset using torchvision and sklearn.

### Task 4: Perform Adversarial Attacks

Use the Fast Gradient Method (FGM) to generate adversarial examples.

### Task 5: Visualize Adversarial Images

Display adversarial images to visualize the impact of the attacks.

### Task 6: Evaluate Model Performance on Adversarial Images

Evaluate the model's performance on adversarial images to assess its robustness.

### Task 7: Evaluate Model Performance on Normal Images

Evaluate the model's performance on clean images for comparison.

### Task 8: Train a Classifier to Detect Adversarial Examples

Train a binary classifier to detect adversarial examples from clean images.

### Task 9: Identify Adversarial Examples From Normal Examples

Use the trained classifier to distinguish between adversarial and clean images.

### Task 10: Train the Model With Adversarial Examples

Use adversarial training to improve the model's robustness against attacks.

### Task 11: Evaluate Model's Performance After Adversarial Training

Assess the model's performance on clean and adversarial images after adversarial training.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

