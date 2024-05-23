# Adversarial Robustness of Neural Networks

This repository contains a Jupyter notebook that demonstrates how to evaluate and improve the adversarial robustness of neural networks using various techniques, including adversarial attacks and training with adversarial examples. The notebook utilizes popular libraries such as PyTorch, timm, and the Adversarial Robustness Toolbox (ART).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Notebook Contents](#notebook-contents)
  - [Task 1: Import Libraries](#task-1-import-libraries)
  - [Task 2: Load the Pretrained Model](#task-2-load-the-pretrained-model)
  - [Task 3: Load the Dataset](#task-3-load-the-dataset)
  - [Task 4: Perform Adversarial Attacks](#task-4-perform-adversarial-attacks)
  - [Task 5: Visualize Adversarial Images](#task-5-visualize-adversarial-images)
  - [Task 6: Evaluate Model Performance on Adversarial Images](#task-6-evaluate-model-performance-on-adversarial-images)
  - [Task 7: Evaluate Model Performance on Normal Images](#task-7-evaluate-model-performance-on-normal-images)
  - [Task 8: Train a Classifier to Detect Adversarial Examples](#task-8-train-a-classifier-to-detect-adversarial-examples)
  - [Task 9: Identify Adversarial Examples From Normal Examples](#task-9-identify-adversarial-examples-from-normal-examples)
  - [Task 10: Train the Model With Adversarial Examples](#task-10-train-the-model-with-adversarial-examples)
  - [Task 11: Evaluate Model's Performance After Adversarial Training](#task-11-evaluate-models-performance-after-adversarial-training)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this notebook, you will need to install the following dependencies:

```bash
pip install pandas numpy opencv-python matplotlib timm adversarial-robustness-toolbox torch torchvision
```

Ensure you have the necessary files in the `/files` directory, including pretrained model weights and adversarial detection models.

## Usage

1. Clone this repository:

```bash
git clone https://github.com/arunprasathjayaprakash/portfolio/tree/266b3c1a356f614c16c016343a28ec76092295fc/adversial_robustness
cd adversarial-robustness
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:

```bash
jupyter notebook
```

Open `Adversarial_Robustness.ipynb` in the Jupyter interface and follow the instructions in the notebook.

## Notebook Contents

### Task 1: Import Libraries

Import necessary libraries and set up the environment for the tasks ahead.

### Task 2: Load the Pretrained Model

Load a pretrained ResNet-34 model, modify its architecture, and load custom weights.

### Task 3: Load the Dataset

Load and transform the CIFAR-10 dataset for training and testing.

### Task 4: Perform Adversarial Attacks

Create adversarial examples using the Fast Gradient Method (FGM).

### Task 5: Visualize Adversarial Images

Display original and adversarial images to understand the differences.

### Task 6: Evaluate Model Performance on Adversarial Images

Evaluate the model's performance on adversarial examples.

### Task 7: Evaluate Model Performance on Normal Images

Evaluate the model's performance on the original dataset.

### Task 8: Train a Classifier to Detect Adversarial Examples

Train a binary classifier to distinguish between adversarial and normal examples.

### Task 9: Identify Adversarial Examples From Normal Examples

Use the trained classifier to detect adversarial examples.

### Task 10: Train the Model With Adversarial Examples

Train the model with both normal and adversarial examples to improve robustness.

### Task 11: Evaluate Model's Performance After Adversarial Training

Evaluate the model's performance after incorporating adversarial training.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.