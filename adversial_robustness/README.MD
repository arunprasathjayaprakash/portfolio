
---

# Adversarial Robustness of Neural Networks

Note: This work is a reflection to explain the understanding of the concept. Models used were pretrained models to explain the importance of adversarial attacks that can affect models in this Generative AI Era.


This repository demonstrates how to implement and evaluate the robustness of neural networks against adversarial attacks using a Jupyter Notebook. The notebook covers loading a pretrained model, performing adversarial attacks, visualizing adversarial images, evaluating model performance, and training a classifier to detect adversarial examples.

## Business Problem

Adversarial attacks on neural networks can cause misclassification in various applications such as financial services, healthcare, autonomous vehicles, and cybersecurity. Understanding and mitigating these attacks can improve the robustness and reliability of machine learning models.

## Tasks and Implementation

### Task 1: Import Libraries

We import necessary libraries including `pandas`, `numpy`, `cv2`, `matplotlib`, `timm` for model handling, `art` for adversarial robustness, and `torch` for deep learning operations.

### Task 2: Load the Pretrained Model

We load a pretrained ResNet34 model using the `timm` library, then modify its first convolutional layer and fully connected layer to match custom weights. This prepares the model for further tasks such as adversarial attacks and evaluation.

### Task 3: Load the Dataset

We load the CIFAR-10 dataset, applying transformations such as random cropping, horizontal flipping, and normalization to the training set, and normalization to the test set. We then prepare data loaders for training and testing.

### Task 4: Perform Adversarial Attacks

We define the loss function and optimizer, then create an ART `PyTorchClassifier` for the ResNet34 model. We scale the training and test images, and use the Fast Gradient Method (FGM) from ART to generate adversarial examples for both datasets.

### Task 5: Visualize Adversarial Images

We display a few original and adversarial test images using the PIL library to visually inspect the effects of the adversarial attack.

### Task 6: Evaluate Model Performance on Adversarial Images

We predict the labels for the adversarial test images using the ART classifier and compute the accuracy to evaluate the model's performance under adversarial conditions.

### Task 7: Evaluate Model Performance on Normal Images

We similarly predict the labels for the normal test images and compute the accuracy to compare the model's performance on clean data.

### Task 8: Train a Classifier to Detect Adversarial Examples

We create and modify a ResNet18 model to serve as a binary classifier to detect adversarial examples. We generate a combined dataset of adversarial and clean images, label them accordingly, and train the binary classifier using ART's `BinaryInputDetector`.

### Task 9: Identify Adversarial Examples From Normal Examples

We use the trained binary classifier to differentiate between adversarial and normal examples, assessing the classifier's ability to detect adversarial attacks.

### Task 10: Train the Model With Adversarial Examples

We create another ResNet18 model and load custom weights. We then use ART's `AdversarialTrainer` to train this model on a combined dataset of adversarial and clean images to improve its robustness against adversarial attacks.

### Task 11: Evaluate Model's Performance After Adversarial Training

Finally, we evaluate the performance of the adversarially trained model on the test dataset to verify the improvement in accuracy and robustness post-adversarial training.
---