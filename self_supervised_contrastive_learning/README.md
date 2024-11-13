# Contrastive Learning with SimCLR-Inspired Approach

This project demonstrates a contrastive learning approach inspired by the SimCLR framework. Using self-supervised learning, the model learns useful image representations by contrasting augmented views of images. The code is implemented in PyTorch and uses the CIFAR-10 dataset as a sample dataset.

```markdown
## Project Structure

- data/: Contains the CIFAR-10 dataset (downloaded automatically).
- scripts/: Contains Python scripts for model training and evaluation.
    - `train.py`: Main training script for contrastive learning.
    - `model.py`: Defines the feature extractor model (SimpleCNN) used for embedding images.
    - `pre-processing.py`: Contains the Transformation pipeline class.
    - `inference.py`: Will contain code for inference

## Requirements

Make sure to install the necessary Python packages:

```bash
pip install torch torchvision
```

## How to Run the Project

1. **Dataset**: The CIFAR-10 dataset will be automatically downloaded to the `data` folder.

2. **Training**: Run the main script to start training the model using contrastive learning.

   ```bash
   python scripts/train.py
   ```

   This script will:
   - Load and augment the CIFAR-10 dataset.
   - Train a CNN model to learn embeddings using contrastive loss.

3. **Fine-Tuning** (Optional): After training, you can fine-tune the model on a labeled dataset for tasks like image classification.

## Code Explanation

- **Model Architecture**:
  The model uses a simple CNN (`SimpleCNN`) as a backbone to extract 128-dimensional embeddings for each image. The model architecture can be found in `scripts/model.py`.

- **Contrastive Loss**:
  We implement the contrastive loss function to bring similar pairs (positive pairs) closer together in the embedding space while pushing dissimilar pairs (negative pairs) farther apart. The loss function is defined in `scripts/contrastive_loss.py`.

- **Data Augmentation**:
  Each image in the dataset is randomly augmented to generate two different views, which act as positive pairs. These augmentations include random resized cropping, horizontal flipping, and color jitter.
```
