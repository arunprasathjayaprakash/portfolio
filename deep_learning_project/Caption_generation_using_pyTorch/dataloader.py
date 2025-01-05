import torch
from torchvision.io import read_image
import os
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels_dir, transform):
        self.image_dir = image_dir
        self.total_images = len(os.listdir(image_dir))
        self.labels = labels_dir
        self.transform = transform

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        image = read_image(self.image_dir[idx])
        labels = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, labels

    def get_images(self):
        pass