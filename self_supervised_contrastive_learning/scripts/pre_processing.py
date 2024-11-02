import os

from torchvision.transforms import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def transform_data(x):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor()
    ])

    return transform(x)


def load_data(default=True):
    if default:
        train_dataset = datasets.CIFAR10(root=os.path.join(os.path.dirname(os.getcwd()),'data'), train=True, transform=transform_data, download=True)
    else:
        train_dataset = datasets.CIFAR10(root=os.path.join(os.path.dirname(os.getcwd()),'data'), train=True, download=True)

    test_dataset = DataLoader(train_dataset, batch_size=64, shuffle=True)

    return train_dataset, test_dataset