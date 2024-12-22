#Load the Transformations pipeline
from torchvision.transforms import v2

def transfomations(train=True):
    if train:
        train_transformations = v2.Compose([
            v2.RandomCrop(size=(512,512)),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return train_transformations
    else:

        test_transformations = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return test_transformations