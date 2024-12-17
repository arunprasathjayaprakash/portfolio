import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from evaluate_model import evaluate

def adverserial_images(model,
                         loss,
                         optimizer,
                         train_data,
                         test_data):
    
    art_model = PyTorchClassifier(
        input_shape=(3,32,32),
        model= model,
        loss= loss,
        optimizer=optimizer,
        nb_classes=10
    )

    #Scaling Train images
    scaled_train = (train_data.data.transpose(0, 3, 1, 2)[0:1000]/255)
    scaled_train_targets =  np.array(train_data.targets)

    #Scaling Test Images
    scaled_test = test_data.data.transpose(0, 3, 1, 2)[0:1000]/255
    scaled_test_targets = np.array(test_data.targets)

    #performing adversal attacks
    adversal_func = FastGradientMethod(art_model,norm=1)
    train_attack = adversal_func.generate(scaled_train,scaled_train_targets)

    test_attack = adversal_func.generate(scaled_test,scaled_test_targets)

    #evaluate model
    metrics = evaluate(art_model,
             test_attack,
             scaled_test_targets)
    
    print(f"Adverserial Metrics: {metrics}")

    return metrics


if __name__ == "__main__":

    import os
    import torchvision.datasets
    from sklearn.model_selection import train_test_split
    from pre_process import transfomations

    train_data = torchvision.datasets.CIFAR10(os.getcwd(),download=True,transform=transfomations)
    test_data = torchvision.datasets.CIFAR10(os.getcwd(),download=True,train=False,transform=transfomations(train=False))
    adverserial_images()
    