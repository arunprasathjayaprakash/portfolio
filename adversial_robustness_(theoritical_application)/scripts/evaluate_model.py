import torch
import numpy as np


def evaluate(model,test_data,test_targets):
    predictions = []
    from tqdm import tqdm
    for image in tqdm(test_data,total=len(test_data)):
        model_predict = model.predict(torch.from_numpy(image).unsqueeze(dim=0))
        predictions.append(np.argmax(model_predict))
    
    return (((np.sum(predictions==test_targets))/len(test_targets)))