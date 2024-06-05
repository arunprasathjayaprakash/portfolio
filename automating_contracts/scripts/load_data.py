import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualize_features import visualize_stats
from tqdm import tqdm
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from transformers import Trainer , TrainingArguments
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score


def load_data(data_path,req_count):
    data = json.load(open(data_path,'r'))
    text_list, hypothesis_list, labels_list = [], [], []
    counter = 0

    while counter != req_count:
        for k, v in data['documents'][counter].items():
            if k == 'annotation_sets':
                for annot_value in v:
                    for ndx_value in annot_value['annotations']:
                        hypothesis_list.append(ndx_value)
                        labels_list.append(data['labels'][ndx_value]['hypothesis'])
                        text_list.append(data['documents'][counter]['text'])

                counter += 1


    if len(text_list) == len(hypothesis_list) == len(labels_list):
        data_df = pd.DataFrame()
        data_df['Text'] = text_list
        data_df['hypothesis'] = hypothesis_list
        data_df['labels'] = labels_list

    return data_df


if __name__ == "__main__":
    data_path = r'C:\csulb_projects\portfolio_projects\automating_contracts\data\train.json'
    train_data = load_data(data_path,35)
    valid_data = load_data(data_path,10)
    test_data = load_data(data_path,5)
    visualize_stats(train_data,'training','Text','labels')