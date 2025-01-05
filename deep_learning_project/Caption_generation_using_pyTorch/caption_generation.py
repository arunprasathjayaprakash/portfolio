import torch
import math
import os
import pickle
import pathlib
import gradio
import pandas
import random , re
from collections import defaultdict
from PIL import Image
from torch import nn
from torchmetrics.text import ROUGEScore
from dataloader import *

random.seed(42)

if __name__ == "__main__":
    IMAGE_DIR = os.path.join(os.getcwd(), 'usercode/flickr-8k')
    TRAIN_TEXT_DIR = os.path.join(os.getcwd(), 'usercode/flickr-8k/text/Flickr_8k.trainImages.txt')
    TEST_TEXT_DIR = os.path.join(os.getcwd(), 'usercode/flickr-8k/text/Flickr_8k.testImages.txt')
    VALID_TEXT_DIR = os.path.join(os.getcwd(), 'usercode/flickr-8k/text/Flickr_8k.devImages.txt')
    CAPTIONS_FILE = os.path.join(os.getcwd(), 'usercode/flickr-8k/text/Flickr8k.token.txt')
    dataset = CustomDataset(IMAGE_DIR, TRAIN_TEXT_DIR, None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=45, shuffle=True)
    images , lables =  next(iter(dataloader))