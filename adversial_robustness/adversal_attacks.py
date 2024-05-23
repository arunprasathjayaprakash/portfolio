import pandas as pd
import numpy as np
import timm
import art
import torch
import cv2
import torchvision.datasets
from IPython.display import display

resnet_34 = timm.create_model('resnet34',pretrained=True)
config_values = resnet_34.default_cfg
resnet_34