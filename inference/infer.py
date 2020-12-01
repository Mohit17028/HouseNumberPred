import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import time
import tarfile
from inference.model_skeleton import Model




def load_model():
    classifier = torch.load('SVHN_model_checkpoint.tar', map_location=torch.device(device))
    model = Model()
    model.load_state_dict(classifier['model_state_dict'])
    model.eval()
    return model

classifier = torch.load('SVHN_model_checkpoint.tar', map_location=device)
inf_model = Model()
inf_model.load_state_dict(classifier['model_state_dict'])
inf_model.eval()