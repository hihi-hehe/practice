import numpy as numpy
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasetsa

if torch.cuda.is_available() :
    DEVICE = torch.device('cuda')
else :
    DEVICE = torch.device('cpu')
print('Using PyTorch version: ', torch.__version__, '\t Device: ', DEVICE)
BATCH_SIZE = 32
EPOCHS = 10

train_dataset = datasets.MNIST(root="/data/MNIST", train = True, download = True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST(root = "/data/MNIST", train = False, transform = transforms.ToTensor())
