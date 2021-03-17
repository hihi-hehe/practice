import numpy as np
import matplotlib.pyplot as plt

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

for(X_train, Y_tain) in train_loader:
    print('X_train: ', X_train.size(), 'type: ', X_train.type())
    print('X_train: ', X_train.size(), 'type: ', X_train.type())
    break

pltsize = 1
plt.figure(figsize=(10*pltsize,pltsize))
   
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28.28), cmap = "gray_r")
    plt.title('Class: ' +str(y_train[i].item()))

class Net(nn.Module):