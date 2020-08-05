import torch
import torchvision
import torchvision.transforms as transforms
import foolbox as fb
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import numpy as np

trainset = torchvision.datasets.CIFAR10(root='D:\\DataSet', train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='D:\\DataSet', train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

images = np.zeros([50000,32,32,3])
labels = np.zeros([50000])
count = 0
for i,j in trainset:
    images[count]=i
    labels[count]=j
    count+=1
images = images.transpose(0,3,1,2)
images = images/255












