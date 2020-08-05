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

MODELPATH = os.getcwd()+'\\ModelSaver'
transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

trainset = torchvision.datasets.CIFAR10(root='D:\\DataSet', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='D:\\DataSet', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


feature_extract=True
model_name ='resnet'
num_classes = 2
batch_size = 32
num_epochs = 15 
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name,num_classes,feature_extract, use_pretrained=True):
    if model_name == 'resnet':
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 32
    return model_ft,input_size

model_ft, input_size = initialize_model(model_name,num_classes,feature_extract,use_pretrained=True)

def train_model(model,trainloader,loss_fn, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            if 








