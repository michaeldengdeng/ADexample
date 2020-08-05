import torch
import os
import torchvision
import foolbox as fb
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
from AlexNet_Cifar10 import AlexNet
import time

batch_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainset = torchvision.datasets.CIFAR10(root='D:\\DataSet', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='D:\\DataSet', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = torch.load('C:\\Users\\Administrator\\Desktop\\大四\\ADexample\\ModelSaver\\AlexNet_CIFAR10.pkl').eval()
model.to(device)
fmodel = fb.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
attack_FGSM = fb.attacks.FGSM(fmodel)
attack_SaliencyMap = fb.attacks.SaliencyMapAttack(fmodel)
attack_DeepFool = fb.attacks.DeepFoolAttack(fmodel)
attack_CW = fb.attacks.CarliniWagnerL2Attack(fmodel)

rows,cols=(228,228)
rotate_angle = range(0,360,120)
#rotate_angle = np.arange(0,1,0.01) 
acc_FGSM = []
acc_SaliencyMap = []
acc_DeepFool = []
acc_CW = []
count=0

for j in rotate_angle:
    rotated_adversarials_FGSM=[]
    rotated_adversarials_SaliencyMap=[]
    rotated_adversarials_DeepFool=[]
    rotated_adversarials_CW=[]
    
    matrix = cv2.getRotationMatrix2D((cols/2,rows/2),j,1)
    
    correct = 0
    total = 0
    correct_FGSM = 0 
    total_FGSM = 0
    correct_SaliencyMap = 0 
    total_SaliencyMap = 0
    correct_DeepFool = 0
    total_DeepFool = 0
    correct_CW = 0
    total_CW = 0
    correct_attack_FGSM = 0
    correct_attack_SaliencyMap = 0
    correct_attack_DeepFool = 0
    correct_attack_CW = 0
    correct_normal_FGSM=0
    
    for data in testloader:
        images, labels = data
        images, labels = images.numpy(),labels.numpy()
        adversarials_FGSM = attack_FGSM(images, labels)
        adversarials_SaliencyMap = attack_SaliencyMap(images, labels)
        adversarials_DeepFool = attack_DeepFool(images, labels)
        adversarials_CW = attack_CW(images, labels)
        
        correct_attack_SaliencyMap += sum(fmodel.forward(adversarials_SaliencyMap).argmax(axis=-1)==labels)
        correct_attack_DeepFool += sum(fmodel.forward(adversarials_DeepFool).argmax(axis=-1)==labels)
        correct_attack_CW += sum(fmodel.forward(adversarials_CW).argmax(axis=-1)==labels)
        

        for i in adversarials_FGSM:
            i=i.transpose(1,2,0)
            i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
            rotated_adversarials_FGSM.append(i)
            rotated_adversarials_FGSM2 = np.array(rotated_adversarials_FGSM)
            rotated_adversarials_FGSM2.resize(100,3,227,227)
        total_FGSM+=100
        correct_FGSM+=sum(fmodel.forward(rotated_adversarials_FGSM2).argmax(axis=-1)==labels)
             
        for i in adversarials_SaliencyMap:
            i=i.transpose(1,2,0)
            i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
            rotated_adversarials_SaliencyMap.append(i)
            rotated_adversarials_SaliencyMap2 = np.array(rotated_adversarials_SaliencyMap)
            rotated_adversarials_SaliencyMap2.resize(100,3,227,227)
        total_SaliencyMap+=100
        correct_SaliencyMap+=sum(fmodel.forward(rotated_adversarials_SaliencyMap2).argmax(axis=-1)==labels)
                
        for i in adversarials_DeepFool:
            i=i.transpose(1,2,0)
            i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
            rotated_adversarials_DeepFool.append(i)
            rotated_adversarials_DeepFool2 = np.array(rotated_adversarials_DeepFool)
            rotated_adversarials_DeepFool2.resize(100,3,227,227)
        total_DeepFool+=100
        correct_DeepFool+=sum(fmodel.forward(rotated_adversarials_DeepFool2).argmax(axis=-1)==labels)
        
        for i in adversarials_CW:
            i=i.transpose(1,2,0)
            i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
            rotated_adversarials_CW.append(i)
            rotated_adversarials_CW2 = np.array(rotated_adversarials_CW)
            rotated_adversarials_CW2.resize(100,3,227,227)
        total_CW+=100
        correct_CW+=sum(fmodel.forward(rotated_adversarials_FGSM2).argmax(axis=-1)==labels)
        
    acc_FGSM.append(correct_FGSM/total_FGSM)
    acc_SaliencyMap.append(correct_SaliencyMap/total_SaliencyMap)
    acc_DeepFool.append(correct_DeepFool/total_DeepFool)
    acc_CW.append(correct_CW/total_CW) 




#18.36














