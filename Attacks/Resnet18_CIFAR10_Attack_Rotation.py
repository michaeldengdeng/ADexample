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
from Resnet18_Cifar10 import ResNet, ResidualBlock
import time

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='D:\\DataSet', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='D:\\DataSet', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = torch.load('C:\\Users\\Administrator\\Desktop\\大四\\ADexample\\ModelSaver\\ResNet18_CIFAR10.pkl').eval()
fmodel = fb.models.PyTorchModel(model, bounds=(-3, 3), num_classes=10)


attack_FGSM = fb.attacks.FGSM(fmodel)
attack_SaliencyMap = fb.attacks.SaliencyMapAttack(fmodel)
attack_DeepFool = fb.attacks.DeepFoolAttack(fmodel)
attack_CW = fb.attacks.CarliniWagnerL2Attack(fmodel)

rows,cols=(32,32)
#rotate_angle = range(0,30,1)
rotate_angle = np.arange(0.8,1.0,0.01) 
acc_FGSM = []
acc_SaliencyMap = []
acc_DeepFool = []
acc_CW = []


for j in rotate_angle:
    rotated_adversarials_FGSM=[]
    rotated_adversarials_SaliencyMap=[]
    rotated_adversarials_DeepFool=[]
    rotated_adversarials_CW=[]
    
    matrix = cv2.getRotationMatrix2D((cols/2,rows/2),0,j)
    
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
    count=0
    
    for data in testloader:
        count+=1
        if count>10:
            break
        else:
            print('degree:',j,'count:',count,'time:',time.clock())
            images, labels = data
            images, labels = images.numpy(),labels.numpy()
            adversarials_FGSM = attack_FGSM(images, labels)
            '''
            adversarials_SaliencyMap = attack_SaliencyMap(images, labels)
            adversarials_DeepFool = attack_DeepFool(images, labels)
            adversarials_CW = attack_CW(images, labels)
            '''
            
            correct_normal_FGSM+=sum(fmodel.forward(images).argmax(axis=-1)==labels)
            correct_attack_FGSM += sum(fmodel.forward(adversarials_FGSM).argmax(axis=-1)==labels)
            '''
            correct_attack_SaliencyMap += sum(fmodel.forward(adversarials_SaliencyMap).argmax(axis=-1)==labels)
            correct_attack_DeepFool += sum(fmodel.forward(adversarials_DeepFool).argmax(axis=-1)==labels)
            correct_attack_CW += sum(fmodel.forward(adversarials_CW).argmax(axis=-1)==labels)
            '''

            for i in adversarials_FGSM:
                i=i.transpose(1,2,0)
                i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
                rotated_adversarials_FGSM.append(i)
                rotated_adversarials_FGSM2 = np.array(rotated_adversarials_FGSM)
                rotated_adversarials_FGSM2.resize(100,3,32,32)
            total_FGSM+=100
            correct_FGSM+=sum(fmodel.forward(rotated_adversarials_FGSM2).argmax(axis=-1)==labels)
    print("Acc:",correct_FGSM/total_FGSM,'Normal Acc:',correct_normal_FGSM/total_FGSM)
    acc_FGSM.append(correct_FGSM/total_FGSM)
'''
        for i in adversarials_SaliencyMap:
            i=i.transpose(1,2,0)
            i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
            rotated_adversarials_SaliencyMap.append(i)
            rotated_adversarials_SaliencyMap2 = np.array(rotated_adversarials_SaliencyMap)
            rotated_adversarials_SaliencyMap2.resize(100,3,32,32)
        total_SaliencyMap+=100
        correct_SaliencyMap+=sum(fmodel.forward(rotated_adversarials_SaliencyMap2).argmax(axis=-1)==labels)
                
        for i in adversarials_DeepFool:
            i=i.transpose(1,2,0)
            i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
            rotated_adversarials_DeepFool.append(i)
            rotated_adversarials_DeepFool2 = np.array(rotated_adversarials_DeepFool)
            rotated_adversarials_DeepFool2.resize(100,3,32,32)
        total_DeepFool+=100
        correct_DeepFool+=sum(fmodel.forward(rotated_adversarials_DeepFool2).argmax(axis=-1)==labels)
        
        for i in adversarials_CW:
            i=i.transpose(1,2,0)
            i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
            rotated_adversarials_CW.append(i)
            rotated_adversarials_CW2 = np.array(rotated_adversarials_CW)
            rotated_adversarials_CW2.resize(100,3,32,32)
        total_CW+=100
        correct_CW+=sum(fmodel.forward(rotated_adversarials_FGSM2).argmax(axis=-1)==labels)
      
    acc_FGSM.append(correct_FGSM/total_FGSM)
    
    acc_SaliencyMap.append(correct_SaliencyMap/total_SaliencyMap)
    acc_DeepFool.append(correct_DeepFool/total_DeepFool)
    acc_CW.append(correct_CW/total_CW) 
    '''
    
fig, ax = plt.subplots()

ax.plot(np.array(rotate_angle), np.array(acc_FGSM), label = 'Accuracy of FGSM')
'''
ax.plot(np.array(rotate_angle), np.array(acc_SaliencyMap), label = 'Accuracy of JSMA')
ax.plot(np.array(rotate_angle), np.array(acc_DeepFool), label = 'Accuracy of DeepFool')
ax.plot(np.array(rotate_angle), np.array(acc_CW), label = 'Accuracy of C&W')
'''
legend = ax.legend(loc='best', shadow = True, fontsize = 'large')
legend.get_frame().set_facecolor('#FFFFFF')

plt.xlabel('Resize Range')
plt.ylabel('Accuracy')
plt.show()
    


               
    #    outputs = fmodel.forward(images).argmax(axis=-1) == labels
    #    total += labels.size
    #    correct += sum(fmodel.forward(images).argmax(axis=-1) == labels)








