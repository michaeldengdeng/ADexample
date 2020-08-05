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
import skimage

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

std_range = np.arange(0, 300,10)
acc_FGSM = []
acc_SaliencyMap = []
acc_DeepFool = []
acc_CW = []


for j in std_range:
    noised_adversarials_FGSM=[]
    noised_adversarials_SaliencyMap=[]
    noised_adversarials_DeepFool=[]
    noised_adversarials_CW=[]
    
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
        if count>1:
            break
        else:
            print('std range:',j,'count:',count,'time:',time.clock())
            images, labels = data
            images, labels = images.numpy(),labels.numpy()
            
            adversarials_FGSM = attack_FGSM(images, labels)
            print("FGSMMM:",time.clock())
            adversarials_SaliencyMap = attack_SaliencyMap(images, labels)
            print("JASMMM:",time.clock())
            adversarials_DeepFool = attack_DeepFool(images, labels)
            print("DPPPPP:",time.clock())
            '''
            adversarials_CW = attack_CW(images, labels)
            print("CWWWW:",time.clock())
            '''
            correct_normal_FGSM+=sum(fmodel.forward(images).argmax(axis=-1)==labels)
            correct_attack_FGSM += sum(fmodel.forward(adversarials_FGSM).argmax(axis=-1)==labels)
            correct_attack_SaliencyMap += sum(fmodel.forward(adversarials_SaliencyMap).argmax(axis=-1)==labels)
            correct_attack_DeepFool += sum(fmodel.forward(adversarials_DeepFool).argmax(axis=-1)==labels)
            #correct_attack_CW += sum(fmodel.forward(adversarials_CW).argmax(axis=-1)==labels)
            
            for i in range(len(adversarials_FGSM)):
                bilateralFilter_FGSM = np.uint8(adversarials_FGSM[i]*255)
                bilateralFilter_FGSM = cv2.bilateralFilter(bilateralFilter_FGSM.transpose(1,2,0),3,j,j)
                bilateralFilter_FGSM = bilateralFilter_FGSM.transpose(2,0,1)
                adversarials_FGSM[i] = np.float32(bilateralFilter_FGSM/255.0) 
            noised_adversarials_FGSM = adversarials_FGSM
            total_FGSM+=100
            correct_FGSM+=sum(fmodel.forward(noised_adversarials_FGSM).argmax(axis=-1)==labels)
            
            for i in range(len(adversarials_SaliencyMap)):
                bilateralFilter_SaliencyMap = np.uint8(adversarials_SaliencyMap[i]*255)
                bilateralFilter_SaliencyMap = cv2.bilateralFilter(bilateralFilter_SaliencyMap.transpose(1,2,0),25,j,j)
                bilateralFilter_SaliencyMap = bilateralFilter_SaliencyMap.transpose(2,0,1)
                adversarials_SaliencyMap[i] = np.float32(bilateralFilter_SaliencyMap/255.0) 
            noised_adversarials_SaliencyMap = adversarials_SaliencyMap
            total_SaliencyMap+=100
            correct_SaliencyMap+=sum(fmodel.forward(noised_adversarials_SaliencyMap).argmax(axis=-1)==labels)
            
            for i in range(len(adversarials_DeepFool)):
                bilateralFilter_DeepFool = np.uint8(adversarials_DeepFool[i]*255)
                bilateralFilter_DeepFool = cv2.bilateralFilter(bilateralFilter_DeepFool.transpose(1,2,0),25,j,j)
                bilateralFilter_DeepFool = bilateralFilter_DeepFool.transpose(2,0,1)
                adversarials_DeepFool[i] = np.float32(bilateralFilter_DeepFool/255.0) 
            noised_adversarials_DeepFool = adversarials_DeepFool
            total_DeepFool+=100
            correct_DeepFool+=sum(fmodel.forward(noised_adversarials_DeepFool).argmax(axis=-1)==labels)

            
            '''
            for i in range(len(adversarials_CW)):
                adversarials_CW[i] = skimage.util.random_noise(adversarials_CW[i], 
                                                                 mode = 's&p', seed = None, 
                                                                 clip = True, amount=j)
            noised_adversarials_CW = adversarials_CW
            total_CW+=100
            correct_CW+=sum(fmodel.forward(noised_adversarials_CW).argmax(axis=-1)==labels)
            '''
    
    print("AccFGSM:",correct_FGSM/total_FGSM,'Normal Acc:',correct_normal_FGSM/total_FGSM)
    acc_FGSM.append(correct_FGSM/total_FGSM)
    print("AccJSMA:",correct_SaliencyMap/total_SaliencyMap,'Normal Acc:',correct_normal_FGSM/total_FGSM)
    acc_SaliencyMap.append(correct_SaliencyMap/total_SaliencyMap)
    print("AccDeepFool:",correct_DeepFool/total_DeepFool,'Normal Acc:',correct_normal_FGSM/total_DeepFool)
    acc_DeepFool.append(correct_DeepFool/total_DeepFool)    
    #print("AccCW:",correct_CW/total_CW,'Normal Acc:',correct_normal_FGSM/total_CW)
    #acc_CW.append(correct_CW/total_CW)
    
    
    
fig, ax = plt.subplots()

ax.plot(np.array(std_range), np.array(acc_FGSM), label = 'Accuracy of FGSM')
ax.plot(np.array(std_range), np.array(acc_SaliencyMap), label = 'Accuracy of JSMA')
ax.plot(np.array(std_range), np.array(acc_DeepFool), label = 'Accuracy of DeepFool')
#ax.plot(np.array(std_range), np.array(acc_CW), label = 'Accuracy of C&W')

legend = ax.legend(loc='best', shadow = True, fontsize = 'small')
legend.get_frame().set_facecolor('#FFFFFF')

plt.xlabel('Amount Range')
plt.ylabel('Accuracy')
plt.show()
    


               
    #    outputs = fmodel.forward(images).argmax(axis=-1) == labels
    #    total += labels.size
    #    correct += sum(fmodel.forward(images).argmax(axis=-1) == labels)








# -*- coding: utf-8 -*-

