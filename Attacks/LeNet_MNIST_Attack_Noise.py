import torch
import os
import torchvision
import foolbox as fb
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from LeNet import Cnn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
batch_size = 128
import cv2
import skimage

trainset = torchvision.datasets.MNIST(root='D:\\DataSet', train=True, download=True)
testset = torchvision.datasets.MNIST(root='D:\\DataSet', train=False)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

model = torch.load('C:\\Users\\Administrator\\Desktop\\大四\\ADexample\\ModelSaver\\LeNet_MNIST.pkl') 

fmodel = fb.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)


images = np.zeros([10000,1,28,28])
labels = np.zeros([10000])
count = 0
for i,j in testset:
    images[count]=i
    labels[count]=j
    count+=1
images = images/255
images = np.array(images, dtype = np.float32)
labels = np.array(labels, dtype = np.int32)

print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))

attack_FGSM = fb.attacks.FGSM(fmodel)
attack_SaliencyMap = fb.attacks.SaliencyMapAttack(fmodel)
attack_DeepFool = fb.attacks.DeepFoolAttack(fmodel)
attack_CW = fb.attacks.CarliniWagnerL2Attack(fmodel)

adversarials_FGSM = attack_FGSM(images[:100], labels[:100])
adversarials_SaliencyMap = attack_SaliencyMap(images[:100], labels[:100])
adversarials_DeepFool = attack_DeepFool(images[:100], labels[:100])
adversarials_CW = attack_CW(images[:100], labels[:100])

'''
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels[:100]))
attack = fb.attacks.FGSM(fmodel, distance = fb.distances.Linf)
adversarials_donunpack = attack(images[:1000], labels[:1000], unpack = False)
adversarial_classes = np.asarray([a.adversarial_class for a in adversarials_donunpack])
print(np.mean(adversarial_classes == labels[:1000]))
'''

std_range = np.arange(0, 0.20, 0.01)
acc_FGSM = []
acc_SaliencyMap = []
acc_DeepFool = []
acc_CW = []

for j in std_range:
    noised_adversarials_FGSM=[]
    noised_adversarials_SaliencyMap=[]
    noised_adversarials_DeepFool=[]
    noised_adversarials_CW=[]
    
    
    for i in range(len(adversarials_FGSM)):
        adversarials_FGSM[i] = skimage.util.random_noise(adversarials_FGSM[i], 
                                                         mode = 'gaussian', seed = None, 
                                                         clip = True, mean = 0, var = j**2)
    noised_adversarials_FGSM = np.array(noised_adversarials_FGSM,dtype=np.float32)
    noised_adversarials_FGSM.resize(100,1,28,28)
    print(np.mean(fmodel.forward(noised_adversarials_FGSM).argmax(axis=-1) == labels[:100]))
    acc_FGSM.append(np.mean(fmodel.forward(noised_adversarials_FGSM).argmax(axis=-1) == labels[:100]))
    
    for i in adversarials_SaliencyMap:
        i = skimage.util.random_noise(i.copy(), mode = 'gaussian', seed = None, clip = True, mean = 0, var = j**2)
    noised_adversarials_SaliencyMap = np.array(noised_adversarials_SaliencyMap,dtype=np.float32)
    noised_adversarials_SaliencyMap.resize(100,1,28,28)
    print(np.mean(fmodel.forward(noised_adversarials_SaliencyMap).argmax(axis=-1) == labels[:100]))
    acc_SaliencyMap.append(np.mean(fmodel.forward(noised_adversarials_SaliencyMap).argmax(axis=-1) == labels[:100]))

    for i in adversarials_DeepFool:
        i = skimage.util.random_noise(i.copy(), mode = 'gaussian', seed = None, clip = True, mean = 0, var = j**2)
    noised_adversarials_DeepFool = np.array(noised_adversarials_DeepFool,dtype=np.float32)
    noised_adversarials_DeepFool.resize(100,1,28,28)
    print(np.mean(fmodel.forward(noised_adversarials_DeepFool).argmax(axis=-1) == labels[:100]))
    acc_DeepFool.append(np.mean(fmodel.forward(noised_adversarials_DeepFool).argmax(axis=-1) == labels[:100]))

    for i in adversarials_CW:
        i = skimage.util.random_noise(i.copy(), mode = 'gaussian', seed = None, clip = True, mean = 0, var = j**2)
    noised_adversarials_CW = np.array(noised_adversarials_CW,dtype=np.float32)
    noised_adversarials_CW.resize(100,1,28,28)
    print(np.mean(fmodel.forward(noised_adversarials_CW).argmax(axis=-1) == labels[:100]))
    acc_CW.append(np.mean(fmodel.forward(noised_adversarials_CW).argmax(axis=-1) == labels[:100]))


fig, ax = plt.subplots()
ax.plot(np.array(std_range), np.array(acc_FGSM), label = 'Accuracy of FGSM')
ax.plot(np.array(std_range), np.array(acc_SaliencyMap), label = 'Accuracy of JSMA')
ax.plot(np.array(std_range), np.array(acc_DeepFool), label = 'Accuracy of DeepFool')
ax.plot(np.array(std_range), np.array(acc_CW), label = 'Accuracy of C&W')

legend = ax.legend(loc='best', shadow = True, fontsize = 'large')
legend.get_frame().set_facecolor('#FFFFFF')

plt.xlabel('Std Range')
plt.ylabel('Accuracy')
plt.show()



print(np.mean(fmodel.forward(adversarials_FGSM).argmax(axis=-1) == labels))
print(np.mean(fmodel.forward(adversarials_SaliencyMap).argmax(axis=-1) == labels))
print(np.mean(fmodel.forward(adversarials_DeepFool).argmax(axis=-1) == labels))
print(np.mean(fmodel.forward(adversarials_CW).argmax(axis=-1) == labels))












































'''
image = images[4]
adversarial = attack(images[:10],labels[:10])[4]

image = image.transpose(1, 2, 0)
adversarial = adversarial.transpose(1, 2, 0)


plt.figure()

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(adversarial)
plt.axis('off')


plt.show()

'''















