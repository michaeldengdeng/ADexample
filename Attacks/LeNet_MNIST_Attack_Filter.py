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

adversarials_FGSM = attack_FGSM(images[:1000], labels[:1000])
'''
adversarials_SaliencyMap = attack_SaliencyMap(images[:10000], labels[:10000])
adversarials_DeepFool = attack_DeepFool(images[:10000], labels[:10000])
adversarials_CW = attack_CW(images[:10000], labels[:10000])


print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels[:100]))
attack = fb.attacks.FGSM(fmodel, distance = fb.distances.Linf)
adversarials_donunpack = attack(images[:1000], labels[:1000], unpack = False)
adversarial_classes = np.asarray([a.adversarial_class for a in adversarials_donunpack])
print(np.mean(adversarial_classes == labels[:1000]))
'''

rows,cols=(28,28)
#rotate_angle = range(0,360,1)
#rotate_angle = np.arange(0,1,0.01) 
std_range = np.arange(0, 20, 1)
acc_FGSM = []
acc_SaliencyMap = []
acc_DeepFool = []
acc_CW = []

for j in std_range:
    rotated_adversarials_FGSM=[]
    rotated_adversarials_SaliencyMap=[]
    rotated_adversarials_DeepFool=[]
    rotated_adversarials_CW=[]

    matrix = cv2.getRotationMatrix2D((cols/2,rows/2),j,1)
    for i in adversarials_FGSM:
        i=i.transpose(1,2,0)
        i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
        rotated_adversarials_FGSM.append(i)
    rotated_adversarials_FGSM = np.array(rotated_adversarials_FGSM)
    rotated_adversarials_FGSM.resize(10000,1,28,28)
    print(np.mean(fmodel.forward(rotated_adversarials_FGSM).argmax(axis=-1) == labels[:10000]))
    acc_FGSM.append(np.mean(fmodel.forward(rotated_adversarials_FGSM).argmax(axis=-1) == labels[:10000]))
    '''
    for i in adversarials_SaliencyMap:
        i=i.transpose(1,2,0)
        i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
        rotated_adversarials_SaliencyMap.append(i)
    rotated_adversarials_SaliencyMap = np.array(rotated_adversarials_SaliencyMap)
    rotated_adversarials_SaliencyMap.resize(10000,1,28,28)
    print(np.mean(fmodel.forward(rotated_adversarials_SaliencyMap).argmax(axis=-1) == labels[:10000]))
    acc_SaliencyMap.append(np.mean(fmodel.forward(rotated_adversarials_SaliencyMap).argmax(axis=-1) == labels[:10000]))

    for i in adversarials_DeepFool:
        i=i.transpose(1,2,0)
        i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
        rotated_adversarials_DeepFool.append(i)
    rotated_adversarials_DeepFool = np.array(rotated_adversarials_DeepFool)
    rotated_adversarials_DeepFool.resize(10000,1,28,28)
    print(np.mean(fmodel.forward(rotated_adversarials_DeepFool).argmax(axis=-1) == labels[:10000]))
    acc_DeepFool.append(np.mean(fmodel.forward(rotated_adversarials_DeepFool).argmax(axis=-1) == labels[:10000]))

    for i in adversarials_CW:
        i=i.transpose(1,2,0)
        i = cv2.warpAffine(i.copy(),matrix,(cols,rows))
        rotated_adversarials_CW.append(i)
    rotated_adversarials_CW = np.array(rotated_adversarials_CW)
    rotated_adversarials_CW.resize(10000,1,28,28)
    print(np.mean(fmodel.forward(rotated_adversarials_CW).argmax(axis=-1) == labels[:10000]))
    acc_CW.append(np.mean(fmodel.forward(rotated_adversarials_CW).argmax(axis=-1) == labels[:10000]))
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


'''
print(np.mean(fmodel.forward(adversarials_FGSM).argmax(axis=-1) == labels))
print(np.mean(fmodel.forward(adversarials_SaliencyMap).argmax(axis=-1) == labels))
print(np.mean(fmodel.forward(adversarials_DeepFool).argmax(axis=-1) == labels))
print(np.mean(fmodel.forward(adversarials_CW).argmax(axis=-1) == labels))
'''











































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














