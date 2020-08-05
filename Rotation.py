import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import foolbox as fb

def infer_img(img,model,t=0,AD=False,Fool=False):
    if AD==False:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img=np.expand_dims(img, axis=0)

    
    #使用预测模式 主要影响droupout和BN层的行为
    if Fool==True:
        img = model.forward(img)
        img.resize(1000)
        output = fb.utils.softmax(img)
        label=np.argmax(output)
        pro = output[label]
        if t!=0:         #t不为0返回指定类别概率
            pro = output[t]
    else:
        #model = models.resnet18(pretrained=True).eval()
        img = Variable(torch.from_numpy(img).float())
        output=F.softmax(model(img),dim=1)    
        label=np.argmax(output.data.numpy())
        pro=output.data.numpy()[0][label]
        if t != 0:
            pro=output.data.numpy()[0][t]
    
    #print("{}={}".format(t,pro))
    return pro



images,labels = fb.utils.samples(dataset='imagenet', batchsize=16, data_format='channels_first', bounds=(0, 1))
image = images[3]


model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = fb.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
attack = fb.attacks.FGSM(fmodel)
adversarials = attack(images, labels)
adversarial = adversarials[3]
adversarial = adversarial.transpose(1,2,0)#CHW->HWC以进行cv2旋转仿射

image = image.transpose(1,2,0)
rows,cols=(224,224)
#rotation_range = range(0,180,10)
rotation_range = [0.8,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.0]
#translation_range = np.arange(0,100,5)
original_pro = []
adv_pro = []



for i in rotation_range:
    #matrix = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
    matrix = cv2.getRotationMatrix2D((cols/2,rows/2),0,i)
    #matrix = np.float32([[1,0,0-i],[0,1,0-i]])
    
    rotate_adv_img = cv2.warpAffine(adversarial.copy(),matrix,(cols,rows))
    prob_243=infer_img(rotate_adv_img.copy(),fmodel,t=990,AD=True,Fool=True) 
    print("rotate={} prob_adv [243, Bull Mastiff]={}".format(i,prob_243))
    adv_pro+= [prob_243]
    
    rotate_img = cv2.warpAffine(image.copy(),matrix,(cols,rows))
    prob=infer_img(rotate_img.copy(),model,t=990,AD=False,Fool=False)
    print("rotate={} prob_ori [243,bull mastiff]={}".format(i,prob))
    original_pro += [prob]
    
matrix60 = cv2.getRotationMatrix2D((cols/2,rows/2),60,1)
rotate_img = cv2.warpAffine(image.copy(),matrix60,(cols,rows))

'''
#Figure 6画图部分
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.array(rotation_range), np.array(original_pro),color='#9370DB', linestyle='-.', label='Probability of Class Bull Mastiff(243)')
#ax.plot(np.array(eps_range), np.array(nb_correct_robust), 'r--', label='Robust classifier')

legend = plt.legend(loc='best', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#FFFFFF')
plt.xlabel('Rotation Range')
plt.ylabel('Prob')

plt.subplot(1,2,2)
plt.imshow(rotate_img)
plt.title("Image after rotation of 60-degree")
plt.show()
'''

#Figure 7画图部分
fig, ax = plt.subplots()
ax.plot(np.array(rotation_range), np.array(adv_pro), color='#9370DB',linestyle='-.', label='Probability of Adversarial')
ax.plot(np.array(rotation_range), np.array(original_pro), 'r',linestyle='-.', label='Probability of Original')

legend = ax.legend(loc='best', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#FFFFFF')

plt.xlabel('Zoom Range')
plt.ylabel('Prob')
plt.show()
