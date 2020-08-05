import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.utils.data.dataloader as Data
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
import torch.nn.functional as F
import foolbox as fb
import matplotlib.pyplot as plt


model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = fb.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

images,labels = fb.utils.samples(dataset='imagenet', batchsize=5, data_format='channels_first', bounds=(0, 1))
image = images[4]
label = labels[4]

image=np.expand_dims(image, axis=0)

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



