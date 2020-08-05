# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:16:59 2020

@author: Akun
"""


import torch
import torchvision
import foolbox as fb
import numpy as np
import torchvision.models as models


model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = fb.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

images,labels = fb.utils.samples(dataset='imagenet', batchsize=10, data_format='channels_first', bounds=(0, 1))
print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))
attack = fb.attacks.FGSM(fmodel)
adversarials = attack(images, labels)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))

attack = fb.attacks.FGSM(fmodel, distance = fb.distances.Linf)
adversarials = attack(images, labels, unpack = False)

adversarial_classes = np.asarray([a.adversarial_class for a in adversarials])
print(labels)
print(adversarial_classes)
print(np.mean(adversarial_classes == labels))

distances = np.asarray([a.distance.value for a in adversarials])
print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))
print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))
print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))

import matplotlib.pyplot as plt

image = images[4]
adversarial = attack(images, labels)[4]

# CHW to HWC
image = image.transpose(1, 2, 0)
adversarial = adversarial.transpose(1, 2, 0)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original/Prediction:{} strawberry'.format(labels[4]))
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial/Prediction:{} trifle'.format(adversarials[4].adversarial_class))
plt.imshow(adversarial)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()

    



























