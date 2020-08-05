import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os


DATASET = 'MNIST'
MODELPATH = os.getcwd()+'\\ModelSaver'
batch_size = 128        # 批的大小
learning_rate = 1e-2    # 学习率
num_epoches = 10        # 遍历训练集的次数





if DATASET == 'CIFAR10':
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = datasets.CIFAR10(root='D:\\DataSet', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
    testset = datasets.CIFAR10(root='D:\\DataSet', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=0)
elif DATASET == 'MNIST':
    trainset = datasets.MNIST(root='D:\\DataSet', train=True, transform=transforms.ToTensor(), download=True)
    testset = datasets.MNIST(root='D:\\DataSet', train=False, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 定义 Convolution Network 模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()    # super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.conv = nn.Sequential(     #padding=2保证输入输出尺寸相同(参数依次是:输入深度，输出深度，ksize，步长，填充)
            nn.Conv2d(in_dim, 6, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120), 
            nn.Linear(120, 84), 
            nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

'''
model = Cnn(1, 10)  # 图片大小是28x28,输入深度是1，最终输出的10类
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))      # .format为输出格式，formet括号里的即为左边花括号的输出
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(trainloader, 1):
        img, label = data
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / (len(trainset)), running_acc / (len(trainset))))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in testloader:
        img, label = data
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        testset)), eval_acc / (len(testset))))
    print()

PATH = MODELPATH+'\\LeNet_MNIST.pkl'
#torch.save(model.state_dict(), PATH)
torch.save(model,PATH)'''
