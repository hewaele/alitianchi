#%%
import torch
import pandas as pd
import os
import numpy as np
import time
import torchvision
import torch.utils.data as td
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda:0')
#%%读取数据集
fashion_mnist_path = './data'
train_data = torchvision.datasets.FashionMNIST(fashion_mnist_path, train=True, download=False,
                                               transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(fashion_mnist_path, train=False, download=False,
                                              transform=torchvision.transforms.ToTensor())


print(type(train_data))
# tensor_train_data = torchvision.transforms.ToTensor()(train_data)
# tensor_test_data = torchvision.transforms.ToTensor()(test_data)
#将数据进行
train_iters = td.DataLoader(train_data, batch_size=64, shuffle=True)
test_iters = td.DataLoader(test_data, batch_size=64, shuffle=True)

#%%
x0 = 0
for xi, yi in train_iters:
    # print(xi.shape, yi.shape)
    # print(xi[0][0])
    x0 = xi
    # t = Image.fromarray(xi[0][0].detach().numpy()*255)
    # t.show()
    break

#调用vgg类
#"D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
vgg16 = torch.nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=True),
                               torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), torch.nn.BatchNorm2d(64),torch.nn.ReLU(inplace=True),
                               torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                               # torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),torch.nn.ReLU(inplace=True),
                               torch.nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
                               torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                               torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
                               # torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),torch.nn.ReLU(inplace=True),
                               # torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),torch.nn.ReLU(inplace=True),
                               torch.nn.Flatten(),
                               torch.nn.Linear(128*7*7, 2048), torch.nn.ReLU(inplace=True),
                               torch.nn.Linear(2048, 512), torch.nn.ReLU(inplace=True),
                               torch.nn.Linear(512, 10), torch.nn.Softmax(dim=1)
                               # torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                               # torch.nn.Conv2d(256, 512, kernel_size=(3, 3)),
                               # torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                               # torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                               # torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                               # torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                               # torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                               # torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                               # torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                               )

# vgg16 = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=(5, 5)), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
#     nn.Linear(120, 84), nn.Sigmoid(),
#     nn.Linear(84, 10))

#执行训练
def init_weight(n):
    if type(n) in [torch.nn.Linear, torch.nn.Conv2d]:
        torch.nn.init.kaiming_normal_(n.weight)
        torch.nn.init.zeros_(n.bias)
    if isinstance(n, nn.BatchNorm2d):
        torch.nn.init.constant_(n.weight, 1)
        torch.nn.init.constant_(n.bias, 1)

vgg16.apply(init_weight)

def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    The evaluate function, and the performance measure is accuracy.
    """
    ret_acc, temp_num = 0., 0
    with torch.no_grad():
        for x, y in data_iter:
            net.eval() # The evaluate mode, and the dropout is closed.
            ret_acc += (net(x.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            temp_num += y.shape[0]

    return ret_acc / temp_num

print(x0.shape)
y_hat = vgg16(x0)
print(y_hat.shape)
optimer = torch.optim.SGD(vgg16.parameters(), lr=0.1)
loss = torch.nn.CrossEntropyLoss()

vgg16 = vgg16.to(device)
print(vgg16)
for opoch in range(300):
    for xi, yi in train_iters:
        xi = xi.to(device)
        yi = yi.to(device)
        y_hat = vgg16(xi)
        l = loss(y_hat, yi)
        optimer.zero_grad()
        l.backward()
        optimer.step()

    ac = evaluate_accuracy(test_iters, vgg16)
    print(opoch, ac)