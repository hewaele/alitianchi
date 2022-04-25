#创建一个网络类，包含前向运算，训练，预测
#%%
import torch
from torch.utils import data as td
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

import sklearn
from sklearn.datasets import _samples_generator
from sklearn import metrics
from sklearn.model_selection import train_test_split

#%%
#创建双月数据集
source_x, source_y = _samples_generator.make_moons(1000, noise=0.1, random_state=20220422)
train_x, test_x, train_y, test_y = train_test_split(source_x, source_y, test_size=0.2, random_state=20220422)
tensor_train_x, tensor_train_y = torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)
tensor_test_x, tensor_test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long)

#创建数据迭代器
batch_size = 16
train_tensor_dataset = td.TensorDataset(tensor_train_x, tensor_train_y)
train_tensor_iter = td.DataLoader(train_tensor_dataset, batch_size, drop_last=False)

#%%
#创建一个多层感知机类
class MyNet(torch.nn.Module):
    def __init__(self, layers_size_list, output_size):
        super().__init__()
        self.layers_size_list = layers_size_list
        self.output_size = output_size

        #现根据输入生成层
        for index, layer in enumerate(self.layers_size_list[:-1]):
            self._modules['liner{}'.format(index)] = torch.nn.Linear(self.layers_size_list[index], self.layers_size_list[index+1])
            self._modules['relu{}'.format(index)] = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(10, 2)
        # self.output_softmax = torch.nn.Softmax(dim=1)
        # self.l1 = torch.nn.Linear(2, 20)
        # self.l2 = torch.nn.Linear(20, 10)
        # self.l3 = torch.nn.Linear(10, 2)
        for mi in self._modules.values():
            print('hello')
            print(mi)


    def forward(self, x):
        # x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        for index, layer in enumerate(self._modules.values()):
            x = layer(x)
        #对函数进行softmax

        # x = self.output_layer(x)
        # print('here')
        output = nn.Softmax(dim=1)(x)
        return output


def train(net, train_dataset_iter, learning_rate, epochs=100, optimer=None):
    #对参数进行初始话
    def init_parameters(layer):
        if type(layer) in [torch.nn.Linear, torch.nn.Conv2d]:
            print(type(layer))
            torch.nn.init.normal_(layer.weight, 0, 0.01)
            torch.nn.init.ones_(layer.bias)
    net.apply(init_parameters)
    if optimer is None:
        optimer = torch.optim.SGD(net.parameters(), learning_rate)
    else:
        optimer = optimer
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for xi, yi in train_dataset_iter:
            y_hat = net(xi)
            tmp_loss = loss(y_hat, yi)
            optimer.zero_grad()
            tmp_loss.backward()
            optimer.step()


        #执行预测查看训练效果
        y_hat = net(tensor_test_x)
        # print(np.argmax(y_hat.detach().numpy(), axis=1))
        tmp_loss = loss(y_hat, tensor_test_y)
        ac = torch.sum(torch.argmax(y_hat.detach(), dim=1) == tensor_test_y) / tensor_test_y.shape[0]

        print('epoch:{} loss: {} ac: {}'.format(epoch, tmp_loss, ac))

#执行训练
net = MyNet([tensor_train_x.shape[1], 20,10], 2)
print(net)
# for pi in net.named_parameters():
#     print(pi)
# print(net.named_parameters())
# y_hat = net(tensor_test_x[:10])
# print(y_hat)
train(net, train_tensor_iter, 0.1)

#%%
#自定义块测试
from torch import nn
from torch.nn import functional as F
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()

        self.input = nn.Linear(2, 20)
        self.hidden = nn.Linear(20, 30)  # 隐藏层
        self.output = nn.Linear(30, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        X = self.test(X)
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

test_net = MLP()
for pi in test_net.named_parameters():
    print(pi)

#%%
