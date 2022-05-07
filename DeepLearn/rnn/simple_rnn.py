import torch
from torch.utils import data as td
from torch.nn import functional as Func
import numpy as np
import tensorboard
import os
import time
import matplotlib.pyplot as plt

#创建一个sin函数生成序列数据
x = np.linspace(0, 20, 1000)
y = np.sin(x)

plt.figure(figsize=(16,2))
plt.plot(x, y)
plt.show()

#生成训练数据测试数据
def creat_data(x, y, features, numstep):
    position = 0
    index = []
    train_x = []
    train_y = []
    while 1:
        index.append(x[position:position+features])
        train_x.append(y[position:position+features])
        train_y.append(y[position+1:position+1+features])
        position += features
        if position+1+features >= len(y):
            break

    return index, train_x, train_y

index, train_x, train_y = creat_data(x, y, 20, 2)
plt.figure(figsize=(16,2))
plt.plot(np.concatenate(index), np.concatenate(train_x))
plt.show()
#创建一个循环神经网络
class SimpleRNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.input_dim = 20
        self.h = torch.zeros((1, hid_dim))
        #注册参数
        self.hb = torch.nn.Parameter(torch.zeros(hid_dim, requires_grad=True))
        self.xh = torch.nn.Parameter(torch.normal(0, 0.01, (input_dim, hid_dim), requires_grad=True))
        self.hh = torch.nn.Parameter(torch.normal(0, 0.01, (hid_dim, hid_dim), requires_grad=True))
        self.o = torch.nn.Parameter(torch.normal(0, 0.01, (hid_dim, output_dim), requires_grad=True))

        #对参数进行注册
        # self.register_parameter('hb', self.hb)
    def forward(self, x):
        h = self.h
        outputs = []
        if self.training:
            for xi in x:
                h = Func.relu(torch.mm(xi, self.xh) + torch.mm(h, self.hh) + self.hb)
                y = torch.mm(h, self.o)
                outputs.append(y)
        else:
            xi = x[0]
            for i in range(self.input_dim):
                # print(xi)
                h = Func.relu(torch.mm(xi, self.xh) + torch.mm(h, self.hh) + self.hb)
                y = torch.mm(h, self.o)
                # print(y)
                xi = y.reshape(xi.shape)
                # print(xi)
                outputs.append(y)
        return outputs

print(SimpleRNN)
test_rnn = SimpleRNN(1, 5, 1)

for pi in test_rnn.parameters():
    print(pi)
xt, yt = train_x[0], train_y[0]
xt = torch.tensor(xt, dtype=torch.float32).reshape((20, 1, 1))
yt = torch.tensor(yt, dtype=torch.float32).reshape((20, 1))

# print(xi.shape)
print(xt)
print(yt)

y_hat = test_rnn(xt)

print(torch.concat(y_hat))

#执行训练
optimer = torch.optim.SGD(test_rnn.parameters(), lr=0.05)
loss = torch.nn.MSELoss()
epochs = 10
test_rnn.train()
for epoch in range(epochs):
    print(epoch)
    for xi, yi in zip(train_x, train_y):
        xi = torch.tensor(xi, dtype=torch.float32).reshape((20, 1, 1))
        yi = torch.tensor(yi, dtype=torch.float32).reshape((20, 1))

        y_hat = test_rnn(xi)
        y_hat = torch.concat(y_hat, 0)
        tmp_loss = loss(yi, y_hat)
        optimer.zero_grad()
        tmp_loss.mean().backward()
        optimer.step()
    print(tmp_loss)

#执行预测
index_list = []
predict_list = []
true_list = []
count = 0
# test_rnn.eval()
for i, xi, yi in zip(index, train_x, train_y):
    # if count % 10 == 0:
    xt = torch.tensor(xi, dtype=torch.float32).reshape((20, 1, 1))
    yt = torch.tensor(yi, dtype=torch.float32).reshape((20, 1))
    index_list.append(i)
    y_hat = test_rnn(xt)
    # print(y_hat)
    predict_list.append(torch.concat(y_hat).reshape((20, 1)).detach().numpy())
    true_list.append(yt.reshape((20, 1)).detach().numpy())
    count += 1

index_list = np.concatenate(index_list)
print(len(index_list))
predict_list = np.concatenate(predict_list)
true_list = np.concatenate(true_list)
plt.figure(figsize=(16, 2))
plt.plot(index_list, predict_list, c='r')
plt.plot(index_list, true_list, c='b')
plt.show()




