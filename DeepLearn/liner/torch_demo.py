import pandas as pd
import os
import torch
import numpy as np
from util.load_my_data import load_gyzq
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
train_data, test_data = load_gyzq()

#数据的读入及转换为tensor
feature_names = [i for i in train_data.columns if i != 'target']
train_x = train_data[feature_names]
train_y = train_data[['target']]
print(train_x.head())
print(train_y.head())
#将数据拆分为训练集和验证集
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

tensor_train_x = torch.tensor(train_x.values, dtype=torch.float32)
tensor_train_y = torch.tensor(train_y.values, dtype=torch.float32).reshape([-1, 1])

tensor_val_x = torch.tensor(val_x.values, dtype=torch.float32)
tensor_val_y = torch.tensor(val_y.values, dtype=torch.float32)

print(tensor_train_x.shape, tensor_train_y.shape)

#创建一个简单的线性回归模型
# w = torch.tensor(np.random.uniform(0, 1, tensor_train_x.shape[1]))
# bais = torch.tensor([1])
# print(w, bais)

w = torch.normal(0, 0.1, size=[tensor_train_x.shape[1]], requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

def liner_model(x, w, b):
    # print(x.dtype, w.dtype, b.dtype)
    y = torch.matmul(x, w)+b
    return y

def loss_function(y_true, y_pre):
    loss_score = (y_true.reshape(y_pre.shape) - y_pre)**2 / 2
    return loss_score/y_pre.shape[0]

def sgd(w, learn_rate, batch_size):
    with torch.no_grad():
        #清零梯度
        w -= learn_rate*w.grad/batch_size
        w.grad.zero_()

#执行训练
model = liner_model
loss = loss_function
learn_rate = 0.01

#创建一颗决策树进行对比
cart = LinearRegression()
cart.fit(train_x, train_y)
y = cart.predict(val_x)
print(metrics.mean_squared_error(val_y, y))

for epoch in range(2):
    for xi, yi in zip(tensor_train_x, tensor_train_y):
        tmp_l = loss_function(yi, liner_model(xi, w, bias))
        #计算梯度
        tmp_l.sum().backward()
        #跟新参数
        sgd(w, learn_rate, 1)

    #执行预测评分模型
    with torch.no_grad():
        pre_y = liner_model(tensor_val_x, w, bias)
        # val_loss = loss_function(tensor_val_y, pre_y)
        print('epoch {}: loss score: {}'.format(epoch, metrics.mean_squared_error(val_y, pre_y)))


#测试数组的引用
y = np.array([1, 2])
y = torch.tensor([1, 2])
def test_copy(y):
    y -= torch.tensor([0, 1])
    print(y)
test_copy(y)
print(y)

#高效版线性回归，使用torch内置函数
from torch import nn
#创建一个层
net = nn.Sequential(nn.Linear(tensor_val_x.shape[-1], 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(1)
ll = nn.MSELoss()
opt = torch.optim.SGD(net.parameters(), lr=0.01)
#执行训练
for epoch in range(50):
    for xi, yi in zip(tensor_train_x, tensor_train_y):
        #计算损失
        loss_score = ll(net(xi), yi)
        #将原始梯度清零
        opt.zero_grad()
        #反向传播计算梯度
        loss_score.backward()
        #更新梯度
        opt.step()
    #执行预测
    tmp_l = ll(net(tensor_val_x), tensor_val_y)
    print(epoch, tmp_l)





