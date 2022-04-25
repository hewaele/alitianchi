#%%
import torch
from torch import nn
import torchvision
import os
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#%%
data_x, data_y = datasets._samples_generator.make_classification(10000, 2,
                                                                 n_informative=2,
                                                                 n_redundant=0,
                                                                 n_repeated=0,
                                                                 n_classes=3,
                                                                 n_clusters_per_class=1, random_state=200)
# data_x, data_y = datasets._samples_generator.make_moons(500, shuffle=True, noise=0.1, random_state=2022)
# plt.plot([0, 1], [0, 1])
plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y)
plt.show()
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=10)

tensor_train_x, tensor_train_y = torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)
tensor_test_x, tensor_test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long)

#将数据迁移到gpu
device = torch.device('cuda:0')
tensor_train_x = tensor_train_x.to(device)
tensor_train_y = tensor_train_y.to(device)
tensor_test_y = tensor_test_y.to(device)
tensor_test_x = tensor_test_x.to(device)

#creat dateloader
from torch.utils.data import TensorDataset, DataLoader
tensor_dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
tensor_dataiter = DataLoader(tensor_dataset, 32, drop_last=False)
print('done')

#%%
from sklearn import metrics
#常见一个简单的softmax分类网络
net = torch.nn.Sequential(torch.nn.Linear(tensor_train_x.shape[1], 100),
                          torch.nn.ReLU(),
                          torch.nn.Linear(100, 50),
                          torch.nn.ReLU(),
                          torch.nn.Linear(50, 3),
                          torch.nn.Softmax())
net_loss = torch.nn.CrossEntropyLoss()
net_optimer = torch.optim.SGD(net.parameters(), 0.1)
#将模型迁移到gpu
net.to(device)
for index, ni in enumerate(net):
    if type(ni) == nn.Linear:
        nn.init.normal_(net[index].weight, std=0.01)
        nn.init.ones_(net[index].bias)
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)

# net.apply(init_weights)
#执行训练
epochs = 100
for epoch in range(epochs):
    for xi, yi in tensor_dataiter:
        y_hat = net(xi)
        tmp_loss = net_loss(y_hat, yi)
        net_optimer.zero_grad()
        tmp_loss.mean().backward()
        net_optimer.step()

    with torch.no_grad():
        y_hat = net(tensor_train_x)
        tmp_loss = net_loss(y_hat, tensor_train_y)
        ac = torch.sum(tensor_train_y == torch.argmax(y_hat, dim=1))/tensor_train_y.shape[0]
        print('epoch:{} tmp_loss:{} accuary:{}'.format(epoch, tmp_loss, ac))

#%%
#使用决策数进行分类
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(train_x, train_y)
y_hat = clf.predict(test_x)

print(metrics.accuracy_score(test_y, y_hat))

#%%
#使用现成的网络绘制分界线
xmin, xmax = test_x[:, 0].min(), test_x[:, 0].max()
ymin, ymax = test_x[:, 1].min(), test_x[:, 1].max()
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01),
                         np.arange(ymin, ymax, 0.1))

print(xx.shape)
#绘制图
z = net(torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32))
z = np.argmax(z.detach().numpy(), axis=1).reshape(xx.shape)

# z = clf.predict(np.stack([xx.ravel(), yy.ravel()], axis=1))
# z = z.reshape(xx.shape)

print(z.shape)

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#分别对应rgb
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.contourf(xx, yy, z)
#绘制点
plt.scatter(test_x[:,0], test_x[:, 1], c=test_y)
plt.show()


