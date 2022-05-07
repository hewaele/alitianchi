#创建一个resnet网络实现 cifar10
#%%
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, FashionMNIST
import torchvision.datasets as td

from PIL import Image
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('./tensorboard')
# for i in range(100):
#     print(i)
#     x = i
#     y = 2*i**2 + 0.1*x + 5
#     acc = i/100
#     #创建单个图 绘制单个曲线
#     writer.add_scalar('acc value', acc, global_step=x)
#     #绘制单图多曲线
#     writer.add_scalars('mul clue', {'y_hat': y, 'acc':acc}, global_step=x)
#     time.sleep(1)

#%%下载

transfunc = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])

# root_path = '../resnet'
# train_data = CIFAR10(root_path, train=True, transform=transfunc, download=True)
# test_data = CIFAR10(root_path, train=False, transform=transfunc, download=True)

root_path = '../simple_vgg/data'
train_data = FashionMNIST(root_path, train=True, transform=transfunc, download=True)
test_data = FashionMNIST(root_path, train=False, transform=transfunc, download=True)
tensor_train = DataLoader(train_data, batch_size=32)
tensor_test = DataLoader(test_data, batch_size=32)

print(len(tensor_train))
print(len(tensor_test))

train_iter =tensor_train
test_iter = tensor_test

#%%
import numpy as np
# xi = next(train_iter)[0][0]
# print(xi.shape)
# image_type = np.uint8(np.array(xi*255)).transpose(1, 2, 0)
# print(image_type)
# fig = Image.fromarray(np.uint8(np.array(xi*255)).transpose(1, 2, 0))
# fig.show()

#%%
#创建resnet block
class ResnetBlock(torch.nn.Module):
    def __init__(self, input_features, output_features, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Sequential(nn.Conv2d(input_features, output_features, 3, self.stride, padding),
                                   nn.BatchNorm2d(output_features),
                                   nn.ReLU())
        self.conv2 = nn.Conv2d(output_features, output_features, 3, 1, padding)
        self.bn = nn.BatchNorm2d(output_features)
        self.conv3 = nn.Conv2d(self.input_features, self.output_features, 1, self.stride, 0)
        self.act = nn.ReLU()

    def forward(self, X):
        hid = self.conv1(X)
        hid = self.conv2(hid)
        hid = self.bn(hid)

        #判断是否对输入进行1x1卷积
        if self.stride == 2:
            X = self.conv3(X)
        hid += X
        #对最后的结果进行激活
        return self.act(hid)

header = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
# print(next(train_iter))
test_block = ResnetBlock(64, 128, 2, 1)
test_net = nn.Sequential(header, test_block)
# print('执行块测试')
# y_hat_test = test_net(next(train_iter)[0])
# print(y_hat_test.shape)
# print('快测试完成')

#%%
#船舰完整网络
header = nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))

b2 = nn.Sequential(ResnetBlock(64, 64, 1, 1), ResnetBlock(64, 64, 1, 1))
b3 = nn.Sequential(ResnetBlock(64, 128, 2, 1), ResnetBlock(128, 128, 1, 1))
b4 = nn.Sequential(ResnetBlock(128, 256, 2, 1), ResnetBlock(256, 256, 1, 1))
b5 = nn.Sequential(ResnetBlock(256, 512, 2, 1), ResnetBlock(512, 512, 1, 1))

out = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 10))
my_net = nn.Sequential(header, b2, b3, b4, b5, out)

#使用科学模式会报错 无法保存网路结构图
writers = SummaryWriter('./resnet_log', comment='resnet')
# with SummaryWriter(comment='LeNet') as w:
#     w.add_graph(my_net, [next(train_iter)[0], ])
writers.add_graph(my_net, [torch.rand((32, 1, 224, 224)), ])

#对网路进行初始化
def init_weight(ni):
    if type(ni) == nn.Conv2d:
        nn.init.kaiming_uniform_(ni.weight)
        nn.init.zeros_(ni.bias)
    elif type(ni) == nn.Linear:
        nn.init.xavier_normal_(ni.weight)
        nn.init.zeros_(ni.bias)
    elif type(ni) == nn.BatchNorm2d:
        nn.init.ones_(ni.weight)
        nn.init.zeros_(ni.bias)

print(my_net)

# my_net.eval()
# print('执行网络测试')
#
# y_hat_test = my_net(next(train_iter)[0])
# print(y_hat_test.shape)
# print('执行网络完成')

#执行网络训练
loss = nn.CrossEntropyLoss()
optimer = torch.optim.SGD(my_net.parameters(), lr=0.008)
epochs = 100
nn.Dropout()
device = torch.device('cuda:0')
#执行网络训练
my_net = my_net.to(device)
for epoch in range(epochs):
    train_ac = 0
    train_count = 0
    train_loss = 0

    for xi, yi in train_iter:
        # print(xi.shape[0])
        xi = xi.to(device)
        yi = yi.to(device)
        y_hat = my_net(xi)
        tmp_loss = loss(y_hat, yi)
        optimer.zero_grad()
        tmp_loss.mean().backward()
        optimer.step()
        train_ac += torch.sum(torch.argmax(y_hat, dim=1) == yi, dtype=torch.float32)
        train_count += xi.shape[0]
        train_loss += tmp_loss.sum()
        # break

    #计算训练平均
    train_ac /= train_count
    train_loss /= train_count

    #查看训练结果
    with torch.no_grad():
        ac = 0
        count = 0
        tmp_loss = 0
        for xi, yi in test_iter:
            xi = xi.to(device)
            yi = yi.to(device)
            y_hat = my_net(xi)
            tmp_loss += loss(y_hat, yi).sum()
            ac += torch.sum(torch.argmax(y_hat, dim=1) == yi, dtype=torch.float32)
            count += xi.shape[0]
            # print(tmp_loss, ac, count)
            # break
        print(ac, count, tmp_loss)
        ac /= count
        tmp_loss /= count

        writers.add_scalars('train_clue',
                            {'val_loss': tmp_loss, 'val_ac': ac,
                            'train_loss': train_loss, 'train_ac': train_ac},
                            global_step=epoch)
        print('epoch:{} train loss: {} train ac: {} val loss: {} val ac: {}'.
              format(epoch, train_loss, train_ac, tmp_loss, ac))

torch.save(my_net, './my_net.pt')

# load_net = torch.load('./my_net.pt')
# print(load_net)