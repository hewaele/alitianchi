import numpy as np
import torch

#%%
#创建一个张量x
#这里有个坑，只能计算浮点数的梯度，不可计算int的梯度
x = torch.arange(4.0, requires_grad=True)
print(x, x.shape)

#%%
#倒数只能为标量，不可为向量或者张量
x.grad.zero_()
y = 2*x
print(x)
#将y这个符合运算看作u，则再第一步计算梯度时u被当成常数
u = y.detach()
print(y.detach())
print(y)
print(y.sum())
y.sum().backward()
print(x.grad)
y.sum().backward()
print(x.grad)
# z = u*x
# z.sum().backward()
# z.sum
# print(x.grad)

#