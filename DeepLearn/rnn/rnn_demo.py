#使用内置库实现简单rnn模型
import torch
from torch.utils import data as td
from torch.nn import functional as Func
import numpy as np
import tensorboard
import os
import time
import re
import matplotlib.pyplot as plt

num_steps = 20

#读取文本信息生成token
def text_process(text_path):
    """
    对文本进行根据句子进行分割，再对单词进行分割
    :param text_path:
    :return:
    """
    fb = open(text_path, 'r', encoding='utf-8')

    sentences = fb.read().replace('\n', ' ').replace('\'', '').split('.')
    sentences_result = []
    for si in sentences[:]:
        si = re.sub('[^a-zA-Z0-9]+', ' ', si.lower())
        if len(si) > 3:
            sentences_result.append(si)

    return sentences_result

#生成词表
def create_vocab(sentences):
    vocab_dic = {'<unk>': 0, '<start>':1, '<eof>': 2, '<pad>':3}
    position = 4
    for si in sentences:
        for wi in si.split():
            if not vocab_dic.get(wi):
                vocab_dic[wi] = position
                position += 1

    return vocab_dic, position

def get_token(sentence, vocab_dic):
    #将句子拆分
    token_result = []
    if not isinstance(sentence, list):
        sentence = sentence.split()
    for si in sentence:
        position = vocab_dic.get(si)
        if position:
            token_result.append(position)
        else:
            token_result.append(0)
    return token_result

text_path = './timemachine.txt'
sentences = text_process(text_path)
vocab_dic, vocab_size = create_vocab(sentences)
ti = get_token(sentences[10], vocab_dic)
#统计句子的长度
len_list = []
i = 0
for si in sentences:
    i+=1
    len_list.append(len(si.split()))
print(i)
from collections import Counter
ri = Counter(len_list)
print(ri)
print(sentences[10])
print(ti)

#生成训练数据
def create_train_data(sentences, vocab_dic):
    xs = []
    ys = []
    for sentence in sentences:
        #判断当前句子是否满足长度要求
        words = sentence.split()
        if len(words) > num_steps:
            x = get_token(words[:num_steps], vocab_dic)
            y = get_token(words[1:num_steps+1], vocab_dic)
        else:
            x = get_token(words[:-2], vocab_dic)
            y = get_token(words[1:-1], vocab_dic)
            #对结果进行填充
            x.extend([3 for i in range(num_steps-len(x))])
            y.extend([3 for i in range(num_steps-len(y))])
        xs.append(x)
        ys.append(y)

    return xs, ys

x, y = create_train_data(sentences, vocab_dic)
print(sentences[1])
print(x[:5])
print(y[:5])

embedding = torch.nn.Embedding(vocab_size, 30)
# print(embedding(torch.LongTensor(x[0:3])))

from sklearn.model_selection import train_test_split

#rnn自动读取批量 循环次数 输入向量长度 隐藏层大小 隐藏层深度
rnn = torch.nn.RNN(30, 50, 2)
ei = embedding(torch.LongTensor(x[0])).reshape(-1, num_steps, 30)
print(ei.shape)
yi, state = rnn(torch.permute(ei, (1, 0, 2)))
print(yi)
print(yi.shape)
print(state.shape)

#创建rnn模型
class MyRNN(torch.nn.Module):
    def __init__(self, rnn_layer, **kwargs):
        super(MyRNN, self).__init__(**kwargs)
        self.rnn_layer = rnn_layer
        self.liner = torch.nn.Linear(50, vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, 50)

    def forward(self, X, state):
        #对输入的x进行embedding
        # x = self.embedding(torch.LongTensor(X))
        #将x传入到rnn中
        y, state = self.rnn_layer(X, state)

        #对获取的结果进行全连接输出分类层
        y = self.liner(y)
        return y, state

#rnn 输入维度为  序列长度  批量长度  输入维度大小
#rnn 输出维度为  序列长度  批量长度  输出维度大小
test_net = MyRNN(rnn)
y_hat, state = test_net(torch.permute(embedding(torch.LongTensor(x[0:3])), (1, 0, 2)), torch.zeros((2, 3, 50)))
print(y_hat)
print(y_hat.shape)
print(state)
print(torch.argmax(y_hat, dim=2).shape)
#计算损失
# yt = torch.permute(torch.Tensor(y[0:3]), (1, 0))
# print(yt.shape)

#将网络的参数进行初始化
print(test_net)
for pi in test_net.parameters():
    print(pi)
loss = torch.nn.CrossEntropyLoss()
optimer = torch.optim.SGD(test_net.parameters(), 0.5)

epochs = 50
for epoch in range(epochs):

    for xi, yi in zip(x, y):
        xi = torch.permute(embedding(torch.LongTensor(xi)).reshape(-1, 20, 30), (1, 0, 2))
        yi = torch.LongTensor(yi).T.reshape(-1)
        # print(yi)
        y_hat, state = test_net(xi, torch.zeros((2, 1, 50)))
        # y_hat = torch.argmax(y_hat, dim=2)
        # print(y_hat.reshape(20, -1).shape)
        tmp_loss = loss(y_hat.reshape((20*1, 4600)), yi).mean()
        optimer.zero_grad()
        tmp_loss.backward()
        optimer.step()
        # print(tmp_loss)
        # break

    #预测第四个句子
    print(x[4])
    print(sentences[4])
    y_hat_test, state_test = test_net(torch.permute(embedding(torch.LongTensor(x[4])).reshape(-1, 20, 30), (1, 0, 2)), torch.zeros((2, 1, 50)))
    y_hat_test = torch.argmax(y_hat_test.reshape((20*1, 4600)), dim=1)
    print(y_hat_test)
    index2word = {vi:ki for ki, vi in vocab_dic.items()}
    labels = [index2word.get(int(pi)) for pi in y_hat_test]
    print(labels)