#%%
import time
import os
import numpy as np
import math
import copy
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_process import MyData

#创建一个Encode类
class EnCode(nn.Module):
    def __init__(self, input_size, vocabulary_size, hidden_size, num_layers, dropout_rate=0.5):
        super(EnCode, self).__init__()
        self.input_size = input_size
        self.embedding_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = 0.5

        self.embedding = nn.Embedding(vocabulary_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, state=None):
        """
        :param x:
        :param state:
        :return:
        """
        #对传入的数据进行排序
        sort_x, sort_x_length, sort_index = self.sort_batch_data(x)
        #对解压之后的数据进行词嵌入
        x = self.dropout(self.embedding(sort_x))
        #对词嵌入之后的结果进行压缩
        x = nn.utils.rnn.pack_padded_sequence(x, sort_x_length, batch_first=True)
        #将结果传入到rnn中
        output, state = self.gru(x, state)

        #对返回的结果进行解压缩
        unpack_output, unpack_len = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

    def sort_batch_data(self, batch_data):
        """
        对传入的一个batch data 数据根据长度进行排序,返回完成排序的一个batch data数据
        :param batch_data:
        :return:
        """
        # nn.utils.rnn.pack_sequence()
        x, x_lengths = batch_data[0], batch_data[1]
        sort_x_length, sort_index = torch.sort(x_lengths, descending=True)
        sort_x = torch.index_select(x, 0, sort_index)

        return sort_x, sort_x_length, sort_index

#创建一个decode类
class DeCode(nn.Module):
    def __init__(self):
        super(DeCode, self).__init__()

    def forward(self, x, state):
        pass

#创建一个完成的seq2seq网络
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

    def forward(self, x, state):
        pass


def pad_data(X, Y, batch_size, padding=0):
    """
    对传入的数据集进行处理，先对数据根据从大到小的排序，然后填充
    :param X: 传入经 train test split 打乱的数据
    :param padding:
    :return: 返回x 以及对应的长度列表
    """
    #将数据转换为tensor格式
    tensor_x = [torch.tensor(xi) for xi in X]
    tensor_y = [torch.tensor(yi) for yi in Y]

    #获取每个数据的长度
    x_lengths = torch.tensor([li.shape[-1] for li in tensor_x])
    y_lengths = torch.tensor([li.shape[-1] for li in tensor_y])
    print(x_lengths)
    print(torch.max(x_lengths))
    print(y_lengths)
    print(torch.max(y_lengths))
    #对数据进行填充
    tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first=True, padding_value=padding)
    tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first=True, padding_value=padding)

    #将获取到的数据转换为
    tensor_dataset = TensorDataset(tensor_x, x_lengths, tensor_y, y_lengths)
    train_iter = DataLoader(tensor_dataset, batch_size, shuffle=True)

    return train_iter

def sort_batch_data(batch_data):
    """
    对传入的一个batch data 数据根据长度进行排序,返回完成排序的一个batch data数据
    :param batch_data:
    :return:
    """
    # nn.utils.rnn.pack_sequence()
    x, x_lengths = batch_data[0], batch_data[1]
    sort_x_length, sort_index = torch.sort(x_lengths, descending=True)
    sort_x = torch.index_select(x, 0, sort_index)

    return sort_x, sort_x_length, sort_index

def pack_data():
    """
    对传入的一个batch数据进行排序
    :return:
    """
    pass

def unpack_data():
    pass

if __name__ == "__main__":
    # x = torch.tensor([[1, 1, 3], [2, 3, 0], [3, 0, 0]])
    # y = torch.nn.Embedding(4, 4)(x)
    # # print(y)
    # p = nn.utils.rnn.pad_sequence([torch.tensor([1, 1, 3]), torch.tensor([2, 3])], batch_first=True)
    # print(p)
    # print(torch.nn.Embedding(4, 4)(p))
    #
    # print(nn.utils.rnn.pack_padded_sequence(y, torch.tensor([3, 2, 1]), batch_first=True))

#句子转换为数字  序列长度不一
#pad  再嵌入 则填充字段也会生成向量 该向量固定  序列不为零 后续会对填充序列删除掉
#想embedding 则生成序列长度不易 词嵌入向量固定 pad之后的序列为零 后续会对填充序列删除掉

    root_data_path = r'E:\hw\personal\python\Machine Learn\DeepLearn\seq2seq\dataset\en-cn'
    test1 = MyData(root_data_path)
    print(' create done')

    x = test1.mul_text2index(test1.x_text[:], language='en')
    y = test1.mul_text2index(test1.y_text[:], language='cn')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20220506)

    train_iter = pad_data(x_train, y_train, batch_size=3)
    for ti in train_iter:
        print(ti)
        #将结果输出出来
        print(test1.mul_index2text(ti[0].detach().numpy(), language='en'))
        print(test1.mul_index2text(ti[2].detach().numpy(), language='cn'))
        ti = sort_batch_data(ti)
        print(test1.mul_index2text(ti[0].detach().numpy(), language='en'))
        print(test1.mul_index2text(ti[2].detach().numpy(), language='cn'))


        #对生成的数据进行embedding
        embeddingg_tensor = nn.Embedding(test1.en_dic_len, 10)(ti[0])
        pack_tensor = nn.utils.rnn.pack_padded_sequence(embeddingg_tensor, ti[1], batch_first=True)

        print(embeddingg_tensor)
        print(ti[1])
        print(pack_tensor)

        break