#%%
import time
import os
import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
# nltk.download('punkt', '../')

# s = nltk.word_tokenize("'this is a test, you know? maybe you don't know.' Hobbls said! haha...")
# s = nltk.sent_tokenize("'this is a test, you know? maybe you don't know.' Hobbls-- said!")
# s = nltk.load("tokenizers/punkt/{0}.pickle".format('english'))
# print(s)

#创建一个文本处理基类
#实现文本读取， tokens生成，文本索引字典生成  字典索引生成
class DataProcess():
    def __init__(self, root_data_path='../dataset/en-cn'):
        self.root_data_path = root_data_path
        self.default_dic = {'<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<PAD>': 0}
        self.x_text, self.y_text = self.load_data(self.root_data_path)

    def load_data(self, root_data_path):
        en = []
        cn = []
        for file in os.listdir(root_data_path):
            # print(file)
            with open(os.path.join(root_data_path, file), 'r', encoding='utf-8') as fb:
                for line in fb.readlines():
                    line = line.replace('    ', '\t').strip().split('\t')
                    en.append(line[0])
                    cn.append(line[1])
        return en, cn

    #创建分词预料库
    def get_text2tokens(self, text_list, language='en'):
        tokens_list = []
        if not isinstance(text_list, list):
            text_list = [text_list]
        if language == 'en':
            for xi in text_list:
                tokens_list.append(['<BOS>'] + nltk.word_tokenize(xi) + ['<EOS>'])
        else:
            for text in text_list:
                tokens_list.append(['<BOS>'] + [wi for wi in text] + ['<EOS>'])
        return tokens_list

    def build_vocabulaty(self, tokens_list, position=4):
        """
        传入tokens列表生成字典
        :param tokens_list:
        :param position:
        :return:
        """
        word_dic = copy.deepcopy(self.default_dic)
        for tokens in tokens_list:
            for token in tokens:
                if word_dic.get(token) is None:
                    word_dic[token] = position
                    position += 1
        #生成索引字典
        index_dic = {vi: ki for ki, vi in word_dic.items()}
        return word_dic, index_dic, len(word_dic)

    def get_tokens_index(self, tokens, vocabulary_dic):
        index = []

        for token in tokens:
            tmp_index = vocabulary_dic.get(token)
            if tmp_index is not None:
                index.append(tmp_index)
            else:
                index.append(vocabulary_dic.get('<UNK>'))
        return index

    def get_multokens_index(self, tokens_list, vocabulary_dic):
        index_list = []
        for tokens in tokens_list:
            index_list.append(self.get_tokens_index(tokens, vocabulary_dic))
        return index_list

    def get_index_token(self, index, index_dic):
        tokens = []
        for i in index:
            tmp_token = index_dic.get(i)
            if tmp_token:
                tokens.append(tmp_token)
            else:
                print('index dic error')
        return tokens

    def get_mulindex_token(self, index_list, index_dic):
        tokens_list = []
        for index in index_list:
            tokens_list.append(self.get_index_token(index, index_dic))
        return tokens_list


#创建一个实际任务数据处理类
class MyData(DataProcess):
    def __init__(self, root_data_path):
        super(MyData, self).__init__(root_data_path)
        self.get_en_cn_vocabulary()

    #获取中因为字典表
    def get_en_cn_vocabulary(self):
        #先获取tokens
        self.en_tokens = self.get_text2tokens(self.x_text, language='en')
        self.cn_tokens = self.get_text2tokens(self.y_text, language='cn')

        #创建字典
        self.en_word_dic, self.en_index_dic, self.en_dic_len = self.build_vocabulaty(self.en_tokens)
        self.cn_word_dic, self.cn_index_dic, self.cn_dic_len = self.build_vocabulaty(self.cn_tokens)


    def get_word_vocabulary(self, language):
        if language == 'en':
            return self.en_word_dic
        else:
            return self.cn_word_dic

    def get_index_vocabulary(self, language):
        if language == 'en':
            return self.en_index_dic
        else:
            return self.cn_index_dic

    def single_text2index(self, text, language='en', cut=False):
        """

        :param text:
        :param cut:
        :return:
        """
        vocabulary = self.get_word_vocabulary(language)

        #将句子转换为tokens
        if not isinstance(text, list):
            text = [text]
        cur_tokens = self.get_text2tokens(text,language)[0]
        cur_index = self.get_tokens_index(cur_tokens, vocabulary)
        return cur_index

    def mul_text2index(self, text, language='en', cut=False):
        vocabulary = self.get_word_vocabulary(language)

        cur_tokens_list = self.get_text2tokens(text, language)
        cur_index_list = self.get_multokens_index(cur_tokens_list, vocabulary)
        return cur_index_list

    def single_index2text(self, index, language='en', cut=False, merge=True):
        """

        :param text:
        :param cut:
        :return:
        """
        if language == 'en':
            vocabulary = self.en_index_dic
        else:
            vocabulary = self.cn_index_dic

        if not isinstance(index, list):
            index = [index]
        cur_tokens = self.get_index_token(index, vocabulary)
        #将token合并
        if language == 'en':
            return ' '.join(cur_tokens)
        else:
            return ''.join(cur_tokens)

    def mul_index2text(self, index_list, language='en', cut=False, merge=True):
        vocabulary = self.get_index_vocabulary(language)

        cur_tokens = self.get_mulindex_token(index_list, vocabulary)

        merge_tokens = []
        for ti in cur_tokens:
            if language == 'en':
                merge_tokens.append(' '.join(ti))
            else:
                merge_tokens.append(''.join(ti))
        return merge_tokens



if __name__ == '__main__':
    root_data_path = r'E:\hw\personal\python\Machine Learn\DeepLearn\seq2seq\dataset\en-cn'
    test1 = MyData(root_data_path)
    print(' create done')

    #%%
    # print(test1.get_text2tokens(['this is a test', 'the string whose method is called'], 'en'))
    # print(test1.cn_word_dic.get('<BOS>'))
    # print(test1.cn_index_dic.get(0))
    # print(test1.default_dic.get('<BOS>'))

    #生成数据
    from sklearn.model_selection import train_test_split
    x = test1.mul_text2index(test1.x_text[:], language='en')
    y = test1.mul_text2index(test1.y_text[:], language='cn')

    print(x[:10])
    print(y[:10])

    # x = test1.get_text2tokens(test1.x_text[:], language='en')
    # y = test1.get_text2tokens(test1.y_text[:], language='cn')
    # print(x)
    # print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20220506)
    print(x_train[:10])
    print(y_test[:10])