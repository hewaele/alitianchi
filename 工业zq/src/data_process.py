import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    source_data_root_path = 'e:/hw/personal/python/机器学习数据集/阿里云工业蒸汽'
    train_data_file = os.path.join(source_data_root_path, 'zhengqi_train.txt')
    test_data_file = os.path.join(source_data_root_path, 'zhengqi_test.txt')
    train_data = pd.read_csv(train_data_file, sep='\t')
    test_data = pd.read_csv(test_data_file, sep='\t')

    #简单查看数据情况
    print(train_data.head())
    print(train_data.describe())

    sns.boxplot(data=train_data)
    # sns.boxplot()

    sns.lineplot(data=[-7.5, 7.5])
    plt.show()

