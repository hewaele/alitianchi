import pandas as pd
import os
import numpy as py

def load_gyzq():
    root_data_path = r'E:\hewaele\python\my_code\alitianchi\dataset\工业蒸汽'
    train_data_file = os.path.join(root_data_path, 'zhengqi_train.txt')
    test_data_file = os.path.join(root_data_path, 'zhengqi_test.txt')

    train_data = pd.read_csv(train_data_file, sep='\t')
    test_data = pd.read_csv(test_data_file, sep='\t')

    return train_data, test_data

if __name__ == '__main__':
    pass


