#%%
#实现数据读取
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.ensemble import _stacking
from utils.load_my_data import load_gyzq
from IPython import display
pd.set_option('display.max_columns', None)
display.display('svg')
import warnings
warnings.filterwarnings('ignore')


#%%
source_data_root_path = 'e:/hw/personal/python/机器学习数据集/阿里云工业蒸汽'
train_data_file = os.path.join(source_data_root_path, 'zhengqi_train.txt')
test_data_file = os.path.join(source_data_root_path, 'zhengqi_test.txt')
train_data = pd.read_csv(train_data_file, sep='\t')
test_data = pd.read_csv(test_data_file, sep='\t')
# train_data, test_data = load_gyzq()
#简单查看数据情况
print(train_data.head())
print(train_data.describe())

feature_columns = [i for i in train_data.columns if i != 'target']
label_columns = ['target']

# sns.boxplot(data=train_data, orient="v")
# plt.show()
#
#%%
#查看数据的分布
rows = 15
cols = 6
#设置图片的尺寸 宽高
figure = plt.figure(figsize=(cols*5, rows*5))
i = 0
for index, column in enumerate(train_data.columns):
    print(column)
    i += 1
    figure = plt.subplot(rows, cols, i)
    #绘制第i行的数据分布
    sns.distplot(train_data[column], fit=stats.norm)
    # ax.set_xlabel(column)
    #绘制正太分布图
    i += 1
    fig = plt.subplot(rows, cols, i)
    stats.probplot(train_data[column], plot=plt)
    # ax.set_xlabel(column)

plt.show()
# plt.savefig('../result_data/tmp_qq.png')

#%%
#查看训练数据特征和测试数据特征的分布情况
i = 0
fig = plt.figure(figsize=(3*4,14*4))
for column in feature_columns:
    train_column_data = train_data[column]
    test_column_data = test_data[column]
    i+=1
    plt.subplot(14, 3, i)
    sns.kdeplot(train_column_data, color='r', shade=True)
    sns.kdeplot(test_column_data, color='b', shade=True)
    plt.xlabel(column)
    plt.ylabel('F')
    plt.legend(['train', 'test'])

plt.show()


#%%
#查看各个特征之间的相关性
plt.figure(figsize=(30, 30))
corr = train_data.corr()

sns.heatmap(train_data.corr(), annot=True, fmt='.2f', square=True, cmap='RdBu_r')

plt.show()
#筛选相关系数大于阈值的列
print(corr['target'])
threhold = 0.2
target_field = corr.index[abs(corr['target']) >= threhold]
no_field = corr.index[abs(corr['target']) < threhold]
print(no_field)
#查看各个特征之间的相关性
plt.figure(figsize=(30, 30))
sns.heatmap(train_data[target_field].corr(), annot=True, fmt='.2f', square=True, cmap='RdBu_r')
plt.show()

#%%
drop_feature_columns = set('V5 V11 V17 V22 '.split() + [i for i in no_field])
process_train_data = train_data.drop(columns=drop_feature_columns)
# print(process_train_data.head())
#将数据进行归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
final_feature_columns = [i for i in process_train_data.columns if i != 'target']
process_data_x = process_train_data[final_feature_columns]
process_data_y = process_train_data[['target']]

scaler.fit(process_data_x)
process_data_x = scaler.transform(process_data_x)
#使用pca进行降维
from sklearn.decomposition import PCA
pca = PCA(n_components=0.98)
pca.fit(process_data_x)
pca_data_x = pca.transform(process_data_x)
print(pca.explained_variance_, pca.explained_variance_ratio_)

#%%
#创建一个线性回归模型简单验证结果
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model = LinearRegression()
#测试原始的数据进行模型训练
source_train_x = train_data[feature_columns].values
source_train_y = train_data['target']

train_x, test_x, train_y, test_y = train_test_split(source_train_x, source_train_y, test_size=0.2, random_state=10, shuffle=True)
model.fit(train_x, train_y)
y_hat = model.predict(test_x)

print(test_y[:5])
print(mean_squared_error(test_y, y_hat))
import xgboost
xgb = xgboost.XGBRegressor(max_depth=4,
                    learning_rate=0.2,
                    n_estimators=300,)
xgb.fit(train_x, train_y)
y_hat = xgb.predict(test_x)
print('d:{} n:{} l:{} mse:{} '.format(8, 100, 0.8, mean_squared_error(test_y, y_hat)))

#%%
model = LinearRegression()
train_x, test_x, train_y, test_y = train_test_split(process_data_x, process_data_y, test_size=0.2, random_state=10, shuffle=True)
model.fit(train_x, train_y)
y_hat = model.predict(test_x)
print(test_y[:5])
print(mean_squared_error(test_y, y_hat))

import xgboost
xgb = xgboost.XGBRegressor(max_depth=5,
                    learning_rate=0.1,
                    n_estimators=300,)
xgb.fit(train_x, train_y)
y_hat = xgb.predict(test_x)
print('d:{} n:{} l:{} mse:{} '.format(8, 100, 0.8, mean_squared_error(test_y, y_hat)))


#%%
model = LinearRegression()
train_x, test_x, train_y, test_y = train_test_split(pca_data_x, process_data_y, test_size=0.2, random_state=10, shuffle=True)
model.fit(train_x, train_y)
y_hat = model.predict(test_x)
print(test_y[:5])
print(mean_squared_error(test_y, y_hat))

import xgboost
xgb = xgboost.XGBRegressor(max_depth=4,
                    learning_rate=0.2,
                    n_estimators=300,)
xgb.fit(train_x, train_y)
y_hat = xgb.predict(test_x)
print('d:{} n:{} l:{} mse:{} '.format(8, 100, 0.8, mean_squared_error(test_y, y_hat)))

#%%
print(np.linspace(0.5,1,10))
#%%
#使用xgboost模型
import xgboost
for d in range(5, 15):
    for l in np.linspace(0.1, 1, 10):
        for n in range(50, 150, 200):
            xgb = xgboost.XGBRegressor(max_depth=8,
                    learning_rate=0.8,
                    n_estimators=100,)
            xgb.fit(train_x, train_y)
            y_hat = xgb.predict(test_x)
            print('d:{} n:{} l:{} mse:{} '.format(d, n, l, mean_squared_error(test_y, y_hat)))


#%%
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost

#创建线性回归，knn, gbdt xgboost进行融合
liner = LinearRegression()
knn = KNeighborsRegressor()
gbdt = GradientBoostingRegressor(n_estimators=200, subsample=0.7, max_depth=5)
xgb = xgboost.XGBRegressor()

# for ai in np.linspace(0.1, 1, 10):
final_estimator = LinearRegression()
stacking_model = StackingRegressor([('lr', liner), ('knn', knn), ('gbdt', gbdt), ('xgb', xgb)], cv=5, final_estimator=final_estimator)
train_x, test_x, train_y, test_y = train_test_split(source_train_x, source_train_y, test_size=0.2, random_state=10, shuffle=True)
stacking_model.fit(train_x, train_y)
y_hat = stacking_model.predict(test_x)

print('惩罚权重：{} train loss: {} test loss: {}'.format('Liner', metrics.mean_squared_error(train_y, stacking_model.predict(train_x)), metrics.mean_squared_error(test_y, y_hat)))

#%%
#实现k折交叉验证
from sklearn import metrics
from sklearn.model_selection import KFold
kf = KFold(shuffle=True, random_state=2022)
train_mse_list = []
test_mse_list = []

for i, j in kf.split(source_train_x, source_train_y):
    #创建一个xgboost进行模型的验证测试
    xgb = xgboost.XGBRegressor(max_depth=4,
                    learning_rate=0.1,
                    n_estimators=200)
    xgb.fit(source_train_x[i], source_train_y[i])
    train_mse = metrics.mean_squared_error(source_train_y[i], xgb.predict(source_train_x[i]))
    test_mse = metrics.mean_squared_error(source_train_y[j], xgb.predict(source_train_x[j]))
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)

    print('train mse:{}  test mse: {}'.format(train_mse, test_mse))

print('mean train mse: {}  mean test mse:  {}'.format(np.array(train_mse_list).mean(), np.array(test_mse_list).mean()))
#%%

import xgboost
test_xgb = xgboost.XGBRegressor()
print(test_xgb)

from sklearn.model_selection import GridSearchCV
search_param = {'max_depth': [i for i in range(3, 8)],
                'learning_rate': np.linspace(0.1, 1, 10),
                'n_estimators': [50, 100, 150, 200, 250, 300]}
clg = GridSearchCV(xgboost.XGBRegressor(), param_grid=search_param, cv=5, return_train_score=True)
clg.fit(source_train_x, source_train_y)
model = clg.best_estimator_
print(model)
print(clg.cv_results_)

#%%
def test_desc(a, b):
    """

    param tip
    ---------
    :a first numbers

    :b second numbers

    :return:
    """








