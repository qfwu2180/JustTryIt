# coding=UTF-8
import numpy as np
import pandas as pd
import tushare as ts
from sklearn.covariance import GraphLassoCV, GraphicalLassoCV
from sklearn.cluster import affinity_propagation
import xlrd


def get_onedata(code, start, end):
    df = ts.get_k_data(code, start=start, end=end)  # dataframe with columns [date, open, close, high, low]
    return df


def get_kdata(codes_list, start, end):
    df = pd.DataFrame()
    for code in codes_list:
        # print('fetching K data of {}...'.format(code))
        df = df.append(get_onedata(code, start, end))
    return df


def preprocess_data(stock_df, min_K_num=1000):
    df = stock_df.copy()
    df['diff'] = df.close - df.open
    df.drop(['open', 'close', 'high', 'low'], axis=1, inplace=True)
    # print(df)
    result_df = None
    for name, group in df[['date', 'diff']].groupby(df.code):
        if len(group.index) < min_K_num: continue
        if result_df is None:
            result_df = group.rename(columns={'diff': name})
        else:
            result_df = pd.merge(result_df, group.rename(columns={'diff': name}), on='date', how='inner')
            # 对应相同的date 排序

    result_df.drop(['date'], axis=1, inplace=True)
    # 将股票数据DataFrame转变为np.ndarray
    stock_dataset = np.array(result_df).astype(np.float64)
    # 数据归一化
    stock_dataset /= np.std(stock_dataset, axis=0)
    return stock_dataset, result_df.columns.tolist()


def preprocess_data2(stock_df, min_K_num=1000):
    df = stock_df.copy()
    df.drop(['open', 'close', 'high', 'low'], axis=1, inplace=True)
    result_df = None
    for name, group in df[['date', 'volume']].groupby(df.code):
        if len(group.index) < min_K_num: continue
        if result_df is None:
            result_df = group.rename(columns={'volume': name})
        else:
            result_df = pd.merge(result_df, group.rename(columns={'volume': name}), on='date', how='inner')
    result_df.drop(['date'], axis=1, inplace=True)
    # 然后将股票数据DataFrame转变为np.ndarray
    stock_dataset = np.array(result_df).astype(np.float64)
    # 数据归一化，此处使用相关性而不是协方差的原因是在结构恢复时更高效
    stock_dataset /= np.std(stock_dataset, axis=0)
    return stock_dataset, result_df.columns.tolist()


data = xlrd.open_workbook('nlmy.xlsx')
table = data.sheet_by_name('Animal_husbandry')
stock_list = table.col_values(colx=1, start_rowx=1, end_rowx=33)
stock_list = [str(i) for i in stock_list]

batch_K_data = get_kdata(stock_list, start='2013-09-01', end='2018-09-01')  # 查看最近五年的数据
# print(batch_K_data.info())

stock_dataset, selected_stocks = preprocess_data(batch_K_data, min_K_num=1100)
stock_dataset2, selected_stocks2 = preprocess_data2(batch_K_data, min_K_num=1100)
print("The selected stocks is:  ",
      selected_stocks)  # 这是实际使用的股票列表stock_dataset,selected_stocks=preprocess_data2(batch_K_data,min_K_num=1100)

# 从相关性中学习其图形结构
edge_model1 = GraphicalLassoCV(cv=3)
edge_model2 = GraphicalLassoCV(cv=3)
# edge_model.fit(stock_dataset)
edge_model1.fit(stock_dataset)
edge_model2.fit(stock_dataset2)

# 使用近邻传播算法构建模型，并训练LassoCV graph
_, labels1 = affinity_propagation(edge_model1.covariance_)
_, labels2 = affinity_propagation(edge_model2.covariance_)

n_labels = max(labels1)
print('Stock Clusters: {}'.format(n_labels + 1))  # 10，即得到10个类别
sz50_df2 = stock_list
# print(sz50_df2)
for i in range(n_labels + 1):
    print('Cluster: {}----> stocks: {}'.format(i, ','.join(np.array(selected_stocks)[labels1 == i])))  # 这个只有股票代码而不是股票名称
    stocks = np.array(selected_stocks)[labels1 == i].tolist()
    # names = sz50_df2.loc[stocks, :].name.tolist()
    # print('Cluster: {}----> stocks: {}'.format(i,','.join(names)))

n_labels = max(labels2)
print('Stock Clusters: {}'.format(n_labels + 1))  # 10，即得到10个类别
sz50_df2 = stock_list
# print(sz50_df2)
for i in range(n_labels + 1):
    print('Cluster: {}----> stocks: {}'.format(i, ','.join(np.array(selected_stocks)[labels2 == i])))  # 这个只有股票代码而不是股票名称
    stocks = np.array(selected_stocks)[labels1 == i].tolist()
    # names = sz50_df2.loc[stocks, :].name.tolist()
    # print('Cluster: {}----> stocks: {}'.format(i,','.join(names)))
