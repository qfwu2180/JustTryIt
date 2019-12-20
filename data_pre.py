import numpy as np
import pandas as pd
from sklearn import preprocessing


class StockData(object):
    # 对每一支股票进行处理
    def __init__(self, stock_code, target_size, num_steps, train_ratio, interval, future):
        """
        一个Stock_Data对象对应一只股票历史数据，在Stock_count中将股票历史交易数据处理成模型需要的训练集和测试集
        :param stock_code: 股票的编码
        :param target_size: 决定target的涨跌幅是几日的平均，例如涨跌幅取一个月后五天内的平均涨跌幅，input_size=5
        :param num_steps: 决定输入LSTM神经网络的时序长度
        :param train_ratio: 训练集占所有样本的比率
        :param interval: interval用于将收盘价处理成interval天内的涨跌幅,interval = batch_size
        :param future: future表示用来预测多少天之后的涨跌幅
        """
        self.stock_code = stock_code
        self.target_size = target_size
        self.num_steps = num_steps
        self.train_ratio = train_ratio
        self.interval = interval
        # interval表示将价格换成涨跌率的时间间隔，比如将50天之内的价格换成涨跌率interval=50，就是用50天内的价格去除以50天之前的最后一天的价格
        self.future = future
        # future表示取多久之后的平均值，比如预测30个交易日之后的10日平均值，future=30，target_size=10

        df = pd.read_csv('Data\\交易数据\\' + stock_code + '.csv')
        '''将该股的收盘价每interval天计算成涨跌率，'''
        close = df['close'].to_numpy().tolist()
        vol = df['vol'].to_numpy().tolist()

        self.close_changed = []
        self.period_vol = []
        for i in range(0, len(close), interval):  # range(start, end, step)
            if (len(close) - i) > interval:
                for j in range(1, interval+1):
                    self.close_changed.append((close[i + j] - close[i]) / close[i])
                    self.period_vol.append(vol[i + j])
            else:
                for j in range(1, len(close) - i):
                    self.close_changed.append((close[i + j] - close[i]) / close[i])
                    self.period_vol.append(vol[i + j])

        '''数据归一化'''
        self.trans_close = preprocessing.minmax_scale(self.close_changed)  # 价格涨跌率归一化
        self.trans_vol = preprocessing.minmax_scale(self.period_vol)       # 日成交量归一化

        '''用归一化后的数据来构建输入data_x, 因为模型需要归一化数据'''
        self.data = list(zip(self.trans_close, self.trans_vol))
        data_length = len(self.data) - num_steps - self.target_size - self.future  # data_x 和 data_y的长度
        self.data_x = np.array([self.data[i:i + num_steps] for i in range(data_length)]).astype('float32')

        '''用未归一化的数据来构建标签data_y，因为归一化后的数据会失真'''
        self.data_y = []
        for i in range(data_length):
            change_y = 0
            for j in range(target_size):
                change_y += self.close_changed[i + num_steps + self.future - j]
            self.data_y.append([change_y / self.target_size])  # data_y是时间点 i 之后 future 天 target_size天内的平均涨跌幅
        self.data_y = np.array(self.data_y).astype('float32')

        '''划分训练集测试集'''
        self.train_x, self.test_x = np.split(self.data_x, [int(len(self.data_x) * self.train_ratio)])
        self.train_y, self.test_y = np.split(self.data_y, [int(len(self.data_y) * self.train_ratio)])


def get_stocks(target_size, num_steps, train_ratio, interval, future):
    """
    获取每只股票对应的Stock_Data对象，读取数据文件中的股票数据
    :return 每只股票的训练集，统一的测试集，股票名,
    """
    stocks_df = pd.read_csv('C:/Users/20929/PycharmProjects/Quant/Data/交易数据/download_stocks_symbol.csv')
    stock_name = stocks_df['code'].values.tolist()
    # 处理成Stock_Data对象
    stock_data = [
        StockData(stock_code=code,
                  target_size=target_size,
                  num_steps=num_steps,
                  train_ratio=train_ratio,
                  interval=interval,
                  future=future)
        for code in stock_name
    ]

    return stock_data
