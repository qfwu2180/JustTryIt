from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.python as tf
from tensorflow.keras import layers
import data_pre

'''get stock data'''
stock_data = data_pre.get_stocks(
    target_size=10,
    num_steps=250,
    train_ratio=0.8,
    interval=30,
    future=30
)
# stock_data[] {
#     :param stock_code: 股票的编码
#     :param target_size: 决定target的涨跌幅是几日的平均，例如涨跌幅取一个月后五天内的平均涨跌幅，input_size=5
#     :param num_steps: 决定输入LSTM神经网络的时序长度
#     :param train_ratio: 训练集占所有样本的比率
#     :param interval: interval用于将收盘价处理成interval天内的涨跌幅
#     :param future: future表示用来预测多少天之后的涨跌幅
#     train_x: shape = [data_length * train_ratio, num_steps, 2]
#     train_y: shape = [data_length * train_ratio, 1]
#     test_x:  shape = [data_length * (1-train_ratio), num_steps, 2]
#     test_y:  shape = [data_length * (1-train_ratio), 1]
# }

'''define hyper parameters:'''
num_steps = 250
batch_size = 20
num_batches = 25
num_epochs = 3
learning_rate = 1e-3

# class LSTM(tf.keras.Model):
#     def __init__(self, batch_size, num_steps):
#         super(LSTM, self).__init__()
#         self.batch_size = batch_size
#         self.num_steps = num_steps
#         self.lstm_layer = tf.keras.layers.LSTMCell(units=128)
#         self.dense = tf.keras.layers.Dense(units=1)
#
#     def call(self, inputs, training=None, mask=None):
#         states = self.lstm_layer.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
#         for t in range(self.num_steps):
#             temp, states = self.lstm_layer(inputs[:, t, :], states)
#         return self.dense(temp)

# define the network

# bulid the model
inputs = tf.keras.Input(shape=(num_steps, 2), name='input')
x = layers.GRU(128)(inputs)
# x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1,)(x)
model = tf.keras.Model(inputs, outputs)
print("model initialized")

# choose the first stock
data = stock_data[0]
# compile model and train and evaluate it
print("stock code: ", data.stock_code)
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
)
model.fit(
    x=tf.convert_to_tensor(data.train_x),
    y=tf.convert_to_tensor(data.train_y),
    batch_size=batch_size,
    epochs=num_epochs,
    verbose=1,
    # validation_split=0.2,
    initial_epoch=0,
    steps_per_epoch=num_batches,
    # validation_freq=2,
    use_multiprocessing=True
)
test_loss = model.evaluate(
    x=data.test_x,
    y=data.test_y,
    verbose=1,
    use_multiprocessing=False
)
print("test_loss:", test_loss)
model.summary()


# model.compile(
#     optimizer=optimizer,
#     loss=tf.keras.losses.mean_absolute_percentage_error(data.data_y, y_pred=model(data.data_x))
# )
#
# history = model.fit(
#     x=data.train_x,
#     y=data.train_y,
#     batch_size=batch_size,
#     epochs=num_epochs,
#     verbose=1,
#     validation_split=0.2,
#     initial_epoch=0,
#     steps_per_epoch=num_batches,
#     validation_freq=2,
#     use_multiprocessing=False
# )
# print(history)
#
# test_loss = model.evaluate(
#     x=data.test_x,
#     y=data.test_y,
#     verbose=1,
#     use_multiprocessing=False
# )
# print(test_loss)
