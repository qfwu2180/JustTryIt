from sklearn.preprocessing import MinMaxScaler
import pickle as pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from copy import deepcopy

OUTPUT_NODE = 1
BATCH_SIZE = 50
LEARNING_RATE_BASE = 1.0
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.00001
MOVING_AVERAGE_DECAY = 0.99


def get_weight(shape, lambda1):
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=1)
    var = tf.Variable(weight_initializer(shape))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var


def train(length,traning_steps):
    global INPUT_NODE
    global LAYER1_NODE
    global LAYER2_NODE
    global LAYER3_NODE
    INPUT_NODE=result1.shape[1]
    LAYER1_NODE = 7
    LAYER2_NODE = 7
    LAYER3_NODE = 7
    LAYER4_NODE = 7

    x = tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None,1],name='y-input')

    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=1)
    bias_initializer = tf.zeros_initializer()
    W_hidden_1 = tf.Variable(weight_initializer([INPUT_NODE, LAYER1_NODE]))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(W_hidden_1))
    bias_hidden_1 = tf.Variable(bias_initializer([LAYER1_NODE]))
    W_hidden_2 = tf.Variable(weight_initializer([LAYER1_NODE, LAYER2_NODE]))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(W_hidden_2))
    bias_hidden_2 = tf.Variable(bias_initializer([LAYER2_NODE]))
    W_hidden_3 = tf.Variable(weight_initializer([LAYER2_NODE, LAYER3_NODE]))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(W_hidden_3))
    bias_hidden_3 = tf.Variable(bias_initializer([LAYER3_NODE]))

    W_hidden_4 = tf.Variable(weight_initializer([LAYER3_NODE, LAYER4_NODE]))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(W_hidden_4))
    bias_hidden_4 = tf.Variable(bias_initializer([LAYER4_NODE]))

    W_out = tf.Variable(weight_initializer([LAYER4_NODE, 1]))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(W_out))
    bias_out = tf.Variable(bias_initializer([1]))

    hidden_1 = tf.nn.relu(tf.add(tf.matmul(x, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    out =tf.add(tf.matmul(hidden_4, W_out), bias_out)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    mse = tf.reduce_mean(tf.square(y_ - out))
    tf.add_to_collection('losses', mse)
    loss = tf.add_n(tf.get_collection('losses'))
    #learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,500, LEARNING_RATE_DECAY,staircase=False)
    train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(0,traning_steps):
            start = (i * BATCH_SIZE) % size
            end = min(start + BATCH_SIZE, size)

            sess.run(train_step, feed_dict={x: result1[start:end], y_: result2[start:end]})

            if i % 1000 == 0:
                pre=sess.run(out, feed_dict={x: result3})
                print(scaler2.inverse_transform(pre)-1)
                #print(sess.run(learning_rate))
                total_mse = sess.run(loss, feed_dict={x: result1, y_: result2})
                print("After %d training step(s), loss on training batch is %g." % (i, math.sqrt(total_mse)))
                total_mse = sess.run(loss, feed_dict={x: result3, y_: result4})
                print("After %d training step(s), loss on testing batch is %g, average is %g.\nStandard average is %g\n" % (i, math.sqrt(total_mse),np.mean(scaler2.inverse_transform(pre))-1,average4))



def find_valid(seq,begin,data1):
    i=0
    temp=begin
    if seq==1:
        for i in range(begin, 4034):
            if data1.iloc[i,0] != '' :
                return data1.index[i]
        if i == 4034:
            for i in range(temp, -1, -1):
                if data1.iloc[i, 0] != '':
                    return data1.index[i]
        return 0
    elif seq==-1:
        for i in range(begin, -1, -1):
            if data1.iloc[i, 0] != '':
                return data1.index[i]
        if i == 0:
            for i in range(temp, 4034):
                if data1.iloc[i,0] != '' :
                    return data1.index[i]
        return 0


def pretreatment(target_date, begin_date,length,interval,num,year,data):
    global result1
    global result2
    global result3
    global result4
    global size
    global scaler2
    global average4
    result1=pd.DataFrame()
    result2 = pd.DataFrame()
    result3 = pd.DataFrame()
    result4 =pd.DataFrame()
    tem5 = []
    name = pd.read_csv('D:\\name.csv', header=None)
    name = name.values

    for k in range(0, num):
        print("Gathering %s"%(name[k,0]))
        retr = balance[balance.iloc[:, 2] == name[k, 0]]
        if retr.empty:
            continue
        pkl_file = open('STOCK/'+name[k,0]+'.pkl', 'rb')
        data1 = pickle.load(pkl_file)

        beginindex = data.loc[:, [begin_date]].values[0, 0]
        terminal = data.loc[:, [target_date]].values[0, 0]
        target_date = find_valid(1, terminal, data1)

        if target_date == 0:
            continue
        terminal = data.loc[:, [target_date]].values[0, 0]
        ll=deepcopy(terminal)

        begin_date = find_valid(-1, beginindex, data1)
        if begin_date == 0:
            continue
        beginindex = data.loc[:, [begin_date]].values[0, 0]
        if (beginindex+length+2*interval)>=terminal:
            continue
        data2=deepcopy(data1)
        data1 = data1[~data1.isin([''])]
        data1 = data1.dropna(axis=0, how='any')
        beginindex = data1.index.get_loc(begin_date)
        terminal = data1.index.get_loc(target_date)
        endindex = terminal - length - 2*interval
        flag=0

        loc=[]

        for i in range(0,year):
            ll=ll-252
            if ll<0:
                continue
            for z in range(0,20):
                if data2.iloc[ll+z, 0] != '':
                    loc.append(data2.index[ll+z])
        print(loc)
        for i in range(len(loc)-1,-1,-1):
            flag=0
            list3 = []
            tem3 = []
            list4 = []
            tem4 = []
            ter = data1.index.get_loc(loc[i])
            endate=data1.index[ter-interval]
            #list3.append(name[k, 0])
            #list3.append(data1.index[ter - interval])
            #list4.append(name[k, 0])
            #list4.append(data1.index[ter - interval])
            for z in range(0,length):
                list3.append(data1.iloc[ter - interval - length + z, 0])
                list4.append(data1.iloc[ter - interval - length + z, 1])

            for g in range(0,retr.shape[0]):
                if endate<retr.iloc[g,3]:
                    flag=1
                    break
                else:
                    if (g<(retr.shape[0]-1)):
                        if (endate < retr.iloc[g + 1, 3]):
                            for w in range(5, 15):
                                list4.append(retr.iloc[g, w])
                            break
                    elif g == (retr.shape[0] - 1):
                        for w in range(5, 15):
                            list4.append(retr.iloc[g, w])
                        break


            if flag==1:
                continue
            list3.append((data1.iloc[ter, 0]+data1.iloc[ter+1, 0]+data1.iloc[ter+2, 0]+data1.iloc[ter+3, 0]+data1.iloc[ter+4, 0])/5)
            tem3.append(list3)
            tem4.append(list4)
            result1 = result1.append(pd.DataFrame(tem3, columns=None), ignore_index=True, sort=False)
            result2 = result2.append(pd.DataFrame(tem4, columns=None), ignore_index=True, sort=False)
        list3 = []
        tem3 = []
        list4 = []
        tem4 = []


        for m in range(0, length):
            list3.append(data1.iloc[terminal-interval-length+m, 0])
            list4.append(data1.iloc[terminal-interval-length+m, 1])

        endate = data1.index[terminal - interval]
        for g in range(0, retr.shape[0]):
            if endate < retr.iloc[g, 3]:
                flag = 1
                break
            else:
                if (g < (retr.shape[0] - 1)):
                    if (endate < retr.iloc[g + 1, 3]):
                        for w in range(5, 15):
                            list4.append(retr.iloc[g, w])
                        break
                elif g == (retr.shape[0] - 1):
                    for w in range(5, 15):
                        list4.append(retr.iloc[g, w])
                    break
        if flag==0:
            list3.append((data1.iloc[terminal, 0]+data1.iloc[terminal+1, 0]+data1.iloc[terminal+2, 0]+data1.iloc[terminal+3, 0]+data1.iloc[terminal+4, 0])/5)
            tem3.append(list3)
            tem4.append(list4)
            result3 = result3.append(pd.DataFrame(tem3, columns=None), ignore_index=True, sort=False)
            result4 = result4.append(pd.DataFrame(tem4, columns=None), ignore_index=True, sort=False)

        for i in range(beginindex, endindex):
            flag=0
            list1 = []
            tem1 = []
            list2 = []
            tem2 = []

            #list1.append(name[k, 0])
            #list1.append(data1.index[i])
            #list2.append(name[k, 0])
            #list2.append(data1.index[i])
            for m in range(0, length):
                list1.append(data1.iloc[i + m, 0])
                list2.append(data1.iloc[i + m, 1])
            endate = data1.index[i+length]
            for g in range(0, retr.shape[0]):
                if endate < retr.iloc[g, 3]:
                    flag = 1
                    break
                else:
                    if (g < (retr.shape[0] - 1)):
                        if (endate < retr.iloc[g + 1, 3]):
                            for w in range(5, 15):
                                list2.append(retr.iloc[g, w])
                            break
                    elif g == (retr.shape[0] - 1):
                        for w in range(5, 15):
                            list2.append(retr.iloc[g, w])
                        break
            if flag==1:

                continue


            sum=0
            for j in range(0,5):
                sum=sum+data1.iloc[i + length + interval+j, 0]

            list1.append(sum/5)
            if i==endindex-1:
                tem5.append(sum/5)
            tem1.append(list1)
            tem2.append(list2)
            result1 = result1.append(pd.DataFrame(tem1, columns=None), ignore_index=True, sort=False)
            result2 = result2.append(pd.DataFrame(tem2, columns=None), ignore_index=True, sort=False)
    shuffle_indices = np.random.permutation(result1.index)
    result1 = result1.reindex(shuffle_indices)
    result2 = result2.reindex(shuffle_indices)
    scaler1 = MinMaxScaler().fit(result2.iloc[:,0:length])
    re_temp=result2.iloc[:,length:result2.shape[1]]
    result2 = scaler1.transform(result2.iloc[:,0:length])
    result2 = pd.DataFrame(result2)
    result2 = pd.concat([result2, re_temp], axis=1)

    re_temp=result4.iloc[:,length:result4.shape[1]]
    result4 = scaler1.transform(result4.iloc[:,0:length])
    result4 = pd.DataFrame(result4)
    result4 = pd.concat([result4, re_temp], axis=1)

    temp = result1.iloc[:, result1.shape[1] - 1]
    result1 = result1.drop([result1.shape[1] - 1], axis=1)
    scaler = MinMaxScaler().fit(result1)
    result1 = scaler.transform(result1)
    result1 = pd.DataFrame(result1)
    result1 = pd.concat([result1, result2], axis=1)

    temp.to_csv('Train_key.csv', encoding="utf-8", header=0, index=0)
    result2 = temp.values.reshape(-1, 1)
    scaler2 = MinMaxScaler().fit(result2)
    result2 = scaler2.transform(result2)

    temp = result3.iloc[:, result3.shape[1] - 1]
    result3 = result3.drop([result3.shape[1] - 1], axis=1)
    result3 = scaler.transform(result3)
    result3 = pd.DataFrame(result3)
    result3 = pd.concat([result3, result4], axis=1)

    temp.to_csv('Test_key.csv', encoding="utf-8", header=0, index=0)
    result4 = temp.values.reshape(-1, 1)
    average4 = np.mean(result4)
    result4 = scaler2.transform(result4)
    result2 = pd.DataFrame(result2)
    result4 = pd.DataFrame(result4)

    result1.to_csv('Standard_Train_data.csv', encoding="utf-8", header=0, index=0)
    result2.to_csv('Standard_Train_key.csv', encoding="utf-8", header=0, index=0)
    result3.to_csv('Standard_Test_data.csv', encoding="utf-8", header=0, index=0)
    result4.to_csv('Standard_Test_key.csv', encoding="utf-8", header=0, index=0)
    result5 = pd.DataFrame(columns=None, data=tem5)
    result5.to_csv('Original_last_key.csv', encoding="utf-8", header=0, index=0)

    size = result1.shape[0]
    result1 = result1.values
    result2 = result2.values
    result3 = result3.values
    result4 = result4.values


def main(argv=None):
    global balance
    target_date = 20180809
    begin_date = 20180101  # 数据库开始时间
    interval = 30  # 数据库结束日期与目标日期的差
    length = 20  # 样本长度
    num = 50  # 考察的股票范围
    year = 2
    traning_steps=200000
    pkl_file = open('index_date.pkl', 'rb')
    data = pickle.load(pkl_file)
    for i in range(target_date, 20020103, -1):
        if i in data.columns:
            target_date = i
            break
    for i in range(begin_date, 20180818):
        if i in data.columns:
            begin_date = i
            break

    balance = pd.read_csv('c.csv', header=None)
    pretreatment(target_date, begin_date,length,interval,num,year,data)
    train(length,traning_steps)


if __name__ == '__main__':
    main()
