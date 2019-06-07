# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:23:26 2019

@author: Knightsll

Function: train LSTM model by various factors in SSE
"""

from atrader import *
from atrader.calcfactor import *
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import datetime as dt
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Activation

fac_0_ = 'RSI'
fac_1_ = 'SMA'
fac_2_ = 'PB'
fac_3_ = 'PE'
filepath = './model/LSTM_full_v1.h5'

a_0 = preprocessing.StandardScaler()
a_1 = preprocessing.StandardScaler()
a_2 = preprocessing.StandardScaler()
a_3 = preprocessing.StandardScaler()
a_4 = preprocessing.StandardScaler()
begin = '2016-01-01'
end = '2018-09-30'

cons_date = dt.datetime.strptime(begin, '%Y-%m-%d') - dt.timedelta(days=1)
hs300 = get_code_list('sz50', cons_date)[['code', 'weight']]
target_list = list(hs300['code'])


data_value = get_kdata(target_list=target_list, frequency='day', fre_num=1, begin_date=begin, end_date=end)
factor_pool = {}
for i in range(50):
    data_factor = get_factor_by_code([fac_0_,fac_1_,fac_2_,fac_3_],target=target_list[i], begin_date=begin, end_date=end)
    factor_pool[i] = data_factor 

fac_0 = np.array([])
fac_1 = np.array([])
fac_2 = np.array([])
fac_3 = np.array([])
stock = np.array([])

for i in range(50):
    if len(factor_pool[i][fac_0_].values.reshape((-1,1))) == len(data_value[target_list[i]]['high'].values):
        fac_0 = np.append(fac_0,factor_pool[i][fac_0_].values.reshape((-1,1)))
        fac_1 = np.append(fac_1,factor_pool[i][fac_1_].values.reshape((-1,1)))
        fac_2 = np.append(fac_2,factor_pool[i][fac_2_].values.reshape((-1,1)))
        fac_3 = np.append(fac_3,factor_pool[i][fac_3_].values.reshape((-1,1)))
        
        high = data_value[target_list[i]]['high'].values
        low = data_value[target_list[i]]['low'].values
        close = data_value[target_list[i]]['close'].values
        stock = np.append(stock, ((high+low+2*close)/4.0).reshape((-1,1)))
        print(np.shape(fac_0),np.shape(stock))

fac_0 = a_0.fit_transform(fac_0.reshape((-1,1)))
fac_1 = a_1.fit_transform(fac_1.reshape((-1,1)))
fac_2 = a_2.fit_transform(fac_2.reshape((-1,1)))
fac_3 = a_3.fit_transform(fac_3.reshape((-1,1)))


stock = a_4.fit_transform(stock.reshape((-1,1)))

"""

划分训练集与数据集

"""
try:
    t_s = len(fac_0)
except:
    t_s = int(len(fac_0)*0.8)
train_x = np.column_stack((fac_0[0:t_s],fac_1[0:t_s],fac_2[0:t_s],fac_3[0:t_s]))
train_x = np.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))
test = np.random.randint(1,len(fac_0),(100,1))
train_y = stock[0:t_s]

test_x = np.column_stack((fac_0[test],fac_1[test],fac_2[test],fac_3[test]))
test_x = np.reshape(test_x,(test_x.shape[0],1,test_x.shape[1]))
print(np.shape(test_x))

test_y = stock[test]

"""

LSTM模型搭建

"""
model = Sequential()
model.add(LSTM(50,input_shape=(1,4),return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))
model.summary()

model.compile(loss='mse', optimizer='rmsprop')

"""

LSTM模型训练

"""

history = model.fit(train_x, train_y, batch_size=50, nb_epoch=1000, validation_split=0.1)



model.save(filepath,overwrite=True, include_optimizer=True )


LS = load_model(filepath)

"""

LSTM模型预测

"""

test_y = test_y.reshape((-1,1))
predict_y = LS.predict(test_x)

"""

数据可视化

"""

plt.figure(figsize=(20,8))
plt.suptitle('Four factors(RSI,SMA,PB,PE) model based on LSTM',fontsize=20)
plt.subplot(1, 2, 1)
plt.title('Predict data and real data',fontsize=20)
print(np.shape(test_y))
print(np.shape(predict_y))

plt.plot(a_4.inverse_transform(test_y), lw=2,label='True data')
plt.plot(a_4.inverse_transform(predict_y),lw=1, label='Predict data')

plt.legend()
plt.subplot(1, 2, 2)
plt.title('Error convergence',fontsize=20)
plt.ylim([-0.05,1.05])
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()






