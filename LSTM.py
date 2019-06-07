 # -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:53:38 2019

@author: ASUS
"""

from atrader import *
from atrader.calcfactor import *
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Activation

def LSTM_2(fac_0,fac_1,filepath,stock_name):
    Data_stand = preprocessing.StandardScaler()
    data_value = get_kdata(target_list=[stock_name], frequency='day', fre_num=1, begin_date='2017-01-01', end_date='2018-06-30')
    data_factor = get_factor_by_code([fac_0,fac_1],stock_name,'2017-01-01','2018-06-30')
    PE = data_factor[fac_0].values.reshape((-1,1))
    PB = data_factor[fac_1].values.reshape((-1,1))

    PE = Data_stand.fit_transform(PE)
    PB = Data_stand.fit_transform(PB)

    high = data_value[stock_name]['high'].values
    low = data_value[stock_name]['low'].values
    stock = (high+low).reshape((-1,1))

    """
    
    划分训练集与数据集
    
    """
    try:
        t_s = len(PE)-100
    except:
        t_s = int(len(PE)*0.8)
    train_x = np.column_stack((PE[0:t_s],PB[0:t_s]))
    train_x = np.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))

    train_y = stock[0:t_s]
    train_y = Data_stand.fit_transform(train_y)
    test_x = np.column_stack((PE[(t_s-1):len(PE)],PB[(t_s-1):len(PE)]))
    test_x = np.reshape(test_x,(test_x.shape[0],1,test_x.shape[1]))


    test_y = stock[t_s:len(PE)]
    test_y = Data_stand.fit_transform(test_y)

    """
    
    LSTM模型搭建
    
    """
    model = Sequential()
    model.add(LSTM(50,input_shape=(1,2),return_sequences=True))
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

    predict_y = LS.predict(test_x)

    """
    
    数据可视化
    
    """

    plt.figure(figsize=(10,10))

    plt.subplot(2, 1, 1)
    plt.title(fac_0+"_"+fac_1+'to Stock Value')
    plt.plot(Data_stand.inverse_transform(test_y), label='True data')
    plt.plot(Data_stand.inverse_transform(predict_y), label='Predict data')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()


def LSTM_3(fac_0,fac_1,fac_2,filepath,stock_name):
    Data_stand = preprocessing.StandardScaler()
    data_value = get_kdata(target_list=[stock_name], frequency='day', fre_num=1, begin_date='2017-01-01', end_date='2018-06-30')
    data_factor = get_factor_by_code([fac_0,fac_1,fac_2],stock_name,'2017-01-01','2018-06-30')
    PE = 1.0/data_factor[fac_0].values.reshape((-1,1))
    PB = data_factor[fac_1].values.reshape((-1,1))
    MA = data_factor[fac_2].values.reshape((-1,1))

    PE = Data_stand.fit_transform(PE)
    PB = Data_stand.fit_transform(PB)
    MA = Data_stand.fit_transform(MA)

    open_v = data_value[stock_name]['open'].values
    stock = open_v.reshape((-1,1))

    """
    
    划分训练集与数据集
    
    """
    try:
        t_s = len(PE)-100
    except:
        t_s = int(len(PE)*0.8)
    train_x = np.column_stack((PE[0:t_s],PB[0:t_s],MA[0:t_s]))
    train_x = np.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))

    train_y = stock[0:t_s]
    train_y = Data_stand.fit_transform(train_y)
    test_x = np.column_stack((PE[(t_s-1):len(PE)],PB[(t_s-1):len(PE)],MA[(t_s-1):len(PE)]))
    test_x = np.reshape(test_x,(test_x.shape[0],1,test_x.shape[1]))


    test_y = stock[t_s:len(PE)]
    test_y = Data_stand.fit_transform(test_y)

    """
    
    LSTM模型搭建
    
    """
    model = Sequential()
    model.add(LSTM(50,input_shape=(1,3),return_sequences=True))
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

    predict_y = LS.predict(test_x)

    """
    
    数据可视化
    
    """

    plt.figure(figsize=(10,10))

    plt.subplot(2, 1, 1)
    plt.title(fac_0+"_"+fac_1+"_"+fac_2+'to Stock Value')
    plt.plot(Data_stand.inverse_transform(test_y), label='True data')
    plt.plot(Data_stand.inverse_transform(predict_y), label='Predict data')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

def LSTM_4(fac_0,fac_1,fac_2,fac_3,filepath,stock_name):
    a_0 = preprocessing.StandardScaler()
    a_1 = preprocessing.StandardScaler()
    a_2 = preprocessing.StandardScaler()
    a_3 = preprocessing.StandardScaler()
    a_4 = preprocessing.StandardScaler()
    
    
    data_value = get_kdata(target_list=[stock_name], frequency='day', fre_num=1, begin_date='2017-01-01', end_date='2018-06-30')
    data_factor = get_factor_by_code([fac_0,fac_1,fac_2,fac_3],stock_name,'2016-01-01','2018-09-30')
    fac_0 = data_factor[fac_0].values.reshape((-1,1))
    fac_1 = data_factor[fac_1].values.reshape((-1,1))
    fac_2 = data_factor[fac_2].values.reshape((-1,1))
    fac_3 = data_factor[fac_3].values.reshape((-1,1))
    
    fac_0 = a_0.fit_transform(fac_0)
    fac_1 = a_1.fit_transform(fac_1)
    fac_2 = a_2.fit_transform(fac_2)
    fac_3 = a_3.fit_transform(fac_3)
    
    open_v = data_value[stock_name]['open'].values
    
    stock = open_v.reshape((-1,1))
    stock = a_4.fit_transform(stock)

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
    model.add(LSTM(50,input_shape=(1,7),return_sequences=True))
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

    plt.figure(figsize=(10,10))

    plt.subplot(2, 1, 1)
    plt.title('Four factors')
    print(np.shape(test_y))
    print(np.shape(predict_y))
    
    plt.plot(a_4.inverse_transform(test_y), label='True data')
    plt.plot(a_4.inverse_transform(predict_y), label='Predict data')

    plt.legend()
    plt.subplot(2, 1, 2)
    plt.ylim([-0.05,1.05])
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()


def LSTM_7(fac_0,fac_1,fac_2,fac_3,fac_4,fac_5,fac_6,filepath,stock_name):
    Data_stand = preprocessing.StandardScaler()
    data_value = get_kdata(target_list=[stock_name], frequency='day', fre_num=1, begin_date='2017-01-01', end_date='2018-06-30')
    data_factor = get_factor_by_code([fac_0,fac_1,fac_2,fac_3,fac_4,fac_5,fac_6],stock_name,'2017-01-01','2018-06-30')
    fac_0 = data_factor[fac_0].values.reshape((-1,1))
    fac_1 = data_factor[fac_1].values.reshape((-1,1))
    fac_2 = data_factor[fac_2].values.reshape((-1,1))
    fac_3 = data_factor[fac_3].values.reshape((-1,1))
    fac_4 = data_factor[fac_4].values.reshape((-1,1))
    fac_5 = data_factor[fac_5].values.reshape((-1,1))
    fac_6 = data_factor[fac_6].values.reshape((-1,1))
    

    fac_0 = Data_stand.fit_transform(fac_0)
    fac_1 = Data_stand.fit_transform(fac_1)
    fac_2 = Data_stand.fit_transform(fac_2)
    fac_3 = Data_stand.fit_transform(fac_3)
    fac_4 = Data_stand.fit_transform(fac_4)
    fac_5 = Data_stand.fit_transform(fac_5)
    fac_6 = Data_stand.fit_transform(fac_6)

    open_v = data_value[stock_name]['open'].values
    stock = open_v.reshape((-1,1))

    """
    
    划分训练集与数据集
    
    """
    try:
        t_s = len(fac_0)-100
    except:
        t_s = int(len(fac_0)*0.8)
    train_x = np.column_stack((fac_0[0:t_s],fac_1[0:t_s],fac_2[0:t_s],fac_3[0:t_s],fac_4[0:t_s],fac_5[0:t_s],fac_6[0:t_s]))
    train_x = np.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))

    train_y = stock[0:t_s]
    train_y = Data_stand.fit_transform(train_y)
    test_x = np.column_stack((fac_0[(t_s-1):len(fac_0)],fac_1[(t_s-1):len(fac_0)],fac_2[(t_s-1):len(fac_0)],fac_3[(t_s-1):len(fac_0)],
                                    fac_4[(t_s-1):len(fac_0)],fac_5[(t_s-1):len(fac_0)],fac_6[(t_s-1):len(fac_0)]))
    test_x = np.reshape(test_x,(test_x.shape[0],1,test_x.shape[1]))


    test_y = stock[t_s:len(fac_0)]
    test_y = Data_stand.fit_transform(test_y)

    """
    
    LSTM模型搭建
    
    """
    model = Sequential()
    model.add(LSTM(50,input_shape=(1,7),return_sequences=True))
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

    predict_y = LS.predict(test_x)

    """
    
    数据可视化
    
    """

    plt.figure(figsize=(10,10))

    plt.subplot(2, 1, 1)
    plt.title('Seven factor')
    plt.plot(Data_stand.inverse_transform(test_y), label='True data')
    plt.plot(Data_stand.inverse_transform(predict_y), label='Predict data')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()


