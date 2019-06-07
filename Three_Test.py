# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:54:25 2019

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
from LTSM_class import LSTM_M_4


class last_night(object):
    __instance = None
    __first_init = False
    flag = False
    def __new__(cls):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance
    def __init__(self):
        if not self.__first_init:
            print("哒哒哒哒哒哒——————————————————")
            self.values = 100
            self.buy = 1
            last_night.__first_init = True

def init(context):
    set_backtest(initial_cash=1000000)  # 设置回测初始信息
    reg_kdata('day', 1)  # 注册K线数据
    reg_factor(['RSI','SMA','PB', 'PE'])
    

def on_data(context: Context):
    DS = preprocessing.StandardScaler()
    data_factor = get_reg_factor(reg_idx=context.reg_factor[0], length = 30, df=True)  
    data_values = get_reg_kdata(reg_idx=context.reg_kdata[0], target_indices=(), length=30, df=True)
    a = LSTM_M_4("./model/LSTM_full.h5")
    high = data_values['high'].values
    low = data_values['low'].values
    close = data_values['close'].values
    stock = ((high+low+2*close)/4.0).reshape((-1,1)).astype('float64')
    
    PE = data_factor['value'][np.where(data_factor['factor'] == 'PE')[0]].values.astype('float64').reshape((-1, 1))
    PB = data_factor['value'][np.where(data_factor['factor'] == 'PB')[0]].values.astype('float64').reshape((-1, 1))
    MA = data_factor['value'][np.where(data_factor['factor'] == 'SMA')[0]].values.astype('float64').reshape((-1, 1))
    RSI = data_factor['value'][np.where(data_factor['factor'] == 'RSI')[0]].values.astype('float64').reshape((-1, 1))

    if np.isnan(stock).any() or np.isnan(PE).any():
        pass
    else:
        stock = DS.fit_transform(stock)
        PE = DS.fit_transform(PE)
        PB = DS.fit_transform(PB)
        MA = DS.fit_transform(MA)
        RSI = DS.fit_transform(RSI)
        last = last_night()
        pre_stock = a.predict(PE, PB, MA, RSI )
        pre_stock = DS.inverse_transform(pre_stock)
        stock = DS.inverse_transform(stock)
        print(context.account(account_idx=0).cash['valid_cash'].values)
        if pre_stock==-1:
            pass
        else:
            if context.account(account_idx=0).cash['valid_cash'].values > 1010000:
                last.buy=2
            if pre_stock < last.values and last.buy==0:
                order_close_all()
                last.buy=1
            elif pre_stock>=last.values and last.buy==1:
                order_percent(account_idx=0, target_idx=0, percent=1.00, side=1, position_effect=1, order_type=2, price=0.0)
                last.buy=0

            last.values = pre_stock
if __name__ == '__main__':  
    start = '2018-06-11'
    end = '2018-09-30'
    target = ['sse.601360']
    run_backtest(strategy_name='Three_factor', file_path='.', target_list=target, frequency='day', fre_num=1, begin_date=start, end_date=end)


