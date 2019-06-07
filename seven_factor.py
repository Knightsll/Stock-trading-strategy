# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 00:52:41 2019

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
from LTSM_class import LSTM_M_7


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
    set_backtest(initial_cash=10000000)  # 设置回测初始信息
    reg_kdata('day', 1)  # 注册K线数据
    reg_factor(['VR','RSI','CCI5','SMA','CHAIKINVOLATILITY','PB', 'PE'])
    

def on_data(context: Context):
    DS = preprocessing.StandardScaler()
    data_factor = get_reg_factor(reg_idx=context.reg_factor[0], length = 30, df=True)  
    data_values = get_reg_kdata(reg_idx=context.reg_kdata[0], target_indices=(), length=30, df=True)
    a = LSTM_M_7("./model/LSTM_7_and_so_on.h5")

    open_v= data_values['open'].values
    stock = open_v.astype('float64').reshape((-1,1))
    
    fac_0 = data_factor['value'][np.where(data_factor['factor'] == 'VR')[0]].values.astype('float64').reshape((-1,1))
    fac_1 = data_factor['value'][np.where(data_factor['factor'] == 'RSI')[0]].values.astype('float64').reshape((-1,1))
    fac_2 = data_factor['value'][np.where(data_factor['factor'] == 'CCI5')[0]].values.astype('float64').reshape((-1,1))
    fac_3 = data_factor['value'][np.where(data_factor['factor'] == 'SMA')[0]].values.astype('float64').reshape((-1,1))
    fac_4 = data_factor['value'][np.where(data_factor['factor'] == 'CHAIKINVOLATILITY')[0]].values.astype('float64').reshape((-1,1))
    fac_5 = data_factor['value'][np.where(data_factor['factor'] == 'PB')[0]].values.astype('float64').reshape((-1,1))
    fac_6 = data_factor['value'][np.where(data_factor['factor'] == 'PE')[0]].values.astype('float64').reshape((-1,1))

    if np.isnan(stock).any() or np.isnan(fac_0).any():
        pass
    else:
        stock = DS.fit_transform(stock)
        fac_0 = DS.fit_transform(fac_0)
        fac_1 = DS.fit_transform(fac_1)
        fac_2 = DS.fit_transform(fac_2)
        fac_3 = DS.fit_transform(fac_3)
        fac_4 = DS.fit_transform(fac_4)
        fac_5 = DS.fit_transform(fac_5)
        fac_6 = DS.fit_transform(fac_6)
        
        last = last_night()
        pre_stock = a.predict(fac_0,fac_1,fac_2,fac_3,fac_4,fac_5,fac_6)
        pre_stock = DS.inverse_transform(pre_stock)
        stock = DS.inverse_transform(stock)
        print("------------  ", pre_stock, "+++++++++++++++",stock[len(stock)-1])
        if pre_stock==-1:
            pass
        else:
            if pre_stock>=last.values and last.buy==1:
                order_volume(account_idx=0, target_idx=0, volume=100000, side=1, position_effect=1, order_type=2)
                last.buy=0
            elif pre_stock< last.values and last.buy==0:
                order_close_all()
                last.buy=1
            last.values = pre_stock
if __name__ == '__main__':  
    start = '2017-05-11'
    end = '2018-01-11'
    target = ['sse.601186']
    run_backtest(strategy_name='seven_factor', file_path='.', target_list=target, frequency='day', fre_num=1, begin_date=start, end_date=end)

