# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:35:34 2019

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
    reg_factor(['MA5','MA60'])
    

def on_data(context: Context):
    data_factor = get_reg_factor(reg_idx=context.reg_factor[0], length = 1, df=True)  
    MA5 = data_factor['value'][np.where(data_factor['factor'] == 'MA5')[0]].values.astype('float64')
    MA60 = data_factor['value'][np.where(data_factor['factor'] == 'MA60')[0]].values.astype('float64')
    last = last_night()
    print(context.account(account_idx=0).cash['valid_cash'].values)
    if context.account(account_idx=0).cash['valid_cash'].values > 30000000:
        last.buy=2
    if MA60-MA5>0  and last.buy==1:
        order_percent(account_idx=0, target_idx=0, percent=0.99, side=1, position_effect=1, order_type=2)
        last.buy=0
    elif MA60-MA5<0 and last.buy==0:
        order_close_all()
        last.buy=1

if __name__ == '__main__':  
    start = '2017-05-11'
    end = '2018-08-30'
    target = ['sse.601800']
    run_backtest(strategy_name='Three_factor', file_path='.', target_list=target, frequency='day', fre_num=1, begin_date=start, end_date=end)


