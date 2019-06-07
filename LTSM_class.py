# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:33:39 2019

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
import time

class LSTM_M_3(object):
    __instance = None
    __first_init = False
    flag = False

    def __new__(cls, path):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self, path):
        if not self.__first_init:
            print("初始化类——————————————————")
            self.model = load_model(path)
            LSTM_M_3.__first_init = True
    """
    划分训练集与数据集
    """
    def predict(self, PE, PB, MA):
        self.PE = PE
        self.PB = PB
        self.MA = MA
        t_s = len(PE)
        self.test_x = np.column_stack((PE[(t_s - 1):len(PE)], PB[(t_s - 1):len(PE)], MA[(t_s - 1):len(PE)]))
        self.test_x = np.reshape(self.test_x, (self.test_x.shape[0], 1, self.test_x.shape[1]))
        """
        LSTM模型预测
        """
        predict_y = self.model.predict(self.test_x)
        return predict_y

class LSTM_M_4(object):
    __instance = None
    __first_init = False
    flag = False

    def __new__(cls, path):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self, path):
        if not self.__first_init:
            print("初始化类——————————LSTM————————")
            self.model = load_model(path)
            LSTM_M_4.__first_init = True
    """
    划分训练集与数据集
    """
    def predict(self, PE, PB, MA, RSI):
        self.PE = PE
        self.PB = PB
        self.MA = MA
        self.RSI = RSI
        t_s = len(PE)
        self.test_x = np.column_stack((PE[(t_s - 1):len(PE)], PB[(t_s - 1):len(PE)], MA[(t_s - 1):len(PE)],RSI[(t_s - 1):len(PE)]))
        self.test_x = np.reshape(self.test_x, (self.test_x.shape[0], 1, self.test_x.shape[1]))
        """
        LSTM模型预测
        """
        predict_y = self.model.predict(self.test_x)
        return predict_y

class RP_M_4(object):
    __instance = None
    __first_init = False
    flag = False

    def __new__(cls, path):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self, path):
        if not self.__first_init:
            print("初始化类———————RP———————————")
            self.model = load_model(path)
            RP_M_4.__first_init = True
    """
    划分训练集与数据集
    """
    def predict(self, PE, PB, MA, RSI):
        self.PE = PE
        self.PB = PB
        self.MA = MA
        self.RSI = RSI
        t_s = len(PE)
        self.test_x = np.column_stack((PE[(t_s - 1):len(PE)], PB[(t_s - 1):len(PE)], MA[(t_s - 1):len(PE)],RSI[(t_s - 1):len(PE)]))
        self.test_x = np.reshape(self.test_x, (self.test_x.shape[0],  self.test_x.shape[1]))
        """
        LSTM模型预测
        """
        predict_y = self.model.predict(self.test_x)
        return predict_y

class LSTM_M_7(object):
    __instance = None
    __first_init = False
    flag = False

    def __new__(cls, path):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self, path):
        if not self.__first_init:
            print("初始化类——————————————————")
            self.model = load_model(path)
            LSTM_M_7.__first_init = True
    """
    划分训练集与数据集
    """
    def predict(self, fac_0, fac_1, fac_2, fac_3, fac_4, fac_5, fac_6):
        self.fac_0 = fac_0
        self.fac_1 = fac_1
        self.fac_2 = fac_2
        self.fac_3 = fac_3
        self.fac_4 = fac_4
        self.fac_5 = fac_5
        self.fac_6 = fac_6
        t_s = len(fac_0)
        self.test_x = np.column_stack((fac_0[(t_s - 1):len(fac_0)], fac_1[(t_s - 1):len(fac_0)], fac_2[(t_s - 1):len(fac_0)],
                                fac_3[(t_s - 1):len(fac_0)],fac_4[(t_s - 1):len(fac_0)],fac_5[(t_s - 1):len(fac_0)],fac_6[(t_s - 1):len(fac_0)]))
        self.test_x = np.reshape(self.test_x, (self.test_x.shape[0], 1, self.test_x.shape[1]))
        """
        LSTM模型预测
        """
        predict_y = self.model.predict(self.test_x)
        return predict_y







