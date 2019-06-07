from atrader import *
import numpy as np
import pandas as pd
import datetime as dt
from atrader.calcfactor import *
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

class choose(object):
    __instance = None
    __first_init = False
    flag = False
    def __new__(cls, index):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance
    def __init__(self,index):
        if not self.__first_init:
            if not isinstance(index,list):
                self.index =index.tolist()
            else:
                self.index = index
            if len(index)==1:
                choose.__first_init = True
            print("——————————{0}————————".format(self.index))

def ps_cal(MV,ORPS):
    if len(MV) > 30:
        temp_OR = np.nan_to_num(ORPS[[29 + i * 30 for i in range(50)]])
        temp_index = np.where(temp_OR == 0)[0]
        temp_OR = np.delete(temp_OR, temp_index)
        temp_MV = np.delete(MV[[29 + i * 30 for i in range(50)]], temp_index)
        temp_PS = temp_MV / temp_OR
        PS_total = np.insert(temp_PS, temp_index, 0)
    else:
        temp_OR = np.nan_to_num(ORPS)
        temp_index = np.where(temp_OR == 0)[0]
        temp_OR = np.delete(temp_OR, temp_index)
        temp_MV = np.delete(MV, temp_index)
        temp_PS = temp_MV / temp_OR
        PS_total = np.insert(temp_PS, temp_index, 0)
    return PS_total

def init(context):
    set_backtest(initial_cash=10000000)
    reg_kdata('day', 1)
    reg_factor(['RSI', 'SMA', 'PB', 'PE', 'MktValue','PCF','OperatingRevenuePS','ROE','HBETA'])

def on_data(context):
    DS = preprocessing.StandardScaler()
    TS = preprocessing.StandardScaler()
    a = LSTM_M_4("./model/LSTM_full_v1.h5")
    t_index = choose([x for x in range(50)])

    data_factor = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=t_index.index, length=30, df=True)
    data_values = get_reg_kdata(reg_idx=context.reg_kdata[0], target_indices=t_index.index, length=30, df=True)
    #print(data_values)
    
    PE_ = data_factor['value'][np.where(data_factor['factor'] == 'PE')[0]].values.astype('float64').reshape((-1, 1))
    PB_ = data_factor['value'][np.where(data_factor['factor'] == 'PB')[0]].values.astype('float64').reshape((-1, 1))
    PCF_ = data_factor['value'][np.where(data_factor['factor'] == 'PCF')[0]].values.astype('float64').reshape((-1, 1))
    RSI_ = data_factor['value'][np.where(data_factor['factor'] == 'RSI')[0]].values.astype('float64').reshape((-1, 1))
    MA = data_factor['value'][np.where(data_factor['factor'] == 'SMA')[0]].values.astype('float64').reshape((-1, 1))
    ROE = data_factor['value'][np.where(data_factor['factor'] == 'ROE')[0]].values.astype('float64').reshape((-1, 1))
    NPGR = data_factor['value'][np.where(data_factor['factor'] == 'NetProfitGrowRate')[0]].values.astype('float64').reshape((-1, 1))
    TAGR = data_factor['value'][np.where(data_factor['factor'] == 'TotalAssetGrowRate')[0]].values.astype('float64').reshape((-1, 1))
    MV = data_factor['value'][np.where(data_factor['factor'] == 'MktValue')[0]].values.astype('float64').reshape((-1, 1))
    ORPS = data_factor['value'][np.where(data_factor['factor'] == 'OperatingRevenuePS')[0]].values.astype('float64').reshape((-1, 1))
    BT_ = data_factor['value'][np.where(data_factor['factor'] == 'HBETA')[0]].values.astype('float64').reshape((-1, 1))
    try:
        PE_total = np.nan_to_num(PE_[[29+i*30 for i in range(50)]])
        PB_total = np.nan_to_num(PB_[[29+i*30 for i in range(50)]])
        PCF_total = np.nan_to_num(PCF_[[29 + i * 30 for i in range(50)]])
        print("start to calculate :", len(MV), len(PCF_),len(ORPS))
        PS_total = ps_cal(MV,ORPS)
        print("end to calculate")

        #ROE_total = np.nan_to_num(ROE[[29+i*30 for i in range(50)]])
        #NPGR_total = np.nan_to_num(NPGR[[29+i*30 for i in range(50)]])
        #TAGR_total = np.nan_to_num(TAGR[[29+i*30 for i in range(50)]])
    except:
        PE_total = np.array([1])
        PB_total = np.array([1])
        PS_total = np.array([1])
        PCF_total = np.array([1])
    #print("PS: ",PS_total)
    if np.isnan(PE_).any():
        pass
    else:
        try:
            pe = TS.fit_transform(PE_total.reshape(-1,1))
            pb = TS.fit_transform(PB_total.reshape(-1,1))
            ps = TS.fit_transform(PS_total.reshape(-1,1))
            pcf = TS.fit_transform(PCF_total.reshape(-1,1))
            #pcf = TS.fit_transform(PCF_total)
            temp_choose = pe+pb+ps+pcf
        except:
            temp_choose = np.array([1])
        print("choose:      ",temp_choose, len(temp_choose))
        t_index = choose(np.where(temp_choose == temp_choose.max())[0])

        try:
            high = data_values['high'].values[[i for i in range((t_index.index[0] + 1) * 30)]]
            low = data_values['low'].values[[i for i in range((t_index.index[0] + 1) * 30)]]
            close = data_values['close'].values[[i for i in range((t_index.index[0] + 1) * 30)]]
            stock = ((high + low + 2 * close) / 4.0).reshape((-1, 1)).astype('float64')

            PE = PE_[[i for i in range((t_index.index[0] + 1) * 30)]]
            PB = PB_[[i for i in range((t_index.index[0] + 1) * 30)]]
            MA = MA[[i for i in range((t_index.index[0] + 1) * 30)]]
            RSI = RSI_[[i for i in range((t_index.index[0] + 1) * 30)]]
        except:

            high = data_values['high'].values
            low = data_values['low'].values
            close = data_values['close'].values
            stock = ((high + low + 2 * close) / 4.0).reshape((-1, 1)).astype('float64')

            PE = PE_.copy()
            PB = PB_.copy()
            MA = MA.copy()
            RSI = RSI_.copy()


        if np.isnan(stock).any():
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
                if context.account(account_idx=0).cash['valid_cash'].values > 12000000:
                    last.buy=2
                if pre_stock < last.values and last.buy==0:
                    order_close_all()
                    last.buy=1
                elif pre_stock>=last.values and last.buy==1:
                    order_percent(account_idx=0, target_idx=t_index.index[0], percent=1.00, side=1, position_effect=1, order_type=2, price=0.0)
                    last.buy=0

                last.values = pre_stock




    
if __name__ == '__main__':
    begin = '2016-08-01'
    end = '2018-08-31'
    cons_date = dt.datetime.strptime(begin, '%Y-%m-%d') - dt.timedelta(days=1)
    hs300 = get_code_list('sz50', cons_date)[['code', 'weight']]
    targetlist = list(hs300['code'])
    run_backtest(strategy_name='Choose_stock',
                 file_path='.',
                 target_list=targetlist,
                 frequency='day',
                 fre_num=1,
                 begin_date=begin,
                 end_date=end,
                 fq=1)

