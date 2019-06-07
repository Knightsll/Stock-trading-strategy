import numpy as np
import pandas as pd
from atrader import *
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm

def init(context):
    set_backtest(initial_cash=10000000)  # 设置回测初始信息
    reg_kdata('day', 1)  # 注册K线数据
    reg_factor(['PE', 'PB', 'MA10'])

def on_data(context: Context):
    data_factor = get_reg_factor(reg_idx=context.reg_factor[0], length = 30, df=True)
    data_values = get_reg_kdata(reg_idx=context.reg_kdata[0], target_indices=(), length=30, df=True)

if __name__ == '__main__':
    start = '2017-05-11'
    end = '2018-06-11'
    target = ['sse.600196']
    run_backtest(strategy_name='Three_factor', file_path='.', target_list=target, frequency='day', fre_num=1, begin_date=start, end_date=end)
