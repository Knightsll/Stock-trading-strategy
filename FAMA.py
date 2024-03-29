# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:42:30 2019

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     多因子
   Description :
   Author :       haoyuan.m
   date：          2018/10/9
-------------------------------------------------
   Change Activity:
                   2018/10/9:
-------------------------------------------------
"""
__author__ = 'haoyuan.m'
from atrader import *
import numpy as np
import pandas as pd
import datetime as dt
'''
本策略每隔1个月定时触发,根据Fama-French三因子模型对每只股票进行回归，得到其alpha值。
假设Fama-French三因子模型可以完全解释市场，则alpha为负表明市场低估该股，因此应该买入。
策略思路：
计算市场收益率、个股的账面市值比和市值,并对后两个进行了分类,
根据分类得到的组合分别计算其市值加权收益率、SMB和HML. 
对各个股票进行回归(假设无风险收益率等于0)得到alpha值.
选取alpha值小于0并为最小的10只股票进入标的池
平掉不在标的池的股票并等权买入在标的池的股票
回测数据:上证50在回测日期前一天的成份股
回测时间:2017-09-01 到2018-05-31 
'''
def init(context):
    set_backtest(initial_cash=10000000)  # 设置回测初始信息
    reg_kdata('day', 1)  # 注册K线数据
    reg_factor(['PB', 'NegMktValue'])
    days = get_trading_days('SSE', '2017-09-01', '2018-06-01')
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()
    context.date = 20
    # 设置开仓的最大资金量
    context.ratio = 0.8
    # 账面市值比的大/中/小分类
    context.BM_BIG = 3.0
    context.BM_MID = 2.0
    context.BM_SMA = 1.0
    # 市值大/小分类
    context.MV_BIG = 2.0
    context.MV_SMA = 1.0
# 计算市值加权的收益率,MV为市值的分类,BM为账目市值比的分类
def market_value_weighted(stocks, MV, BM):
    select = stocks[(stocks.NegMktValue_Class == MV) & (stocks.BM_Class == BM)]
    market_value = select['NegMktValue'].values
    mv_total = np.sum(market_value)
    mv_weighted = [mv / mv_total for mv in market_value]
    stock_return = select['return'].values
    # 返回市值加权的收益率的和
    return_total = []
    for i in range(len(mv_weighted)):
        return_total.append(mv_weighted[i] * stock_return[i])
    return_total = np.sum(return_total)
    return return_total
def on_data(context):
    if dt.datetime.strftime(context.now, '%Y-%m-%d') not in context.month_begin:  # 调仓频率为月
        return
    factor = get_reg_factor(context.reg_factor[0], target_indices=[x for x in range(50)], length=1, df=True)
    PB = factor[factor['factor'] == 'PB'].rename(columns={'value': 'PB'}).drop('factor', axis=1).set_index('target_idx')
    NegMktValue = factor[factor['factor'] == 'NegMktValue'].rename(columns={'value': 'NegMktValue'}). \
        drop('factor', axis=1).set_index('target_idx')
    # 计算账面市值比,为P/B的倒数
    PB['BM'] = 1 / PB.PB
    # 计算市值的50%的分位点,用于后面的分类
    size_gate = NegMktValue['NegMktValue'].quantile(0.50)
    # 计算账面市值比的30%和70%分位点,用于后面的分类
    bm_gate = [PB['BM'].quantile(0.30), PB['BM'].quantile(0.70)]
    x_return = []
    kdata = get_reg_kdata(context.reg_kdata[0], target_indices=[x for x in range(50)], length=context.date + 1,
                          fill_up=True, df=True)
    if kdata['close'].isna().any():  # 如果数据不满21天则跳过
        return
    kdatalist = [kdata[kdata['target_idx'] == x] for x in pd.unique(kdata.target_idx)]
    for target in kdatalist:
        # 计算收益率
        stock_return = target.close.iloc[-1] / target.close.iloc[0] - 1
        target_idx = target.target_idx.iloc[0]
        BM = PB['BM'].loc[target_idx]
        market_value = NegMktValue['NegMktValue'].loc[target_idx]
        # 获取[股票代码. 股票收益率, 账面市值比的分类, 市值的分类, 流通市值]
        if BM < bm_gate[0]:
            if market_value < size_gate:
                label = [target_idx, stock_return, context.BM_SMA, context.MV_SMA, market_value]
            else:
                label = [target_idx, stock_return, context.BM_SMA, context.MV_BIG, market_value]
        elif BM < bm_gate[1]:
            if market_value < size_gate:
                label = [target_idx, stock_return, context.BM_MID, context.MV_SMA, market_value]
            else:
                label = [target_idx, stock_return, context.BM_MID, context.MV_BIG, market_value]
        elif market_value < size_gate:
            label = [target_idx, stock_return, context.BM_BIG, context.MV_SMA, market_value]
        else:
            label = [target_idx, stock_return, context.BM_BIG, context.MV_BIG, market_value]
        if len(x_return) == 0:
            x_return = label
        else:
            x_return = np.vstack([x_return, label])
    stocks = pd.DataFrame(data=x_return,
                          columns=['target_idx', 'return', 'BM_Class', 'NegMktValue_Class', 'NegMktValue'])
    stocks.set_index('target_idx', inplace=True)
    # 获取小市值组合的市值加权组合收益率
    smb_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_SMA, context.BM_MID) +
             market_value_weighted(stocks, context.MV_SMA, context.BM_BIG)) / 3
    # 获取大市值组合的市值加权组合收益率
    smb_b = (market_value_weighted(stocks, context.MV_BIG, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_MID) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 3
    smb = smb_s - smb_b
    # 获取大账面市值比组合的市值加权组合收益率
    hml_b = (market_value_weighted(stocks, context.MV_SMA, context.BM_BIG) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 2
    # 获取小账面市值比组合的市值加权组合收益率
    hml_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_SMA)) / 2
    hml = hml_b - hml_s
    market_close = get_reg_kdata(context.reg_kdata[0], target_indices=[50],
                                 length=context.date + 1, fill_up=True, df=True).close
    market_return = market_close.iloc[-1] / market_close.iloc[0] - 1
    coff_pool = []
    # 对每只股票进行回归获取其alpha值
    for stock in stocks.index:
        x_value = np.array([[market_return], [smb], [hml], [1.0]])
        y_value = np.array([stocks['return'][stock]])
        # OLS估计系数
        coff = np.linalg.lstsq(x_value.T, y_value)[0][3]
        coff_pool.append(coff)
    # 获取alpha最小并且小于0的10只的股票进行操作(若少于10只则全部买入)
    stocks['alpha'] = coff_pool
    stocks = stocks[stocks.alpha < 0].sort_values(by='alpha').head(10)
    symbols_pool = stocks.index.tolist()
    positions = context.account().positions
    # 平不在标的池的股票
    for target_idx in positions.target_idx.astype(int):
        if target_idx not in symbols_pool:
            if positions['volume_long'].iloc[target_idx] > 0:
                order_volume(account_idx=0, target_idx=target_idx,
                             volume=int(positions['volume_long'].iloc[target_idx]),
                             side=2, position_effect=2, order_type=2, price=0)
                # print('市价单平不在标的池的', context.target_list[target_idx])
    # 获取股票的权重
    percent = context.ratio / len(symbols_pool)
    # 买在标的池中的股票
    for target_idx in symbols_pool:
        order_target_percent(account_idx=0, target_idx=int(target_idx), target_percent=percent, side=1, order_type=2,
                             price=0)
        # print(context.target_list[int(target_idx)], '以市价单调多仓到仓位', percent)
if __name__ == '__main__':
    begin = '2017-08-01'
    end = '2018-05-31'
    cons_date = dt.datetime.strptime(begin, '%Y-%m-%d') - dt.timedelta(days=1)
    hs300 = get_code_list('sz50', cons_date)[['code', 'weight']]
    targetlist = list(hs300['code'])
    targetlist.append('sse.000016')
    run_backtest(strategy_name='Fama多因子',
                 file_path='FAMA.py',
                 target_list=targetlist,
                 frequency='day',
                 fre_num=1,
                 begin_date=begin,
                 end_date=end,
                 fq=1)
