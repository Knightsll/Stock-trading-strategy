from atrader import *
from atrader.calcfactor import *
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from LTSM_class import LSTM_M_4,RP_M_4
import pandas as pd


def P_R(real,pre,thres):
    thres_0 = pre.copy()
    print(thres_0)
    thres_0[thres_0<thres]=0
    print(thres_0)
    thres_0[thres_0!=0]=1
    print(thres_0)
    TP = len(np.where((2*real+thres_0)==3)[0])
    TN = len(np.where((2*real+thres_0)==0)[0])
    FN = len(np.where((2*real+thres_0)==1)[0])
    FP = len(np.where((2*real+thres_0)==2)[0])
    if TP+FP!=0:
        P = TP/(float(TP+FP))
    else:
        P=0
    if TP+FN!=0:
        R = TP/(float(TP+FN))
    else:
        R=0
    return P,R


fac_0_ = 'RSI'
fac_1_ = 'SMA'
fac_2_ = 'PB'
fac_3_ = 'PE'

stock_name = 'sse.600000'

a = LSTM_M_4("./model/LSTM_full.h5")
b = RP_M_4("./model/BP_full.h5")
Data_stand = preprocessing.StandardScaler()
data_value = get_kdata(target_list=[stock_name], frequency='day', fre_num=1, begin_date='2018-04-01', end_date='2018-06-30')
data_factor = get_factor_by_code([fac_0_,fac_1_,fac_2_,fac_3_],stock_name,'2018-04-01','2018-06-30')

PB = data_factor[fac_0_].values.reshape((-1,1))
PE = data_factor[fac_1_].values.reshape((-1,1))
MA = data_factor[fac_2_].values.reshape((-1,1))
RSI = data_factor[fac_3_].values.reshape((-1,1))

t_PB = Data_stand.fit_transform(PB)
t_PE = Data_stand.fit_transform(PE)
t_MA = Data_stand.fit_transform(MA)
t_RSI = Data_stand.fit_transform(RSI)


high = data_value[stock_name]['high'].values
low = data_value[stock_name]['low'].values
close = data_value[stock_name]['close'].values
stock = ((high+low+2*close)/4.0).reshape((-1,1))

pre_v = np.array([])
pre_v_1 = np.array([])
for i in range(len(PE)):
    pre_v = np.append(pre_v, a.predict(t_PB[i],t_PE[i],t_MA[i],t_RSI[i]))
    pre_v_1 = np.append(pre_v_1, b.predict(t_PB[i],t_PE[i],t_MA[i],t_RSI[i]))
pre_v = Data_stand.inverse_transform(pre_v)
pre_v_1 = Data_stand.inverse_transform(pre_v_1)

pre_v = pre_v[0:(len(pre_v)-1)]
pre_v_1 = pre_v_1[0:(len(pre_v_1)-1)]
stock = stock[1:(len(stock)),::]
diff_p = np.diff(pre_v)
diff_p_1 = np.diff(pre_v_1)
diff_s = np.diff(stock.reshape(len(stock)))

real = diff_s.copy()
real[real<0]=0
real[real!=0]=1

P_sum = np.array([])
R_sum = np.array([])

P_sum_1 = np.array([])
R_sum_1 = np.array([])

for i in range(40):
    temp_p, temp_r = P_R(real,diff_p,i*(diff_p.min()/20.0))
    temp_p_1, temp_r_1 = P_R(real,diff_p_1,i*(diff_p_1.min()/20.0))
    P_sum = np.append(P_sum, temp_p)
    R_sum = np.append(R_sum, temp_r)
    P_sum_1 = np.append(P_sum_1, temp_p_1)
    R_sum_1 = np.append(R_sum_1, temp_r_1)

clf = joblib.load("SVM_classifer.m")
vr = np.column_stack((t_PB,t_PE,t_MA,t_RSI))
probas = clf.predict_proba(vr)

precision, recall, _ = precision_recall_curve(real, probas[:-2,1])

plt.figure(figsize = (8,8))
plt.title('P_R',fontsize=30)
plt.xlim([-0.05,1.05])
plt.xlabel('Recall',fontsize=20)
plt.ylim([-0.05,1.05])
plt.ylabel('Precision',fontsize=20)
plt.step(-np.sort(-np.append(R_sum,[1,0])),np.insert(P_sum,[-1,0],[1,0]),lw=2,label='LSTM')
plt.step(-np.sort(-np.append(R_sum_1,[1,0])),np.insert(P_sum_1,[-1,0],[1,0]),lw=2,label='BPNN')
plt.step(recall,np.sort(precision),lw=2,label = 'SVM')
plt.plot([0.01*x for x in range(100)],[0.01*x for x in range(100)], lw=4)
plt.legend()
