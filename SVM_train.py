import numpy as np
import pandas as pd
from atrader import *
from atrader.calcfactor import *
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
import datetime as dt

fac_0_ = 'RSI'
fac_1_ = 'SMA'
fac_2_ = 'PB'
fac_3_ = 'PE'
a_0 = preprocessing.StandardScaler()
a_1 = preprocessing.StandardScaler()
a_2 = preprocessing.StandardScaler()
a_3 = preprocessing.StandardScaler()
a_4 = preprocessing.StandardScaler()
begin = '2017-08-01'
end = '2018-05-31'

cons_date = dt.datetime.strptime(begin, '%Y-%m-%d') - dt.timedelta(days=1)
hs300 = get_code_list('sz50', cons_date)[['code', 'weight']]
target_list = list(hs300['code'])
fac = ['RSI','SMA','PB', 'PE']

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


diff = np.diff(stock.reshape(len(stock)))

judge = diff.copy().reshape((-1,1))
judge[judge<0]=0
judge[judge!=0]=1

try:
    t_s = len(fac_0)-1
except:
    t_s = int(len(fac_0)*0.8)
train_x = np.column_stack((fac_0[0:t_s],fac_1[0:t_s],fac_2[0:t_s],fac_3[0:t_s]))
train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1]))

test = np.random.randint(0,len(fac_0)-1,(100,1))
train_y = judge[0:t_s]

test_x = np.column_stack((fac_0[test],fac_1[test],fac_2[test],fac_3[test]))
test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1]))
print(np.shape(test_x))

model = svm.SVC(kernel='rbf',C=0.4,gamma=1,probability=True)
model.fit(train_x, train_y)

print(model.score(train_x, train_y))
probas = model.predict_proba(test_x)


precision, recall, _ = precision_recall_curve(judge[test].reshape((100,1)), probas[:,1])

plt.step(recall, precision, lw=1)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: ')
joblib.dump(model, "SVM_classifer.m")
