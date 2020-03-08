import numpy as np
import pandas as pd
import random as rd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm,skew
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

print('asdfasdfasdf')
sales=pd.read_csv("../input/sales_train.csv")
item=pd.read_csv("../input/items.csv")
sub=pd.read_csv("../input/sample_submission.csv")
shops=pd.read_csv("../input/shops.csv")
test=pd.read_csv("../input/test.csv")
print(test.head())

sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x,'%d.%m.%Y'))
print(sales.info())

monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

print(monthly_sales.head(20))

x=item.groupby(['item_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()

plt.figure(figsize=(8,4))
ax = sns.barplot(x.item_category_id,x.item_id,alpha=0.8)
plt.title('Item per Category')
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()

ts=sales.groupby(['date_block_num'])['item_cnt_day'].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.show()

plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(), label='Rolling Mean')
plt.plot(ts.rolling(window=12,center=False).std(), label='Rolling Std')
plt.legend()
plt.show()

import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model='multiplicative')
fig=res.plot()
plt.show()

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model='additive')
fig=res.plot()
plt.show()


def test_stationarity(timeseries):
    print('Result of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(ts)
from pandas import Series as Series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob


ts = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts)
plt.plot(new_ts)
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts,12)       # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()
plt.show()

test_stationarity(new_ts)

best_aic=np.inf
best_order=None
best_mdl=None
rng=range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl=smt.ARMA(new_ts.values,order=(i,j)).fit(method='mle',trend='nc')
            tmp_aic=tmp_mdl.aic
            if tmp_aic<best_aic:
                best_aic = tmp_aic
                best_order = (i,j)
                mest_mdl=tmp_mdl
        except: continue
print('aic:{:6.5} | order:{}'.format(best_aic,best_order))

