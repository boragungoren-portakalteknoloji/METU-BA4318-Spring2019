import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # plotting
from sklearn.metrics import mean_squared_error # to calculate error
from math import sqrt # to calculate root MSE from MSE
#  very popular statistical package
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing

import warnings
warnings.filterwarnings("ignore") # turns off warnings in shell (below)

df = pd.read_csv("Simple Dataset.txt", sep='\t')
# print(df.axes)
# print(df)
# 'Stable Sales', 'Increasing Sales',
# 'Sales with Some Fluctuation', 'Increasing with Some Fluctuation',
# 'Increasing with Increasing Fluctuation',
# 'Increasing with Decreasing Fluctuation'
seriesname = 'Increasing with Decreasing Fluctuation'
interest = df[seriesname]
size = len(interest)
train = df[0:size-1]
test = df[size-1:]
# print("train:", train, "test:", test)
darray = np.asarray(interest)
trainarray = np.asarray(train[seriesname])
testarray = np.asarray(test[seriesname])
# print(interest)
# print(darray)

# sm.tsa.seasonal_decompose(interest).plot()
# result = sm.tsa.stattools.adfuller(interest)
# plt.show()

test['naive'] = darray[-2] # last one observed on training
RMSE = sqrt ( mean_squared_error(test[seriesname],test['naive']) )
print("RMSE for Naive:", RMSE)

test['simple mean'] = train[seriesname].mean()
RMSE = sqrt ( mean_squared_error(test[seriesname],test['simple mean']) )
RMSE = round(RMSE,2)
print("RMSE for Simple Mean:", RMSE)

windowsize = 5
test['moving average'] = train[seriesname].rolling(windowsize).mean().iloc[-1]
# last mean in series of means, each calculated on a rolling window
# of size windowsize
RMSE = sqrt ( mean_squared_error(test[seriesname],test['moving average']) )
RMSE = round(RMSE,2)
print("RMSE for Moving Average:", RMSE)

alpha = 0.5
slope = 0.1
model = Holt(trainarray)
fit = model.fit(smoothing_level = alpha, smoothing_slope = slope)
test['Holt'] = fit.forecast( len(test) )
RMSE = sqrt ( mean_squared_error(test[seriesname],test['Holt']) )
RMSE = round(RMSE,2)
print("RMSE for Holt:", RMSE)
# Holt(trainarray).fit(smoothing_level = alpha, smoothing_slope = slope).forecast( len(test) )

# Grid search for alpha and slope
best_rmse = 100000
best_alpha = 0.5
best_slope = 0.1
model = Holt(trainarray) # need to create just once
for alpha in np.linspace(0,1,11):
    for slope in np.linspace(0,1,11):
        fit = model.fit(smoothing_level = alpha, smoothing_slope = slope)
        test['Holt'] = fit.forecast( len(test) )
        current_rmse = sqrt ( mean_squared_error(test[seriesname],test['Holt']) )
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_alpha = alpha
            best_slope = slope
RMSE = round(best_rmse,2)
print("Holt Grid Search: Best alpha is", alpha, "and best slope is", slope, "with RMSE:", RMSE)

months = 6 # Is this a valid assumption!
model = ExponentialSmoothing(trainarray,
                             seasonal_periods=months,
                             trend='add',
                             seasonal='mult',
                             damped=True)
fit = model.fit()
test['Holt-Winters'] = fit.forecast(len(test))
RMSE = sqrt ( mean_squared_error(test[seriesname],test['Holt-Winters']) )
RMSE = round(RMSE,2)
print("RMSE for Holt-Winters:", RMSE)
