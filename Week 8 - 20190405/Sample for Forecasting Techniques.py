import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

#Importing data
# dataframe
df = pd.read_csv("Simple Dataset.txt", sep='\t')
# print(df.axes)
seriesname = 'Stable Sales'
series = df[seriesname]
sarray = np.asarray(series)
print("Estimating", seriesname)
print("Series Data:", sarray)
#'Stable Sales', 'Increasing Sales',
# 'Sales with Some Fluctuation', 'Increasing with Some Fluctuation',
# 'Increasing with Increasing Fluctuation',
# 'Increasing with Deccreasing Fluctuation'
size = len(series)
train = series[0:size-5]
trainarray= np.asarray(train)
test = series[size-5:]
testarray = np.asarray(test)
print("Training data:", trainarray, "Test data:", testarray)

