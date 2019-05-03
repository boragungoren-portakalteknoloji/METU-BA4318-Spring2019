# imports for stats libraries
from math import sqrt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt #,ExponentialSmoothing,  
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

# The following lines are to suppress warning messages.
import warnings
warnings.filterwarnings("ignore")

# Load data into a dataframe object, also create a copy of raw data in an array
original = pd.read_csv("Patients.csv", sep=';')

dataframe = original.head(len(original)-1)
dataframe.info()
name='Incoming'

freq = 12 #12 months per year
series = dataframe[name]
numbers = np.asarray(series,dtype='int')
result = sm.tsa.seasonal_decompose(numbers,freq=12,model='Additive')
# Additive model means y(t) = Level + Trend + Seasonality + Noise
result.plot()
plt.show() # Uncomment to reshow plot, saved as Figure 1. 
# Now using techniques to estimate, from simple to more complex

# Function for Naive
def estimate_naive(df, seriesname):
     numbers = np.asarray ( df[seriesname] )
     return float( numbers[-1] )
    
naive = round(estimate_naive (dataframe, name),0)
print ("Naive estimation:", naive)

# Function for Simple Average
def estimate_simple_average(df,seriesname):
    avg = df[seriesname].mean()
    return avg

simpleaverage = round( estimate_simple_average(dataframe, name), 0)
print("Simple average estimation:", simpleaverage)

# Function for Moving Average
def estimate_moving_average(df,seriesname,windowsize):
    avg = df[seriesname].rolling(windowsize).mean().iloc[-1]
    return avg

months = 12 # observed period is 12 months
movingaverage = round(estimate_moving_average(dataframe,name, months),0)
print("Moving average estimation for last ", months, " months: ", movingaverage)

# Function for Simple Exponential Smoothing
def estimate_ses(df, seriesname, alpha=0.2):
    numbers = np.asarray(df[seriesname])
    estimate = SimpleExpSmoothing(numbers).fit(smoothing_level=alpha,optimized=False).forecast(1)
    return estimate

alpha = 0.2
ses = round ( estimate_ses(dataframe, name, alpha)[0], 0)
print("Exponential smoothing estimation with alpha =", alpha, ": ", ses)

# Trend estimation with Holt
def estimate_holt(df, seriesname, alpha=0.2, slope=0.1):
    numbers = np.asarray(df[seriesname])
    model = Holt(numbers)
    fit = model.fit(alpha,slope)
    estimate = fit.forecast(1)[-1]
    return estimate

alpha = 0.2
slope = 0.1
holt = round(estimate_holt(dataframe,name,alpha, slope),0)
print("Holt trend estimation with alpha =", alpha, ", and slope =", slope, ": ", holt)

# Trend and seasonality estimation with Holt-Winters
def estimate_holtwinters(df, seriesname, periods=10, trendtype='additive',
                         seasontype='additive', damptype=False, boxcox=False,
                         trendmuchlarger=False):
    if trendmuchlarger == False:
        numbers = np.asarray(df[seriesname])
        model = ExponentialSmoothing(numbers, trend=trendtype, seasonal=seasontype,
                                     damped=damptype, seasonal_periods=periods)
        fit = model.fit(optimized=True, use_boxcox=boxcox, remove_bias=True)
        estimate = fit.predict(1)[-1]
        return estimate
    else:
        numbers = np.asarray(df[seriesname])
        # calculate a simple moving average trend estimate
        windowsize = 2
        trendestimate = np.asarray( df[seriesname].rolling(windowsize).mean() )
        # If seasonality is additive then subtract the trend estimate from the original series
        # print("trendestimate:", trendestimate)
        smooth = []
        if seasontype == 'additive':
            for n,t in zip(numbers,trendestimate):
                result = n - t
                smooth.append(n-t)
        # else (if seasonality is multiplicative, then divide the original series by the trend estimate
        else:
            for n,t in numbers,trendestimate:
                smooth.append(n / t)
        # Convert list to array
        smooth = np.asarray(smooth)
        # replace any nan values with 0
        smooth[np.isnan(smooth)] = 0
        # print("smooth:", smooth)
        #apply Holt-Winters to smoothed data
        model = ExponentialSmoothing(smooth, trend=trendtype, seasonal=seasontype,
                                     damped=damptype, seasonal_periods=periods)
        fit = model.fit(optimized=True, use_boxcox=boxcox, remove_bias=True)
        hwestimate = fit.predict(1)[-1]
        # print("hwe:", hwestimate)
        # Then update the prediction by adding/multiplying the trend estimate
        if seasontype == 'additive':
            hwestimate = hwestimate + trendestimate[-1]
        else:
            hwestimate = hwestimate * trendestimate[-1]
        return hwestimate
    
periods = 12 # This is the number of periods assumed
trend = "additive" # 
season = "additive" # Variations of seasonality is more or less same, as observed in the decomposition
damptype = True
hw = round(estimate_holtwinters(df=dataframe, seriesname=name,
                                periods=periods, trendtype=trend, seasontype=season,
                                damptype=damptype, trendmuchlarger=True), 0)
print("Holt Winters seasonal estimation with", periods, "periods:", hw)

target = original[name].tail(1)[1]
print("Target was:",target)
