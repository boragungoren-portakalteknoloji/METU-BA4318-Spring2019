import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt 
# from sklearn.metrics import mean_squared_error

def estimate_holt(df, seriesname, alpha=0.2, slope=0.1, trend="add"):
    numbers = np.asarray(df[seriesname], dtype='float')
    model = Holt(numbers)
    fit = model.fit(alpha, slope, trend)
    estimate = fit.forecast(1)[-1]
    return estimate

def decomp(frame,name,f,mod='Additive'):
    #frame['Date'] = pd.to_datetime(frame['Date'])
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show() # Uncomment to reshow plot, saved as Figure 1.
    return result

#Series Descriptions	
#TP.KTF10	Personal (TRY)(Flow Data, %)-Level
#TP.KTF101	Personal (TRY)(Including Real Person Overdraft Account)(Flow Data, %)-Level
#TP.KTF11	Vehicle (TRY)(Flow Data, %)-Level
#TP.KTF12	Housing (TRY)(Flow Data, %)-Level
#TP.KTF17	Commercial (TRY)(Flow Data, %)-Level
#TP.KTF17.EUR	Commercial Loans (EUR)(Flow Data, %)-Level
#TP.KTF17.USD	Commercial Loans (USD)(Flow Data, %)-Level
#TP.KTF18	Commercial Loans (TRY)(Excluding Corporate Overdraft Account and Corporate Credit Cards)(Flow Data, %)-Level
#TP.KTFTUK	Consumer Loan (TRY)(Personal+Vehicle+Housing)(Flow Data, %)-Level
#TP.KTFTUK01	Consumer Loan (TRY)(Personal+Vehicle+Housing)(Including Real Person Overdraft Account)(Flow Data, %)-Level

# The following lines are to suppress warning messages.
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("Interest Rates.txt", sep='\t')
# replace , with . 
df = df.stack().str.replace(',','.').unstack()
df.set_index('Date')
df.info() # not all columns have same number of elements
print("Percentage of missing values")
print( df.isna().mean().round(4)*100)  # get percentage of missing values
seriesname = 'TP KTF18' # Commercial Loans (TRY)(Excluding Corporate Overdraft Account and Corporate Credit Cards)

print("Method 1 - dropna()")
df2 = df.dropna()
df2.info()
print("Percentage of missing values")
print( df2.isna().mean().round(4)*100)  # get percentage of missing values
result = decomp(df2,seriesname,f=52) # 52 weeks per year
# Estimate 'TP KTF18' for next 1 period
interest = round( estimate_holt(df2, seriesname, alpha=0.2, slope=0.1, trend="add"), 2)
print("Estimation on next week's interest rate is:", interest)

print("Method 2 - fillna")
# Load data
df = pd.read_csv("Interest Rates.txt", sep='\t')
# replace , with . 
df = df.stack().str.replace(',','.').unstack()
df.set_index('Date')
# forward fill, propagate non-null values forward
df3 = df.fillna(method ='ffill')
# backward fill, propagate non-null values backward
df3 = df3.fillna(method ='bfill') 
# alternative, use number # df.fillna(0, inplace = True)
df3.info()
print("Percentage of missing values")
print( df3.isna().mean().round(4)*100)  # get percentage of missing values
result = decomp(df3,seriesname,f=52) # 52 weeks per yearEstimate 'TP KTF18' for next 1 period

interest = round( estimate_holt(df3, seriesname, alpha=0.2, slope=0.1, trend="add"), 2)
print("Estimation on next week's interest rate is:", interest)






