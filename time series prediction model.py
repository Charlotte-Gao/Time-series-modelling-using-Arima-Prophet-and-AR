#!/usr/bin/env python
# coding: utf-8

# In[9]:


#import lightgbm as lgb
import numpy as np
import pandas as pd

from fbprophet import Prophet
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

np.random.seed(2020)


# In[10]:


df = pd.read_csv("shuttle services.csv") #, parse_dates=['date'], index_col='date')
df.set_index("date", drop=False, inplace=True)
df.head()


# In[11]:


# plot target variable shuttle over time
plt.title("Monthly shuttle services between 2012 and 2019")
df.shuttle.plot(figsize=(14, 7))


# In[12]:


#test_stationarity(df)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(df['shuttle'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


# ## Feature Engineering
# 
# Almost every time series problem will have some external features or some internal feature engineering to help the model.
# 
# Let’s add some basic features like lag values of available numeric features that are widely used for time series problems. Since we need to predict the shuttle value for a month (I wish you had daily data!!!), we cannot use the feature values of the same day since they will be unavailable at actual inference time. We need to use statistics like mean, standard deviation of their lagged values.
# 
# We will use three sets of lagged values, one previous day, one looking back 7 days and another looking back 30 days as a proxy for last week and last month metrics.

# In[13]:


df.reset_index(drop=True, inplace=True)
lag_features = ["shuttle"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

df.fillna(df.mean(), inplace=True)

df.set_index("date", drop=False, inplace=True)
df.head()


# For boosting models, it is very useful to add datetime features like hour, day, month, as applicable to provide the model information about the time component in the data. For time series models it is not explicitly required to pass this information but we could do so and we will in this notebook so that all models are compared on the exact same set of features.

# In[14]:


df.Date = pd.to_datetime(df.date, format="%b-%y")
df["monthnum"] = df.Date.dt.month
df["week"] = df.Date.dt.week
df["day"] = df.Date.dt.day
df["day_of_week"] = df.Date.dt.dayofweek
df["Date"] = df.Date
df.head()


# In[15]:


df_train = df[df.Date < "2019"]
df_valid = df[df.Date >= "2019"]

exogenous_features = ["shuttle_mean_lag3",
                      "shuttle_mean_lag7",
                      "shuttle_mean_lag30",
                      "shuttle_std_lag3",
                      "shuttle_std_lag7",
                      "shuttle_std_lag30",
                      "monthnum", "week", "day", "day_of_week"]


# In[16]:


model = auto_arima(df_train.shuttle, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.shuttle, exogenous=df_train[exogenous_features])

forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
df_valid["Forecast_ARIMAX"] = forecast


# In[17]:


df_valid[["shuttle", "Forecast_ARIMAX"]].plot(figsize=(14, 7))
plt.title("forecast ARIMAX model vs actual figure")


# In[18]:


model_fbp = Prophet(daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=False
                   #).add_seasonality(
                   #                  name='monthly',
                   #                  period=30.5,
                   #                  fourier_order=12
                   #).add_seasonality(
                   #                  name='daily',
                   #                  period=1,
                   #                  fourier_order=15
                   #).add_seasonality(
                   #                  name='weekly',
                   #                  period=7,
                   #                  fourier_order=20
                   #).add_seasonality(
                   #                  name='yearly',
                   #                  period=365.25,
                   #                  fourier_order=20
                   ).add_seasonality(
                                     name='quaterly',
                                     period=365.25/4,
                                     fourier_order=5,
                                     prior_scale=30)

for feature in exogenous_features:
    model_fbp.add_regressor(feature)
    
model_fbp.fit(df_train[["Date", "shuttle"] + exogenous_features].rename(columns={"Date": "ds", "shuttle": "y"}))

forecast = model_fbp.predict(df_valid[["Date", "shuttle"] + exogenous_features].rename(columns={"Date": "ds"}))

df_valid["Forecast_Prophet"] = forecast.yhat.values

model_fbp.plot_components(forecast)


# In[19]:


df_valid[["shuttle", "Forecast_ARIMAX", "Forecast_Prophet"]].plot(figsize=(14, 7))
plt.title("forecasted ARIMAX and Proohet vs actual figure ")


# In[20]:


print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.shuttle, df_valid.Forecast_ARIMAX)))
print("RMSE of Prophet:", np.sqrt(mean_squared_error(df_valid.shuttle, df_valid.Forecast_Prophet)))
print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.shuttle, df_valid.Forecast_ARIMAX))
print("MAE of Prophet:", mean_absolute_error(df_valid.shuttle, df_valid.Forecast_Prophet))


# More information on prophet can be found at:
# 
# https://facebook.github.io/prophet/docs/quick_start.html

# In[21]:


pd.plotting.lag_plot(df['shuttle'])


# Because we don’t have many data points, this particular lag_plot() doesn’t look terribly convincing, but there is some correlation in there (along with some possible outliers).
# 
# Like good data scientists/statisticians, we don’t want to just rely on a visual representation of correlation though, so we’ll use the idea of autocorrelation plots to look at correlations of our data.
# 
# Using pandas, you can plot an autocorrelation plot using this command:

# In[22]:


pd.plotting.autocorrelation_plot(df['shuttle'])


# The resulting chart contains a few lines on it separate from the autocorrelation function. The dark horizontal line at zero just denotes the zero line, the lighter full horizontal lines is the 95% confidence level and the dashed horizontal lines are 99% confidence levels, which means that correlations are more significant if they occur at those levels.

# In[38]:


# Actual vs Fitted
from datetime import datetime
import statsmodels.tsa.ar_model as AR
model1 = AR.AR(df['shuttle'])
model_fit1 = model1.fit(method='mle')
#predictions=AR.AR.predict(params=model_fit.params, start=datetime(2015, 7, 1), end=datetime(2019, 12, 1))
predictions = model_fit1.predict(dynamic=False)
plt.figure(figsize=(14,7))
predictions.plot(figsize=(14,7),label='prediction')
plt.plot(df['shuttle'],label='shuttle')
plt.xlabel('date')
plt.ylabel('shuttle services')
plt.title("AR model vs actual")
plt.show()


# In[24]:


#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

print("RMSE of AR:", np.sqrt(mean_squared_error(df['shuttle'],predictions )))
print("MAE of AR:", mean_absolute_error(df['shuttle'], predictions))


# In[25]:


from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
mod = ARMA(np.array(df_train["shuttle"]), order=(4,0))
result = mod.fit()


# In[26]:


best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5) # [0,1,2,3,4,5]
for i in rng:
    for j in rng:
        try:
            tmp_mdl = ARMA(np.array(df_train["shuttle"]), order=(i, j)).fit(
                method='mle', trend='nc'
            )
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))


# In[27]:


print(best_mdl.summary())


# In[28]:


import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


# In[29]:


_ = tsplot(best_mdl.resid) #, lags=max_lag)


# The ACF and PACF are showing no significant autocorrelation. The QQ and Probability Plots show the residuals are approximately normal with heavy tails. However, this model's residuals do NOT look like white noise! Look at the highlighted areas of obvious conditional heteroskedasticity (conditional volatility) that the model has not captured. 

# In[30]:


from statsmodels.tsa.arima_model import ARIMA
# pick best order and final model based on aic

best_aic = np.inf 
best_order = None
best_mdl = None

pq_rng = range(5) # [0,1,2,3,4]
d_rng = range(2) # [0,1]
for i in pq_rng:
    for d in d_rng:
        for j in pq_rng:
            try:
                tmp_mdl = ARIMA(np.array(df_train["shuttle"]), order=(i,d,j)).fit(method='mle', trend='nc')
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d, j)
                    best_mdl = tmp_mdl
            except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

# ARIMA model resid plot
_ = tsplot(best_mdl.resid, lags=30)


# In[ ]:




