print('ello')


#%%
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller



df = pd.read_csv(r'sales.csv')
df
# %%
df.isna().sum()
df.dropna( subset=['Postal Code'],inplace=True)

print(df.duplicated().sum())
# Remove duplicates (if any)
df = df.drop_duplicates()

#%%
# Group the data by state and calculate total sales
sales_by_state = df.groupby('State')['Sales'].sum()

# Plot total sales per state
plt.figure(figsize=(10, 8))
sales_by_state.plot(kind='bar')
plt.title('Total Sales per State')
plt.xlabel('State')
plt.ylabel('Total Sales')
plt.show()



#%%

# Choose two other meaningful columns to plot based on the sales data
# For example, let's choose 'Category' and 'Segment'
sales_by_category = df.groupby('Category')['Sales'].sum()
sales_by_segment = df.groupby('Segment')['Sales'].sum()


#%%
# Plot total sales per category
plt.figure(figsize=(10, 8))
sales_by_category.plot(kind='bar')
plt.title('Total Sales per Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()

# Plot total sales per segment
plt.figure(figsize=(10, 8))
sales_by_segment.plot(kind='bar')
plt.title('Total Sales per Segment')
plt.xlabel('Segment')
plt.ylabel('Total Sales')
plt.show()

# %%

df['Order_Date']

#%%
# Plot the total sales data per month
df['Order_Date'] = pd.to_datetime(df['Order_Date']
                                  ,format= '%d/%m/%Y'
                                  )



#%%
# df['Month'] = df['Order_Date'].dt.month
df['Month'] = df['Order_Date'].dt.to_period('M')

sales_by_month = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(10, 8))
sales_by_month.plot(kind='line')
plt.title('Total Sales per Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()




# Plot the total sales per month for each unique category
for category in df['Category'].unique():
    sales_by_month_category = df[df['Category'] == category].groupby('Month')['Sales'].sum()
    plt.figure(figsize=(10, 8))
    sales_by_month_category.plot(kind='line')
    plt.title(f'Total Sales per Month')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.show()



#%%
# df['Month'] = df['Order_Date'].dt.month
# sales_by_month = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(10, 8))
sales_by_month.plot(kind='line',label='Total')





# Plot the total sales per month for each unique category
for category in df['Category'].unique():
    sales_by_month_category = df[df['Category'] == category].groupby('Month')['Sales'].sum()
    sales_by_month_category.plot(kind='line',label=category)
 

plt.title(f'Total Sales per Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.legend()
plt.show()


# Q2

#%%
# df.isna().sum()

# #%%
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(25,10))
# sns.barplot(x='State',y='Sales',data=df)
# plt.show()


# # %%
# numeric_cols = df.select_dtypes(include=[np.number]).columns
# categorical_cols = df.select_dtypes(include=['object']).columns

# df_cat = df[categorical_cols]
# df_num =df[numeric_cols]
# #%%
# Q1 =df_num.quantile(.25)
# Q3 =df_num.quantile(.75)
# IQR=Q3-Q1

# # %%
# outliers = ((df_num < (Q1-1.5 * IQR))| (df_num > (Q3+1.5 * IQR)))
# # %%
# print(outliers)









# # %%
# y=df['Sales']


# # %%
# df2 =df.copy()
# df2.drop(columns=['Sales'],inplace=True)

# # %%
# print(df.isnull().sum())
# df.dropna( subset=['Postal Code'],inplace=True)
# print(df.isnull().sum())
# #%%





# Q2



df.set_index('Order_Date',inplace=True)


#%%

monthly_sales =df['Sales'].resample('M').sum()
#%%

# a. Plot the monthly decomposition sales for trends and seasonality
result = seasonal_decompose(monthly_sales,model='multiplicative')
result.plot()
plt.show()
# The plot shows the trend, seasonal, and residual components of the sales data.


#%%

# b. Generate the ACF and PACF plots
plot_acf(monthly_sales)
plt.show()
# The ACF plot shows the autocorrelation of the sales data at different lags.

plot_pacf(monthly_sales)
plt.show()
# The PACF plot shows the partial autocorrelation of the sales data at different lags.

# c. Check for stationarity using the Dickey-Fuller test
result = adfuller(monthly_sales)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
# If the p-value is less than 0.05, the sales data is stationary. Otherwise, it's non-stationary.
# %%



# Q3
# Import necessary libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#%%
train_data, test_data = train_test_split(monthly_sales,test_size=0.2,random_state=42,shuffle=False)

#%%

# Assuming sales_by_month is your time series
# Fit the Holt-Winters exponential smoothing model
model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Print the model parameters
print(model_fit.params)

# Assess the model fit/performance
print(model_fit.summary())


#%%
# Forecast the sales for the next 12 months
forecast = model_fit.forecast(steps=12)

# Plot the original data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Train')
plt.plot(forecast, label='Forecast')
plt.legend(loc='best')
plt.title('Holt-Winters Exponential Smoothing')
plt.show()
# %%


import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

#%%

predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

# Calculate error metrics
mae = mean_absolute_error(test_data, predictions)
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100


#%%
# Print error metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
# %%



# Q4

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Fit the AR model
model_ar = ARIMA(train_data, order=(1, 0, 0))
model_ar_fit = model_ar.fit()


#%%
# Fit the MA model
model_ma = ARIMA(train_data, order=(0, 0, 1))
model_ma_fit = model_ma.fit()


#%%
# Forecast the next 12 months with AR model
forecast_ar = model_ar_fit.forecast(steps=12)


#%%
# Forecast the next 12 months with MA model
forecast_ma = model_ma_fit.forecast(steps=12)



#%%
# Plot the original data, the AR forecast, and the MA forecast
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data.values, label='Original')
plt.plot(pd.date_range(train_data.index[-1], periods=12, freq='M'), forecast_ar, label='AR Forecast')
plt.plot(pd.date_range(train_data.index[-1], periods=12, freq='M'), forecast_ma, label='MA Forecast')
plt.legend()
plt.show()
# %%




import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


# Define the ARIMA model
model_arima = ARIMA(train_data, order=(1, 1, 1))

# Fit the ARIMA model
model_arima_fit = model_arima.fit(disp=False)


# Forecast the next 12 months with ARIMA model
forecast_arima = model_arima_fit.forecast(steps=12)

# Define the SARIMA model
model_sarima = SARIMAX(monthly_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Fit the SARIMA model
model_sarima_fit = model_sarima.fit(disp=False)

# Forecast the next 12 months with SARIMA model
forecast_sarima = model_sarima_fit.predict(len(monthly_sales), len(monthly_sales) + 11)

# Plot the original data, the ARIMA forecast, and the SARIMA forecast
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, label='Original')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), forecast_arima[0], label='ARIMA Forecast')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), forecast_sarima, label='SARIMA Forecast')
plt.legend()
plt.show()
# %%



# Define the ARIMA model
model_arima = ARIMA(monthly_sales, order=(1, 1, 1))

# Fit the ARIMA model
model_arima_fit = model_arima.fit()


# Forecast the next 12 months with ARIMA model
forecast_arima = model_arima_fit.forecast(steps=12)

# Define the SARIMA model
model_sarima = SARIMAX(monthly_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Fit the SARIMA model
model_sarima_fit = model_sarima.fit()

# Forecast the next 12 months with SARIMA model
forecast_sarima = model_sarima_fit.predict(len(monthly_sales), len(monthly_sales) + 11)

# Plot the original data, the ARIMA forecast, and the SARIMA forecast
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, label='Original')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), forecast_arima, label='ARIMA Forecast')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), forecast_sarima, label='SARIMA Forecast')
plt.legend()
plt.show()
# %%
# Forecast the next 12 months with each model
forecast_ar = model_ar_fit.forecast(steps=12)
forecast_ma = model_ma_fit.forecast(steps=12)
forecast_arima = model_arima_fit.forecast(steps=12)
forecast_sarima = model_sarima_fit.forecast(steps=12)
forecast_holt_winters = model_fit.forecast(steps=12)

# Calculate and print the MAE and RMSE for each model
models = ['AR', 'MA', 'ARIMA', 'SARIMA', 'Holt-Winters']
forecasts = [forecast_ar, forecast_ma, forecast_arima, forecast_sarima, forecast_holt_winters]

for model, forecast in zip(models, forecasts):
    mae = mean_absolute_error(monthly_sales[-12:], forecast)
    rmse = np.sqrt(mean_squared_error(monthly_sales[-12:], forecast))
    print(f"{model} Model: MAE = {mae}, RMSE = {rmse}")
# %%
# Fit the models and make forecasts
models = {
    'AR': ARIMA(train_data, order=(1, 0, 0)),
    'MA': ARIMA(train_data, order=(0, 0, 1)),
    'ARIMA': ARIMA(train_data, order=(1, 1, 1)),
    'SARIMA': SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
    'Holt-Winters': ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=12)
}



#%%
forecasts = {}
for name, model in models.items():
    model_fit = model.fit()
    forecasts[name] = model_fit.forecast(steps=len(test_data))


#%%
# Calculate and print the MAE and RMSE for each model
for name, forecast in forecasts.items():
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    print(f"{name} Model: MAE = {mae}, RMSE = {rmse}")
# %%
