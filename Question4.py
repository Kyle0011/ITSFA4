#%%
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


df = pd.read_csv(r'sales.csv')

# %%
# a)
df.isna().sum()
df.dropna( subset=['Postal Code'],inplace=True)
print(df.duplicated().sum())

# Remove duplicates (if any)
df = df.drop_duplicates()

df['Order_Date'] = pd.to_datetime(df['Order_Date']
                                  ,format= '%d/%m/%Y'
                                  )


#%%
# df['Month'] = df['Order_Date'].dt.month
df['Month'] = df['Order_Date'].dt.to_period('M')

sales_by_month = df.groupby('Month')['Sales'].sum()
#%%
df.set_index('Order_Date',inplace=True)
monthly_sales =df['Sales'].resample('M').sum()
train_data, test_data = train_test_split(monthly_sales,test_size=0.2,random_state=42,shuffle=False)


# Q4
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
#4.1
# a) Fit the AR model
model_ar = ARIMA(monthly_sales, order=(1, 0, 0))
model_ar_fit = model_ar.fit()


#%%
# Fit the MA model
model_ma = ARIMA(monthly_sales, order=(0, 0, 1))
model_ma_fit = model_ma.fit()


#%%
# Forecast the next 12 months with AR model
forecast_ar = model_ar_fit.forecast(steps=12)


#%%
# Forecast the next 12 months with MA model
forecast_ma = model_ma_fit.forecast(steps=12)


#%%
# b) Plot the original data, the AR forecast, and the MA forecast
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, label='Original')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), forecast_ar, label='AR Forecast')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), forecast_ma, label='MA Forecast')
plt.legend()
plt.title('AR & MA')
plt.show()


# %%


# 4.2

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt



# Define the ARIMA model
model_arima = ARIMA(monthly_sales, order=(1, 1, 1))

# Fit the ARIMA model
model_arima_fit = model_arima.fit()


# Forecast the next 12 months with ARIMA model
forecast_arima = model_arima_fit.forecast(steps=12)
# Plot the original data, the ARIMA forecast
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, label='Original')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), forecast_arima, label='ARIMA Forecast')
plt.title('ARIMA')
plt.legend()
plt.show()


# 4.3

#%%
# Define the SARIMA model
model_sarima = SARIMAX(monthly_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Fit the SARIMA model
model_sarima_fit = model_sarima.fit()

# Forecast the next 12 months with SARIMA model
forecast_sarima = model_sarima_fit.predict(len(monthly_sales), len(monthly_sales) + 11)

# Plot the original data, the SARIMA forecast
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, label='Original')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=12, freq='M'), forecast_sarima, label='SARIMA Forecast')
plt.legend()
plt.title('SARIMA')

plt.show()

#%%


# Compare all models
# %%
# Fit the models and make forecasts
models = {
    'AR': ARIMA(train_data, order=(1, 0, 0)),
    'MA': ARIMA(train_data, order=(0, 0, 1)),
    'ARIMA': ARIMA(train_data, order=(1, 1, 1)),
    'SARIMA': SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
    'Holt-Winters': ExponentialSmoothing(train_data, trend='add',seasonal='add', seasonal_periods=12)
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


