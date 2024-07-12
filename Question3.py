#%%
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
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

# %%
#%%
# Q3
# Import necessary libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%
# Assuming sales_by_month is your time series
# Fit the Holt-Winters exponential smoothing model
model = ExponentialSmoothing(monthly_sales, trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Print the model parameters
print(model_fit.params)

# Assess the model fit/performance
print(model_fit.summary())

# b)


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


#%%
# Forecast the sales for the next 12 months
forecast = model_fit.forecast(steps=12)

# Plot the original data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales, label='Train')
plt.plot(forecast, label='Forecast')
plt.legend(loc='best')
plt.title('Holt-Winters Exponential Smoothing')
plt.show()






# %%
