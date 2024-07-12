#%%
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


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

# Q2
#%%
df.set_index('Order_Date',inplace=True)
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
