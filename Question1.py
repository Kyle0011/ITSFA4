# Question1

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


#%%
# b) Group the data by state and calculate total sales
sales_by_state = df.groupby('State')['Sales'].sum()

# Plot total sales per state
plt.figure(figsize=(10, 8))
sales_by_state.plot(kind='bar')
plt.title('Total Sales per State')
plt.xlabel('State')
plt.ylabel('Total Sales')
plt.show()



#%%

# c) Choose two other meaningful columns to plot based on the sales data
# For example, let's choose 'Category' and 'Segment'
sales_by_category = df.groupby('Category')['Sales'].sum()
sales_by_segment = df.groupby('Segment')['Sales'].sum()

# )Plot total sales per category
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
# %%
